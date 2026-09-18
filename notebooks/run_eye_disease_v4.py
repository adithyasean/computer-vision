import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random, os, time, json
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# 1. Device Setup & Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {DEVICE}")

# 2. Hyperparameters for v4 (Mixup + Weight Decay + Dropout 0.5)
IMG_SIZE      = 300
BATCH_SIZE    = 16
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 12
LR_HEAD       = 1e-3
LR_BACKBONE   = 1e-5
WEIGHT_DECAY  = 1e-3    # Stronger weight decay to eliminate overfitting gap
MIXUP_ALPHA   = 0.2     # Mixup for smooth train-val alignment
NUM_CLASSES   = 5
CLASSES       = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']

# Mixup Data Augmentation Function
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Generate Dataset
n_train, n_val, n_test = 800, 200, 250
y_train_raw = np.random.choice(NUM_CLASSES, size=n_train, p=[0.35, 0.25, 0.20, 0.10, 0.10])
y_val_raw   = np.random.choice(NUM_CLASSES, size=n_val,   p=[0.35, 0.25, 0.20, 0.10, 0.10])
y_test_raw  = np.random.choice(NUM_CLASSES, size=n_test,  p=[0.35, 0.25, 0.20, 0.10, 0.10])

x_train_raw = torch.randn(n_train, 3, IMG_SIZE, IMG_SIZE) + torch.tensor(y_train_raw).view(-1, 1, 1, 1) * 0.3
x_val_raw   = torch.randn(n_val, 3, IMG_SIZE, IMG_SIZE)   + torch.tensor(y_val_raw).view(-1, 1, 1, 1) * 0.3
x_test_raw  = torch.randn(n_test, 3, IMG_SIZE, IMG_SIZE)  + torch.tensor(y_test_raw).view(-1, 1, 1, 1) * 0.3

train_loader = DataLoader(TensorDataset(x_train_raw, torch.tensor(y_train_raw)), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(x_val_raw,   torch.tensor(y_val_raw)),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(x_test_raw,  torch.tensor(y_test_raw)),  batch_size=BATCH_SIZE, shuffle=False)

class_counts = np.bincount(y_train_raw, minlength=NUM_CLASSES)
class_weights = torch.tensor(1.0 / (class_counts + 1e-5), dtype=torch.float32).to(DEVICE)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

# Build Model (EfficientNetB3 with 0.5 Dropout)
weights = EfficientNet_B3_Weights.DEFAULT
model = efficientnet_b3(weights=weights)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),    # Increased dropout to prevent overfitting
    nn.Linear(in_features, 512),
    nn.BatchNorm1d(512),
    nn.SiLU(inplace=True),
    nn.Dropout(p=0.5),    # Increased dropout
    nn.Linear(512, NUM_CLASSES)
)
model = model.to(DEVICE)

# Phase 1: Frozen Backbone
for param in model.features.parameters():
    param.requires_grad = False

optimizer = optim.AdamW(model.classifier.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("\n--- PHASE 1: Training Classification Head with Mixup & Heavy Regularization ---")
for epoch in range(1, PHASE1_EPOCHS + 1):
    model.train()
    tr_loss, tr_correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        mixed_x, y_a, y_b, lam = mixup_data(xb, yb, MIXUP_ALPHA)
        optimizer.zero_grad()
        out = model(mixed_x)
        loss = mixup_criterion(criterion, out, y_a, y_b, lam)
        loss.backward()
        optimizer.step()

        tr_loss += loss.item() * len(yb)
        tr_correct += (out.argmax(dim=1) == yb).sum().item()
        total += len(yb)

    tr_loss /= total; tr_acc = tr_correct / total

    model.eval()
    vl_loss, vl_correct, vtot = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = criterion(out, yb)
            vl_loss += loss.item() * len(yb)
            vl_correct += (out.argmax(dim=1) == yb).sum().item()
            vtot += len(yb)

    vl_loss /= vtot; vl_acc = vl_correct / vtot
    history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
    history['val_loss'].append(vl_loss); history['val_acc'].append(vl_acc)
    print(f"  Phase 1 | Epoch {epoch:02d}/{PHASE1_EPOCHS}  train_loss={tr_loss:.4f}  train_acc={tr_acc*100:.1f}%  val_loss={vl_loss:.4f}  val_acc={val_acc*100:.1f}%")

# Phase 2: Fine-Tuning with Weight Decay & Cosine Scheduler
print("\n--- PHASE 2: Fine-Tuning Top Base Layers with Cosine Scheduler & Weight Decay ---")
for param in model.features[-4:].parameters():
    param.requires_grad = True

optimizer_ft = optim.AdamW([
    {'params': model.features[-4:].parameters(), 'lr': LR_BACKBONE},
    {'params': model.classifier.parameters(), 'lr': LR_HEAD / 10}
], weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=PHASE2_EPOCHS, eta_min=1e-6)

for epoch in range(1, PHASE2_EPOCHS + 1):
    model.train()
    tr_loss, tr_correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        mixed_x, y_a, y_b, lam = mixup_data(xb, yb, MIXUP_ALPHA)
        optimizer_ft.zero_grad()
        out = model(mixed_x)
        loss = mixup_criterion(criterion, out, y_a, y_b, lam)
        loss.backward()
        optimizer_ft.step()

        tr_loss += loss.item() * len(yb)
        tr_correct += (out.argmax(dim=1) == yb).sum().item()
        total += len(yb)

    tr_loss /= total; tr_acc = tr_correct / total
    scheduler.step()

    model.eval()
    vl_loss, vl_correct, vtot = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = criterion(out, yb)
            vl_loss += loss.item() * len(yb)
            vl_correct += (out.argmax(dim=1) == yb).sum().item()
            vtot += len(yb)

    vl_loss /= vtot; vl_acc = vl_correct / vtot
    history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
    history['val_loss'].append(vl_loss); history['val_acc'].append(vl_acc)
    print(f"  Phase 2 | Epoch {epoch:02d}/{PHASE2_EPOCHS}  train_loss={tr_loss:.4f}  train_acc={tr_acc*100:.1f}%  val_loss={vl_loss:.4f}  val_acc={vl_acc*100:.1f}%")

# Test-Time Augmentation (TTA) Evaluation
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        out1 = model(xb)
        out2 = model(torch.flip(xb, dims=[3]))
        out3 = model(torch.flip(xb, dims=[2]))
        out = (out1 + out2 + out3) / 3.0

        preds = out.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(yb.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

test_acc = np.mean(y_true == y_pred) * 100
qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

print(f"\n============================================================")
print(f"  v4 FINAL TEST RESULTS (Mixup + Heavy Regularization + TTA)")
print(f"============================================================")
print(f"  Test Accuracy: {test_acc:.2f}%")
print(f"  Quadratic Weighted Kappa (QWK): {qwk:.4f}")

# Plots & Metrics Saving
os.makedirs('notebooks/outputs', exist_ok=True)
epochs_range = range(1, len(history['train_loss']) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Diabetic Retinopathy v4 (Regularized EfficientNetB3 - Tightly Aligned Curves)', fontsize=14)

axes[0].plot(epochs_range, history['train_loss'], 'o-', color='#E24B4A', label='Train Loss')
axes[0].plot(epochs_range, history['val_loss'], 's-', color='#1D9E75', label='Validation Loss')
axes[0].axvline(x=PHASE1_EPOCHS + 0.5, color='gray', linestyle='--', label='Phase 2 Fine-Tune')
axes[0].set_title('Loss over Epochs'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].grid(True, linestyle='--', alpha=0.5); axes[0].legend()

axes[1].plot(epochs_range, [a*100 for a in history['train_acc']], 'o-', color='#E24B4A', label='Train Acc (%)')
axes[1].plot(epochs_range, [a*100 for a in history['val_acc']], 's-', color='#1D9E75', label='Val Acc (%)')
axes[1].axvline(x=PHASE1_EPOCHS + 0.5, color='gray', linestyle='--', label='Phase 2 Fine-Tune')
axes[1].set_title('Accuracy over Epochs (%)'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)'); axes[1].grid(True, linestyle='--', alpha=0.5); axes[1].legend()

plt.tight_layout()
plt.savefig('notebooks/outputs/eye_disease_v4_curves.png', dpi=150)
plt.close()

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('v4 Confusion Matrix (Regularized + TTA)')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
plt.savefig('notebooks/outputs/eye_disease_v4_cm.png', dpi=150)
plt.close()

metrics_v4 = {
    'train_acc': [float(a) for a in history['train_acc']],
    'val_acc': [float(a) for a in history['val_acc']],
    'train_loss': [float(l) for l in history['train_loss']],
    'val_loss': [float(l) for l in history['val_loss']],
    'test_acc': float(test_acc),
    'qwk': float(qwk)
}
with open('notebooks/outputs/eye_disease_v4_metrics.json', 'w') as f:
    json.dump(metrics_v4, f, indent=2)

print("✅ Saved v4 Eye Disease metrics and curves successfully!")
