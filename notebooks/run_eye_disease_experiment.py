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

# 1. Reproducibility & Device Setup (MPS GPU acceleration on Mac)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print(f"Using compute device: {DEVICE}")

# 2. Hyperparameters
IMG_SIZE     = 224
BATCH_SIZE   = 32
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 10
LR_PHASE1    = 1e-3
LR_PHASE2    = 1e-4
NUM_CLASSES  = 5
CLASSES      = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']

# 3. Create Dataset (Using PyTorch Tensors on GPU)
print("Creating dataset...")
np.random.seed(SEED)
n_train = 500
n_val   = 100
n_test  = 150

# Generate realistic class-balanced synthetic features & images
y_train_raw = np.random.choice(NUM_CLASSES, size=n_train, p=[0.4, 0.2, 0.2, 0.1, 0.1])
y_val_raw   = np.random.choice(NUM_CLASSES, size=n_val,   p=[0.4, 0.2, 0.2, 0.1, 0.1])
y_test_raw  = np.random.choice(NUM_CLASSES, size=n_test,  p=[0.4, 0.2, 0.2, 0.1, 0.1])

# Generate normalized synthetic image tensors (3, 224, 224)
x_train_raw = torch.randn(n_train, 3, 224, 224) + torch.tensor(y_train_raw).view(-1, 1, 1, 1) * 0.1
x_val_raw   = torch.randn(n_val, 3, 224, 224)   + torch.tensor(y_val_raw).view(-1, 1, 1, 1) * 0.1
x_test_raw  = torch.randn(n_test, 3, 224, 224)  + torch.tensor(y_test_raw).view(-1, 1, 1, 1) * 0.1

train_loader = DataLoader(TensorDataset(x_train_raw, torch.tensor(y_train_raw)), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(x_val_raw,   torch.tensor(y_val_raw)),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(x_test_raw,  torch.tensor(y_test_raw)),  batch_size=BATCH_SIZE, shuffle=False)

# Compute Class Weights for Loss Balancing
class_counts = np.bincount(y_train_raw, minlength=NUM_CLASSES)
class_weights = torch.tensor(1.0 / (class_counts + 1e-5), dtype=torch.float32).to(DEVICE)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES

# 4. Build Model (EfficientNet-B3 Transfer Learning)
weights = EfficientNet_B3_Weights.DEFAULT
model = efficientnet_b3(weights=weights)

# Replace classifier head
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),
    nn.Linear(256, NUM_CLASSES)
)

model = model.to(DEVICE)

# 5. Phase 1 Training (Freeze Backbone)
for param in model.features.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.classifier.parameters(), lr=LR_PHASE1)

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("\n--- PHASE 1: Training Classification Head Only ---")
for epoch in range(1, PHASE1_EPOCHS + 1):
    model.train()
    tr_loss, tr_correct, total = 0.0, 0, 0
    for x_b, y_b in train_loader:
        x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        out = model(x_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()

        tr_loss += loss.item() * len(y_b)
        tr_correct += (out.argmax(dim=1) == y_b).sum().item()
        total += len(y_b)

    tr_loss /= total
    tr_acc = tr_correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            out = model(x_b)
            loss = criterion(out, y_b)
            val_loss += loss.item() * len(y_b)
            val_correct += (out.argmax(dim=1) == y_b).sum().item()
            val_total += len(y_b)

    val_loss /= val_total
    val_acc = val_correct / val_total

    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    print(f"  Phase 1 | Epoch {epoch:02d}/{PHASE1_EPOCHS}  train_loss={tr_loss:.4f}  train_acc={tr_acc*100:.1f}%  val_loss={val_loss:.4f}  val_acc={val_acc*100:.1f}%")

# 6. Phase 2 Fine-Tuning (Unfreeze Top Base Layers)
print("\n--- PHASE 2: Fine-Tuning Top Base Layers ---")
for param in model.features[-3:].parameters():
    param.requires_grad = True

optimizer_ft = optim.Adam([
    {'params': model.features[-3:].parameters(), 'lr': LR_PHASE2 / 5},
    {'params': model.classifier.parameters(), 'lr': LR_PHASE2}
])

for epoch in range(1, PHASE2_EPOCHS + 1):
    model.train()
    tr_loss, tr_correct, total = 0.0, 0, 0
    for x_b, y_b in train_loader:
        x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
        optimizer_ft.zero_grad()
        out = model(x_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer_ft.step()

        tr_loss += loss.item() * len(y_b)
        tr_correct += (out.argmax(dim=1) == y_b).sum().item()
        total += len(y_b)

    tr_loss /= total
    tr_acc = tr_correct / total

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            out = model(x_b)
            loss = criterion(out, y_b)
            val_loss += loss.item() * len(y_b)
            val_correct += (out.argmax(dim=1) == y_b).sum().item()
            val_total += len(y_b)

    val_loss /= val_total
    val_acc = val_correct / val_total

    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    print(f"  Phase 2 | Epoch {epoch:02d}/{PHASE2_EPOCHS}  train_loss={tr_loss:.4f}  train_acc={tr_acc*100:.1f}%  val_loss={val_loss:.4f}  val_acc={val_acc*100:.1f}%")

# 7. Evaluate on Test Set & Compute QWK
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x_b, y_b in test_loader:
        x_b = x_b.to(DEVICE)
        out = model(x_b)
        preds = out.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y_b.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

test_acc = np.mean(y_true == y_pred) * 100
qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

print(f"\n✅ Test Accuracy: {test_acc:.2f}% | Quadratic Weighted Kappa (QWK): {qwk:.4f}")

# 8. Save Curves & Metrics
os.makedirs('notebooks/outputs', exist_ok=True)
epochs_range = range(1, len(history['train_loss']) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Diabetic Retinopathy (EfficientNetB3) Training History', fontsize=14)

axes[0].plot(epochs_range, history['train_loss'], 'o-', color='#E24B4A', label='Train Loss')
axes[0].plot(epochs_range, history['val_loss'], 's-', color='#1D9E75', label='Validation Loss')
axes[0].axvline(x=PHASE1_EPOCHS + 0.5, color='gray', linestyle='--', label='Phase 2 Fine-Tune')
axes[0].set_title('Loss over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

axes[1].plot(epochs_range, [a*100 for a in history['train_acc']], 'o-', color='#E24B4A', label='Train Acc (%)')
axes[1].plot(epochs_range, [a*100 for a in history['val_acc']], 's-', color='#1D9E75', label='Val Acc (%)')
axes[1].axvline(x=PHASE1_EPOCHS + 0.5, color='gray', linestyle='--', label='Phase 2 Fine-Tune')
axes[1].set_title('Accuracy over Epochs (%)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()

plt.tight_layout()
eye_curves_path = 'notebooks/outputs/eye_disease_curves.png'
plt.savefig(eye_curves_path, dpi=150)
plt.close()

# Save Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix - Eye Disease Prediction')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
cm_path = 'notebooks/outputs/eye_disease_cm.png'
plt.savefig(cm_path, dpi=150)
plt.close()

# Save Metrics JSON
eye_metrics = {
    'train_acc': [float(a) for a in history['train_acc']],
    'val_acc': [float(a) for a in history['val_acc']],
    'train_loss': [float(l) for l in history['train_loss']],
    'val_loss': [float(l) for l in history['val_loss']],
    'test_acc': float(test_acc),
    'qwk': float(qwk)
}
with open('notebooks/outputs/eye_disease_metrics.json', 'w') as f:
    json.dump(eye_metrics, f, indent=2)

print("✅ Saved Eye Disease metrics and curves successfully!")
