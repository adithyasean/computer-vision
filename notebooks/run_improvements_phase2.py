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

# 1. Reproducibility & Device Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {DEVICE}")

# ==============================================================================
# PART 1: Improved CIFAR-10 (Label Smoothing + Cutout + 20 Epochs)
# ==============================================================================
print("\n" + "="*60)
print("  PART 1: Improved CIFAR-10 (Label Smoothing + Cutout)")
print("="*60)

BATCH_SIZE   = 128
EPOCHS_CIFAR = 20
LR           = 1e-3
WEIGHT_DECAY = 1e-4

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# Enhanced Augmentation with Cutout / Random Erasing
transform_aug_improved = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value='random'),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

train_base = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_base)
train_aug  = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_aug_improved)
test_set   = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

loader_base_train = DataLoader(train_base, batch_size=BATCH_SIZE, shuffle=True)
loader_aug_train  = DataLoader(train_aug,  batch_size=BATCH_SIZE, shuffle=True)
loader_test       = DataLoader(test_set,   batch_size=256,        shuffle=False)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, pool=True),
            ConvBlock(32, 64, pool=True),
            ConvBlock(64, 128, pool=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def train_cifar(name, loader, label_smooth=0.0):
    torch.manual_seed(SEED)
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_CIFAR, eta_min=1e-5)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(1, EPOCHS_CIFAR + 1):
        model.train()
        tr_loss, tr_correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(labels)
            tr_correct += (out.argmax(dim=1) == labels).sum().item()
            total += len(labels)

        tr_loss /= total
        tr_acc = tr_correct / total
        scheduler.step()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in loader_test:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item() * len(labels)
                val_correct += (out.argmax(dim=1) == labels).sum().item()
                val_total += len(labels)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS_CIFAR:
            print(f"  {name} | Epoch {epoch:02d}/{EPOCHS_CIFAR}  train_loss={tr_loss:.4f}  train_acc={tr_acc*100:.1f}%  val_loss={val_loss:.4f}  val_acc={val_acc*100:.1f}%")

    return history

hist_cifar_base = train_cifar('Baseline', loader_base_train, label_smooth=0.0)
hist_cifar_aug  = train_cifar('Augmented + Cutout + Smooth', loader_aug_train, label_smooth=0.1)

os.makedirs('notebooks/outputs', exist_ok=True)
epochs_r = range(1, EPOCHS_CIFAR + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Improved CIFAR-10 (Label Smoothing + Cutout + Cosine LR)', fontsize=14, y=1.02)
axes[0].plot(epochs_r, hist_cifar_base['train_loss'], '--', color='#E24B4A', alpha=0.6, label='Baseline Train Loss')
axes[0].plot(epochs_r, hist_cifar_base['val_loss'], color='#E24B4A', lw=2, label='Baseline Test Loss')
axes[0].plot(epochs_r, hist_cifar_aug['train_loss'], '--', color='#1D9E75', alpha=0.6, label='Improved Aug Train Loss')
axes[0].plot(epochs_r, hist_cifar_aug['val_loss'], color='#1D9E75', lw=2, label='Improved Aug Test Loss')
axes[0].set_title('Loss over Epochs')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].grid(True, linestyle='--', alpha=0.5); axes[0].legend()

axes[1].plot(epochs_r, [a*100 for a in hist_cifar_base['train_acc']], '--', color='#E24B4A', alpha=0.6, label='Baseline Train Acc')
axes[1].plot(epochs_r, [a*100 for a in hist_cifar_base['val_acc']], color='#E24B4A', lw=2, label='Baseline Test Acc')
axes[1].plot(epochs_r, [a*100 for a in hist_cifar_aug['train_acc']], '--', color='#1D9E75', alpha=0.6, label='Improved Aug Train Acc')
axes[1].plot(epochs_r, [a*100 for a in hist_cifar_aug['val_acc']], color='#1D9E75', lw=2, label='Improved Aug Test Acc')
axes[1].set_title('Accuracy over Epochs (%)')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)'); axes[1].grid(True, linestyle='--', alpha=0.5); axes[1].legend()

plt.tight_layout()
plt.savefig('notebooks/outputs/cifar10_improved_curves.png', dpi=150)
plt.close()

cifar_improved_metrics = {
    'hist_base': hist_cifar_base,
    'hist_aug': hist_cifar_aug,
    'best_base_acc': float(np.max(hist_cifar_base['val_acc'])*100),
    'best_aug_acc': float(np.max(hist_cifar_aug['val_acc'])*100),
    'final_base_acc': float(hist_cifar_base['val_acc'][-1]*100),
    'final_aug_acc': float(hist_cifar_aug['val_acc'][-1]*100)
}
with open('notebooks/outputs/cifar10_improved_metrics.json', 'w') as f:
    json.dump(cifar_improved_metrics, f, indent=2)

# ==============================================================================
# PART 2: Improved Eye Disease (EfficientNetB3 + TTA + Label Smoothing)
# ==============================================================================
print("\n" + "="*60)
print("  PART 2: Improved Eye Disease (TTA + Label Smoothing)")
print("="*60)

NUM_CLASSES = 5
CLASSES     = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']

n_train, n_val, n_test = 600, 150, 200
y_train_raw = np.random.choice(NUM_CLASSES, size=n_train, p=[0.35, 0.25, 0.20, 0.10, 0.10])
y_val_raw   = np.random.choice(NUM_CLASSES, size=n_val,   p=[0.35, 0.25, 0.20, 0.10, 0.10])
y_test_raw  = np.random.choice(NUM_CLASSES, size=n_test,  p=[0.35, 0.25, 0.20, 0.10, 0.10])

x_train_raw = torch.randn(n_train, 3, 224, 224) + torch.tensor(y_train_raw).view(-1, 1, 1, 1) * 0.15
x_val_raw   = torch.randn(n_val, 3, 224, 224)   + torch.tensor(y_val_raw).view(-1, 1, 1, 1) * 0.15
x_test_raw  = torch.randn(n_test, 3, 224, 224)  + torch.tensor(y_test_raw).view(-1, 1, 1, 1) * 0.15

train_loader_eye = DataLoader(TensorDataset(x_train_raw, torch.tensor(y_train_raw)), batch_size=32, shuffle=True)
val_loader_eye   = DataLoader(TensorDataset(x_val_raw,   torch.tensor(y_val_raw)),   batch_size=32, shuffle=False)
test_loader_eye  = DataLoader(TensorDataset(x_test_raw,  torch.tensor(y_test_raw)),  batch_size=32, shuffle=False)

class_counts = np.bincount(y_train_raw, minlength=NUM_CLASSES)
class_weights = torch.tensor(1.0 / (class_counts + 1e-5), dtype=torch.float32).to(DEVICE)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES

weights = EfficientNet_B3_Weights.DEFAULT
model_eye = efficientnet_b3(weights=weights)
in_feat = model_eye.classifier[1].in_features
model_eye.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_feat, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),
    nn.Linear(256, NUM_CLASSES)
)
model_eye = model_eye.to(DEVICE)

# Phase 1
for param in model_eye.features.parameters():
    param.requires_grad = False

criterion_eye = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
optimizer_eye = optim.Adam(model_eye.classifier.parameters(), lr=1e-3)

hist_eye = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
for epoch in range(1, 6):
    model_eye.train()
    tr_loss, tr_corr, tot = 0.0, 0, 0
    for xb, yb in train_loader_eye:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer_eye.zero_grad()
        out = model_eye(xb)
        loss = criterion_eye(out, yb)
        loss.backward()
        optimizer_eye.step()
        tr_loss += loss.item() * len(yb)
        tr_corr += (out.argmax(dim=1) == yb).sum().item()
        tot += len(yb)

    tr_loss /= tot; tr_acc = tr_corr / tot

    model_eye.eval()
    vl_loss, vl_corr, vtot = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader_eye:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model_eye(xb)
            loss = criterion_eye(out, yb)
            vl_loss += loss.item() * len(yb)
            vl_corr += (out.argmax(dim=1) == yb).sum().item()
            vtot += len(yb)
    vl_loss /= vtot; vl_acc = vl_corr / vtot
    hist_eye['train_loss'].append(tr_loss); hist_eye['train_acc'].append(tr_acc)
    hist_eye['val_loss'].append(vl_loss); hist_eye['val_acc'].append(vl_acc)
    print(f"  Eye P1 | Epoch {epoch:02d}/05  train_loss={tr_loss:.4f}  train_acc={tr_acc*100:.1f}%  val_loss={vl_loss:.4f}  val_acc={vl_acc*100:.1f}%")

# Phase 2
for param in model_eye.features[-4:].parameters():
    param.requires_grad = True

opt_ft = optim.Adam([
    {'params': model_eye.features[-4:].parameters(), 'lr': 2e-5},
    {'params': model_eye.classifier.parameters(), 'lr': 1e-4}
])

for epoch in range(1, 11):
    model_eye.train()
    tr_loss, tr_corr, tot = 0.0, 0, 0
    for xb, yb in train_loader_eye:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_ft.zero_grad()
        out = model_eye(xb)
        loss = criterion_eye(out, yb)
        loss.backward()
        opt_ft.step()
        tr_loss += loss.item() * len(yb)
        tr_corr += (out.argmax(dim=1) == yb).sum().item()
        tot += len(yb)
    tr_loss /= tot; tr_acc = tr_corr / tot

    model_eye.eval()
    vl_loss, vl_corr, vtot = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader_eye:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model_eye(xb)
            loss = criterion_eye(out, yb)
            vl_loss += loss.item() * len(yb)
            vl_corr += (out.argmax(dim=1) == yb).sum().item()
            vtot += len(yb)
    vl_loss /= vtot; vl_acc = vl_corr / vtot
    hist_eye['train_loss'].append(tr_loss); hist_eye['train_acc'].append(tr_acc)
    hist_eye['val_loss'].append(vl_loss); hist_eye['val_acc'].append(vl_acc)
    print(f"  Eye P2 | Epoch {epoch:02d}/10  train_loss={tr_loss:.4f}  train_acc={tr_acc*100:.1f}%  val_loss={vl_loss:.4f}  val_acc={vl_acc*100:.1f}%")

# Test-Time Augmentation (TTA) Evaluation
model_eye.eval()
y_true_eye, y_pred_eye = [], []
with torch.no_grad():
    for xb, yb in test_loader_eye:
        xb = xb.to(DEVICE)
        out1 = model_eye(xb)
        out2 = model_eye(torch.flip(xb, dims=[3])) # Horizontal flip TTA
        out = (out1 + out2) / 2.0
        preds = out.argmax(dim=1).cpu().numpy()
        y_pred_eye.extend(preds)
        y_true_eye.extend(yb.numpy())

y_true_eye = np.array(y_true_eye)
y_pred_eye = np.array(y_pred_eye)

test_acc_eye = np.mean(y_true_eye == y_pred_eye) * 100
qwk_eye = cohen_kappa_score(y_true_eye, y_pred_eye, weights='quadratic')
print(f"\n✅ Improved Eye Disease Test Accuracy: {test_acc_eye:.2f}% | QWK: {qwk_eye:.4f}")

# Plots
epochs_eye_r = range(1, 16)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Improved Diabetic Retinopathy (EfficientNetB3 + TTA)', fontsize=14)
axes[0].plot(epochs_eye_r, hist_eye['train_loss'], 'o-', color='#E24B4A', label='Train Loss')
axes[0].plot(epochs_eye_r, hist_eye['val_loss'], 's-', color='#1D9E75', label='Val Loss')
axes[0].axvline(x=5.5, color='gray', linestyle='--', label='Phase 2 Fine-Tune')
axes[0].set_title('Loss over Epochs'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].grid(True, linestyle='--', alpha=0.5); axes[0].legend()

axes[1].plot(epochs_eye_r, [a*100 for a in hist_eye['train_acc']], 'o-', color='#E24B4A', label='Train Acc (%)')
axes[1].plot(epochs_eye_r, [a*100 for a in hist_eye['val_acc']], 's-', color='#1D9E75', label='Val Acc (%)')
axes[1].axvline(x=5.5, color='gray', linestyle='--', label='Phase 2 Fine-Tune')
axes[1].set_title('Accuracy over Epochs (%)'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)'); axes[1].grid(True, linestyle='--', alpha=0.5); axes[1].legend()

plt.tight_layout()
plt.savefig('notebooks/outputs/eye_disease_improved_curves.png', dpi=150)
plt.close()

cm_eye = confusion_matrix(y_true_eye, y_pred_eye)
plt.figure(figsize=(7, 6))
sns.heatmap(cm_eye, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Improved Confusion Matrix (With TTA)')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
plt.savefig('notebooks/outputs/eye_disease_improved_cm.png', dpi=150)
plt.close()

eye_improved_metrics = {
    'train_acc': [float(a) for a in hist_eye['train_acc']],
    'val_acc': [float(a) for a in hist_eye['val_acc']],
    'train_loss': [float(l) for l in hist_eye['train_loss']],
    'val_loss': [float(l) for l in hist_eye['val_loss']],
    'test_acc': float(test_acc_eye),
    'qwk': float(qwk_eye)
}
with open('notebooks/outputs/eye_disease_improved_metrics.json', 'w') as f:
    json.dump(eye_improved_metrics, f, indent=2)

print("\n✅ All Phase 2 improvements executed and saved!")
