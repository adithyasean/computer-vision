import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random, os, time, json

# 1. Reproducibility & Device Setup
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
BATCH_SIZE   = 128
EPOCHS       = 15      # 15 epochs per model for fast, clean, textbook-quality curves
LR           = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 0

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# 3. Data Transformations
transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

transform_aug = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# Remove incomplete archive if present to ensure clean extraction
tar_file = './data/cifar-10-python.tar.gz'
if os.path.exists(tar_file) and os.path.getsize(tar_file) < 160 * 1024 * 1024:
    print("Cleaning incomplete data archive...")
    os.remove(tar_file)

train_base = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_base)
train_aug  = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_aug)
test_set   = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

loader_base_train = DataLoader(train_base, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
loader_aug_train  = DataLoader(train_aug,  batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
loader_test       = DataLoader(test_set,   batch_size=256,        shuffle=False, num_workers=NUM_WORKERS)

CLASSES = test_set.classes
print(f"Classes: {CLASSES}")
print(f"Train samples: {len(train_base):,} | Test samples: {len(test_set):,}")

# 4. Model Architecture (Simple 3-Block CNN)
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
            ConvBlock(3, 32, pool=True),    # 32x32 -> 16x16
            ConvBlock(32, 64, pool=True),   # 16x16 -> 8x8
            ConvBlock(64, 128, pool=True),  # 8x8 -> 4x4
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

# 5. Training Loop & Evaluation
def train_one_epoch(model, loader, criterion, optimizer, scheduler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    scheduler.step()
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * len(labels)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total

def run_experiment(name, train_loader):
    print(f"\n=======================================================")
    print(f"  Training: {name}")
    print(f"=======================================================")

    torch.manual_seed(SEED)
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
        val_loss, val_acc = evaluate(model, loader_test, criterion)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Epoch {epoch:02d}/{EPOCHS}  train_loss={tr_loss:.4f}  train_acc={tr_acc*100:.1f}%  val_loss={val_loss:.4f}  val_acc={val_acc*100:.1f}%")

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s | Final Test Accuracy: {history['val_acc'][-1]*100:.2f}%")
    return model, history

if __name__ == '__main__':
    model_base, hist_base = run_experiment('Baseline (no augmentation)', loader_base_train)
    model_aug,  hist_aug  = run_experiment('Augmented', loader_aug_train)

    os.makedirs('notebooks/outputs', exist_ok=True)
    epochs_range = range(1, EPOCHS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CIFAR-10 Data Augmentation Comparison (Baseline vs Augmented)', fontsize=14, y=1.02)

    # Loss plot
    axes[0].plot(epochs_range, hist_base['train_loss'], '--', color='#E24B4A', alpha=0.6, label='Baseline Train Loss')
    axes[0].plot(epochs_range, hist_base['val_loss'], color='#E24B4A', lw=2, label='Baseline Test Loss')
    axes[0].plot(epochs_range, hist_aug['train_loss'], '--', color='#1D9E75', alpha=0.6, label='Augmented Train Loss')
    axes[0].plot(epochs_range, hist_aug['val_loss'], color='#1D9E75', lw=2, label='Augmented Test Loss')
    axes[0].set_title('Loss over Epochs', fontsize=12)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()

    # Accuracy plot
    axes[1].plot(epochs_range, [a*100 for a in hist_base['train_acc']], '--', color='#E24B4A', alpha=0.6, label='Baseline Train Acc')
    axes[1].plot(epochs_range, [a*100 for a in hist_base['val_acc']], color='#E24B4A', lw=2, label='Baseline Test Acc')
    axes[1].plot(epochs_range, [a*100 for a in hist_aug['train_acc']], '--', color='#1D9E75', alpha=0.6, label='Augmented Train Acc')
    axes[1].plot(epochs_range, [a*100 for a in hist_aug['val_acc']], color='#1D9E75', lw=2, label='Augmented Test Acc')
    axes[1].set_title('Accuracy over Epochs (%)', fontsize=12)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    curve_path = 'notebooks/outputs/cifar10_curves.png'
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"\n✅ Curves plot saved to {curve_path}")

    metrics = {
        'hist_base': hist_base,
        'hist_aug': hist_aug,
        'best_base_acc': max(hist_base['val_acc']) * 100,
        'best_aug_acc': max(hist_aug['val_acc']) * 100,
        'final_base_acc': hist_base['val_acc'][-1] * 100,
        'final_aug_acc': hist_aug['val_acc'][-1] * 100,
    }
    with open('notebooks/outputs/cifar10_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✅ Metrics saved to notebooks/outputs/cifar10_metrics.json")
