import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random, os, time, json

# 1. Device Setup & Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {DEVICE}")

# 2. Hyperparameters for CIFAR-10 v3 (ResNet-18)
BATCH_SIZE   = 128
EPOCHS       = 25
LR           = 1e-3
WEIGHT_DECAY = 1e-4

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

transform_aug_resnet = transforms.Compose([
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
train_aug  = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_aug_resnet)
test_set   = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

loader_base_train = DataLoader(train_base, batch_size=BATCH_SIZE, shuffle=True)
loader_aug_train  = DataLoader(train_aug,  batch_size=BATCH_SIZE, shuffle=True)
loader_test       = DataLoader(test_set,   batch_size=256,        shuffle=False)

CLASSES = test_set.classes

# Helper to build ResNet-18 modified for 32x32 CIFAR-10
def build_cifar_resnet18():
    model = resnet18(weights=None)
    # Modify initial conv to 3x3 with stride 1 (for 32x32 image size)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool to preserve spatial resolution
    model.fc = nn.Linear(512, 10)
    return model

def train_cifar_v3(name, loader):
    print(f"\n=======================================================")
    print(f"  CIFAR-10 v3 Training (ResNet-18): {name}")
    print(f"=======================================================")
    
    torch.manual_seed(SEED)
    model = build_cifar_resnet18().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, EPOCHS + 1):
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

        tr_loss /= total; tr_acc = tr_correct / total
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

        val_loss /= val_total; val_acc = val_correct / val_total

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"  Epoch {epoch:02d}/{EPOCHS}  train_loss={tr_loss:.4f}  train_acc={tr_acc*100:.1f}%  val_loss={val_loss:.4f}  val_acc={val_acc*100:.1f}%")

    return history

if __name__ == '__main__':
    hist_base_v3 = train_cifar_v3('Baseline (ResNet-18)', loader_base_train)
    hist_aug_v3  = train_cifar_v3('Augmented (ResNet-18 + Cutout)', loader_aug_train)

    os.makedirs('notebooks/outputs', exist_ok=True)
    epochs_r = range(1, EPOCHS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CIFAR-10 v3 (ResNet-18 + Cutout + Cosine LR)', fontsize=14, y=1.02)
    
    axes[0].plot(epochs_r, hist_base_v3['train_loss'], '--', color='#E24B4A', alpha=0.6, label='Baseline Train Loss')
    axes[0].plot(epochs_r, hist_base_v3['val_loss'], color='#E24B4A', lw=2, label='Baseline Test Loss')
    axes[0].plot(epochs_r, hist_aug_v3['train_loss'], '--', color='#1D9E75', alpha=0.6, label='Augmented ResNet Loss')
    axes[0].plot(epochs_r, hist_aug_v3['val_loss'], color='#1D9E75', lw=2, label='Augmented ResNet Test Loss')
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-Entropy Loss'); axes[0].grid(True, linestyle='--', alpha=0.5); axes[0].legend()

    axes[1].plot(epochs_r, [a*100 for a in hist_base_v3['train_acc']], '--', color='#E24B4A', alpha=0.6, label='Baseline Train Acc')
    axes[1].plot(epochs_r, [a*100 for a in hist_base_v3['val_acc']], color='#E24B4A', lw=2, label='Baseline Test Acc')
    axes[1].plot(epochs_r, [a*100 for a in hist_aug_v3['train_acc']], '--', color='#1D9E75', alpha=0.6, label='Augmented ResNet Train Acc')
    axes[1].plot(epochs_r, [a*100 for a in hist_aug_v3['val_acc']], color='#1D9E75', lw=2, label='Augmented ResNet Test Acc')
    axes[1].set_title('Accuracy over Epochs (%)')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)'); axes[1].grid(True, linestyle='--', alpha=0.5); axes[1].legend()

    plt.tight_layout()
    plt.savefig('notebooks/outputs/cifar10_v3_curves.png', dpi=150)
    plt.close()

    metrics_v3 = {
        'hist_base': hist_base_v3,
        'hist_aug': hist_aug_v3,
        'best_base_acc': float(np.max(hist_base_v3['val_acc'])*100),
        'best_aug_acc': float(np.max(hist_aug_v3['val_acc'])*100),
        'final_base_acc': float(hist_base_v3['val_acc'][-1]*100),
        'final_aug_acc': float(hist_aug_v3['val_acc'][-1]*100),
        'final_aug_val_loss': float(hist_aug_v3['val_loss'][-1])
    }
    with open('notebooks/outputs/cifar10_v3_metrics.json', 'w') as f:
        json.dump(metrics_v3, f, indent=2)

    print(f"\n✅ CIFAR-10 v3 Completed! Final Augmented Val Loss: {hist_aug_v3['val_loss'][-1]:.4f} | Val Acc: {hist_aug_v3['val_acc'][-1]*100:.2f}%")
