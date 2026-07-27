import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random, os, time, json

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# Hyperparameters
BATCH_SIZE   = 128
EPOCHS       = 15      # Efficient 15 epochs for quick, high-accuracy comparison
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA  = 0.2
LABEL_SMOOTH = 0.1
NUM_WORKERS  = 2

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# Transforms
transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

transform_improved = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), value='random'),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# Datasets (ImageFolder for fast local loading)
data_train_path = './data/temp_cifar10/cifar10/train'
data_test_path  = './data/temp_cifar10/cifar10/test'

train_base = torchvision.datasets.ImageFolder(root=data_train_path, transform=transform_base)
train_imp  = torchvision.datasets.ImageFolder(root=data_train_path, transform=transform_improved)
test_set   = torchvision.datasets.ImageFolder(root=data_test_path,  transform=transform_test)

loader_base_train = DataLoader(train_base, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
loader_imp_train  = DataLoader(train_imp,  batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
loader_test       = DataLoader(test_set,   batch_size=256,        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Mixup
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# SimpleCNN (from original comparison)
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
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,  64,  pool=True),
            ConvBlock(64, 128, pool=True),
            ConvBlock(128, 256, pool=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ResNet-18
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18CIFAR, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Training Helpers
def train_one_epoch(model, loader, criterion, optimizer, scheduler, use_mixup=False):
    model.train()
    total_loss, correct, total = 0., 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        if use_mixup:
            imgs_m, y_a, y_b, lam = mixup_data(imgs, labels, alpha=MIXUP_ALPHA)
            outputs = model(imgs_m)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            correct += (lam * (outputs.argmax(1) == y_a).sum().item() + (1 - lam) * (outputs.argmax(1) == y_b).sum().item())
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
        
    scheduler.step()
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0., 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

def train_model(name, model, train_loader, use_mixup=False):
    print(f'\n' + '='*55)
    print(f'  Training: {name}')
    print('='*55)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH if use_mixup else 0.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_t = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, use_mixup=use_mixup)
        va_loss, va_acc = evaluate(model, loader_test, criterion)
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)
        
        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f'  Epoch {epoch:02d}/{EPOCHS}  '
                  f'train_loss={tr_loss:.4f}  train_acc={tr_acc*100:.1f}%  '
                  f'val_loss={va_loss:.4f}  val_acc={va_acc*100:.1f}%', flush=True)
                  
    elapsed = time.time() - start_t
    print(f'\n  ✔ Final Test Accuracy ({name}): {history["val_acc"][-1]*100:.2f}% | Best: {max(history["val_acc"])*100:.2f}% (in {elapsed:.1f}s)')
    return model, history

if __name__ == '__main__':
    print("Starting Improved CIFAR-10 Experiment...")
    model_baseline = SimpleCNN().to(DEVICE)
    model_baseline, hist_baseline = train_model('Baseline SimpleCNN', model_baseline, loader_base_train, use_mixup=False)

    model_resnet = ResNet18CIFAR().to(DEVICE)
    model_resnet, hist_resnet = train_model('Improved ResNet-18 + Mixup', model_resnet, loader_imp_train, use_mixup=True)

    # Plot
    epochs_range = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, hist_baseline['train_loss'], 'b--', label='Baseline Train Loss')
    axes[0].plot(epochs_range, hist_baseline['val_loss'], 'b-', label='Baseline Val Loss')
    axes[0].plot(epochs_range, hist_resnet['train_loss'], 'g--', label='ResNet-18 Train Loss')
    axes[0].plot(epochs_range, hist_resnet['val_loss'], 'g-', label='ResNet-18 Val Loss')
    axes[0].set_title('Training & Validation Loss Comparison', fontsize=12)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, [acc*100 for acc in hist_baseline['train_acc']], 'b--', label='Baseline Train Acc')
    axes[1].plot(epochs_range, [acc*100 for acc in hist_baseline['val_acc']], 'b-', label='Baseline Val Acc')
    axes[1].plot(epochs_range, [acc*100 for acc in hist_resnet['train_acc']], 'g--', label='ResNet-18 Train Acc')
    axes[1].plot(epochs_range, [acc*100 for acc in hist_resnet['val_acc']], 'g-', label='ResNet-18 Val Acc')
    axes[1].set_title('Training & Validation Accuracy Comparison (%)', fontsize=12)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('notebooks/improved_resnet18_comparison.png', dpi=120, bbox_inches='tight')
    print("Saved notebooks/improved_resnet18_comparison.png")

    # Summary
    final_base = hist_baseline['val_acc'][-1] * 100
    final_res  = hist_resnet['val_acc'][-1] * 100
    best_base  = max(hist_baseline['val_acc']) * 100
    best_res   = max(hist_resnet['val_acc']) * 100

    gap_base = (hist_baseline['train_acc'][-1] - hist_baseline['val_acc'][-1]) * 100
    gap_res  = (hist_resnet['train_acc'][-1] - hist_resnet['val_acc'][-1]) * 100

    print('\n' + '='*55)
    print('      ACCURACY IMPROVEMENT SUMMARY — CIFAR-10')
    print('='*55)
    print(f'  {"Metric":<30} {"Baseline SimpleCNN":>18}  {"Improved ResNet-18":>18}')
    print('-'*55)
    print(f'  {"Final Test Accuracy":<30} {final_base:>17.2f}%  {final_res:>17.2f}%')
    print(f'  {"Best Test Accuracy":<30} {best_base:>17.2f}%  {best_res:>17.2f}%')
    print(f'  {"Overfitting Gap (Train - Val)":<30} {gap_base:>17.2f}%  {gap_res:>17.2f}%')
    print('='*55)
    print(f'  Accuracy Improvement:  +{final_res - final_base:.2f}%')
    print(f'  Generalization Gap:    {gap_base:.2f}% -> {gap_res:.2f}%')
    print('='*55)
