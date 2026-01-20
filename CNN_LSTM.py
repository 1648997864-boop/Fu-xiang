import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy as np

CONFIG = {
    'train_folder': r'G:\LO3a\train',
    'val_folder': r'G:\LO3a\val',
    'save_dir': r'G:\model\CNN_LSTM',
    'input_size': (224, 224),
    'batch_size': 64,
    'epochs': 20,
    'lr': 1e-4,
    'num_workers': 4
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)

class SequenceDataset(Dataset):
    def __init__(self, folder, transform=None, class_to_idx=None):
        self.folder = folder
        self.transform = transform

        self.files = [f for f in os.listdir(folder) if f.lower().endswith('.png') and '-' in f]

        all_names = [f.split('-', 1)[0] for f in self.files]

        if class_to_idx is None:
            self.unique_classes = sorted(list(set(all_names)))
            self.class_to_idx = {name: i for i, name in enumerate(self.unique_classes)}
        else:
            self.class_to_idx = class_to_idx

        self.labels = [self.class_to_idx[name] for name in all_names]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.files[idx])
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image.unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

class CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.flatten_dim = 256 * 14 * 14
        self.lstm = nn.LSTM(input_size=self.flatten_dim, hidden_size=512, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        b, t, c, h, w = x.size()
        c_in = x.view(b * t, c, h, w)
        c_out = self.cnn(c_in)

        r_in = c_out.view(b, t, -1)
        r_out, _ = self.lstm(r_in)
        return self.classifier(r_out[:, -1, :])


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Using Device: {device}")

    transform = transforms.Compose([
        transforms.Resize(CONFIG['input_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading Datasets...")
    train_ds = SequenceDataset(CONFIG['train_folder'], transform)
    val_ds = SequenceDataset(CONFIG['val_folder'], transform, class_to_idx=train_ds.class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)

    num_classes = len(train_ds.class_to_idx)
    model = CNNLSTM(num_classes=num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_loss = float('inf')
    start_time = time.time()

    print(f"Starting Training ({num_classes} classes)...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                v_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                v_total += labels.size(0)
                v_correct += (pred == labels).sum().item()

        tr_loss, tr_acc = running_loss / len(train_loader), correct / total
        vl_loss, vl_acc = v_loss / len(val_loader), v_correct / v_total

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        print(
            f"Epoch {epoch + 1}/{CONFIG['epochs']} - Tr_Loss: {tr_loss:.4f}, Tr_Acc: {tr_acc:.4f} | Vl_Loss: {vl_loss:.4f}, Vl_Acc: {vl_acc:.4f}")

        scheduler.step(vl_loss)

        if vl_loss < best_loss:
            best_loss = vl_loss
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'best_cnn_lstm.pth'))
            print("Model improved and saved.")

    epochs_range = range(1, CONFIG['epochs'] + 1)
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='#1f77b4', lw=2)
    plt.plot(epochs_range, history['val_loss'], label='Val Loss', color='#ff7f0e', lw=2, linestyle='--')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc', color='#2ca02c', lw=2)
    plt.plot(epochs_range, history['val_acc'], label='Val Acc', color='#d62728', lw=2, linestyle='--')
    plt.title('CNN&LSTM', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['save_dir'], 'CNN&LSTM.png'))
    plt.show()

    print(f"Total Time: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == '__main__':
    train_model()