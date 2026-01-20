import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None, class_to_idx=None):
        self.folder_path = folder_path
        self.transform = transform

        self.filenames = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]

        all_file_classes = [f.split('-', 1)[0] for f in self.filenames]

        if class_to_idx is None:
            unique_classes = sorted(list(set(all_file_classes)))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        else:
            self.class_to_idx = class_to_idx

        self.labels = [self.class_to_idx[name] for name in all_file_classes]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.filenames[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=5, padding=2),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.5)
            )

        self.features = nn.Sequential(
            conv_block(3, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            conv_block(64, 64)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


if __name__ == '__main__':
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_folder = r'G:\LO3a\train'
    val_folder = r'G:\LO3a\val'
    save_path = r'G:\model\baselineCNN'
    os.makedirs(save_path, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ImageDataset(train_folder, train_transform)
    val_dataset = ImageDataset(val_folder, val_transform, class_to_idx=train_dataset.class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    num_classes = len(train_dataset.class_to_idx)
    print(f"Total classes detected: {num_classes}")

    model = BaselineCNN(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 20
    best_acc = 0.0
    train_loss_history = []
    val_loss_history = []

    print("Starting training...")
    for epoch in range(epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = validate(model, val_loader, criterion, device)

        train_loss_history.append(t_loss)
        val_loss_history.append(v_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'BaselineCNN_model.pth'))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_loss_history, label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), val_loss_history, label='Val Loss', marker='s')
    plt.title('baselineCNN')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    print(f"Loss curve saved to {save_path}\\loss_curve.png")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f}s | Best Val Acc: {best_acc:.4f}")