import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import time
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

class LargeImageDataset(Dataset):
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
            self.unique_classes = sorted(list(self.class_to_idx.keys()))

        self.labels = [self.class_to_idx[name] for name in all_names]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.files[idx])
        image = Image.open(path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == '__main__':
    start_time = time.time()

    train_folder = r'G:\LO3a\train'
    val_folder = r'G:\LO3a\val'
    model_save_dir = r'G:\model\optizationCNN'
    os.makedirs(model_save_dir, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = LargeImageDataset(train_folder, train_transform)
    val_ds = LargeImageDataset(val_folder, train_transform, class_to_idx=train_ds.class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(train_ds.class_to_idx)
    print(f"Total detected classes: {num_classes}")

    print(f"Class Mapping: {train_ds.class_to_idx}")
    with open(os.path.join(model_save_dir, 'classes.txt'), 'w') as f:
        for name, idx in train_ds.class_to_idx.items():
            f.write(f"{idx}: {name}\n")

    model = CNNModel(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    epochs = 20
    train_losses, val_losses = [], []

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                v_loss += loss.item()

        avg_val_loss = v_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(model_save_dir, 'optimizationcnn.pth'))
    print(f"Model saved to {model_save_dir}")


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', color='red', linestyle='--', linewidth=2)
    plt.title('Training and Validation Loss Curve', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (CrossEntropy)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(model_save_dir, 'loss_curve.png'))
    print("Loss curve image saved.")
    plt.show()

    print(f"Total training time: {time.time() - start_time:.2f}s")