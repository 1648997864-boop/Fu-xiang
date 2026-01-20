import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import time
from torch.amp import GradScaler, autocast

class SpatiotemporalDataset(Dataset):
    def __init__(self, base_root, time_dirs, subset, transform=None, class_to_idx=None):
        self.base_root = base_root
        self.time_dirs = time_dirs
        self.subset = subset
        self.transform = transform

        ref_dir = os.path.join(base_root, time_dirs[0], subset)
        if not os.path.exists(ref_dir):
            raise FileNotFoundError(f"无法找到参考目录: {ref_dir}")

        all_files = [f for f in os.listdir(ref_dir) if f.lower().endswith('.png') and '-' in f]

        if class_to_idx is None:
            all_names = [f.split('-', 1)[0] for f in all_files]
            self.unique_classes = sorted(list(set(all_names)))
            self.class_to_idx = {name: i for i, name in enumerate(self.unique_classes)}
        else:
            self.class_to_idx = class_to_idx

        self.valid_list = []
        print(f"正在解析 {subset} 集并匹配 {time_dirs} 时间维度样本...")

        for fname in all_files:
            try:
                prefix = fname.rsplit('-', 1)[0]
                class_name = prefix.split('-', 1)[0]
                label = self.class_to_idx[class_name]

                paths = []
                match_success = True

                for t_dir in time_dirs:
                    target_fname = f"{prefix}-{t_dir}.png"
                    p = os.path.join(base_root, t_dir, subset, target_fname)

                    if os.path.exists(p):
                        paths.append(p)
                    else:
                        match_success = False
                        break

                if match_success:
                    self.valid_list.append((paths, label))
            except Exception:
                continue

        print(f"{subset} 集匹配完成: 找到了 {len(self.valid_list)} 组序列。")

    def __len__(self):
        return len(self.valid_list)

    def __getitem__(self, idx):
        paths, label = self.valid_list[idx]
        sequence = []
        for p in paths:
            img = Image.open(p).convert('RGB')
            if self.transform:
                img = self.transform(img)
            sequence.append(img)
        return torch.stack(sequence), torch.tensor(label, dtype=torch.long)

class CNN_BiLSTM_Model(nn.Module):
    def __init__(self, num_classes=22):
        super(CNN_BiLSTM_Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.feature_dim = 256 * 14 * 14
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=512,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),
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

def main():
    BASE_ROOT = r'G:\LSTM_after'
    TIME_DIRS = ['0.5s', '1s', '2s', '4s']
    SAVE_DIR = r'G:\model\Bilstm'
    os.makedirs(SAVE_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 1e-4

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("准备数据中...")
    train_ds = SpatiotemporalDataset(BASE_ROOT, TIME_DIRS, 'training', transform)
    val_ds = SpatiotemporalDataset(BASE_ROOT, TIME_DIRS, 'validation', transform,
                                   class_to_idx=train_ds.class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(train_ds.class_to_idx)
    model = CNN_BiLSTM_Model(num_classes=num_classes).to(DEVICE)

    with open(os.path.join(SAVE_DIR, 'bilstm_classes.txt'), 'w') as f:
        for name, idx in train_ds.class_to_idx.items():
            f.write(f"{idx}: {name}\n")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler()

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        start_t = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                v_loss += criterion(outputs, labels).item()
                _, pred = torch.max(outputs, 1)
                v_total += labels.size(0)
                v_correct += (pred == labels).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = v_loss / len(val_loader)
        val_acc = v_correct / v_total

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Time: {time.time() - start_t:.1f}s | "
              f"Tr_Loss: {avg_train_loss:.4f} | Val_Loss: {avg_val_loss:.4f} | Val_Acc: {val_acc:.4f}")

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_cnn_bilstm.pth'))
            print(f"最佳模型已保存")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.savefig(os.path.join(SAVE_DIR, 'bilstm_training_plot.png'))
    plt.show()


if __name__ == "__main__":
    main()