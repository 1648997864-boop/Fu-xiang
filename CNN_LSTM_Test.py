import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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


class MyDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.files = [f for f in os.listdir(folder) if f.lower().endswith('.png') and '-' in f]

        all_names = [f.split('-', 1)[0] for f in self.files]
        self.unique_classes = sorted(list(set(all_names)))
        self.class_to_idx = {name: i for i, name in enumerate(self.unique_classes)}
        self.labels = [self.class_to_idx[name] for name in all_names]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.files[idx])
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img.unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TEST_FOLDER = r"G:\LO3a\test"
    MODEL_PATH = r"G:\model\CNN_LSTM\best_cnn_lstm.pth"
    SAVE_DIR = r"G:\model\CNN_LSTM\test_results"
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到权重文件 {MODEL_PATH}")
        exit()

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_ds = MyDataset(TEST_FOLDER, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    class_names = test_ds.unique_classes
    num_classes = len(class_names)

    model = CNNLSTM(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded. Testing on {num_classes} classes.")

    y_true, y_pred = [], []

    print("Running Inference...")
    with torch.no_grad():
        for imgs, labels in test_loader:
            outputs = model(imgs.to(device))
            preds = torch.argmax(outputs, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())

    acc = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report_dict).transpose()

    csv_path = os.path.join(SAVE_DIR, "test_metrics.csv")
    df_report.to_csv(csv_path, encoding='utf-8-sig')


    print(f"报告已保存至: {csv_path}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Counts) - Acc: {acc:.4f}')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(SAVE_DIR, "CNN&LSTM_counts.png"), dpi=300, bbox_inches='tight')
    plt.show()

    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Normalized %)')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(SAVE_DIR, "CNN&LSTM_percent.png"), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"所有结果已保存至: {SAVE_DIR}")