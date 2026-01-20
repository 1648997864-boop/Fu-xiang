import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=5, padding=2),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            conv_block(3, 16), conv_block(16, 32),
            conv_block(32, 64), conv_block(64, 64)
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


class ImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
        all_names = [f.split('-', 1)[0] for f in self.files]
        self.unique_classes = sorted(list(set(all_names)))
        self.class_to_idx = {name: i for i, name in enumerate(self.unique_classes)}
        self.labels = [self.class_to_idx[name] for name in all_names]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.files[idx])
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.labels[idx], dtype=torch.long)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_folder = r'G:\LO3a\test'
    model_path = r'G:\model\baselineCNN\BaselineCNN_model.pth'
    save_dir = r'G:\model\baselineCNN\test_results'
    os.makedirs(save_dir, exist_ok=True)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = ImageDataset(test_folder, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    class_names = test_dataset.unique_classes
    num_classes = len(class_names)

    model = BaselineCNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    df_report = pd.DataFrame(report_dict).transpose()

    csv_path = os.path.join(save_dir, 'BaselineCNN_test.csv')
    df_report.to_csv(csv_path, encoding='utf-8-sig')

    print(f"表格已保存到: {csv_path}")

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"BaselineCNN (Counts) - Acc: {acc:.4f}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'BaselineCNN_counts.png'), dpi=300)
    plt.show()

    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title("BaselineCNN (Percentages)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'BaselineCNN_percent.png'), dpi=300)
    plt.show()