import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

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
    model_weight_path = r'G:\model\optizationCNN\optimizationcnn.pth'
    save_dir = r'G:\model\optizationCNN'
    os.makedirs(save_dir, exist_ok=True)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_ds = ImageDataset(test_folder, test_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    class_names = test_ds.unique_classes
    num_classes = len(class_names)

    model = CNNModel(num_classes).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    print("Performing predictions...")
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds, all_labels, all_probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)

    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    csv_path = os.path.join(save_dir, 'evaluation_report.csv')
    df_report.to_csv(csv_path, encoding='utf-8-sig')
    print(f"CSV report saved to: {csv_path}")

    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(16, 14))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title('optimizationCNN (Counts)')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(save_dir, 'optimizationCNN_counts.png'), dpi=300, bbox_inches='tight')
    plt.show()

    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(16, 14))
    sns.heatmap(conf_matrix_percent, annot=True, fmt=".2f", cmap="Greens", xticklabels=class_names,
                yticklabels=class_names)
    plt.title('optimizationCNN (Percentages)')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(save_dir, 'optimizationCNN_percent.png'), dpi=300, bbox_inches='tight')
    plt.show()

    lb = LabelBinarizer()
    lb.fit(range(num_classes))
    y_true_one_hot = lb.transform(all_labels)
    plt.figure(figsize=(12, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], all_probs[:, i])
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right", fontsize='x-small', ncol=2)
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()