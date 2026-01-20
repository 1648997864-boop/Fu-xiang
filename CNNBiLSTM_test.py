import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

BASE_FOLDER = r"G:\LSTM_after"
TIME_DIRS = ["0.5s", "1s", "2s", "4s"]
SUBSET = "evaluation"
MODEL_PATH = r"G:\model\Bilstm\best_cnn_bilstm.pth"
NUM_CLASSES = 210
IMG_SIZE = 224
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            nn.Linear(512 * 2, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        b, t, c, h, w = x.size()
        c_in = x.view(b * t, c, h, w)
        c_out = self.cnn(c_in)
        r_in = c_out.view(b, t, -1)
        r_out, _ = self.lstm(r_in)
        return self.classifier(r_out[:, -1, :])


def build_file_list(base, time_dirs, subset):
    samples = defaultdict(dict)
    ref_folder = os.path.join(base, time_dirs[0], subset)
    if not os.path.exists(ref_folder):
        raise FileNotFoundError(f"找不到参考目录: {ref_folder}")

    all_fnames = [f for f in os.listdir(ref_folder) if f.lower().endswith(".png") and '-' in f]

    all_class_names = sorted(list(set([f.split('-', 1)[0] for f in all_fnames])))
    class_to_idx = {name: i for i, name in enumerate(all_class_names)}

    final_list = []
    print(f"正在跨时间维度匹配样本 (子集: {subset})...")

    for fname in all_fnames:
        prefix = fname.rsplit('-', 1)[0]
        class_name = prefix.split('-', 1)[0]
        label = class_to_idx[class_name]

        paths = []
        match_success = True
        for t in time_dirs:
            target_fname = f"{prefix}-{t}.png"
            p = os.path.join(base, t, subset, target_fname)
            if os.path.exists(p):
                paths.append(p)
            else:
                match_success = False
                break

        if match_success:
            final_list.append((paths, label, prefix))

    print(f"匹配成功: 找到 {len(final_list)} 组完整的四维时空序列")
    return final_list, all_class_names


class MultiTimeDataset(Dataset):
    def __init__(self, file_list):
        self.data = file_list
        self.tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        paths, label, key = self.data[idx]
        imgs = [self.tf(Image.open(p).convert("RGB")) for p in paths]
        return torch.stack(imgs), torch.tensor(label), key

def test_model():
    RESULTS_DIR = r"G:\model\Bilstm"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    file_list, class_names = build_file_list(BASE_FOLDER, TIME_DIRS, SUBSET)
    dataset = MultiTimeDataset(file_list)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = CNN_BiLSTM_Model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_labels, all_preds, all_keys = [], [], []

    print("开始")
    with torch.no_grad():
        for x, y, keys in loader:
            x = x.to(DEVICE)
            outputs = model(x)
            preds = outputs.argmax(1)

            all_labels.extend(y.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_keys.extend(keys)

    acc = accuracy_score(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True,
                                        zero_division=0)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(os.path.join(RESULTS_DIR, "classification_report.csv"), encoding='utf-8-sig')

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"CNN-BiLSTM Confusion Matrix (Counts) - Acc: {acc:.4f}")
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(RESULTS_DIR, "cm_counts.png"), dpi=300, bbox_inches='tight')

    plt.figure(figsize=(16, 14))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title("CNN-BiLSTM Confusion Matrix (Normalized %)")
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(RESULTS_DIR, "cm_percent.png"), dpi=300, bbox_inches='tight')
    plt.show()

    df_preds = pd.DataFrame({"ID_Prefix": all_keys, "True_Label": all_labels, "Predicted_Label": all_preds})
    df_preds.to_csv(os.path.join(RESULTS_DIR, "detailed_predictions.csv"), index=False)

    print(f"结果已保存至: {RESULTS_DIR}")


if __name__ == "__main__":
    test_model()