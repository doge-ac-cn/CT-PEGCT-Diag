#!/usr/bin/env python
"""cnn_framework: mask 先验形态消融 (变体 C: 灰度+距离通道; 变体 D: 灰度+mask+距离 3ch)
- 与 dl_ablation 完全同协议; 通过 --variant 选择通道
- 对照: A=1ch灰度 (0.803/0.875), B=2ch灰度+二值mask (0.808/0.887)
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

VARIANTS = {'dist': [0, 2], 'both': [0, 1, 2]}

class ROIDataset(Dataset):
    def __init__(self, manifest, split_set, channels):
        self.items = [m for m in manifest if m['id'] in split_set]
        self.channels = channels
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        m = self.items[idx]
        three = np.load(f'experiments/cnn_framework/data/images/{m["id"]}.npy')
        three[0] = three[0] / 255.0
        x = torch.tensor(three[self.channels], dtype=torch.float32)
        return x, m['class']

class Simple3DCNN(nn.Module):
    def __init__(self, n_classes=3, in_channels=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1), nn.BatchNorm3d(128), nn.ReLU(), nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))
    def forward(self, x):
        return self.classifier(self.features(x))

def train(model, loader, optimizer, criterion, device):
    model.train()
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_prob = [], []
    for x, y in loader:
        x = x.to(device)
        all_prob.append(torch.softmax(model(x), dim=1).cpu().numpy())
        all_y.extend(y.numpy())
    all_prob = np.concatenate(all_prob); all_y = np.array(all_y)
    y_pred = all_prob.argmax(axis=1)
    return (accuracy_score(all_y, y_pred), roc_auc_score(all_y, all_prob, multi_class='ovr'),
            confusion_matrix(all_y, y_pred), all_y, all_prob)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=list(VARIANTS), default='dist')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    channels = VARIANTS[args.variant]

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'variant={args.variant} channels={channels} device={device}')

    manifest = json.load(open('experiments/cnn_framework/data/manifest.json'))
    split = json.load(open('datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
    train_set = set(split['MT']['train'] + split['IT']['train'] + split['MGCT']['train']) - {'MT_411'}
    test_set = set(split['MT']['test'] + split['IT']['test'] + split['MGCT']['test']) - {'MT_411'}
    train_loader = DataLoader(ROIDataset(manifest, train_set, channels), batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(ROIDataset(manifest, test_set, channels), batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f'train: {len(train_loader.dataset)}, test: {len(test_loader.dataset)}')

    model = Simple3DCNN(n_classes=3, in_channels=len(channels)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0
    for epoch in range(args.epochs):
        loss = train(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc, auc, cm, _, _ = evaluate(model, test_loader, device)
            print(f'epoch {epoch+1}: loss={loss:.4f} | test ACC={acc:.4f} AUC={auc:.4f}', flush=True)
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), f'experiments/cnn_framework/model_{args.variant}_s{args.seed}_best.pt')

    acc, auc, cm, all_y, all_prob = evaluate(model, test_loader, device)
    print(f'\n=== cnn_framework {args.variant} FINAL (channels={channels}) ===')
    print(f'ACC={acc:.4f} AUC(ovr)={auc:.4f} best_auc={best_auc:.4f}')
    print(cm)
    results = {'variant': args.variant, 'channels': channels, 'acc': float(acc), 'auc': float(auc),
               'best_auc': float(best_auc), 'cm': cm.tolist(),
               'per_class_auc': {n: float(roc_auc_score((all_y == i).astype(int), all_prob[:, i]))
                                 for i, n in enumerate(['MT', 'IT', 'MGCT'])}}
    with open(f'experiments/cnn_framework/results_{args.variant}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('saved ->', f'experiments/cnn_framework/results_{args.variant}.json')

if __name__ == '__main__':
    main()
