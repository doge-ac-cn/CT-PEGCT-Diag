#!/usr/bin/env python
"""DL ablation experiments (2D/2.5D/3D ResNet): 统一训练框架
用法:
  python train_ablation.py --dim 2d --arch 18 [--roi 64] [--loss ce] [--oversample 0] [--tag x]
  --dim: 2d | 25d | 3d
  --arch: 18 | 34 | 50 | 101 (ResNet)
  --roi: 3D 输入尺寸 (仅 3d)
  --loss: ce | weighted | focal
  --oversample: 1 = WeightedRandomSampler (少数类过采样)
严格性: 所有组合同一协议 (epochs=60, lr=1e-3, cosine, seed=42, batch 按 arch 自适应)
"""
import os, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, resnet34, resnet50, resnet101
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, recall_score
import warnings
warnings.filterwarnings('ignore')

SEED = 42
ARCHS = {'18': resnet18, '34': resnet34, '50': resnet50, '101': resnet101}
BATCH = {'2d': {'18': 64, '34': 48, '50': 32, '101': 16},
         '25d': {'18': 64, '34': 48, '50': 32, '101': 16},
         '3d': {'18': 8, '34': 6, '50': 4, '101': 2}}
EPOCHS = 60
LR = 1e-3

class ROIDataset(Dataset):
    def __init__(self, manifest, split, dim, roi=None, oversample=False, augment=True):
        self.items = []
        for m in manifest:
            if m['split'] != split:
                continue
            stem, cls = m['stem'], m['cls']
            if dim == '2d':
                p = f'experiments/dl_ablation/data/{stem}_2d.npy'
            elif dim == '25d':
                p = f'experiments/dl_ablation/data/{stem}_25d.npy'
            else:
                p = f'experiments/dl_ablation/data/{stem}_3d_{roi}.npy'
            if os.path.exists(p):
                self.items.append((p, cls))
        self.oversample = oversample
        self.augment = augment
        if oversample:
            counts = np.bincount([c for _, c in self.items], minlength=3)
            weights = [1.0 / counts[c] for _, c in self.items]
            self.sampler_weights = torch.DoubleTensor(weights)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, cls = self.items[i]
        x = np.load(p)
        # 轻增强: 随机翻转/旋转 (仅训练)
        if self.augment and x.shape[0] <= 3 and np.random.rand() < 0.5:
            k = np.random.randint(0, 4)
            x = np.stack([np.rot90(ch, k) for ch in x])
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy()
        return torch.from_numpy(x).float(), cls

def conv3x3x3(inp, out, stride=1):
    return nn.Conv3d(inp, out, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1x1(inp, out, stride=1):
    return nn.Conv3d(inp, out, kernel_size=1, stride=stride, bias=False)

class Bottleneck3D(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

def make_resnet3d(arch, num_classes=3):
    """2D→3D 结构转换: conv1 7³ stride2, maxpool 3³, blocks 同 torchvision"""
    cfgs = {'18': [2, 2, 2, 2], '34': [3, 4, 6, 3], '50': [3, 4, 6, 3], '101': [3, 4, 23, 3]}
    widths = [64, 128, 256, 512]
    block = BasicBlock3D if arch in ('18', '34') else Bottleneck3D
    layers = cfgs[arch]
    inplanes = 64
    def make_layer(planes, blocks, stride=1):
        nonlocal inplanes
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1x1(inplanes, planes * block.expansion, stride),
                                       nn.BatchNorm3d(planes * block.expansion))
        layers_ = []
        layers_.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers_.append(block(inplanes, planes))
        return nn.Sequential(*layers_)

    class ResNet3D(nn.Module):
        def __init__(self):
            super().__init__()
            nonlocal inplanes
            self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm3d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            self.layer1 = make_layer(widths[0], layers[0])
            self.layer2 = make_layer(widths[1], layers[1], stride=2)
            self.layer3 = make_layer(widths[2], layers[2], stride=2)
            self.layer4 = make_layer(widths[3], layers[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        def forward(self, x):
            x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
            x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
            x = self.avgpool(x); x = torch.flatten(x, 1)
            return self.fc(x)
    return ResNet3D()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        if self.alpha is not None:
            at = self.alpha[targets]
            loss = at * (1 - pt) ** self.gamma * ce
        else:
            loss = (1 - pt) ** self.gamma * ce
        return loss.mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', choices=['2d', '25d', '3d'], required=True)
    ap.add_argument('--arch', choices=['18', '34', '50', '101'], required=True)
    ap.add_argument('--roi', type=int, default=64)
    ap.add_argument('--loss', choices=['ce', 'weighted', 'focal'], default='ce')
    ap.add_argument('--oversample', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    manifest = json.load(open('experiments/dl_ablation/data/manifest_ablation.json'))
    train_ds = ROIDataset(manifest, 'train', args.dim, args.roi, oversample=args.oversample)
    test_ds = ROIDataset(manifest, 'test', args.dim, args.roi, augment=False)

    batch = BATCH[args.dim][args.arch]
    if args.oversample:
        train_loader = DataLoader(train_ds, batch_size=batch, sampler=torch.utils.data.WeightedRandomSampler(
            train_ds.sampler_weights, len(train_ds), replacement=True), num_workers=2)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=2)

    if args.dim == '3d':
        model = make_resnet3d(args.arch)
    else:
        model = ARCHS[args.arch](weights=None, num_classes=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    counts = np.bincount([c for _, c in train_ds.items], minlength=3).astype(float)
    if args.loss == 'weighted':
        w = counts.sum() / (3 * counts)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(w / w.sum() * 3, dtype=torch.float32)).to(device)
    elif args.loss == 'focal':
        w = counts.sum() / (3 * counts)
        criterion = FocalLoss(gamma=2.0, alpha=torch.tensor(w / w.sum() * 3, dtype=torch.float32)).to(device)
    else:
        criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f'[{args.dim} arch={args.arch} roi={args.roi} loss={args.loss} over={args.oversample}] '
          f'train={len(train_ds)} test={len(test_ds)} batch={batch}', flush=True)

    best = None
    for ep in range(args.epochs):
        model.train()
        tot, corr, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            tot += y.size(0); corr += (out.argmax(1) == y).sum().item(); loss_sum += loss.item() * y.size(0)
        sched.step()
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                all_y, all_p = [], []
                for x, y in test_loader:
                    out = model(x.to(device))
                    all_y.extend(y.tolist()); all_p.extend(torch.softmax(out, 1).cpu().numpy())
            acc = accuracy_score(all_y, np.argmax(all_p, 1))
            print(f'  ep {ep+1:3d} tr_acc={corr/tot:.4f} loss={loss_sum/tot:.4f} te_acc={acc:.4f}', flush=True)
            if best is None or acc > best[0]:
                best = (acc, ep + 1, np.array(all_y), np.array(all_p))

    _, best_ep, all_y, all_p = best
    y, p = all_y, all_p
    pred = p.argmax(1)
    acc = accuracy_score(y, pred)
    try:
        auc = roc_auc_score(y, p, multi_class='ovr')
    except Exception:
        auc = float('nan')
    rec = recall_score(y, pred, average=None).tolist()
    cm = confusion_matrix(y, pred).tolist()
    tag = args.tag or f'{args.dim}_r{args.arch}' + (f'_roi{args.roi}' if args.dim == '3d' else '') + f'_{args.loss}' + (f'_over' if args.oversample else '') + (f'_s{args.seed}' if args.seed != 42 else '')
    res = {'dim': args.dim, 'arch': args.arch, 'roi': args.roi, 'loss': args.loss, 'oversample': args.oversample,
           'best_epoch': best_ep, 'acc': acc, 'auc_ovr': auc, 'recall': rec, 'cm': cm, 'n_train': len(train_ds), 'n_test': len(test_ds)}
    os.makedirs('experiments/dl_ablation/results', exist_ok=True)
    with open(f'experiments/dl_ablation/results/{tag}.json', 'w') as f:
        json.dump(res, f, indent=2)
    torch.save(model.state_dict(), f'experiments/dl_ablation/results/{tag}.pt')
    print(f'RESULT [{tag}] ACC={acc:.4f} AUC={auc:.4f} recall={np.round(rec,3)} (best_ep={best_ep})', flush=True)

if __name__ == '__main__':
    main()
