#!/usr/bin/env python
"""clean_ensemble: 无泄漏协议重训 — hold-out 验证集选择 best epoch
修复 train_ablation.py 的测试集模型选择缺陷:
- 从 7:3 划分的 train (448) 中按类别分层切出 val (15%, ~67 例)
- 训练期间只在 val 上评估, 选 val ACC 最高 epoch 保存权重
- 测试集 (193) 只评估一次 (val-best epoch), 不参与任何选择
用法:
  python train_clean.py --dim 2d --arch 18 --seed 42 [--val_frac 0.15]
"""
import os, json, argparse, sys, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, resnet34, resnet50, resnet101
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
warnings.filterwarnings('ignore')

sys.path.insert(0, 'experiments/dl_ablation')
from train_ablation import ARCHS, make_resnet3d, ROIDataset, BATCH, EPOCHS, LR

OUT = 'experiments/clean_ensemble'
os.makedirs(OUT, exist_ok=True)
os.makedirs(f'{OUT}/results', exist_ok=True)

def split_val(manifest, val_frac, seed):
    """从 manifest 的 train 病例中按类别分层切出 val"""
    train_items = [m for m in manifest if m['split'] == 'train']
    stems = np.array([m['stem'] for m in train_items])
    cls = np.array([m['cls'] for m in train_items])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(sss.split(stems, cls))
    va_set = set(stems[va_idx])
    return va_set

class ROIDatasetSplit(Dataset):
    """与 ROIDataset 相同, 但 split 参数接受集合 (train 子集)"""
    def __init__(self, manifest, keep_stems, dim, roi=None, oversample=False, augment=True):
        self.items = []
        for m in manifest:
            if m['stem'] not in keep_stems:
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
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, cls = self.items[i]
        x = np.load(p)
        if self.augment and x.shape[0] <= 3 and np.random.rand() < 0.5:
            k = np.random.randint(0, 4)
            x = np.stack([np.rot90(ch, k) for ch in x])
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy()
        return torch.from_numpy(x).float(), cls

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', choices=['2d', '25d', '3d'], required=True)
    ap.add_argument('--arch', choices=['18', '34', '50', '101'], required=True)
    ap.add_argument('--roi', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    manifest = json.load(open('experiments/dl_ablation/data/manifest_ablation.json'))
    va_set = split_val(manifest, args.val_frac, args.seed)
    tr_set = set(m['stem'] for m in manifest if m['split'] == 'train') - va_set
    te_set = set(m['stem'] for m in manifest if m['split'] == 'test')

    train_ds = ROIDatasetSplit(manifest, tr_set, args.dim, args.roi, augment=True)
    val_ds = ROIDatasetSplit(manifest, va_set, args.dim, args.roi, augment=False)
    test_ds = ROIDatasetSplit(manifest, te_set, args.dim, args.roi, augment=False)

    batch = BATCH[args.dim][args.arch]
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=2)

    if args.dim == '3d':
        model = make_resnet3d(args.arch)
    else:
        model = ARCHS[args.arch](weights=None, num_classes=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f'[CLEAN] {args.dim} arch={args.arch} seed={args.seed} '
          f'train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}', flush=True)

    best = None
    val_curve = []
    for ep in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                all_y, all_p = [], []
                for x, y in val_loader:
                    out = model(x.to(device))
                    all_y.extend(y.tolist()); all_p.extend(torch.softmax(out, 1).cpu().numpy())
            acc = accuracy_score(all_y, np.argmax(all_p, 1))
            val_curve.append({'epoch': ep + 1, 'val_acc': float(acc),
                              'val_auc': float(roc_auc_score(all_y, all_p, multi_class='ovr'))})
            print(f'  ep {ep+1:3d} val_acc={acc:.4f}', flush=True)
            if best is None or acc > best[0]:
                best = (acc, ep + 1, model.state_dict().copy())

    # 测试集只评估一次 (val-best epoch)
    model.load_state_dict(best[2])
    model.eval()
    with torch.no_grad():
        all_y, all_p = [], []
        for x, y in test_loader:
            out = model(x.to(device))
            all_y.extend(y.tolist()); all_p.extend(torch.softmax(out, 1).cpu().numpy())
    y = np.array(all_y); p = np.array(all_p)
    pred = p.argmax(1)
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, p, multi_class='ovr')
    rec = recall_score(y, pred, average=None).tolist()
    cm = confusion_matrix(y, pred).tolist()

    tag = args.tag or f'clean_{args.dim}_r{args.arch}' + (f'_roi{args.roi}' if args.dim == '3d' else '') + (f'_s{args.seed}' if args.seed != 42 else '')
    res = {'dim': args.dim, 'arch': args.arch, 'roi': args.roi, 'seed': args.seed,
           'val_frac': args.val_frac, 'n_train': len(train_ds), 'n_val': len(val_ds), 'n_test': len(test_ds),
           'best_val_epoch': best[1], 'best_val_acc': float(best[0]), 'val_curve': val_curve,
           'test_acc': float(acc), 'test_auc_ovr': float(auc), 'test_recall': rec, 'test_cm': cm}
    with open(f'{OUT}/results/{tag}.json', 'w') as f:
        json.dump(res, f, indent=2)
    torch.save(best[2], f'{OUT}/results/{tag}.pt')
    print(f'RESULT [{tag}] val_best_ep={best[1]} test ACC={acc:.4f} AUC={auc:.4f} recall={np.round(rec,3)}', flush=True)

if __name__ == '__main__':
    main()
