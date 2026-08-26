#!/usr/bin/env python
"""lodo_train_dl.py — LODO（留一解剖部位域）2D ResNet18 ensemble

协议与论文一致: epochs=60, lr=1e-3 AdamW cosine, test-selected epoch, 5 seeds 概率平均
自定义 split: train=其余大域, test=留出域 (选epoch), rare=罕见域(最终评估,不参与选epoch)
用法: python lodo_train_dl.py --domain ovary [--seeds 42,0,1,123,2024]
输出: experiments/lodo/results/{domain}/s{seed}.json/.pt + ensemble.json + all_seeds.json

依赖:
  - experiments/dl_ablation/train_ablation.py（ARCHS/BATCH/EPOCHS/LR，与论文 DL 训练同一框架）
  - experiments/dl_ablation/data/{stem}_2d.npy（由 prepare_ablation.py 生成）
  - experiments/dl_ablation/data/manifest_ablation.json
  - experiments/lodo/lodo_splits.json
GPU: 建议 NVIDIA GPU（RTX 3090 上单域 5 seeds 约 16 min）
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, 'results')
DATA = os.path.join(REPO, 'experiments', 'dl_ablation', 'data')
LODO_JSON = os.path.join(HERE, 'lodo_splits.json')

sys.path.insert(0, os.path.join(REPO, 'experiments', 'dl_ablation'))
from train_ablation import ARCHS, BATCH, EPOCHS, LR  # noqa: E402


class LodoDataset(Dataset):
    def __init__(self, manifest, stems, dim, augment):
        self.items = []
        stem_set = set(stems)
        for m in manifest:
            stem = m['stem']
            if stem not in stem_set:
                continue
            p = f'{DATA}/{stem}_2d.npy' if dim == '2d' else f'{DATA}/{stem}_25d.npy'
            if os.path.exists(p):
                self.items.append((p, m['cls']))
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


def train_one(domain, seed, manifest, tr_stems, te_stems, ra_stems, dim='2d', arch='18'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds = LodoDataset(manifest, tr_stems, dim, augment=True)
    test_ds = LodoDataset(manifest, te_stems, dim, augment=False)
    rare_ds = LodoDataset(manifest, ra_stems, dim, augment=False)
    batch = BATCH[dim][arch]
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=2)

    model = ARCHS[arch](weights=None, num_classes=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    print(f'[{domain} s{seed}] train={len(train_ds)} test(sel)={len(test_ds)} rare={len(rare_ds)}', flush=True)

    best = None
    for ep in range(EPOCHS):
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
        if (ep + 1) % 10 == 0 or ep == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                all_y, all_p = [], []
                for x, y in test_loader:
                    out = model(x.to(device))
                    all_y.extend(y.tolist()); all_p.extend(torch.softmax(out, 1).cpu().numpy())
            acc = accuracy_score(all_y, np.argmax(all_p, 1))
            print(f'  ep {ep + 1:3d} tr_acc={corr / tot:.4f} te_acc={acc:.4f}', flush=True)
            if best is None or acc > best[0]:
                best = (acc, ep + 1, np.array(all_y), np.array(all_p))

    _, best_ep, y, p = best
    pred = p.argmax(1)
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, p, multi_class='ovr')
    rec = recall_score(y, pred, average=None).tolist()
    cm = confusion_matrix(y, pred).tolist()

    # 罕见域最终评估
    rare_res = None
    if len(rare_ds) > 0:
        model.eval()
        with torch.no_grad():
            ry, rp = [], []
            for x, yy in DataLoader(rare_ds, batch_size=batch, shuffle=False):
                out = model(x.to(device))
                ry.extend(yy.tolist()); rp.extend(torch.softmax(out, 1).cpu().numpy())
        ry, rp = np.array(ry), np.array(rp)
        rare_res = {'acc': float(accuracy_score(ry, rp.argmax(1))),
                    'auc': float(roc_auc_score(ry, rp, multi_class='ovr')),
                    'recall': [float(r) for r in recall_score(ry, rp.argmax(1), average=None)],
                    'cm': confusion_matrix(ry, rp.argmax(1)).tolist()}

    res = {'domain': domain, 'seed': seed, 'best_epoch': best_ep,
           'test_acc': float(acc), 'test_auc': float(auc), 'test_recall': [float(r) for r in rec],
           'test_cm': cm, 'rare': rare_res, 'n_train': len(train_ds), 'n_test': len(test_ds)}
    d_out = os.path.join(OUT, domain)
    os.makedirs(d_out, exist_ok=True)
    json.dump(res, open(os.path.join(d_out, f's{seed}.json'), 'w'), indent=2)
    torch.save(model.state_dict(), os.path.join(d_out, f's{seed}.pt'))
    print(f'RESULT [{domain} s{seed}] best_ep={best_ep} ACC={acc:.4f} AUC={auc:.4f} recall={np.round(rec, 3)}', flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True)
    ap.add_argument('--seeds', default='42,0,1,123,2024')
    args = ap.parse_args()
    domain = args.domain
    seeds = [int(s) for s in args.seeds.split(',')]
    manifest = json.load(open(os.path.join(DATA, 'manifest_ablation.json')))
    lodo = json.load(open(LODO_JSON))
    tr = lodo[domain]['train']; te = lodo[domain]['test']; ra = lodo[domain]['rare_test']
    all_res = []
    for seed in seeds:
        all_res.append(train_one(domain, seed, manifest, tr, te, ra))
    json.dump(all_res, open(os.path.join(OUT, domain, 'all_seeds.json'), 'w'), indent=2)
    # 概率平均 ensemble (重新加载 .pt 推理)
    probs = []
    for seed in seeds:
        model = ARCHS['18'](weights=None, num_classes=3)
        model.load_state_dict(torch.load(os.path.join(OUT, domain, f's{seed}.pt'), map_location='cpu'))
        model.eval()
        with torch.no_grad():
            p = []
            for x, _ in DataLoader(LodoDataset(manifest, te + ra, '2d', augment=False), batch_size=64):
                p.append(torch.softmax(model(x), 1).numpy())
        probs.append(np.concatenate(p))
    p_ens = np.mean(probs, axis=0)
    y_ens = np.array([m['cls'] for m in manifest if m['stem'] in set(te + ra)])
    ens = {'domain': domain, 'n_test_plus_rare': len(y_ens),
           'test+rare_acc': float(accuracy_score(y_ens, p_ens.argmax(1))),
           'test+rare_auc': float(roc_auc_score(y_ens, p_ens, multi_class='ovr')),
           'test+rare_recall': [float(r) for r in recall_score(y_ens, p_ens.argmax(1), average=None)],
           'test+rare_cm': confusion_matrix(y_ens, p_ens.argmax(1)).tolist()}
    json.dump(ens, open(os.path.join(OUT, domain, 'ensemble.json'), 'w'), indent=2)
    print(f'ENSEMBLE [{domain}] ACC={ens["test+rare_acc"]:.4f} AUC={ens["test+rare_auc"]:.4f} '
          f'recall={np.round(ens["test+rare_recall"], 3)}', flush=True)


if __name__ == '__main__':
    main()
