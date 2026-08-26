#!/usr/bin/env python
"""clean-protocol cascade DL evaluation (40 cases) (40 例, fold0-val ∩ test)
用 clean 重训的 3-seed dist/gray CNN 集成, recomputes the GT vs pred-mask scenario for the distance-prior 3D CNN
- dist: 2ch [灰度, 距离]; gray: 1ch [灰度]
- reference (old protocol) pred_mask ACC: dist 0.800 / gray 0.750 (+5pt)
"""
import os, sys, json, warnings
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix
warnings.filterwarnings('ignore')

sys.path.insert(0, 'experiments/cnn_framework')
from prepare_data import crop_roi_triple, window_and_normalize
from train_cnn_variants import Simple3DCNN

OUT = 'experiments/clean_ensemble'
SRC = 'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'
PRED_MASK_DIR = 'experiments/seg_eval/pred_masks'
SEEDS = [42, 123, 2024]
CAT2DIR = {'MTs': 'MTs', 'ITs': 'ITs', 'MGCTs': 'MGCTs'}

def cat_of(stem):
    for cat in CAT2DIR:
        if os.path.exists(f'{SRC}/{cat}/Images/{stem}.nii.gz'):
            return cat
    return None

@torch.no_grad()
def predict_ens(models, x, device):
    probs = []
    for m in models:
        m.eval()
        p = torch.softmax(m(x.to(device)), dim=1).cpu().numpy()
        probs.append(p)
    return np.mean(probs, axis=0)

def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else 'dist'
    channels = {'dist': [0, 2], 'gray': [0]}[variant]
    print(f'=== clean cascade eval: variant={variant} channels={channels} ===')

    cmp = np.loadtxt('experiments/seg_eval/roi_classification_compare.csv', delimiter=',',
                     dtype=str, skiprows=1, usecols=(0,))
    stems = [s for s in cmp]
    print(f'cases: {len(stems)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    for s in SEEDS:
        m = Simple3DCNN(n_classes=3, in_channels=len(channels)).to(device)
        m.load_state_dict(torch.load(f'{OUT}/results/clean_{variant}_s{s}.pt', map_location=device))
        models.append(m)
    print(f'loaded clean 3-seed {variant} models on {device}')

    X_gt, X_pred, y_true, skipped = [], [], [], []
    for stem in stems:
        cat = cat_of(stem)
        if cat is None:
            skipped.append(stem); continue
        cls = {'MTs': 0, 'ITs': 1, 'MGCTs': 2}[cat]
        img = nib.load(f'{SRC}/{cat}/Images/{stem}.nii.gz').get_fdata()
        gt = nib.load(f'{SRC}/{cat}/Masks/{stem}.nii.gz').get_fdata()
        pm = f'{PRED_MASK_DIR}/{stem}.nii.gz'
        if not os.path.exists(pm):
            skipped.append(stem); continue
        pred = nib.load(pm).get_fdata()
        img_w = window_and_normalize(img)
        roi_gt = crop_roi_triple(img_w, gt)
        roi_pred = crop_roi_triple(img_w, pred)
        if roi_gt is None or roi_pred is None:
            skipped.append(stem); continue
        X_gt.append(roi_gt); X_pred.append(roi_pred); y_true.append(cls)
    print(f'usable: {len(X_gt)} (skipped {len(skipped)}: {skipped[:10]})')

    y_true = np.array(y_true)
    X_gt = np.stack(X_gt)[:, channels]; X_gt[:, 0] /= 255.0
    X_pred = np.stack(X_pred)[:, channels]; X_pred[:, 0] /= 255.0
    t_gt = torch.tensor(X_gt, dtype=torch.float32)
    t_pred = torch.tensor(X_pred, dtype=torch.float32)

    p_gt = predict_ens(models, t_gt, device)
    p_pred = predict_ens(models, t_pred, device)

    print(f'\n=== clean_ensembleg 级联 DL (clean 3-seed {variant}) ===')
    res = {}
    for tag, p in [('GT-mask', p_gt), ('Pred-mask', p_pred)]:
        pred = p.argmax(1)
        acc = accuracy_score(y_true, pred)
        auc = roc_auc_score(y_true, p, multi_class='ovr') if len(set(y_true)) >= 3 else np.nan
        rec = recall_score(y_true, pred, average=None)
        cm = confusion_matrix(y_true, pred)
        print(f'{tag:9s} n={len(y_true)} ACC={acc:.4f} AUC={auc:.4f} recall={np.round(rec,3)}')
        print(cm)
        res[tag.lower().replace('-', '_')] = {'acc': float(acc), 'auc': float(auc),
                                               'recall': rec.tolist(), 'cm': cm.tolist()}
    agree = (p_gt.argmax(1) == p_pred.argmax(1)).mean()
    diff = np.where(p_gt.argmax(1) != p_pred.argmax(1))[0]
    print(f'agreement={agree:.4f} n_disagree={len(diff)}')
    for i in diff:
        print(f'  {stems[i]}: true={y_true[i]} gt={p_gt[i].argmax()} pred={p_pred[i].argmax()}')
    res['agreement'] = float(agree); res['n_disagree'] = int(len(diff))
    res['n_cases'] = int(len(y_true))
    res['baselines_old_protocol'] = {'gray_cnn_cascade': 0.750, 'dist_cnn_cascade': 0.800}
    with open(f'{OUT}/results_cascade_clean_{variant}.json', 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f'saved -> {OUT}/results_cascade_clean_{variant}.json')

if __name__ == '__main__':
    main()
