#!/usr/bin/env python
"""cascade_eval 图表: 级联误差传导可视化 (113 例)
Fig A: GT ROI vs Pred ROI 三分类概率对比 (每例连线, 按类别着色)
Fig B: 不一致病例分类概率变化热图
Fig C: MGCT→MT 误判代表病例的分割过分割可视化 (CT + GT mask + Pred mask)
"""
import os, json, warnings
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.isdir(BASE):
    raise RuntimeError('BASE path not found; run from a checkout of the repository')
OUT = f'{BASE}/experiments/cascade_eval/figs'
os.makedirs(OUT, exist_ok=True)
SRC = f'{BASE}/datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'
CAT2CLS = {'MTs': 0, 'ITs': 1, 'MGCTs': 2}
CLS_NAMES = ['MT', 'IT', 'MGCT']
COLORS = {'MT': '#4C72B0', 'IT': '#DD8452', 'MGCT': '#55A868'}

def main():
    z = np.load(f'{BASE}/experiments/bootstrap/cascade_prob_113.npz')
    y_te, pred_gt, pred_pr = z['y_te'], z['pred_gt'], z['pred_pr']
    prob_gt, prob_pr = z['prob_gt'], z['prob_pr']

    # 病例顺序 (与 eval_cached 一致: sorted stems ∩ pred_map)
    split = json.load(open(f'{BASE}/datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
    test_stems = set(split['MT']['test'] + split['IT']['test'] + split['MGCT']['test']) - {'MT_411'}
    cat_of = {}
    for cat in ['MTs', 'ITs', 'MGCTs']:
        for f in os.listdir(f'{SRC}/{cat}/Masks'):
            cat_of[f.replace('.nii.gz', '')] = cat
    pred_map = json.load(open(f'{BASE}/experiments/cascade_eval/pred_map.json'))
    stems = sorted(s for s in test_stems if s in cat_of and s in pred_map)
    assert len(stems) == len(y_te), f'{len(stems)} vs {len(y_te)}'

    diff_idx = np.where(pred_gt != pred_pr)[0]

    # ---- Fig A: 概率变化 (仅不一致病例) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for cls in range(3):
        ax = axes[cls]
        idxs = [i for i in diff_idx if y_te[i] == cls]
        for i in idxs:
            ax.plot([0, 1], [prob_gt[i, cls], prob_pr[i, cls]], 'o-', alpha=0.7,
                    color=COLORS[CLS_NAMES[cls]], lw=1.2, ms=4)
        ax.set_title(f'True={CLS_NAMES[cls]} (n={len(idxs)})')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['GT ROI', 'Pred ROI'])
        ax.set_ylabel('P(true class)')
        ax.axhline(1/3, color='gray', ls='--', lw=0.8, alpha=0.6)
        ax.set_ylim(0, 1.05)
    fig.suptitle(f'Cascade error propagation: 13/113 disagreements (GT ROI vs Pred ROI)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f'{OUT}/figA_prob_shift.png', dpi=150)
    plt.close(fig)

    # ---- Fig B: 不一致病例概率热图 ----
    fig, ax = plt.subplots(figsize=(9, 6))
    n = len(diff_idx)
    data = np.vstack([prob_gt[diff_idx], prob_pr[diff_idx]])
    im = ax.imshow(data, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['GT ROI', 'Pred ROI'])
    ax.set_xticks(np.arange(3 * n))
    ax.set_xticklabels([f'{CLS_NAMES[j % 3]}' for j in range(3 * n)], fontsize=7)
    # 每例分隔线
    for i in range(1, n):
        ax.axvline(3 * i - 0.5, color='white', lw=0.6, alpha=0.5)
    labels = [f'{stems[i]} ({cat_of[stems[i]]})' for i in diff_idx]
    ax.set_title('Disagreement cases: P(class) GT ROI (top) vs Pred ROI (bottom)')
    cb = fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/figB_prob_heatmap.png', dpi=150)
    plt.close(fig)

    # ---- Fig C: MGCT→MT 代表病例过分割可视化 (前3例) ----
    mgct_diffs = [i for i in diff_idx if y_te[i] == 2 and pred_gt[i] == 2 and pred_pr[i] == 0]
    sel = mgct_diffs[:3]
    fig = plt.figure(figsize=(14, 4.5))
    for k, i in enumerate(sel):
        stem = stems[i]; cat = cat_of[stem]
        img = nib.load(f'{SRC}/{cat}/Images/{stem}.nii.gz').get_fdata()
        gt = nib.load(f'{SRC}/{cat}/Masks/{stem}.nii.gz').get_fdata() > 0
        pr = nib.load(pred_map[stem]).get_fdata() > 0
        # 选肿瘤中心轴
        zz = int(np.argwhere(gt | pr)[:, 0].mean())
        sl = img[zz]
        vmin, vmax = np.percentile(sl[sl > -1024], [5, 95])
        ax = fig.add_subplot(1, len(sel), k + 1)
        ax.imshow(sl, cmap='gray', vmin=vmin, vmax=vmax)
        ax.contour(gt[zz], colors='lime', linewidths=1.2, levels=[0.5])
        ax.contour(pr[zz], colors='red', linewidths=1.2, levels=[0.5])
        ax.set_title(f'{stem}\nGT P(MGCT)={prob_gt[i,2]:.2f} → Pred ROI P(MGCT)={prob_pr[i,2]:.2f}\nGT P(MT)={prob_gt[i,0]:.2f} → Pred P(MT)={prob_pr[i,0]:.2f}', fontsize=9)
        ax.axis('off')
    fig.suptitle('MGCT→MT misclassifications: GT mask (green) vs nnUNet pred mask (red) — over-segmentation', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(f'{OUT}/figC_mgct_overseg.png', dpi=150)
    plt.close(fig)

    # 保存不一致列表
    rows = [{'stem': stems[i], 'cat': cat_of[stems[i]], 'true': int(y_te[i]),
             'pred_gt': int(pred_gt[i]), 'pred_pred': int(pred_pr[i]),
             'prob_gt': prob_gt[i].round(3).tolist(), 'prob_pred': prob_pr[i].round(3).tolist()}
            for i in diff_idx]
    json.dump(rows, open(f'{OUT}/disagreements.json', 'w'), indent=2)
    print(f'figs saved to {OUT}: figA_prob_shift.png, figB_prob_heatmap.png, figC_mgct_overseg.png')
    print(f'disagreements: {len(rows)}; MGCT→MT: {len([r for r in rows if r["true"]==2 and r["pred_gt"]==2 and r["pred_pred"]==0])}')

if __name__ == '__main__':
    main()
