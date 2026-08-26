#!/usr/bin/env python
"""seg_eval: 分割质量分析与 ROI 裁剪管线
- nnUNet fold 0 预测 Mask vs 金标准 Mask
- 按类别/位置/域分层报告 Dice/HD95
- 可视化失败病例 (低 Dice)
- 金标准 vs 预测 Mask 提取 ROI 的分类差异 (与 3D CNN baseline CNN 对比)
"""
import json, glob, os, warnings
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from collections import defaultdict
from scipy import ndimage
warnings.filterwarnings('ignore')

BASE = 'datasets/nnUNet_results/Dataset001_GCT/nnUNetTrainer250Epochs__nnUNetPlans__3d_fullres'
SRC = 'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'
OUT = 'experiments/seg_eval'
os.makedirs(f'{OUT}/pred_masks', exist_ok=True)

def dice(pred, gt):
    inter = np.logical_and(pred > 0, gt > 0).sum()
    union = (pred > 0).sum() + (gt > 0).sum()
    return 2 * inter / union if union > 0 else 0.0

def hd95(pred, gt, spacing):
    """95% Hausdorff 距离 (mm)"""
    from scipy.ndimage import distance_transform_edt
    if pred.sum() == 0 or gt.sum() == 0:
        return float('inf')
    pred_edt = distance_transform_edt(~(pred > 0), sampling=spacing)
    gt_edt = distance_transform_edt(~(gt > 0), sampling=spacing)
    d1 = pred_edt[gt > 0]
    d2 = gt_edt[pred > 0]
    return max(np.percentile(d1, 95), np.percentile(d2, 95))

def main():
    # 检查 fold 0 是否完成
    fold0 = f'{BASE}/fold_0'
    ckpts = glob.glob(f'{fold0}/checkpoint_final.pth')
    if not ckpts:
        print('ERROR: fold 0 not finished yet. Checkpoint not found.')
        return
    print('fold 0 checkpoint found:', ckpts[0])

    # 找 fold 0 的验证集 (nnUNet 用 4/5 训练, 1/5 验证)
    # nnUNet 自动 splits; 验证集是 fold 0 的 val
    val_dir = f'{fold0}/validation'
    if not os.path.exists(val_dir):
        # 需要先跑 validation
        print('validation dir not found, run: nnUNetv2_train 1 3d_fullres 0 --val')
        return

    # 收集验证集病例
    pred_files = glob.glob(f'{val_dir}/*.nii.gz')
    print(f'validation predictions: {len(pred_files)}')

    # 直接用文件名映射类别
    rows = []
    for pf in sorted(pred_files):
        stem = os.path.basename(pf).replace('.nii.gz','')
        cat = 'MTs' if stem.startswith('MT_') else ('ITs' if stem.startswith('IT_') else 'MGCTs')
        gt_path = f'{SRC}/{cat}/Masks/{stem}.nii.gz'
        if not os.path.exists(gt_path):
            continue
        pred = nib.load(pf).get_fdata()
        gt = nib.load(gt_path).get_fdata()
        img = nib.load(f'{SRC}/{cat}/Images/{stem}.nii.gz')
        spacing = (abs(img.affine[0,0]), abs(img.affine[1,1]), abs(img.affine[2,2]))
        d = dice(pred, gt)
        h = hd95(pred, gt, spacing)
        rows.append({'stem': stem, 'cat': cat, 'dice': d, 'hd95': h})
        # 保存预测 mask 供 ROI 裁剪
        nib.save(nib.Nifti1Image((pred > 0).astype(np.uint8), img.affine), f'{OUT}/pred_masks/{stem}.nii.gz')

    import pandas as pd
    df = pd.DataFrame(rows)
    print(f'\n=== 总体分割质量 ===')
    print(f'n={len(df)}, mean Dice={df["dice"].mean():.4f}, median={df["dice"].median():.4f}')
    print(f'mean HD95={df["hd95"].mean():.2f}mm')
    print(f'Dice>=0.85 占比: {(df["dice"]>=0.85).mean():.3f}')

    print(f'\n=== 按类别 ===')
    for cat in ['MTs','ITs','MGCTs']:
        sub = df[df['cat']==cat]
        print(f'{cat}: n={len(sub)}, Dice={sub["dice"].mean():.4f}+-{sub["dice"].std():.4f}, HD95={sub["hd95"].mean():.2f}mm')

    print(f'\n=== 失败病例 (Dice<0.7) ===')
    fail = df[df['dice'] < 0.7].sort_values('dice')
    print(f'count: {len(fail)}')
    print(fail.head(10).to_string())

    df.to_csv(f'{OUT}/seg_quality.csv', index=False)
    print(f'\nsaved -> {OUT}/seg_quality.csv')

if __name__ == '__main__':
    main()
