#!/usr/bin/env python
"""seg_eval 第三部分: 金标准 Mask vs nnUNet 预测 Mask 提取 ROI 的分类对比
- 用训练好的 3D CNN (3D CNN baseline) 分别对 金标准 ROI 和 预测 Mask ROI 分类
- 量化分割误差对分类的影响传导
"""
import json, os, warnings
import numpy as np
import nibabel as nib
import torch
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
warnings.filterwarnings('ignore')

SRC = 'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'
OUT = 'experiments/seg_eval'
TARGET_SIZE = 64
MARGIN = 8

def window_and_normalize(arr, level=35, width=350):
    lower = level - width / 2.0
    upper = level + width / 2.0
    arr = np.clip(arr, lower, upper)
    arr = (arr - lower) / (upper - lower) * 255.0
    return arr.astype(np.float32)

def crop_roi_from_mask(img_arr, mask_arr):
    """用给定 mask 裁剪 ROI"""
    coords = np.argwhere(mask_arr > 0)
    if len(coords) == 0:
        return None
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    z0 = max(0, z0 - MARGIN); y0 = max(0, y0 - MARGIN); x0 = max(0, x0 - MARGIN)
    z1 = min(img_arr.shape[0], z1 + MARGIN); y1 = min(img_arr.shape[1], y1 + MARGIN); x1 = min(img_arr.shape[2], x1 + MARGIN)
    crop = img_arr[z0:z1, y0:y1, x0:x1]
    zoom = [TARGET_SIZE / s for s in crop.shape]
    return ndimage.zoom(crop, zoom, order=1, mode='nearest')

def main():
    # 检查模型
    model_path = 'experiments/cnn_baseline/model_base_best.pt'
    if not os.path.exists(model_path):
        print('ERROR: 3D CNN baseline model not found:', model_path)
        return
    import sys
    sys.path.insert(0, 'experiments/cnn_baseline')
    from train_cnn import Simple3DCNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Simple3DCNN(n_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print('model loaded')

    # 预测 Mask (seg_eval 第一部分生成)
    pred_dir = f'{OUT}/pred_masks'
    if not os.path.exists(pred_dir) or len(os.listdir(pred_dir)) == 0:
        print('ERROR: pred masks not found. Run eval_seg.py first.')
        return

    split = json.load(open('datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
    test_stems = set(split['MT']['test'] + split['IT']['test'] + split['MGCT']['test']) - {'MT_411'}
    cat_map = {}
    for cat in ['MTs', 'ITs', 'MGCTs']:
        for f in os.listdir(f'{SRC}/{cat}/Masks'):
            cat_map[f.replace('.nii.gz','')] = cat

    # 对测试集每个病例: 金标准 ROI vs 预测 ROI → 模型预测
    print('computing ROI classification comparison...')
    rows = []
    with torch.no_grad():
        for stem in sorted(test_stems):
            if stem not in cat_map:
                continue
            cat = cat_map[stem]
            img = nib.load(f'{SRC}/{cat}/Images/{stem}.nii.gz').get_fdata()
            gt_mask = nib.load(f'{SRC}/{cat}/Masks/{stem}.nii.gz').get_fdata()
            pred_mask_path = f'{pred_dir}/{stem}.nii.gz'
            if not os.path.exists(pred_mask_path):
                continue
            pred_mask = nib.load(pred_mask_path).get_fdata()
            img_w = window_and_normalize(img)
            roi_gt = crop_roi_from_mask(img_w, gt_mask)
            roi_pred = crop_roi_from_mask(img_w, pred_mask)
            if roi_gt is None or roi_pred is None:
                continue
            def classify(roi):
                x = torch.tensor(roi/255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                prob = torch.softmax(model(x), dim=1).cpu().numpy()[0]
                return prob.argmax(), prob
            pred_gt, prob_gt = classify(roi_gt)
            pred_pred, prob_pred = classify(roi_pred)
            rows.append({'stem': stem, 'cat': cat, 'true': {'MTs':0,'ITs':1,'MGCTs':2}[cat],
                         'pred_gt': pred_gt, 'pred_pred': pred_pred,
                         'agree': pred_gt == pred_pred})

    import pandas as pd
    df = pd.DataFrame(rows)
    print(f'\n=== 分割 Mask 对分类的影响 (测试集 {len(df)} 例) ===')
    agree_rate = df['agree'].mean()
    print(f'金标准 vs 预测 Mask 分类一致率: {agree_rate:.4f}')

    # 金标准 ROI 分类准确率 vs 预测 Mask ROI 分类准确率
    acc_gt = (df['pred_gt'] == df['true']).mean()
    acc_pred = (df['pred_pred'] == df['true']).mean()
    print(f'金标准 ROI 分类 ACC: {acc_gt:.4f}')
    print(f'预测 Mask ROI 分类 ACC: {acc_pred:.4f}')
    print(f'差异: {acc_pred - acc_gt:+.4f}')

    # 不一致病例
    diff = df[~df['agree']]
    print(f'\n不一致病例: {len(diff)}')
    print(diff.head(10).to_string())

    df.to_csv(f'{OUT}/roi_classification_compare.csv', index=False)
    print(f'\nsaved -> {OUT}/roi_classification_compare.csv')

if __name__ == '__main__':
    main()
