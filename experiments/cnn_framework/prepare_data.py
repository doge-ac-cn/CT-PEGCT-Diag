#!/usr/bin/env python
"""cnn_framework 数据准备: 3 通道 ROI (灰度 + 二值 mask + 距离变换软 mask)
- 距离变换: 肿瘤内部=到边界的欧氏距离(归一化到[0,1]), 外部=0
- 输出: (3,64,64,64), 供 train_cnn_variants.py 按变体取通道子集
"""
import os, json
import numpy as np
import nibabel as nib
from scipy import ndimage

SRC = 'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'
OUT = 'experiments/cnn_framework/data'
os.makedirs(f'{OUT}/images', exist_ok=True)

TARGET_SIZE = 64
MARGIN = 8

def window_and_normalize(arr, level=35, width=350):
    lower = level - width / 2.0
    upper = level + width / 2.0
    arr = np.clip(arr, lower, upper)
    arr = (arr - lower) / (upper - lower) * 255.0
    return arr.astype(np.float32)

def crop_roi_triple(img_arr, mask_arr):
    coords = np.argwhere(mask_arr > 0)
    if len(coords) == 0:
        return None
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    z0 = max(0, z0 - MARGIN); y0 = max(0, y0 - MARGIN); x0 = max(0, x0 - MARGIN)
    z1 = min(img_arr.shape[0], z1 + MARGIN); y1 = min(img_arr.shape[1], y1 + MARGIN); x1 = min(img_arr.shape[2], x1 + MARGIN)
    crop_img = img_arr[z0:z1, y0:y1, x0:x1]
    crop_mask = mask_arr[z0:z1, y0:y1, x0:x1]
    zoom_factors = [TARGET_SIZE / s for s in crop_img.shape]
    resized_img = ndimage.zoom(crop_img, zoom_factors, order=1, mode='nearest')
    resized_mask = ndimage.zoom(crop_mask, zoom_factors, order=0, mode='nearest')
    resized_mask = (resized_mask > 0.5).astype(np.float32)
    # 距离变换: 内部到边界距离, 归一化
    dist_in = ndimage.distance_transform_edt(resized_mask)
    dmax = dist_in.max()
    dist_norm = dist_in / dmax if dmax > 0 else dist_in
    return np.stack([resized_img, resized_mask, dist_norm], axis=0)

def main():
    cats = {'MTs': 0, 'ITs': 1, 'MGCTs': 2}
    split = json.load(open('datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
    manifest = []
    for cat, cls in cats.items():
        stems = split[cat[:-1]]['train'] + split[cat[:-1]]['test']
        for stem in stems:
            if stem == 'MT_411':
                continue
            img = nib.load(f'{SRC}/{cat}/Images/{stem}.nii.gz').get_fdata()
            mask = nib.load(f'{SRC}/{cat}/Masks/{stem}.nii.gz').get_fdata()
            roi = crop_roi_triple(window_and_normalize(img), mask)
            if roi is None:
                print(f'SKIP empty mask: {stem}')
                continue
            np.save(f'{OUT}/images/{stem}.npy', roi)
            manifest.append({'id': stem, 'class': cls})
    with open(f'{OUT}/manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'DONE: {len(manifest)} 3-channel ROIs -> {OUT}/')
    chk = np.load(f'{OUT}/images/{manifest[0]["id"]}.npy')
    print('example shape:', chk.shape, '| ch1 mask uniq:', np.unique(chk[1]),
          '| ch2 dist range:', chk[2].min(), chk[2].max())

if __name__ == '__main__':
    main()
