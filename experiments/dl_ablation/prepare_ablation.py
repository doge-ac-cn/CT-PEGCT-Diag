#!/usr/bin/env python
"""DL ablation experiments (2D/2.5D/3D ResNet): 统一数据准备
生成输入:
  2D  : 病灶最大层面 (GT mask 沿层轴投影面积最大) -> 单层 224x224 (3通道复制)
  2.5D: 最大层面 ±1 共 3 层 -> 3通道 224x224
  3D  : bbox+margin 裁剪 -> roi_size^3 (32/48/64/96/128)
统一预处理: 窗宽窗位(35/350) + 归一化 0-255
"""
import os, json
import numpy as np
import nibabel as nib
from scipy import ndimage

SRC = 'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'
OUT = 'experiments/dl_ablation/data'
os.makedirs(OUT, exist_ok=True)

TARGET_2D = 224
MARGIN = 8
ROI_SIZES = [32, 48, 64, 96, 128]

def layer_axis(img):
    return int(np.argmax(np.abs(img.affine.diagonal()[:3])))

def window_and_normalize(arr, level=35, width=350):
    lower = level - width / 2.0
    upper = level + width / 2.0
    arr = np.clip(arr, lower, upper)
    arr = (arr - lower) / (upper - lower) * 255.0
    return arr.astype(np.float32)

def crop_resize(arr, mask2d=None, size=TARGET_2D, margin=MARGIN):
    """2D bbox 裁剪 + resize"""
    if mask2d is None:
        mask2d = arr > 0
    coords = np.argwhere(mask2d > 0)
    if len(coords) == 0:
        return None
    y0, x0 = coords.min(axis=0); y1, x1 = coords.max(axis=0) + 1
    y0 = max(0, y0 - margin); x0 = max(0, x0 - margin)
    y1 = min(arr.shape[0], y1 + margin); x1 = min(arr.shape[1], x1 + margin)
    crop = arr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    zoom = [size / s for s in crop.shape]
    return ndimage.zoom(crop, zoom, order=1, mode='nearest')

def crop_roi3d(img_arr, mask_arr, roi_size, margin=MARGIN):
    coords = np.argwhere(mask_arr > 0)
    if len(coords) == 0:
        return None
    lo = coords.min(axis=0); hi = coords.max(axis=0) + 1
    lo = np.maximum(0, lo - margin); hi = np.minimum(np.array(img_arr.shape), hi + margin)
    crop = img_arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    zoom = [roi_size / s for s in crop.shape]
    return ndimage.zoom(crop, zoom, order=1, mode='nearest')

def main():
    cats = {'MTs': 0, 'ITs': 1, 'MGCTs': 2}
    split = json.load(open('datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
    manifest = []
    for cat, cls in cats.items():
        key = cat[:-1]  # MT/IT/MGCT
        stems = split[key]['train'] + split[key]['test']
        for stem in stems:
            if stem == 'MT_411':
                continue
            img_path = f'{SRC}/{cat}/Images/{stem}.nii.gz'
            mask_path = f'{SRC}/{cat}/Masks/{stem}.nii.gz'
            img = nib.load(img_path)
            arr = img.get_fdata()
            mask = nib.load(mask_path).get_fdata()
            ax = layer_axis(img)
            arr_w = window_and_normalize(arr)

            # ---- 2D: 最大层面 ----
            proj = mask.sum(axis=tuple(i for i in range(3) if i != ax))
            zmax = int(proj.argmax())
            sl = [slice(None)] * 3
            sl[ax] = zmax
            img2d = arr_w[tuple(sl)]
            mask2d = mask[tuple(sl)]
            if mask2d.sum() > 0:
                roi2d = crop_resize(img2d, mask2d, TARGET_2D)
            else:
                roi2d = None
            if roi2d is not None:
                np.save(f'{OUT}/{stem}_2d.npy', np.stack([roi2d]*3, axis=0).astype(np.float32))

            # ---- 2.5D: 最大层面 ±1 ----
            planes = []
            ok = True
            for dz in [-1, 0, 1]:
                zz = zmax + dz
                if zz < 0 or zz >= arr.shape[ax]:
                    ok = False
                    break
                sl = [slice(None)] * 3
                sl[ax] = zz
                planes.append(arr_w[tuple(sl)])
            if ok:
                # 联合 bbox (三层 mask 并集) 保证病灶在画面内
                joint = np.zeros_like(planes[0])
                for dz in [-1, 0, 1]:
                    zz = zmax + dz
                    sl = [slice(None)] * 3
                    sl[ax] = zz
                    joint = joint + (mask[tuple(sl)] > 0)
                rois25d = []
                for p in planes:
                    r = crop_resize(p, joint, TARGET_2D)
                    if r is None:
                        ok = False
                        break
                    rois25d.append(r)
                if ok:
                    np.save(f'{OUT}/{stem}_25d.npy', np.stack(rois25d, axis=0).astype(np.float32))

            # ---- 3D: 多档 ROI ----
            for rs in ROI_SIZES:
                roi3d = crop_roi3d(arr_w, mask, rs)
                if roi3d is not None:
                    np.save(f'{OUT}/{stem}_3d_{rs}.npy', roi3d[None].astype(np.float32))

            manifest.append({'stem': stem, 'cat': cat, 'cls': cls, 'split': 'train' if stem in split[key]['train'] else 'test'})
            if len(manifest) % 100 == 0:
                print(f'processed {len(manifest)}', flush=True)

    json.dump(manifest, open(f'{OUT}/manifest.json', 'w'), indent=2)
    # 同时写出 manifest_ablation.json：下游脚本（train_ablation.py 等）统一读取该文件，
    # 避免因文件名不一致导致找不到 manifest 的断点
    json.dump(manifest, open(f'{OUT}/manifest_ablation.json', 'w'), indent=2)
    print(f'done: {len(manifest)} cases')

if __name__ == '__main__':
    main()
