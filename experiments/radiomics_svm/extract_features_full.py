#!/usr/bin/env python
"""radiomics_svm: PyRadiomics 全量特征提取 (original + wavelet + LoG, 对齐 OnekeyAI 官方协议)
协议: 窗宽窗位(35/350) → 归一化 0-255 → 1mm³ 重采样 → 1316 特征
"""
import os, json, glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

SRC = 'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'
OUT = 'experiments/radiomics_svm'
os.makedirs(OUT, exist_ok=True)

def window_and_normalize(image_sitk, level=35, width=350):
    arr = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
    lower = level - width / 2.0
    upper = level + width / 2.0
    arr = np.clip(arr, lower, upper)
    arr = (arr - lower) / (upper - lower) * 255.0
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image_sitk)
    return out

def main():
    cats = {'MTs': 0, 'ITs': 1, 'MGCTs': 2}
    rows = []
    split = json.load(open('datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
    all_cases = []
    for cat, cls in cats.items():
        for stem in split[cat[:-1]]['train'] + split[cat[:-1]]['test']:
            all_cases.append((cat, stem, cls))

    extractor = featureextractor.RadiomicsFeatureExtractor(f'{OUT}/radiomics_params_full.json')

    for i, (cat, stem, cls) in enumerate(all_cases):
        img_path = f'{SRC}/{cat}/Images/{stem}.nii.gz'
        mask_path = f'{SRC}/{cat}/Masks/{stem}.nii.gz'
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f'SKIP missing: {stem}')
            continue
        try:
            img = sitk.ReadImage(img_path)
            mask = sitk.ReadImage(mask_path)
            img_w = window_and_normalize(img)
            feat = extractor.execute(img_w, mask)
            row = {'Patient_ID': stem, 'class': cls}
            for k, v in feat.items():
                if not k.startswith('diagnostics'):
                    row[k] = float(v)
            rows.append(row)
        except Exception as e:
            print(f'ERROR {stem}: {e}')
        if (i + 1) % 100 == 0:
            print(f'progress: {i+1}/{len(all_cases)}', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/features_full.csv', index=False)
    print(f'DONE: {len(df)} cases, {df.shape[1]-2} features -> {OUT}/features_full.csv')
    print('class distribution:', df['class'].value_counts().to_dict())

if __name__ == '__main__':
    main()
