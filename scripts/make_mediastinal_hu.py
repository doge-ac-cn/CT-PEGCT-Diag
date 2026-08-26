#!/usr/bin/env python
"""make_mediastinal_hu: 生成 Figure 7(B) 所需数据（修复缺失的 mediastinal_hu.npz）

从 clean 测试集纵隔病例的 CT + 金标准 mask 中提取病灶内体素 HU 值，
保存为 experiments/clean_ensemble/mediastinal_hu.npz（键 = Patient_ID，值 = HU 数组）。
下游 scripts/make_fig7_subgroup.py 读取该文件绘制 IT vs MT HU 分布。
"""
import numpy as np
import pandas as pd
import nibabel as nib

SRC = 'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'
EXP = 'experiments/clean_ensemble'
OUT = f'{EXP}/mediastinal_hu.npz'

stems = list(np.load(f'{EXP}/clean_stems.npy', allow_pickle=True))

# 读取各病例部位信息
meta_rows = []
for cat in ['MTs', 'ITs', 'MGCTs']:
    md = pd.read_excel(f'{SRC}/{cat}/{cat[:-1]}_patient.xlsx')
    for _, r in md.iterrows():
        meta_rows.append({'Patient_ID': r['Patient_ID'],
                          'cat': cat,
                          'location': str(r['Location']).strip().lower()})
meta = pd.DataFrame(meta_rows).set_index('Patient_ID')

# 筛选纵隔病例（与 make_fig7_subgroup.py 口径一致）
med_stems = [s for s in stems if meta.loc[s, 'location'] == 'mediastinum']
print(f'mediastinal test cases: {len(med_stems)}')

hu_map = {}
for stem in med_stems:
    cat = meta.loc[stem, 'cat']
    img = nib.load(f'{SRC}/{cat}/Images/{stem}.nii.gz').get_fdata()
    m = nib.load(f'{SRC}/{cat}/Masks/{stem}.nii.gz').get_fdata() > 0
    vox = img[m]
    hu_map[stem] = vox.astype(np.float64)
    print(f'  {stem}: {vox.size} voxels, median {np.median(vox):.0f} HU')

np.savez(OUT, **hu_map)
print('saved ->', OUT)
