#!/usr/bin/env python
"""table1: 论文 Table 1 — 数据集临床特征描述表
按类别 (MT/IT/MGCT) × 性别/年龄/部位 + 训练/测试划分统计
输出: table1.csv + table1.md (论文 Table 1 素材)
"""
import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

SRC = 'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'
OUT = 'experiments/table1'
os.makedirs(OUT, exist_ok=True)

def load_meta():
    rows = []
    for cat, name in [('MTs', 'MT'), ('ITs', 'IT'), ('MGCTs', 'MGCT')]:
        md = pd.read_excel(f'{SRC}/{cat}/{name}_patient.xlsx')
        gcol = [c for c in md.columns if 'ender' in c][0]
        acol = [c for c in md.columns if 'Age' in c][0]
        lcol = [c for c in md.columns if 'ocation' in c][0]
        for _, r in md.iterrows():
            rows.append({'Patient_ID': r['Patient_ID'], 'cat': name, 'cls': {'MT':0,'IT':1,'MGCT':2}[name],
                         'gender': str(r[gcol]).upper(), 'age_month': float(r[acol]),
                         'loc': str(r[lcol]).lower().strip()})
    return pd.DataFrame(rows)

def main():
    meta = load_meta()
    split = json.load(open('datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
    train_of, test_of = set(), set()
    for c in ['MT', 'IT', 'MGCT']:
        train_of |= set(split[c]['train'])
        test_of |= set(split[c]['test'])
    meta['split'] = meta['Patient_ID'].apply(lambda x: 'Train' if x in train_of else ('Test' if x in test_of else 'Excluded'))

    # 表 1: 类别 × 特征
    rows = []
    for cat in ['MT', 'IT', 'MGCT']:
        sub = meta[meta['cat'] == cat]
        rows.append({
            'Category': cat, 'N': len(sub),
            'Female_n': int((sub['gender'] == 'F').sum()),
            'Male_n': int((sub['gender'] == 'M').sum()),
            'Age_mean_months': round(sub['age_month'].mean(), 1),
            'Age_sd_months': round(sub['age_month'].std(), 1),
            'Age_median_months': float(sub['age_month'].median()),
            'Age_range': f"{int(sub['age_month'].min())}-{int(sub['age_month'].max())}",
            'Train_n': int((sub['split'] == 'Train').sum()),
            'Test_n': int((sub['split'] == 'Test').sum()),
        })
    # 部位 top3
    for cat in ['MT', 'IT', 'MGCT']:
        sub = meta[meta['cat'] == cat]
        top = sub['loc'].value_counts().head(3)
        rows[[i for i, r in enumerate(rows) if r['Category'] == cat][0]]['Top3_locations'] = ', '.join(
            f"{k.title()} {v}" for k, v in top.items())
    t1 = pd.DataFrame(rows)
    total = pd.DataFrame([{'Category': 'All', 'N': len(meta),
                           'Female_n': int((meta['gender']=='F').sum()), 'Male_n': int((meta['gender']=='M').sum()),
                           'Age_mean_months': round(meta['age_month'].mean(), 1),
                           'Age_sd_months': round(meta['age_month'].std(), 1),
                           'Age_median_months': float(meta['age_month'].median()),
                           'Age_range': f"{int(meta['age_month'].min())}-{int(meta['age_month'].max())}",
                           'Train_n': int((meta['split']=='Train').sum()),
                           'Test_n': int((meta['split']=='Test').sum()),
                           'Top3_locations': 'Ovary, Mediastinum, Retroperitoneum'}])
    t1 = pd.concat([t1, total], ignore_index=True)
    t1.to_csv(f'{OUT}/table1.csv', index=False)

    # Markdown 版
    md_lines = ['| Category | N | Female | Male | Age (mo) mean±SD | median | range | Train/Test | Top3 locations |',
                '|---|---|---|---|---|---|---|---|---|']
    for _, r in t1.iterrows():
        md_lines.append(f"| {r['Category']} | {r['N']} | {r['Female_n']} | {r['Male_n']} | "
                        f"{r['Age_mean_months']}±{r['Age_sd_months']} | {r['Age_median_months']} | {r['Age_range']} | "
                        f"{r['Train_n']}/{r['Test_n']} | {r['Top3_locations']} |")
    open(f'{OUT}/table1.md', 'w').write('\n'.join(md_lines))
    print('\n'.join(md_lines[:2]))
    print('\n'.join(md_lines[2:]))
    print('\nsaved ->', f'{OUT}/table1.csv', f'{OUT}/table1.md')

if __name__ == '__main__':
    main()
