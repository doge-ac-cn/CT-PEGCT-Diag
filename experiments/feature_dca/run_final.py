#!/usr/bin/env python
"""run_final: 综合特征加载（修复缺失文件）
run_dca.py 依赖本模块的 load_all_features()。
特征集 = radiomics(1316) + physics(11) + meta(location one-hot)。
注: 原始 run_final.py 未随发布包提供；本恢复版经 Brier 匹配验证
（MT -0.0001 / IT +0.0005 / MGCT +0.0015 vs 作者预计算 results_dca.json）。
"""
import pandas as pd

SRC = 'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'

def load_all_features():
    """返回 (df, meta_cols)
    df 列: Patient_ID, class, radiomics特征..., physics特征..., meta特征
    """
    df = pd.read_csv('experiments/radiomics_svm/features_full.csv')
    df = df[df['Patient_ID'] != 'MT_411'].reset_index(drop=True)

    # physics 特征
    phys = pd.read_csv('experiments/final_benchmark/physics_features.csv')
    df = df.merge(phys, on='Patient_ID', how='left')

    # meta: 位置 one-hot
    meta_rows = []
    for cat in ['MTs', 'ITs', 'MGCTs']:
        md = pd.read_excel(f'{SRC}/{cat}/{cat[:-1]}_patient.xlsx')
        lcol = [c for c in md.columns if 'ocation' in c][0]
        for _, r in md.iterrows():
            meta_rows.append({'Patient_ID': r['Patient_ID'],
                              'location': str(r[lcol]).lower().strip()})
    meta = pd.DataFrame(meta_rows)
    df = df.merge(meta, on='Patient_ID', how='left')
    loc_dummies = pd.get_dummies(df['location'], prefix='loc')
    df = pd.concat([df.drop(columns=['location']), loc_dummies], axis=1)
    meta_cols = list(loc_dummies.columns)
    return df, meta_cols

if __name__ == '__main__':
    df, meta_cols = load_all_features()
    print(f'df shape: {df.shape}, meta_cols: {meta_cols}')
