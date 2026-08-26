#!/usr/bin/env python
"""physics_priors: 影像物理先验 (路线3)
假设: MT 含脂肪钙化、MGCT 实性 → 肿瘤内 HU 分布、脂肪/钙化占比是 MT vs MGCT 的物理判据
方法: 基于 Mask 提取物理特征 → 加入 radiomics_svm 特征集 → DeLong 检验 AUC 提升
物理特征:
  - fat_fraction: HU < -30 的体素占比 (脂肪)
  - calc_fraction: HU > 100 的体素占比 (钙化)
  - hu_mean/median/std/p5/p95: 肿瘤内 HU 统计
  - solid_fraction: HU > 20 占比 (实性成分)
"""
import json, glob, warnings
import numpy as np, pandas as pd
import nibabel as nib
from scipy import stats
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
warnings.filterwarnings('ignore')

SRC = 'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'

def extract_physics(stem):
    """提取物理先验特征"""
    for cat in ['MTs', 'ITs', 'MGCTs']:
        import os
        img_p = f'{SRC}/{cat}/Images/{stem}.nii.gz'
        mask_p = f'{SRC}/{cat}/Masks/{stem}.nii.gz'
        if os.path.exists(img_p) and os.path.exists(mask_p):
            img = nib.load(img_p).get_fdata()
            mask = nib.load(mask_p).get_fdata()
            hu = img[mask > 0]
            if len(hu) == 0:
                return None
            n = len(hu)
            return {
                'fat_fraction': float((hu < -30).mean()),
                'calc_fraction': float((hu > 100).mean()),
                'solid_fraction': float((hu > 20).mean()),
                'hu_mean': float(hu.mean()),
                'hu_median': float(np.median(hu)),
                'hu_std': float(hu.std()),
                'hu_p5': float(np.percentile(hu, 5)),
                'hu_p95': float(np.percentile(hu, 95)),
                'hu_skew': float(stats.skew(hu)),
                'hu_kurt': float(stats.kurtosis(hu)),
                'n_voxels': int(n),
            }
    return None

def delong_test(auc1, auc2, n, corr):
    """简化的 DeLong 检验 (双相关样本 AUC 比较)"""
    # 使用 Hanley-McNeil 近似
    se1 = np.sqrt((auc1 * (1 - auc1) + (n - 1) * (auc1 / (2 - auc1) - auc1**2)) / n)
    se2 = np.sqrt((auc2 * (1 - auc2) + (n - 1) * (auc2 / (2 - auc2) - auc2**2)) / n)
    se_diff = np.sqrt(se1**2 + se2**2 - 2 * corr * se1 * se2)
    z = (auc1 - auc2) / se_diff if se_diff > 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

def main():
    df = pd.read_csv('experiments/radiomics_svm/features_full.csv')
    df = df[df['Patient_ID'] != 'MT_411'].reset_index(drop=True)

    # 提取物理特征
    print('extracting physics features...')
    phys_rows = []
    for stem in df['Patient_ID']:
        p = extract_physics(stem)
        if p is None:
            print('MISSING:', stem)
        else:
            phys_rows.append({'Patient_ID': stem, **p})
    phys = pd.DataFrame(phys_rows)
    print(f'physics features extracted: {len(phys)} cases')

    # 合并
    df = df.merge(phys, on='Patient_ID', how='left')
    split = json.load(open('datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
    train_stems = set(split['MT']['train'] + split['IT']['train'] + split['MGCT']['train']) - {'MT_411'}
    test_stems = set(split['MT']['test'] + split['IT']['test'] + split['MGCT']['test']) - {'MT_411'}
    train_df = df[df['Patient_ID'].isin(train_stems)]; test_df = df[df['Patient_ID'].isin(test_stems)]

    phys_cols = ['fat_fraction', 'calc_fraction', 'solid_fraction', 'hu_mean', 'hu_median',
                 'hu_std', 'hu_p5', 'hu_p95', 'hu_skew', 'hu_kurt', 'n_voxels']
    feat_cols = [c for c in df.columns if c not in ('Patient_ID', 'class') + tuple(phys_cols)]

    print('\n=== 物理特征判别力 (单变量 AUC, MT vs MGCT) ===')
    mt_mask = (train_df['class'] == 0)
    mgct_mask = (train_df['class'] == 2)
    for col in phys_cols:
        y = train_df.loc[mt_mask | mgct_mask, 'class'].values
        y = (y == 2).astype(int)  # MGCT=1
        x = train_df.loc[mt_mask | mgct_mask, col].values
        if len(np.unique(x)) > 1:
            auc = roc_auc_score(y, x) if np.median(x[y==1]) > np.median(x[y==0]) else roc_auc_score(y, -x)
            print(f'  {col}: single-var AUC={auc:.4f}')

    def run_svm(X_tr, y_tr, X_te, y_te, label):
        vt = VarianceThreshold(threshold=1e-6)
        X_tr_v = vt.fit_transform(X_tr); X_te_v = vt.transform(X_te)
        corr = np.corrcoef(X_tr_v.T)
        keep = [i for i in range(X_tr_v.shape[1]) if all(abs(corr[i, j]) < 0.95 for j in range(i))]
        pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', probability=True, random_state=42))])
        gs = GridSearchCV(pipe, {'svm__C': [0.1, 1, 10], 'svm__gamma': ['scale', 0.001, 0.01]},
                          cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc_ovr', n_jobs=4)
        gs.fit(X_tr_v[:, keep], y_tr)
        y_prob = gs.predict_proba(X_te_v[:, keep])
        y_pred = gs.predict(X_te_v[:, keep])
        acc = accuracy_score(y_te, y_pred)
        auc = roc_auc_score(y_te, y_prob, multi_class='ovr')
        print(f'{label}: ACC={acc:.4f} AUC={auc:.4f}')
        return acc, auc

    print('\n=== 模型对比 (三分类) ===')
    y_tr = train_df['class'].values; y_te = test_df['class'].values
    X_tr_base = train_df[feat_cols].values; X_te_base = test_df[feat_cols].values
    X_tr_full = train_df[feat_cols + phys_cols].values; X_te_full = test_df[feat_cols + phys_cols].values

    acc1, auc1 = run_svm(X_tr_base, y_tr, X_te_base, y_te, '影像特征 only')
    acc2, auc2 = run_svm(X_tr_full, y_tr, X_te_full, y_te, '影像特征 + 物理先验')

    # 物理特征单独
    acc3, auc3 = run_svm(train_df[phys_cols].values, y_tr, test_df[phys_cols].values, y_te, '物理先验 only')

    # 保存
    results = {
        'radiomics_only': {'acc': float(acc1), 'auc': float(auc1)},
        'radiomics_plus_physics': {'acc': float(acc2), 'auc': float(auc2)},
        'physics_only': {'acc': float(acc3), 'auc': float(auc3)},
        'physics_cols': phys_cols
    }
    with open('experiments/physics_priors/results_physics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nsaved -> experiments/physics_priors/results_physics.json')

if __name__ == '__main__':
    main()
