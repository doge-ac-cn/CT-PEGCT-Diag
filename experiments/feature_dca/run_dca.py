#!/usr/bin/env python
"""feature_dcab: DCA 决策曲线分析 + 校准曲线
基于 feature_dca 的 SVM 综合特征模型, 生成 DCA 与校准曲线图
"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
warnings.filterwarnings('ignore')

SRC = 'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'
OUT = 'experiments/feature_dca'

# 复用 feature_dca 的加载函数
import sys
sys.path.insert(0, 'experiments/feature_dca')
from run_final import load_all_features

def net_benefit(thresholds, y_true, y_prob, label_idx):
    """单类 one-vs-rest DCA 净收益"""
    y_bin = (y_true == label_idx).astype(int)
    nb = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = ((pred == 1) & (y_bin == 1)).sum()
        fp = ((pred == 1) & (y_bin == 0)).sum()
        n = len(y_bin)
        # 净收益 = TP/n - FP/n * (t/(1-t))
        nb.append(tp / n - fp / n * (t / (1 - t)))
    return np.array(nb)

def main():
    df, meta_cols = load_all_features()
    split = json.load(open('datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
    train_stems = set(split['MT']['train'] + split['IT']['train'] + split['MGCT']['train']) - {'MT_411'}
    test_stems = set(split['MT']['test'] + split['IT']['test'] + split['MGCT']['test']) - {'MT_411'}
    train_df = df[df['Patient_ID'].isin(train_stems)]; test_df = df[df['Patient_ID'].isin(test_stems)]

    phys_cols = ['fat_fraction', 'calc_fraction', 'solid_fraction', 'hu_mean', 'hu_median',
                 'hu_std', 'hu_p5', 'hu_p95', 'hu_skew', 'hu_kurt', 'n_voxels']
    feat_cols = [c for c in df.columns if c not in ('Patient_ID', 'class') + tuple(phys_cols) + tuple(meta_cols)]
    all_cols = feat_cols + phys_cols + meta_cols

    X_tr = train_df[all_cols].values; y_tr = train_df['class'].values
    X_te = test_df[all_cols].values; y_te = test_df['class'].values
    vt = VarianceThreshold(threshold=1e-6)
    X_tr_v = vt.fit_transform(X_tr); X_te_v = vt.transform(X_te)
    keep = list(range(X_tr_v.shape[1]))
    if X_tr_v.shape[1] >= 2:
        try:
            corr = np.corrcoef(X_tr_v.T)
            keep = [i for i in range(X_tr_v.shape[1]) if all(abs(corr[i, j]) < 0.95 for j in range(i))]
        except Exception:
            pass
    X_tr_r, X_te_r = X_tr_v[:, keep], X_te_v[:, keep]
    pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=10, gamma=0.001, probability=True, random_state=42))])
    pipe.fit(X_tr_r, y_tr)
    y_prob = pipe.predict_proba(X_te_r)

    names = ['MT', 'IT', 'MGCT']
    thresholds = np.linspace(0.05, 0.95, 50)

    # === DCA 图 (三类的 one-vs-rest) ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, name in enumerate(names):
        ax = axes[i]
        nb_model = net_benefit(thresholds, y_te, y_prob[:, i], i)
        # 全治疗 (treat all)
        prev = (y_te == i).mean()
        nb_all = prev - (1 - prev) * thresholds / (1 - thresholds)
        # 全不治疗 = 0
        ax.plot(thresholds, nb_model, 'b-', lw=2, label=f'SVM ({name})')
        ax.plot(thresholds, nb_all, 'g--', lw=1.5, label='Treat all')
        ax.axhline(0, color='r', ls=':', lw=1.5, label='Treat none')
        ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
        ax.set_title(f'{name} (prevalence={prev:.3f})')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/dca.png', dpi=150)
    plt.close()
    print('DCA saved ->', f'{OUT}/dca.png')

    # === 校准曲线 (三类的 one-vs-rest) ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, name in enumerate(names):
        ax = axes[i]
        y_bin = (y_te == i).astype(int)
        prob_true, prob_pred = calibration_curve(y_bin, y_prob[:, i], n_bins=8)
        brier = brier_score_loss(y_bin, y_prob[:, i])
        ax.plot(prob_pred, prob_true, 'bo-', label=f'{name} (Brier={brier:.3f})')
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect')
        ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed fraction')
        ax.set_title(f'{name} calibration')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/calibration.png', dpi=150)
    plt.close()
    print('Calibration saved ->', f'{OUT}/calibration.png')

    # 保存数值
    results = {
        'brier_per_class': {n: float(brier_score_loss((y_te==i).astype(int), y_prob[:,i])) for i, n in enumerate(names)},
        'prevalence': {n: float((y_te==i).mean()) for i, n in enumerate(names)},
        'dca': f'{OUT}/dca.png', 'calibration': f'{OUT}/calibration.png'
    }
    with open(f'{OUT}/results_dca.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('saved ->', f'{OUT}/results_dca.json')

if __name__ == '__main__':
    main()
