#!/usr/bin/env python
"""bootstrap: SVM 主模型 bootstrap 95% CI (critic A3)
对纯组学 SVM (class_weight=None, 方案1 主结果) 在测试集 193 例上
bootstrap 1000 次 → ACC/AUC 95% CI + 每类召回 CI
"""
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
warnings.filterwarnings('ignore')

OUT = 'experiments/bootstrap'
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)
N_BOOT = 1000

def main():
    df = pd.read_csv('experiments/radiomics_svm/features_full.csv')
    df = df[df['Patient_ID'] != 'MT_411'].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in ('Patient_ID', 'class')]
    split = json.load(open('datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
    train_stems = set(split['MT']['train'] + split['IT']['train'] + split['MGCT']['train']) - {'MT_411'}
    test_stems = set(split['MT']['test'] + split['IT']['test'] + split['MGCT']['test']) - {'MT_411'}
    tr = df[df['Patient_ID'].isin(train_stems)]
    te = df[df['Patient_ID'].isin(test_stems)]
    X_tr, y_tr = tr[feat_cols].values, tr['class'].values
    X_te, y_te = te[feat_cols].values, te['class'].values

    vt = VarianceThreshold(threshold=1e-6)
    X_tr_v = vt.fit_transform(X_tr)
    nan_cols = np.isnan(X_tr_v).any(axis=0)
    if nan_cols.sum():
        X_tr_v = X_tr_v[:, ~nan_cols]
        keep_cols = np.where(~nan_cols)[0]
    else:
        keep_cols = np.arange(X_tr_v.shape[1])
    keep = list(range(X_tr_v.shape[1]))
    if X_tr_v.shape[1] >= 2:
        try:
            corr = np.corrcoef(X_tr_v.T)
            keep = [i for i in range(X_tr_v.shape[1]) if all(abs(corr[i, j]) < 0.95 for j in range(i))]
        except Exception:
            pass
    X_tr_r = X_tr_v[:, keep]
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('svm', SVC(kernel='rbf', C=10, gamma=0.001, probability=True, random_state=42))])
    pipe.fit(X_tr_r, y_tr)

    # 测试集变换 (与训练同列)
    X_te_v = vt.transform(X_te)[:, keep]
    prob = pipe.predict_proba(X_te_v)
    pred = pipe.predict(X_te_v)
    acc0 = accuracy_score(y_te, pred)
    auc0 = roc_auc_score(y_te, prob, multi_class='ovr')
    rec0 = recall_score(y_te, pred, average=None)
    print(f'point estimates: ACC={acc0:.4f} AUC={auc0:.4f} recall={np.round(rec0,3)}')

    # bootstrap
    n = len(y_te)
    accs, aucs, recs = [], [], []
    for b in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        accs.append(accuracy_score(y_te[idx], pred[idx]))
        aucs.append(roc_auc_score(y_te[idx], prob[idx], multi_class='ovr'))
        recs.append(recall_score(y_te[idx], pred[idx], average=None))
    accs, aucs = np.array(accs), np.array(aucs)
    recs = np.array(recs)
    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    result = {
        'n_test': int(n), 'n_boot': N_BOOT,
        'acc': float(acc0), 'acc_ci95': ci(accs),
        'auc_ovr': float(auc0), 'auc_ci95': ci(aucs),
        'recall': [float(r) for r in rec0],
        'recall_ci95': [ci(recs[:, i]) for i in range(3)],
    }
    print(f'ACC 95% CI: {result["acc_ci95"]}')
    print(f'AUC 95% CI: {result["auc_ci95"]}')
    print(f'Recall CI:  {result["recall_ci95"]}')
    json.dump(result, open(f'{OUT}/results_bootstrap.json', 'w'), indent=2)
    print(f'saved -> {OUT}/results_bootstrap.json')


if __name__ == '__main__':
    main()
