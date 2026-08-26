#!/usr/bin/env python
"""lodo_svm.py — LODO（留一解剖部位域）SVM 分类

5 个留出域: 训练域 = 其余 4 大域 (不含罕见域), 测试 = 留出域 + 罕见域(11例)
模型: 与论文协议一致 — VarianceThreshold + corr<0.95 去冗余 + 训练域内 5 折 GridSearch
      (C∈[0.1,1,10,100], gamma∈[scale,0.001,0.01,0.1], RBF, class_weight=balanced, roc_auc_ovr)
输出: 本目录 lodo_svm_results.json + lodo_svm_summary.csv

依赖:
  - experiments/radiomics_svm/features_full.csv
  - experiments/lodo/domain_labels.csv + lodo_splits.json
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FEAT_CSV = os.path.join(REPO, 'experiments', 'radiomics_svm', 'features_full.csv')
LABELS_CSV = os.path.join(HERE, 'domain_labels.csv')
LODO_JSON = os.path.join(HERE, 'lodo_splits.json')

feat = pd.read_csv(FEAT_CSV)
feat = feat[feat['Patient_ID'] != 'MT_411'].reset_index(drop=True)
lodo = json.load(open(LODO_JSON))
labels = pd.read_csv(LABELS_CSV)
df = feat.merge(labels[['Patient_ID', 'domain']], on='Patient_ID')
feat_cols = [c for c in feat.columns if c not in ('Patient_ID', 'class')]
X = df[feat_cols].values
y = df['class'].values
stems = df['Patient_ID'].values


def eval_split(dom):
    tr = list(lodo[dom]['train']); te = list(lodo[dom]['test']); rare = list(lodo[dom]['rare_test'])
    m_tr = np.isin(stems, tr); m_te = np.isin(stems, te); m_ra = np.isin(stems, rare)
    X_tr, y_tr = X[m_tr], y[m_tr]
    # 特征筛选 (训练域内)
    vt = VarianceThreshold(threshold=1e-6); Xv = vt.fit_transform(X_tr)
    corr = np.corrcoef(Xv.T)
    keep = [i for i in range(Xv.shape[1]) if all(abs(corr[i, j]) < 0.95 for j in range(i))]
    Xv = Xv[:, keep]
    print(f'[{dom}] train={len(tr)} test={len(te)} rare={len(rare)} | feats: {Xv.shape[1]}', flush=True)
    # GridSearch (训练域内 5 折)
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('svm', SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'))])
    param_grid = {'svm__C': [0.1, 1, 10, 100], 'svm__gamma': ['scale', 0.001, 0.01, 0.1]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc_ovr', n_jobs=8, verbose=0)
    gs.fit(Xv, y_tr)
    # 测试域 + 罕见域
    res = {'domain': dom, 'n_train': len(tr), 'n_test': len(te), 'n_rare': len(rare),
           'n_features': int(Xv.shape[1]), 'best_params': gs.best_params_, 'cv_auc': float(gs.best_score_)}
    for tag, mask in [('test', m_te), ('rare', m_ra), ('test+rare', m_te | m_ra)]:
        if mask.sum() == 0:
            continue
        Xt = vt.transform(X[mask])[:, keep]
        yt = y[mask]
        yp = gs.predict(Xt); ypr = gs.predict_proba(Xt)
        acc = accuracy_score(yt, yp)
        auc = roc_auc_score(yt, ypr, multi_class='ovr')
        cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
        recall = (cm / np.maximum(cm.sum(1, keepdims=True), 1)).diagonal().tolist()
        res[f'{tag}_acc'] = round(float(acc), 4)
        res[f'{tag}_auc'] = round(float(auc), 4)
        res[f'{tag}_recall'] = [round(float(r), 4) for r in recall]
        res[f'{tag}_cm'] = cm.tolist()
        res[f'{tag}_n'] = int(mask.sum())
    # 错误样本 (test 域)
    Xt = vt.transform(X[m_te])[:, keep]; yt = y[m_te]
    yp = gs.predict(Xt)
    wrong = stems[m_te][yp != yt].tolist()
    res['test_wrong_cases'] = wrong
    print(f"  [{dom}] test ACC={res['test_acc']} AUC={res['test_auc']} recall={res['test_recall']} | wrong={len(wrong)}", flush=True)
    return res


results = [eval_split(d) for d in ['ovary', 'testis', 'retroperitoneum', 'sacrococcyx', 'mediastinum']]
json.dump(results, open(os.path.join(HERE, 'lodo_svm_results.json'), 'w'), ensure_ascii=False, indent=2)

# 汇总表
rows = []
for r in results:
    rows.append({'domain': r['domain'], 'n_train': r['n_train'], 'n_test': r['n_test'],
                 'feats': r['n_features'], 'cv_auc': r['cv_auc'],
                 'best_C': r['best_params']['svm__C'], 'best_gamma': r['best_params']['svm__gamma'],
                 'test_ACC': r['test_acc'], 'test_AUC': r['test_auc'],
                 'MT_recall': r['test_recall'][0], 'IT_recall': r['test_recall'][1], 'MGCT_recall': r['test_recall'][2],
                 'rare_ACC': r.get('rare_acc'), 'rare_AUC': r.get('rare_auc'),
                 'test+rare_ACC': r.get('test+rare_acc')})
pd.DataFrame(rows).to_csv(os.path.join(HERE, 'lodo_svm_summary.csv'), index=False)
print('\n=== LODO-SVM 汇总 ===')
print(pd.DataFrame(rows).to_string(index=False))
print(f'\nsaved -> {HERE}/lodo_svm_results.json, lodo_svm_summary.csv')
