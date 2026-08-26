#!/usr/bin/env python
"""pca_drift.py — LODO 域特征漂移诊断（PCA 基线）

对应论文 Supplementary（site hold-out 特征漂移分析）：
1) 全量 641 例 PCA（保留 95% 方差）：域间质心距离矩阵 + 域内分散度
2) per-LODO 训练域 PCA：每个留出域到训练质心的距离
3) 内部测试集（7:3 split, 193例）到训练质心距离作为同分布基线
4) 罕见域（other, 11例）距离

输出（写入本脚本所在目录）：domain_distances.json + pca_drift.png

依赖：
  - experiments/radiomics_svm/features_full.csv（由 extract_features_full.py 生成）
  - experiments/lodo/domain_labels.csv + lodo_splits.json（make_domain_splits.py）
  - datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json
"""
import json
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FEAT_CSV = os.path.join(REPO, 'experiments', 'radiomics_svm', 'features_full.csv')
LABELS_CSV = os.path.join(HERE, 'domain_labels.csv')
LODO_JSON = os.path.join(HERE, 'lodo_splits.json')
SPLIT_JSON = os.path.join(REPO, 'datasets', 'R23_CT-PEGCT-Diag', 'split_7to3_seed42.json')

# ---------- 数据 ----------
feat = pd.read_csv(FEAT_CSV)
feat = feat[feat['Patient_ID'] != 'MT_411'].reset_index(drop=True)
labels = pd.read_csv(LABELS_CSV)  # Patient_ID, class, domain, domain_raw
df = feat.merge(labels[['Patient_ID', 'domain']], on='Patient_ID')
X = df[[c for c in feat.columns if c not in ('Patient_ID', 'class')]].values
y = df['class'].values
domains = df['domain'].values
stems = df['Patient_ID'].values
print(f'data: {df.shape[0]} cases x {X.shape[1]} features')

# 方差筛选 (全量拟合, 仅去常数特征, 与 SVM 协议一致)
vt = VarianceThreshold(threshold=1e-6)
Xv = vt.fit_transform(X)
print(f'after variance filter: {Xv.shape[1]}')

# 标准化 (训练域拟合; 全量基线用全量拟合)
sc_all = StandardScaler().fit(Xv)
Xs = sc_all.transform(Xv)

# ---------- 1) 全量 PCA ----------
pca_all = PCA(n_components=0.95, random_state=42)
Z_all = pca_all.fit_transform(Xs)
print(f'PCA components (95% var): {Z_all.shape[1]} (explained {pca_all.explained_variance_ratio_.sum():.3f})')

# 域质心 & 距离矩阵
uniq = ['ovary', 'testis', 'retroperitoneum', 'sacrococcyx', 'mediastinum', 'other']
cent = {d: Z_all[domains == d].mean(0) for d in uniq}
dist_mat = {}
for a in uniq:
    dist_mat[a] = {}
    for b in uniq:
        dist_mat[a][b] = round(float(np.linalg.norm(cent[a] - cent[b])), 3)

# 域内分散度 (平均到质心距离)
spread = {d: float(np.mean(np.linalg.norm(Z_all[domains == d] - cent[d], axis=1))) for d in uniq}
print('\n=== 域内分散度 (平均到质心距离, PCA95) ===')
for d in uniq:
    print(f'  {d:16s} spread={spread[d]:.3f}  N={(domains == d).sum()}')

print('\n=== 域间质心距离矩阵 ===')
print('            ' + ''.join(f'{d[:5]:>8s}' for d in uniq))
for a in uniq:
    print(f'{a[:12]:>12s} ' + ''.join(f'{dist_mat[a][b]:8.2f}' for b in uniq))

# ---------- 2) per-LODO 训练域 PCA ----------
lodo = json.load(open(LODO_JSON))

lodo_dist = {}
for dom in ['ovary', 'testis', 'retroperitoneum', 'sacrococcyx', 'mediastinum']:
    tr_mask = np.isin(stems, list(lodo[dom]['train']))
    te_mask = np.isin(stems, list(lodo[dom]['test']))
    sc = StandardScaler().fit(Xv[tr_mask])
    pca = PCA(n_components=0.95, random_state=42).fit(sc.transform(Xv[tr_mask]))
    Ztr = pca.transform(sc.transform(Xv[tr_mask]))
    Zte = pca.transform(sc.transform(Xv[te_mask]))
    ctr = Ztr.mean(0)
    d_test = np.linalg.norm(Zte - ctr, axis=1)
    d_train = np.linalg.norm(Ztr - ctr, axis=1)
    lodo_dist[dom] = {'pca_components': Ztr.shape[1],
                      'test_to_train_centroid_mean': round(float(d_test.mean()), 3),
                      'test_to_train_centroid_median': round(float(np.median(d_test)), 3),
                      'train_spread_median': round(float(np.median(d_train)), 3)}

# 3) 内部测试集同分布基线 (7:3 split, 训练集 PCA)
split = json.load(open(SPLIT_JSON))
tr7 = set(split['MT']['train'] + split['IT']['train'] + split['MGCT']['train']) - {'MT_411'}
te7 = set(split['MT']['test'] + split['IT']['test'] + split['MGCT']['test']) - {'MT_411'}
sc7 = StandardScaler().fit(Xv[np.isin(stems, list(tr7))])
pca7 = PCA(n_components=0.95, random_state=42).fit(sc7.transform(Xv[np.isin(stems, list(tr7))]))
Ztr7 = pca7.transform(sc7.transform(Xv[np.isin(stems, list(tr7))]))
Zte7 = pca7.transform(sc7.transform(Xv[np.isin(stems, list(te7))]))
ctr7 = Ztr7.mean(0)
d7 = np.linalg.norm(Zte7 - ctr7, axis=1)
internal_dist = float(np.mean(d7))
internal_dist_median = float(np.median(d7))
# 离群样本 (内部测试集, mean 口径下 > 训练分散度 3 倍)
train_spread7 = np.linalg.norm(Ztr7 - ctr7, axis=1)
outliers7 = stems[np.isin(stems, list(te7))][d7 > max(3 * np.median(d7), 100)].tolist()
print('\n  内部测试集离群样本 (dist > threshold):', outliers7)

# 4) 罕见域 (other) 距离: 用全量 PCA 空间 (其他域为训练代表)
Zother = Z_all[domains == 'other']
dist_other = float(np.mean(np.linalg.norm(Zother - Z_all[domains != 'other'].mean(0), axis=1)))

print('\n=== per-LODO 测试域到训练质心距离 (PCA95, median稳健) ===')
for dom in lodo_dist:
    d = lodo_dist[dom]
    print(f'  LODO-{dom:16s} median={d["test_to_train_centroid_median"]:.3f}  '
          f'(mean={d["test_to_train_centroid_mean"]:.3f}, train spread med={d["train_spread_median"]:.3f})')
print(f'  内部测试集基线 (同分布): median={internal_dist_median:.3f} (mean={internal_dist:.3f})')
print(f'  罕见域 other -> 全量质心: {dist_other:.3f}')

# ---------- 保存 ----------
result = {'domain_centroid_dist_matrix': dist_mat,
          'domain_spread': spread,
          'per_lodo_test_dist': lodo_dist,
          'internal_test_baseline_mean': round(internal_dist, 3),
          'internal_test_baseline_median': round(internal_dist_median, 3),
          'internal_test_outliers': outliers7,
          'rare_domain_dist': round(dist_other, 3),
          'pca_components_95pct_all': int(Z_all.shape[1]),
          'note': '距离=PCA95空间欧氏距离(StandardScaler后), median为主指标(mean被离群拉高)'}
json.dump(result, open(os.path.join(HERE, 'domain_distances.json'), 'w'), ensure_ascii=False, indent=2)

# ---------- 可视化 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
colors = {'ovary': '#e15759', 'testis': '#4e79a7', 'retroperitoneum': '#59a14f',
          'sacrococcyx': '#f28e2b', 'mediastinum': '#b07aa1', 'other': '#9c755f'}
# 左: PCA 前2主成分散点
for d in uniq:
    m = domains == d
    axes[0].scatter(Z_all[m, 0], Z_all[m, 1], s=8, alpha=0.5, color=colors[d], label=f'{d} (N={m.sum()})')
axes[0].set_xlabel(f'PC1 ({pca_all.explained_variance_ratio_[0] * 100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca_all.explained_variance_ratio_[1] * 100:.1f}%)')
axes[0].set_title('PCA95 (前2主成分): 641例按部位域着色')
axes[0].legend(fontsize=8, markerscale=2)
# 右: 距离条形图
doms = list(lodo_dist.keys())
vals = [lodo_dist[d]['test_to_train_centroid_median'] for d in doms]
axes[1].bar(doms, vals, color=[colors[d] for d in doms])
axes[1].axhline(internal_dist_median, ls='--', color='k', label=f'内部测试集基线(median) {internal_dist_median:.2f}')
axes[1].set_ylabel('测试域→训练质心距离 (PCA95)')
axes[1].set_title('per-LODO 域外距离 vs 同分布基线')
axes[1].legend()
for i, v in enumerate(vals):
    axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'pca_drift.png'), dpi=150)
print(f'\nsaved: {HERE}/domain_distances.json, pca_drift.png')
