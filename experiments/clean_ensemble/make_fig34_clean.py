#!/usr/bin/env python
"""No-leakage Fig 3 (ROC) + Fig 4 (confusion matrix) generation
- Fig 3: clean 概率 (clean_ensemble) + balanced SVM (calibration 管线重算, 固定超参无泄漏)
  曲线: ensemble (0.902/0.962), radiomics SVM (0.886/0.950),
        balanced full-feature SVM (0.881/0.952), 2D R18 5-seed (0.865/0.923),
        official baseline (0.726/0.885, 标量用星号)
- Fig 4: clean ensemble 混淆矩阵 (n=193)
输出: experiments/clean_ensemble/figures/fig3_roc_clean.png, fig4_cm_clean.png
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

OUT = 'experiments/clean_ensemble'
FIGOUT = f'{OUT}/figures'
os.makedirs(FIGOUT, exist_ok=True)

# ---------- clean 概率 ----------
y = np.load(f'{OUT}/clean_y.npy')
p_svm = np.load(f'{OUT}/clean_p_svm.npy')
p_dl5 = np.load(f'{OUT}/clean_p_2d_avg5.npy')
p_ens = (p_svm + p_dl5) / 2
stems = list(np.load(f'{OUT}/clean_stems.npy'))

# ---------- balanced SVM (calibration 管线, 固定超参) ----------
split = json.load(open('datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
train_stems = set(split['MT']['train'] + split['IT']['train'] + split['MGCT']['train']) - {'MT_411'}
df = pd.read_csv('experiments/radiomics_svm/features_full.csv')
df = df[df['Patient_ID'] != 'MT_411'].reset_index(drop=True)
phys = pd.read_csv('experiments/final_benchmark/physics_features.csv')
df = df.merge(phys, on='Patient_ID', how='left')
meta_rows = []
for cat in ['MTs', 'ITs', 'MGCTs']:
    md = pd.read_excel(f'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag/{cat}/{cat[:-1]}_patient.xlsx')
    g = [c for c in md.columns if 'ender' in c][0]
    a = [c for c in md.columns if 'Age' in c][0]
    l = [c for c in md.columns if 'ocation' in c][0]
    for _, r in md.iterrows():
        meta_rows.append({'Patient_ID': r['Patient_ID'], 'gender': str(r[g]).upper(),
                          'age': float(r[a]), 'location': str(r[l]).lower().strip()})
meta = pd.DataFrame(meta_rows)
meta['is_female'] = (meta['gender'] == 'F').astype(int)
meta['age_log'] = np.log1p(meta['age'])
dummies = pd.get_dummies(meta['location'], prefix='loc')
meta = pd.concat([meta, dummies], axis=1)
meta_cols = ['is_female', 'age_log'] + [c for c in meta.columns if c.startswith('loc_')]
df = df.merge(meta[['Patient_ID'] + meta_cols], on='Patient_ID', how='left')
phys_cols = ['fat_fraction', 'calc_fraction', 'solid_fraction', 'hu_mean', 'hu_median',
             'hu_std', 'hu_p5', 'hu_p95', 'hu_skew', 'hu_kurt', 'n_voxels']
feat_cols = [c for c in df.columns if c not in ('Patient_ID', 'class') + tuple(phys_cols) + tuple(meta_cols)]
all_cols = feat_cols + phys_cols + meta_cols
tr = df[df['Patient_ID'].isin(train_stems)]
te = df[df['Patient_ID'].isin(set(stems))]
df[all_cols] = df[all_cols].astype(float)
X_tr, y_tr = tr[all_cols].values.astype(float), tr['class'].values
X_te = te.set_index('Patient_ID').loc[stems, all_cols].values.astype(float)
vt = VarianceThreshold(threshold=1e-6)
X_tr_v = vt.fit_transform(X_tr); X_te_v = vt.transform(X_te)
if np.isnan(X_tr_v).any():
    nan_cols = np.isnan(X_tr_v).any(axis=0)
    X_tr_v = X_tr_v[:, ~nan_cols]; X_te_v = X_te_v[:, ~nan_cols]
corr = np.corrcoef(X_tr_v.T)
keep = [i for i in range(X_tr_v.shape[1]) if all(abs(corr[i, j]) < 0.95 for j in range(i))]
pipe = Pipeline([('scaler', StandardScaler()),
                 ('svm', SVC(kernel='rbf', C=10, gamma=0.001, class_weight='balanced',
                             probability=True, random_state=42))])
pipe.fit(X_tr_v[:, keep], y_tr)
p_bal = pipe.predict_proba(X_te_v[:, keep])
bal_acc = accuracy_score(y, p_bal.argmax(1))
bal_auc = roc_auc_score(y, p_bal, multi_class='ovr')
print(f'balanced SVM: ACC={bal_acc:.4f} AUC={bal_auc:.4f} (expect 0.881/0.952)')

# ---------- Fig 3: ROC ----------
methods = [
    ('5-seed radiomics×DL ensemble', p_ens, '#55A868'),
    ('Radiomics SVM', p_svm, '#C44E52'),
    ('Full-feature balanced SVM', p_bal, '#8172B2'),
    ('2D ResNet18 (5-seed)', p_dl5, '#4C72B0'),
]
plt.figure(figsize=(6.8, 6.2))
for name, p, c in methods:
    fprs, tprs = [], []
    for k in range(3):
        fpr, tpr, _ = roc_curve((y == k).astype(int), p[:, k])
        fprs.append(fpr); tprs.append(tpr)
    fpr_u = np.unique(np.concatenate(fprs))
    tpr_interp = np.mean([np.interp(fpr_u, f, t) for f, t in zip(fprs, tprs)], axis=0)
    tpr_interp[0] = 0.0
    auc = roc_auc_score(y, p, multi_class='ovr')
    plt.plot(fpr_u, tpr_interp, lw=2.2, color=c, label=f'{name} (AUC={auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
# Official baseline: 原论文仅报告标量 AUC, 未发表逐点 ROC 曲线.
# 只保留星号标记, 不在图内写字; AUC 与其他模型一起在右下角图例中报告.
plt.scatter([0.13], [0.155], marker='*', s=260, color='#7f7f7f', zorder=5,
            label='Official baseline (AUC=0.885)')
plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
plt.title('Macro-averaged one-vs-rest ROC (held-out test set, n=193)')
plt.legend(loc='lower right', fontsize=9)
plt.xlim(-0.02, 1.02); plt.ylim(-0.02, 1.02)
plt.tight_layout()
plt.savefig(f'{FIGOUT}/fig3_roc_clean.png', dpi=300)
print('saved ->', f'{FIGOUT}/fig3_roc_clean.png')

# ---------- Fig 4: confusion matrix (clean ensemble) ----------
pred = p_ens.argmax(1)
cm = confusion_matrix(y, pred)
rec = recall_score(y, pred, average=None)
acc = accuracy_score(y, pred)
labels = ['MT', 'IT', 'MGCT']
fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
im = ax.imshow(cm, cmap='Blues')
thresh = cm.max() / 2.
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=16,
                color='white' if cm[i, j] > thresh else 'black')
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(labels); ax.set_yticklabels(labels)
ax.set_xlabel('Predicted subtype', fontsize=12)
ax.set_ylabel('True subtype', fontsize=12)
ax.set_title(f'5-seed radiomics×DL ensemble (ACC {acc:.3f})', fontsize=13)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Number of cases', fontsize=10)
fig.savefig(f'{FIGOUT}/fig4_cm_clean.png', dpi=300)
print('saved ->', f'{FIGOUT}/fig4_cm_clean.png')
print('CM:', cm.tolist(), 'recall:', np.round(rec, 3).tolist())
