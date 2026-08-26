#!/usr/bin/env python
"""calibration: 最终统一管线下的校准与决策曲线 (balanced SVM, C=10, gamma=0.001)
目的: feature_dca 的校准/DCA 基于旧管线 (无 class_weight); 论文最终数字采用 final_benchmark
      统一管线 (balanced) → 校准/DCA 需重算
输出: 每类 Brier + ECE + 校准数据 + DCA 数据
"""
import json, os, warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score, recall_score
warnings.filterwarnings('ignore')

OUT = 'experiments/calibration'
os.makedirs(OUT, exist_ok=True)
SEED = 42

split = json.load(open('datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
train_stems = set(split['MT']['train'] + split['IT']['train'] + split['MGCT']['train']) - {'MT_411'}
test_stems = set(split['MT']['test'] + split['IT']['test'] + split['MGCT']['test']) - {'MT_411'}

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
te = df[df['Patient_ID'].isin(test_stems)]
df[all_cols] = df[all_cols].astype(float)
X_tr, y_tr = tr[all_cols].values.astype(float), tr['class'].values
X_te, y_te = te[all_cols].values.astype(float), te['class'].values

vt = VarianceThreshold(threshold=1e-6)
X_tr_v = vt.fit_transform(X_tr); X_te_v = vt.transform(X_te)
# NaN 检查与保护
if np.isnan(X_tr_v).any():
    nan_cols = np.isnan(X_tr_v).any(axis=0)
    print(f'  drop {nan_cols.sum()} NaN columns')
    X_tr_v = X_tr_v[:, ~nan_cols]
    X_te_v = X_te_v[:, ~nan_cols]
if X_tr_v.shape[1] < 2:
    raise RuntimeError('too few features')
corr = np.corrcoef(X_tr_v.T)
keep = [i for i in range(X_tr_v.shape[1]) if all(abs(corr[i, j]) < 0.95 for j in range(i))]
X_tr_r, X_te_r = X_tr_v[:, keep], X_te_v[:, keep]
print(f'train {len(tr)}, test {len(te)}, features {len(keep)}')

pipe = Pipeline([('scaler', StandardScaler()),
                 ('svm', SVC(kernel='rbf', C=10, gamma=0.001, class_weight='balanced',
                             probability=True, random_state=SEED))])
pipe.fit(X_tr_r, y_tr)
proba = pipe.predict_proba(X_te_r)
pred = pipe.predict(X_te_r)
print(f'ACC={accuracy_score(y_te, pred):.4f} AUC(ovr)={roc_auc_score(y_te, proba, multi_class="ovr"):.4f}')
print(f'recall={np.round(recall_score(y_te, pred, average=None), 3)}')

# 1) 每类 Brier + 校准 (分箱)
cat_names = ['MT', 'IT', 'MGCT']
briers = {}
for c in range(3):
    yb = (y_te == c).astype(int)
    b = brier_score_loss(yb, proba[:, c])
    briers[cat_names[c]] = float(b)
    print(f'Brier {cat_names[c]}: {b:.4f}')

# 2) ECE (整体, 按 max prob 分箱)
maxp = proba.max(axis=1)
bins = np.linspace(0, 1, 11)
ece = 0.0
for i in range(10):
    m = (maxp >= bins[i]) & (maxp < bins[i + 1])
    if m.sum() == 0:
        continue
    conf = maxp[m].mean()
    acc = (pred[m] == y_te[m]).mean()
    ece += (m.sum() / len(y_te)) * abs(conf - acc)
print(f'ECE: {ece:.4f}')

# 3) DCA 数据 (三分类 one-vs-rest 净获益, 阈值网格)
threshs = np.arange(0.05, 0.96, 0.05)
dca = []
for c in range(3):
    yb = (y_te == c).astype(int)
    prev = yb.mean()
    for t in threshs:
        tp = ((proba[:, c] >= t) & (yb == 1)).sum()
        fp = ((proba[:, c] >= t) & (yb == 0)).sum()
        n = len(yb)
        # 净获益 = TP/n - FP/n * (t/(1-t))  (标准净获益公式)
        nb = tp / n - fp / n * (t / (1 - t))
        # 全治疗 (treat all) 和全不治疗基线
        dca.append({'cls': cat_names[c], 'thresh': float(t), 'net_benefit': float(nb)})
dca_df = pd.DataFrame(dca)
dca_df.to_csv(f'{OUT}/dca_data.csv', index=False)

# 校准数据 (每类分箱)
cal_rows = []
for c in range(3):
    for i in range(10):
        m = (proba[:, c] >= bins[i]) & (proba[:, c] < bins[i + 1])
        if m.sum() >= 5:
            cal_rows.append({'cls': cat_names[c], 'bin': bins[i],
                             'pred': float(proba[m, c].mean()),
                             'obs': float((y_te[m] == c).mean()), 'n': int(m.sum())})
cal_df = pd.DataFrame(cal_rows)
cal_df.to_csv(f'{OUT}/calibration_data.csv', index=False)

res = {
    'acc': float(accuracy_score(y_te, pred)),
    'auc_ovr': float(roc_auc_score(y_te, proba, multi_class='ovr')),
    'recall': recall_score(y_te, pred, average=None).tolist(),
    'brier': briers,
    'ece': float(ece),
    'n_features': int(len(keep)),
    'note': '统一管线 balanced SVM C=10 gamma=0.001, 669→筛选特征',
}
json.dump(res, open(f'{OUT}/results_calibration.json', 'w'), indent=2, ensure_ascii=False)
print(f'\nsaved -> experiments/calibration/')
