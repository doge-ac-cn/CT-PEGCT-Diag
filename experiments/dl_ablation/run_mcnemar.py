#!/usr/bin/env python
"""McNemar tests: 消融结论显著性检验 (McNemar) + 补充表 S2
- 逐例预测: 2D/2.5D/3D R18, SVM, 集成, 位置感知集成
- McNemar 配对检验: 2D vs 2.5D, 2D vs 3D, 2D vs SVM, 集成 vs SVM, 集成 vs DL
- 补充表 S2: 全部模型详细指标
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
from scipy.stats import binom
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
warnings.filterwarnings('ignore')

sys.path.insert(0, 'experiments/dl_ablation')
from train_ablation import ARCHS, make_resnet3d, ROIDataset

SRC = 'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'

def predict_dl(manifest, dim, weight_path):
    ds = ROIDataset(manifest, 'test', dim, 64, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    if dim == '3d':
        model = make_resnet3d('18')
    else:
        model = ARCHS['18'](weights=None, num_classes=3)
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()
    stems, p = [], []
    with torch.no_grad():
        for x, y in loader:
            p.extend(torch.softmax(model(x), 1).numpy())
    stems = [os.path.basename(pp).replace('.npy', '').replace('_2d', '').replace('_25d', '').replace('_3d_64', '') for pp, _ in ds.items]
    return stems, np.array(p)

def mcnemar(y1, y2, y_true):
    """配对 McNemar: 比较两个分类器预测。返回 p 值 (二项精确检验)"""
    b = int(((y1 == y_true) & (y2 != y_true)).sum())   # 1对2错
    c = int(((y1 != y_true) & (y2 == y_true)).sum())   # 1错2对
    n = b + c
    if n == 0:
        return 1.0, b, c
    # 二项检验: p = P(X <= min(b,c) | n, 0.5)*2
    p = 2 * binom.cdf(min(b, c), n, 0.5)
    return min(p, 1.0), b, c

def main():
    manifest = json.load(open('experiments/dl_ablation/data/manifest_ablation.json'))
    stems, p_2d = predict_dl(manifest, '2d', 'experiments/dl_ablation/results/2d_r18_ce.pt')
    _, p_25 = predict_dl(manifest, '25d', 'experiments/dl_ablation/results/25d_r18_ce.pt')
    _, p_3d = predict_dl(manifest, '3d', 'experiments/dl_ablation/results/3d_r18_roi64_ce.pt')

    meta_rows = []
    for cat in ['MTs', 'ITs', 'MGCTs']:
        md = pd.read_excel(f'{SRC}/{cat}/{cat[:-1]}_patient.xlsx')
        lcol = [c for c in md.columns if 'ocation' in c][0]
        for _, r in md.iterrows():
            meta_rows.append({'Patient_ID': r['Patient_ID'], 'cat': cat, 'location': str(r[lcol]).lower().strip()})
    meta = pd.DataFrame(meta_rows).set_index('Patient_ID').loc[stems]
    y_true = np.array([{'MTs': 0, 'ITs': 1, 'MGCTs': 2}[c] for c in meta['cat']])
    loc = meta['location'].values

    # SVM
    df = pd.read_csv('experiments/radiomics_svm/features_full.csv')
    df = df[df['Patient_ID'] != 'MT_411'].reset_index(drop=True)
    split = json.load(open('datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
    train_stems = set(split['MT']['train'] + split['IT']['train'] + split['MGCT']['train']) - {'MT_411'}
    feat_cols = [c for c in df.columns if c not in ('Patient_ID', 'class')]
    train_df = df[df['Patient_ID'].isin(train_stems)]
    X_tr = train_df[feat_cols].values; y_tr = train_df['class'].values
    vt = VarianceThreshold(threshold=1e-6)
    X_tr_v = vt.fit_transform(X_tr)
    keep = list(range(X_tr_v.shape[1]))
    try:
        corr = np.corrcoef(X_tr_v.T)
        keep = [i for i in range(X_tr_v.shape[1]) if all(abs(corr[i, j]) < 0.95 for j in range(i))]
    except Exception:
        pass
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('svm', SVC(kernel='rbf', C=10, gamma=0.001, probability=True, random_state=42))])
    pipe.fit(X_tr_v[:, keep], y_tr)
    test_df = df[df['Patient_ID'].isin(set(stems))].set_index('Patient_ID')
    p_svm = pipe.predict_proba(vt.transform(test_df.loc[stems, feat_cols].values)[:, keep])

    # 集成 & 位置感知
    med = loc == 'mediastinum'
    p_ens = (p_2d + p_svm) / 2
    p_sel = (np.where(med[:, None], p_25, p_2d) + p_svm) / 2

    preds = {'2D R18': p_2d.argmax(1), '2.5D R18': p_25.argmax(1), '3D R18': p_3d.argmax(1),
             'SVM': p_svm.argmax(1), 'Ensemble': p_ens.argmax(1), 'Loc-aware': p_sel.argmax(1)}
    probs = {'2D R18': p_2d, '2.5D R18': p_25, '3D R18': p_3d, 'SVM': p_svm,
             'Ensemble': p_ens, 'Loc-aware': p_sel}

    # McNemar 矩阵
    print('=== McNemar 配对检验 ===')
    pairs = [('2D R18', '2.5D R18'), ('2D R18', '3D R18'), ('2D R18', 'SVM'),
             ('Ensemble', 'SVM'), ('Ensemble', '2D R18'), ('Loc-aware', 'Ensemble')]
    results = {}
    for a, b in pairs:
        pval, bcnt, ccnt = mcnemar(preds[a], preds[b], y_true)
        print(f'{a:12s} vs {b:12s}: p={pval:.4f} (b={bcnt}, c={ccnt}) {"**显著**" if pval < 0.05 else ""}')
        results[f'{a}_vs_{b}'] = {'p': float(pval), 'b': int(bcnt), 'c': int(ccnt)}
    pv23 = results['2D R18_vs_3D R18']['p']
    pv_ens_svm = results['Ensemble_vs_SVM']['p']
    print(f'\n核心结论: 2D vs 3D p={pv23:.4f}; Ensemble vs SVM p={pv_ens_svm:.4f}')

    # 补充表 S2
    rows = []
    for name, pr in preds.items():
        acc = accuracy_score(y_true, pr)
        auc = roc_auc_score(y_true, probs[name], multi_class='ovr')
        rec = recall_score(y_true, pr, average=None)
        rows.append({'model': name, 'acc': acc, 'auc': auc,
                     'mt_recall': rec[0], 'it_recall': rec[1], 'mgct_recall': rec[2]})
    s2 = pd.DataFrame(rows)
    print('\n=== 补充表 S2 ===')
    print(s2.round(3).to_string(index=False))
    s2.to_csv('experiments/dl_ablation/supp_table_S2.csv', index=False)

    json.dump({'mcnemar': results,
               's2': s2.round(4).to_dict('records')},
              open('experiments/dl_ablation/results_mcnemar.json', 'w'), indent=2)
    print('\nsaved -> experiments/dl_ablation/supp_table_S2.csv, results_mcnemar.json')

if __name__ == '__main__':
    main()
