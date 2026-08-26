#!/usr/bin/env python
"""cascade_eval 第三部分: 从缓存加载特征 → feature_dca SVM → GT vs Pred ROI 级联评估 (113例)
特征缓存: experiments/cascade_eval/cache_gt|pred/*.npz (由 extract_cached.py 生成)
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.isdir(BASE):
    raise RuntimeError('BASE path not found; run from a checkout of the repository')
OUT = f'{BASE}/experiments/cascade_eval'
sys.path.insert(0, f'{BASE}/experiments/feature_dca')
sys.path.insert(0, f'{BASE}/experiments/physics_priors')

CAT2CLS = {'MTs': 0, 'ITs': 1, 'MGCTs': 2}

def load_meta():
    SRC = f'{BASE}/datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'
    rows = []
    for cat in ['MTs', 'ITs', 'MGCTs']:
        md = pd.read_excel(f'{SRC}/{cat}/{cat[:-1]}_patient.xlsx')
        g = [c for c in md.columns if 'ender' in c][0]
        a = [c for c in md.columns if 'Age' in c][0]
        l = [c for c in md.columns if 'ocation' in c][0]
        for _, r in md.iterrows():
            rows.append({'Patient_ID': r['Patient_ID'], 'gender': str(r[g]).upper(),
                         'age': float(r[a]), 'location': str(r[l]).lower().strip()})
    meta = pd.DataFrame(rows)
    meta['is_female'] = (meta['gender'] == 'F').astype(int)
    meta['age_log'] = np.log1p(meta['age'])
    dummies = pd.get_dummies(meta['location'], prefix='loc')
    return pd.concat([meta, dummies], axis=1)

def load_cache(cache_dir, stems, feat_cols, phys_cols, meta=None, meta_cols=None):
    rows = []
    for s in stems:
        f = f'{cache_dir}/{s}.npz'
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        d = dict(z['radiomics'])
        p = dict(z['phys']) if len(z['phys']) else {}
        row = {c: d.get(c, np.nan) for c in feat_cols}
        row.update({k: v for k, v in p.items() if k in phys_cols})
        # bootstrapb 修复: 测试集元数据必须用真实值 (与训练集特征模式一致), 而非均值填充
        if meta is not None and meta_cols and s in meta.index:
            m = meta.loc[s]
            for c in meta_cols:
                row[c] = float(m[c])
        rows.append((s, row))
    return rows

def main():
    split = json.load(open(f'{BASE}/datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
    train_stems = set(split['MT']['train'] + split['IT']['train'] + split['MGCT']['train']) - {'MT_411'}
    test_stems = set(split['MT']['test'] + split['IT']['test'] + split['MGCT']['test']) - {'MT_411'}
    cat_of = {}
    SRC = f'{BASE}/datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'
    for cat in ['MTs', 'ITs', 'MGCTs']:
        for f in os.listdir(f'{SRC}/{cat}/Masks'):
            cat_of[f.replace('.nii.gz', '')] = cat

    # 1) 训练集特征 (radiomics_svm GT 全量)
    df = pd.read_csv(f'{BASE}/experiments/radiomics_svm/features_full.csv')
    df = df[df['Patient_ID'] != 'MT_411'].reset_index(drop=True)
    from run_physics import extract_physics as extract_physics_gt
    phys = []
    for stem in df['Patient_ID']:
        p = extract_physics_gt(stem)
        if p:
            phys.append({'Patient_ID': stem, **p})
    df = df.merge(pd.DataFrame(phys), on='Patient_ID', how='left')
    meta = load_meta()
    meta_cols = ['is_female', 'age_log'] + [c for c in meta.columns if c.startswith('loc_')]
    meta_idx = meta.set_index('Patient_ID')
    df = df.merge(meta[['Patient_ID'] + meta_cols], on='Patient_ID', how='left')
    phys_cols = ['fat_fraction', 'calc_fraction', 'solid_fraction', 'hu_mean', 'hu_median',
                 'hu_std', 'hu_p5', 'hu_p95', 'hu_skew', 'hu_kurt', 'n_voxels']
    feat_cols = [c for c in df.columns if c not in ('Patient_ID', 'class') + tuple(phys_cols) + tuple(meta_cols)]
    all_cols = feat_cols + phys_cols + meta_cols
    tr = df[df['Patient_ID'].isin(train_stems)]
    X_tr, y_tr = tr[all_cols].values, tr['class'].values
    vt = VarianceThreshold(threshold=1e-6)
    X_tr_v = vt.fit_transform(X_tr)
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
    print(f'SVM trained: {len(tr)} cases, {len(keep)} features', flush=True)

    # 2) 测试集 (有预测 mask 的 113 例)
    pred_map = json.load(open(f'{OUT}/pred_map.json'))
    stems = sorted(s for s in test_stems if s in cat_of and s in pred_map)
    gt_rows = load_cache(f'{OUT}/cache_gt', stems, feat_cols, phys_cols, meta=meta_idx, meta_cols=meta_cols)
    pr_rows = load_cache(f'{OUT}/cache_pred', stems, feat_cols, phys_cols, meta=meta_idx, meta_cols=meta_cols)
    print(f'cache loaded: gt={len(gt_rows)}, pred={len(pr_rows)}', flush=True)

    def build_matrix(rows):
        X = np.full((len(rows), len(all_cols)), np.nan)
        for i, (s, row) in enumerate(rows):
            for j, c in enumerate(all_cols):
                if c in row:
                    X[i, j] = row[c]
        return X

    X_gt = build_matrix(gt_rows); X_pr = build_matrix(pr_rows)
    # NaN 填充: 用训练集该列均值 (从训练矩阵计算, 保持分布一致)
    tr_df = tr[all_cols]
    for j, c in enumerate(all_cols):
        col_mean = tr_df[c].mean()
        if np.isnan(col_mean):
            col_mean = 0.0
        X_gt[np.isnan(X_gt[:, j]), j] = col_mean
        X_pr[np.isnan(X_pr[:, j]), j] = col_mean
    y_te = np.array([CAT2CLS[cat_of[s]] for s, _ in gt_rows])
    # 对齐: 相同 stem 顺序
    X_gt_r = vt.transform(X_gt)[:, keep]
    X_pr_r = vt.transform(X_pr)[:, keep]

    def report(name, Xr):
        pred = pipe.predict(Xr); prob = pipe.predict_proba(Xr)
        acc = accuracy_score(y_te, pred)
        auc = roc_auc_score(y_te, prob, multi_class='ovr')
        rec = recall_score(y_te, pred, average=None)
        print(f'{name:10s} n={len(y_te)} ACC={acc:.4f} AUC={auc:.4f} recall={np.round(rec,3)}', flush=True)
        return pred, prob, acc, auc, rec

    pred_gt, prob_gt, acc_gt, auc_gt, rec_gt = report('GT ROI', X_gt_r)
    pred_pr, prob_pr, acc_pr, auc_pr, rec_pr = report('Pred ROI', X_pr_r)

    agree = (pred_gt == pred_pr).mean()
    diff_idx = np.where(pred_gt != pred_pr)[0]
    print(f'\n一致率: {agree:.4f} ({len(gt_rows)} 例); 不一致 {len(diff_idx)}')
    for i in diff_idx:
        s, _ = gt_rows[i]
        print(f'  {s:10s} {cat_of[s]:6s} true={y_te[i]} gt_pred={pred_gt[i]} pred_mask_pred={pred_pr[i]}')

    res = {'n': int(len(gt_rows)), 'gt_acc': float(acc_gt), 'pred_acc': float(acc_pr),
           'gt_auc': float(auc_gt), 'pred_auc': float(auc_pr),
           'agreement': float(agree), 'n_disagree': int(len(diff_idx))}
    json.dump(res, open(f'{OUT}/results_cascade_113.json', 'w'), indent=2)
    np.savez(f'{OUT}/cascade_113.npz', y_te=y_te, pred_gt=pred_gt, pred_pr=pred_pr, prob_gt=prob_gt, prob_pr=prob_pr)
    print('\nsaved -> results_cascade_113.json')

if __name__ == '__main__':
    main()
