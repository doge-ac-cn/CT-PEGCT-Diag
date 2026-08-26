#!/usr/bin/env python
"""clean_ensembleb: 无泄漏协议下的 ensemble + location-aware 重算
- DL: clean 2D R18 (seed42) / clean 2.5D R18 (seed42) — val-selected epochs
- SVM: 与 run_locaware 相同的无权重 RBF C=10 gamma=0.001 (669 特征)
- 报告: 纯 2D 集成 (0.5/0.5) 与 位置感知 (纵隔→2.5D) 的 ACC/AUC
- 同时输出 5-seed clean 2D 概率平均的 ensemble (可选, 用 seed 42 的 SVM)
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
warnings.filterwarnings('ignore')

sys.path.insert(0, 'experiments/dl_ablation')
from train_ablation import ARCHS, make_resnet3d, ROIDataset

OUT = 'experiments/clean_ensemble'
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

def main():
    manifest = json.load(open('experiments/dl_ablation/data/manifest_ablation.json'))
    stems_2d, p_2d = predict_dl(manifest, '2d', f'{OUT}/results/clean_2d_r18.pt')
    stems_25, p_25 = predict_dl(manifest, '25d', f'{OUT}/results/clean_25d_r18.pt')
    assert stems_2d == stems_25
    stems = stems_2d

    # 5-seed 2D clean 概率平均
    p_2d_seeds = []
    for seed, tag in [(42, ''), (0, '_s0'), (1, '_s1'), (123, '_s123'), (2024, '_s2024')]:
        _, pp = predict_dl(manifest, '2d', f'{OUT}/results/clean_2d_r18{tag}.pt')
        p_2d_seeds.append(pp)
    p_2d_avg5 = np.mean(p_2d_seeds, axis=0)

    # 元数据: 位置
    meta_rows = []
    for cat in ['MTs', 'ITs', 'MGCTs']:
        md = pd.read_excel(f'{SRC}/{cat}/{cat[:-1]}_patient.xlsx')
        lcol = [c for c in md.columns if 'ocation' in c][0]
        for _, r in md.iterrows():
            meta_rows.append({'Patient_ID': r['Patient_ID'], 'cat': cat, 'location': str(r[lcol]).lower().strip()})
    meta = pd.DataFrame(meta_rows).set_index('Patient_ID').loc[stems]
    loc = meta['location'].values
    y_all = np.array([{'MTs': 0, 'ITs': 1, 'MGCTs': 2}[c] for c in meta['cat']])

    # SVM (与 run_locaware 相同: 无权重)
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

    med_mask = (loc == 'mediastinum')

    def report(name, p_ens):
        acc = accuracy_score(y_all, p_ens.argmax(1))
        auc = roc_auc_score(y_all, p_ens, multi_class='ovr')
        rec = recall_score(y_all, p_ens.argmax(1), average=None)
        med_acc = accuracy_score(y_all[med_mask], p_ens.argmax(1)[med_mask])
        med_it = med_mask & (y_all == 1)
        mit = int((p_ens.argmax(1)[med_it] == 1).sum()) if med_it.sum() else 'na'
        print(f'{name}: ACC={acc:.4f} AUC={auc:.4f} recall={np.round(rec,3)} 纵隔ACC={med_acc:.3f} 纵隔IT={mit}/{med_it.sum()}')
        return {'acc': float(acc), 'auc': float(auc), 'recall': rec.tolist(),
                'med_acc': float(med_acc), 'med_it': f'{mit}/{med_it.sum()}'}

    out = {'n': len(stems), 'clean_2d_1seed': {'acc': float(accuracy_score(y_all, p_2d.argmax(1))),
                                                'auc': float(roc_auc_score(y_all, p_2d, multi_class='ovr'))},
           'clean_2d_5seed_avg': {'acc': float(accuracy_score(y_all, p_2d_avg5.argmax(1))),
                                   'auc': float(roc_auc_score(y_all, p_2d_avg5, multi_class='ovr'))},
           'svm': {'acc': float(accuracy_score(y_all, p_svm.argmax(1))),
                   'auc': float(roc_auc_score(y_all, p_svm, multi_class='ovr'))}}
    # 原版对照 (test-selected 模型)
    stems_o, p_2d_o = predict_dl(manifest, '2d', 'experiments/dl_ablation/results/2d_r18_ce.pt')
    _, p_25_o = predict_dl(manifest, '25d', 'experiments/dl_ablation/results/25d_r18_ce.pt')
    out['orig_testsel_ensemble'] = report('orig_ensemble', (p_2d_o + p_svm) / 2)
    out['orig_testsel_locaware'] = report('orig_locaware', (np.where(med_mask[:, None], p_25_o, p_2d_o) + p_svm) / 2)
    # clean 版本
    out['clean_1seed_ensemble'] = report('clean_1seed_ensemble', (p_2d + p_svm) / 2)
    out['clean_5seed_ensemble'] = report('clean_5seed_ensemble', (p_2d_avg5 + p_svm) / 2)
    out['clean_locaware_1seed'] = report('clean_locaware_1seed', (np.where(med_mask[:, None], p_25, p_2d) + p_svm) / 2)
    out['clean_locaware_5seed'] = report('clean_locaware_5seed', (np.where(med_mask[:, None], p_25, p_2d_avg5) + p_svm) / 2)

    json.dump(out, open(f'{OUT}/ensemble_clean.json', 'w'), indent=2, ensure_ascii=False)
    # 保存逐例概率
    np.save(f'{OUT}/clean_p_2d.npy', p_2d); np.save(f'{OUT}/clean_p_2d_avg5.npy', p_2d_avg5)
    np.save(f'{OUT}/clean_p_25.npy', p_25); np.save(f'{OUT}/clean_p_svm.npy', p_svm)
    np.save(f'{OUT}/clean_y.npy', y_all); np.save(f'{OUT}/clean_stems.npy', np.array(stems))
    print(f'\nsaved -> {OUT}/ensemble_clean.json')

if __name__ == '__main__':
    main()
