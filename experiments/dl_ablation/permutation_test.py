#!/usr/bin/env python
"""voxel permutation control: 体素置换对照 — 信号在直方图还是空间纹理?
- 对测试集 ROI: 原始 vs 随机置换体素 (保留 HU 直方图, 破坏空间纹理)
- 全量组学特征 (原始/置换) → 同协议 SVM (训练集原始特征训练)
- 若置换后性能不变 → 空间纹理无贡献, 直方图足够
"""
import os, json, warnings
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from radiomics import featureextractor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
warnings.filterwarnings('ignore')

SRC = 'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'
OUT = 'experiments/dl_ablation'
os.makedirs(f'{OUT}/permuted', exist_ok=True)
SEED = 42

def window_and_normalize(image_sitk, level=35, width=350):
    arr = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
    lower = level - width / 2.0
    upper = level + width / 2.0
    arr = np.clip(arr, lower, upper)
    arr = (arr - lower) / (upper - lower) * 255.0
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image_sitk)
    return out

def permute_roi(img_sitk, mask_sitk, seed=SEED):
    """在 mask 内随机置换体素 (保持直方图), 输出新 mask (保留原 spacing 信息用于特征提取)"""
    arr = sitk.GetArrayFromImage(img_sitk).copy()
    m = sitk.GetArrayFromImage(mask_sitk) > 0
    rng = np.random.RandomState(seed)
    vox = arr[m]
    perm = rng.permutation(vox)
    arr[m] = perm
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img_sitk)
    return out

def extract(img_path, mask_path, params='experiments/radiomics_svm/radiomics_params_full.json'):
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    img = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(mask_path)
    img_w = window_and_normalize(img)
    feat = extractor.execute(img_w, mask)
    return {k: float(v) for k, v in feat.items() if not k.startswith('diagnostics')}

def main():
    split = json.load(open('datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
    train_stems = set(split['MT']['train'] + split['IT']['train'] + split['MGCT']['train']) - {'MT_411'}
    test_stems = sorted(set(split['MT']['test'] + split['IT']['test'] + split['MGCT']['test']) - {'MT_411'})

    # 1) 训练集: 原始特征 (复用 features_full.csv 训练)
    df = pd.read_csv('experiments/radiomics_svm/features_full.csv')
    df = df[df['Patient_ID'] != 'MT_411'].reset_index(drop=True)
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
    print(f'SVM trained: {len(keep)} features', flush=True)

    # 2) 测试集: 原始 + 置换特征
    cat_map = {}
    for cat in ['MTs', 'ITs', 'MGCTs']:
        for f in os.listdir(f'{SRC}/{cat}/Images'):
            cat_map[f.replace('.nii.gz', '')] = cat

    rows = []
    for i, stem in enumerate(test_stems):
        cat = cat_map[stem]
        img_p = f'{SRC}/{cat}/Images/{stem}.nii.gz'
        mask_p = f'{SRC}/{cat}/Masks/{stem}.nii.gz'
        try:
            f_orig = extract(img_p, mask_p)
            # 置换 (每例不同 seed 保证独立)
            img_sitk = sitk.ReadImage(img_p)
            mask_sitk = sitk.ReadImage(mask_p)
            perm_sitk = permute_roi(img_sitk, mask_sitk, seed=SEED + i)
            perm_path = f'{OUT}/permuted/{stem}.nii.gz'
            sitk.WriteImage(perm_sitk, perm_path)
            f_perm = extract(perm_path, mask_p)
        except Exception as e:
            print(f'ERROR {stem}: {e}', flush=True)
            continue
        cls = {'MTs': 0, 'ITs': 1, 'MGCTs': 2}[cat]
        for tag, f in [('orig', f_orig), ('perm', f_perm)]:
            row = {'stem': stem, 'tag': tag, 'cls': cls}
            for c in feat_cols:
                row[c] = f.get(c, np.nan)
            rows.append(row)
        if (i + 1) % 30 == 0:
            print(f'{i+1}/{len(test_stems)}', flush=True)

    ev = pd.DataFrame(rows)
    print('\n=== 体素置换对照 ===')
    for tag in ['orig', 'perm']:
        sub = ev[ev['tag'] == tag]
        X = vt.transform(sub[feat_cols].values)[:, keep]
        p = pipe.predict_proba(X)
        pred = p.argmax(1)
        y = sub['cls'].values
        acc = accuracy_score(y, pred)
        auc = roc_auc_score(y, p, multi_class='ovr')
        rec = recall_score(y, pred, average=None)
        print(f'{tag:4s}: ACC={acc:.4f} AUC={auc:.4f} recall={np.round(rec,3)} (n={len(sub)})')

    # 3) 特征类别分解: firstorder vs texture (置换后两者应差异大)
    fo_cols = [c for c in feat_cols if 'firstorder' in c]
    tx_cols = [c for c in feat_cols if ('glcm' in c or 'glrlm' in c or 'glszm' in c or 'gldm' in c or 'ngtdm' in c)]
    print(f'\n特征类别: firstorder={len(fo_cols)}, texture={len(tx_cols)}')
    for grp_name, cols in [('firstorder', fo_cols), ('texture', tx_cols)]:
        # 单独用该类别特征训练 (5折CV训练集内, 简化: 直接用原训练集训练)
        Xg = train_df[cols].values
        vtg = VarianceThreshold(threshold=1e-6)
        Xg_v = vtg.fit_transform(Xg)
        kg = list(range(Xg_v.shape[1]))
        try:
            corr = np.corrcoef(Xg_v.T)
            kg = [i for i in range(Xg_v.shape[1]) if all(abs(corr[i, j]) < 0.95 for j in range(i))]
        except Exception:
            pass
        pipeg = Pipeline([('scaler', StandardScaler()),
                          ('svm', SVC(kernel='rbf', C=10, gamma=0.001, probability=True, random_state=42))])
        pipeg.fit(Xg_v[:, kg], y_tr)
        print(f'\n{grp_name} 单独:')
        for tag in ['orig', 'perm']:
            sub = ev[ev['tag'] == tag]
            X = vtg.transform(sub[cols].values)[:, kg]
            p = pipeg.predict_proba(X)
            y = sub['cls'].values
            acc = accuracy_score(y, p.argmax(1))
            auc = roc_auc_score(y, p, multi_class='ovr')
            print(f'  {tag}: ACC={acc:.4f} AUC={auc:.4f}')

    out = {'n_test': len(test_stems), 'note': 'see log for full results'}
    json.dump(out, open(f'{OUT}/results_permutation.json', 'w'), indent=2)
    print('\ndone')

if __name__ == '__main__':
    main()
