#!/usr/bin/env python
"""radiomics_svm: 官方影像组学 SVM 三分类基线复现
- 特征: 1316 全量 PyRadiomics (original+wavelet+LoG), 窗宽窗位 35/350, 1mm³
- 筛选: 方差>0 + 训练集内相关性去冗余 (官方 ICC>0.80 的替代, 因无重复分割数据)
- 模型: SVM (RBF), 训练集内 5 折调参
- 评估: 独立测试集 (7:3), ACC/AUC/每类灵敏度/特异度
"""
import json, warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report)
warnings.filterwarnings('ignore')

def main():
    df = pd.read_csv('experiments/radiomics_svm/features_full.csv')
    # 剔除 MT_411 (Mask 损坏)
    df = df[df['Patient_ID'] != 'MT_411'].reset_index(drop=True)
    print(f'cases after removing MT_411: {len(df)}')

    split = json.load(open('datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
    # 训练/测试病例集合 (注意 MT_411 若在划分中也要剔除)
    train_stems = set(split['MT']['train'] + split['IT']['train'] + split['MGCT']['train'])
    test_stems = set(split['MT']['test'] + split['IT']['test'] + split['MGCT']['test'])
    train_stems.discard('MT_411')
    test_stems.discard('MT_411')

    train_df = df[df['Patient_ID'].isin(train_stems)]
    test_df = df[df['Patient_ID'].isin(test_stems)]
    print(f'train: {len(train_df)}, test: {len(test_df)}')
    print('train class dist:', train_df['class'].value_counts().to_dict())
    print('test class dist:', test_df['class'].value_counts().to_dict())

    feat_cols = [c for c in df.columns if c not in ('Patient_ID', 'class')]
    X_train = train_df[feat_cols].values
    y_train = train_df['class'].values
    X_test = test_df[feat_cols].values
    y_test = test_df['class'].values

    # 1. 方差筛选 (训练集内, 防泄漏)
    vt = VarianceThreshold(threshold=1e-6)
    X_train_v = vt.fit_transform(X_train)
    X_test_v = vt.transform(X_test)
    print(f'after variance filter: {X_train_v.shape[1]} features')

    # 2. 相关性去冗余 (训练集内, 皮尔逊 >0.95 保留其一)
    corr = np.corrcoef(X_train_v.T)
    keep = []
    dropped = 0
    for i in range(X_train_v.shape[1]):
        if all(abs(corr[i, j]) < 0.95 for j in keep):
            keep.append(i)
        else:
            dropped += 1
    X_train_r = X_train_v[:, keep]
    X_test_r = X_test_v[:, keep]
    print(f'after corr filter: {X_train_r.shape[1]} features (dropped {dropped})')

    # 3. 训练集内 5 折 GridSearch 调 SVM
    param_grid = {'svm__C': [0.1, 1, 10, 100], 'svm__gamma': ['scale', 0.001, 0.01, 0.1]}
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc_ovr', n_jobs=4, verbose=0)
    gs.fit(X_train_r, y_train)
    print(f'best params: {gs.best_params_}, best CV AUC: {gs.best_score_:.4f}')

    # 4. 测试集评估
    y_pred = gs.predict(X_test_r)
    y_prob = gs.predict_proba(X_test_r)
    acc = accuracy_score(y_test, y_pred)
    auc_ovr = roc_auc_score(y_test, y_prob, multi_class='ovr')
    print(f'\n=== TEST SET ===')
    print(f'Overall ACC: {acc:.4f} | Overall AUC(ovr): {auc_ovr:.4f}')
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=['MT','IT','MGCT'], digits=4))

    # 每类 AUC (one-vs-rest)
    print('\nPer-class AUC (ovr):')
    for i, name in enumerate(['MT','IT','MGCT']):
        auc = roc_auc_score((y_test == i).astype(int), y_prob[:, i])
        print(f'  {name}: {auc:.4f}')

    # 保存
    results = {
        'n_train': len(train_df), 'n_test': len(test_df),
        'n_features_after_filter': len(keep),
        'best_params': gs.best_params_,
        'cv_auc': float(gs.best_score_),
        'test_acc': float(acc), 'test_auc_ovr': float(auc_ovr),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'official_ref': {'acc': 0.726, 'auc': 0.885}
    }
    with open('experiments/radiomics_svm/svm_results.json','w') as f:
        json.dump(results, f, indent=2)
    print('\nresults saved -> experiments/radiomics_svm/svm_results.json')

if __name__ == '__main__':
    main()
