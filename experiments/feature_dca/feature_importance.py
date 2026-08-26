#!/usr/bin/env python
"""特征重要性分析: SVM 影像组学特征的可解释性
- 用 sklearn permutation_importance 评估 radiomics_svm SVM 的 Top 特征
- 识别哪些影像特征驱动 MT/IT/MGCT 分类
"""
import json, warnings
import numpy as np, pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

df = pd.read_csv('experiments/radiomics_svm/features_full.csv')
df = df[df['Patient_ID'] != 'MT_411'].reset_index(drop=True)
split = json.load(open('datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
train_stems = set(split['MT']['train'] + split['IT']['train'] + split['MGCT']['train']) - {'MT_411'}
test_stems = set(split['MT']['test'] + split['IT']['test'] + split['MGCT']['test']) - {'MT_411'}
train_df = df[df['Patient_ID'].isin(train_stems)]; test_df = df[df['Patient_ID'].isin(test_stems)]
feat_cols = [c for c in df.columns if c not in ('Patient_ID', 'class')]
X_tr, y_tr = train_df[feat_cols].values, train_df['class'].values
X_te, y_te = test_df[feat_cols].values, test_df['class'].values

vt = VarianceThreshold(threshold=1e-6)
X_tr_v = vt.fit_transform(X_tr); X_te_v = vt.transform(X_te)
corr = np.corrcoef(X_tr_v.T)
keep = [i for i in range(X_tr_v.shape[1]) if all(abs(corr[i,j]) < 0.95 for j in range(i))]
kept_cols = [feat_cols[i] for i in range(len(feat_cols)) if vt.get_support()[i]]
kept_cols = [kept_cols[i] for i in keep]

pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=10, gamma=0.001, random_state=42))])
pipe.fit(X_tr_v[:, keep], y_tr)

# permutation importance (测试集, 用 accuracy 作为评分)
print('computing permutation importance (this takes a few minutes)...')
result = permutation_importance(pipe, X_te_v[:, keep], y_te, n_repeats=10,
                                scoring='accuracy', n_jobs=4, random_state=42)
imp = pd.DataFrame({
    'feature': kept_cols,
    'importance': result.importances_mean,
    'std': result.importances_std
}).sort_values('importance', ascending=False)

print('\n=== Top 20 重要特征 ===')
print(imp.head(20).to_string(index=False))

# 特征族分布
imp['family'] = imp['feature'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[1] if len(x.split('_'))>1 else x)
print('\n=== 特征族汇总 ===')
fam = imp.groupby('family')['importance'].sum().sort_values(ascending=False)
print(fam.head(10).to_string())

imp.to_csv('experiments/feature_dca/feature_importance.csv', index=False)
print('\nsaved -> experiments/feature_dca/feature_importance.csv')
