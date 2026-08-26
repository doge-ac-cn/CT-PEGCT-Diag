#!/usr/bin/env python
"""论文级可视化: 三分类 ROC 对比 + 混淆矩阵热图
数据: radiomics_svm SVM, 3D CNN baseline CNN, 官方基线
"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
warnings.filterwarnings('ignore')

OUT = 'experiments/feature_dca'
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'figure.dpi': 150})

# === 重算 SVM 概率 (radiomics_svm 配置) ===
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
pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=10, gamma=0.001, probability=True, random_state=42))])
pipe.fit(X_tr_v[:, keep], y_tr)
svm_prob = pipe.predict_proba(X_te_v[:, keep])
print('SVM probs computed')

# === 图1: 三分类 ROC (SVM vs CNN vs 官方) ===
# CNN 概率 (从 3D CNN baseline 模型重算)
import torch, sys
sys.path.insert(0, 'experiments/cnn_baseline')
from train_cnn import Simple3DCNN, ROIDataset
torch.manual_seed(42); np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Simple3DCNN(n_classes=3).to(device)
model.load_state_dict(torch.load('experiments/cnn_baseline/model_base_best.pt', map_location=device))
model.eval()
manifest = json.load(open('experiments/cnn_baseline/data/manifest.json'))
test_ds = ROIDataset(manifest, test_stems)
from torch.utils.data import DataLoader
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)
cnn_prob = []
with torch.no_grad():
    for x, _ in test_loader:
        cnn_prob.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
cnn_prob = np.concatenate(cnn_prob)
print('CNN probs computed:', cnn_prob.shape)

names = ['MT', 'IT', 'MGCT']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, name in enumerate(names):
    ax = axes[i]
    y_bin = (y_te == i).astype(int)
    # SVM
    fpr, tpr, _ = roc_curve(y_bin, svm_prob[:, i])
    a = auc(fpr, tpr)
    ax.plot(fpr, tpr, colors[i], lw=2, label=f'SVM (AUC={a:.3f})')
    # CNN
    fpr2, tpr2, _ = roc_curve(y_bin, cnn_prob[:, i])
    a2 = auc(fpr2, tpr2)
    ax.plot(fpr2, tpr2, '--', color=colors[i], lw=1.5, alpha=0.7, label=f'CNN (AUC={a2:.3f})')
    # 官方
    official = {'MT': 0.843, 'IT': 0.797, 'MGCT': 0.869}[name]
    ax.plot([0,1], [0,1], 'k:', lw=1, label=f'Official SVM (AUC={official:.3f})')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{name} ROC')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/roc_comparison.png')
plt.close()
print('ROC saved ->', f'{OUT}/roc_comparison.png')

# === 图2: SVM 混淆矩阵热图 ===
y_pred = svm_prob.argmax(axis=1)
cm = confusion_matrix(y_te, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(cm, display_labels=names)
disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')
ax.set_title('SVM Confusion Matrix (Test Set)')
plt.tight_layout()
plt.savefig(f'{OUT}/confusion_matrix.png')
plt.close()
print('Confusion matrix saved ->', f'{OUT}/confusion_matrix.png')

# === 图3: 连续谱可视化 (solid_fraction 箱线图) ===
import nibabel as nib, glob, os
SRC = 'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'
rows = []
for cat, cls in [('MTs',0),('ITs',1),('MGCTs',2)]:
    for mp in sorted(glob.glob(f'{SRC}/{cat}/Masks/*.nii.gz')):
        stem = os.path.basename(mp).replace('.nii.gz','')
        if stem == 'MT_411': continue
        img = nib.load(f'{SRC}/{cat}/Images/{stem}.nii.gz').get_fdata()
        mask = nib.load(mp).get_fdata()
        hu = img[mask > 0]
        if len(hu) == 0: continue
        rows.append({'class': cls, 'solid': (hu > 20).mean(), 'fat': (hu < -30).mean()})
pdf = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
bp = axes[0].boxplot([pdf[pdf['class']==c]['solid'] for c in [0,1,2]],
                     tick_labels=names, patch_artist=True, showfliers=False)
colors_bp = ['#1f77b4', '#ff7f0e', '#2ca02c']
for patch, c in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(c); patch.set_alpha(0.6)
axes[0].set_title('Solid Fraction (HU>20) by Class')
axes[0].set_ylabel('Proportion of tumor voxels')
axes[0].grid(alpha=0.3)
bp2 = axes[1].boxplot([pdf[pdf['class']==c]['fat'] for c in [0,1,2]],
                      tick_labels=names, patch_artist=True, showfliers=False)
for patch, c in zip(bp2['boxes'], colors_bp):
    patch.set_facecolor(c); patch.set_alpha(0.6)
axes[1].set_title('Fat Fraction (HU<-30) by Class')
axes[1].set_ylabel('Proportion of tumor voxels')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/continuum.png')
plt.close()
print('Continuum saved ->', f'{OUT}/continuum.png')
print('\nAll figures generated.')
