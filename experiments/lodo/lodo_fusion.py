#!/usr/bin/env python
"""lodo_fusion.py — LODO 融合评估与完整报告

1) 每域: DL 5-seed ensemble 概率 + SVM 概率 → 平均融合（论文 ensemble 协议）
2) 汇总表: 内部基线 / SVM / DL / 融合 (test 与 test+rare 双口径)
3) 漂移-性能散点图 (PCA median 距离 vs ACC)
4) 错误案例归纳 (SVM/DL/融合 一致性)

输出: 本目录 lodo_final_report.md + lodo_summary.csv + lodo_all_results.json + drift_perf.png

依赖（需先运行）:
  - make_domain_splits.py → domain_labels.csv / lodo_splits.json
  - pca_drift.py → domain_distances.json
  - lodo_svm.py → LODO-SVM 结果（本脚本按论文固定超参 C=10 γ=0.001 重新拟合）
  - lodo_train_dl.py → experiments/lodo/results/{domain}/s{seed}.pt（5 seeds 权重）
  - experiments/radiomics_svm/features_full.csv
  - experiments/dl_ablation/data/{stem}_2d.npy + manifest_ablation.json
"""
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FEAT_CSV = os.path.join(REPO, 'experiments', 'radiomics_svm', 'features_full.csv')
LABELS_CSV = os.path.join(HERE, 'domain_labels.csv')
LODO_JSON = os.path.join(HERE, 'lodo_splits.json')
DIST_JSON = os.path.join(HERE, 'domain_distances.json')
D116 = os.path.join(HERE, 'results')
DATA = os.path.join(REPO, 'experiments', 'dl_ablation', 'data')
DOMAINS = ['ovary', 'testis', 'retroperitoneum', 'sacrococcyx', 'mediastinum']
SEEDS = [42, 0, 1, 123, 2024]

sys.path.insert(0, os.path.join(REPO, 'experiments', 'dl_ablation'))
from train_ablation import ARCHS  # noqa: E402

feat = pd.read_csv(FEAT_CSV)
feat = feat[feat['Patient_ID'] != 'MT_411'].reset_index(drop=True)
lodo = json.load(open(LODO_JSON))
labels = pd.read_csv(LABELS_CSV)
df = feat.merge(labels[['Patient_ID', 'domain']], on='Patient_ID')
feat_cols = [c for c in feat.columns if c not in ('Patient_ID', 'class')]
X = df[feat_cols].values; y = df['class'].values; stems = df['Patient_ID'].values
manifest = json.load(open(os.path.join(DATA, 'manifest_ablation.json')))
stem2cls = {m['stem']: m['cls'] for m in manifest}


def load_dl_probs(domain, stems_list):
    """加载 5 seed .pt 推理, 返回 (n,3) 概率均值"""
    probs = []
    model = ARCHS['18'](weights=None, num_classes=3)
    for seed in SEEDS:
        model.load_state_dict(torch.load(os.path.join(D116, domain, f's{seed}.pt'), map_location='cpu'))
        model.eval()
        ps = []
        with torch.no_grad():
            for stem in stems_list:
                x = torch.from_numpy(np.load(os.path.join(DATA, f'{stem}_2d.npy'))).float().unsqueeze(0)
                ps.append(torch.softmax(model(x), 1).numpy()[0])
        probs.append(np.array(ps))
    return np.mean(probs, axis=0)


def fit_svm(tr_stems):
    X_tr = X[np.isin(stems, tr_stems)]; y_tr = y[np.isin(stems, tr_stems)]
    vt = VarianceThreshold(threshold=1e-6); Xv = vt.fit_transform(X_tr)
    corr = np.corrcoef(Xv.T)
    keep = [i for i in range(Xv.shape[1]) if all(abs(corr[i, j]) < 0.95 for j in range(i))]
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('svm', SVC(kernel='rbf', C=10, gamma=0.001, probability=True,
                                 random_state=42, class_weight='balanced'))])
    pipe.fit(Xv[:, keep], y_tr)
    return pipe, vt, keep


def predict_svm(pipe, vt, keep, stems_list):
    idx = {s: i for i, s in enumerate(stems)}
    Xt = np.stack([X[idx[s]] for s in stems_list])  # 严格按 stems_list 顺序
    Xt = vt.transform(Xt)[:, keep]
    return pipe.predict_proba(Xt), pipe.predict(Xt)  # (proba, decision-argmax 口径)


def evaluate(y_true, p):
    pred = p.argmax(1)
    return {'acc': float(accuracy_score(y_true, pred)),
            'auc': float(roc_auc_score(y_true, p, multi_class='ovr')),
            'recall': [float(r) for r in recall_score(y_true, pred, average=None)],
            'cm': confusion_matrix(y_true, pred).tolist()}


rows = []
for dom in DOMAINS:
    tr = lodo[dom]['train']; te = lodo[dom]['test']; rare = lodo[dom]['rare_test']
    te_stems = te + rare
    y_te = np.array([stem2cls[s] for s in te_stems])
    # DL ensemble
    p_dl = load_dl_probs(dom, te_stems)
    # SVM (proba + predict 双口径)
    pipe, vt, keep = fit_svm(tr)
    p_svm, pred_svm = predict_svm(pipe, vt, keep, te_stems)
    # 融合
    p_ens = (p_dl + p_svm) / 2
    r_dl = evaluate(y_te, p_dl); r_svm = evaluate(y_te, p_svm); r_ens = evaluate(y_te, p_ens)
    # 纯 test (不含 rare)
    n_te = len(te)
    r_dl_test = evaluate(y_te[:n_te], p_dl[:n_te])
    r_svm_test = evaluate(y_te[:n_te], p_svm[:n_te])
    r_svm_test_pred = {'acc': float(accuracy_score(y_te[:n_te], pred_svm[:n_te]))}  # predict 口径
    r_ens_test = evaluate(y_te[:n_te], p_ens[:n_te])
    # 错误案例
    wrong_dl = [s for s, pr in zip(te_stems, p_dl.argmax(1)) if stem2cls[s] != pr]
    wrong_svm = [s for s, pr in zip(te_stems, pred_svm) if stem2cls[s] != pr]  # predict 口径
    wrong_ens = [s for s, pr in zip(te_stems, p_ens.argmax(1)) if stem2cls[s] != pr]
    both_wrong = sorted(set(wrong_dl) & set(wrong_svm))
    rows.append({'domain': dom, 'n_test': n_te, 'n_rare': len(rare),
                 'svm_acc': r_svm_test['acc'], 'svm_auc': r_svm_test['auc'],
                 'svm_acc_pred': r_svm_test_pred['acc'],
                 'svm_recall': r_svm_test['recall'],
                 'dl_acc': r_dl_test['acc'], 'dl_auc': r_dl_test['auc'],
                 'dl_recall': r_dl_test['recall'],
                 'ens_acc': r_ens_test['acc'], 'ens_auc': r_ens_test['auc'],
                 'ens_recall': r_ens_test['recall'],
                 'ens_acc_tr': r_ens['acc'], 'ens_auc_tr': r_ens['auc'], 'ens_recall_tr': r_ens['recall'],
                 'wrong_dl': wrong_dl, 'wrong_svm': wrong_svm, 'wrong_ens': wrong_ens,
                 'both_wrong': both_wrong})
    print(f"[{dom}] SVM ACC={r_svm_test['acc']:.4f}(proba)/{r_svm_test_pred['acc']:.4f}(pred) | "
          f"DL ACC={r_dl_test['acc']:.4f} | ENS ACC={r_ens_test['acc']:.4f} | "
          f"ENS+rare ACC={r_ens['acc']:.4f} | both_wrong={len(both_wrong)}", flush=True)

json.dump(rows, open(os.path.join(HERE, 'lodo_all_results.json'), 'w'), ensure_ascii=False, indent=2)

# ---- 汇总表 ----
summ = pd.DataFrame([{k: r[k] for k in ['domain', 'n_test', 'n_rare', 'svm_acc', 'svm_acc_pred',
                                        'svm_auc', 'dl_acc', 'dl_auc', 'ens_acc', 'ens_auc',
                                        'ens_acc_tr', 'ens_auc_tr']} for r in rows])
for k in ['svm', 'dl', 'ens']:
    summ[f'{k}_recall'] = [r[f'{k}_recall'] for r in rows]
summ.to_csv(os.path.join(HERE, 'lodo_summary.csv'), index=False)

# 漂移-性能 (PCA median 距离, 与 pca_drift.py)
dist = json.load(open(DIST_JSON))['per_lodo_test_dist']
fig, ax = plt.subplots(figsize=(7, 5.5))
colors = {'ovary': '#e15759', 'testis': '#4e79a7', 'retroperitoneum': '#59a14f',
          'sacrococcyx': '#f28e2b', 'mediastinum': '#b07aa1'}
for r in rows:
    d = r['domain']
    ax.scatter(dist[d]['test_to_train_centroid_median'], r['ens_acc'], s=120, color=colors[d], label=d, zorder=3)
    ax.annotate(d, (dist[d]['test_to_train_centroid_median'] + 0.5, r['ens_acc'] + 0.008), fontsize=10)
ax.axhline(0.902, ls='--', color='gray', label='内部5折融合基线 0.902')
ax.axhline(0.886, ls=':', color='gray', label='内部SVM基线 0.886')
ax.set_xlabel('PCA域外漂移距离 (median, pca_drift.py)')
ax.set_ylabel('LODO 融合 ACC (test域)')
ax.set_title('LODO: 特征漂移 vs 融合性能')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'drift_perf.png'), dpi=150)

# ---- Markdown 报告 ----
md = ['# LODO（留一解剖部位域）验证报告 — CT-PEGCT-Diag EGCT 三分类', '']
md.append('## 协议')
md.append('- 641 例（剔除 MT_411），解剖部位域：ovary 213 / testis 141 / retroperitoneum 134 / '
          'sacrococcyx 97（合并 sacrococcyx+sacrococcygeal）/ mediastinum 45 / other 11（罕见，仅测试）')
md.append('- 每个留出域：训练 = 其余 4 大域（不含罕见域），测试 = 留出域 + 罕见域')
md.append('- SVM：radiomics 1316→方差+corr 去冗余，RBF C=10 γ=0.001，class_weight=balanced（与论文协议一致）')
md.append('- SVM ACC 口径：单模型用 `predict`（decision function，与论文基线 0.886 一致）；'
          '融合用 `predict_proba`（Platt 校准，概率平均所需）。两者 argmax 约有 5% 样本不一致（SVC 已知特性）')
md.append('- DL：2D ResNet18，60 epochs，AdamW 1e-3 cosine，test-selected epoch，5 seeds（42/0/1/123/2024）概率平均')
md.append('- 融合：DL ensemble 与 SVM 概率平均（论文 ensemble 协议）')
md.append('')
md.append('## 逐域结果（test 域，不含罕见域；SVM 为 predict 口径）')
md.append('| 留出域 | N | SVM ACC | DL ACC | 融合 ACC | SVM AUC | DL AUC | 融合 AUC | 融合 recall (MT/IT/MGCT) |')
md.append('|---|---|---|---|---|---|---|---|---|')
for r in rows:
    rec = '/'.join(f'{v:.3f}' for v in r['ens_recall'])
    md.append(f"| {r['domain']} | {r['n_test']} | {r['svm_acc_pred']:.3f} | {r['dl_acc']:.3f} | "
              f"**{r['ens_acc']:.3f}** | {r['svm_auc']:.3f} | {r['dl_auc']:.3f} | {r['ens_auc']:.3f} | {rec} |")
md.append('')
md.append('> 注：SVM proba 口径 ACC（与融合同口径）：'
          + '、'.join(f"{r['domain']} {r['svm_acc']:.3f}" for r in rows))
md.append('')
md.append('## 内部基线对比')
md.append('- 论文 5 折内部测试（193 例）：SVM ACC 0.886 / AUC 0.950；DL ensemble 0.881 / 0.939；融合 0.902 / 0.962')
n_total = sum(r['n_test'] for r in rows)
w_svm = np.sum([r['svm_acc_pred'] * r['n_test'] for r in rows]) / n_total
w_dl = np.sum([r['dl_acc'] * r['n_test'] for r in rows]) / n_total
w_ens = np.sum([r['ens_acc'] * r['n_test'] for r in rows]) / n_total
md.append(f"- LODO 平均（5 域，测试集加权）：SVM ACC {w_svm:.3f}、DL ACC {w_dl:.3f}、融合 ACC {w_ens:.3f}")
md.append('')
md.append('## 漂移-性能关系（pca_drift.py PCA median 距离）')
md.append('| 域 | PCA 域外距离 (median) | 融合 ACC |')
md.append('|---|---|---|')
for r in rows:
    d = dist[r['domain']]['test_to_train_centroid_median']
    md.append(f"| {r['domain']} | {d:.1f} | {r['ens_acc']:.3f} |")
md.append('- 内部测试集同分布基线：PCA median 距离 28.5')
md.append('')
md.append('## 错误案例（test 域，SVM 与 DL 同时错）')
for r in rows:
    md.append(f"- **{r['domain']}**（{len(r['both_wrong'])} 例）："
              f"{', '.join(r['both_wrong']) if r['both_wrong'] else '无'}")
md.append('')
md.append('## 罕见域（other，11 例，仅测试）')
for r in rows:
    md.append(f"- {r['domain']} 训练模型：融合 ACC {r['ens_acc_tr']:.3f}（含 rare {r['n_rare']} 例）")
with open(os.path.join(HERE, 'lodo_final_report.md'), 'w') as f:
    f.write('\n'.join(md))

print('\n报告已保存:', os.path.join(HERE, 'lodo_final_report.md'))
print(pd.DataFrame([{k: r[k] for k in ['domain', 'svm_acc', 'dl_acc', 'ens_acc', 'ens_auc', 'ens_acc_tr']}
                    for r in rows]).to_string(index=False))
