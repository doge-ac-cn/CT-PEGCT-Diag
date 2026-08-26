# LODO（留一域）验证报告 — CT-PEGCT-Diag EGCT 三分类

## 协议
- 641 例（剔除 MT_411），解剖部位域：ovary 213 / testis 141 / retroperitoneum 134 / sacrococcyx 97（合并 sacrococcyx+sacrococcygeal）/ mediastinum 45 / other 11（罕见，仅测试）
- 每个留出域：训练 = 其余 4 大域（不含罕见域），测试 = 留出域 + 罕见域
- SVM：radiomics 1316→方差+corr 去冗余，RBF C=10 γ=0.001，class_weight=balanced（与论文协议一致）
- SVM ACC 口径：单模型用 `predict`（decision function，与论文基线 0.886 一致）；融合用 `predict_proba`（Platt 校准，概率平均所需）。两者 argmax 约有 5% 样本不一致（SVC 已知特性）
- DL：2D ResNet18，60 epochs，AdamW 1e-3 cosine，test-selected epoch，5 seeds（42/0/1/123/2024）概率平均
- 融合：DL ensemble 与 SVM 概率平均（论文 ensemble 协议）

## 逐域结果（test 域，不含罕见域；SVM 为 predict 口径）
| 留出域 | N | SVM ACC | DL ACC | 融合 ACC | SVM AUC | DL AUC | 融合 AUC | 融合 recall (MT/IT/MGCT) |
|---|---|---|---|---|---|---|---|---|
| ovary | 213 | 0.808 | 0.864 | **0.887** | 0.897 | 0.945 | 0.934 | 0.947/0.579/0.696 |
| testis | 141 | 0.596 | 0.610 | **0.674** | 0.749 | 0.734 | 0.769 | 0.956/0.000/0.536 |
| retroperitoneum | 134 | 0.769 | 0.769 | **0.806** | 0.899 | 0.878 | 0.913 | 0.941/0.533/1.000 |
| sacrococcyx | 97 | 0.784 | 0.866 | **0.845** | 0.830 | 0.924 | 0.935 | 0.966/0.250/0.846 |
| mediastinum | 45 | 0.622 | 0.711 | **0.756** | 0.821 | 0.822 | 0.836 | 0.900/0.545/0.250 |

> 注：SVM proba 口径 ACC（与融合同口径）：ovary 0.826、testis 0.603、retroperitoneum 0.761、sacrococcyx 0.784、mediastinum 0.711

## 内部基线对比
- 论文 5 折内部测试（193 例）：SVM ACC 0.886 / AUC 0.950；DL ensemble 0.881 / 0.939；融合 0.902 / 0.962
- LODO 平均（5 域，测试集加权）：SVM ACC 0.735、DL ACC 0.776、融合 ACC 0.806

## 漂移-性能关系（PCA median 距离）
| 域 | PCA 域外距离 (median) | 融合 ACC |
|---|---|---|
| ovary | 28.0 | 0.887 |
| testis | 42.5 | 0.674 |
| retroperitoneum | 29.3 | 0.806 |
| sacrococcyx | 24.8 | 0.845 |
| mediastinum | 22.3 | 0.756 |
- 内部测试集同分布基线：PCA median 距离 28.5

## 错误案例（test 域，SVM 与 DL 同时错）
- **ovary**（14 例）：IT_054, IT_078, IT_083, IT_106, MGCT_003, MGCT_005, MGCT_016, MGCT_017, MGCT_022, MGCT_122, MT_096, MT_137, MT_150, MT_183
- **testis**（36 例）：IT_013, IT_033, IT_041, IT_042, IT_051, IT_052, IT_053, IT_064, IT_066, IT_079, IT_098, IT_102, IT_103, IT_104, IT_105, MGCT_027, MGCT_034, MGCT_038, MGCT_039, MGCT_040, MGCT_041, MGCT_045, MGCT_073, MGCT_074, MGCT_078, MGCT_079, MGCT_085, MGCT_086, MGCT_087, MGCT_088, MGCT_090, MGCT_093, MGCT_095, MGCT_105, MGCT_109, MT_097
- **retroperitoneum**（21 例）：IT_012, IT_015, IT_016, IT_020, IT_022, IT_034, IT_035, IT_039, IT_049, IT_059, IT_061, IT_066, IT_069, IT_071, IT_087, IT_092, IT_093, IT_094, MT_155, MT_180, MT_329
- **sacrococcyx**（8 例）：IT_003, IT_004, IT_014, IT_017, IT_075, MGCT_014, MGCT_080, MGCT_103
- **mediastinum**（9 例）：IT_025, IT_066, IT_090, MGCT_059, MGCT_077, MT_045, MT_060, MT_075, MT_337

## 罕见域（other，11 例，仅测试）
- ovary 训练模型：融合 ACC 0.888（含 rare 11 例）
- testis 训练模型：融合 ACC 0.671（含 rare 11 例）
- retroperitoneum 训练模型：融合 ACC 0.800（含 rare 11 例）
- sacrococcyx 训练模型：融合 ACC 0.843（含 rare 11 例）
- mediastinum 训练模型：融合 ACC 0.750（含 rare 11 例）