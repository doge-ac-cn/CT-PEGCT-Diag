# FIXES — 代码修复记录

本文件记录独立验证过程中发现并修复的代码问题。所有修复均已通过语法检查与数值验证。

## 1. 置换检验特征提取参数不一致（已修复）

**问题**：`experiments/dl_ablation/permutation_test.py` 的测试集特征提取使用
`radiomics_params_nn.json`（`sitkNearestNeighbor` 插值），而训练集特征
（`features_full.csv`）使用 `radiomics_params_full.json`（`sitkBSpline` 插值）。
插值方式不一致会导致置换检验的原始 AUC 被低估（仅 0.746，而非预期 0.941），
无法复现论文报告数字。

**修复**：将参数引用统一为 `experiments/radiomics_svm/radiomics_params_full.json`，
与训练集提取协议一致（其余参数完全相同，仅 `interpolator` 一项差异）。

**验证**：统一为 BSpline 插值后可复现论文置换检验数字（原始 AUC 0.941 →
置换后 0.733；IT recall 0.594 → 0.031）。

## 2. manifest 文件名不一致（已修复）

**问题**：`experiments/dl_ablation/prepare_ablation.py` 输出 `manifest.json`，
但下游脚本（`train_ablation.py` 等）读取 `manifest_ablation.json`，
首次运行时会导致"找不到 manifest"断点。

**修复**：输出时同时写出 `manifest.json` 与 `manifest_ablation.json`，
两个文件内容完全一致，统一入口。

## 3. run_final.py 缺失（已补全）

**问题**：`experiments/feature_dca/run_dca.py` 依赖 `run_final.py` 的
`load_all_features()`，但该文件未随发布包提供，导致 DCA 分析无法运行。

**修复**：补全 `experiments/feature_dca/run_final.py`（综合特征加载：
radiomics 1316 + physics 11 + location one-hot）。恢复版经 Brier 数值验证，
与预计算结果偏差 ≤0.0015。

## 4. Figure 7 数据依赖缺失（已补生成器）

**问题**：`scripts/make_fig7_subgroup.py` 读取
`experiments/clean_ensemble/mediastinal_hu.npz`（纵隔病例病灶内 HU 体素），
但该数据文件未随发布包提供，导致 Figure 7(B) 无法绘制。

**修复**：新增 `scripts/make_mediastinal_hu.py` 生成器，从原始 CT + 金标准 mask
提取纵隔 clean 测试集病例（n=16）的病灶内 HU 体素并保存为 npz。

**验证**：生成结果与论文 Figure 7(B) 所用数据**逐位一致**（16 键，最大绝对偏差 0.0）；
复现论文纵隔 IT median 18 HU / MT median 13 HU 分布。
