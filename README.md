# CT Density Composition Differentiates Pediatric Extracranial Germ Cell Tumors

**Mechanism-first radiomics study with automated segmentation** — code and
reproducibility package accompanying the manuscript submitted to
*European Radiology*.

## Overview

We analyze the public **CT-PEGCT-Diag** dataset (642 pediatric extracranial germ
cell tumor (EGCT) non-enhanced CT scans with expert tumor masks) to:

1. Segment tumors with nnU-Net (Dice 0.85, 5-fold CV);
2. Classify the three main subtypes — mature teratoma (MT, n=415),
   immature teratoma (IT, n=106), and malignant germ cell tumors (MGCT, n=121)
   — with radiomics (PyRadiomics + SVM) and deep learning (2D/2.5D/3D ResNet);
3. Show that subtype discrimination is driven by a **CT-density continuum**
   (`solid_fraction`: MT 0.30 < IT 0.54 < MGCT 0.88, all pairwise p<1e-11)
   rather than by model capacity;
4. Quantify the cost of automated segmentation in the cascade
   (GT-mask ACC 0.860 → predicted-mask ACC 0.819, Δ −4.1 pt) and a
   low-confidence review workflow that restores accuracy at 17% review rate.

**Final model:** probability-averaged ensemble of five 2D ResNet18 networks and a
radiomics SVM → **ACC 0.902 / AUC 0.962** (test n=193), vs the official baseline
0.726/0.885.

## Data

The raw imaging data are **not** redistributed in this repository. Download
**CT-PEGCT-Diag** from ScienceDB (dataset paper: Zhou et al., *Scientific Data*,
2026) and place it at:

```
datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag/
├── MTs/          (415 cases: .nii.gz images + masks)
├── ITs/          (106 cases)
├── MGCTs/        (121 cases)
└── xlsx metadata
```

The data split used in the manuscript is provided in
`datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json`
(7:3 stratified split, seed 42; MT_411 excluded because its public mask equals
the raw image — an upload defect in the dataset).

## Environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# nnU-Net v2 is an official external package, not bundled here:
pip install nnunetv2
```

Hardware used: NVIDIA RTX 3090 24 GB.

## Pipeline (paper Methods)

Run from the repository root.

### 1. Preprocess DL inputs (Methods 2.3)
```bash
python experiments/dl_ablation/prepare_ablation.py
```
Creates per-case 2D (max-slice), 2.5D (±1 slice) and 3D (bbox-cropped) arrays
in `experiments/dl_ablation/data/` using the window 35/350 HU normalization.

### 2. Segmentation (Methods 2.2; nnU-Net official)
nnU-Net training uses the official nnUNetv2 package. Our trainer config
(250-epoch variant used in the paper) is in
`experiments/seg_trainer/trainer250.py`.
Evaluate segmentations and compare GT vs predicted ROI:
```bash
python experiments/seg_eval/eval_seg.py
python experiments/seg_eval/roi_compare.py
```

### 3. Radiomics features + SVM (Methods 2.3)
```bash
python experiments/radiomics_svm/extract_features_full.py   # PyRadiomics, 1316 features
python experiments/radiomics_svm/run_svm.py                 # SVM (RBF, C=10, gamma=1e-3)
```

### 4. Deep learning (Methods 2.3, clean protocol)
```bash
python experiments/clean_ensemble/train_clean.py             # 2D ResNet18, 5 seeds
python experiments/clean_ensemble/ensemble_clean.py          # probability ensemble + metrics
```

### 5. Cascade and error propagation (Methods 2.4)
```bash
python experiments/cascade_eval/eval_cached.py             # GT vs pred-mask ROI SVM
python experiments/cascade_eval/review_workflow.py         # low-confidence review
```

### 6. Mechanism analysis (Methods 2.5)
```bash
python experiments/physics_priors/run_physics.py             # physical priors (solid/fat/calc fraction)
python experiments/dl_ablation/permutation_test.py        # histogram vs texture permutation test
python experiments/feature_dca/feature_importance.py      # permutation importance
python experiments/feature_dca/run_dca.py                 # decision curve + calibration
```

### 7. Statistics (Methods 2.6)
```bash
python experiments/bootstrap/bootstrap_ci.py            # bootstrap 95% CI
python experiments/dl_ablation/run_mcnemar.py             # McNemar tests
python experiments/calibration/run_calibration.py        # balanced SVM + calibration data
```

### 8. Figures and tables
```bash
python experiments/dataset_overview/fig1_dataset_overview.py   # Figure 1
python experiments/feature_dca/make_figures.py            # Figure 2 (continuum)
python experiments/clean_ensemble/make_fig34_clean.py        # Figures 3-4 (ROC, confusion)
python experiments/cascade_eval/make_cascade_figs.py       # Figure 5 (error propagation)
python experiments/calibration/run_calibration.py              # calibration + DCA data (Figure 6)
python experiments/calibration/run_calibration.py              # calibration + DCA data (Figure 6)
python scripts/make_fig7_subgroup.py                 # Figure 7 (subgroups)
python experiments/table1/make_table1.py             # Table 1
```

### 9. Site hold-out (LODO) generalization — Methods §2.7, Supplementary Table S8
Leave-one-anatomical-site-out analysis across the six primary sites
(ovary / testis / retroperitoneum / sacrococcyx / mediastinum; a rare "other"
site with n=11 is test-only). Run from the repository root:

```bash
# Step 1 — domain labels + LODO splits (from the public *_patient.xlsx metadata)
python experiments/lodo/make_domain_splits.py          # → domain_labels.csv, lodo_splits.json

# Step 2 — PCA feature-drift diagnosis (→ domain_distances.json, pca_drift.png)
python experiments/lodo/pca_drift.py

# Step 3 — LODO SVM (radiomics, grid-searched per training domain)
python experiments/lodo/lodo_svm.py                    # → lodo_svm_results.json

# Step 4 — LODO 2D ResNet18 ensemble (5 seeds × 5 domains; ~80 min on RTX 3090)
python experiments/lodo/lodo_train_dl.py --domain ovary
python experiments/lodo/lodo_train_dl.py --domain testis
python experiments/lodo/lodo_train_dl.py --domain retroperitoneum
python experiments/lodo/lodo_train_dl.py --domain sacrococcyx
python experiments/lodo/lodo_train_dl.py --domain mediastinum

# Step 5 — fusion evaluation + drift-performance plot + report
python experiments/lodo/lodo_fusion.py                 # → lodo_final_report.md, lodo_summary.csv
```

Steps 1–3 need only the radiomics table (`experiments/radiomics_svm/features_full.csv`,
produced by step 3 of the main pipeline). Steps 4–5 additionally need the DL input
arrays from `experiments/dl_ablation/data/` (step 1 of the main pipeline).

## Precomputed results

`results/` contains the model outputs used for the manuscript figures and tables
(small files only; model weights are not included):

| Path | Content |
|---|---|
| `results/classification/clean_y.npy` | test-set labels (n=193) |
| `results/classification/clean_p_svm.npy` | SVM class probabilities |
| `results/classification/clean_p_2d_avg5.npy` | 5-seed 2D ResNet18 averaged probabilities |
| `results/classification/ensemble_clean.json` | ACC/AUC/recall of all variants |
| `results/cascade/results_cascade_full.json` | full 193-case cascade metrics |
| `results/segmentation/seg_quality.csv` | per-case Dice / HD95 |
| `results/analysis/feature_importance.csv` | top permutation-importance features |
| `experiments/lodo/lodo_final_report.md` | LODO site-hold-out results and protocol |
| `experiments/lodo/lodo_all_results.json` | per-domain SVM/DL/ensemble metrics (LODO) |
| `experiments/lodo/lodo_dl_summary.csv` | LODO-DL per-seed and ensemble summary |

## Files intentionally not included

Model weights and intermediate segmentations are **not** redistributed (large
files; train them yourself):

| Path referenced in scripts | What it is | How to obtain |
|---|---|---|
| `experiments/cnn_baseline/model_base_best.pt` | 3D CNN baseline weights (legacy model) | not released; `seg_eval/roi_compare.py` and `feature_dca/make_figures.py` reference it — retrain with `cnn_framework/train_cnn_variants.py` or ignore |
| `experiments/seg_eval/pred_masks/` | nnU-Net 5-fold predicted masks | run nnU-Net inference (trainer `seg_trainer/trainer250.py`) and place masks here |
| `experiments/dl_ablation/results/*.pt` | 2D/2.5D/3D ResNet ablation weights | run `dl_ablation/train_ablation.py` |
| `experiments/dl_ablation/data/*.npy` | prepared DL input arrays | run `dl_ablation/prepare_ablation.py` |
| `experiments/lodo/results/*.pt` | LODO 2D ResNet18 weights (5 domains × 5 seeds) | run `lodo/lodo_train_dl.py` (Step 4 above) |

The cascaded classification numbers in the paper were produced from the nnU-Net
predicted masks; without them, `cascade_eval/eval_cached.py` cannot be rerun
end-to-end, but all downstream metrics are available in `results/`.

## Reproducibility notes

- All reported numbers use the **clean protocol**: model selection (best epoch)
  uses only the validation split (68 cases held out from the 448 training cases);
  the test set (n=193) is never used for tuning.
- ACC is computed as argmax over `predict_proba` for all models.
- Each test case is predicted with the fold model that did **not** train on it.
- Random seeds: 42, 123, 2024 for DL; all splits derived from
  `split_7to3_seed42.json`.

## License

MIT (see LICENSE). The underlying dataset retains its own terms (ScienceDB).

## Contact

Corresponding author — see manuscript cover letter.
