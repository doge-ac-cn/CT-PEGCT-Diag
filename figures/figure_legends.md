# Figure & Table Legends (ER format)

## Figure 1. Dataset overview.
(A) Composition of the CT-PEGCT-Diag cohort (n=642; MT 415, IT 106, MGCT 121) with stacked bars showing anatomical location distribution. (B) Representative axial CT slices (window 35/350 HU) with tumor contours for each subtype at low/mid/high solid fraction (SF). (C) Box plots of solid fraction (fraction of voxels >20 HU within the tumor mask) by subtype, with individual patient values overlaid. Pairwise Mann-Whitney U tests: all p<1e-11.

## Figure 2. Density continuum and information-theoretic characterization.
(A) Kernel density estimates of solid fraction by subtype. (B) Single-variable ROC of solid fraction: overall ovr AUC 0.951; IT 1-vs-rest AUC 0.57–0.61 (MT 0.876, MGCT 0.967). (C) Classification error of the optimal 1D threshold classifier by subtype (IT recall 0.062). SF, solid fraction; ovr, one-vs-rest.

## Figure 3. Model performance comparison.
ROC curves (one-vs-rest) on the held-out test set (n=193) for the radiomics×DL ensemble (ACC 0.902/AUC 0.962), radiomics SVM (0.886/0.950), full-feature balanced SVM (0.881/0.952), 2D ResNet18 (five-network mean, 0.865±0.015/0.923±0.010), and official baseline (0.726/0.885).

## Figure 4. Confusion matrices of the final ensemble.
Rows: true subtype; columns: predicted subtype. MT 121/125 (recall 0.968), IT 18/32 (0.562), MGCT 35/36 (0.972). Errors concentrate at IT boundaries (9 IT→MT, 5 IT→MGCT, 4 MT→IT, 1 MGCT→MT).

## Figure 5. Segmentation error propagation.
(A) Dice distribution by subtype (5-fold cross-validation; IT 0.917, MT 0.847, MGCT 0.802). (B) Paired ACC comparison: GT-mask ROI vs nnU-Net-predicted-mask ROI (SVM: 0.900→0.825 on 40 cases; full 193-case: 0.860→0.819). (C) Solid-fraction shift in misclassified vs correctly classified cases (median delta −0.022 vs +0.008, p=0.041, Wilcoxon). (D) Low-confidence review workflow: ACC vs review rate (tau=0.6, 17% review → ACC 0.900 = GT level).

## Figure 6. Calibration and decision curves.
(A) Reliability diagrams (ECE: ensemble 0.055, SVM 0.062). (B) Decision curve analysis: net benefit vs threshold probability for the ensemble, SVM, and DL; the ensemble shows the highest net benefit at thresholds ≥0.15.

## Figure 7. Subgroup stratification and the mediastinal blind spot.
(A) ACC by location (sacrococcygeal 0.970 [32/33], ovary 0.947 [54/57], testis 0.907 [39/43], retroperitoneum 0.829 [34/41], mediastinum 0.812 [13/16]; sacrococcyx and sacrococcygeal records merged as one anatomical site), sex (F 0.928 [116/125], M 0.853 [58/68]), and age (≤5y 0.882 [112/127], 6–12y 0.926 [50/54], >12y 1.000 [12/12]). (B) Mediastinal-IT HU distributions: IT median 18 HU vs MT 13 HU, 55% overlap at threshold 15.5 HU. (C) Dimension comparison on mediastinal IT recall: 2D 0.5 (3/6), 2.5D 0.5 (3/6); no reliable dimensional gain.

## Table 1. Dataset characteristics.
Subtype counts, sex distribution, age, anatomical location, scanner, and imaging parameters of the CT-PEGCT-Diag cohort (n=642; 641 analyzable after excluding MT_411 with a corrupted mask).

## Table 2. Classification performance comparison (test set, n=193).
ACC, AUC (ovr), per-class recall, and calibration (ECE) for all models under the unified pipeline. Official baseline from the dataset paper. McNemar: ensemble vs SVM p=0.63 (n.s.); ensemble vs five-network DL average p=0.22 (n.s.).

| Method | ACC | AUC (ovr) | IT recall | ECE |
|---|---|---|---|---|
| Radiomics×DL ensemble (final) | 0.902 | 0.962 | 0.562 | 0.055 |
| Radiomics SVM (official replication) | 0.886 | 0.950 | 0.594 | 0.062 |
| Full-feature balanced SVM | 0.881 | 0.952 | 0.625 | — |
| 2D ResNet18 (five-network mean) | 0.865 ± 0.015 | 0.923 ± 0.010 | — | — |
| Location-adaptive ensemble (2.5D mediastinum) | 0.902 | 0.961 | 0.562 | — |
| Official baseline | 0.726 | 0.885 | — | — |

## Table 3. Subgroup performance of the final model.
ACC by location (sacrococcygeal 0.970 [32/33], ovary 0.947 [54/57], testis 0.907 [39/43], retroperitoneum 0.829 [34/41], mediastinum 0.812 [13/16]), sex (F 0.928, M 0.853), and age (≤5y 0.882, 6–12y 0.926, >12y 1.000); mediastinal-IT recall highlighted (0.5, 3/6). Error types: 9/19 IT→MT, 5 IT→MGCT, 4 MT→IT, 1 MGCT→MT.

## Table 4. Mechanism evidence chain.
Summary of evidence for the density-continuum mechanism: solid-fraction ordering, IT 1D blind zone, multi-dimensional first-order recovery, causal permutation test, SHAP signal decomposition, and clinical safety profile. Cross-referenced to Figures 2, 3 and Supplementary Tables S2–S3.
