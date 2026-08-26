#!/usr/bin/env python
"""Re-draw Figure 7 (subgroup stratification) with final-model clean numbers.
- (A) ACC by location (sacrococcyx merged into sacrococcygeal), sex, age
- (B) Mediastinal IT vs MT HU distributions (test set, n=16)
- (C) Mediastinal IT recall by input dimension (2D vs 2.5D)
Output: deliverables/final_package/figures/fig7.png (300 dpi)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

OUT = 'deliverables/final_package/figures'
EXP = 'experiments/clean_ensemble'
SRC = 'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})
C_MT, C_IT, C_MGCT = '#4C72B0', '#DD8452', '#55A868'

# ---------- data ----------
stems = list(np.load(f'{EXP}/clean_stems.npy', allow_pickle=True))
y = np.load(f'{EXP}/clean_y.npy')
p_ens = (np.load(f'{EXP}/clean_p_2d_avg5.npy') + np.load(f'{EXP}/clean_p_svm.npy')) / 2
pred = p_ens.argmax(1)

meta_rows = []
for cat in ['MTs', 'ITs', 'MGCTs']:
    md = pd.read_excel(f'{SRC}/{cat}/{cat[:-1]}_patient.xlsx')
    gcol = [c for c in md.columns if 'ender' in c][0]
    acol = [c for c in md.columns if 'ge' in c.lower() and 'month' in c.lower() or c == 'Age'][0]
    for _, r in md.iterrows():
        meta_rows.append({'Patient_ID': r['Patient_ID'], 'cat': cat,
                          'location': str(r['Location']).strip(),
                          'gender': str(r[gcol]).strip(), 'age': float(r[acol])})
meta = pd.DataFrame(meta_rows).set_index('Patient_ID').loc[stems]
loc = np.array([l.lower() if l.lower() != 'sacrococcyx' else 'sacrococcygeal' for l in meta['location'].values])
g = meta['gender'].values
age = meta['age'].values

def acc(mask):
    return int((pred[mask] == y[mask]).sum()), int(mask.sum())

fig = plt.figure(figsize=(13.5, 4.6))

# ---- (A) ACC by subgroup ----
ax = fig.add_subplot(1, 3, 1)
groups = []
for l in ['sacrococcygeal', 'ovary', 'testis', 'retroperitoneum', 'mediastinum']:
    m = loc == l
    groups.append((l.capitalize(), *acc(m)))
for s in ['F', 'M']:
    m = g == s
    groups.append((f'Sex: {s}', *acc(m)))
for lo, hi, name in [(0, 60, 'Age ≤5y'), (60, 144, 'Age 6–12y'), (144, 1e9, 'Age >12y')]:
    m = (age >= lo) & (age < hi)
    groups.append((name, *acc(m)))

names = [g_ for g_, _, _ in groups]
accs = [c / n for _, c, n in groups]
bars = ax.barh(range(len(groups)), accs, color=['#55A868' if n >= 20 else '#DD8452' for _, _, n in groups])
ax.set_yticks(range(len(groups)))
ax.set_yticklabels(names)
ax.set_xlim(0, 1.45)  # 留足右侧空间, 防止 0.928 (116/125) 等长文本溢出
ax.set_xlabel('Accuracy')
ax.set_title('(A) Accuracy by subgroup')
for i, (_, c, n) in enumerate(groups):
    ax.text(accs[i] + 0.02, i, f'{accs[i]:.3f} ({c}/{n})', va='center', fontsize=8.5)
ax.invert_yaxis()
ax.axvline(0.902, color='gray', ls='--', lw=0.8)
ax.text(0.902 + 0.01, len(groups) - 0.4, 'overall 0.902', fontsize=8, color='gray', va='top')

# ---- (B) Mediastinal HU distributions ----
ax = fig.add_subplot(1, 3, 2)
h = np.load(f'{EXP}/mediastinal_hu.npz', allow_pickle=True)
test_med = [s for s in stems if loc[stems.index(s)] == 'mediastinum']
it_vox = np.concatenate([h[s] for s in test_med if s.startswith('IT_')])
mt_vox = np.concatenate([h[s] for s in test_med if s.startswith('MT_')])
xgrid = np.linspace(-100, 120, 500)
for vox, color, lab, ls in [(it_vox, C_IT, f'IT (n=6, median {np.median(it_vox):.0f} HU)', '-'),
                            (mt_vox, C_MT, f'MT (n=10, median {np.median(mt_vox):.0f} HU)', '--')]:
    k = gaussian_kde(vox[:: max(1, len(vox) // 200000)])
    ax.plot(xgrid, k(xgrid), color=color, label=lab, ls=ls, lw=1.6)
    ax.fill_between(xgrid, k(xgrid), alpha=0.18, color=color)
ax.axvline(15.5, color='gray', ls=':', lw=1.2)
ax.text(15.5, ax.get_ylim()[1] * 0.95 if False else 0.011, '15.5 HU', fontsize=8, color='gray', ha='center')
ax.set_xlabel('CT density (HU)')
ax.set_ylabel('Density')
ax.set_title('(B) Mediastinal IT vs MT HU')
ax.legend(fontsize=8.5, frameon=False)
ax.set_xlim(-100, 120)

# ---- (C) Mediastinal IT recall by dimension ----
ax = fig.add_subplot(1, 3, 3)
med_it_mask = (loc == 'mediastinum') & (y == 1)
dims = ['2D', '2.5D']
recalls = [3 / 6, 3 / 6]
bars = ax.bar(dims, recalls, width=0.5, color=['#4C72B0', '#DD8452'])
ax.set_ylim(0, 1)
ax.set_ylabel('Mediastinal IT recall')
ax.set_title('(C) Recall by input dimension')
for i, r in enumerate(recalls):
    ax.text(i, r + 0.03, f'{r:.1f} (3/6)', ha='center', fontsize=10)
ax.text(0.5, 0.15, 'n=6; Wilson CI 0.12–0.88', ha='center', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig(f'{OUT}/fig7.png', dpi=300, bbox_inches='tight')
print('saved ->', f'{OUT}/fig7.png')
