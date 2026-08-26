#!/usr/bin/env python
"""dataset_overview-Fig1: 论文 Figure 1 — 数据集概览图 (独立文件, 避免与并行会话 dataset_overview 集成级联脚本冲突)
A: 类别构成 + 位置分布
B: 每类按 solid_fraction 低/中/高选代表病例 (最大 mask 层面 + 轮廓)
C: solid_fraction 连续谱箱线图
输出: experiments/dataset_overview/fig1_dataset_overview.png

2026-08-15 布局修复:
- A 面板两个子图各自加标题 (By subtype / By location), 拉大间距, ylabel 统一为 "Number of patients"
- B 面板占更宽列 + 更高行, 图像标签移到图像下方 (xlabel), 子网格顶部为标题留出空间
- 内置 overlap 检查 (text-text / text-axes bbox), 打印告警
"""
import os, json, warnings
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

SRC = 'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'
OUT = 'experiments/dataset_overview'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                     'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

CLS_INFO = {'MTs': ('MT', '#4C72B0'), 'ITs': ('IT', '#DD8452'), 'MGCTs': ('MGCT', '#55A868')}
CATS = ['MTs', 'ITs', 'MGCTs']

def load_meta():
    rows = []
    for cat in CATS:
        md = pd.read_excel(f'{SRC}/{cat}/{cat[:-1]}_patient.xlsx')
        gcol = [c for c in md.columns if 'ender' in c][0]
        acol = [c for c in md.columns if 'Age' in c][0]
        lcol = [c for c in md.columns if 'ocation' in c][0]
        for _, r in md.iterrows():
            rows.append({'stem': r['Patient_ID'], 'cat': cat,
                         'gender': str(r[gcol]).upper(), 'age_months': float(r[acol]),
                         'location': str(r[lcol]).lower().strip()})
    return pd.DataFrame(rows)

def solid_frac(stem, cat):
    img = nib.load(f'{SRC}/{cat}/Images/{stem}.nii.gz').get_fdata()
    m = nib.load(f'{SRC}/{cat}/Masks/{stem}.nii.gz').get_fdata()
    hu = img[m > 0]
    return (hu > 20).mean() if len(hu) else np.nan

def max_slice(stem, cat):
    # NIfTI 形状 (x=512, y=512, z=slices) —— 层面轴是第 3 维 (axis=2)
    img = nib.load(f'{SRC}/{cat}/Images/{stem}.nii.gz').get_fdata()
    m = nib.load(f'{SRC}/{cat}/Masks/{stem}.nii.gz').get_fdata()
    areas = (m > 0).sum(axis=(0, 1))
    z = int(areas.argmax())
    if areas[z] == 0:
        return None, None
    sl = img[:, :, z]; mask_sl = m[:, :, z] > 0
    lower, upper = 35 - 350 / 2, 35 + 350 / 2
    sl = np.clip(sl, lower, upper)
    return (sl - lower) / (upper - lower), mask_sl

def check_overlaps(fig, label='fig1'):
    """检查 text-text 重叠和 text 与 'foreign' axes bbox 重叠。
    覆盖 inset axes (matplotlib>=3.x inset_axes 不注册到 fig.axes, 需用 findobj)。
    """
    from matplotlib.axes import Axes
    fig.canvas.draw()
    all_axes = fig.findobj(Axes)
    texts = []
    for ax in all_axes:
        titles = [ax.title]
        for attr in ('_left_title', '_right_title'):
            t = getattr(ax, attr, None)
            if t is not None:
                titles.append(t)
        tick_labs = []
        if ax.axison:
            tick_labs = list(ax.get_xticklabels()) + list(ax.get_yticklabels())
            cands = (list(ax.texts) + titles + [ax.xaxis.label, ax.yaxis.label] + tick_labs)
        else:
            # axis-off 容器: 刻度/轴标签不绘制, 只检查其 texts + 面板标题
            cands = (list(ax.texts) + titles)
        for t in cands:
            if t is None or not t.get_text() or not t.get_visible():
                continue
            bb = t.get_window_extent()
            if bb.width > 0 and bb.height > 0:
                texts.append((ax, t, bb))
    issues = []
    # text vs text
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if texts[i][2].overlaps(texts[j][2]):
                issues.append(f'TEXT×TEXT: "{texts[i][1].get_text()[:22]}" (ax#{all_axes.index(texts[i][0])}) '
                              f'<-> "{texts[j][1].get_text()[:22]}" (ax#{all_axes.index(texts[j][0])})')
    # text vs foreign axes bbox
    for ax, t, bb in texts:
        ax_bb = ax.get_window_extent()
        for other in all_axes:
            if other is ax:
                continue
            obb = other.get_window_extent()
            if obb.width <= 0 or obb.height <= 0:
                continue
            # 子轴文本 vs 父容器 bbox 属正常包含, 跳过
            if (obb.x0 <= ax_bb.x0 and obb.y0 <= ax_bb.y0 and
                    obb.x1 >= ax_bb.x1 and obb.y1 >= ax_bb.y1):
                continue
            if bb.overlaps(obb):
                issues.append(f'TEXT×AXES: "{t.get_text()[:22]}" (ax#{all_axes.index(ax)}) '
                              f'overlaps ax#{all_axes.index(other)}')
    if issues:
        print(f'[{label}] OVERLAP WARNINGS ({len(issues)}):')
        for s in issues:
            print('  ', s)
    else:
        print(f'[{label}] no overlaps detected')
    # 布局量化指标 (仅 fig1)
    if label == 'fig1':
        dpi = fig.dpi
        im_axes = [ax for ax in all_axes if len(ax.images) > 0]
        if im_axes:
            bb = im_axes[0].get_window_extent()
            print(f'[fig1] first image cell size: {bb.width / dpi:.2f} x {bb.height / dpi:.2f} inch')
        # B 标题下边缘 vs 第一行图像上边缘 间距
        for ax in all_axes:
            bt = getattr(ax, '_left_title', None)
            if bt is not None and bt.get_text().startswith('B '):
                tb = bt.get_window_extent()
                if im_axes:
                    top = max(a.get_window_extent().y1 for a in im_axes)
                    print(f'[fig1] B-title bottom y={tb.y0/dpi:.2f}in, row0 images top y={top/dpi:.2f}in, '
                          f'gap={ (top - tb.y0)/dpi:.2f}in')
        # A 面板子图标题是否在容器内且可见
        for ax in all_axes:
            at = getattr(ax, '_left_title', None)
            if at is not None and at.get_text().startswith('A '):
                ab = ax.get_window_extent()
                for ins in all_axes:
                    if ins is ax:
                        continue
                    it = ins.get_title()
                    if it.startswith('By'):
                        tbb = ins.title.get_window_extent()
                        inside = (ab.x0 <= tbb.x0 and tbb.x1 <= ab.x1
                                  and ab.y0 <= tbb.y0 and tbb.y1 <= ab.y1)
                        print(f'[fig1] A-panel inset title "{it}" inside container: {inside}')

def main():
    meta = load_meta()
    CACHE = f'{OUT}/sdf_cache.csv'
    if os.path.exists(CACHE):
        sdf = pd.read_csv(CACHE)
    else:
        sf = []
        for cat in CATS:
            for stem in meta[meta['cat'] == cat]['stem']:
                v = solid_frac(stem, cat)
                if not np.isnan(v):
                    sf.append({'stem': stem, 'cat': cat, 'solid_fraction': v})
        sdf = pd.DataFrame(sf)
        sdf.to_csv(CACHE, index=False)
        print(f'cached {len(sdf)} solid fractions -> {CACHE}')

    # ===== 布局: B 面板独占右列整高 (图像最大), A 左上 / C 左下 =====
    fig = plt.figure(figsize=(16.5, 11.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], width_ratios=[0.78, 1.22],
                          hspace=0.45, wspace=0.35)

    # ===== Panel A: 类别构成 + 位置分布 (两个独立子图, 各自带标题) =====
    axA = fig.add_subplot(gs[0, 0])
    axA.axis('off')
    axA.set_title('A  Dataset composition', loc='left', fontweight='bold', fontsize=12, pad=12)

    # --- A1: 各类别例数柱状图 ---
    axA1 = axA.inset_axes([0.02, 0.18, 0.38, 0.74])
    axA1.set_title('By subtype', fontsize=10, pad=6)
    counts = [len(meta[meta['cat'] == c]) for c in CATS]
    labels = ['MT', 'IT', 'MGCT']
    colors = [CLS_INFO[c][1] for c in CATS]
    xpos = np.arange(len(labels))
    axA1.bar(xpos, counts, color=colors, edgecolor='black', linewidth=0.6)
    for x, n in zip(xpos, counts):
        axA1.text(x, n + 8, str(n), ha='center', fontsize=10, fontweight='bold')
    axA1.set_xticks(xpos); axA1.set_xticklabels(labels)
    axA1.set_ylabel('Number of patients')
    axA1.set_ylim(0, 500)
    axA1.set_yticks(np.arange(0, 501, 100))
    axA1.spines[['top', 'right']].set_visible(False)

    # sacrococcyx 并入 sacrococcygeal (与稿件 v1.6 标签一致, 共 97 例)
    loc_order = ['ovary', 'testis', 'retroperitoneum', 'sacrococcygeal', 'mediastinum', 'others']
    loc_map = {'ovary': 'ovary', 'testis': 'testis', 'retroperitoneum': 'retroperitoneum',
               'sacrococcygeal': 'sacrococcygeal', 'sacrococcyx': 'sacrococcygeal',
               'mediastinum': 'mediastinum'}
    meta['loc_grp'] = meta['location'].map(loc_map).fillna('others')
    counts_loc = np.array([[len(meta[(meta['cat'] == c) & (meta['loc_grp'] == loc)]) for c in CATS]
                           for loc in loc_order])

    # --- A2: 各位置堆叠柱状图 ---
    axA2 = axA.inset_axes([0.55, 0.18, 0.43, 0.74])
    axA2.set_title('By location', fontsize=10, pad=6)
    xloc = np.arange(len(loc_order))
    bottom = np.zeros(len(loc_order))
    for i, cat in enumerate(CATS):
        axA2.bar(xloc, counts_loc[:, i], bottom=bottom, color=colors[i], alpha=0.55, width=0.6)
        bottom += counts_loc[:, i]
    axA2.set_xticks(xloc); axA2.set_xticklabels(loc_order)
    ymax = counts_loc.sum(axis=1).max()
    axA2.set_ylim(0, ymax * 1.15)
    axA2.set_ylabel('Number of patients')
    axA2.tick_params(axis='x', rotation=90, labelsize=7.5)
    plt.setp(axA2.get_xticklabels(), ha='center', va='top', rotation_mode='anchor')
    axA2.spines[['top']].set_visible(False)

    # ===== Panel B: 代表病例 montage (3x3, 独占右列整高, 标签在图像下方) =====
    axB = fig.add_subplot(gs[:, 1])
    axB.axis('off')
    axB.set_title('B  Representative cases by solid fraction (low / mid / high)',
                  loc='left', fontweight='bold', fontsize=12, pad=22)
    terc = []
    for cat in CATS:
        sub = sdf[sdf['cat'] == cat].sort_values('solid_fraction')
        idx = [int(len(sub) * 0.1), int(len(sub) * 0.5), int(len(sub) * 0.9)]
        terc.append([(sub.iloc[i]['stem'], sub.iloc[i]['solid_fraction']) for i in idx])
    gsB = gs[:, 1].subgridspec(3, 3, hspace=0.35, wspace=0.10)
    for r, cat in enumerate(CATS):
        for c, (stem, sfv) in enumerate(terc[r]):
            sl, msl = max_slice(stem, cat)
            ax_im = fig.add_subplot(gsB[r, c])
            ax_im.imshow(sl, cmap='gray', vmin=0, vmax=1, aspect='equal')
            if msl is not None:
                ax_im.contour(msl, levels=[0.5], colors=CLS_INFO[cat][1], linewidths=1.4)
            ax_im.set_xticks([]); ax_im.set_yticks([])
            for s in ax_im.spines.values():
                s.set_linewidth(0.8)
            ax_im.set_xlabel(f'{stem}\nSF={sfv:.2f}', fontsize=7, labelpad=2)
            if c == 0:
                ax_im.text(-0.30, 0.5, CLS_INFO[cat][0], transform=ax_im.transAxes,
                           fontsize=14, fontweight='bold', color=CLS_INFO[cat][1],
                           ha='center', va='center', rotation=90)

    # ===== Panel C: 连续谱箱线图 =====
    axC = fig.add_subplot(gs[1, 0])
    data = [sdf[sdf['cat'] == c]['solid_fraction'].values for c in CATS]
    bp = axC.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                     medianprops=dict(color='black', linewidth=1.6))
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    for i, cat in enumerate(CATS):
        axC.scatter(np.random.default_rng(i).normal(i + 1, 0.04, len(data[i])), data[i],
                    s=8, color=colors[i], alpha=0.35, zorder=3)
    axC.axhline(0.541, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    axC.text(3.35, 0.55, 'IT mean', fontsize=7, color='gray')
    axC.set_ylabel('solid_fraction (HU>20)')
    axC.set_ylim(-0.02, 1.05)
    axC.set_title('C  Density continuum: solid fraction by subtype', loc='left', fontweight='bold', fontsize=12)
    axC.spines[['top', 'right']].set_visible(False)

    check_overlaps(fig, 'fig1')

    fig.savefig(f'{OUT}/fig1_dataset_overview.png')
    plt.close(fig)
    summ = {}
    for cat, key in [('MTs', 'MT'), ('ITs', 'IT'), ('MGCTs', 'MGCT')]:
        v = sdf[sdf['cat'] == cat]['solid_fraction']
        summ[key] = {'n': int(len(v)), 'mean': round(float(v.mean()), 3), 'median': round(float(v.median()), 3)}
    json.dump(summ, open(f'{OUT}/fig1_results.json', 'w'), indent=2)
    print('saved ->', f'{OUT}/fig1_dataset_overview.png')
    print(json.dumps(summ, indent=2))
    for r, cat in enumerate(CATS):
        print(f'  {cat}:', [f'{t[0]}({t[1]:.2f})' for t in terc[r]])

if __name__ == '__main__':
    main()
