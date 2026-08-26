#!/usr/bin/env python
"""cascade_eval 并行缓存版: 测试集预测 Mask → 组学+物理特征 (多进程, 每例缓存, 断点续传)
用法: python extract_cached.py [--workers 8]
输出: experiments/cascade_eval/cache_gt/*.npz, cache_pred/*.npz
"""
import os, sys, json, argparse, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from scipy import stats
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = f'{BASE}/datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag'
OUT = f'{BASE}/experiments/cascade_eval'
CACHE_GT = f'{OUT}/cache_gt'
CACHE_PRED = f'{OUT}/cache_pred'
os.makedirs(CACHE_GT, exist_ok=True); os.makedirs(CACHE_PRED, exist_ok=True)

PARAMS = f'{BASE}/experiments/radiomics_svm/radiomics_params_full.json'
FEAT_COLS = None  # 由第一个完成的病例确定

def window_and_normalize(image_sitk, level=35, width=350):
    arr = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
    lower = level - width / 2.0; upper = level + width / 2.0
    arr = np.clip(arr, lower, upper)
    arr = (arr - lower) / (upper - lower) * 255.0
    out = sitk.GetImageFromArray(arr); out.CopyInformation(image_sitk)
    return out

def extract_one(args):
    stem, img_path, mask_path = args
    try:
        from radiomics import featureextractor
        extractor = featureextractor.RadiomicsFeatureExtractor(PARAMS)
        img = sitk.ReadImage(img_path); mask = sitk.ReadImage(mask_path)
        feat = extractor.execute(window_and_normalize(img), mask)
        radiomics = {k: float(v) for k, v in feat.items() if not k.startswith('diagnostics')}
    except Exception as e:
        return stem, None, str(e)
    try:
        img2 = nib.load(img_path).get_fdata(); m2 = nib.load(mask_path).get_fdata()
        hu = img2[m2 > 0]
        phys = None
        if len(hu) > 0:
            phys = {'fat_fraction': float((hu < -30).mean()), 'calc_fraction': float((hu > 100).mean()),
                    'solid_fraction': float((hu > 20).mean()), 'hu_mean': float(hu.mean()),
                    'hu_median': float(np.median(hu)), 'hu_std': float(hu.std()),
                    'hu_p5': float(np.percentile(hu, 5)), 'hu_p95': float(np.percentile(hu, 95)),
                    'hu_skew': float(stats.skew(hu)), 'hu_kurt': float(stats.kurtosis(hu)),
                    'n_voxels': int(len(hu))}
    except Exception as e:
        return stem, None, f'phys: {e}'
    return stem, (radiomics, phys), None

def process(stems, img_of, mask_of, cache_dir, workers):
    done = set(f[:-4] for f in os.listdir(cache_dir) if f.endswith('.npz'))
    todo = [s for s in stems if s not in done]
    print(f'cache {os.path.basename(cache_dir)}: total {len(stems)}, done {len(done)}, todo {len(todo)}', flush=True)
    if not todo:
        return
    tasks = [(s, img_of[s], mask_of[s]) for s in todo]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(extract_one, t): t[0] for t in tasks}
        n = 0
        for fut in as_completed(futs):
            stem, res, err = fut.result()
            if res is None:
                print(f'  ERROR {stem}: {err}', flush=True)
                continue
            radiomics, phys = res
            np.savez(f'{cache_dir}/{stem}.npz', radiomics=np.array(list(radiomics.items()), dtype=object),
                     phys=np.array(list(phys.items()), dtype=object) if phys else np.array([]))
            n += 1
            if n % 20 == 0:
                print(f'  progress {n}/{len(todo)}', flush=True)

def load_cache(cache_dir, stems, feat_cols):
    rows = []
    for s in stems:
        f = f'{cache_dir}/{s}.npz'
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        d = dict(z['radiomics'])
        p = dict(z['phys']) if len(z['phys']) else {}
        row = {c: d.get(c, np.nan) for c in feat_cols}
        row.update({k: v for k, v in p.items()})
        rows.append((s, row))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    split = json.load(open(f'{BASE}/datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
    test_stems = set(split['MT']['test'] + split['IT']['test'] + split['MGCT']['test']) - {'MT_411'}
    cat_of = {}
    for cat in ['MTs', 'ITs', 'MGCTs']:
        for f in os.listdir(f'{SRC}/{cat}/Masks'):
            cat_of[f.replace('.nii.gz', '')] = cat
    pred_map = json.load(open(f'{OUT}/pred_map.json'))
    stems = sorted(s for s in test_stems if s in cat_of and s in pred_map)
    img_of = {s: f'{SRC}/{cat_of[s]}/Images/{s}.nii.gz' for s in stems}
    gt_mask = {s: f'{SRC}/{cat_of[s]}/Masks/{s}.nii.gz' for s in stems}
    pred_mask = {s: pred_map[s] for s in stems}
    print(f'test cases with pred: {len(stems)}', flush=True)

    process(stems, img_of, gt_mask, CACHE_GT, args.workers)
    process(stems, img_of, pred_mask, CACHE_PRED, args.workers)
    print('done', flush=True)

if __name__ == '__main__':
    main()
