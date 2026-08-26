#!/usr/bin/env python
"""bootstrap: 权威级联评估 (管线 = authoritative cascade pipeline 完全复刻 + 保存逐例概率)
用途: 统一论文级联数字 (113/193 例均用此管线), 输出供复核工作流与三规模汇总表
输入: experiments/cascade_eval/cache_gt|cache_pred (extract_cached.py 产出)
输出: experiments/bootstrap/cascade_eval_<n>.json + cascade_prob_<n>.npz (stem/y/prob_gt/prob_pr/pred_gt/pred_pr)
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.isdir(BASE):
    raise RuntimeError('BASE path not found; run from a checkout of the repository')
os.chdir(BASE)
OUT = f'{BASE}/experiments/bootstrap'
os.makedirs(OUT, exist_ok=True)
CACHE = f'{BASE}/experiments/cascade_eval'
sys.path.insert(0, f'{BASE}/experiments/physics_priors')

PHYS_COLS = ['fat_fraction','calc_fraction','solid_fraction','hu_mean','hu_median',
             'hu_std','hu_p5','hu_p95','hu_skew','hu_kurt','n_voxels']

def load_meta():
    rows = []
    for cat in ['MTs','ITs','MGCTs']:
        md = pd.read_excel(f'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag/{cat}/{cat[:-1]}_patient.xlsx')
        g=[c for c in md.columns if 'ender' in c][0]; a=[c for c in md.columns if 'Age' in c][0]
        l=[c for c in md.columns if 'ocation' in c][0]
        for _,r in md.iterrows():
            rows.append({'Patient_ID':r['Patient_ID'],'gender':str(r[g]).upper(),
                         'age':float(r[a]),'location':str(r[l]).lower().strip()})
    meta=pd.DataFrame(rows); meta['is_female']=(meta['gender']=='F').astype(int)
    meta['age_log']=np.log1p(meta['age'])
    dd=pd.get_dummies(meta['location'],prefix='loc')
    return pd.concat([meta,dd],axis=1)

def main():
    split=json.load(open('datasets/R23_CT-PEGCT-Diag/split_7to3_seed42.json'))
    train_stems=set(split['MT']['train']+split['IT']['train']+split['MGCT']['train'])-{'MT_411'}
    test_stems=set(split['MT']['test']+split['IT']['test']+split['MGCT']['test'])-{'MT_411'}

    # 训练 (完全复刻 authoritative cascade pipeline)
    df=pd.read_csv('experiments/radiomics_svm/features_full.csv')
    df=df[df['Patient_ID']!='MT_411'].reset_index(drop=True)
    from run_physics import extract_physics as extract_physics_gt
    phys=[]
    for stem in df['Patient_ID']:
        p=extract_physics_gt(stem)
        if p: phys.append({'Patient_ID':stem,**p})
    df=df.merge(pd.DataFrame(phys),on='Patient_ID',how='left')
    meta=load_meta()
    meta_cols=['is_female','age_log']+[c for c in meta.columns if c.startswith('loc_')]
    df=df.merge(meta[['Patient_ID']+meta_cols],on='Patient_ID',how='left')
    feat_cols=[c for c in df.columns if c not in ('Patient_ID','class')+tuple(PHYS_COLS)+tuple(meta_cols)]
    all_cols=feat_cols+PHYS_COLS+meta_cols
    tr=df[df['Patient_ID'].isin(train_stems)]
    vt=VarianceThreshold(threshold=1e-6); Xtr_v=vt.fit_transform(tr[all_cols].values)
    keep=list(range(Xtr_v.shape[1]))
    if Xtr_v.shape[1]>=2:
        try:
            corr=np.corrcoef(Xtr_v.T)
            keep=[i for i in range(Xtr_v.shape[1]) if all(abs(corr[i,j])<0.95 for j in range(i))]
        except Exception: pass
    pipe=Pipeline([('scaler',StandardScaler()),
                   ('svm',SVC(kernel='rbf',C=10,gamma=0.001,probability=True,random_state=42))])
    pipe.fit(Xtr_v[:,keep], tr['class'].values)
    print(f'SVM trained: {len(tr)} cases, {len(keep)} feats', flush=True)

    stems=sorted(s for s in test_stems if os.path.exists(f'{CACHE}/cache_gt/{s}.npz') and os.path.exists(f'{CACHE}/cache_pred/{s}.npz'))
    print(f'cached test cases: {len(stems)}', flush=True)
    cat_of={}
    for cat in ['MTs','ITs','MGCTs']:
        for f in os.listdir(f'datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag/{cat}/Masks'):
            cat_of[f.replace(".nii.gz","")]=cat
    rows=[]
    for s in stems:
        g=np.load(f'{CACHE}/cache_gt/{s}.npz',allow_pickle=True)
        p=np.load(f'{CACHE}/cache_pred/{s}.npz',allow_pickle=True)
        m=meta[meta['Patient_ID']==s]
        mrow={c:float(m[c].iloc[0]) for c in meta_cols if c in m.columns}
        gfeat=dict(zip(g['radiomics'][:,0],g['radiomics'][:,1].astype(float)))
        pfeat=dict(zip(p['radiomics'][:,0],p['radiomics'][:,1].astype(float)))
        gphys=dict(zip(g['phys'][:,0],g['phys'][:,1].astype(float)))
        pphys=dict(zip(p['phys'][:,0],p['phys'][:,1].astype(float)))
        for tag,f,ph in [('gt',gfeat,gphys),('pred',pfeat,pphys)]:
            row={'stem':s,'cat':cat_of[s],'true':{'MTs':0,'ITs':1,'MGCTs':2}[cat_of[s]],'mask':tag}
            for c in feat_cols: row[c]=f.get(c,np.nan)
            for c in PHYS_COLS: row[c]=ph.get(c,np.nan)
            row.update(mrow); rows.append(row)
    ev=pd.DataFrame(rows)
    X=vt.transform(ev[all_cols].values)[:,keep]
    ev['prob']=list(pipe.predict_proba(X)); ev['pred']=pipe.predict(X)

    print(f'\n=== 权威级联 (n={len(stems)}) ===', flush=True)
    for tag in ['gt','pred']:
        sub=ev[ev['mask']==tag]
        acc=accuracy_score(sub['true'],sub['pred'])
        auc=roc_auc_score(sub['true'],np.vstack(sub['prob']),multi_class='ovr')
        rec=recall_score(sub['true'],sub['pred'],average=None)
        print(f'{tag:4s} ROI: n={len(sub)} ACC={acc:.4f} AUC={auc:.4f} recall[MT,IT,MGCT]={np.round(rec,3)}', flush=True)
    pivot=ev.pivot(index='stem',columns='mask',values='pred')
    pivot['true']=ev[ev['mask']=='gt'].set_index('stem')['true']
    pivot['agree']=(pivot['gt']==pivot['pred'])
    print(f'GT vs Pred 一致率: {pivot["agree"].mean():.4f} ({len(pivot)} 例)', flush=True)
    diff=pivot[~pivot['agree']]
    if len(diff):
        print('不一致病例:')
        print(diff.to_string())
    res={'n':int(len(stems)),
         'gt_acc':float(accuracy_score(ev[ev["mask"]=='gt']['true'],ev[ev["mask"]=='gt']['pred'])),
         'pred_acc':float(accuracy_score(ev[ev["mask"]=='pred']['true'],ev[ev["mask"]=='pred']['pred'])),
         'gt_auc':float(roc_auc_score(ev[ev["mask"]=='gt']['true'],np.vstack(ev[ev["mask"]=='gt']['prob']),multi_class='ovr')),
         'pred_auc':float(roc_auc_score(ev[ev["mask"]=='pred']['true'],np.vstack(ev[ev["mask"]=='pred']['prob']),multi_class='ovr')),
         'agreement':float(pivot['agree'].mean()),'n_disagree':int(len(diff))}
    json.dump(res,open(f'{OUT}/cascade_eval_{len(stems)}.json','w'),indent=2)

    # 保存逐例概率 (供复核工作流)
    gt_ev=ev[ev['mask']=='gt'].set_index('stem').loc[stems]
    pr_ev=ev[ev['mask']=='pred'].set_index('stem').loc[stems]
    np.savez(f'{OUT}/cascade_prob_{len(stems)}.npz',
             stems=np.array(stems), y_te=gt_ev['true'].values.astype(int),
             prob_gt=np.vstack(gt_ev['prob']), prob_pr=np.vstack(pr_ev['prob']),
             pred_gt=gt_ev['pred'].values.astype(int), pred_pr=pr_ev['pred'].values.astype(int))
    ev.to_csv(f'{OUT}/cascade_eval_{len(stems)}.csv', index=False)
    print(f'saved -> {OUT}/cascade_eval_{len(stems)}.json/.csv, cascade_prob_{len(stems)}.npz')

if __name__=='__main__':
    main()
