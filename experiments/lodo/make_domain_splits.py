#!/usr/bin/env python
"""make_domain_splits.py — 生成 LODO（留一解剖部位域）划分

对应论文 Methods §2.7（site hold-out 泛化分析）：
- 解剖部位标签来自 CT-PEGCT-Diag 公开元数据（*_patient.xlsx 的 Location 列）；
- 清洗规则：strip + lower；sacrococcyx / sacrococcygeal 合并为 sacrococcyx；
- 罕见部位（abdomen / vagina / head and neck / maxillary sinus，合计 11 例）
  合并为 "other"，仅作为测试域，不参与任何 LODO 训练；
- 剔除 MT_411（公开 mask 与原始图像相同，上传缺陷，与主实验一致）。

输出（写入本脚本所在目录）：
  domain_labels.csv   Patient_ID,class,domain,domain_raw （641 行）
  lodo_splits.json    每个大域的 train/test/rare_test 病例清单

依赖：
  - 原始数据已解压至 <repo>/datasets/R23_CT-PEGCT-Diag/CT-PEGCT-Diag/
    （ScienceDB 公开下载，见 README）
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO, 'datasets', 'R23_CT-PEGCT-Diag', 'CT-PEGCT-Diag')
SPLIT_JSON = os.path.join(REPO, 'datasets', 'R23_CT-PEGCT-Diag', 'split_7to3_seed42.json')

CATS = {'MTs': 'MT', 'ITs': 'IT', 'MGCTs': 'MGCT'}
CLS = {'MTs': 0, 'ITs': 1, 'MGCTs': 2}
CAT_DIR = {'MTs': 'MTs', 'ITs': 'ITs', 'MGCTs': 'MGCTs'}
XLSX_NAME = {'MTs': 'MT_patient.xlsx', 'ITs': 'IT_patient.xlsx', 'MGCTs': 'MGCT_patient.xlsx'}

# 部位规范化：合并大小写/别名，罕见部位 → other
RARE = {'abdomen', 'vagina', 'head and neck', 'maxillary sinus'}


def normalize(loc: str) -> str:
    s = loc.strip().lower()
    if s == 'sacrococcygeal':
        s = 'sacrococcyx'
    return 'other' if s in RARE else s


def main():
    rows = []
    for cat, pre in CATS.items():
        xlsx = os.path.join(DATA, CAT_DIR[cat], XLSX_NAME[cat])
        df = pd.read_excel(xlsx)
        loc_col = [c for c in df.columns if 'Location' in str(c)][0]
        for _, r in df.iterrows():
            pid = str(r['Patient_ID'])
            rows.append({'Patient_ID': pid, 'class': CLS[cat],
                         'domain': normalize(str(r[loc_col])),
                         'domain_raw': str(r[loc_col]).strip().lower()})
    lab = pd.DataFrame(rows)
    lab = lab[lab['Patient_ID'] != 'MT_411'].reset_index(drop=True)

    # 行序与 features_full.csv 保持一致（论文数字口径）
    feat_order = pd.read_csv(os.path.join(REPO, 'experiments', 'radiomics_svm', 'features_full.csv'),
                             usecols=['Patient_ID'])
    feat_order = feat_order[feat_order['Patient_ID'] != 'MT_411']
    order = {pid: i for i, pid in enumerate(feat_order['Patient_ID'])}
    lab = lab.sort_values('Patient_ID', key=lambda s: s.map(order)).reset_index(drop=True)

    dom_counts = lab['domain'].value_counts().sort_index()
    print('domain 分布:')
    print(dom_counts.to_string())

    # 大域（参与训练/留出），other 仅测试
    big = ['ovary', 'testis', 'retroperitoneum', 'sacrococcyx', 'mediastinum']
    rare = sorted(lab.loc[lab['domain'] == 'other', 'Patient_ID'].tolist())
    lodo = {}
    for d in big:
        test = sorted(lab.loc[lab['domain'] == d, 'Patient_ID'].tolist())
        train = sorted(lab.loc[~lab['domain'].isin([d, 'other']), 'Patient_ID'].tolist())
        lab_d = lab.set_index('Patient_ID')
        lodo[d] = {'train': train, 'test': test, 'rare_test': rare,
                   'n_train': len(train), 'n_test': len(test), 'n_rare_test': len(rare),
                   'train_cls': {str(k): int(v) for k, v in lab_d.loc[train, 'class'].value_counts().items()},
                   'test_cls': {str(k): int(v) for k, v in lab_d.loc[test, 'class'].value_counts().items()}}
        print(f'{d:16s} train={len(train)} test={len(test)} rare={len(rare)}')

    lab.to_csv(os.path.join(HERE, 'domain_labels.csv'), index=False)
    with open(os.path.join(HERE, 'lodo_splits.json'), 'w') as f:
        json.dump(lodo, f, ensure_ascii=False, indent=2)
    print(f'\n已写入: {HERE}/domain_labels.csv, lodo_splits.json')


if __name__ == '__main__':
    main()
