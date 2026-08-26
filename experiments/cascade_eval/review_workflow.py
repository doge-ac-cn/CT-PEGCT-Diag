#!/usr/bin/env python
"""cascade_evalb: 低置信度人工复核工作流 — 113 例公平级联扩展验证
方法 (同 low-confidence review protocol): 置信度 = max(prob_pred); 阈值 τ 以下病例改用 GT ROI 预测 (模拟人工复核)
目标: 验证 40 例结论 (τ=0.6, 17% 复核, ACC 0.825→0.900) 在 113 例上是否稳健
输出: 阈值-复核率-挽回率曲线 + 工作流 ACC 对比表
"""
import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.isdir(BASE):
    raise RuntimeError('BASE path not found; run from a checkout of the repository')
import argparse
OUT = f'{BASE}/experiments/cascade_eval'

def main():
    global N
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=113)
    args = ap.parse_args()
    N = args.n
    z = np.load(f'{OUT}/../bootstrap/cascade_prob_{N}.npz')
    y_te, pred_gt, pred_pr = z['y_te'], z['pred_gt'], z['pred_pr']
    prob_gt, prob_pr = z['prob_gt'], z['prob_pr']
    n = len(y_te)
    conf = prob_pr.max(axis=1)
    auto_correct = pred_pr == y_te
    gt_correct = pred_gt == y_te

    # 1) 置信度能否区分误判?
    print('=== 置信度 (max prob of pred ROI) 区分误判 ===')
    print(f'误判病例 mean conf: {conf[~auto_correct].mean():.4f} (n={int((~auto_correct).sum())})')
    print(f'正确病例 mean conf: {conf[auto_correct].mean():.4f} (n={int(auto_correct.sum())})')
    from scipy.stats import mannwhitneyu
    try:
        u, p = mannwhitneyu(conf[~auto_correct], conf[auto_correct], alternative='less')
        print(f'Mann-Whitney (误判<正确): p={p:.4f}')
    except Exception as e:
        print(f'MWU err: {e}')

    # 2) 复核工作流
    print('\n=== 低置信度复核策略 (阈值 τ) ===')
    print(f'{"tau":>6s} {"review%":>8s} {"workflow_ACC":>12s} {"auto_only":>9s} {"rescued":>7s} {"hurt":>5s}')
    rows = []
    for tau in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        review = conf < tau
        final_pred = np.where(review, pred_gt, pred_pr)
        acc_wf = accuracy_score(y_te, final_pred)
        acc_auto = accuracy_score(y_te, pred_pr)
        rescued = int((review & (~auto_correct) & gt_correct).sum())
        hurt = int((review & auto_correct & (~gt_correct)).sum())
        print(f'{tau:6.2f} {review.mean()*100:7.1f}% {acc_wf:12.4f} {acc_auto:9.4f} {rescued:7d} {hurt:5d}')
        rows.append({'tau': tau, 'review_rate': float(review.mean()), 'workflow_acc': float(acc_wf),
                     'auto_acc': float(acc_auto), 'rescued': rescued, 'hurt': hurt})

    # 3) 关键阈值详情
    tau60 = [r for r in rows if abs(r['tau'] - 0.6) < 1e-9][0]
    print(f'\nτ=0.6: 复核率 {tau60["review_rate"]*100:.1f}%, ACC {tau60["auto_acc"]:.4f} → {tau60["workflow_acc"]:.4f} (Δ{tau60["workflow_acc"]-tau60["auto_acc"]:+.4f})')
    # 最优阈值
    best = max(rows, key=lambda r: r['workflow_acc'])
    print(f'最优 τ={best["tau"]}: ACC {best["workflow_acc"]:.4f}, 复核率 {best["review_rate"]*100:.1f}%')

    # 4) 曲线图
    fig, ax1 = plt.subplots(figsize=(7, 5))
    taus = [r['tau'] for r in rows]
    ax1.plot(taus, [r['workflow_acc'] for r in rows], 'o-', color='#4C72B0', label='Workflow ACC')
    ax1.axhline(accuracy_score(y_te, pred_gt), color='green', ls='--', lw=1, label=f'GT-ROI ACC ({accuracy_score(y_te, pred_gt):.3f})')
    ax1.axhline(accuracy_score(y_te, pred_pr), color='red', ls='--', lw=1, label=f'Auto-only ACC ({accuracy_score(y_te, pred_pr):.3f})')
    ax1.set_xlabel('Confidence threshold τ'); ax1.set_ylabel('ACC'); ax1.legend(loc='lower right')
    ax2 = ax1.twinx()
    ax2.plot(taus, [r['review_rate']*100 for r in rows], 's--', color='gray', alpha=0.7, label='Review rate')
    ax2.set_ylabel('Review rate (%)'); ax2.set_ylim(0, 110)
    ax1.set_title(f'Low-confidence review workflow (113-case fair cascade)')
    fig.tight_layout()
    fig.savefig(f'{OUT}/figs/figD_review_workflow_{N}.png', dpi=150)
    plt.close(fig)

    json.dump({'n': N, 'conf_mean': {'wrong': float(conf[~auto_correct].mean()), 'correct': float(conf[auto_correct].mean())},
               'tau_curve': rows, 'best': best, 'tau60': tau60,
               'gt_acc': float(accuracy_score(y_te, pred_gt)), 'auto_acc': float(accuracy_score(y_te, pred_pr))},
              open(f'{OUT}/results_review_{N}.json', 'w'), indent=2)
    print(f'\nsaved -> results_review_{N}.json, figs/figD_review_workflow_{N}.png')

if __name__ == '__main__':
    main()
