"""
evaluate.py
用真实标签评估模型输出的逾期概率质量。

核心指标:
- AUC      : 区分能力,>0.7 算可用,>0.8 优秀
- KS       : 风控行业标准,>0.3 可用,>0.4 优秀
- Brier Score : 概率校准度,越小越好
- 分层捕获率 : Top 10% 高风险样本中真实坏客户的占比
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np


def load_predictions(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def auc_score(y_true, y_score):
    """手算 AUC,避免依赖 sklearn"""
    pairs = sorted(zip(y_score, y_true), reverse=True)
    n_pos = sum(y_true); n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = 0; fp = 0; auc = 0; prev_score = None
    prev_tp = 0; prev_fp = 0
    for s, y in pairs:
        if s != prev_score:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_score = s; prev_tp = tp; prev_fp = fp
        if y == 1: tp += 1
        else: fp += 1
    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
    return auc / (n_pos * n_neg)


def ks_score(y_true, y_score):
    """KS = max(TPR - FPR)"""
    pairs = sorted(zip(y_score, y_true), reverse=True)
    n_pos = sum(y_true); n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = 0; fp = 0; ks = 0
    for _, y in pairs:
        if y == 1: tp += 1
        else: fp += 1
        ks = max(ks, tp / n_pos - fp / n_neg)
    return ks


def brier_score(y_true, y_score):
    return float(np.mean([(p - y) ** 2 for p, y in zip(y_score, y_true)]))


def lift_at_k(y_true, y_score, k_pct=0.1):
    n = len(y_true)
    base_rate = sum(y_true) / n
    if base_rate == 0:
        return float("nan")
    pairs = sorted(zip(y_score, y_true), reverse=True)
    top_k = pairs[:max(1, int(n * k_pct))]
    top_rate = sum(y for _, y in top_k) / len(top_k)
    return top_rate / base_rate


def confusion_at_threshold(y_true, y_score, threshold):
    tp = fp = tn = fn = 0
    for y, s in zip(y_true, y_score):
        pred = 1 if s >= threshold else 0
        if pred == 1 and y == 1: tp += 1
        elif pred == 1 and y == 0: fp += 1
        elif pred == 0 and y == 0: tn += 1
        else: fn += 1
    return tp, fp, tn, fn


def main():
    path = Path(__file__).parent.parent / "output" / "predictions.jsonl"
    rows = load_predictions(path)

    # 过滤掉解析失败的
    valid = [r for r in rows if r["model_output"].get("overdue_probability", -1) >= 0]
    print(f"有效样本: {len(valid)}/{len(rows)}")

    y_true = [r["true_label"] for r in valid]
    y_score = [r["model_output"]["overdue_probability"] for r in valid]

    print("\n========== 整体表现 ==========")
    print(f"  AUC          : {auc_score(y_true, y_score):.4f}")
    print(f"  KS           : {ks_score(y_true, y_score):.4f}")
    print(f"  Brier Score  : {brier_score(y_true, y_score):.4f}")
    print(f"  Top 10% Lift : {lift_at_k(y_true, y_score, 0.1):.2f}x")
    print(f"  Top 20% Lift : {lift_at_k(y_true, y_score, 0.2):.2f}x")

    print("\n========== 按真实风险等级分组 ==========")
    by_sev = {}
    for r in valid:
        sev = r["true_severity"]
        by_sev.setdefault(sev, []).append(r["model_output"]["overdue_probability"])
    for sev in ["excellent", "good", "medium", "bad"]:
        if sev in by_sev:
            scores = by_sev[sev]
            print(f"  {sev:10s} | n={len(scores):4d} | 平均PD: {np.mean(scores):.2%} | "
                  f"中位数: {np.median(scores):.2%} | 最大: {np.max(scores):.2%}")

    print("\n========== 风险等级预测一致性 ==========")
    level_dist = Counter(r["model_output"]["risk_level"] for r in valid)
    for k, v in level_dist.most_common():
        print(f"  {k}: {v} ({v/len(valid):.1%})")

    print("\n========== 拒绝策略效果(threshold=0.4) ==========")
    tp, fp, tn, fn = confusion_at_threshold(y_true, y_score, 0.4)
    total_bad = tp + fn; total_good = fp + tn
    bad_capture = tp / total_bad if total_bad else 0
    good_loss = fp / total_good if total_good else 0
    print(f"  坏客户捕获率: {bad_capture:.1%}  (拒掉的坏客户/全部坏客户)")
    print(f"  好客户误杀率: {good_loss:.1%}    (误拒的好客户/全部好客户)")
    print(f"  混淆矩阵: TP={tp}, FP={fp}, TN={tn}, FN={fn}")


if __name__ == "__main__":
    main()

