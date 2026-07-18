"""
calibrate.py
对 LLM 原始逾期概率做概率校准 (Probability Calibration)。

动机:
    Qwen3.5-4B 相比 2B 更严格地执行 "DTI>70% 高风险" 等规则,排序能力 (AUC) 更强,
    但输出的绝对 PD 系统性偏高 (excellent 组平均 PD 达 18%),导致 Brier Score 变差、
    固定阈值 0.4 下好客户误杀率偏高。

    校准是一个【单调映射】: 只改变 PD 的绝对刻度,不改变样本排序,
    因此 AUC/KS 不变,而 Brier、阈值决策质量显著改善。

方法:
    - Isotonic Regression (保序回归, PAV 算法),纯 numpy 实现,无需 sklearn
    - 5 折 Out-of-Fold: 每折用其余 4 折拟合、在本折预测,避免用同一批样本
      "既拟合又评估" 带来的乐观偏差 (数据泄漏)

用法:
    python calibrate.py              # 读取 predictions.jsonl, 输出校准前后对比
"""

import json
from pathlib import Path

import numpy as np

from evaluate import auc_score, ks_score, brier_score, lift_at_k, confusion_at_threshold


# ------------------------------------------------------------
#  保序回归 (Pool Adjacent Violators)
# ------------------------------------------------------------

def isotonic_fit_predict(x_train: np.ndarray, y_train: np.ndarray):
    """拟合单调非降的 x->y 映射,返回一个 predict(x_new) 函数。"""
    order = np.argsort(x_train, kind="mergesort")
    xs = x_train[order]
    ys = y_train[order].astype(float)

    # PAV: 维护若干 (均值, 权重) 块,遇到逆序即向前合并
    vals, wts = [], []
    for v in ys:
        vals.append(float(v)); wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2 = vals.pop(), wts.pop()
            v1, w1 = vals.pop(), wts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)

    # 把块均值展开回每个训练点
    yhat = np.empty(len(ys))
    idx = 0
    for v, w in zip(vals, wts):
        cnt = int(round(w))
        yhat[idx:idx + cnt] = v
        idx += cnt

    # 以唯一 x 为锚点做分段线性插值 (与 sklearn 行为一致)
    ux = np.unique(xs)
    uy = np.array([yhat[np.where(xs == xv)[0][-1]] for xv in ux])

    def predict(x_new: np.ndarray) -> np.ndarray:
        return np.clip(np.interp(x_new, ux, uy), 0.0, 1.0)

    return predict


def oof_calibrate(x: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """5 折 out-of-fold 保序校准,返回与输入等长的校准后概率。"""
    n = len(x)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    cal = np.empty(n)
    for f in range(k):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(k) if j != f])
        predict = isotonic_fit_predict(x[tr], y[tr])
        cal[te] = predict(x[te])
    return cal


def best_threshold(y_true, y_score):
    """按 Youden's J (=TPR-FPR, 即 KS 取得点) 选最优阈值。"""
    ths = np.unique(y_score)
    best_j, best_t = -1, 0.5
    for t in ths:
        tp, fp, tn, fn = confusion_at_threshold(y_true, y_score, t)
        tpr = tp / (tp + fn) if (tp + fn) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        if tpr - fpr > best_j:
            best_j, best_t = tpr - fpr, t
    return best_t


def report_operating_point(y_true, y_score, threshold, tag):
    tp, fp, tn, fn = confusion_at_threshold(y_true, y_score, threshold)
    total_bad = tp + fn; total_good = fp + tn
    cap = tp / total_bad if total_bad else 0
    loss = fp / total_good if total_good else 0
    print(f"  [{tag}] 阈值={threshold:.3f} | 坏客户捕获率 {cap:.1%} | 好客户误杀率 {loss:.1%} | "
          f"TP={tp} FP={fp} TN={tn} FN={fn}")


def main():
    path = Path(__file__).parent.parent / "output" / "predictions.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    valid = [r for r in rows if r["model_output"].get("overdue_probability", -1) >= 0]

    y = np.array([r["true_label"] for r in valid])
    raw = np.array([r["model_output"]["overdue_probability"] for r in valid])

    cal = oof_calibrate(raw, y, k=5, seed=0)

    print(f"有效样本: {len(valid)}/{len(rows)}  |  真实违约率(base rate): {y.mean():.2%}\n")

    print("========== 校准前后对比 (排序类指标不受校准影响) ==========")
    print(f"  {'指标':<14}{'原始 PD':>12}{'校准后 PD':>14}")
    print(f"  {'AUC':<14}{auc_score(y, raw):>12.4f}{auc_score(y, cal):>14.4f}")
    print(f"  {'KS':<14}{ks_score(y, raw):>12.4f}{ks_score(y, cal):>14.4f}")
    print(f"  {'Brier Score':<14}{brier_score(y, raw):>12.4f}{brier_score(y, cal):>14.4f}   <- 校准核心收益")
    print(f"  {'Top10% Lift':<14}{lift_at_k(y, raw, 0.1):>11.2f}x{lift_at_k(y, cal, 0.1):>13.2f}x")

    print("\n========== 校准后按真实等级的 PD 分布 ==========")
    by = {}
    for r, c in zip(valid, cal):
        by.setdefault(r["true_severity"], []).append(c)
    for sev in ["excellent", "good", "medium", "bad"]:
        v = np.array(by.get(sev, []))
        if len(v):
            print(f"  {sev:10s} | n={len(v):4d} | 平均PD {v.mean():.2%} | 中位数 {np.median(v):.2%} | 最大 {v.max():.2%}")

    print("\n========== 拒绝策略 (校准修复阈值语义) ==========")
    report_operating_point(y, raw, 0.4, "原始@0.4")
    report_operating_point(y, cal, 0.4, "校准@0.4")
    t = best_threshold(y, cal)
    report_operating_point(y, cal, t, "校准@最优(Youden)")

    # 写回校准后的概率
    out = Path(__file__).parent.parent / "output" / "predictions_calibrated.jsonl"
    ci = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            if r["model_output"].get("overdue_probability", -1) >= 0:
                r["model_output"]["calibrated_probability"] = round(float(cal[ci]), 4)
                ci += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✓ 已写出校准结果: {out}")


if __name__ == "__main__":
    main()
