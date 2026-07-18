"""
risk_model.py
把 LLM 零样本风险分数当作【无监督特征】,验证它给下游监督风控模型带来的增量。

思路:
    Qwen 在不接触标签的前提下，对每份征信报告输出一个风险分数
    (overdue_probability)。该分数因此是一个 "label-free / 无监督" 特征，
    可以和传统结构化特征 (DTI、利用率、逾期次数……) 一起喂进下游监督模型。

    本脚本用 5 折 Out-of-Fold 对比两套特征:
        A. baseline      : 仅传统结构化特征
        B. baseline+llm  : 传统特征 + LLM 风险分数
    若 B 的 OOF AUC/KS 高于 A，则说明 LLM 分数带来了传统特征之外的增量信号。

实现:
    - 纯 numpy 逻辑回归 (标准化 + L2 + 梯度下降)，与本仓库 "不依赖 sklearn" 一致
    - 每折仅用训练折拟合标准化参数与权重，在留出折预测，避免数据泄漏
    - 复用 evaluate.py 的 AUC / KS / Brier 实现

用法:
    python risk_model.py       # 读取 ../output/predictions.jsonl, 打印对比并写 metrics json
"""

import json
import re
from pathlib import Path

import numpy as np

from evaluate import auc_score, ks_score, brier_score, lift_at_k, confusion_at_threshold

SEED = 42
N_FOLDS = 5

# 传统结构化特征(数值型)
NUMERIC_FEATURES = [
    "age", "monthly_income", "loan_count", "credit_card_count",
    "total_loan_balance", "credit_card_utilization", "high_util_cards",
    "current_overdue_amount", "overdue_account_count", "overdue_count_24m",
    "recent_overdue_3m", "civil_judgments", "enforcement_records",
    "tax_arrears", "inquiry_3m", "inquiry_6m", "loan_approval_inquiry_3m", "dti",
]


def severity_to_ordinal(s: str) -> float:
    """最长逾期等级 -> 序数: 无逾期=0, M1=1, M2=2, ... """
    m = re.search(r"M(\d+)", s or "")
    return float(m.group(1)) if m else 0.0


def load_matrix(pred_path: Path):
    rows = [json.loads(l) for l in open(pred_path, encoding="utf-8")]
    y, base, llm = [], [], []
    for r in rows:
        f = r["features"]
        y.append(int(r["true_label"]))
        vec = [float(f[k]) for k in NUMERIC_FEATURES]
        vec.append(severity_to_ordinal(f.get("max_overdue_severity")))
        base.append(vec)
        llm.append(float(r["model_output"]["overdue_probability"]))
    return np.array(y), np.array(base), np.array(llm)


def _sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def logreg_fit(X, y, l2=1e-2, lr=0.5, iters=4000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        p = _sigmoid(X @ w + b)
        err = p - y
        w -= lr * (X.T @ err / n + l2 * w)
        b -= lr * err.mean()
    return w, b


def oof_predict(X, y, folds):
    """返回每个样本的 out-of-fold 预测概率。"""
    oof = np.zeros(len(y))
    for k in range(N_FOLDS):
        val = folds == k
        tr = ~val
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd == 0, 1.0, sd)
        Xtr = (X[tr] - mu) / sd
        Xval = (X[val] - mu) / sd
        w, b = logreg_fit(Xtr, y[tr])
        oof[val] = _sigmoid(Xval @ w + b)
    return oof


def report(name, y, score):
    tp, fp, tn, fn = confusion_at_threshold(y, score, 0.4)
    return {
        "model": name,
        "AUC": round(auc_score(y, score), 4),
        "KS": round(ks_score(y, score), 4),
        "Brier": round(brier_score(y, score), 4),
        "Lift@10%": round(lift_at_k(y, score, 0.1), 2),
    }


def main():
    root = Path(__file__).resolve().parent.parent
    y, base, llm = load_matrix(root / "output" / "predictions.jsonl")

    rng = np.random.default_rng(SEED)
    folds = rng.permutation(len(y)) % N_FOLDS  # 固定分折，两套特征共用

    base_oof = oof_predict(base, y, folds)
    combo_oof = oof_predict(np.column_stack([base, llm]), y, folds)

    results = [
        report("LLM 分数单独 (raw, 无下游模型)", y, llm),
        report("下游模型 · 仅传统特征 (OOF)", y, base_oof),
        report("下游模型 · 传统 + LLM 分数 (OOF)", y, combo_oof),
    ]

    print(f"\n样本: n={len(y)}  正类={int(y.sum())} ({y.mean()*100:.2f}%)  "
          f"特征: 传统 {base.shape[1]} 维 + LLM 分数 1 维\n")
    print(f"{'模型':<34}{'AUC':>9}{'KS':>9}{'Brier':>9}{'Lift@10%':>10}")
    print("-" * 71)
    for r in results:
        print(f"{r['model']:<34}{r['AUC']:>9}{r['KS']:>9}{r['Brier']:>9}{r['Lift@10%']:>10}")

    d_auc = results[2]["AUC"] - results[1]["AUC"]
    d_ks = results[2]["KS"] - results[1]["KS"]
    print("-" * 71)
    print(f"加入 LLM 分数后的增量:  ΔAUC={d_auc:+.4f}  ΔKS={d_ks:+.4f}")
    print("\n注: 标签为合成、正类仅 95 条，OOF 估计存在方差，增量方向需谨慎解读。")

    out = root / "output" / "downstream_metrics.json"
    out.write_text(json.dumps({
        "n": int(len(y)), "n_pos": int(y.sum()),
        "n_traditional_features": int(base.shape[1]),
        "results": results,
        "delta_auc": round(d_auc, 4), "delta_ks": round(d_ks, 4),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out.relative_to(root)}")


if __name__ == "__main__":
    main()
