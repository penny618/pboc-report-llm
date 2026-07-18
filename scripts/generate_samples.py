"""
generate_samples.py
基于一份征信报告样本,按照风控建模的真实分布衍生 N 份征信报告。

设计要点:
1. 按风险等级混合采样: 优质 50% / 中等 30% / 高风险 15% / 严重风险 5%
2. 每个字段按合理分布生成,字段之间保持业务一致性
   (例如: 高负债 → 利用率高 → 逾期概率提升 → 查询次数多)
3. 同步生成"真实标签" label_overdue_30d_in_6m (未来 6 个月内逾期 30+ 天)
   用于后续模型评估,但模型推理时不输入该字段
"""

import json
import random
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)
np.random.seed(42)
# DTI 校准用独立随机流,避免扰动主流(保证其余字段与旧数据逐字节一致,便于 A/B)
dti_rng = random.Random(12345)
# 固定报告锚点日期:避免用 datetime.now() 导致日期字段随运行日变化,保证逐字节可复现
REPORT_DATE = datetime(2026, 5, 9)

# ---------- 配置区 ----------
N_SAMPLES = 1200
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(OUTPUT_DIR)


# 风险等级分布(参考银行真实坏账率)
RISK_TIERS = [
    ("excellent", 0.50, 0.01),
    ("good",      0.30, 0.05),
    ("medium",    0.15, 0.25),
    ("bad",       0.05, 0.65),
]

SURNAMES = ["张","王","李","赵","刘","陈","杨","黄","周","吴","徐","孙","胡","朱","高","林","何","郭","马","罗"]
GIVEN = ["伟","芳","娜","秀英","敏","静","丽","强","磊","军","洋","勇","艳","杰","娟","涛","明","超","秀兰","霞"]
CITIES = ["北京","上海","广州","深圳","杭州","成都","武汉","西安","南京","重庆","苏州","天津","长沙","郑州","青岛"]
EDUCATION = ["高中","大专","本科","硕士","博士"]
MARITAL = ["未婚","已婚","离异","丧偶"]
OCCUPATIONS = ["工程师","教师","医生","会计","销售","公务员","个体经营","管理人员","技术员","服务员"]
EMPLOYERS = ["某科技公司","某商贸公司","某事业单位","某制造企业","某金融机构","某教育机构","某医疗机构","某互联网公司"]
BANKS = ["工商银行","建设银行","农业银行","中国银行","招商银行","中信银行","民生银行","交通银行","平安银行","浦发银行","兴业银行"]
LOAN_TYPES = ["个人住房贷款","个人消费贷款","个人经营贷款","汽车贷款"]


def gen_id_card_masked():
    prefix = random.choice(["110101","310101","440101","330101","510101","420101"])
    suffix = "".join(random.choices("0123456789", k=4))
    return f"{prefix}********{suffix}"


def gen_overdue_history(months: int, severity: str):
    """
    生成最近 N 个月的逾期状态序列:
    0=未逾期, 1=M1(1-30天), 2=M2(31-60), 3=M3, ..., 7=呆账
    """
    if severity == "excellent":
        return [0] * months
    elif severity == "good":
        h = [0] * months
        if random.random() < 0.3 and months > 0:
            h[random.randint(0, months - 1)] = 1
        return h
    elif severity == "medium":
        h = [0] * months
        n_overdue = min(months, random.randint(2, 5))
        for _ in range(n_overdue):
            h[random.randint(0, months - 1)] = random.choice([1,1,1,2])
        return h
    else:  # bad
        h = [0] * months
        n_overdue = min(months, random.randint(5, 12))
        for _ in range(n_overdue):
            h[random.randint(0, months - 1)] = random.choice([1,2,2,3,3,4])
        return h


def summarize_overdue(history_lists):
    if not history_lists:
        return 0, 0, 0
    flat = [v for h in history_lists for v in h]
    overdue_count = sum(1 for v in flat if v >= 1)
    max_overdue_months = max(flat) if flat else 0
    overdue_account_count = sum(1 for h in history_lists if (h and max(h) >= 1))
    return overdue_count, max_overdue_months, overdue_account_count


def gen_one_report(report_idx: int, severity: str) -> dict:
    # ---------- 个人基本信息 ----------
    age = int(np.clip(np.random.normal(38, 10), 22, 65))
    education = random.choice(EDUCATION)
    edu_factor = {"高中":0.6,"大专":0.8,"本科":1.0,"硕士":1.4,"博士":1.8}[education]
    age_factor = min(1.5, 0.8 + (age - 22) * 0.02)
    base_income = 8000 * edu_factor * age_factor
    monthly_income = int(np.clip(np.random.normal(base_income, base_income * 0.3), 3000, 80000))

    personal_info = {
        "name": random.choice(SURNAMES) + random.choice(GIVEN),
        "id_card_masked": gen_id_card_masked(),
        "gender": random.choice(["男","女"]),
        "age": age,
        "marital_status": random.choice(MARITAL),
        "education": education,
        "employer": random.choice(EMPLOYERS),
        "occupation": random.choice(OCCUPATIONS),
        "monthly_income": monthly_income,
        "residence_city": random.choice(CITIES),
        "residence_years": random.randint(1, max(1, min(20, age - 18))),
    }

    # ---------- 贷款 ----------
    if severity == "excellent":
        n_loans = random.choices([0,1,2], weights=[0.3,0.5,0.2])[0]
    elif severity == "good":
        n_loans = random.choices([0,1,2,3], weights=[0.2,0.4,0.3,0.1])[0]
    elif severity == "medium":
        n_loans = random.choices([1,2,3,4], weights=[0.2,0.3,0.3,0.2])[0]
    else:
        n_loans = random.choices([2,3,4,5,6], weights=[0.1,0.2,0.3,0.25,0.15])[0]

    loans = []
    total_loan_balance = 0
    loan_overdue_histories = []

    for _ in range(n_loans):
        loan_type = random.choice(LOAN_TYPES)
        if loan_type == "个人住房贷款":
            amount = random.randint(500_000, 3_000_000); term = random.choice([240, 300, 360])
        elif loan_type == "汽车贷款":
            amount = random.randint(80_000, 400_000); term = random.choice([36, 48, 60])
        else:
            amount = random.randint(20_000, 500_000); term = random.choice([12, 24, 36, 60])

        elapsed = random.randint(1, max(2, term - 1))
        balance = max(0, int(amount * (1 - elapsed / term) * random.uniform(0.95, 1.05)))
        monthly_payment = int(amount / term * random.uniform(1.05, 1.2))
        history = gen_overdue_history(min(24, elapsed), severity)
        loan_overdue_histories.append(history)

        loans.append({
            "loan_type": loan_type,
            "lender": random.choice(BANKS),
            "amount": amount, "balance": balance,
            "term_months": term, "remaining_months": term - elapsed,
            "monthly_payment": monthly_payment,
            "start_date": (REPORT_DATE - timedelta(days=elapsed*30)).strftime("%Y-%m-%d"),
            "status": "正常" if (not history or max(history) < 3) else "关注",
            "overdue_history_24m": history,
        })
        total_loan_balance += balance

    # ---------- 信用卡 ----------
    if severity == "excellent":
        n_cards = random.choices([1,2,3,4], weights=[0.1,0.3,0.4,0.2])[0]
        utilization_base = random.uniform(0.05, 0.30)
    elif severity == "good":
        n_cards = random.choices([1,2,3,4,5], weights=[0.1,0.2,0.3,0.25,0.15])[0]
        utilization_base = random.uniform(0.20, 0.55)
    elif severity == "medium":
        n_cards = random.choices([2,3,4,5,6], weights=[0.1,0.2,0.3,0.25,0.15])[0]
        utilization_base = random.uniform(0.55, 0.85)
    else:
        n_cards = random.choices([3,4,5,6,7,8], weights=[0.1,0.15,0.2,0.25,0.2,0.1])[0]
        utilization_base = random.uniform(0.85, 1.05)

    credit_cards = []
    total_limit = 0; total_used = 0
    cc_overdue_histories = []

    for _ in range(n_cards):
        limit = random.choice([10_000, 20_000, 30_000, 50_000, 80_000, 100_000, 150_000])
        util = float(np.clip(np.random.normal(utilization_base, 0.1), 0, 1.2))
        used = int(limit * util)
        history = gen_overdue_history(24, severity)
        cc_overdue_histories.append(history)

        credit_cards.append({
            "issuer": random.choice(BANKS),
            "card_type": "信用卡",
            "credit_limit": limit, "used_amount": used,
            "utilization": round(util, 4),
            "issue_date": (REPORT_DATE - timedelta(days=random.randint(180, 3650))).strftime("%Y-%m-%d"),
            "status": "正常" if max(history) < 3 else "关注",
            "min_payment_overdue_24m": sum(1 for v in history if v >= 1),
            "overdue_history_24m": history,
        })
        total_limit += limit; total_used += used

    # ---------- 按风险等级校准 DTI(因果一致的关键修正) ----------
    # 原实现里 monthly_payment = amount/term,而 amount 与收入无关,
    # 短期大额消费/经营贷会产生"月供 > 数倍月收入"的天价 DTI(如 DTI=600%)。
    # 这里按风险等级抽取目标负债收入比,再把总债务月供按余额占比回填到各笔贷款,
    # 使 DTI 落在真实业务区间且与风险等级单调一致。
    # (amount/term 仍作为贷款的授信事实保留,月供以实际债务负担为准。)
    TARGET_DTI_BAND = {
        "excellent": (0.10, 0.40),
        "good":      (0.30, 0.60),
        "medium":    (0.55, 0.90),
        "bad":       (0.85, 1.60),
    }[severity]
    target_dti = dti_rng.uniform(*TARGET_DTI_BAND)
    cc_min_total = sum(c["used_amount"] * 0.1 for c in credit_cards)  # 与 feature_engineering 口径一致
    loan_debt_service = max(0.0, target_dti * monthly_income - cc_min_total)
    bal_sum = sum(l["balance"] for l in loans)
    for l in loans:
        share = (l["balance"] / bal_sum) if bal_sum > 0 else (1.0 / len(loans))
        l["monthly_payment"] = max(200, int(loan_debt_service * share))

    # ---------- 汇总 ----------
    all_histories = loan_overdue_histories + cc_overdue_histories
    overdue_count_24m, max_overdue_months_24m, overdue_account_count = summarize_overdue(all_histories)

    current_overdue_amount = 0
    if severity in ("medium", "bad"):
        for loan in loans:
            if loan["overdue_history_24m"] and loan["overdue_history_24m"][-1] >= 1:
                current_overdue_amount += loan["monthly_payment"]
        for card in credit_cards:
            if card["overdue_history_24m"][-1] >= 1:
                current_overdue_amount += int(card["used_amount"] * 0.1)

    summary = {
        "loan_account_count": len(loans),
        "credit_card_account_count": len(credit_cards),
        "total_loan_balance": total_loan_balance,
        "total_credit_card_used": total_used,
        "total_credit_card_limit": total_limit,
        "credit_card_utilization": round(total_used / total_limit, 4) if total_limit else 0,
        "guarantee_count": random.choices([0,1,2], weights=[0.85,0.12,0.03])[0],
        "overdue_account_count": overdue_account_count,
        "current_overdue_amount": current_overdue_amount,
        "max_overdue_months_24m": max_overdue_months_24m,
        "overdue_count_24m": overdue_count_24m,
    }

    # ---------- 公共记录 ----------
    if severity == "bad":
        public_records = {
            "civil_judgments": random.choices([0,1,2], weights=[0.6,0.3,0.1])[0],
            "enforcement_records": random.choices([0,1], weights=[0.7,0.3])[0],
            "tax_arrears": random.choices([0,1], weights=[0.85,0.15])[0],
            "administrative_penalties": random.choices([0,1], weights=[0.85,0.15])[0],
        }
    else:
        public_records = {"civil_judgments":0,"enforcement_records":0,"tax_arrears":0,"administrative_penalties":0}

    # ---------- 查询记录 ----------
    if severity == "excellent":
        q3, q6, q12 = random.randint(0,2), random.randint(1,4), random.randint(2,8)
    elif severity == "good":
        q3, q6, q12 = random.randint(1,4), random.randint(3,8), random.randint(5,15)
    elif severity == "medium":
        q3, q6, q12 = random.randint(4,10), random.randint(8,18), random.randint(15,30)
    else:
        q3, q6, q12 = random.randint(8,20), random.randint(15,35), random.randint(30,60)

    inquiry_records = {
        "inquiry_count_3m": q3, "inquiry_count_6m": q6, "inquiry_count_12m": q12,
        "loan_approval_inquiry_3m": min(q3, random.randint(0, max(1, q3 // 2))),
        "credit_card_approval_inquiry_3m": min(q3, random.randint(0, max(1, q3 // 2))),
        "self_inquiry_3m": random.randint(0, 2),
    }

    # ---------- 真实标签(评估用) ----------
    base_pd = {t[0]: t[2] for t in RISK_TIERS}[severity]
    label = 1 if random.random() < base_pd else 0

    return {
        "report_id": f"PBC-2026-{report_idx:05d}",
        "report_date": REPORT_DATE.strftime("%Y-%m-%d"),
        "personal_info": personal_info,
        "summary": summary,
        "loans": loans,
        "credit_cards": credit_cards,
        "public_records": public_records,
        "inquiry_records": inquiry_records,
        "_meta": {"severity_tag": severity, "label_overdue_30d_in_6m": label}
    }


def main():
    severities = []
    for tier_name, ratio, _ in RISK_TIERS:
        severities.extend([tier_name] * int(N_SAMPLES * ratio))
    while len(severities) < N_SAMPLES:
        severities.append("good")
    random.shuffle(severities)

    print(f"开始生成 {N_SAMPLES} 份征信报告样本...")
    label_dist = {"excellent":0,"good":0,"medium":0,"bad":0}
    label_count = 0

    for i, sev in enumerate(severities, start=1):
        report = gen_one_report(i, sev)
        label_dist[sev] += 1
        label_count += report["_meta"]["label_overdue_30d_in_6m"]
        out_path = OUTPUT_DIR / f"report_{i:05d}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    index = {
        "total": N_SAMPLES,
        "severity_distribution": label_dist,
        "true_overdue_rate": round(label_count / N_SAMPLES, 4),
        "files": [f"report_{i:05d}.json" for i in range(1, N_SAMPLES + 1)],
    }
    with open(OUTPUT_DIR / "_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"✓ 完成! 共生成 {N_SAMPLES} 个文件")
    print(f"  风险分层: {label_dist}")
    print(f"  真实逾期率: {index['true_overdue_rate']:.2%}")
    print(f"  输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

