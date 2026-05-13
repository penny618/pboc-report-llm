"""
feature_engineering.py
将原始征信报告 JSON 压缩为 Qwen 友好的"特征摘要文本"。

为什么不直接喂原始 JSON?
- 原始 JSON 一份约 2-5K tokens, 1200 份 = 200万+ tokens, 推理慢、贵
- LLM 对结构化数值 token 化效率低,关键风险信号被淹没在低价值字段中
- 摘要文本: 约 300-500 tokens, 保留全部风险关键变量,可读性强
"""

from typing import Dict, Any
import json, sys
from pathlib import Path


def map_overdue_severity(max_overdue: int) -> str:
    """逾期等级翻译"""
    return {0:"无逾期", 1:"M1(1-30天)", 2:"M2(31-60天)",
            3:"M3(61-90天)", 4:"M4(91-120天)",
            5:"M5(121-150天)", 6:"M6(151-180天)",
            7:"呆账"}.get(max_overdue, f"M{max_overdue}")


def calc_dti(report: Dict[str, Any]) -> float:
    """计算负债收入比 (Debt-to-Income)"""
    monthly_income = report["personal_info"]["monthly_income"]
    if monthly_income <= 0:
        return float("inf")
    monthly_loan_pay = sum(l["monthly_payment"] for l in report["loans"])
    # 信用卡按已用额度的 10% 估算最低还款
    monthly_cc_min = sum(c["used_amount"] * 0.1 for c in report["credit_cards"])
    return round((monthly_loan_pay + monthly_cc_min) / monthly_income, 4)


def extract_features(report: Dict[str, Any]) -> Dict[str, Any]:
    """从原始报告提取关键风险变量"""
    pi = report["personal_info"]
    s = report["summary"]
    pr = report["public_records"]
    iq = report["inquiry_records"]

    # 高利用率卡数
    high_util_cards = sum(1 for c in report["credit_cards"] if c["utilization"] >= 0.8)

    # 最近 3 个月有逾期的账户数
    recent_overdue_3m = 0
    for l in report["loans"]:
        if l["overdue_history_24m"] and any(v >= 1 for v in l["overdue_history_24m"][-3:]):
            recent_overdue_3m += 1
    for c in report["credit_cards"]:
        if any(v >= 1 for v in c["overdue_history_24m"][-3:]):
            recent_overdue_3m += 1

    return {
        "age": pi["age"],
        "education": pi["education"],
        "occupation": pi["occupation"],
        "monthly_income": pi["monthly_income"],
        "marital_status": pi["marital_status"],
        "loan_count": s["loan_account_count"],
        "credit_card_count": s["credit_card_account_count"],
        "total_loan_balance": s["total_loan_balance"],
        "credit_card_utilization": s["credit_card_utilization"],
        "high_util_cards": high_util_cards,
        "current_overdue_amount": s["current_overdue_amount"],
        "overdue_account_count": s["overdue_account_count"],
        "max_overdue_severity": map_overdue_severity(s["max_overdue_months_24m"]),
        "overdue_count_24m": s["overdue_count_24m"],
        "recent_overdue_3m": recent_overdue_3m,
        "civil_judgments": pr["civil_judgments"],
        "enforcement_records": pr["enforcement_records"],
        "tax_arrears": pr["tax_arrears"],
        "inquiry_3m": iq["inquiry_count_3m"],
        "inquiry_6m": iq["inquiry_count_6m"],
        "loan_approval_inquiry_3m": iq["loan_approval_inquiry_3m"],
        "dti": calc_dti(report),
    }


def build_summary_text(report: Dict[str, Any]) -> str:
    """生成中文摘要,用于 LLM 输入"""
    f = extract_features(report)
    income_w = f["monthly_income"] / 10000

    text = f"""【个人基本信息】
姓名: {report['personal_info']['name']}, 年龄: {f['age']}岁, 学历: {f['education']}, 婚姻: {f['marital_status']}
职业: {f['occupation']}, 月收入: {income_w:.1f}万元

【信贷账户概要】
- 贷款账户数: {f['loan_count']} 个, 贷款余额合计: {f['total_loan_balance']/10000:.1f} 万元
- 信用卡数: {f['credit_card_count']} 张, 总体利用率: {f['credit_card_utilization']:.1%} (高利用率卡数: {f['high_util_cards']})
- 负债收入比(DTI): {f['dti']:.2%}

【逾期表现】
- 当前逾期金额: {f['current_overdue_amount']:,} 元
- 24 个月内逾期账户数: {f['overdue_account_count']}, 累计逾期次数: {f['overdue_count_24m']}
- 最长逾期等级: {f['max_overdue_severity']}
- 最近 3 个月发生逾期的账户数: {f['recent_overdue_3m']}

【公共记录】
- 民事判决: {f['civil_judgments']} 项, 强制执行: {f['enforcement_records']} 项, 税收欠缴: {f['tax_arrears']} 项

【近期信贷需求(查询)】
- 近 3 月机构查询: {f['inquiry_3m']} 次 (其中贷款审批 {f['loan_approval_inquiry_3m']} 次)
- 近 6 月机构查询: {f['inquiry_6m']} 次"""
    return text

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

INPUT_DIR = Path("./data/samples")
OUTPUT_DIR = Path("./data/smp_prompt")

def process_one_file(report_path: Path, output_dir: Path = OUTPUT_DIR):
    report = load_json(report_path)
    summary_text = build_summary_text(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (report_path.stem + ".txt")
    with out_path.open("w", encoding="utf-8") as f:
        f.write(summary_text)

    #print(f"[完成] {report_path} -> {out_path}")


def process_batch(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    start_id: int = 1,
    end_id: int = 1200,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(start_id, end_id + 1):
        report_path = input_dir / f"report_{idx:05d}.json"
        if not report_path.exists():
            print(f"[跳过] 文件不存在: {report_path}")
            continue

        try:
            process_one_file(report_path, output_dir)
        except Exception as e:
            print(f"[失败] {report_path}: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 传入单个文件路径时，处理单文件
        process_one_file(Path(sys.argv[1]))
    else:
        # 默认批量处理
        process_batch()





