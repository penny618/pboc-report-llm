"""
qwen_inference.py
Qwen 本地模型批量解读征信报告。

支持两种模式:
- transformers 模式 (默认): 简单易用,适合 7B/14B 单卡推理
- vllm 模式: 高吞吐,适合大规模批量推理 (推荐 1000+ 样本)

模型推荐:
- Qwen2.5-7B-Instruct (16GB 显存可跑, 速度快)
- Qwen2.5-14B-Instruct (单 A100 40G 推荐, 准确度更高)
- Qwen2.5-72B-Instruct (多卡, 生产环境推荐)

输出: 每份征信报告对应一个 JSON,含
    - overdue_probability  : 逾期概率 (0-1)
    - risk_level           : 风险等级 [低/中/高/极高]
    - key_drivers          : 关键判断依据 (列表)
    - explanation          : 自然语言解释
    - recommendation       : 信贷决策建议 [批准/有条件批准/拒绝]
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Any

from feature_engineering import build_summary_text, extract_features


# ============================================================
#  Prompt 设计 - 这是整个项目最关键的部分
# ============================================================

SYSTEM_PROMPT = """你是一名资深银行风控建模专家,精通中国人民银行征信报告解读和个人信贷违约预测。

你的任务: 根据用户提供的征信报告摘要,评估其【未来 6 个月内发生 30 天以上逾期的概率】(PD, Probability of Default)。

评估方法论:
1. 重点关注"现金流压力" - DTI 负债收入比 > 70% 是高风险信号
2. 重点关注"履约历史" - 24 个月内逾期表现是最强的预测变量
3. 重点关注"近期信贷渴求度" - 近 3 月查询次数 > 6 次提示资金紧张
4. 重点关注"公共负面记录" - 任何司法记录都应大幅提升 PD 估计
5. 信用卡利用率 > 80% 提示流动性紧张

风险分层参考:
- 低风险  (PD < 5%):   征信干净, DTI < 50%, 查询合理, 收入稳定
- 中风险  (PD 5-15%):  轻微利用率偏高或近期查询略多, 但无实质逾期
- 高风险  (PD 15-40%): 出现 M1/M2 逾期, 或 DTI > 70%, 或近 3 月查询过多
- 极高风险(PD > 40%):  当前有逾期、M3+、公共记录、或多项叠加

严格按以下 JSON 格式输出,不要输出其他任何内容:
{
  "overdue_probability": <0-1 之间的浮点数,保留 4 位小数>,
  "risk_level": "<低风险|中风险|高风险|极高风险>",
  "key_drivers": [
    "<驱动因素1: 具体数据 + 影响方向>",
    "<驱动因素2>",
    "<驱动因素3>"
  ],
  "explanation": "<2-3 句话综合解释,先结论后理由>",
  "recommendation": "<批准|有条件批准|拒绝>",
  "suggested_credit_limit_ratio": <相对于申请额度的建议批复比例 0-1, 0 表示拒绝>
}"""


USER_PROMPT_TEMPLATE = """请评估以下征信报告:

{summary_text}

请输出 JSON 格式的风险评估结果。"""


# ============================================================
#  推理引擎
# ============================================================

class QwenInferenceEngine:
    """统一推理接口,支持 transformers / vllm 两种 backend"""

    def __init__(self, model_path: str, backend: str = "transformers",
                 max_new_tokens: int = 512, temperature: float = 0.1):
        self.model_path = model_path
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._init_backend()

    def _init_backend(self):
        if self.backend == "transformers":
            self._init_transformers()
        elif self.backend == "vllm":
            self._init_vllm()
        else:
            raise ValueError(f"未知 backend: {self.backend}")

    def _init_transformers(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[transformers] 加载模型: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print("[transformers] 模型加载完毕")

    def _init_vllm(self):
        from vllm import LLM, SamplingParams
        print(f"[vllm] 加载模型: {self.model_path}")
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            max_num_seqs=1,
            dtype="bfloat16",
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=0.9,
            max_tokens=self.max_new_tokens,
        )
        # tokenizer 用于构造 chat template
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        print("[vllm] 模型加载完毕")

    def _build_prompt(self, summary_text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(summary_text=summary_text)},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_batch(self, summary_texts: List[str]) -> List[str]:
        prompts = [self._build_prompt(t) for t in summary_texts]
        if self.backend == "vllm":
            outputs = self.llm.generate(prompts, self.sampling_params)
            return [o.outputs[0].text for o in outputs]
        else:
            return [self._generate_single_hf(p) for p in prompts]

    def _generate_single_hf(self, prompt: str) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = output_ids[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)


# ============================================================
#  解析 LLM 输出
# ============================================================

def parse_llm_json(raw: str) -> Dict[str, Any]:
    """从 LLM 输出中抽取 JSON,处理常见错误"""
    # 1) 优先匹配 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        json_str = m.group(1)
    else:
        # 2) 抓第一个完整的 { ... }
        m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL)
        if not m:
            raise ValueError(f"未找到 JSON: {raw[:200]}")
        json_str = m.group(0)

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败: {e}, 原文: {json_str[:200]}")

    # 字段校验与裁剪
    pd = float(result.get("overdue_probability", 0.0))
    pd = max(0.0, min(1.0, pd))
    result["overdue_probability"] = round(pd, 4)
    result.setdefault("risk_level", "未知")
    result.setdefault("key_drivers", [])
    result.setdefault("explanation", "")
    result.setdefault("recommendation", "未知")
    result.setdefault("suggested_credit_limit_ratio", 0.5)
    return result


# ============================================================
#  Mock 模式 (无 GPU 时演示用,基于规则的评分,模拟 LLM 输出)
# ============================================================

def mock_inference(summary_text: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """规则版评分,用于无 GPU 环境演示。逻辑模拟 LLM 的判断过程。"""
    pd = 0.02
    drivers = []

    # 逾期历史 - 最强信号
    if features["current_overdue_amount"] > 0:
        pd += 0.30
        drivers.append(f"当前存在逾期金额 {features['current_overdue_amount']:,} 元(强烈负面)")
    if features["max_overdue_severity"] not in ("无逾期",):
        sev = features["max_overdue_severity"]
        if "M1" in sev:
            pd += 0.05; drivers.append(f"24月内出现 {sev} 逾期")
        elif "M2" in sev:
            pd += 0.15; drivers.append(f"24月内出现 {sev} 逾期(显著负面)")
        elif "M3" in sev or "M4" in sev:
            pd += 0.30; drivers.append(f"24月内出现 {sev} 逾期(严重负面)")
        else:
            pd += 0.45; drivers.append(f"24月内出现 {sev}(极严重)")
    if features["recent_overdue_3m"] > 0:
        pd += 0.10 * min(features["recent_overdue_3m"], 3)
        drivers.append(f"最近 3 月有 {features['recent_overdue_3m']} 个账户发生逾期(近期恶化)")

    # DTI
    dti = features["dti"]
    if dti > 1.0:
        pd += 0.15
        drivers.append(f"负债收入比 {dti:.0%} 已超过月收入(严重现金流压力)")
    elif dti > 0.7:
        pd += 0.08
        drivers.append(f"负债收入比 {dti:.0%} 偏高")
    elif dti > 0.5:
        pd += 0.03
        drivers.append(f"负债收入比 {dti:.0%} 中等偏高")

    # 信用卡利用率
    util = features["credit_card_utilization"]
    if util > 0.9:
        pd += 0.08
        drivers.append(f"信用卡总体利用率 {util:.0%}(流动性紧张)")
    elif util > 0.7:
        pd += 0.04
        drivers.append(f"信用卡利用率 {util:.0%} 偏高")

    # 公共记录
    if features["civil_judgments"] > 0:
        pd += 0.20; drivers.append(f"存在 {features['civil_judgments']} 项民事判决")
    if features["enforcement_records"] > 0:
        pd += 0.25; drivers.append(f"存在 {features['enforcement_records']} 项强制执行记录(失信)")
    if features["tax_arrears"] > 0:
        pd += 0.10; drivers.append("存在税收欠缴记录")

    # 查询行为
    if features["inquiry_3m"] >= 8:
        pd += 0.08
        drivers.append(f"近 3 月被查询 {features['inquiry_3m']} 次(资金紧张信号)")
    elif features["inquiry_3m"] >= 5:
        pd += 0.04
        drivers.append(f"近 3 月被查询 {features['inquiry_3m']} 次(略多)")

    # 正面因素
    if features["overdue_count_24m"] == 0 and features["civil_judgments"] == 0:
        drivers.append("征信报告干净,无逾期及负面记录(正面)")
    if dti < 0.3:
        drivers.append(f"DTI {dti:.0%} 处于健康水平(正面)")

    pd = max(0.001, min(0.99, pd))

    if pd < 0.05:
        level, rec, ratio = "低风险", "批准", 1.0
    elif pd < 0.15:
        level, rec, ratio = "中风险", "批准", 0.8
    elif pd < 0.40:
        level, rec, ratio = "高风险", "有条件批准", 0.4
    else:
        level, rec, ratio = "极高风险", "拒绝", 0.0

    explanation = (
        f"根据 24 个月履约表现、{dti:.0%} 的负债收入比和近期 {features['inquiry_3m']} 次查询行为,"
        f"该客户被评估为{level}, 6 个月内逾期 30+ 概率约 {pd:.1%}。"
        f"建议{rec}。"
    )

    return {
        "overdue_probability": round(pd, 4),
        "risk_level": level,
        "key_drivers": drivers[:6],
        "explanation": explanation,
        "recommendation": rec,
        "suggested_credit_limit_ratio": ratio,
    }


# ============================================================
#  批处理主流程
# ============================================================

def process_one(report_path: Path, engine, mock: bool = False) -> Dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = build_summary_text(report)
    features = extract_features(report)

    if mock:
        result = mock_inference(summary, features)
    else:
        raw = engine.generate_batch([summary])[0]
        try:
            result = parse_llm_json(raw)
        except Exception as e:
            result = {
                "overdue_probability": -1, "risk_level": "解析失败",
                "key_drivers": [], "explanation": str(e),
                "recommendation": "需人工复核", "suggested_credit_limit_ratio": 0,
                "_raw_output": raw,
            }

    return {
        "report_id": report["report_id"],
        "true_label": report.get("_meta", {}).get("label_overdue_30d_in_6m"),
        "true_severity": report.get("_meta", {}).get("severity_tag"),
        "features": features,
        "model_output": result,
    }


def batch_inference_vllm(report_paths: List[Path], engine, batch_size: int = 32) -> List[Dict[str, Any]]:
    """vllm 模式: 真正的批量推理,加速 5-10x"""
    results = []
    for i in range(0, len(report_paths), batch_size):
        chunk = report_paths[i:i + batch_size]
        reports = [json.loads(p.read_text(encoding="utf-8")) for p in chunk]
        summaries = [build_summary_text(r) for r in reports]
        features_list = [extract_features(r) for r in reports]
        raws = engine.generate_batch(summaries)

        for r, feats, raw in zip(reports, features_list, raws):
            try:
                model_out = parse_llm_json(raw)
            except Exception as e:
                model_out = {"overdue_probability": -1, "risk_level": "解析失败",
                             "explanation": str(e), "_raw_output": raw}
            results.append({
                "report_id": r["report_id"],
                "true_label": r.get("_meta", {}).get("label_overdue_30d_in_6m"),
                "true_severity": r.get("_meta", {}).get("severity_tag"),
                "features": feats,
                "model_output": model_out,
            })
        print(f"  已处理 {min(i + batch_size, len(report_paths))}/{len(report_paths)}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="../data/samples")
    parser.add_argument("--output_dir", default="../output")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--backend", choices=["transformers","vllm","mock"], default="mock")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个,0 表示全部")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    report_files = sorted([p for p in in_dir.glob("report_*.json")])
    if args.limit > 0:
        report_files = report_files[:args.limit]
    print(f"共 {len(report_files)} 份报告待处理 [backend={args.backend}]")

    if args.backend == "mock":
        engine = None
    else:
        engine = QwenInferenceEngine(args.model_path, backend=args.backend)

    t0 = time.time()
    if args.backend == "vllm":
        results = batch_inference_vllm(report_files, engine, args.batch_size)
    else:
        results = []
        for i, p in enumerate(report_files, 1):
            r = process_one(p, engine, mock=(args.backend == "mock"))
            results.append(r)
            if i % 100 == 0:
                print(f"  已处理 {i}/{len(report_files)}, 耗时 {time.time()-t0:.1f}s")

    # 保存所有结果
    out_path = out_dir / "predictions.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✓ 完成! 总耗时 {time.time()-t0:.1f}s, 结果保存至 {out_path}")
    print(f"  平均每份: {(time.time()-t0)/len(report_files)*1000:.0f} ms")


if __name__ == "__main__":
    main()

