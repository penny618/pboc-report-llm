"""
input_manifest.py
把"喂给 Qwen 的输入"钉成可审计指纹。

背景:
    推理时的 Qwen 输入 = 静态 SYSTEM_PROMPT + USER_PROMPT_TEMPLATE.format(现算摘要)。
    摘要由 feature_engineering.build_summary_text 从 data/samples 现算，无随机 / 时间 /
    locale 依赖，因此给定样本即确定。但 predictions.jsonl 未持久化该输入，无法离线核对。

本脚本对每份样本重算完整 Qwen 输入并记录 sha256，输出:
    - output/input_manifest.jsonl  : 每行 {report_id, file, input_sha256}
    - 控制台 : 静态提示词哈希 + 整集输入指纹(对全部 report_id+hash 再取一次 sha256)

用途:
    任何人重跑本脚本，只要整集指纹一致，即证明"喂给模型的输入"逐字节相同，
    与推理在何时、用哪台机器无关。

    python input_manifest.py
"""

import hashlib
import json
from pathlib import Path

from feature_engineering import build_summary_text
from qwen_inference import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def full_input(summary_text: str) -> str:
    """与 qwen_inference._build_messages 一致的完整输入文本(不含模型侧 chat 模板包裹)。"""
    return SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE.format(summary_text=summary_text)


def main():
    root = Path(__file__).resolve().parent.parent
    samples = sorted((root / "data" / "samples").glob("report_*.json"))

    sys_hash = sha256(SYSTEM_PROMPT)
    tmpl_hash = sha256(USER_PROMPT_TEMPLATE)

    entries = []
    for p in samples:
        report = json.loads(p.read_text(encoding="utf-8"))
        h = sha256(full_input(build_summary_text(report)))
        entries.append({
            "report_id": report["report_id"],
            "file": p.name,
            "input_sha256": h,
        })

    entries.sort(key=lambda e: e["report_id"])
    out = root / "output" / "input_manifest.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # 整集输入指纹:对 "report_id:hash" 拼接串再取一次 sha256
    fingerprint = sha256("\n".join(f"{e['report_id']}:{e['input_sha256']}" for e in entries))

    print(f"样本数              : {len(entries)}")
    print(f"SYSTEM_PROMPT sha256: {sys_hash}")
    print(f"USER_TEMPLATE sha256: {tmpl_hash}")
    print(f"整集输入指纹        : {fingerprint}")
    print(f"已写入 {out.relative_to(root)}")


if __name__ == "__main__":
    main()
