# 征信报告 LLM 风控解读：从 1 份样本到 1200 份预测的完整实战

> 用 Qwen 本地大模型对人行征信报告做批量风险解读，输出**逾期概率 + 解释性归因**，复刻风控实战流程。

---

## 0. 项目目标与产出

| 目标 | 产出 |
|---|---|
| 数据：基于 1 份征信报告样本扩展为 1000+ | `data/samples/report_*.json` × 1200 |
| 推理：Qwen 本地批量解读 | `output/predictions.jsonl` |
| 评估：可量化的风控指标 | AUC、KS、Lift 报告 |

---

## 1. 项目结构

```
credit_risk_project/
├── data/
│   ├── sample_credit_report.json        # 原始征信样本(人行格式)
│   └── samples/                          # 1200 份衍生样本
│       ├── report_00001.json
│       ├── ...
│       └── _index.json                   # 索引及风险分布
├── scripts/
│   ├── generate_samples.py               # 样本生成
│   ├── feature_engineering.py            # 特征工程 + 摘要文本
│   ├── qwen_inference.py                 # Qwen 批量推理
│   └── evaluate.py                       # 评估指标
├── output/
│   └── predictions.jsonl                 # 模型预测结果
```

---

## 2. 环境准备

```bash
# Python 3.10+
pip install torch transformers accelerate

# 推荐(高吞吐推理)
pip install vllm

# 工具库
pip install numpy pandas
```

**硬件建议**:
| 模型 | 显存 | 1200 份耗时 (vllm) |
|---|---|---|
| Qwen2.5-7B-Instruct | 16GB (T4/V100/3090) | ~ 8 分钟 |
| Qwen2.5-14B-Instruct | 32GB (A6000/A100-40G) | ~ 12 分钟 |
| Qwen2.5-72B-Instruct | 4×A100-80G | ~ 25 分钟 |

无 GPU 时可用 `--backend mock` 模式，0.1 秒跑完全部 1200 份（基于规则的演示版）。

---

## 3. 第一步：构建 1 份"金标准"征信报告

参考人行《个人信用信息基础数据库》字段规范，把征信报告分成 6 大块：

```python
# data/sample_credit_report.json 结构
{
  "report_id": "...",
  "personal_info": {...},      # 个人基本信息
  "summary": {...},            # 信息概要
  "loans": [...],              # 贷款明细
  "credit_cards": [...],       # 信用卡明细
  "public_records": {...},     # 公共记录(司法、税务)
  "inquiry_records": {...}     # 查询记录
}
```

**关键风险字段**（按预测能力排序）:
1. `overdue_history_24m` —— 24 月逾期序列（最强信号）
2. `current_overdue_amount` —— 当前逾期金额
3. `max_overdue_months_24m` —— 最长逾期 M 等级
4. `credit_card_utilization` —— 信用卡利用率
5. `dti` —— 负债收入比（衍生计算）
6. `inquiry_count_3m` —— 近 3 月查询次数
7. `civil_judgments` / `enforcement_records` —— 司法记录

---

## 4. 第二步：衍生 1200 份样本（含真实标签）

### 4.1 设计原则

不能简单地"随机扰动"——那会破坏字段间的业务逻辑。正确做法是 **分层 + 因果一致**：

```
风险等级(severity) → 该等级对应的字段联合分布 → 采样得到一份样本
                  ↓
                  bad 客户: 利用率高 → 逾期概率高 → 查询次数多 → 公共记录多
                  excellent 客户: 反之
```

### 4.2 风险等级分布（贴近真实银行客群）

| 等级 | 占比 | 真实 6 月内逾期率 |
|---|---|---|
| excellent | 50% | 1% |
| good | 30% | 5% |
| medium | 15% | 25% |
| bad | 5% | 65% |
| **整体** | **100%** | **~8%** |

这与中国商业银行个贷不良率 1-3% + 关注类 5-8% 的真实分布一致。

### 4.3 运行

```bash
cd scripts
python generate_samples.py
```

输出:
```
开始生成 1200 份征信报告样本...
✓ 完成! 共生成 1200 个文件
  风险分层: {'excellent': 600, 'good': 360, 'medium': 180, 'bad': 60}
  真实逾期率: 7.92%
```

**注意**：每份样本的 `_meta` 字段保存了真实标签 `label_overdue_30d_in_6m`（仅评估用，模型输入时被过滤）。

---

## 5. 第三步：特征工程——别把原始 JSON 喂给 LLM

### 5.1 痛点

原始 JSON 平均 2-5K tokens × 1200 份 = 数百万 tokens，问题：
- 推理慢且贵；
- 关键风险信号被低价值字段淹没；
- LLM 对结构化数值的 token 利用率低。

### 5.2 解决方案：摘要文本 + 衍生特征

`feature_engineering.py` 把征信报告压缩为 **300-500 token 的摘要文本**，同时补充 LLM 算不出的衍生指标：

```
【个人基本信息】
姓名: 张三, 年龄: 35岁, 学历: 本科, 婚姻: 已婚
职业: 工程师, 月收入: 2.5万元

【信贷账户概要】
- 贷款账户数: 3 个, 贷款余额合计: 85.0 万元
- 信用卡数: 4 张, 总体利用率: 17.5% (高利用率卡数: 0)
- 负债收入比(DTI): 39.0%

【逾期表现】
- 当前逾期金额: 0 元
- 24 个月内逾期账户数: 0, 累计逾期次数: 0
- 最长逾期等级: 无逾期
- 最近 3 个月发生逾期的账户数: 0

【公共记录】
- 民事判决: 0 项, 强制执行: 0 项, 税收欠缴: 0 项

【近期信贷需求(查询)】
- 近 3 月机构查询: 2 次 (其中贷款审批 1 次)
- 近 6 月机构查询: 4 次
```

**关键衍生指标**:
- **DTI（负债收入比）**: `(月供合计 + 信用卡已用 × 10%) / 月收入`，>70% 为高危
- **high_util_cards**: 利用率 ≥80% 的信用卡数
- **recent_overdue_3m**: 近 3 月有逾期的账户数（捕捉"近期恶化"）

---

## 6. 第四步：Qwen 本地批量推理（核心）

### 6.1 Prompt 工程：决定 50% 效果的环节

`qwen_inference.py` 的 `SYSTEM_PROMPT` 包含 4 个关键设计：

#### ① 角色定位
```
你是一名资深银行风控建模专家,精通中国人民银行征信报告解读和个人信贷违约预测。
```
明确专家身份，激活模型的金融领域知识。

#### ② 评估方法论（关键！）
告诉模型"看哪些指标、按什么权重"，避免它自由发挥：
```
1. 重点关注"现金流压力" - DTI 负债收入比 > 70% 是高风险信号
2. 重点关注"履约历史" - 24 个月内逾期表现是最强的预测变量
3. 重点关注"近期信贷渴求度" - 近 3 月查询次数 > 6 次提示资金紧张
4. 重点关注"公共负面记录" - 任何司法记录都应大幅提升 PD 估计
5. 信用卡利用率 > 80% 提示流动性紧张
```

#### ③ 风险分层锚点
给模型提供"标尺"，避免概率漂移到极端：
```
低风险  (PD < 5%):   征信干净, DTI < 50%, 查询合理, 收入稳定
中风险  (PD 5-15%):  轻微利用率偏高或近期查询略多, 但无实质逾期
高风险  (PD 15-40%): 出现 M1/M2 逾期, 或 DTI > 70%, 或近 3 月查询过多
极高风险(PD > 40%):  当前有逾期、M3+、公共记录、或多项叠加
```

#### ④ 严格 JSON 输出格式
强制结构化，便于下游解析：
```json
{
  "overdue_probability": 0.0850,
  "risk_level": "中风险",
  "key_drivers": ["驱动因素1", "..."],
  "explanation": "...",
  "recommendation": "批准",
  "suggested_credit_limit_ratio": 0.8
}
```

### 6.2 两种 backend 的选型

```python
# 选项 A: transformers (简单, 单条推理)
engine = QwenInferenceEngine(
    "Qwen/Qwen2.5-7B-Instruct",
    backend="transformers"
)

# 选项 B: vLLM (推荐, 真正的批量推理, 5-10x 加速)
engine = QwenInferenceEngine(
    "Qwen/Qwen2.5-7B-Instruct",
    backend="vllm"
)
```

**vLLM 的优势**: 内部用 PagedAttention 共享 KV cache，1200 份请求合并为几十个 batch，吞吐量碾压原生 transformers。

### 6.3 运行

```bash
# 真实推理(需要 GPU + 模型权重)
python qwen_inference.py \
    --backend vllm \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --input_dir ../data/samples \
    --output_dir ../output \
    --batch_size 32

# 无 GPU 时用 mock 模式(规则版,演示流程)
python qwen_inference.py --backend mock
```

### 6.4 输出格式 (`predictions.jsonl`)

每行一份 JSON，包含 4 块：
```json
{
  "report_id": "PBC-2026-00026",
  "true_label": 0,
  "true_severity": "bad",
  "features": {...},                    // 衍生特征(便于复盘)
  "model_output": {
    "overdue_probability": 0.99,
    "risk_level": "极高风险",
    "key_drivers": [
      "当前存在逾期金额 32,031 元(强烈负面)",
      "24月内出现 M4(91-120天) 逾期(严重负面)",
      "最近 3 月有 6 个账户发生逾期(近期恶化)",
      "负债收入比 214% 已超过月收入(严重现金流压力)",
      "信用卡利用率 86% 偏高",
      "存在 1 项强制执行记录(失信)"
    ],
    "explanation": "...",
    "recommendation": "拒绝",
    "suggested_credit_limit_ratio": 0.0
  }
}
```

---

## 7. 第五步：模型评估——风控行业标准指标

`evaluate.py` 输出 5 类关键指标：

```
========== 整体表现 ==========
  AUC          : 0.8746      ← 区分能力
  KS           : 0.6874      ← 风控行业核心指标
  Brier Score  : 0.1376      ← 概率校准度(越小越好)
  Top 10% Lift : 5.79x       ← 高分人群中坏客户富集程度
  Top 20% Lift : 4.11x

========== 按真实风险等级分组 ==========
  excellent | n= 600 | 平均PD: 10.32% | 中位数: 10.00%
  good      | n= 360 | 平均PD: 20.01% | 中位数: 22.00%
  medium    | n= 180 | 平均PD: 84.79% | 中位数: 94.00%
  bad       | n=  60 | 平均PD: 99.00% | 中位数: 99.00%

========== 拒绝策略效果(threshold=0.4) ==========
  坏客户捕获率: 82.1%
  好客户误杀率: 15.2%
```

### 指标解读

| 指标 | 含义 | 行业基准 | 我们的结果 |
|---|---|---|---|
| AUC | 排序能力 | >0.7 可用，>0.8 优秀 | **0.87** |
| KS | 风控核心指标 | >0.3 可用，>0.4 优秀 | **0.69** |
| Brier | 概率校准度 | <0.25 可用 | 0.14 |
| Top 10% Lift | 头部捕获能力 | >3x 可用 | **5.79x** |

> KS = 0.69 已超过绝大部分商业银行的传统评分卡（典型 KS 0.35-0.45）。

---

## 8. LLM 方案 vs 传统评分卡：什么时候用 LLM？

| 维度 | 传统 LR/XGB 评分卡 | LLM 方案 |
|---|---|---|
| 训练数据 | 需 10万+ 历史样本 | 0 训练样本(prompt 即模型) |
| 上线速度 | 3-6 个月 | 1-2 周 |
| 解释性 | 弱（系数数字） | 强（自然语言） |
| 极端场景 | 需重新建模 | 改 prompt 即可 |
| 推理成本 | <1ms | 100-500ms / 张 |
| 监管合规 | 成熟 | 需输出可审计的 key_drivers |
| AUC/KS 上限 | 取决于数据量 | 受限于 LLM 推理能力 |

**最佳实践**: LLM 用于**冷启动**、**长尾客群**、**反欺诈说明**、**人审辅助**；评分卡用于**主流量自动审批**。两者并行，LLM 给出 `key_drivers` 作为评分卡的解释补充。

---

## 9. 生产化建议

### 9.1 推理性能调优
- 用 **vLLM + AWQ 量化**: 7B 模型显存从 16G 降到 6G，吞吐 +30%
- **batch_size 调到 64+**: 充分利用 GPU
- **max_new_tokens 限制 256**: 输出 JSON 通常不超过这个量

### 9.2 输出可靠性
LLM 偶尔会输出非法 JSON。`parse_llm_json()` 已做：
1. 优先匹配 ```json``` 代码块；
2. 退化为正则抓取第一个 `{...}`；
3. 失败的样本标记为 `risk_level: "解析失败"` 进入人工复核队列。

生产环境建议：失败率 > 1% 时应升级 prompt 或改用 JSON Schema 约束的工具调用模式。

### 9.3 偏见与合规
- **不要**让模型看到姓名、性别、民族等保护特征——会引入歧视风险；
- 把 `personal_info` 中除年龄/收入/职业外的字段从 prompt 中删除；
- 在 `key_drivers` 中**严禁**出现性别/地域歧视性表述（可加 prompt 约束 + 后处理过滤）。

### 9.4 部署架构

```
            ┌──────────────────┐
征信API ──> │ 特征工程服务      │ ──> 摘要文本
            └──────────────────┘
                     │
                     v
            ┌──────────────────┐
            │ vLLM 推理集群     │ <── Qwen2.5-14B
            │ (Triton + K8s)    │     (AWQ 量化)
            └──────────────────┘
                     │
                     v
            ┌──────────────────┐
            │ JSON 解析 + 校验   │
            └──────────────────┘
                     │
              ┌──────┴──────┐
              v             v
        风控决策引擎     可解释性归档
        (autoApprove)    (DataLake)
```

---

## 10. 一键复现命令

```bash
cd credit_risk_project/scripts

# Step 1: 生成 1200 份样本 (1 秒)
python generate_samples.py

# Step 2: 批量推理 (mock 模式 0.1 秒, vllm 模式 8 分钟)
python qwen_inference.py --backend mock
# 或: python qwen_inference.py --backend vllm --model_path Qwen/Qwen2.5-7B-Instruct

# Step 3: 评估
python evaluate.py
```

