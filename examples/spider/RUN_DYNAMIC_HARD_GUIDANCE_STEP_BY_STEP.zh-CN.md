# 动态 Hard + Write-only Guidance 运行手册（逐步）

本文档说明如何把整个流程从头到尾跑通：

1) 挖掘 Hard 样本（raw-only）
2) 离线生成两档 guidance 正文（L1/L2，强模型）
3)（可选）Ignite 试跑选择 `min_level`（L1/L2）
4) 训练：round-based `p_guided` 退火 + per-sample 脱拐 + rollout group 一致性

所有约束保持不变：**不改 WRITE/CHECK/REWRITE 模板文本**、不做 schema replace、且只允许 **write-only guidance**。

---

## 0) 工作目录（重要）

建议把工作目录切到 Spider example 下，确保相对路径都能正确解析：

```bash
cd /home/storage/wenbinxing/ltf/passk/agent-lightning/examples/spider
```

---

## 1) 环境准备

### 1.1 激活环境

```bash
conda activate ltf_agent
```

### 1.2 安装必要依赖（如果你的环境还没有）

Spider example 需要 LangGraph/LangChain + SQL 相关库（参考 `README.md`）：

```bash
pip install "langgraph<1.0" "langchain[openai]<1.0" "langchain-community" "langchain-text-splitters<1.0" sqlparse nltk
```

此外你还需要确保当前环境已经安装了 Agent-lightning + VERL + vLLM 等训练依赖（通常你的工程环境里已处理；如未安装，请沿用项目已有安装方式）。

---

## 2) 数据集与路径（必须存在）

本流程要求 Spider 数据位于 `VERL_SPIDER_DATA_DIR` 指向的目录中（默认使用当前目录下的 `data/`）。

### 2.1 快速检查数据文件

至少需要：

- `data/train_spider.parquet`
- `data/test_dev.parquet`
- `data/database/`（训练用 SQLite DB）
- `data/test_database/`（验证/测试用 SQLite DB）

### 2.2 设置 `VERL_SPIDER_DATA_DIR`（强烈推荐）

```bash
export VERL_SPIDER_DATA_DIR="$(pwd)/data"
```

如果不设置，该变量默认值为 `"data"`（相对当前工作目录）。

---

## 3) 启动 OpenAI 兼容 Endpoint（用于 raw mining 和 ignite）

`mine_hard_round.py` 与 `ignite_guidance.py` 会直接调用 agent，需要一个 OpenAI 兼容的 Chat Completions endpoint 来服务 **训练模型**。

你可以选择两种方式：

### 方式 A：使用已有的 OpenAI 兼容服务

设置：

```bash
export OPENAI_API_BASE="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="dummy"
```

### 方式 B：本地启动 vLLM OpenAI server（示例）

如果你使用 vLLM，可启动一个 server（示例命令，按你的 GPU/模型调整）：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --port 8000 \
  --host 127.0.0.1
```

然后在另一个 shell 里：

```bash
export OPENAI_API_BASE="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="dummy"
```

---

## 4) Round 0：挖掘 HARD（raw-only K=8）

产物：

- `outputs/round0/round_0_hard_samples.jsonl`
- `outputs/round0/raw_success_state.round000.json`

命令：

```bash
python scripts/mine_hard_round.py \
  --input data/train_spider.parquet \
  --round 0 \
  --output-dir outputs/round0 \
  --k 8 \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --tokenizer-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --endpoint "$OPENAI_API_BASE"
```

说明：

- **Hard 规则严格**：仅当 K=8 次 raw rollout 全失败（`max_reward==0`）才标 HARD。
- `sample_id` 稳定：`sha256(db_id + "\n" + question)`。
- 默认 hard_samples 输出 **不包含** `gold_query`（降低泄漏风险）。
  - 如你仅用于离线排查，额外加：`--dump-gold-for-debug`。

---

## 5) Round 0：生成 Guidance（离线强模型）

该步骤调用 **强模型** 生成 guidance 正文（不包含 marker/call_tag）。

### 5.1 设置强模型 endpoint

如果你用独立的强模型服务（OpenAI/自托管/其他），请设置：

```bash
export OPENAI_API_BASE="https://<your-strong-endpoint>/v1"
export OPENAI_API_KEY="<your-key>"
```

### 5.2 执行生成

产物：

- `outputs/round0/round_0_guidance.jsonl`

命令：

```bash
python scripts/generate_guidance.py \
  --hard-samples outputs/round0/round_0_hard_samples.jsonl \
  --output outputs/round0/round_0_guidance.jsonl \
  --tokenizer-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --guidance-model gpt-4.1 \
  --guidance-endpoint "$OPENAI_API_BASE"
```

说明：

- token 上限用 **训练模型 tokenizer** 计数：
  - L1 正文 ≤ 110 tokens
  - L2 正文 ≤ 240 tokens
- 默认输入上下文 **不提供 gold**（更干净）。

---

## 6)（可选）Round 0：Ignite 试跑选择 `min_level`

Ignite 会对每条 hard 样本先用 L1 再用 L2（每档 K′=4）快速试跑，然后输出 `min_level`。

产物：

- `outputs/round0/round_0_ignite.jsonl`

命令（示例：用更小的 0.5B 模型作为 ignite 模型）：

```bash
python scripts/ignite_guidance.py \
  --hard-samples outputs/round0/round_0_hard_samples.jsonl \
  --guidance outputs/round0/round_0_guidance.jsonl \
  --output outputs/round0/round_0_ignite.jsonl \
  --round 0 \
  --k 4 \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --tokenizer-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --endpoint "$OPENAI_API_BASE"
```

---

## 7) 训练 Round 0（raw/guided 混合 + 退火 + 脱拐）

### 7.1 训练会自动做什么

当你传入 `--round/--hard-samples/--guidance/...` 后，训练脚本会：

- 加载离线产物，并将配置传入 trainer（`agentlightning_guidance`）。
- 在训练侧计算同样的稳定 `sample_id`（与离线一致）。
- 在训练阶段自动应用：
  - **仅 HARD 才允许 guided**（非 hard 一律 raw）
  - **round-based** `p_guided`（或 `--p-guided` 覆盖）
  - **per-sample 脱拐**：若 `ever_raw_success==True`，永远强制 raw（只读，不写回文件）
  - **rollout group 一致性**：对每个 `sample_id` 做一次决策并缓存，同组所有 rollouts 一致
- 当 guidance 启用时，训练脚本会自动切换到 **legacy `fit_v0`**，从而使用 daemon 的 v0 proxy 路径（本项目的 guided prompt 分段/预算/超预算处理逻辑就是实现于 v0 proxy）。

### 7.2 启动训练

```bash
python train_sql_agent.py qwen \
  --round 0 \
  --hard-samples outputs/round0/round_0_hard_samples.jsonl \
  --guidance outputs/round0/round_0_guidance.jsonl \
  --ignite outputs/round0/round_0_ignite.jsonl \
  --raw-success-state outputs/round0/raw_success_state.round000.json
```

可选：覆盖退火概率：

```bash
python train_sql_agent.py qwen \
  --round 0 \
  --p-guided 0.8 \
  --hard-samples outputs/round0/round_0_hard_samples.jsonl \
  --guidance outputs/round0/round_0_guidance.jsonl \
  --raw-success-state outputs/round0/raw_success_state.round000.json
```

---

## 8) 日志中需要重点关注的点（快速验收）

### 8.1 训练侧 metrics / 计数器

训练日志中应出现 `guidance/*` 指标（例）：

- `guidance/n_guided_selected`
- `guidance/n_force_raw_success`
- `guidance/n_force_raw_non_hard`
- `guidance/n_force_raw_p_guided`
- `guidance/n_force_raw_missing_guidance`

### 8.2 Proxy 侧 baseline 门控 + guided 统计

proxy 会周期性输出 stats（例）：

- `n_guided_write_seen`
- `n_guided_blocked_no_baseline`（baseline 自检未完成前，guided 会被阻塞并降级 raw）
- `n_guided_blocked_baseline_mismatch`（baseline 不匹配，guided 会被禁用并降级 raw）
- `n_suffix_oversize`, `n_oversize_trimmed_to_fit`, `n_oversize_degraded_to_raw`

如果你发现 guided 一直被阻塞，请确认：

- 至少出现过一次 **足够长的 raw** 请求（prompt > 4096，且非 stream），这样 baseline 自检才能触发；
- 你的 vLLM 后端会返回 `prompt_token_ids`，并且其截断方向与预期一致（必须是 **tail**）。

---

## 9) Round 1/2/3 循环

每一轮建议按相同顺序执行：

1) 用当前 checkpoint/模型服务 endpoint 做 raw-only K=8 hard mining
2) 生成 guidance（强模型）
3)（可选）ignite
4) 用 `--round r` 以及对应产物路径启动训练

默认 `p_guided` 退火策略为：

- Round0=1.0
- Round1=0.5
- Round2=0.25
- Round3=0.0

也可用 `--p-guided` 覆盖。

---

## 10) 常见问题排查

- **路径错误**：建议从 `examples/spider/` 目录执行，或把 `data/*` 换成绝对路径。
- **endpoint 未设置**：`mine_hard_round.py`/`ignite_guidance.py` 需要 `--endpoint` 或 `OPENAI_API_BASE`。
- **guided 一直 blocked**：baseline 自检未触发（需要长 raw prompt），或 baseline mismatch（非 tail）。
- **GPU OOM**：降低 batch size / rollout n / 模型尺寸，或使用更小的配置。

