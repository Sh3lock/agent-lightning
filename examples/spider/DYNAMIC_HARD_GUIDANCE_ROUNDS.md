# Dynamic Hard Sampling + Write-only Guidance (Round-based)

本文档总结本次在 `Agent-lightning + VERL` 的 Spider Text-to-SQL 强化学习训练项目中，为“Hard 样本点火”引入的 **round-based** 方案相关代码改动与新增脚本（GRPO + Pass@k 改进背景）。

> 重要说明（当前实现覆盖范围）
>
> - 已实现：`write-only guidance` 注入（Agent 侧）、Proxy 侧基于 marker+CALL_TAG 的 token 级分段与预算控制（含超预算闭环截正文重算 suffix + 降级 raw + 诊断日志）、离线 hard mining / guidance 生成 / ignite 试跑 / raw_success_state 版本化输出。
> - 已接入：训练阶段已在 `agentlightning/verl/trainer.py` 中完成 **hard-only guidance、round-based p_guided 退火、per-sample 脱拐、组内一致性、stable sample_id 注入**。可选的“L2 成功后降 L1”的强度退火未默认启用（如需可再补）。

---

## 1. 设计目标与约束回顾

目标：提升 Hard 样本探索成功率（从 0 拉到 ε），并通过 round-based 退火逐步回归 raw 分布，且 **稳定/可控/可回滚**。

关键约束（本次实现严格遵循的部分）：

- 不改 `WRITE/CHECK/REWRITE` prompt 与 `table_info` 逻辑；不做 schema replace。
- 只做 **write-only guidance**：仅在 `write_query` 的那次 LLM 调用追加 guidance message；`check_query`/`rewrite_query` 不读 guidance、不追加 message。
- Proxy 侧 guidance 分段逻辑 **仅当 marker+CALL_TAG 同时命中** 才触发；否则一律 raw。
- 线上超预算处理：**禁止截 suffix**；只能截 guidance 正文并重算 suffix；迭代上限 + 可降级 raw + 诊断日志。
- baseline 一致性校验：Proxy 需要自检 vLLM 的 raw 截断方向一致后，才对 raw 显式设置 `truncate_prompt_tokens=4096`。
- tokenizer/模板可追踪：离线脚本与 Proxy 启动日志记录 `model_path / revision / chat_template_hash`。

---

## 2. 代码改动总览

### 2.1 Agent：write-only guidance 注入

文件：`examples/spider/sql_agent.py`

实现点：

1) LangGraph `State` 新增字段：

- `guidance: str`
- `guidance_level: str`（期望值：`L1` / `L2` / 空）

2) `LitSQLAgent.rollout()` 通过 `agent.invoke()` 的输入将 guidance 注入到 state：

- raw：`{\"question\": q, \"guidance\": \"\", \"guidance_level\": \"\"}`
- guided：`{\"question\": q, \"guidance\": body, \"guidance_level\": \"L1\"|\"L2\"}`

3) `SQLAgent.write_query()` 仅在 `guidance_level in {L1, L2}` 时 append 一条 `HumanMessage`，格式固定为三段行结构：

```\n[AGL_GUIDANCE_L1]  或 [AGL_GUIDANCE_L2]\nCALL_TAG=WRITE\n<guidance 正文>\n```

> 这条 message 只会出现在 WRITE 调用；CHECK/REWRITE 不受影响。

---

### 2.2 Proxy：marker+CALL_TAG gated 的 token 级分段 + 预算与降级

文件：`agentlightning/verl/daemon.py`（仅 `_start_proxy_server_v0` 的 Flask proxy 路径）

实现点：

#### 2.2.1 触发条件（严格）

仅当请求 `messages[-1].content` 同时满足：

- 第一行严格等于 `[AGL_GUIDANCE_L1]` 或 `[AGL_GUIDANCE_L2]`
- 第二行严格等于 `CALL_TAG=WRITE`

才进入 guided 分段逻辑；否则一律 raw 逻辑。

#### 2.2.2 suffix 的 token 口径与分段语义

命中 guided 时：

- `raw_messages = messages[:-1]`
- `guidance_message = messages[-1]`
- `raw_body_ids = tokenizer.apply_chat_template(raw_messages, add_generation_prompt=False, tokenize=True, tools=tools)`
- `full_ids = tokenizer.apply_chat_template(raw_messages + [guidance_message], add_generation_prompt=True, tokenize=True, tools=tools)`
- `suffix_ids = full_ids[len(raw_body_ids):]`

预算（suffix tokens 口径）：

- L1: `budget=128`
- L2: `budget=256`

若 `len(suffix_ids) <= budget`：

- 设置 `truncate_prompt_tokens = 4096 + len(suffix_ids)`
- 目标语义：让 vLLM 实际保留 `last4096(raw_body_ids) + suffix_ids`

#### 2.2.3 超预算处理（禁止截 suffix；只截正文并重算 suffix）

若 `suffix_len > budget`：

- 解析 guidance message content 的三段结构，只缩短第三段（正文），marker 与 CALL_TAG 行保持不变。
- 通过 `tokenizer.encode/decode` 对正文 token 做截断，并重建 content：
  - `content = marker + \"\\n\" + \"CALL_TAG=WRITE\" + \"\\n\" + new_body_text.strip()`
- 重算 `full_ids/suffix_ids`，循环直到：
  - `suffix_len <= budget`，或
  - 达到迭代上限（当前实现 `max_trim_iters=8`），或
  - 正文已空仍超预算

单调收敛保证：

- 每轮 `target_len` 严格下降：`target_len = min(int(target_len*0.7), target_len-1)`，确保不会卡死。

若正文清空仍超预算：

- 降级为 raw：`messages = raw_messages`，并记录诊断日志与计数 `n_oversize_degraded_to_raw`。

诊断日志包含：

- marker 档位（L1/L2）
- budget
- suffix_len_when_body_empty
- tools 是否启用、tools 数量
- call_tag、route
- （如可取）`x-rollout-id` / `x-attempt-id`

#### 2.2.4 baseline 自检：通过后才对 raw 显式 truncate=4096

raw 请求路径：

- 在 `raw_truncation_state.checked == False` 且满足可计算 token（非 stream，且 prompt 足够长）时：
  - 计算 `baseline_check_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, tools=tools)`
  - 先不设置 `truncate_prompt_tokens`
  - 等后端返回 `prompt_token_ids` 后进行对比：
    - 若 `prompt_token_ids == baseline_check_ids[-4096:]`，说明截断方向为 **tail**，自检通过：
      - `raw_truncation_state.enabled = True`
      - 后续 raw 请求统一设置 `truncate_prompt_tokens=4096`
    - 否则记录 `n_suffix_consistency_mismatch` 与错误日志（模式 head/unknown）

#### 2.2.5 统计计数器

当前在 proxy closure 内以 `stats` 维护并周期性输出（每 100 请求或每分钟）：

- `n_guided_write_seen`
- `n_suffix_oversize`
- `n_oversize_trimmed_to_fit`
- `n_oversize_degraded_to_raw`
- `n_suffix_consistency_mismatch`（raw baseline mismatch）

#### 2.2.6 tokenizer/模板版本追踪

Proxy 启动时记录：

- `model_path = tokenizer.name_or_path`
- `revision = tokenizer.revision 或 tokenizer.init_kwargs[\"revision\"]`
- `chat_template_hash = sha256(tokenizer.chat_template)`

---

## 3. 新增离线脚本（Round-based 工作流的产物生成）

目录：`examples/spider/scripts/`

### 3.1 Hard mining：raw-only K=8 全失败才标 HARD

脚本：`examples/spider/scripts/mine_hard_round.py`

核心规则：

- `sample_id = sha256(db_id + \"\\n\" + question)`（稳定、跨轮可对齐；不依赖 gold）
- raw prompt 下采样 `K=8`（默认）：
  - `max(reward) == 0` 才标记为 HARD（严格）
- hard 判定仅基于 raw；guided 不影响 hard 分桶。

输出：

1) `round_{r}_hard_samples.jsonl`：每条包含至少：

- `sample_id, db_id, question, pred, reward_stats`

（当前实现额外写入 `gold_query`，便于后续本地评估/排查；如不需要可后续移除。）

2) `raw_success_state.roundXXX.json`（版本化，供训练阶段只读使用）：

- `items[sample_id].ever_raw_success: bool`
- 可选 `first_success_round`

运行示例：

```bash\npython examples/spider/scripts/mine_hard_round.py \\\n  --input data/train_spider.parquet \\\n  --round 0 \\\n  --output-dir outputs/round0 \\\n  --k 8 \\\n  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \\\n  --tokenizer-model Qwen/Qwen2.5-Coder-1.5B-Instruct \\\n  --endpoint http://127.0.0.1:8000/v1\n```

> 注意：raw sampling 需要 `--endpoint` 或环境变量 `OPENAI_API_BASE`（脚本已显式校验）。

---

### 3.2 Guidance 生成：两档正文（不含 marker/call_tag）

脚本：`examples/spider/scripts/generate_guidance.py`

输入：

- `round_r_hard_samples.jsonl`

输出：

- `round_r_guidance.jsonl`（每条包含 `guidance_l1` / `guidance_l2` 的正文，不含 marker/call_tag）

约束：

- L1 正文 tokens ≤ 110
- L2 正文 tokens ≤ 240
- token 计数使用 **训练模型 tokenizer**（`--tokenizer-model/--tokenizer-revision`）
- 默认不提供 gold；可用 `--include-gold`（但强模型有严格指令：不得输出/复述 gold SQL）

运行示例：

```bash\npython examples/spider/scripts/generate_guidance.py \\\n  --hard-samples outputs/round0/round_0_hard_samples.jsonl \\\n  --output outputs/round0/round_0_guidance.jsonl \\\n  --tokenizer-model Qwen/Qwen2.5-Coder-1.5B-Instruct \\\n  --guidance-model gpt-4.1 \\\n  --guidance-endpoint $OPENAI_API_BASE\n```

实现细节：

- 若超限，会让强模型“重写更短”至 `--max-retries` 次；仍超限则做 token-level 兜底截断并输出（保证落盘可用）。

---

### 3.3 Ignite（可选）：0.5B 试跑判定 min_level

脚本：`examples/spider/scripts/ignite_guidance.py`

流程（对每条 HARD）：

- 用 L1 试跑 `K'=4`（默认 `--k 4`）
- 若仍 `max_reward==0`，再用 L2 试跑 `K'=4`
- 输出 `min_level ∈ {L1, L2, none}`，以及每档试跑 reward 统计

运行示例：

```bash\npython examples/spider/scripts/ignite_guidance.py \\\n  --hard-samples outputs/round0/round_0_hard_samples.jsonl \\\n  --guidance outputs/round0/round_0_guidance.jsonl \\\n  --output outputs/round0/round_0_ignite.jsonl \\\n  --round 0 \\\n  --k 4 \\\n  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \\\n  --tokenizer-model Qwen/Qwen2.5-Coder-1.5B-Instruct \\\n  --endpoint http://127.0.0.1:8000/v1\n```

> 注意：ignite sampling 同样需要 `--endpoint` 或 `OPENAI_API_BASE`。

---

## 4. 训练阶段接入（已落地）

当前训练阶段已补齐以下“调度/退火/组内一致性”逻辑（落地在 `agentlightning/verl/trainer.py`）：

1) **Round-based 混合训练**：对 HARD 样本按 `p_guided(round)` 采样 guided（Round0=1.0, Round1=0.5, Round2=0.25, Round3=0.0）。
2) **per-sample 脱拐**：一旦 `raw_success_state.items[sample_id].ever_raw_success==True`，该 sample 后续强制 `p_guided=0`（完全 raw）。
3) **GRPO 组内一致性**：同一 sample 的 rollout group 内所有 rollouts 使用同一 variant（raw 或 guided）与一致 guidance_level（L1/L2）。
4) **强度退火（可选）**：若某样本必须用 L2 点着，一旦 guided 成功后优先降到 L1 再尝试（当前未默认启用）。

建议接入点（后续工作方向）：

- 在数据构建/采样处（生成 `Task` 或 `task` dict 的地方）为每条样本写入：
  - `guidance` / `guidance_level`
  - 以及用于可追踪的 `sample_id`
- 将 `raw_success_state.roundXXX.json` 作为训练只读输入，避免在线写回造成漂移。

---

## 5. 验收标准对照表（已实现部分）

- Hard 判定：raw K=8 全失败才标 HARD（已实现于 `mine_hard_round.py`）。
- sample_id 稳定：sha256(db_id + \"\\n\" + question)（已实现于 `mine_hard_round.py`，ignite 复用一致逻辑）。
- write-only guidance：仅 `write_query` append message；check/rewrite 不动（已实现于 `sql_agent.py`）。
- Proxy 触发条件：marker+CALL_TAG 同时命中才 guided；否则 raw（已实现于 `daemon.py` v0 proxy）。
- 超预算处理：不截 suffix；截正文并重算 suffix；迭代上限；可降级 raw + 诊断日志（已实现于 `daemon.py`）。
- raw baseline 自检：通过后才对 raw 显式 truncate=4096（已实现于 `daemon.py`，仅当可做非 stream 且 prompt 足够长的请求触发一次自检）。
- tokenizer 信息可追踪：离线脚本与 Proxy 记录 model_path/revision/template hash（已实现）。
- 统计计数器：`n_guided_write_seen/n_suffix_oversize/n_oversize_trimmed_to_fit/n_oversize_degraded_to_raw`（已实现；另有 `n_suffix_consistency_mismatch`）。

---

## 6. 文件清单

- 修改：
  - `examples/spider/sql_agent.py`
  - `agentlightning/verl/daemon.py`
- 新增：
  - `examples/spider/scripts/mine_hard_round.py`
  - `examples/spider/scripts/generate_guidance.py`
  - `examples/spider/scripts/ignite_guidance.py`
  - `examples/spider/DYNAMIC_HARD_GUIDANCE_ROUNDS.md`（本文档）
