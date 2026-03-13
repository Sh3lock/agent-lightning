# Step-by-Step Runbook (Dynamic Hard + Write-only Guidance)

This document explains how to run the full workflow end-to-end:

1) Mine hard samples (raw-only)
2) Generate L1/L2 guidance bodies (offline, strong model)
3) (Optional) Ignite to choose `min_level` (L1/L2)
4) Train with round-based `p_guided` annealing + per-sample detachment + rollout-group consistency

All constraints remain: **no changes to WRITE/CHECK/REWRITE templates**, no schema replace, and **write-only guidance**.

---

## 0) Working Directory (important)

Use this example directory as your working directory so relative paths resolve correctly:

```bash
cd /home/storage/wenbinxing/ltf/passk/agent-lightning/examples/spider
```

---

## 1) Environment Setup

### 1.1 Activate your env

```bash
conda activate ltf_agent
```

### 1.2 Install required Python deps (if not already)

The Spider example needs LangGraph/LangChain + SQL utilities (see `README.md`):

```bash
pip install "langgraph<1.0" "langchain[openai]<1.0" "langchain-community" "langchain-text-splitters<1.0" sqlparse nltk
```

You also need Agent-lightning + VERL + vLLM dependencies installed in this env (project-specific; follow your existing setup).

---

## 2) Dataset / Paths (must exist)

This workflow expects Spider assets under `VERL_SPIDER_DATA_DIR` (default is `data/` under current directory).

### 2.1 Quick check your local files

You should have at least:

- `data/train_spider.parquet`
- `data/test_dev.parquet`
- `data/database/` (SQLite DBs)
- `data/test_database/` (SQLite DBs for val/test)

### 2.2 Set `VERL_SPIDER_DATA_DIR` (recommended)

```bash
export VERL_SPIDER_DATA_DIR="$(pwd)/data"
```

If you don’t set it, the scripts default to `"data"` relative to the current working directory.

---

## 3) Start an OpenAI-Compatible Endpoint for *raw mining* and *ignite*

`mine_hard_round.py` and `ignite_guidance.py` run the agent directly and need an OpenAI-compatible Chat Completions endpoint for the **training model**.

You have two common options:

### Option A: Use an existing OpenAI-compatible service

Set:

```bash
export OPENAI_API_BASE="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="dummy"
```

### Option B: Start a local vLLM OpenAI server (example)

If you use vLLM, start a server (example command; adjust to your GPU/model):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --port 8000 \
  --host 127.0.0.1
```

Then in another shell:

```bash
export OPENAI_API_BASE="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="dummy"
```

---

## 4) Round 0: Mine HARD (raw-only K=8)

Outputs:

- `outputs/round0/round_0_hard_samples.jsonl`
- `outputs/round0/raw_success_state.round000.json`

Command:

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

Notes:

- **Hard rule is strict:** HARD iff `max_reward==0` across K=8 raw rollouts.
- `sample_id` is stable: `sha256(db_id + "\n" + question)`.
- By default, hard samples output does **not** include `gold_query` (to reduce leak risk).
  - If you need it for offline debugging only: add `--dump-gold-for-debug`.

---

## 5) Round 0: Generate Guidance (offline strong model)

This step calls a **strong model** to produce guidance bodies only (no marker/call_tag).

### 5.1 Set strong model endpoint

If using a separate strong endpoint (could be OpenAI, hosted, etc):

```bash
export OPENAI_API_BASE="https://<your-strong-endpoint>/v1"
export OPENAI_API_KEY="<your-key>"
```

### 5.2 Run generation

Outputs:

- `outputs/round0/round_0_guidance.jsonl`

Command:

```bash
python scripts/generate_guidance.py \
  --hard-samples outputs/round0/round_0_hard_samples.jsonl \
  --output outputs/round0/round_0_guidance.jsonl \
  --tokenizer-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --guidance-model gpt-4.1 \
  --guidance-endpoint "$OPENAI_API_BASE"
```

Notes:

- Token limits are enforced using the **training tokenizer**:
  - L1 body ≤ 110 tokens
  - L2 body ≤ 240 tokens
- Default is **no gold in context**.

---

## 6) (Optional) Round 0: Ignite to choose `min_level`

Ignite quickly tries L1 then L2 on each hard sample (K′=4 each) and outputs `min_level`.

Outputs:

- `outputs/round0/round_0_ignite.jsonl`

Command (example with a smaller model as the ignite model):

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

## 7) Train Round 0 (raw/guided mix + annealing + detachment)

### 7.1 What training will do automatically

When you pass `--round/--hard-samples/--guidance/...`, the training script:

- Loads offline artifacts and passes them into the trainer config (`agentlightning_guidance`).
- Computes the same stable `sample_id` in training.
- Applies:
  - **hard-only guided** (non-hard always raw)
  - **round-based** `p_guided` schedule (or `--p-guided` override)
  - **per-sample detachment**: if `ever_raw_success==True`, force raw forever (read-only state)
  - **rollout group consistency**: one decision per `sample_id`, cached for all its rollouts
- Switches to **legacy `fit_v0`** automatically when guidance is enabled, so the daemon uses the v0 proxy path where guided prompt segmentation/truncation logic is implemented.

### 7.2 Run training

```bash
python train_sql_agent.py qwen \
  --round 0 \
  --hard-samples outputs/round0/round_0_hard_samples.jsonl \
  --guidance outputs/round0/round_0_guidance.jsonl \
  --ignite outputs/round0/round_0_ignite.jsonl \
  --raw-success-state outputs/round0/raw_success_state.round000.json
```

Optional: override annealing probability:

```bash
python train_sql_agent.py qwen \
  --round 0 \
  --p-guided 0.8 \
  --hard-samples outputs/round0/round_0_hard_samples.jsonl \
  --guidance outputs/round0/round_0_guidance.jsonl \
  --raw-success-state outputs/round0/raw_success_state.round000.json
```

---

## 8) What to Watch in Logs (quick sanity)

### 8.1 Training-side metrics/counters

Training logs should include `guidance/*` counters (e.g.):

- `guidance/n_guided_selected`
- `guidance/n_force_raw_success`
- `guidance/n_force_raw_non_hard`
- `guidance/n_force_raw_p_guided`
- `guidance/n_force_raw_missing_guidance`

### 8.2 Proxy-side baseline gate + guided stats

The proxy logs periodic stats including (examples):

- `n_guided_write_seen`
- `n_guided_blocked_no_baseline` (guided blocked until baseline check succeeds)
- `n_guided_blocked_baseline_mismatch` (baseline mismatch -> guided disabled)
- `n_suffix_oversize`, `n_oversize_trimmed_to_fit`, `n_oversize_degraded_to_raw`

If guided is always blocked, ensure:

- you have at least one **raw** request with a prompt longer than 4096 tokens (non-stream) so the baseline check can run;
- the backend vLLM returns `prompt_token_ids` and its truncation matches the expected **tail** direction.

---

## 9) Round 1/2/3 Loop

Repeat for each round:

1) Mine hard with the *current* checkpoint/model serving endpoint (raw-only K=8)
2) Generate guidance (strong model)
3) (Optional) Ignite
4) Train with `--round r` and the matching artifact paths

Default `p_guided` schedule is:

- Round0=1.0
- Round1=0.5
- Round2=0.25
- Round3=0.0

You can override via `--p-guided`.

---

## 10) Common Failure Modes

- **Path errors:** run from `examples/spider/` or use absolute paths for `data/*`.
- **Missing endpoint:** `mine_hard_round.py`/`ignite_guidance.py` require `--endpoint` or `OPENAI_API_BASE`.
- **Guided blocked forever:** baseline self-check never triggers (need a long raw request) or baseline mismatch (non-tail).
- **GPU OOM:** reduce batch size / rollout n / model size, or use the provided smaller configs.

