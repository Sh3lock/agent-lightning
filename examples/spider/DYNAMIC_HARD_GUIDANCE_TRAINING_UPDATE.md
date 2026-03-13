# Training-Stage Guidance Integration Update

This document summarizes the **second wave** of changes that complete the training-stage integration for the round-based dynamic hard sampling + write-only guidance workflow.

---

## Scope of This Update

- Training-stage integration: stable `sample_id`, per-sample guidance decisions, hard-only guidance, round-based `p_guided` annealing, per-sample raw-success detachment, and rollout-group consistency.
- Training CLI wiring to load offline artifacts (hard/guidance/ignite/raw_success_state) and pass round config into the trainer.
- Proxy gating: guided truncation now blocked until raw baseline self-check passes.
- Hard-mining output tightened: `gold_query` suppressed by default (debug-only flag).
- Documentation updated to mark training-stage completion.

---

## Files Changed

- `agentlightning/verl/trainer.py`
  - Added stable `sample_id` computation and training-side guidance decision logic.
  - Integrated round-based `p_guided` schedule, per-sample detachment, and rollout-group consistency.
  - Injected `guidance/guidance_level/sample_id` into the batch before rollout.
- `examples/spider/train_sql_agent.py`
  - New CLI flags to point to round artifacts and configure `p_guided`.
  - Config wiring via `agentlightning_guidance`.
- `agentlightning/verl/daemon.py`
  - Guided prompt segmentation now gated on raw baseline self-check.
  - Guided requests degrade to raw if baseline is unverified or mismatched.
- `examples/spider/scripts/mine_hard_round.py`
  - `gold_query` output is now opt-in via `--dump-gold-for-debug`.
- `examples/spider/DYNAMIC_HARD_GUIDANCE_ROUNDS.md`
  - Updated to mark training-stage integration as completed.

---

## Training Integration Details

### 1) Stable `sample_id` (required)

Implemented in `agentlightning/verl/trainer.py`:

```python
def _stable_sample_id(db_id: str, question: str) -> str:
    payload = f"{db_id}\n{question}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
```

- Matches `mine_hard_round.py` exactly: `sha256(db_id + "\n" + question)` (UTF-8).
- `sample_id` is injected into the batch alongside `guidance` and `guidance_level`.

### 2) Training reads offline artifacts and applies round-based annealing

Trainer consumes `agentlightning_guidance` from config and loads:

- `round` (int)
- `hard_samples` (jsonl)
- `guidance` (jsonl)
- `ignite` (jsonl; optional)
- `raw_success_state` (json; read-only)
- `p_guided` override (optional)

Default round schedule:

- Round0 = 1.0
- Round1 = 0.5
- Round2 = 0.25
- Round3 = 0.0

If `p_guided` is provided, it overrides the round schedule.

### 3) Per-sample detachment (required)

If `raw_success_state.items[sample_id].ever_raw_success == True`, the trainer forces:

- `guidance = ""`
- `guidance_level = ""`

This is **read-only**; the trainer does not write back to the file.

### 4) Rollout group consistency (required)

Guidance decisions are made **once per sample_id** and cached:

- Every rollout in the same group gets the same `guidance` and `guidance_level`.
- Deterministic selection is based on `sample_id` hash prefix (no per-attempt randomness).

### 5) Guidance level selection (required)

Selection logic (minimal, controlled):

- If `ignite.min_level == "L2"` -> use L2.
- Else -> use L1.
- If ignite missing -> default to L1.
- If guidance body missing -> force raw.

---

## Training CLI + Config Wiring

`examples/spider/train_sql_agent.py` adds:

- `--round`
- `--hard-samples`
- `--guidance`
- `--ignite` (optional)
- `--raw-success-state`
- `--p-guided` (override)

These populate `config["agentlightning_guidance"]`, which is read by the trainer.

---

## Proxy Baseline Gate (robustness)

Guided truncation now requires raw baseline self-check to be **validated**:

- If baseline is not checked, guided requests degrade to raw.
- If baseline mismatch (non-tail), guided requests degrade to raw.
- When guided is blocked and baseline is not checked, proxy attempts a baseline check on the raw path.

---

## Hard-Mining Output: `gold_query` Suppressed by Default

`examples/spider/scripts/mine_hard_round.py` no longer writes gold SQL unless:

```
--dump-gold-for-debug
```

This reduces leak risk and keeps guidance generation clean by default.

---

## Minimal Training Command

```bash
python examples/spider/train_sql_agent.py qwen \
  --round 0 \
  --hard-samples outputs/round0/round_0_hard_samples.jsonl \
  --guidance outputs/round0/round_0_guidance.jsonl \
  --ignite outputs/round0/round_0_ignite.jsonl \
  --raw-success-state outputs/round0/raw_success_state.round000.json
```

Optional override:

```
--p-guided 0.5
```

---

## Notes / Remaining Optional Work

- The optional strength annealing rule (L2 success -> try L1 next) is **not** enabled here.
- All constraints remain: no prompt/template changes, write-only guidance, no rewrite/check guidance, no schema replace.

