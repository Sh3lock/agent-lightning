# Spider Config2 Repro Guide

This document describes the maintained entry points for reproducing the Spider `GRPO`, `Pass@k`, and `Pass@k + Guided` experiments on a new machine after cloning this repository.

## Scope

This guide only covers the maintained config2 workflow:

- `run_spider_config2_train_matrix.sh`
- `run_spider_config2_eval_matrix.sh`
- `run_eval_dual_gpu_and_report.sh`
- `examples/spider/run_round0_mid_pipeline.sh`

Historical scripts under the repo root with older machine-specific paths are not the recommended entry points for future runs.

## What Stays Local

Do not commit the following:

- checkpoints
- training logs
- eval matrix outputs
- wandb local runs
- temporary Ray directories
- generated figures / csv files

The repository `.gitignore` is configured to exclude these artifacts.

## Required Inputs

Before running, provide:

1. Spider data under `examples/spider/data/` or an equivalent compatible layout.
2. A model path or model ID through:
   - `SPIDER_MODEL_PATH=/abs/path/to/Qwen2.5-Coder-0.5B-Instruct`
   - or a valid HuggingFace model ID reachable by the environment.
3. An artifact directory for logs / ckpts:
   - `ARTIFACT_ROOT=/abs/path/to/output_root`

Optional:

- `PYTHON_BIN=python3`
- `RAY_TMPDIR_BASE=/abs/path/to/tmp`
- `SPIDER_N_RUNNERS=24`
- `ROLLOUT_N=8`
- `SPIDER_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=12`
- `SPIDER_ROLLOUT_GPU_MEMORY_UTILIZATION=0.7`

## Train

Run the three-way comparison:

```bash
export SPIDER_MODEL_PATH=/abs/path/to/Qwen2.5-Coder-0.5B-Instruct
export ARTIFACT_ROOT=$PWD/.artifacts
export RAY_TMPDIR_BASE=$ARTIFACT_ROOT/tmp

MODE=full RUN_SET=grpo,passk,guided RUN_PARALLEL=1 \
GPU_GRPO=0 GPU_PASSK=1 GPU_GUIDED=2 \
bash run_spider_config2_train_matrix.sh
```

Outputs go under:

- `$ARTIFACT_ROOT/examples/spider/log/`
- `$ARTIFACT_ROOT/examples/spider/ckpt/`
- `$ARTIFACT_ROOT/examples/spider/ray/`

## Round0 Pipeline for Guided

The guided workflow uses:

- hard sample mining
- offline guidance generation
- optional ignite decisions

Run:

```bash
export SPIDER_MODEL_PATH=/abs/path/to/Qwen2.5-Coder-0.5B-Instruct
export STRONG_API_BASE=https://<provider>/v1
export STRONG_API_KEY=<your-key>

bash examples/spider/run_round0_mid_pipeline.sh
```

Outputs go under:

- `examples/spider/outputs/round0_full/`

## Eval

After training, evaluate:

```bash
export SPIDER_MODEL_PATH=/abs/path/to/Qwen2.5-Coder-0.5B-Instruct
export ARTIFACT_ROOT=$PWD/.artifacts
export GRPO_CKPT=/abs/path/to/grpo/best
export PASSK_CKPT=/abs/path/to/passk/best
export GUIDED_CKPT=/abs/path/to/guided/best

GPU_EVAL=0 bash run_spider_config2_eval_matrix.sh
```

Outputs go under:

- `$ARTIFACT_ROOT/examples/spider/eval_matrix/`

## Auto Report Helper

To generate a markdown report for Pass@k and Guided evals:

```bash
export PASSK_CKPT=/abs/path/to/passk/best
export GUIDED_CKPT=/abs/path/to/guided/best
export ARTIFACT_ROOT=$PWD/.artifacts

bash run_eval_dual_gpu_and_report.sh
```

The report is written to:

- `$ARTIFACT_ROOT/eval_reports/eval_status.md`

## Portable Path Rules

For future changes, keep these rules:

1. Never commit machine-specific absolute paths in maintained config2 scripts.
2. Prefer environment variables over hard-coded local paths.
3. Keep outputs under `ARTIFACT_ROOT`.
4. Keep model resolution through `SPIDER_MODEL_PATH` when using local weights.
