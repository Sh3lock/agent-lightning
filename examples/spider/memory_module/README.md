## Memento Memory Module (Spider Example)

This directory hosts memory-enhancement components that are only activated when `MEMENTO_ENABLE=1`.

### Skeletonizer Safety
- The SQL skeletonizer replaces table names, column names, and literals with placeholders (e.g., `_tab1`, `_col1`, `_val_num`).
- If `sqlglot` is missing or SQL parsing fails, it returns `failed=True` and an empty skeleton string instead of the original SQL.

These rules prevent cross-database leakage by ensuring no real identifiers survive in the skeleton output.

### CaseBank Safety
- The CaseBank exposes a `skeleton_only` policy that never returns database-specific cases.
- Tiered retrieval prefers same-DB cases but falls back to skeleton cases only when allowed by policy.

### Lazy Initialization
- Vector stores and embedding models are initialized lazily: runtime creation is lightweight and does not load embedding models.

### Scoring Semantics
- Retrieval scores are cosine similarity (L2-normalized dot product); higher is always more similar.

### Error Fix Bank
- Error normalization maps raw errors to stable `error_type` labels and extracts relevant entities.
- Fix hints are retrieved with the same cosine similarity scoring and filtered by `error_type`/`dialect`.

### Static Validator
- A lightweight validator flags missing tables/columns and ambiguous references using schema text.
- When enabled, validation can force `THE QUERY IS INCORRECT.` to trigger rewrite.

### CaseBank Persistence
- Set `MEMENTO_CASEBANK_DIR=/path/to/casebanks` to load persisted indexes at runtime.
- A `manifest.json` is written alongside the indexes with embedder metadata, similarity definition, build split, and counts.
- Runtime will warn (or fail with `MEMENTO_CASEBANK_STRICT=1`) if the manifest is incompatible.
- Build indexes offline via:
  - `python examples/spider/scripts/build_casebanks.py --input data/train_spider.parquet --split train --workers 4`
  - `python examples/spider/scripts/build_casebanks.py --input data/test_dev.parquet --split dev --workers 4`
- Split filtering is enforced at runtime via `MEMENTO_CASEBANK_ALLOW_SPLIT` (default: `train`).

### Evaluation/Smoke
- `python examples/spider/scripts/eval_sql_agent.py --input data/test_dev.parquet --mode baseline`
- `python examples/spider/scripts/eval_sql_agent.py --input data/test_dev.parquet --mode memento`
- `python examples/spider/scripts/train_smoke.py --input data/train_spider.parquet --limit 20`
