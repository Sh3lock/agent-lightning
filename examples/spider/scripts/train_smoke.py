import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sql_agent import SQLAgent, _maybe_init_memento_runtime, evaluate_query, read_memento_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train-mode smoke for SQLAgent (no RL training).")
    parser.add_argument("--input", required=True, help="Train parquet.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output_dir", default="examples/spider/artifacts")
    return parser.parse_args()


def _resolve_db_path(spider_dir: Path, db_id: str) -> Path:
    path = spider_dir / "database" / db_id / f"{db_id}.sqlite"
    if path.exists():
        return path
    alt = spider_dir / "test_database" / db_id / f"{db_id}.sqlite"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Database not found for {db_id}")


def _load_schema(db_path: Path) -> str:
    schema_path = db_path.parent / "schema.sql"
    if schema_path.exists():
        return schema_path.read_text(encoding="utf-8")
    return "No schema available."


def main() -> None:
    args = _parse_args()
    spider_dir = Path(os.environ.get("VERL_SPIDER_DATA_DIR", "data"))

    os.environ["MEMENTO_ENABLE"] = "1"
    os.environ.setdefault("MEMENTO_TRAIN_POLICY", "skeleton_only")
    os.environ.setdefault("MEMENTO_EVAL_POLICY", "tiered")

    # Ensure train entrypoint is importable (smoke for compatibility).
    import train_sql_agent as _train_sql_agent  # noqa: F401

    df = pd.read_parquet(args.input)
    if args.limit > 0:
        df = df.head(args.limit)
    rows = df.to_dict(orient="records")

    retrieval_counts: Counter[str] = Counter()
    metrics: Dict[str, Any] = {"total": 0, "exec_correct": 0}
    leakage_detected = False

    for row in rows:
        question = row["question"]
        db_id = row["db_id"]
        ground_truth = row["query"]
        db_path = _resolve_db_path(spider_dir, db_id)
        schema = _load_schema(db_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_db = Path(temp_dir) / db_path.name
            shutil.copyfile(db_path, temp_db)

            agent = SQLAgent(
                f"sqlite:///{temp_db}",
                db_schema=schema,
            )
            agent.db_id = db_id
            memento_config = read_memento_config()
            agent.memento_config = memento_config
            if memento_config.enable:
                agent.memento_policy = memento_config.train_policy
                agent.memento_runtime = _maybe_init_memento_runtime(memento_config)

            graph = agent.graph()
            state = graph.invoke(  # type: ignore
                {"question": question},
                {"recursion_limit": 100},
            )

            reward = evaluate_query(state["query"], ground_truth, str(temp_db), raise_on_error=False)
            metrics["total"] += 1
            metrics["exec_correct"] += int(reward == 1.0)
            retrieval = state.get("memento_retrieval", {})
            retrieval_counts[retrieval.get("result_type", "none")] += 1
            if retrieval.get("result_type") == "specific":
                leakage_detected = True

    summary = {
        "mode": "train_smoke",
        "total": metrics["total"],
        "exec_at_1": metrics["exec_correct"] / max(metrics["total"], 1),
        "retrieval_counts": dict(retrieval_counts),
        "leakage_detected": leakage_detected,
        "env": {
            "MEMENTO_ENABLE": os.environ.get("MEMENTO_ENABLE", ""),
            "MEMENTO_TRAIN_POLICY": os.environ.get("MEMENTO_TRAIN_POLICY", ""),
            "MODEL": os.environ.get("MODEL", ""),
            "OPENAI_API_BASE": os.environ.get("OPENAI_API_BASE", ""),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"train_smoke_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
