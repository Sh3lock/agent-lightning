import argparse
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sql_agent import (
    SQLAgent,
    _maybe_init_memento_runtime,
    evaluate_query,
    read_memento_config,
)


def _score_summary(scores: List[float]) -> Dict[str, Any]:
    if not scores:
        return {"count": 0}
    ordered = sorted(scores)
    count = len(ordered)
    mean = sum(ordered) / count
    p50_idx = int(0.5 * (count - 1))
    p90_idx = int(0.9 * (count - 1))
    return {
        "count": count,
        "mean": mean,
        "min": ordered[0],
        "max": ordered[-1],
        "p50": ordered[p50_idx],
        "p90": ordered[p90_idx],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SQLAgent on Spider parquet.")
    parser.add_argument("--input", required=True, help="Parquet file (dev/test).")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--mode", choices=["baseline", "memento"], default="baseline")
    parser.add_argument("--output_dir", default="examples/spider/artifacts")
    parser.add_argument("--use_test_db", action="store_true", default=True)
    return parser.parse_args()


def _resolve_db_path(spider_dir: Path, db_id: str, use_test_db: bool) -> Path:
    if use_test_db:
        path = spider_dir / "test_database" / db_id / f"{db_id}.sqlite"
    else:
        path = spider_dir / "database" / db_id / f"{db_id}.sqlite"
    if path.exists():
        return path
    alt = spider_dir / "database" / db_id / f"{db_id}.sqlite"
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

    if args.mode == "baseline":
        os.environ["MEMENTO_ENABLE"] = "0"
    else:
        os.environ["MEMENTO_ENABLE"] = "1"
        os.environ.setdefault("MEMENTO_EVAL_POLICY", "tiered")

    df = pd.read_parquet(args.input)
    if args.limit > 0:
        df = df.head(args.limit)
    rows = df.to_dict(orient="records")

    metrics: Dict[str, Any] = defaultdict(int)
    validation_counts: Counter[str] = Counter()
    retrieval_counts: Counter[str] = Counter()
    specific_scores: List[float] = []
    skeleton_scores: List[float] = []
    llm_correct_count = 0
    validator_override_count = 0
    rewrite_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})
    prompt_lengths: List[int] = []
    output_lengths: List[int] = []
    turns: List[int] = []

    results: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        question = row["question"]
        db_id = row["db_id"]
        ground_truth = row["query"]
        db_path = _resolve_db_path(spider_dir, db_id, args.use_test_db)
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
                agent.memento_policy = memento_config.eval_policy
                agent.memento_runtime = _maybe_init_memento_runtime(memento_config)

            graph = agent.graph()
            state = graph.invoke(  # type: ignore
                {"question": question},
                {"recursion_limit": 100},
            )

            reward = evaluate_query(state["query"], ground_truth, str(temp_db), raise_on_error=False)
            metrics["total"] += 1
            metrics["exec_correct"] += int(reward == 1.0)

            turns.append(state.get("num_turns", 0))
            prompt_lengths.append(state.get("prompt_table_info_chars", 0))
            output_lengths.append(len(state.get("query", "")))

            validation = state.get("validation_error")
            if validation and not validation.get("ok", True):
                validation_counts[validation.get("error_type", "Unknown")] += 1
            llm_feedback = state.get("llm_feedback_raw", "")
            if "THE QUERY IS CORRECT" in llm_feedback:
                llm_correct_count += 1
                if validation and not validation.get("ok", True):
                    validator_override_count += 1

            retrieval = state.get("memento_retrieval", {})
            retrieval_counts[retrieval.get("result_type", "none")] += 1
            specific_score = retrieval.get("specific_score")
            if isinstance(specific_score, (int, float)):
                specific_scores.append(float(specific_score))
            skeleton_score = retrieval.get("skeleton_score")
            if isinstance(skeleton_score, (int, float)):
                skeleton_scores.append(float(skeleton_score))

            normalized_error_type = state.get("normalized_error_type")
            if normalized_error_type:
                rewrite_stats[normalized_error_type]["total"] += 1
                if reward == 1.0:
                    rewrite_stats[normalized_error_type]["success"] += 1

            results.append(
                {
                    "idx": idx,
                    "db_id": db_id,
                    "reward": reward,
                    "num_turns": state.get("num_turns", 0),
                    "query": state.get("query", ""),
                    "memento_retrieval": retrieval,
                    "validation_error": validation,
                    "prompt_table_info_chars": state.get("prompt_table_info_chars", 0),
                    "query_chars": len(state.get("query", "")),
                    "normalized_error_type": normalized_error_type,
                }
            )

    exec_at_1 = metrics["exec_correct"] / max(metrics["total"], 1)
    summary = {
        "mode": args.mode,
        "total": metrics["total"],
        "exec_at_1": exec_at_1,
        "avg_turns": sum(turns) / max(len(turns), 1),
        "avg_prompt_chars": sum(prompt_lengths) / max(len(prompt_lengths), 1),
        "avg_query_chars": sum(output_lengths) / max(len(output_lengths), 1),
        "validation_counts": dict(validation_counts),
        "static_validator_override_rate": (
            validator_override_count / max(llm_correct_count, 1)
        ),
        "static_validator_override_count": validator_override_count,
        "llm_correct_count": llm_correct_count,
        "retrieval_counts": dict(retrieval_counts),
        "best_score": {
            "specific": _score_summary(specific_scores),
            "skeleton": _score_summary(skeleton_scores),
        },
        "rewrite_success_by_error_type": {
            error_type: {
                "total": counts["total"],
                "success": counts["success"],
                "success_rate": counts["success"] / max(counts["total"], 1),
            }
            for error_type, counts in rewrite_stats.items()
        },
        "env": {
            "MEMENTO_ENABLE": os.environ.get("MEMENTO_ENABLE", ""),
            "MEMENTO_EVAL_POLICY": os.environ.get("MEMENTO_EVAL_POLICY", ""),
            "MODEL": os.environ.get("MODEL", ""),
            "OPENAI_API_BASE": os.environ.get("OPENAI_API_BASE", ""),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"eval_summary_{args.mode}_{stamp}.json"
    details_path = output_dir / f"eval_details_{args.mode}_{stamp}.jsonl"
    summary_csv = output_dir / f"eval_summary_{args.mode}_{stamp}.csv"
    details_csv = output_dir / f"eval_details_{args.mode}_{stamp}.csv"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with details_path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    pd.DataFrame(results).to_csv(details_csv, index=False)
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {summary_path}")
    print(f"[saved] {details_path}")
    print(f"[saved] {summary_csv}")
    print(f"[saved] {details_csv}")


if __name__ == "__main__":
    main()
