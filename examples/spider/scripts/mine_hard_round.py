#!/usr/bin/env python
"""Mine hard samples for a training round using raw-only rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
SPIDER_DIR = SCRIPT_DIR.parent
sys.path.append(str(SPIDER_DIR))

from sql_agent import SQLAgent, evaluate_query  # noqa: E402

logger = logging.getLogger(__name__)


def stable_sample_id(db_id: str, question: str) -> str:
    payload = f"{db_id}\n{question}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_tokenizer(model_path: str, revision: Optional[str]) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, trust_remote_code=True)
    return tokenizer


def _log_tokenizer_info(tokenizer: AutoTokenizer) -> None:
    model_path = getattr(tokenizer, "name_or_path", "unknown")
    revision = getattr(tokenizer, "revision", None)
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if revision is None and isinstance(init_kwargs, dict):
        revision = init_kwargs.get("revision")
    chat_template = getattr(tokenizer, "chat_template", None)
    chat_template_hash = (
        hashlib.sha256(chat_template.encode("utf-8")).hexdigest() if chat_template is not None else "unknown"
    )
    logger.info(
        "Tokenizer: model_path=%s revision=%s chat_template_hash=%s",
        model_path,
        revision,
        chat_template_hash,
    )


def _iter_records(df: pd.DataFrame, limit: int) -> Iterable[Dict[str, Any]]:
    records = df.to_dict(orient="records")
    if limit > 0:
        return records[:limit]
    return records


def _sample_id_from_record(record: Dict[str, Any], db_id_field: str, question_field: str) -> Optional[str]:
    db_id = record.get(db_id_field)
    question = record.get(question_field)
    if not db_id or not question:
        return None
    return stable_sample_id(str(db_id), str(question))


def _resolve_db_path(spider_dir: Path, db_id: str, use_test_split: bool) -> Path:
    base = spider_dir / ("test_database" if use_test_split else "database")
    return base / db_id / f"{db_id}.sqlite"


def _load_schema(db_path: Path) -> str:
    schema_path = db_path.parent / "schema.sql"
    if schema_path.exists():
        return schema_path.read_text(encoding="utf-8")
    return "No schema available."


def _run_rollout(
    *,
    question: str,
    ground_truth: str,
    db_path: Path,
    schema: str,
    model: str,
    endpoint: Optional[str],
    temperature: float,
    max_turns: int,
    table_info_truncate: int,
    execution_truncate: int,
) -> tuple[str, float]:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_db = Path(temp_dir) / db_path.name
            shutil.copyfile(db_path, temp_db)
            agent = SQLAgent(
                f"sqlite:///{temp_db}",
                max_turns=max_turns,
                table_info_truncate=table_info_truncate,
                execution_truncate=execution_truncate,
                debug=False,
                db_schema=schema,
                endpoint=endpoint,
                verl_replacement={"model": model, "temperature": temperature},
            ).graph()
            result = agent.invoke(
                {"question": question, "guidance": "", "guidance_level": ""},
                {"recursion_limit": 100},
            )
            query = result.get("query", "") if isinstance(result, dict) else ""
            reward = evaluate_query(query, ground_truth, str(temp_db), raise_on_error=False)
        return query, reward
    except Exception as exc:
        logger.error("Rollout failed: %s", exc)
        return "", 0.0


def _process_sample(
    *,
    sample: Dict[str, Any],
    k: int,
    spider_dir: Path,
    use_test_split: bool,
    model: str,
    endpoint: Optional[str],
    temperature: float,
    max_turns: int,
    table_info_truncate: int,
    execution_truncate: int,
    question_field: str,
    db_id_field: str,
    query_field: str,
) -> Dict[str, Any]:
    question = sample[question_field]
    db_id = sample[db_id_field]
    ground_truth = sample[query_field]
    sample_id = stable_sample_id(db_id, question)

    db_path = _resolve_db_path(spider_dir, db_id, use_test_split)
    if not db_path.exists():
        logger.error("Database missing for sample_id=%s: %s", sample_id, db_path)
        rewards = [0.0] * k
        preds = [""] * k
    else:
        schema = _load_schema(db_path)
        rewards = []
        preds = []
        for _ in range(k):
            pred, reward = _run_rollout(
                question=question,
                ground_truth=ground_truth,
                db_path=db_path,
                schema=schema,
                model=model,
                endpoint=endpoint,
                temperature=temperature,
                max_turns=max_turns,
                table_info_truncate=table_info_truncate,
                execution_truncate=execution_truncate,
            )
            preds.append(pred)
            rewards.append(float(reward))

    max_reward = max(rewards) if rewards else 0.0
    best_idx = rewards.index(max_reward) if rewards else 0
    hard = max_reward <= 0.0
    return {
        "sample_id": sample_id,
        "db_id": db_id,
        "question": question,
        "gold_query": ground_truth,
        "pred": preds[best_idx] if preds else "",
        "reward_stats": {"rewards": rewards, "max_reward": max_reward},
        "hard": hard,
        "ever_raw_success": max_reward > 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to train parquet file.")
    parser.add_argument("--round", type=int, required=True, help="Round index for output versioning.")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument("--k", type=int, default=8, help="Number of raw samples per item.")
    parser.add_argument("--model", required=True, help="Model name for raw sampling.")
    parser.add_argument("--endpoint", default=None, help="OpenAI-compatible API base.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--max-turns", type=int, default=3, help="Max agent turns.")
    parser.add_argument("--table-info-truncate", type=int, default=2048, help="Table info truncate length.")
    parser.add_argument("--execution-truncate", type=int, default=2048, help="Execution result truncate length.")
    parser.add_argument("--num-samples", type=int, default=-1, help="Limit number of samples.")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers.")
    parser.add_argument("--use-test-split", action="store_true", help="Use test_database instead of database.")
    parser.add_argument("--prev-raw-success", default=None, help="Path to previous raw_success_state file.")
    parser.add_argument("--question-field", default="question", help="Field name for question.")
    parser.add_argument("--db-id-field", default="db_id", help="Field name for db id.")
    parser.add_argument("--query-field", default="query", help="Field name for gold query.")
    parser.add_argument("--tokenizer-model", required=True, help="Model path for tokenizer logging.")
    parser.add_argument("--tokenizer-revision", default=None, help="Tokenizer revision.")
    parser.add_argument(
        "--dump-gold-for-debug",
        action="store_true",
        help="Include gold SQL in hard samples output (debug only).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing raw_success_state and hard samples in output dir.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    endpoint = args.endpoint or os.environ.get("OPENAI_API_BASE")
    if not endpoint:
        raise ValueError("Missing --endpoint and OPENAI_API_BASE; required for raw sampling.")

    tokenizer = _load_tokenizer(args.tokenizer_model, args.tokenizer_revision)
    _log_tokenizer_info(tokenizer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hard_output = output_dir / f"round_{args.round}_hard_samples.jsonl"
    state_output = output_dir / f"raw_success_state.round{args.round:03d}.json"

    prev_items: Dict[str, Dict[str, Any]] = {}
    if args.resume and state_output.exists():
        prev_data = json.loads(state_output.read_text(encoding="utf-8"))
        prev_items = prev_data.get("items", {})
        logger.info("Resume enabled: loaded %s processed samples", len(prev_items))
    elif args.prev_raw_success:
        prev_path = Path(args.prev_raw_success)
        if prev_path.exists():
            prev_data = json.loads(prev_path.read_text(encoding="utf-8"))
            prev_items = prev_data.get("items", {})

    df = pd.read_parquet(args.input)
    samples = list(_iter_records(df, args.num_samples))
    if args.resume and prev_items:
        filtered = []
        for record in samples:
            sample_id = _sample_id_from_record(record, args.db_id_field, args.question_field)
            if sample_id is None or sample_id not in prev_items:
                filtered.append(record)
        samples = filtered
        logger.info("Pending samples after resume filter: %s", len(samples))
    total = len(samples)
    logger.info("Loaded %s samples", total)

    items: Dict[str, Dict[str, Any]] = dict(prev_items)
    processed = 0
    hard_count = 0
    spider_dir = Path(os.environ.get("VERL_SPIDER_DATA_DIR", "data"))

    def handle_result(result: Dict[str, Any], file_handle) -> None:
        nonlocal processed, hard_count
        processed += 1
        sample_id = result["sample_id"]
        entry = items.get(sample_id, {"ever_raw_success": False})
        if result["ever_raw_success"]:
            if not entry.get("ever_raw_success"):
                entry["ever_raw_success"] = True
                entry["first_success_round"] = args.round
        items[sample_id] = entry
        if result["hard"]:
            hard_count += 1
            record = {
                "sample_id": result["sample_id"],
                "db_id": result["db_id"],
                "question": result["question"],
                "pred": result["pred"],
                "reward_stats": result["reward_stats"],
            }
            if args.dump_gold_for_debug:
                record["gold_query"] = result["gold_query"]
            file_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        if processed % 50 == 0 or processed == total:
            logger.info("Processed %s/%s samples (hard=%s)", processed, total, hard_count)

    open_mode = "a" if args.resume and hard_output.exists() else "w"
    with hard_output.open(open_mode, encoding="utf-8") as output_handle:
        if args.num_workers <= 1:
            for sample in samples:
                result = _process_sample(
                    sample=sample,
                    k=args.k,
                    spider_dir=spider_dir,
                    use_test_split=args.use_test_split,
                    model=args.model,
                    endpoint=endpoint,
                    temperature=args.temperature,
                    max_turns=args.max_turns,
                    table_info_truncate=args.table_info_truncate,
                    execution_truncate=args.execution_truncate,
                    question_field=args.question_field,
                    db_id_field=args.db_id_field,
                    query_field=args.query_field,
                )
                handle_result(result, output_handle)
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [
                    executor.submit(
                        _process_sample,
                        sample=sample,
                        k=args.k,
                        spider_dir=spider_dir,
                        use_test_split=args.use_test_split,
                        model=args.model,
                        endpoint=args.endpoint,
                        temperature=args.temperature,
                        max_turns=args.max_turns,
                        table_info_truncate=args.table_info_truncate,
                        execution_truncate=args.execution_truncate,
                        question_field=args.question_field,
                        db_id_field=args.db_id_field,
                        query_field=args.query_field,
                    )
                    for sample in samples
                ]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        handle_result(result, output_handle)
                    except Exception as exc:
                        logger.error("Sample processing failed: %s", exc)

    state_payload = {
        "round": args.round,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "items": items,
    }
    state_output.write_text(json.dumps(state_payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s hard samples to %s", hard_count, hard_output)
    logger.info("Wrote raw_success_state to %s", state_output)


if __name__ == "__main__":
    main()
