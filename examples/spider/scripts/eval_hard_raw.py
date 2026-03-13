#!/usr/bin/env python
"""Evaluate raw success on hard samples (no guidance)."""

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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
SPIDER_DIR = SCRIPT_DIR.parent
sys.path.append(str(SPIDER_DIR))

from sql_agent import SQLAgent, evaluate_query  # noqa: E402

logger = logging.getLogger(__name__)


def stable_sample_id(db_id: str, question: str) -> str:
    payload = f"{db_id}\n{question}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_db_path(spider_dir: Path, db_id: str, use_test_split: bool) -> Path:
    base = spider_dir / ("test_database" if use_test_split else "database")
    return base / db_id / f"{db_id}.sqlite"


def _load_schema(db_path: Path) -> str:
    schema_path = db_path.parent / "schema.sql"
    if schema_path.exists():
        return schema_path.read_text(encoding="utf-8")
    return "No schema available."


def _get_gold_query(record: Dict[str, Any]) -> str:
    for key in ("gold_query", "query", "gold"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _run_rollout(
    *,
    question: str,
    ground_truth: str,
    db_path: Path,
    schema: str,
    model: str,
    endpoint: str,
    temperature: float,
    max_turns: int,
    table_info_truncate: int,
    execution_truncate: int,
) -> Tuple[str, float]:
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


def _iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_existing_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    existing_ids = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = record.get("sample_id")
            if sample_id:
                existing_ids.add(sample_id)
    return existing_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-samples", required=True, help="Path to round_r_hard_samples.jsonl.")
    parser.add_argument("--output", required=True, help="Output JSONL path for raw eval results.")
    parser.add_argument("--k", type=int, default=8, help="Number of raw samples per item.")
    parser.add_argument("--model", required=True, help="Model name for raw sampling.")
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible API base.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--max-turns", type=int, default=3, help="Max agent turns.")
    parser.add_argument("--table-info-truncate", type=int, default=2048, help="Table info truncate length.")
    parser.add_argument("--execution-truncate", type=int, default=2048, help="Execution result truncate length.")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel workers.")
    parser.add_argument("--use-test-split", action="store_true", help="Use test_database instead of database.")
    parser.add_argument("--resume", action="store_true", help="Skip samples already present in output.")
    parser.add_argument("--log-every", type=int, default=50, help="Log progress every N samples.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    hard_path = Path(args.hard_samples)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = _load_existing_ids(output_path) if args.resume else set()
    records = list(_iter_records(hard_path))
    if existing_ids:
        records = [r for r in records if r.get("sample_id") not in existing_ids]
        logger.info("Resume enabled: pending samples %s", len(records))

    spider_dir = SPIDER_DIR
    total = len(records)
    logger.info("Loaded %s hard samples", total)

    success_count = 0
    processed = 0

    def process_record(record: Dict[str, Any]) -> Dict[str, Any]:
        db_id = record.get("db_id", "")
        question = record.get("question", "")
        ground_truth = _get_gold_query(record)
        sample_id = record.get("sample_id") or stable_sample_id(str(db_id), str(question))
        if not db_id or not question or not ground_truth:
            return {
                "sample_id": sample_id,
                "db_id": db_id,
                "question": question,
                "rewards": [],
                "max_reward": 0.0,
                "error": "missing_required_fields",
            }

        db_path = _resolve_db_path(spider_dir, str(db_id), args.use_test_split)
        schema = _load_schema(db_path)
        rewards: List[float] = []
        preds: List[str] = []
        for _ in range(args.k):
            pred, reward = _run_rollout(
                question=str(question),
                ground_truth=str(ground_truth),
                db_path=db_path,
                schema=schema,
                model=args.model,
                endpoint=args.endpoint,
                temperature=args.temperature,
                max_turns=args.max_turns,
                table_info_truncate=args.table_info_truncate,
                execution_truncate=args.execution_truncate,
            )
            preds.append(pred)
            rewards.append(float(reward))
        max_reward = max(rewards) if rewards else 0.0
        return {
            "sample_id": sample_id,
            "db_id": db_id,
            "question": question,
            "rewards": rewards,
            "max_reward": max_reward,
            "preds": preds,
        }

    open_mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(open_mode, encoding="utf-8") as out_f:
        if args.num_workers <= 1:
            for record in records:
                result = process_record(record)
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                processed += 1
                if result.get("max_reward", 0.0) > 0:
                    success_count += 1
                if processed % args.log_every == 0:
                    logger.info("Processed %s/%s samples", processed, total)
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [executor.submit(process_record, record) for record in records]
                for future in as_completed(futures):
                    result = future.result()
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    processed += 1
                    if result.get("max_reward", 0.0) > 0:
                        success_count += 1
                    if processed % args.log_every == 0:
                        logger.info("Processed %s/%s samples", processed, total)

    if processed == 0:
        logger.warning("No samples evaluated.")
        return
    success_rate = success_count / processed
    logger.info("Raw success: %s/%s = %.4f", success_count, processed, success_rate)


if __name__ == "__main__":
    main()
