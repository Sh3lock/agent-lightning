#!/usr/bin/env python
"""Quickly test guidance levels (L1 then L2) on hard samples."""

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
from typing import Any, Dict, Optional

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


def _resolve_db_path(spider_dir: Path, db_id: str, use_test_split: bool) -> Path:
    base = spider_dir / ("test_database" if use_test_split else "database")
    return base / db_id / f"{db_id}.sqlite"


def _load_schema(db_path: Path) -> str:
    schema_path = db_path.parent / "schema.sql"
    if schema_path.exists():
        return schema_path.read_text(encoding="utf-8")
    return "No schema available."


def _run_guided_rollout(
    *,
    question: str,
    ground_truth: str,
    db_path: Path,
    schema: str,
    guidance_text: str,
    guidance_level: str,
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
                {"question": question, "guidance": guidance_text, "guidance_level": guidance_level},
                {"recursion_limit": 100},
            )
            query = result.get("query", "") if isinstance(result, dict) else ""
            reward = evaluate_query(query, ground_truth, str(temp_db), raise_on_error=False)
        return query, reward
    except Exception as exc:
        logger.error("Guided rollout failed: %s", exc)
        return "", 0.0


def _load_guidance_map(path: Path) -> Dict[str, Dict[str, Any]]:
    guidance_map: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = record.get("sample_id")
            if sample_id:
                guidance_map[sample_id] = record
    return guidance_map


def _load_ground_truth_map(path: Path, question_field: str, db_id_field: str, query_field: str) -> Dict[str, str]:
    df = pd.read_parquet(path)
    mapping: Dict[str, str] = {}
    for record in df.to_dict(orient="records"):
        sample_id = stable_sample_id(record[db_id_field], record[question_field])
        mapping[sample_id] = record[query_field]
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-samples", required=True, help="Path to round_r_hard_samples.jsonl.")
    parser.add_argument("--guidance", required=True, help="Path to round_r_guidance.jsonl.")
    parser.add_argument("--output", required=True, help="Output ignite jsonl path.")
    parser.add_argument("--round", type=int, required=True, help="Round index for output tagging.")
    parser.add_argument("--k", type=int, default=4, help="Trials per guidance level.")
    parser.add_argument("--model", required=True, help="Model name for ignite sampling.")
    parser.add_argument("--endpoint", default=None, help="OpenAI-compatible API base.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--max-turns", type=int, default=3, help="Max agent turns.")
    parser.add_argument("--table-info-truncate", type=int, default=2048, help="Table info truncate length.")
    parser.add_argument("--execution-truncate", type=int, default=2048, help="Execution result truncate length.")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers.")
    parser.add_argument("--use-test-split", action="store_true", help="Use test_database instead of database.")
    parser.add_argument("--dataset", default=None, help="Parquet dataset for ground truth if missing.")
    parser.add_argument("--question-field", default="question", help="Field name for question.")
    parser.add_argument("--db-id-field", default="db_id", help="Field name for db id.")
    parser.add_argument("--query-field", default="query", help="Field name for gold query.")
    parser.add_argument("--tokenizer-model", required=True, help="Model path for tokenizer logging.")
    parser.add_argument("--tokenizer-revision", default=None, help="Tokenizer revision.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume by skipping sample_ids already present in the output file.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    endpoint = args.endpoint or os.environ.get("OPENAI_API_BASE")
    if not endpoint:
        raise ValueError("Missing --endpoint and OPENAI_API_BASE; required for ignite sampling.")

    tokenizer = _load_tokenizer(args.tokenizer_model, args.tokenizer_revision)
    _log_tokenizer_info(tokenizer)

    guidance_map = _load_guidance_map(Path(args.guidance))
    ground_truth_map: Dict[str, str] = {}
    if args.dataset:
        ground_truth_map = _load_ground_truth_map(
            Path(args.dataset),
            args.question_field,
            args.db_id_field,
            args.query_field,
        )

    spider_dir = Path(os.environ.get("VERL_SPIDER_DATA_DIR", "data"))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = set()
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as existing_f:
            for line in existing_f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = record.get("sample_id")
                if sample_id is not None:
                    existing_ids.add(sample_id)
        logger.info("Resume enabled: %s samples already processed", len(existing_ids))

    def process_record(record: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = record.get("sample_id")
        question = record.get("question")
        db_id = record.get("db_id")
        if not sample_id or not question or not db_id:
            return {"sample_id": sample_id, "min_level": "none", "error": "missing_fields"}

        guidance = guidance_map.get(sample_id)
        if not guidance:
            return {"sample_id": sample_id, "min_level": "none", "error": "missing_guidance"}

        ground_truth = record.get("gold_query") or ground_truth_map.get(sample_id)
        if not ground_truth:
            return {"sample_id": sample_id, "min_level": "none", "error": "missing_ground_truth"}

        db_path = _resolve_db_path(spider_dir, db_id, args.use_test_split)
        if not db_path.exists():
            return {"sample_id": sample_id, "min_level": "none", "error": "missing_database"}

        schema = _load_schema(db_path)
        result_payload: Dict[str, Any] = {
            "sample_id": sample_id,
            "db_id": db_id,
            "question": question,
            "round": args.round,
        }

        def run_level(level: str, guidance_text: str) -> Dict[str, Any]:
            rewards = []
            preds = []
            for _ in range(args.k):
                pred, reward = _run_guided_rollout(
                    question=question,
                    ground_truth=ground_truth,
                    db_path=db_path,
                    schema=schema,
                    guidance_text=guidance_text,
                    guidance_level=level,
                    model=args.model,
                    endpoint=endpoint,
                    temperature=args.temperature,
                    max_turns=args.max_turns,
                    table_info_truncate=args.table_info_truncate,
                    execution_truncate=args.execution_truncate,
                )
                preds.append(pred)
                rewards.append(float(reward))
            return {"rewards": rewards, "max_reward": max(rewards) if rewards else 0.0, "preds": preds}

        l1_text = guidance.get("guidance_l1", "")
        l2_text = guidance.get("guidance_l2", "")
        l1_stats = run_level("L1", l1_text) if l1_text is not None else {"rewards": [], "max_reward": 0.0}
        if l1_stats["max_reward"] > 0:
            result_payload.update({"min_level": "L1", "l1_stats": l1_stats})
            return result_payload

        l2_stats = run_level("L2", l2_text) if l2_text is not None else {"rewards": [], "max_reward": 0.0}
        if l2_stats["max_reward"] > 0:
            result_payload.update({"min_level": "L2", "l1_stats": l1_stats, "l2_stats": l2_stats})
            return result_payload

        result_payload.update({"min_level": "none", "l1_stats": l1_stats, "l2_stats": l2_stats})
        return result_payload

    processed = 0
    with open(args.hard_samples, "r", encoding="utf-8") as in_f:
        records = [json.loads(line) for line in in_f if line.strip()]
    if existing_ids:
        records = [
            record for record in records
            if record.get("sample_id") not in existing_ids
        ]
        logger.info("Pending ignite samples: %s", len(records))

    open_mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(open_mode, encoding="utf-8") as out_f:

        if args.num_workers <= 1:
            for record in records:
                result = process_record(record)
                out_f.write(json.dumps(result, ensure_ascii=True) + "\n")
                processed += 1
                if processed % 50 == 0:
                    logger.info("Processed %s samples", processed)
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [executor.submit(process_record, record) for record in records]
                for future in as_completed(futures):
                    result = future.result()
                    out_f.write(json.dumps(result, ensure_ascii=True) + "\n")
                    processed += 1
                    if processed % 50 == 0:
                        logger.info("Processed %s samples", processed)

    logger.info("Wrote ignite results for %s samples to %s", processed, output_path)


if __name__ == "__main__":
    main()
