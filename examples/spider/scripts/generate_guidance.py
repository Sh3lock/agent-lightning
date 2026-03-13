#!/usr/bin/env python
"""Generate L1/L2 guidance bodies for hard samples using a strong model (Optimized Version)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# NOTE: Hardcoded defaults for the strong model endpoint/key.
DEFAULT_STRONG_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_STRONG_API_KEY = "sk-9378054100064c45973b40a986ab529f"
DEFAULT_STRONG_MODEL = "qwen3-235b-a22b-instruct-2507"


def _load_tokenizer(model_path: str, revision: Optional[str]) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, trust_remote_code=True)
    return tokenizer


def _log_tokenizer_info(tokenizer: AutoTokenizer) -> None:
    model_path = getattr(tokenizer, "name_or_path", "unknown")
    revision = getattr(tokenizer, "revision", None)
    chat_template_hash = "unknown"
    if getattr(tokenizer, "chat_template", None):
        chat_template_hash = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
    
    logger.info(
        "Tokenizer: model_path=%s revision=%s chat_template_hash=%s",
        model_path,
        revision,
        chat_template_hash,
    )


def _count_tokens(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


class _RateLimiter:
    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self._next_time = time.monotonic()

    def wait(self) -> None:
        sleep_time = 0.0
        with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                sleep_time = self._next_time - now
                self._next_time += self.interval_s
            else:
                self._next_time = now + self.interval_s
        if sleep_time > 0:
            time.sleep(sleep_time)


def _generate_guidance(
    *,
    llm: Any,
    tokenizer: AutoTokenizer,
    record: Dict[str, Any],
    max_tokens: int,
    max_retries: int,
    gold_field: str,
    level: str = "L1",
    rate_limiter: Optional[_RateLimiter] = None,
) -> tuple[str, int, int]:
    """
    Generates guidance using distinct personas for L1 (Reviewer) and L2 (Teacher).
    """
    
    # 1. 提取关键字段
    question = record.get("question", "N/A")
    gold_sql = record.get(gold_field, "SELECT * FROM unknown") # 必须有 Gold SQL 才能做对比诊断
    
    # 处理 pred 可能是列表的情况，取最后一个或转为字符串
    pred_raw = record.get("pred", "")
    if isinstance(pred_raw, list):
        pred_sql = pred_raw[-1] if pred_raw else ""
    else:
        pred_sql = str(pred_raw)

    # 2. 定义 System Prompt：核心优化点
    if level == "L1":
        # L1: 严格的代码审查员 (Reviewer) - 一针见血
        system_prompt = (
            "You are a strict SQL Code Reviewer. Your task is to diagnose the error by comparing the User's SQL with the Hidden Gold SQL.\n"
            "INSTRUCTIONS:\n"
            "1. Identify the SINGLE most critical error (e.g., wrong column, missing filter, incorrect GROUP BY).\n"
            "2. Provide a short, direct hint pointing to the error location.\n"
            "CONSTRAINTS:\n"
            " - Do NOT reveal the Gold SQL or exact column names directly.\n"
            " - Do NOT output SQL code blocks.\n"
            " - Instead of 'Use WHERE id=1', say 'Check your filtering condition for the ID'.\n"
            f" - Keep it strictly under {max_tokens} tokens."
        )
    else:
        # L2: 数据库讲师 (Instructor) - 解释逻辑
        system_prompt = (
            "You are an expert Database Instructor. The user has written an incorrect SQL query. You have access to the Gold SQL as truth.\n"
            "INSTRUCTIONS:\n"
            "1. Analyze the LOGICAL gap between the User's attempt and the Gold SQL.\n"
            "2. Explain the correct reasoning step-by-step (e.g., 'To find the max per group, you must group by X first...').\n"
            "CONSTRAINTS:\n"
            " - Do NOT output SQL code blocks or verbatim queries.\n"
            " - Do NOT give the answer directly. Guide the user to derive it.\n"
            " - Use natural language to describe SQL operations.\n"
            f" - Keep guidance helpful but concise, under {max_tokens} tokens."
        )

    # 3. 定义 User Prompt：结构化对比
    # 明确展示 Question, Wrong SQL 和 Gold SQL，强迫模型进行对比
    user_prompt = (
        f"Analyze this Text-to-SQL error:\n\n"
        f"QUESTION: {question}\n"
        f"USER WRONG PREDICTION: {pred_sql}\n"
        f"GOLD TRUTH SQL (Hidden Reference): {gold_sql}\n\n"
        f"Task: Provide {level} guidance following the system constraints."
    )

    guidance_text = ""
    attempts = 0

    for attempt in range(max_retries):
        attempts = attempt + 1
        
        if attempt == 0:
            # 第一次生成
            if rate_limiter is not None:
                rate_limiter.wait()
            response = llm.invoke([
                SystemMessage(content=system_prompt), 
                HumanMessage(content=user_prompt)
            ])
        else:
            # 后续生成：使用 Summarization 策略而非简单的 Shorten
            shorten_prompt = (
                "The previous guidance was too long or verbose. "
                "REWRITE it to be more dense and precise. "
                "Focus ONLY on the most critical error correction. "
                "Discard generic filler words. "
                "Maintain the rule: NO SQL CODE allowed. "
                f"Strict Limit: {max_tokens} tokens."
            )
            if rate_limiter is not None:
                rate_limiter.wait()
            response = llm.invoke([
                SystemMessage(content=shorten_prompt), 
                HumanMessage(content=guidance_text)
            ])
        
        guidance_text = str(getattr(response, "content", "")).strip()
        token_count = _count_tokens(tokenizer, guidance_text)
        
        # 如果长度符合要求，直接返回
        if token_count <= max_tokens:
            return guidance_text, token_count, attempts

    # 如果重试多次仍超长，强制截断（作为最后的兜底，防止 crash）
    trimmed_ids = tokenizer.encode(guidance_text, add_special_tokens=False)[:max_tokens]
    trimmed_text = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()
    # 截断可能导致句子不完整，追加一个省略号
    if not trimmed_text.endswith("."):
        trimmed_text += "..."
        
    return trimmed_text, _count_tokens(tokenizer, trimmed_text), attempts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-samples", required=True, help="Path to round_r_hard_samples.jsonl.")
    parser.add_argument("--output", required=True, help="Output guidance jsonl path.")
    parser.add_argument("--tokenizer-model", required=True, help="Training model path for token counting.")
    parser.add_argument("--tokenizer-revision", default=None, help="Tokenizer revision.")
    parser.add_argument(
        "--guidance-model",
        default=DEFAULT_STRONG_MODEL,
        help="Strong model name for guidance (defaults to hardcoded model).",
    )
    parser.add_argument(
        "--guidance-endpoint",
        default=None,
        help="OpenAI-compatible API base (defaults to hardcoded STRONG endpoint).",
    )
    parser.add_argument(
        "--guidance-api-key",
        default=None,
        help="API key for strong endpoint (defaults to hardcoded key).",
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="Guidance sampling temperature.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for shortening guidance.")
    # L1 限制通常较短，用于 Quick Hint
    parser.add_argument("--l1-limit", type=int, default=110, help="Max tokens for L1 guidance body.")
    # L2 限制稍长，用于 Reasoning Chain
    parser.add_argument("--l2-limit", type=int, default=240, help="Max tokens for L2 guidance body.")
    parser.add_argument("--gold-field", default="gold_query", help="Field for gold SQL in input jsonl.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=64,
        help="Concurrent workers for guidance generation.",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.2,
        help="Minimum interval (seconds) between requests across all workers.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume by skipping sample_ids already present in the output file.",
    )
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    tokenizer = _load_tokenizer(args.tokenizer_model, args.tokenizer_revision)
    _log_tokenizer_info(tokenizer)

    guidance_endpoint = args.guidance_endpoint or DEFAULT_STRONG_API_BASE
    guidance_api_key = args.guidance_api_key or DEFAULT_STRONG_API_KEY

    llm = init_chat_model(
        args.guidance_model,
        model_provider="openai",
        openai_api_base=guidance_endpoint,
        openai_api_key=guidance_api_key,
        temperature=args.temperature,
        max_retries=1,
        # 模型本身的 max_tokens 设置得稍大一点，给它生成的空间，我们在代码里用 _count_tokens 截断
        max_tokens=1024, 
    )

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
        logger.info("Resume enabled: %s samples already generated", len(existing_ids))

    with open(args.hard_samples, "r", encoding="utf-8") as in_f:
        records = [json.loads(line) for line in in_f if line.strip()]
    if existing_ids:
        records = [
            record for record in records
            if record.get("sample_id") not in existing_ids
        ]
        logger.info("Pending guidance samples: %s", len(records))

    rate_limiter = _RateLimiter(args.request_interval)
    write_lock = threading.Lock()
    progress_lock = threading.Lock()
    processed = 0

    def process_record(record: Dict[str, Any]) -> Dict[str, Any]:
        l1_text, l1_tokens, l1_attempts = _generate_guidance(
            llm=llm,
            tokenizer=tokenizer,
            record=record,
            max_tokens=args.l1_limit,
            max_retries=args.max_retries,
            gold_field=args.gold_field,
            level="L1",
            rate_limiter=rate_limiter,
        )
        l2_text, l2_tokens, l2_attempts = _generate_guidance(
            llm=llm,
            tokenizer=tokenizer,
            record=record,
            max_tokens=args.l2_limit,
            max_retries=args.max_retries,
            gold_field=args.gold_field,
            level="L2",
            rate_limiter=rate_limiter,
        )
        return {
            "sample_id": record.get("sample_id"),
            "db_id": record.get("db_id"),
            "question": record.get("question"),
            "guidance_l1": l1_text,
            "guidance_l2": l2_text,
            "guidance_l1_tokens": l1_tokens,
            "guidance_l2_tokens": l2_tokens,
            "guidance_l1_attempts": l1_attempts,
            "guidance_l2_attempts": l2_attempts,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

    open_mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(open_mode, encoding="utf-8") as out_f:
        if args.num_workers <= 1:
            for record in records:
                out_record = process_record(record)
                out_f.write(json.dumps(out_record, ensure_ascii=True) + "\n")
                processed += 1
                if processed % 50 == 0:
                    logger.info("Generated guidance for %s samples", processed)
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [executor.submit(process_record, record) for record in records]
                for future in as_completed(futures):
                    try:
                        out_record = future.result()
                    except Exception as exc:
                        logger.error("Guidance generation failed: %s", exc)
                        continue
                    with write_lock:
                        out_f.write(json.dumps(out_record, ensure_ascii=True) + "\n")
                    with progress_lock:
                        processed += 1
                        if processed % 50 == 0:
                            logger.info("Generated guidance for %s samples", processed)

    logger.info("Wrote guidance for %s samples to %s", processed, output_path)


if __name__ == "__main__":
    main()
