#!/usr/bin/env python
"""Check guidance token lengths against L1/L2 limits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import AutoTokenizer


def _count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _update_stats(stats: Dict[str, int], value: int) -> None:
    stats["count"] += 1
    stats["sum"] += value
    if value > stats["max"]:
        stats["max"] = value
    if value < stats["min"]:
        stats["min"] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guidance", required=True, help="Path to round_r_guidance.jsonl.")
    parser.add_argument("--tokenizer-model", required=True, help="Tokenizer model path.")
    parser.add_argument("--tokenizer-revision", default=None, help="Tokenizer revision.")
    parser.add_argument("--l1-limit", type=int, default=110, help="L1 token limit.")
    parser.add_argument("--l2-limit", type=int, default=240, help="L2 token limit.")
    parser.add_argument("--max-show", type=int, default=10, help="Max oversize samples to show.")
    args = parser.parse_args()

    path = Path(args.guidance)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model, revision=args.tokenizer_revision, trust_remote_code=True
    )

    l1_stats = {"count": 0, "sum": 0, "min": 10**9, "max": 0}
    l2_stats = {"count": 0, "sum": 0, "min": 10**9, "max": 0}
    l1_overs: List[Tuple[str, int]] = []
    l2_overs: List[Tuple[str, int]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = str(record.get("sample_id", ""))

            l1_text = str(record.get("guidance_l1", "") or "")
            l2_text = str(record.get("guidance_l2", "") or "")
            l1_len = _count_tokens(tokenizer, l1_text)
            l2_len = _count_tokens(tokenizer, l2_text)

            _update_stats(l1_stats, l1_len)
            _update_stats(l2_stats, l2_len)

            if l1_len > args.l1_limit:
                l1_overs.append((sample_id, l1_len))
            if l2_len > args.l2_limit:
                l2_overs.append((sample_id, l2_len))

    def _avg(stats: Dict[str, int]) -> float:
        if stats["count"] == 0:
            return 0.0
        return stats["sum"] / stats["count"]

    print("Guidance length summary")
    print(f"- total: {l1_stats['count']}")
    print(f"- l1_limit: {args.l1_limit}")
    print(f"- l2_limit: {args.l2_limit}")
    print(f"- l1_min: {l1_stats['min']} l1_max: {l1_stats['max']} l1_avg: {_avg(l1_stats):.2f}")
    print(f"- l2_min: {l2_stats['min']} l2_max: {l2_stats['max']} l2_avg: {_avg(l2_stats):.2f}")
    print(f"- l1_oversize: {len(l1_overs)}")
    print(f"- l2_oversize: {len(l2_overs)}")

    if l1_overs:
        print("- l1_oversize_samples:")
        for sample_id, length in l1_overs[: args.max_show]:
            print(f"  - {sample_id}: {length}")
        if len(l1_overs) > args.max_show:
            print(f"  ... and {len(l1_overs) - args.max_show} more")
    if l2_overs:
        print("- l2_oversize_samples:")
        for sample_id, length in l2_overs[: args.max_show]:
            print(f"  - {sample_id}: {length}")
        if len(l2_overs) > args.max_show:
            print(f"  ... and {len(l2_overs) - args.max_show} more")


if __name__ == "__main__":
    main()
