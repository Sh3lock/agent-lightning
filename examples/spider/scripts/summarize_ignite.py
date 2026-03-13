#!/usr/bin/env python
"""Summarize ignite guidance results by success level."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict


def _max_reward(stats: Dict[str, Any]) -> float:
    if not isinstance(stats, dict):
        return 0.0
    try:
        return float(stats.get("max_reward", 0.0))
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ignite", required=True, help="Path to round_r_ignite.jsonl.")
    args = parser.parse_args()

    path = Path(args.ignite)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    total = 0
    min_level_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    l1_stats_success = 0
    l2_stats_success = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            min_level = str(record.get("min_level", "none"))
            min_level_counts[min_level] += 1
            if "error" in record:
                error_counts[str(record.get("error"))] += 1

            if _max_reward(record.get("l1_stats", {})) > 0:
                l1_stats_success += 1
            if _max_reward(record.get("l2_stats", {})) > 0:
                l2_stats_success += 1

    guided_success = min_level_counts.get("L1", 0) + min_level_counts.get("L2", 0)

    print("Ignite summary")
    print(f"- total: {total}")
    print(f"- guidance_success_total: {guided_success}")
    print(f"- guidance_success_L1: {min_level_counts.get('L1', 0)}")
    print(f"- guidance_success_L2: {min_level_counts.get('L2', 0)}")
    print(f"- min_level_none: {min_level_counts.get('none', 0)}")
    if error_counts:
        print("- errors:")
        for key, val in sorted(error_counts.items()):
            print(f"  - {key}: {val}")

    # Optional cross-checks based on stats payloads.
    print(f"- l1_stats_max_reward_gt_0: {l1_stats_success}")
    print(f"- l2_stats_max_reward_gt_0: {l2_stats_success}")


if __name__ == "__main__":
    main()
