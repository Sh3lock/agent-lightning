#!/usr/bin/env python3
from pathlib import Path

# 硬编码输入路径
INPUT_PATH = "/home/storage/wenbinxing/ltf/passk/agent-lightning/examples/spider/log/20251220_073132_config_passk_stage1_qwen05b/gpu_monitor.csv"

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    input_path = Path(INPUT_PATH)
    df = pd.read_csv(input_path)
    df.columns = [col.strip() for col in df.columns]
    if "timestamp" not in df.columns or "mem.used" not in df.columns or "util.gpu" not in df.columns:
        raise SystemExit(f"Unexpected columns in {input_path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["name"] = df.get("name", "").astype(str).str.strip()
    df["mem.used"] = pd.to_numeric(df["mem.used"], errors="coerce")
    df["util.gpu"] = pd.to_numeric(df["util.gpu"], errors="coerce")
    df = df.dropna(subset=["timestamp", "mem.used", "util.gpu"])
    df = df.sort_values("timestamp")

    fig, (ax_mem, ax_util) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name, group in df.groupby("name"):
        label = name if name else "GPU"
        ax_mem.plot(group["timestamp"], group["mem.used"], label=label)
        ax_util.plot(group["timestamp"], group["util.gpu"], label=label)

    ax_mem.set_title("GPU Memory Used Over Time")
    ax_mem.set_ylabel("Memory Used (MiB)")
    ax_mem.grid(True, alpha=0.3)

    ax_util.set_title("GPU Utilization Over Time")
    ax_util.set_ylabel("Utilization (%)")
    ax_util.set_xlabel("Time")
    ax_util.grid(True, alpha=0.3)

    if df["name"].nunique(dropna=True) > 1:
        ax_mem.legend(loc="upper right")
        ax_util.legend(loc="upper right")

    fig.tight_layout()

    output_path = input_path.with_name("gpu_monitor_plot.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"[saved] {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
