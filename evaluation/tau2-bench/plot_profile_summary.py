#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Generate cross-model summary plots from tau2-bench `profile_analysis.json`.

Outputs PNGs to `profile_charts/`:
- summary_latency_percentiles.png: P50/P90/P99 (ms) per model
- summary_under5s_and_counts.png: <5s% and call counts per model
- summary_prefill_decode_p50.png: Prefill/Decode P50 (ms) where available

Usage:
  cd evaluation/tau2-bench
  python plot_profile_summary.py --profile-json profile_analysis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _get(d: dict[str, Any], path: list[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _pretty_model(name: str) -> str:
    # shorten for charts
    name = name.replace("expert:", "")
    name = name.replace("Qwen/Qwen3-Next-80B-A3B-Instruct-FP8", "Qwen3-Next-80B-A3B")
    name = name.replace("Qwen/Qwen3-32B-FP8", "Qwen3-32B-FP8")
    name = name.replace("openai/gpt-oss-20b", "gpt-oss-20b")
    name = name.replace("user_sim:gpt-5", "user_sim:gpt-5")
    return name


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--profile-json", type=str, default="profile_analysis.json")
    p.add_argument("--out-dir", type=str, default="profile_charts")
    args = p.parse_args()

    src = Path(args.profile_json).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"profile json not found: {src}")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(src.read_text(encoding="utf-8"))

    # Keep a stable ordering with the most important actors first.
    order = [
        "orchestrator",
        "expert:openai/gpt-oss-20b",
        "expert:Qwen/Qwen3-32B-FP8",
        "expert:Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
        "user_sim:gpt-5",
    ]
    keys = [k for k in order if k in data] + [k for k in data.keys() if k not in order]

    rows = []
    for k in keys:
        stats = _get(data, [k, "stats"], {}) or {}
        if not stats or not isinstance(stats, dict):
            continue
        rows.append(
            {
                "key": k,
                "name": _pretty_model(k),
                "count": float(stats.get("count", 0) or 0),
                "p50": float(stats.get("p50", 0) or 0),
                "p90": float(stats.get("p90", 0) or 0),
                "p99": float(stats.get("p99", 0) or 0),
                "under5": float(stats.get("under_5s_pct", 0) or 0),
                "prefill_p50": float(_get(data, [k, "prefill_stats", "p50"], 0) or 0),
                "decode_p50": float(_get(data, [k, "decode_stats", "p50"], 0) or 0),
                "has_prefill_decode": float(_get(data, [k, "prefill_stats", "count"], 0) or 0) > 0,
            }
        )

    if not rows:
        raise SystemExit("No rows found in profile_analysis.json")

    names = [r["name"] for r in rows]
    y = list(range(len(rows)))

    # Plot 1: Percentiles (ms) — horizontal grouped bars
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    h = 0.22
    ax.barh([yy - h for yy in y], [r["p50"] for r in rows], height=h, label="P50", color="#1f77b4")
    ax.barh(y, [r["p90"] for r in rows], height=h, label="P90", color="#ff7f0e")
    ax.barh([yy + h for yy in y], [r["p99"] for r in rows], height=h, label="P99", color="#2ca02c")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Tau2-Bench profiling: Latency percentiles by model")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(out_dir / "summary_latency_percentiles.png", dpi=200)
    plt.close(fig)

    # Plot 2: Under 5s% + Counts
    fig, ax1 = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    under5 = [r["under5"] for r in rows]
    counts = [r["count"] for r in rows]
    ax1.barh(y, under5, color="#9467bd", alpha=0.85)
    ax1.set_xlim(0, 100)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names)
    ax1.invert_yaxis()
    ax1.set_xlabel("<5s %")
    ax1.set_title("Tau2-Bench profiling: <5s% and call volume")
    ax1.grid(axis="x", alpha=0.25)
    for yy, val in zip(y, under5):
        ax1.text(min(val + 1.0, 99.0), yy, f"{val:.1f}%", va="center", fontsize=9)

    ax2 = ax1.twiny()
    ax2.plot(counts, y, "o", color="#111111", alpha=0.8)
    ax2.set_xlabel("Count (calls)")
    # make counts axis not too busy
    ax2.grid(False)
    fig.savefig(out_dir / "summary_under5s_and_counts.png", dpi=200)
    plt.close(fig)

    # Plot 3: Prefill/Decode P50 (ms) where available
    rows_pd = [r for r in rows if r["has_prefill_decode"]]
    if rows_pd:
        names_pd = [r["name"] for r in rows_pd]
        y2 = list(range(len(rows_pd)))
        fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
        ax.barh([yy - 0.18 for yy in y2], [r["prefill_p50"] for r in rows_pd], height=0.35, label="Prefill P50", color="#17becf")
        ax.barh([yy + 0.18 for yy in y2], [r["decode_p50"] for r in rows_pd], height=0.35, label="Decode P50", color="#d62728")
        ax.set_yticks(y2)
        ax.set_yticklabels(names_pd)
        ax.invert_yaxis()
        ax.set_xlabel("Latency (ms)")
        ax.set_title("Tau2-Bench profiling: Prefill/Decode breakdown (P50)")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="lower right")
        fig.savefig(out_dir / "summary_prefill_decode_p50.png", dpi=200)
        plt.close(fig)

    print(f"Wrote summary plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


