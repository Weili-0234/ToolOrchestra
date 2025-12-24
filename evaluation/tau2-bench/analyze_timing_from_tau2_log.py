#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Analyze tau2 structured logs (TAU2_LOG_FILE) and compute timing statistics + equal-width bin distributions.

This script is intentionally lightweight and depends only on numpy + matplotlib.

Example:
  python analyze_timing_from_tau2_log.py \
    --log-path logs/airline_profile_20_c10/tau2_airline.log

Aggregate across a run directory (all domains):
  python analyze_timing_from_tau2_log.py \
    --log-dir logs/full_c48_profile_20251224_123456
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Any

import numpy as np
import matplotlib.pyplot as plt


def _extract_float(regex: str, s: str) -> Optional[float]:
    m = re.search(regex, s)
    return float(m.group(1)) if m else None


def _stats_ms(values: list[float]) -> Optional[dict[str, Any]]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return None
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def _hist_bins(values: list[float], bins: int) -> Optional[dict[str, Any]]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return None
    lo = float(a.min())
    hi = float(a.max())
    if hi == lo:
        hi = lo + 1e-9
    edges = np.linspace(lo, hi, int(bins) + 1)
    counts, _ = np.histogram(a, bins=edges)
    perc = (counts / a.size) * 100.0
    return {
        "edges": edges.tolist(),
        "counts": counts.tolist(),
        "percent": perc.tolist(),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--log-path", type=str, help="Path to a single TAU2_LOG_FILE (e.g. tau2_airline.log)")
    src.add_argument("--log-dir", type=str, help="Directory containing tau2_*.log files (e.g. a run --log-dir)")
    p.add_argument("--log-glob", type=str, default="tau2_*.log", help="Glob inside --log-dir (default: tau2_*.log)")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default: same as log file)")
    p.add_argument("--out-prefix", type=str, default="timing", help="Output filename prefix")
    p.add_argument("--bins", type=int, default=10, help="Number of equal-width bins for distributions (default: 10)")
    args = p.parse_args()

    log_paths: list[Path] = []
    if args.log_path:
        lp = Path(args.log_path).expanduser().resolve()
        if not lp.exists():
            raise SystemExit(f"log path not found: {lp}")
        log_paths = [lp]
        default_out = lp.parent
        source_label = lp.name
    else:
        ld = Path(args.log_dir).expanduser().resolve()
        if not ld.exists():
            raise SystemExit(f"log dir not found: {ld}")
        log_paths = sorted([p for p in ld.glob(args.log_glob) if p.is_file()])
        if not log_paths:
            raise SystemExit(f"no logs matched in {ld} with glob={args.log_glob}")
        default_out = ld
        source_label = ld.name

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{args.out_prefix}_stats.json"

    # Read all logs (aggregate)
    lines: list[str] = []
    for lp in log_paths:
        lines.extend(lp.read_text(encoding="utf-8", errors="replace").splitlines())

    tool_call_ms: list[float] = []
    expert_call_ms: list[float] = []
    vllm_infer_ms: list[float] = []
    vllm_prefill_ms: list[float] = []
    vllm_decode_ms: list[float] = []
    vllm_prefill_len: list[float] = []
    vllm_decode_len: list[float] = []

    for line in lines:
        if "type=tool_call" in line:
            # NOTE: keep regex as a raw string; escape '.' as '\.' (NOT '\\.').
            v = _extract_float(r"duration_ms=([0-9]+(?:\.[0-9]+)?)", line)
            if v is not None:
                tool_call_ms.append(v)
        if "type=expert_call" in line:
            v = _extract_float(r"duration_ms=([0-9]+(?:\.[0-9]+)?)", line)
            if v is not None:
                expert_call_ms.append(v)
        if "type=llm_call" in line and "toolorchestra_vllm_infer_ms=" in line:
            v = _extract_float(r"toolorchestra_vllm_infer_ms=([0-9]+(?:\.[0-9]+)?)", line)
            if v is not None:
                vllm_infer_ms.append(v)
            v = _extract_float(r"toolorchestra_vllm_prefill_ms=([0-9]+(?:\.[0-9]+)?)", line)
            if v is not None:
                vllm_prefill_ms.append(v)
            v = _extract_float(r"toolorchestra_vllm_decode_ms=([0-9]+(?:\.[0-9]+)?)", line)
            if v is not None:
                vllm_decode_ms.append(v)
            v = _extract_float(r"toolorchestra_vllm_prefill_len=([0-9]+(?:\.[0-9]+)?)", line)
            if v is not None:
                vllm_prefill_len.append(v)
            v = _extract_float(r"toolorchestra_vllm_decode_len=([0-9]+(?:\.[0-9]+)?)", line)
            if v is not None:
                vllm_decode_len.append(v)

    summary = {
        "tool_call": {"unit": "ms", "stats": _stats_ms(tool_call_ms), "hist": _hist_bins(tool_call_ms, args.bins)},
        "expert_call": {"unit": "ms", "stats": _stats_ms(expert_call_ms), "hist": _hist_bins(expert_call_ms, args.bins)},
        "toolorchestra_vllm_infer": {"unit": "ms", "stats": _stats_ms(vllm_infer_ms), "hist": _hist_bins(vllm_infer_ms, args.bins)},
        "toolorchestra_vllm_prefill": {"unit": "ms", "stats": _stats_ms(vllm_prefill_ms), "hist": _hist_bins(vllm_prefill_ms, args.bins)},
        "toolorchestra_vllm_decode": {"unit": "ms", "stats": _stats_ms(vllm_decode_ms), "hist": _hist_bins(vllm_decode_ms, args.bins)},
        "toolorchestra_vllm_prefill_len": {"unit": "tok", "stats": _stats_ms(vllm_prefill_len), "hist": _hist_bins(vllm_prefill_len, args.bins)},
        "toolorchestra_vllm_decode_len": {"unit": "tok", "stats": _stats_ms(vllm_decode_len), "hist": _hist_bins(vllm_decode_len, args.bins)},
    }

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Console summary
    for key, label in [
        ("tool_call", "tool_call.duration_ms"),
        ("expert_call", "expert_call.duration_ms"),
        ("toolorchestra_vllm_infer", "llm_call.toolorchestra_vllm_infer_ms"),
        ("toolorchestra_vllm_prefill", "llm_call.toolorchestra_vllm_prefill_ms"),
        ("toolorchestra_vllm_decode", "llm_call.toolorchestra_vllm_decode_ms"),
        ("toolorchestra_vllm_prefill_len", "llm_call.toolorchestra_vllm_prefill_len"),
        ("toolorchestra_vllm_decode_len", "llm_call.toolorchestra_vllm_decode_len"),
    ]:
        s = summary[key]["stats"]
        unit = summary[key].get("unit", "")
        print(f"=== {label} ===")
        if not s:
            print("no data\n")
            continue
        print(
            f"n={s['n']} mean={s['mean_ms']:.2f}{unit} std={s['std_ms']:.2f}{unit} "
            f"min={s['min_ms']:.2f}{unit} max={s['max_ms']:.2f}{unit}"
        )
        h = summary[key]["hist"]
        edges = h["edges"]
        perc = h["percent"]
        for i in range(args.bins):
            if unit == "tok":
                print(f"[{edges[i]:.0f}, {edges[i+1]:.0f}]: {perc[i]:5.1f}%")
            else:
                print(f"[{edges[i]:.2f}, {edges[i+1]:.2f}]: {perc[i]:5.1f}%")
        print()

    # Plot: exactly 7 PNGs (one per metric)
    plots = [
        ("tool_call", "Local python tool calls (type=tool_call) duration_ms"),
        ("expert_call", "Expert LLM-as-tool calls (type=expert_call) duration_ms"),
        ("toolorchestra_vllm_infer", "ToolOrchestra vLLM infer_ms"),
        ("toolorchestra_vllm_prefill", "ToolOrchestra vLLM prefill_ms (TTFT)"),
        ("toolorchestra_vllm_decode", "ToolOrchestra vLLM decode_ms"),
        ("toolorchestra_vllm_prefill_len", "ToolOrchestra vLLM prefill_len (prompt tokens)"),
        ("toolorchestra_vllm_decode_len", "ToolOrchestra vLLM decode_len (completion tokens)"),
    ]
    png_paths: list[Path] = []
    for key, title in plots:
        h = summary[key]["hist"]
        s = summary[key]["stats"]
        unit = summary[key].get("unit", "")
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
        if not h or not s:
            ax.set_title(title)
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_axis_off()
        else:
            edges = np.asarray(h["edges"], dtype=float)
            perc = np.asarray(h["percent"], dtype=float)
            centers = (edges[:-1] + edges[1:]) / 2.0
            width = (edges[1] - edges[0]) * 0.9
            bars = ax.bar(centers, perc, width=width, align="center")
            
            # Add percentage labels on top of each bar
            for bar, pct in zip(bars, perc):
                height = bar.get_height()
                if height > 0:  # Only label non-zero bars
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{pct:.1f}%',
                            ha='center', va='bottom', fontsize=8)
            
            ax.set_title(f"{title} (n={s['n']}, mean={s['mean_ms']:.2f}{unit}, std={s['std_ms']:.2f}{unit})")
            ax.set_xlabel(unit)
            ax.set_ylabel("%")
        out_png = out_dir / f"{args.out_prefix}_{key}.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        png_paths.append(out_png)

    print(f"Wrote {json_path}")
    for pth in png_paths:
        print(f"Wrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


