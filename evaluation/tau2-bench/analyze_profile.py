#!/usr/bin/env python3
"""
Analyze tau2-bench profiling logs and generate latency plots/reports.

## What experiment are these plots for?
This script is used for the Tau2-Bench **OSS expert** evaluation run recorded in:
`evaluation/tau2-bench/WORKLOG_20251230.md` (Date: 2025-12-30).

**Setup (from the worklog):**
- **Agent / orchestrator**: Nemotron-Orchestrator-8B
- **Experts (tools)**:
  - Expert-1: openai/gpt-oss-20b
  - Expert-2: Qwen/Qwen3-32B-FP8
  - Expert-3: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
- Plus Tau2-Bench local python tools (logged as `type=tool_call`)

## Inputs
- Tau2-Bench domain profiling logs emitted with `--log-level PROFILE`, e.g.:
  - `evaluation/tau2-bench/logs_oss_full/tau2_airline.log`
  - `evaluation/tau2-bench/logs_oss_full/tau2_retail.log`
  - `evaluation/tau2-bench/logs_oss_full/tau2_telecom.log`

## Outputs
- `profile_analysis.json`: aggregated latency stats per model/role
- `experiment_report_<timestamp>.md`: markdown report
- `profile_charts/*.png`: figures including:
  - per-model latency distribution + CDF
  - stacked **all experts** latency distribution (truncated to 10s)
  - stacked **all tools (experts + local python)** latency distribution (truncated to 10s)

## Notes on the figures
- Histograms are plotted as **percent of calls per bin** (not raw counts).
- Some plots **truncate/collapse the tail** into the last visible bin for readability (see on-plot notes).
"""

import re
import sys
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, PNG charts will not be generated")


def parse_profile_logs(log_path: str) -> Dict[str, Any]:
    """
    Parse [PROFILE] entries from tau2 log file.

    Returns dict with:
    - latencies: Dict[model_name, List[duration_ms]]
    - prefill: Dict[model_name, List[prefill_ms]]
    - decode: Dict[model_name, List[decode_ms]]
    - input_tokens: Dict[model_name, List[int]]
    - output_tokens: Dict[model_name, List[int]]
    - tool_calls: Dict[(domain, tool_name), List[duration_ms]]
    - step_times: Dict[domain, List[duration_ms]]
    """

    # Enhanced patterns to capture all metrics
    expert_pattern = re.compile(
        r'\[PROFILE\].*?type=expert_call.*?model=([^\s]+).*?duration_ms=([\d.]+)'
        r'(?:.*?prefill_ms=([\d.]+))?'
        r'(?:.*?decode_ms=([\d.]+))?'
        r'(?:.*?input_tokens=(\d+))?'
        r'(?:.*?output_tokens=(\d+))?'
    )

    llm_call_pattern = re.compile(
        r'\[PROFILE\].*?type=llm_call.*?model=([^\s]+).*?duration_ms=([\d.]+)'
        r'(?:.*?prefill_ms=([\d.]+))?'
        r'(?:.*?decode_ms=([\d.]+))?'
        r'(?:.*?input_tokens=(\d+))?'
        r'(?:.*?output_tokens=(\d+))?'
    )

    user_sim_pattern = re.compile(
        r'\[USER_JUDGE\].*?type=(\w+).*?model=([^\s]+).*?duration_ms=([\d.]+)'
    )

    tool_call_pattern = re.compile(
        r'\[PROFILE\].*?type=tool_call.*?function=([^\s]+).*?duration_ms=([\d.]+)'
    )

    step_pattern = re.compile(
        r'\[PROFILE\].*?type=step_complete.*?duration_ms=([\d.]+)'
    )

    # Task context pattern to get domain
    task_pattern = re.compile(r'task=([^\s]+)')

    result = {
        "latencies": defaultdict(list),
        "prefill": defaultdict(list),
        "decode": defaultdict(list),
        "input_tokens": defaultdict(list),
        "output_tokens": defaultdict(list),
        "tool_calls": defaultdict(list),
        "step_times": defaultdict(list),
    }

    with open(log_path, 'r') as f:
        for line in f:
            # Expert calls
            if 'type=expert_call' in line:
                match = expert_pattern.search(line)
                if match:
                    model = match.group(1)
                    duration = float(match.group(2))
                    result["latencies"][f"expert:{model}"].append(duration)

                    if match.group(3):
                        result["prefill"][f"expert:{model}"].append(float(match.group(3)))
                    if match.group(4):
                        result["decode"][f"expert:{model}"].append(float(match.group(4)))
                    if match.group(5):
                        result["input_tokens"][f"expert:{model}"].append(int(match.group(5)))
                    if match.group(6):
                        result["output_tokens"][f"expert:{model}"].append(int(match.group(6)))
                continue

            # LLM calls (orchestrator)
            if 'type=llm_call' in line:
                match = llm_call_pattern.search(line)
                if match:
                    model = match.group(1)
                    duration = float(match.group(2))
                    if 'Orchestrator' in model or 'Nemotron' in model or 'orchestrator' in model.lower():
                        result["latencies"]["orchestrator"].append(duration)

                        if match.group(3):
                            result["prefill"]["orchestrator"].append(float(match.group(3)))
                        if match.group(4):
                            result["decode"]["orchestrator"].append(float(match.group(4)))
                        if match.group(5):
                            result["input_tokens"]["orchestrator"].append(int(match.group(5)))
                        if match.group(6):
                            result["output_tokens"]["orchestrator"].append(int(match.group(6)))
                continue

            # User simulation calls
            match = user_sim_pattern.search(line)
            if match:
                event_type = match.group(1)
                model = match.group(2)
                duration = float(match.group(3))
                result["latencies"][f"user_sim:{model}"].append(duration)
                continue

            # Tool calls
            match = tool_call_pattern.search(line)
            if match:
                function = match.group(1)
                duration = float(match.group(2))
                # Extract domain from task context if available
                task_match = task_pattern.search(line)
                domain = "unknown"
                if task_match:
                    task_id = task_match.group(1)
                    # Try to infer domain from task_id or log context
                    for d in ["retail", "telecom", "airline", "mock"]:
                        if d in task_id.lower() or d in line.lower():
                            domain = d
                            break
                result["tool_calls"][(domain, function)].append(duration)
                continue

            # Step completion
            match = step_pattern.search(line)
            if match:
                duration = float(match.group(1))
                # Try to get domain from context
                for d in ["retail", "telecom", "airline", "mock"]:
                    if d in line.lower():
                        result["step_times"][d].append(duration)
                        break
                else:
                    result["step_times"]["unknown"].append(duration)

    return result


def compute_stats(values: List[float]) -> Dict[str, Any]:
    """Compute enhanced statistics with full percentile range and <5s%."""
    if not values:
        return {"count": 0}

    arr = np.array(values)
    under_5s = np.sum(arr < 5000) / len(arr) * 100  # <5s percentage

    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p20": float(np.percentile(arr, 20)),
        "p25": float(np.percentile(arr, 25)),
        "p30": float(np.percentile(arr, 30)),
        "p40": float(np.percentile(arr, 40)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p60": float(np.percentile(arr, 60)),
        "p70": float(np.percentile(arr, 70)),
        "p80": float(np.percentile(arr, 80)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "under_5s_pct": float(under_5s),
    }


def compute_histogram(values: List[float], n_bins: int = 10) -> Dict[str, Any]:
    """Compute n-bin histogram of values."""
    if not values:
        return {"bins": [], "counts": [], "edges": []}

    arr = np.array(values)
    counts, edges = np.histogram(arr, bins=n_bins)

    bin_labels = []
    for i in range(len(edges) - 1):
        bin_labels.append(f"{edges[i]:.1f}-{edges[i+1]:.1f}")

    return {
        "bins": bin_labels,
        "counts": counts.tolist(),
        "edges": edges.tolist()
    }


def print_histogram(name: str, hist: Dict, width: int = 40):
    """Print ASCII histogram."""
    if not hist["counts"]:
        print(f"  No data for {name}")
        return

    max_count = max(hist["counts"]) if hist["counts"] else 1

    print(f"\n  10-bin Latency Distribution (ms):")
    print(f"  {'Bin Range':<20} {'Count':>6} {'Histogram':<{width}}")
    print(f"  {'-'*20} {'-'*6} {'-'*width}")

    for bin_label, count in zip(hist["bins"], hist["counts"]):
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = '#' * bar_len
        print(f"  {bin_label:<20} {count:>6} {bar}")


def analyze_and_print(data: Dict[str, Any]):
    """Analyze latencies and print report."""
    latencies = data["latencies"]

    print("=" * 70)
    print("TAU2-BENCH PROFILING ANALYSIS")
    print("=" * 70)

    sorted_keys = sorted(latencies.keys())

    for name in sorted_keys:
        values = latencies[name]
        stats = compute_stats(values)
        hist = compute_histogram(values, n_bins=10)

        print(f"\n{'='*70}")
        print(f"MODEL: {name}")
        print(f"{'='*70}")

        if stats["count"] == 0:
            print("  No data")
            continue

        print(f"\n  Latency Statistics (ms):")
        print(f"  {'Count:':<12} {stats['count']:>10}")
        print(f"  {'Mean:':<12} {stats['mean']:>10.2f}")
        print(f"  {'Std:':<12} {stats['std']:>10.2f}")
        print(f"  {'Min:':<12} {stats['min']:>10.2f}")
        print(f"  {'Max:':<12} {stats['max']:>10.2f}")
        print(f"  {'P10:':<12} {stats['p10']:>10.2f}")
        print(f"  {'P25:':<12} {stats['p25']:>10.2f}")
        print(f"  {'P40:':<12} {stats['p40']:>10.2f}")
        print(f"  {'P50:':<12} {stats['p50']:>10.2f}")
        print(f"  {'P90:':<12} {stats['p90']:>10.2f}")
        print(f"  {'P95:':<12} {stats['p95']:>10.2f}")
        print(f"  {'P99:':<12} {stats['p99']:>10.2f}")
        print(f"  {'<5s%:':<12} {stats['under_5s_pct']:>10.2f}")

        # Print prefill/decode breakdown if available
        if name in data["prefill"] and data["prefill"][name]:
            prefill_stats = compute_stats(data["prefill"][name])
            decode_stats = compute_stats(data["decode"].get(name, []))
            print(f"\n  Prefill/Decode Breakdown (ms):")
            print(f"  {'Prefill Mean:':<16} {prefill_stats.get('mean', 0):>10.2f}")
            print(f"  {'Prefill P50:':<16} {prefill_stats.get('p50', 0):>10.2f}")
            print(f"  {'Decode Mean:':<16} {decode_stats.get('mean', 0):>10.2f}")
            print(f"  {'Decode P50:':<16} {decode_stats.get('p50', 0):>10.2f}")

        # Print token stats if available
        if name in data["input_tokens"] and data["input_tokens"][name]:
            input_stats = compute_stats([float(x) for x in data["input_tokens"][name]])
            output_stats = compute_stats([float(x) for x in data["output_tokens"].get(name, [])])
            print(f"\n  Token Statistics:")
            print(f"  {'Input Mean:':<16} {input_stats.get('mean', 0):>10.2f}")
            print(f"  {'Output Mean:':<16} {output_stats.get('mean', 0):>10.2f}")

        print_histogram(name, hist)

    print(f"\n{'='*70}")
    print("END OF ANALYSIS")
    print("=" * 70)


def generate_png_charts(data: Dict[str, Any], output_dir: Path):
    """Generate matplotlib PNG charts for each model."""
    if not HAS_MATPLOTLIB:
        print("Skipping PNG chart generation (matplotlib not available)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    latencies = data["latencies"]

    # Consistent colors (requested: 3 experts distinct; all python local tools one color)
    COLOR_EXPERT1 = "#1f77b4"  # gpt-oss-20b
    COLOR_EXPERT2 = "#2ca02c"  # Qwen3-32B-FP8
    COLOR_EXPERT3 = "#ff7f0e"  # Qwen3-Next-80B-A3B
    COLOR_PYTOOLS = "#7f7f7f"  # local python tools
    COLOR_OTHER = "#9467bd"

    def _role_and_tool_label(model_key: str) -> str:
        """
        Return a concise label for what this latency series represents in this tau2-bench setup.
        We keep this intentionally lightweight and based on WORKLOG_20251230.md's mapping.
        """
        if model_key == "orchestrator":
            return "Role: orchestrator (agent core)"
        if model_key.startswith("user_sim:"):
            return "Role: user simulator"
        if model_key.startswith("expert:"):
            # Map the 3 expert tools used in this experiment
            if "openai/gpt-oss-20b" in model_key:
                return "Tool: Expert-1 (gpt-oss-20b)"
            if "Qwen/Qwen3-32B-FP8" in model_key:
                return "Tool: Expert-2 (Qwen3-32B-FP8)"
            if "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8" in model_key:
                return "Tool: Expert-3 (Qwen3-Next-80B-A3B)"
            return "Tool: expert"
        return "Role: unknown"

    def _series_color(model_key: str) -> str:
        if model_key.startswith("expert:") and ("openai/gpt-oss-20b" in model_key):
            return COLOR_EXPERT1
        if model_key.startswith("expert:") and ("Qwen/Qwen3-32B-FP8" in model_key):
            return COLOR_EXPERT2
        if model_key.startswith("expert:") and ("Qwen/Qwen3-Next-80B-A3B-Instruct-FP8" in model_key):
            return COLOR_EXPERT3
        if model_key == "orchestrator":
            return COLOR_OTHER
        if model_key.startswith("user_sim:"):
            return COLOR_OTHER
        return COLOR_OTHER

    def _fmt_ms(x: float) -> str:
        # Adaptive formatting so very small values don't appear as 0.0.
        ax = abs(float(x))
        if ax < 1.0:
            return f"{x:.3f}"
        if ax < 10.0:
            return f"{x:.2f}"
        if ax < 100.0:
            return f"{x:.1f}"
        return f"{x:.0f}"

    def _fmt_s(x: float) -> str:
        ax = abs(float(x))
        if ax < 0.01:
            return f"{x:.4f}"
        if ax < 0.1:
            return f"{x:.3f}"
        if ax < 1.0:
            return f"{x:.2f}"
        if ax < 10.0:
            return f"{x:.2f}"
        return f"{x:.1f}"

    def _add_stats_box(ax, stats: Dict[str, Any], role_label: str):
        # Expanded stats box with more percentiles as requested.
        # Using adaptive ms formatting so very small values don't appear as 0.0.
        txt = (
            f"{role_label}\n"
            f"n={stats['count']}  <5s={stats['under_5s_pct']:.1f}%\n"
            f"mean={_fmt_ms(stats['mean'])}ms  std={_fmt_ms(stats['std'])}ms\n"
            f"p10={_fmt_ms(stats['p10'])} p20={_fmt_ms(stats['p20'])} p25={_fmt_ms(stats['p25'])}\n"
            f"p40={_fmt_ms(stats['p40'])} p50={_fmt_ms(stats['p50'])} p75={_fmt_ms(stats['p75'])}\n"
            f"p90={_fmt_ms(stats['p90'])} p95={_fmt_ms(stats['p95'])} p99={_fmt_ms(stats['p99'])}"
        )
        ax.text(
            0.98,
            0.98,
            txt,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc", boxstyle="round,pad=0.35"),
        )

    def _make_edges(x_max_s: float, bin_w_s: float) -> np.ndarray:
        # Ensure last edge lands on a multiple of bin_w_s
        bin_w_s = float(bin_w_s)
        x_max_s = float(np.ceil(x_max_s / bin_w_s) * bin_w_s)
        n = int(round(x_max_s / bin_w_s))
        return np.arange(0.0, (n + 1) * bin_w_s, bin_w_s, dtype=float)

    def _hist_percent(vals_s: np.ndarray, edges_s: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(vals_s, bins=edges_s)
        total = max(len(vals_s), 1)
        return counts.astype(float) / total * 100.0

    def _hist_counts(vals_s: np.ndarray, edges_s: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(vals_s, bins=edges_s)
        return counts.astype(float)

    def _plot_hist_percent(ax, vals_ms: np.ndarray, edges_s: np.ndarray, color: str, label: str, collapse_at_s: float | None):
        vals_s = vals_ms / 1000.0
        if collapse_at_s is not None:
            vals_s = vals_s.copy()
            vals_s[vals_s >= collapse_at_s] = (edges_s[-1] - 1e-6)
        pct = _hist_percent(vals_s, edges_s)
        ax.bar(
            edges_s[:-1],
            pct,
            width=np.diff(edges_s),
            align="edge",
            edgecolor="black",
            alpha=0.8,
            color=color,
            label=label,
        )
        return pct

    def _annotate_cdf_percentiles(
        ax,
        vals_s_full: np.ndarray,
        *,
        x_cap_s: float,
        percentiles: list[int],
    ):
        """
        Denote percentile -> latency mapping directly on the CDF panel.
        We draw a small marker at (x, p) (clamped to x_cap_s) and print the label at the right edge at y=p.
        """
        if vals_s_full.size == 0:
            return
        xlim = ax.get_xlim()
        x_right = xlim[1] - 0.015 * (xlim[1] - xlim[0])
        for p in percentiles:
            x = float(np.percentile(vals_s_full, p))
            x_plot = min(x, x_cap_s - 1e-6)
            ax.scatter([x_plot], [p], s=16, color="#444444", zorder=6, label="_nolegend_")
            if x <= x_cap_s:
                label = f"P{p}={_fmt_s(x)}s"
            else:
                label = f"P{p}={_fmt_s(x)}s (> {x_cap_s:g}s)"
            ax.text(
                x_right,
                p,
                label,
                ha="right",
                va="center",
                fontsize=8,
                color="#333333",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.8),
            )

    for name, values in latencies.items():
        if not values:
            continue

        # Layout: histogram on top, CDF below. Share x-axis so ticks align.
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(14, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.0]},
        )

        # Histogram
        ax1 = axes[0]
        arr = np.array(values)

        stats = compute_stats(values)
        role_label = _role_and_tool_label(name)

        # We plot histogram as "% of calls per bin", not raw count.
        # Binning: 0.5s bins for experts (requested), with a gpt-oss-20b special tail bin.
        is_gptoss = name.startswith("expert:") and ("openai/gpt-oss-20b" in name)
        if is_gptoss:
            # 0.5s bins up to 9s; rightmost shown tick label should be real max seconds (tail)
            collapse_at_s = 9.0
            x_max_display_s = 9.5  # last visible bin [9.0, 9.5) represents tail (>=9.0s)
            edges_s = _make_edges(x_max_display_s, 0.5)
            max_s = float(np.max(arr) / 1000.0)
            series_color = _series_color(name)
            _plot_hist_percent(
                ax1,
                arr,
                edges_s,
                color=series_color,
                label="Histogram (% per 0.5s bin; tail collapsed into last bin)",
                collapse_at_s=collapse_at_s,
            )
            ax1.set_xlim(0.0, x_max_display_s)
            # Ticks: keep clean; label the last tick with the real max.
            ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5]
            ax1.set_xticks(ticks)
            ax1.set_xticklabels([f"{t:g}s" if t != 9.5 else f"{max_s:.0f}s (tail)" for t in ticks])
        else:
            # Default: show up to ~p99 (slightly padded) and collapse tail into the last bin.
            x_max_s = float(np.ceil((stats["p99"] * 1.10) / 1000.0 / 0.5) * 0.5)
            x_max_s = max(x_max_s, 3.0)
            edges_s = _make_edges(x_max_s, 0.5)
            series_color = _series_color(name)
            _plot_hist_percent(
                ax1,
                arr,
                edges_s,
                color=series_color,
                label="Histogram (% per 0.5s bin; tail collapsed)",
                collapse_at_s=x_max_s,
            )
            ax1.set_xlim(0.0, float(edges_s[-1]))
            # Ticks every 1s for readability.
            max_tick = float(edges_s[-1])
            tick_vals = list(np.arange(0.0, max_tick + 1e-9, 1.0))
            ax1.set_xticks(tick_vals)
            ax1.set_xticklabels([f"{t:g}s" for t in tick_vals])

        ax1.axvline(stats["p50"] / 1000.0, color='r', linestyle='--', label=f"P50: {stats['p50']:.0f}ms")
        ax1.axvline(stats["p95"] / 1000.0, color='orange', linestyle='--', label=f"P95: {stats['p95']:.0f}ms")
        ax1.axvline(5.0, color='green', linestyle=':', label=f"5s SLA (<5s={stats['under_5s_pct']:.1f}%)")
        ax1.set_xlabel("Latency (s)")
        ax1.set_ylabel("Percent of calls (%)")
        ax1.set_title("Histogram (% of calls per 0.5s bin)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        _add_stats_box(ax1, stats, role_label)

        # CDF
        ax2 = axes[1]
        # Use the same truncated/collapsed view as the histogram for alignment.
        vals_s_full = (arr / 1000.0).copy()
        vals_s = vals_s_full.copy()
        if is_gptoss:
            vals_s[vals_s >= 9.0] = edges_s[-1] - 1e-6
            note = f"CDF shown up to 9s; tail collapsed (max={float(np.max(arr)/1000.0):.0f}s)"
        else:
            # collapse at the right edge
            collapse_at_s = float(edges_s[-1])
            vals_s[vals_s >= collapse_at_s] = edges_s[-1] - 1e-6
            note = f"CDF x-axis truncated at {edges_s[-1]:g}s; tail collapsed"
        sorted_vals = np.sort(vals_s)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100
        ax2.plot(sorted_vals, cdf, 'b-', linewidth=2)
        ax2.axvline(5.0, color='green', linestyle=':', label=f"5s SLA (<5s={stats['under_5s_pct']:.1f}%)")
        # Mark the actual CDF at 5s explicitly
        ax2.scatter([5.0], [stats["under_5s_pct"]], color="green", s=18, zorder=5)
        ax2.annotate(
            f"<5s: {stats['under_5s_pct']:.1f}%",
            xy=(5.0, stats["under_5s_pct"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color="green",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
        )
        ax2.set_xlabel("Latency (s)")
        ax2.set_ylabel("CDF (%)")
        ax2.set_title("CDF (aligned x-axis)")
        ax2.set_ylim(0, 100)
        ax2.text(
            0.02,
            0.06,
            note,
            transform=ax2.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="#dddddd", boxstyle="round,pad=0.25"),
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # Keep CDF cleaner; the histogram already has the full stats box.
        _annotate_cdf_percentiles(
            ax2,
            vals_s_full,
            x_cap_s=float(edges_s[-1]),
            percentiles=[10, 20, 25, 40, 50, 75, 90, 95],
        )

        fig.suptitle(
            f"Tau2-Bench tool use latency distribution — {role_label} — model={name}",
            fontsize=12,
            y=0.98,
        )

        # Ensure top axis also shows x tick labels (requested).
        ax1.tick_params(axis="x", labelbottom=True)

        plt.tight_layout()

        # Sanitize filename
        safe_name = name.replace('/', '_').replace(':', '_')
        chart_path = output_dir / f"{safe_name}_latency.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"  Generated: {chart_path}")

    # --- All experts (stacked) + All tools (experts + local python tools) ---
    # Both truncated to 10s with 0.5s bins (requested) and show component mix per bin.
    def _expert_component_series() -> list[tuple[str, np.ndarray, str]]:
        comps: list[tuple[str, np.ndarray, str]] = []
        for k, v in latencies.items():
            if not (k.startswith("expert:") and v):
                continue
            label = _role_and_tool_label(k).replace("Tool: ", "")
            comps.append((label, np.array(v, dtype=float), _series_color(k)))
        # Keep stable ordering: Expert-1/2/3 if present
        def _rank(lbl: str) -> int:
            if "Expert-1" in lbl:
                return 0
            if "Expert-2" in lbl:
                return 1
            if "Expert-3" in lbl:
                return 2
            return 99
        comps.sort(key=lambda x: _rank(x[0]))
        return comps

    def _python_tool_values() -> np.ndarray:
        vals: list[float] = []
        for (_domain, _tool), v in data.get("tool_calls", {}).items():
            if v:
                vals.extend(v)
        return np.array(vals, dtype=float)

    def _plot_stacked_components(out_name: str, title: str, components: list[tuple[str, np.ndarray, str]]):
        all_vals = np.concatenate([c[1] for c in components if len(c[1]) > 0]) if components else np.array([], dtype=float)
        if all_vals.size == 0:
            return
        stats = compute_stats(all_vals.tolist())

        x_max_s = 10.0
        # Requested: all-tools plot uses finer bins (0.2s); keep others at 0.5s.
        bin_w_s = 0.2 if out_name == "all_tools_including_python_latency.png" else 0.5
        edges_s = _make_edges(x_max_s, bin_w_s)
        total_all = max(int(all_vals.size), 1)

        # Layout: Top (Histogram) + Bottom (CDF), increased height to avoid overlap.
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(16, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 0.6], "hspace": 0.15},
        )
        ax1, ax2 = axes[0], axes[1]

        bottom = np.zeros(len(edges_s) - 1, dtype=float)
        for label, vals_ms, color in components:
            vals_s = (vals_ms / 1000.0).copy()
            vals_s[vals_s >= x_max_s] = edges_s[-1] - 1e-6
            # Normalize by GLOBAL total so stacked bars sum to 100% max.
            counts = _hist_counts(vals_s, edges_s)
            pct = counts / total_all * 100.0
            ax1.bar(
                edges_s[:-1],
                pct,
                width=np.diff(edges_s),
                align="edge",
                edgecolor="black",
                alpha=0.85,
                color=color,
                label=f"{label} (n={len(vals_ms)})",
                bottom=bottom,
            )
            bottom += pct

        ax1.axvline(5.0, color="green", linestyle=":", label=f"5s SLA (<5s={stats['under_5s_pct']:.1f}%)")
        ax1.set_xlabel("Latency (s)")
        ax1.set_ylabel("Percent of calls (%)")
        ax1.set_title(f"Histogram (% of calls per {bin_w_s:g}s bin, stacked by component)")
        ax1.grid(True, alpha=0.3)
        # Move legend outside to the right to avoid overlapping with the Stats Box
        ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=10, framealpha=0.9)
        _add_stats_box(ax1, stats, title)
        ax1.set_xlim(0.0, edges_s[-1])
        tick_step = 1.0
        tick_vals = list(np.arange(0.0, edges_s[-1] + 1e-9, tick_step))
        ax1.set_xticks(tick_vals)
        ax1.set_xticklabels([f"{t:g}s" for t in tick_vals])
        ax1.tick_params(axis="x", labelbottom=True)  # Show x-ticks on top panel too

        # Denote total % on every bin (use the stacked total in `bottom`)
        # Keep labels horizontal as requested.
        for i, total_pct in enumerate(bottom):
            x_center = edges_s[i] + (edges_s[i + 1] - edges_s[i]) / 2.0
            # Avoid placing text exactly at y=0
            y = float(total_pct) + 0.2 if total_pct > 0 else 0.2
            ax1.text(
                x_center,
                y,
                f"{total_pct:.2f}%",
                ha="center",
                va="bottom",
                fontsize=5 if bin_w_s <= 0.2 else 7,
                rotation=0,
                color="#111111",
                clip_on=True,
            )
        
        all_s = (all_vals / 1000.0).copy()
        all_s[all_s >= x_max_s] = edges_s[-1] - 1e-6
        sorted_vals = np.sort(all_s)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100
        ax2.plot(sorted_vals, cdf, "b-", linewidth=2)

        ax2.axvline(5.0, color="green", linestyle=":", label="5s SLA")
        ax2.scatter([5.0], [stats["under_5s_pct"]], color="green", s=18, zorder=5)
        ax2.annotate(
            f"<5s: {stats['under_5s_pct']:.1f}%",
            xy=(5.0, stats["under_5s_pct"]),
            xytext=(8, -12),
            textcoords="offset points",
            fontsize=10,
            color="green",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
        )
        ax2.set_xlabel("Latency (s)")
        ax2.set_ylabel("CDF (%)")
        ax2.set_title("CDF (aligned x-axis)")
        ax2.set_ylim(0, 100)
        ax2.text(
            0.98,
            0.08,
            "CDF x-axis truncated at 10s; tail collapsed into last bin",
            transform=ax2.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="#dddddd", boxstyle="round,pad=0.25"),
        )
        ax2.grid(True, alpha=0.3)
        # Denote percentiles directly on the CDF curve (requested).
        _annotate_cdf_percentiles(
            ax2,
            (all_vals / 1000.0).copy(),
            x_cap_s=float(edges_s[-1]),
            percentiles=[10, 20, 25, 40, 50, 75, 90, 95],
        )

        fig.suptitle(title, fontsize=13, y=0.98)
        # Manually adjust spacing to avoid overlap/clipping, allocating room for right legend
        plt.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.82, hspace=0.25)
        # plt.tight_layout() # Removed to respect manual adjustments
        chart_path = output_dir / out_name
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"  Generated: {chart_path}")

    # All experts (3 components)
    expert_components = _expert_component_series()
    _plot_stacked_components(
        out_name="experts_all3_combined_latency.png",
        title="Tau2-Bench all experts tool use latency distribution (stacked by expert)",
        components=expert_components,
    )

    # All tools (experts + local python tools as tool calls)
    py_vals = _python_tool_values()
    components_all_tools = list(expert_components)
    if py_vals.size > 0:
        components_all_tools.append(("local python tools", py_vals, COLOR_PYTOOLS))
    _plot_stacked_components(
        out_name="all_tools_including_python_latency.png",
        title="Tau2-Bench all tools (experts + local python) tool use latency distribution (stacked by component)",
        components=components_all_tools,
    )


def generate_markdown_report(data: Dict[str, Any], output_path: Path):
    """Generate detailed markdown report."""
    latencies = data["latencies"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Tau2-Bench OSS Evaluation Report",
        f"**Date**: {timestamp}",
        "",
        "## 1. Model Latency Statistics (ms)",
        "",
        "| Model | Count | Mean | Std | Min | Max | P10 | P25 | P40 | P50 | P90 | P95 | P99 | <5s% |",
        "|-------|-------|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|",
    ]

    for name in sorted(latencies.keys()):
        values = latencies[name]
        if not values:
            continue
        stats = compute_stats(values)
        lines.append(
            f"| {name} | {stats['count']} | {stats['mean']:.1f} | {stats['std']:.1f} | "
            f"{stats['min']:.1f} | {stats['max']:.1f} | {stats['p10']:.1f} | {stats['p25']:.1f} | "
            f"{stats['p40']:.1f} | {stats['p50']:.1f} | {stats['p90']:.1f} | {stats['p95']:.1f} | "
            f"{stats['p99']:.1f} | {stats['under_5s_pct']:.1f} |"
        )

    # Prefill/Decode breakdown
    lines.extend([
        "",
        "## 2. vLLM Prefill/Decode Breakdown (ms)",
        "",
        "| Model | Prefill Mean | Prefill P50 | Decode Mean | Decode P50 | Input Tokens | Output Tokens |",
        "|-------|--------------|-------------|-------------|------------|--------------|---------------|",
    ])

    for name in sorted(data["prefill"].keys()):
        prefill_vals = data["prefill"].get(name, [])
        decode_vals = data["decode"].get(name, [])
        input_vals = data["input_tokens"].get(name, [])
        output_vals = data["output_tokens"].get(name, [])

        if not prefill_vals:
            continue

        prefill_stats = compute_stats(prefill_vals)
        decode_stats = compute_stats(decode_vals)
        input_mean = np.mean(input_vals) if input_vals else 0
        output_mean = np.mean(output_vals) if output_vals else 0

        lines.append(
            f"| {name} | {prefill_stats.get('mean', 0):.1f} | {prefill_stats.get('p50', 0):.1f} | "
            f"{decode_stats.get('mean', 0):.1f} | {decode_stats.get('p50', 0):.1f} | "
            f"{input_mean:.0f} | {output_mean:.0f} |"
        )

    # Throughput
    lines.extend([
        "",
        "## 3. Throughput (tokens/sec)",
        "",
        "| Model | Total Calls | Avg Input tok/call | Avg Output tok/call | Throughput (output tok/s) |",
        "|-------|-------------|--------------------|--------------------|---------------------------|",
    ])

    for name in sorted(latencies.keys()):
        lat_vals = latencies[name]
        input_vals = data["input_tokens"].get(name, [])
        output_vals = data["output_tokens"].get(name, [])

        if not lat_vals or not output_vals:
            continue

        total_output = sum(output_vals)
        total_time_s = sum(lat_vals) / 1000  # ms to sec
        throughput = total_output / total_time_s if total_time_s > 0 else 0

        lines.append(
            f"| {name} | {len(lat_vals)} | {np.mean(input_vals) if input_vals else 0:.0f} | "
            f"{np.mean(output_vals):.0f} | {throughput:.1f} |"
        )

    # Local tool latency
    lines.extend([
        "",
        "## 4. Local Tool Latency (ms)",
        "",
        "| Domain | Tool | Count | Mean | P50 | P95 |",
        "|--------|------|-------|------|-----|-----|",
    ])

    for (domain, tool), values in sorted(data["tool_calls"].items()):
        if not values:
            continue
        stats = compute_stats(values)
        lines.append(
            f"| {domain} | {tool} | {stats['count']} | {stats['mean']:.1f} | "
            f"{stats['p50']:.1f} | {stats['p95']:.1f} |"
        )

    # Step times
    lines.extend([
        "",
        "## 5. Per-Step Completion Time (ms)",
        "",
        "| Domain | Count | Mean | P50 | P95 | P99 |",
        "|--------|-------|------|-----|-----|-----|",
    ])

    for domain, values in sorted(data["step_times"].items()):
        if not values:
            continue
        stats = compute_stats(values)
        lines.append(
            f"| {domain} | {stats['count']} | {stats['mean']:.1f} | "
            f"{stats['p50']:.1f} | {stats['p95']:.1f} | {stats['p99']:.1f} |"
        )

    lines.extend([
        "",
        "---",
        f"*Generated by analyze_profile.py at {timestamp}*",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Markdown report saved to: {output_path}")


def main():
    if len(sys.argv) < 2:
        # Default: analyze all tau2_*.log files in logs_oss/
        log_dirs = [Path("logs_oss"), Path("logs_oss_full")]
        log_files = []
        for log_dir in log_dirs:
            if log_dir.exists():
                log_files.extend(log_dir.glob("tau2_*.log"))

        if not log_files:
            print("Usage: python analyze_profile.py <log_file> [log_file2 ...]")
            print("Or run from tau2-bench dir to auto-detect logs_oss/tau2_*.log")
            sys.exit(1)
    else:
        log_files = [Path(f) for f in sys.argv[1:]]

    # Aggregate data from all log files
    aggregated = {
        "latencies": defaultdict(list),
        "prefill": defaultdict(list),
        "decode": defaultdict(list),
        "input_tokens": defaultdict(list),
        "output_tokens": defaultdict(list),
        "tool_calls": defaultdict(list),
        "step_times": defaultdict(list),
    }

    for log_file in log_files:
        print(f"Parsing: {log_file}")
        data = parse_profile_logs(str(log_file))
        for key in aggregated:
            if isinstance(aggregated[key], defaultdict):
                for k, v in data[key].items():
                    aggregated[key][k].extend(v)

    # Print analysis to terminal
    analyze_and_print(aggregated)

    # Generate JSON output
    json_output = {}
    for name, values in aggregated["latencies"].items():
        json_output[name] = {
            "stats": compute_stats(values),
            "histogram": compute_histogram(values, n_bins=10),
            "prefill_stats": compute_stats(aggregated["prefill"].get(name, [])),
            "decode_stats": compute_stats(aggregated["decode"].get(name, [])),
        }

    output_path = Path("profile_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"\nJSON output saved to: {output_path}")

    # Generate PNG charts
    chart_dir = Path("profile_charts")
    generate_png_charts(aggregated, chart_dir)

    # Generate markdown report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(f"experiment_report_{timestamp}.md")
    generate_markdown_report(aggregated, report_path)


if __name__ == "__main__":
    main()
