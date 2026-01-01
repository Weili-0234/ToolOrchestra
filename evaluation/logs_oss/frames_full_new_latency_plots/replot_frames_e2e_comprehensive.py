#!/usr/bin/env python3
"""
FRAMES E2E Latency Comprehensive Plotting Script
=================================================

Context:
This script visualizes the End-to-End (E2E) latency distributions for the
"FRAMES OSS Full New" experiment (Dec 31, 2025).

Experiment Details:
- Task: FRAMES Benchmark (Full Run, 824 tasks)
- Models:
    - Search: openai/gpt-oss-20b (vLLM) + Wiki Retrieval (GPU FAISS optimized)
    - Answer: Qwen/Qwen3-32B-FP8 (vLLM)
    - Enhance Reasoning: Qwen/Qwen2.5-Coder-14B-Instruct (vLLM)
- Data Source: `outputs_oss_frames_full_new/*.json` (Per-task output JSONs)

Outputs:
1. Three Per-Tool Stacked Histograms (Search, Answer, Enhance):
   - X-axis: Latency (seconds), truncated to tool-specific ranges.
   - Y-axis (Left): % of calls per bin.
   - Y-axis (Right): CDF (Cumulative Distribution Function) %.
   - Stacking: Breakdown by component (e.g., LLM vs Retrieval).
   - Annotations: P50/P90/P99 percentiles, tail bin (>max) flush right.

2. One "All Tools" Stacked Histogram:
   - X-axis: Latency (0-30s).
   - Stacking: Breakdown by Tool Type (Search vs Answer vs Enhance).
   - Includes CDF and percentiles on the aggregate distribution.

Usage:
  cd /home/junxiong/haokang/ToolOrchestra/evaluation
  python logs_oss/frames_full_new_latency_plots/replot_frames_e2e_comprehensive.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# --- Configuration ---

# Path to the experiment outputs (relative to evaluation/ dir)
LOG_GLOB = "outputs_oss_frames_full_new/*.json"

# Plot settings per tool
CONFIGS = {
    "search": {
        "x_max": 15.0, "bins": 30, "color_base": "green",
        "components": [("_query_llm_ms", "Query LLM"), ("_retrieval_ms", "Retrieval")]
    },
    "answer": {
        "x_max": 20.0, "bins": 40, "color_base": "orange",
        "components": [("_expert_ms", "Expert"), ("_judge_ms", "Judge")]
    },
    "enhance_reasoning": {
        "x_max": 25.0, "bins": 50, "color_base": "blue",
        "components": [("_llm_ms", "LLM"), ("_exec_ms", "Exec")]
    }
}

# --- Data Collection ---

def load_data(glob_pattern):
    data = {k: [] for k in CONFIGS}
    all_data = []

    paths = list(Path(".").glob(glob_pattern))
    if not paths:
        script_dir = Path(__file__).parent
        paths = list(script_dir.glob("../../" + glob_pattern))

    print(f"Found {len(paths)} JSON files matching {glob_pattern}")

    for f in paths:
        try:
            d = json.load(open(f))
            for turn, responses in d.get("all_tool_responses", {}).items():
                for r in responses:
                    tool = r.get("tool")
                    if tool not in CONFIGS: continue

                    cfg = CONFIGS[tool]
                    comp_vals = []
                    total = 0.0
                    for field, label in cfg["components"]:
                        v = r.get(field) or 0.0
                        comp_vals.append(v / 1000.0)
                        total += v

                    total_s = total / 1000.0
                    if total_s > 0:
                        data[tool].append({"total": total_s, "comps": comp_vals})
                        all_data.append({"total": total_s, "tool": tool})
        except Exception:
            pass

    return data, all_data

# --- Plotting Utils ---

def get_colors(base, n=2):
    if base == "green": return ["#a1d99b", "#31a354", "#006d2c"][:n]
    if base == "orange": return ["#fdbe85", "#e6550d", "#a63603"][:n]
    if base == "blue": return ["#9ecae1", "#3182bd", "#08519c"][:n]
    return ["gray"] * n

def plot_histogram(items, x_max, n_bins, title, out_path, mode="component_ratio", category_labels=None, colors=None):
    if not items:
        print(f"No items to plot for {title}")
        return

    totals = np.array([x["total"] for x in items])
    n_total = len(totals)
    true_max = totals.max()

    edges = np.linspace(0, x_max, n_bins + 1)
    bin_width = edges[1] - edges[0]

    main_mask = totals <= x_max
    tail_mask = totals > x_max

    n_cats = len(category_labels)

    if mode == "component_ratio":
        bin_counts = np.zeros(n_bins + 1)
        bin_comp_sums = np.zeros((n_bins + 1, n_cats))

        bin_idxs = np.searchsorted(edges, totals, side='right') - 1
        bin_idxs[tail_mask] = n_bins
        bin_idxs = np.clip(bin_idxs, 0, n_bins)

        for i, idx in enumerate(bin_idxs):
            bin_counts[idx] += 1
            bin_comp_sums[idx] += items[i]["comps"]

        total_time_in_bin = bin_comp_sums.sum(axis=1)
        ratios = np.divide(bin_comp_sums, total_time_in_bin[:, None], out=np.zeros_like(bin_comp_sums), where=total_time_in_bin[:, None]!=0)

        raw_heights_pct = (bin_counts / n_total) * 100.0
        stack_heights = ratios * raw_heights_pct[:, None]

    elif mode == "categorical_count":
        cat_map = {name: i for i, name in enumerate(category_labels)}
        stack_heights = np.zeros((n_bins + 1, n_cats))

        bin_idxs = np.searchsorted(edges, totals, side='right') - 1
        bin_idxs[tail_mask] = n_bins
        bin_idxs = np.clip(bin_idxs, 0, n_bins)

        for i, idx in enumerate(bin_idxs):
            cat = items[i]["category"]
            c_idx = cat_map.get(cat, -1)
            if c_idx >= 0:
                stack_heights[idx, c_idx] += 1

        stack_heights = (stack_heights / n_total) * 100.0

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(14, 7), constrained_layout=True)
    ax2 = ax1.twinx()

    bottoms_main = np.zeros(n_bins)
    bottom_tail = 0.0

    for c in range(n_cats):
        heights = stack_heights[:n_bins, c]
        ax1.bar(edges[:-1], heights, width=bin_width, bottom=bottoms_main, align='edge',
                color=colors[c], edgecolor='black', linewidth=0.3, alpha=0.85, label=category_labels[c])
        bottoms_main += heights

    tail_pos = x_max
    for c in range(n_cats):
        h = stack_heights[n_bins, c]
        ax1.bar(tail_pos, h, width=bin_width, bottom=bottom_tail, align='edge',
                color=colors[c], edgecolor='black', linewidth=0.3, alpha=0.6, hatch='//')
        bottom_tail += h

    sorted_vals = np.sort(totals)
    y_vals = np.arange(1, n_total + 1) / n_total * 100.0
    x_plotted = np.clip(sorted_vals, 0, x_max)
    ax2.plot(x_plotted, y_vals, color='red', linewidth=1.5, label='CDF')
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("CDF (%)", color='red', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='red')

    ps = [50, 80, 90, 99]
    p_vals = np.percentile(totals, ps)
    for p, v in zip(ps, p_vals):
        if v <= x_max:
            ax1.axvline(v, color='red', linestyle=':', linewidth=1, alpha=0.8)
            ax1.text(v, ax1.get_ylim()[1], f"P{p}\n{v:.1f}s", color='red', ha='center', va='bottom', fontsize=8,
                     bbox=dict(facecolor='white', edgecolor='red', alpha=0.7, boxstyle='round,pad=0.1'))

    xticks = list(np.arange(0, x_max + 0.01, max(1.0, x_max/10)))
    ax1.set_xticks(xticks + [tail_pos + bin_width/2])
    ax1.set_xticklabels([f"{int(x)}s" for x in xticks] + [f"{int(true_max)}s (tail)"])

    ax1.set_xlim(0, x_max + bin_width * 1.5)
    ax1.set_xlabel("Latency (seconds)")
    ax1.set_ylabel("% of all calls")

    mean = totals.mean()
    med = np.median(totals)
    p80 = np.percentile(totals, 80)
    p90 = np.percentile(totals, 90)
    lt5s_pct = (totals < 5).mean() * 100
    ax1.set_title(f"{title}\nn={n_total}, mean={mean:.2f}s, p50={med:.2f}s, p80={p80:.2f}s, p90={p90:.2f}s | <5s: {lt5s_pct:.1f}% | bin={bin_width:.2f}s")

    ax1.grid(axis='y', alpha=0.2)
    ax1.axvline(x_max, color='gray', linestyle='--', linewidth=1)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")
    plt.close(fig)

# --- Main Execution ---

if __name__ == "__main__":
    print("Collecting FRAMES latency data...")
    per_tool, all_tool = load_data(LOG_GLOB)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"logs_oss/frames_full_new_latency_plots")

    # Print summary stats
    print("\n=== FRAMES Latency Summary ===")
    for tool, items in per_tool.items():
        if items:
            totals = np.array([x["total"] for x in items])
            print(f"{tool}: n={len(totals)}, mean={totals.mean():.2f}s, p50={np.median(totals):.2f}s, p80={np.percentile(totals, 80):.2f}s, p90={np.percentile(totals, 90):.2f}s, <5s={100*(totals<5).mean():.1f}%")
    print()

    # 1. Per-Tool Plots
    for tool, items in per_tool.items():
        cfg = CONFIGS[tool]
        cat_labels = [c[1] for c in cfg["components"]]
        colors = get_colors(cfg["color_base"], len(cat_labels))

        plot_histogram(
            items,
            x_max=cfg["x_max"],
            n_bins=cfg["bins"],
            title=f"FRAMES E2E Latency: {tool}",
            out_path=out_dir / f"frames_full_new_e2e_{tool}.png",
            mode="component_ratio",
            category_labels=cat_labels,
            colors=colors
        )

    # 2. All-Tools Plot
    all_items_cat = [{"total": x["total"], "category": x["tool"]} for x in all_tool]
    cats = ["search", "answer", "enhance_reasoning"]
    cols = ["tab:green", "tab:orange", "tab:blue"]

    plot_histogram(
        all_items_cat,
        x_max=30.0,
        n_bins=60,
        title="FRAMES E2E Latency: ALL Tools",
        out_path=out_dir / "frames_full_new_e2e_all_tools.png",
        mode="categorical_count",
        category_labels=cats,
        colors=cols
    )

    print(f"\nAll plots saved to: {out_dir.resolve()}")
