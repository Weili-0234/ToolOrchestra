#!/usr/bin/env python3
"""
FRAMES E2E Latency Comprehensive Plotting Script (Revised)
==========================================================
Context:
This script visualizes the End-to-End (E2E) latency distributions for the
"FRAMES OSS Full New" experiment (Dec 31, 2025).

Features:
- Stacked histograms showing component breakdown (or tool breakdown).
- x-axis truncation with tail bin.
- Continuous CDF overlay (right y-axis).
- P10-P90 percentile lines (Green, dashed, single height).
- Comprehensive stats in title.

Usage:
  python replot_frames_e2e_comprehensive.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# --- Configuration ---
LOG_GLOB = "outputs_oss_frames_full_new/*.json"

CONFIGS = {
    "search": {
        "x_max": 15.0,  # FRAMES search is slower, set appropriate limit
        "bins": 30,     # 0.5s bins
        "color_base": "green",
        "components": [("_query_llm_ms", "LLM"), ("_retrieval_ms", "Retrieval")]
    },
    "answer": {
        "x_max": 20.0,
        "bins": 40,
        "color_base": "orange",
        "components": [("_expert_ms", "Expert"), ("_judge_ms", "Judge")]
    },
    "enhance_reasoning": {
        "x_max": 25.0,
        "bins": 50,
        "color_base": "blue",
        "components": [("_llm_ms", "LLM"), ("_exec_ms", "Exec")]
    }
}

def get_colors(base, n=2):
    if base == "green": return ["#a1d99b", "#31a354", "#006d2c"][:n]
    if base == "orange": return ["#fdbe85", "#e6550d", "#a63603"][:n]
    if base == "blue": return ["#9ecae1", "#3182bd", "#08519c"][:n]
    return ["gray"] * n

def load_data(glob_pattern):
    per_tool_data = {k: [] for k in CONFIGS}
    all_tool_data = [] 
    
    root_dir = Path("evaluation") if Path("evaluation").exists() else Path(".")
    # Adjust path if running from subdir or root
    if not list(root_dir.glob(glob_pattern)):
        # try one level up if not found, or absolute path if needed.
        # But user environment seems to run from repo root usually.
        pass

    files = list(root_dir.glob(glob_pattern))
    if not files:
        # try absolute path based on user context
        files = list(Path("/home/junxiong/haokang/ToolOrchestra/evaluation").glob(glob_pattern))

    print(f"Found {len(files)} JSON files matching {glob_pattern}")

    for f in files:
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
                        per_tool_data[tool].append({"total": total_s, "comps": comp_vals})
                        all_tool_data.append({"total": total_s, "tool": tool})
        except:
            pass
    return per_tool_data, all_tool_data

def plot_histogram(
    items, 
    x_max, 
    n_bins, 
    title, 
    out_path, 
    mode="component_ratio", 
    category_labels=None, 
    colors=None,
    extra_legend_info=None
):
    if not items: return

    totals = np.array([x["total"] for x in items])
    n_total = len(totals)
    true_max = totals.max()
    
    edges = np.linspace(0, x_max, n_bins + 1)
    bin_width = edges[1] - edges[0]
    
    main_mask = totals <= x_max
    tail_mask = totals > x_max
    
    # --- Stack Data ---
    if mode == "component_ratio":
        n_cats = len(items[0]["comps"])
        bin_counts = np.zeros(n_bins + 1)
        bin_comp_sums = np.zeros((n_bins + 1, n_cats))
        
        bin_idxs = np.searchsorted(edges, totals, side='right') - 1
        bin_idxs[tail_mask] = n_bins 
        bin_idxs = np.clip(bin_idxs, 0, n_bins)
        
        for i, idx in enumerate(bin_idxs):
            bin_counts[idx] += 1
            bin_comp_sums[idx] += items[i]["comps"]
            
        total_time_in_bin = bin_comp_sums.sum(axis=1)
        ratios = np.zeros_like(bin_comp_sums)
        mask = total_time_in_bin > 0
        ratios[mask] = bin_comp_sums[mask] / total_time_in_bin[mask][:, None]
        
        raw_heights_pct = (bin_counts / n_total) * 100.0
        stack_heights = ratios * raw_heights_pct[:, None]
        
    elif mode == "categorical_count":
        n_cats = len(category_labels)
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
    fig, ax1 = plt.subplots(figsize=(14, 8), constrained_layout=True)
    ax2 = ax1.twinx() # CDF
    
    # 1. Histogram
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

    # 2. CDF Line
    sorted_vals = np.sort(totals)
    y_vals = np.arange(1, n_total + 1) / n_total * 100.0
    x_plotted = np.clip(sorted_vals, 0, x_max)
    ax2.plot(x_plotted, y_vals, color='red', linewidth=1.5, label='CDF')
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("CDF (%)", color='red', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 3. Percentiles P10..P90 (Green, Single Height)
    ps = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    p_vals = np.percentile(totals, ps)
    
    # Calculate fixed height for labels (above the highest bar)
    max_bar_h = 0
    if n_bins > 0:
        max_bar_h = np.max(np.sum(stack_heights[:n_bins], axis=1))
    tail_h = np.sum(stack_heights[n_bins])
    y_ref = max(max_bar_h, tail_h)
    label_y = y_ref + 1.5  # Fixed height offset
    
    for p, v in zip(ps, p_vals):
        if v <= x_max:
            ax1.axvline(v, color='green', linestyle=':', linewidth=1, alpha=0.8)
            ax1.text(v, label_y, f"P{p}\n{v:.1f}s", color='green', ha='center', va='bottom', fontsize=8,
                     bbox=dict(facecolor='white', edgecolor='green', alpha=0.7, boxstyle='round,pad=0.1'))

    # 4. Ticks & Limits
    xticks = list(np.arange(0, x_max + 0.01, max(1.0, x_max/10)))
    ax1.set_xticks(xticks + [tail_pos + bin_width/2])
    ax1.set_xticklabels([f"{int(x)}s" for x in xticks] + [f"{int(true_max)}s (tail)"])
    
    ax1.set_xlim(0, x_max + bin_width * 1.5)
    ax1.set_ylim(0, label_y + 5.0) # Ensure headroom
    
    # Add extra info to legend if provided (e.g. for All Tools plot)
    if extra_legend_info:
        # Create a dummy handle for text
        from matplotlib.patches import Rectangle
        extra_h = [Rectangle((0,0), 1, 1, fill=False, edgecolor="none", visible=False) for _ in extra_legend_info]
        h1 = h1 + extra_h
        l1 = l1 + extra_legend_info

    ax1.set_xlabel("Latency (seconds)")
    ax1.set_ylabel("% of all calls")
    
    mean = totals.mean()
    med = np.median(totals)
    p90 = np.percentile(totals, 90)
    ax1.set_title(f"{title}\nn={n_total}, mean={mean:.2f}s, p50={med:.2f}s, p90={p90:.2f}s | bin={bin_width:.2f}s")
    
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right')
    
    ax1.grid(axis='y', alpha=0.2)
    ax1.axvline(x_max, color='gray', linestyle='--', linewidth=1)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")
    plt.close(fig)

# --- Main Execution ---

if __name__ == "__main__":
    print("Collecting FRAMES latency data...")
    per_tool, all_tool = load_data(LOG_GLOB)

    out_dir = Path("logs_oss/frames_full_new_latency_plots")
    
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
    all_items_cat = [{"total": x["total"], "category": x["tool"]} for x in all_tool]
        
    cats = ["search", "answer", "enhance_reasoning"]
    cols = ["tab:green", "tab:orange", "tab:blue"]
    total_calls = len(all_items_cat)
    counts = {c: 0 for c in cats}
    for item in all_items_cat:
        counts[item["category"]] += 1
    
    legend_info = [
        f"{c}: {counts[c]} ({counts[c]/total_calls*100:.1f}%)" 
        for c in cats
    ]

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
