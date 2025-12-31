#!/usr/bin/env python3
"""
Analyze tau2-bench profiling logs.
Outputs per-expert latency stats with enhanced metrics:
- Percentiles: p10, p25, p40, p50, p90, p95, p99
- SLA compliance: <5s%
- Prefill/decode breakdown for vLLM models
- PNG charts with matplotlib
- Detailed markdown report
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
    """Compute enhanced statistics including p10, p25, p40 and <5s%."""
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
        "p25": float(np.percentile(arr, 25)),
        "p40": float(np.percentile(arr, 40)),
        "p50": float(np.percentile(arr, 50)),
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

    for name, values in latencies.items():
        if not values:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1 = axes[0]
        arr = np.array(values)
        ax1.hist(arr, bins=20, edgecolor='black', alpha=0.7)
        ax1.axvline(np.percentile(arr, 50), color='r', linestyle='--', label=f'P50: {np.percentile(arr, 50):.1f}ms')
        ax1.axvline(np.percentile(arr, 95), color='orange', linestyle='--', label=f'P95: {np.percentile(arr, 95):.1f}ms')
        ax1.axvline(5000, color='green', linestyle=':', label='5s SLA')
        ax1.set_xlabel('Latency (ms)')
        ax1.set_ylabel('Count')
        ax1.set_title(f'{name} - Latency Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # CDF
        ax2 = axes[1]
        sorted_vals = np.sort(arr)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100
        ax2.plot(sorted_vals, cdf, 'b-', linewidth=2)
        ax2.axhline(50, color='r', linestyle='--', alpha=0.5, label='P50')
        ax2.axhline(95, color='orange', linestyle='--', alpha=0.5, label='P95')
        ax2.axvline(5000, color='green', linestyle=':', label='5s SLA')
        ax2.set_xlabel('Latency (ms)')
        ax2.set_ylabel('Percentile (%)')
        ax2.set_title(f'{name} - CDF')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Sanitize filename
        safe_name = name.replace('/', '_').replace(':', '_')
        chart_path = output_dir / f"{safe_name}_latency.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"  Generated: {chart_path}")


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
