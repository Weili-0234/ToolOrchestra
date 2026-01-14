#!/usr/bin/env python3
"""
Watch a Continuum restart-per-C sweep TSV and append windowed metrics into a markdown table.

This is intended to be run as a lightweight background process (e.g. via nohup).

Sweep TSV format (tab-separated, header included):
  ts  C  trials_completed  trials_per_min  scheduler_timestamps_path  out_dir

For each completed out_dir, we extract (10–130min window):
  - throughput: trials/min + steps/sec
  - vLLM server metrics from prefix_cache_timeseries.csv (sampled every 2s)
  - per-request cached_tokens/prompt_tokens from orchestrator_usage.jsonl
  - GPU SM util mean from gpu_sm_util_timeseries.csv
  - Continuum preemption breakdown from scheduler_timestamps (see analyze_continuum_preemption_breakdown.py)

Output markdown is append-only per (out_dir), and includes precise metric definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x != x:  # NaN
            return "n/a"
        return f"{x:.6f}"
    return str(x)


def _ensure_report(report_md: Path) -> None:
    if report_md.exists():
        return
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(
        "\n".join(
            [
                "# HLE PreExp: DP=1 vLLM-Continuum (restart per C) (10–130min window)",
                "",
                "- Each C is a cold restart of the Continuum orchestrator backend + router (clears KV/prefix cache).",
                "- `t0` = first successful orchestrator response that logged usage (from `orchestrator_usage.jsonl`).",
                "- Window = `[t0+10min, t0+130min]` = `[600s, 7800s]` after `t0` (2 hours).",
                "- vLLM `/metrics` sampled every 2s into `prefix_cache_timeseries.csv`.",
                "",
                "## Metric Definitions",
                "- `server_hit_ratio_prefill_tokens`: token-level prefill/prefix-cache hit ratio from vLLM `/metrics` counters, aggregated over the window as `sum(Δprefix_cache_hits_total) / sum(Δprefix_cache_queries_total)`.",
                "- `request_hit_avg`: per-request average of `cached_tokens/prompt_tokens` across successful orchestrator responses in the window (from `orchestrator_usage.jsonl`).",
                "- `request_hit_token_weighted`: token-weighted version of the above: `sum(cached_tokens)/sum(prompt_tokens)` in the window.",
                "- `kv_cache_usage_mean_perc`: mean of vLLM gauge `vllm:kv_cache_usage_perc` over the window (units are the raw vLLM gauge, typically 0–1 or 0–100 depending on build).",
                "- `gpu_sm_util_mean`: mean of NVML-sampled SM utilization (%) over the window (sampled every 2s).",
                "- `e2e_s`: window-weighted mean of vLLM histogram `vllm:e2e_request_latency_seconds` (prefill+decode end-to-end request latency).",
                "- `decode_s`: window-weighted mean of vLLM histogram `vllm:request_decode_time_seconds`.",
                "- `decode_over_e2e`: `decode_s / e2e_s` (window means).",
                "- `delta_preemptions_mean_per_2s`: mean of every-2s `Δ(vllm:num_preemptions*)` over the window (from vLLM `/metrics`).",
                "- `preemptions_per_sec`: `sum(Δpreemptions) / window_duration_sec` (from vLLM `/metrics`).",
                "- `preempt_events_total`: Continuum scheduler timestamp events `Request_evicted_from_running_queue_time` in the window (from `scheduler_timestamps`).",
                "- `preemption_rate_events_per_request` (MAIN): `preempt_events_total / total_requests` within the window (requests counted by `Request_arrival_time`).",
                "- `request_preempt_rate`: fraction of requests (arrivals) that experienced ≥1 preemption event within the window.",
                "- `job_preempt_rate`: fraction of jobs (unique job_id with ≥1 arrival in window) that experienced ≥1 preemption event within the window.",
                "- Preemption event types (from `scheduler_timestamps`):",
                "  - `running_with_pin`: preempted while running AND previous step's KV was still pinned at this request arrival.",
                "  - `running_after_unpin`: preempted while running AND previous step's KV had been unpinned before this request arrival.",
                "  - `pinned_with_pin`: a pinned KV cache entry (after a request finished) was evicted during its pin TTL.",
                "",
                "## Results",
                "",
                "### Run Context",
                "| C | t0_iso | window_start_iso | window_end_iso | out_dir | window_summary | scheduler_timestamps |",
                "| --- | --- | --- | --- | --- | --- | --- |",
                "",
                "### Throughput & Cache",
                "| C | trials/min | steps/sec | server_hit_ratio_prefill_tokens | request_hit_avg | request_hit_token_weighted | kv_cache_usage_mean_perc | gpu_sm_util_mean |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                "",
                "### Latency & Server Preemption",
                "| C | e2e_s | decode_s | decode_over_e2e | delta_preemptions_mean_per_2s | preemptions_per_sec |",
                "| --- | --- | --- | --- | --- | --- |",
                "",
                "### Continuum Preemption Rates (`scheduler_timestamps`)",
                "| C | total_requests | preempt_events_total | preemption_rate_events_per_request | preempted_requests | request_preempt_rate | total_jobs | jobs_preempted | job_preempt_rate |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                "",
                "### Continuum Preemption Types",
                "| C | preempt_run_with_pin | preempt_run_with_pin_share | preempt_run_after_unpin | preempt_run_after_unpin_share | preempt_pinned_with_pin | preempt_pinned_with_pin_share |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _state_path(report_md: Path) -> Path:
    return report_md.with_suffix(report_md.suffix + ".state.json")


def _load_state(report_md: Path) -> Dict[str, Any]:
    p = _state_path(report_md)
    if not p.exists():
        return {"processed_out_dirs": []}
    try:
        obj = json.loads(_read_text(p))
        if isinstance(obj, dict):
            if "processed_out_dirs" not in obj or not isinstance(obj["processed_out_dirs"], list):
                obj["processed_out_dirs"] = []
            return obj
    except Exception:
        pass
    return {"processed_out_dirs": []}


def _save_state(report_md: Path, state: Dict[str, Any]) -> None:
    p = _state_path(report_md)
    p.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _iter_tsv_rows(tsv: Path) -> List[Dict[str, str]]:
    if not tsv.exists():
        return []
    out: List[Dict[str, str]] = []
    with tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            out.append({k.strip(): (v.strip() if isinstance(v, str) else "") for k, v in row.items()})
    return out


def _gpu_sm_mean(gpu_csv: Path, t_start: float, t_end: float) -> Optional[float]:
    if not gpu_csv.exists():
        return None
    sm_sum = 0.0
    sm_n = 0
    try:
        with gpu_csv.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    ts = float(row.get("ts_unix", "nan"))
                    sm = float(row.get("sm_util", "nan"))
                except Exception:
                    continue
                if ts != ts or sm != sm:
                    continue
                if ts < t_start or ts > t_end:
                    continue
                sm_sum += sm
                sm_n += 1
    except Exception:
        return None
    return (sm_sum / sm_n) if sm_n > 0 else None


def _run_json(python_bin: str, argv: List[str]) -> Dict[str, Any]:
    out = subprocess.check_output([python_bin, *argv], text=True)
    return json.loads(out)


def _extract_one(
    *,
    python_bin: str,
    out_dir: Path,
    scheduler_timestamps: Path,
    window_start_sec: float,
    window_end_sec: float,
    repo_root: Path,
) -> Optional[Dict[str, Any]]:
    window_summary = out_dir / "window_summary.json"
    usage_jsonl = out_dir / "orchestrator_usage.jsonl"
    eval_log = out_dir / "eval_driver.log"
    gpu_csv = out_dir / "gpu_sm_util_timeseries.csv"

    if not window_summary.exists() or not usage_jsonl.exists() or not eval_log.exists():
        return None

    ws = _load_json(window_summary)
    if not isinstance(ws, dict) or ws.get("ok") is not True:
        return None
    key = f"{int(window_start_sec)}-{int(window_end_sec)}"
    win = (ws.get("windows") or {}).get(key) or {}
    if not isinstance(win, dict):
        return None

    t0_unix = ws.get("t0_first_request_unix")
    if not isinstance(t0_unix, (int, float)):
        return None
    t_start = float(t0_unix) + float(window_start_sec)
    t_end = float(t0_unix) + float(window_end_sec)

    metrics = win.get("metrics") or {}
    usage = win.get("response_usage") or {}
    trials = win.get("trials") or {}
    pre = (metrics.get("preemptions") or {}) if isinstance(metrics, dict) else {}
    lat = (metrics.get("latency_means_seconds") or {}) if isinstance(metrics, dict) else {}

    # steps/sec from [PROFILE] type=step_complete markers in eval_driver.log
    steps_tool = repo_root / "oss-ToolOrchestra" / "scripts" / "preexp" / "summarize_steps_per_sec_window.py"
    steps_out: Dict[str, Any] = {}
    try:
        steps_out = _run_json(
            python_bin,
            [
                str(steps_tool),
                "--usage-jsonl",
                str(usage_jsonl),
                "--eval-log",
                str(eval_log),
                "--start-offset-sec",
                str(window_start_sec),
                "--end-offset-sec",
                str(window_end_sec),
            ],
        )
    except Exception:
        steps_out = {}
    steps_block = steps_out.get("steps") if isinstance(steps_out, dict) else None
    if not isinstance(steps_block, dict):
        steps_block = {}

    # Continuum preemption breakdown from scheduler_timestamps
    preempt_tool = (
        repo_root
        / "oss-ToolOrchestra"
        / "scripts"
        / "hle_preexp"
        / "analyze_continuum_preemption_breakdown.py"
    )
    preempt_out: Dict[str, Any] = {}
    try:
        preempt_out = _run_json(
            python_bin,
            [
                str(preempt_tool),
                "--timestamps",
                str(scheduler_timestamps),
                "--usage-jsonl",
                str(usage_jsonl),
                "--start-offset-sec",
                str(window_start_sec),
                "--end-offset-sec",
                str(window_end_sec),
            ],
        )
    except Exception:
        preempt_out = {}

    totals = (preempt_out.get("totals") or {}) if isinstance(preempt_out, dict) else {}
    br = (preempt_out.get("preemption_event_breakdown") or {}) if isinstance(preempt_out, dict) else {}

    def _br(name: str) -> Tuple[Optional[int], Optional[float]]:
        x = br.get(name) if isinstance(br, dict) else None
        if not isinstance(x, dict):
            return None, None
        ev = x.get("events")
        sh = x.get("share")
        return (int(ev) if isinstance(ev, int) else None, float(sh) if isinstance(sh, (int, float)) else None)

    rwp_ev, rwp_sh = _br("running_with_pin")
    rau_ev, rau_sh = _br("running_after_unpin")
    pwp_ev, pwp_sh = _br("pinned_with_pin")

    m = re.search(r"_c(?P<c>\d+)_", out_dir.name)
    c_val = int(m.group("c")) if m else 0

    return {
        "C": c_val,
        "t0_iso": ws.get("t0_first_request_iso"),
        "window_start_iso": win.get("window_start_iso"),
        "window_end_iso": win.get("window_end_iso"),
        "trials_per_min": (float(trials.get("trials_per_sec")) * 60.0) if isinstance(trials.get("trials_per_sec"), (int, float)) else None,
        "steps_per_sec": steps_block.get("steps_per_sec"),
        "server_hit_ratio_prefill_tokens": metrics.get("hit_ratio") if isinstance(metrics, dict) else None,
        "request_hit_avg": usage.get("cached_tokens_ratio_request_avg") if isinstance(usage, dict) else None,
        "request_hit_token_weighted": usage.get("cached_tokens_ratio_token_weighted") if isinstance(usage, dict) else None,
        "kv_cache_usage_mean_perc": metrics.get("kv_cache_usage_mean_perc") if isinstance(metrics, dict) else None,
        "gpu_sm_util_mean": _gpu_sm_mean(gpu_csv, t_start, t_end),
        "e2e_s": lat.get("e2e_request_latency_seconds") if isinstance(lat, dict) else None,
        "decode_s": lat.get("request_decode_time_seconds") if isinstance(lat, dict) else None,
        "decode_over_e2e": metrics.get("decode_over_e2e_ratio") if isinstance(metrics, dict) else None,
        "delta_preemptions_mean_per_2s": (pre.get("delta_preemptions_mean_per_interval") if isinstance(pre, dict) else None),
        "preemptions_per_sec": (pre.get("preemptions_per_sec") if isinstance(pre, dict) else None),
        "total_requests": totals.get("total_requests"),
        "preempt_events_total": totals.get("preempt_events_total"),
        "preemption_rate_events_per_request": totals.get("preemption_rate_events_per_request"),
        "preempted_requests": totals.get("preempted_requests"),
        "request_preempt_rate": totals.get("request_preempt_rate"),
        "total_jobs": totals.get("total_jobs"),
        "jobs_preempted": totals.get("jobs_preempted"),
        "job_preempt_rate": totals.get("job_preempt_rate"),
        "preempt_run_with_pin": rwp_ev,
        "preempt_run_with_pin_share": rwp_sh,
        "preempt_run_after_unpin": rau_ev,
        "preempt_run_after_unpin_share": rau_sh,
        "preempt_pinned_with_pin": pwp_ev,
        "preempt_pinned_with_pin_share": pwp_sh,
        "scheduler_timestamps": str(scheduler_timestamps),
        "out_dir": str(out_dir),
        "window_summary": str(window_summary),
    }


def _insert_table_row(lines: List[str], *, header: str, row_line: str) -> List[str]:
    header_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == header:
            header_idx = i
            break
    if header_idx < 0:
        return lines + [row_line]

    # Expect separator next; append after last contiguous table row.
    start = header_idx + 2
    end = start
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        end += 1
    return lines[:end] + [row_line] + lines[end:]


def _append_rows(*, report_md: Path, repo_root: Path, row: Dict[str, Any]) -> None:
    def rel(p: Optional[str]) -> str:
        if not p:
            return "n/a"
        try:
            rp = Path(p).resolve()
            rr = repo_root.resolve()
            return str(rp.relative_to(rr))
        except Exception:
            return p

    ctx_header = "| C | t0_iso | window_start_iso | window_end_iso | out_dir | window_summary | scheduler_timestamps |"
    thr_header = "| C | trials/min | steps/sec | server_hit_ratio_prefill_tokens | request_hit_avg | request_hit_token_weighted | kv_cache_usage_mean_perc | gpu_sm_util_mean |"
    lat_header = "| C | e2e_s | decode_s | decode_over_e2e | delta_preemptions_mean_per_2s | preemptions_per_sec |"
    pr_header = "| C | total_requests | preempt_events_total | preemption_rate_events_per_request | preempted_requests | request_preempt_rate | total_jobs | jobs_preempted | job_preempt_rate |"
    pt_header = "| C | preempt_run_with_pin | preempt_run_with_pin_share | preempt_run_after_unpin | preempt_run_after_unpin_share | preempt_pinned_with_pin | preempt_pinned_with_pin_share |"

    ctx_row = (
        "| "
        + " | ".join(
            [
                _fmt(row.get("C")),
                _fmt(row.get("t0_iso")),
                _fmt(row.get("window_start_iso")),
                _fmt(row.get("window_end_iso")),
                f"`{rel(row.get('out_dir'))}`",
                f"`{rel(row.get('window_summary'))}`",
                f"`{rel(row.get('scheduler_timestamps'))}`",
            ]
        )
        + " |\n"
    )
    thr_row = (
        "| "
        + " | ".join(
            [
                _fmt(row.get("C")),
                _fmt(row.get("trials_per_min")),
                _fmt(row.get("steps_per_sec")),
                _fmt(row.get("server_hit_ratio_prefill_tokens")),
                _fmt(row.get("request_hit_avg")),
                _fmt(row.get("request_hit_token_weighted")),
                _fmt(row.get("kv_cache_usage_mean_perc")),
                _fmt(row.get("gpu_sm_util_mean")),
            ]
        )
        + " |\n"
    )
    lat_row = (
        "| "
        + " | ".join(
            [
                _fmt(row.get("C")),
                _fmt(row.get("e2e_s")),
                _fmt(row.get("decode_s")),
                _fmt(row.get("decode_over_e2e")),
                _fmt(row.get("delta_preemptions_mean_per_2s")),
                _fmt(row.get("preemptions_per_sec")),
            ]
        )
        + " |\n"
    )
    pr_row = (
        "| "
        + " | ".join(
            [
                _fmt(row.get("C")),
                _fmt(row.get("total_requests")),
                _fmt(row.get("preempt_events_total")),
                _fmt(row.get("preemption_rate_events_per_request")),
                _fmt(row.get("preempted_requests")),
                _fmt(row.get("request_preempt_rate")),
                _fmt(row.get("total_jobs")),
                _fmt(row.get("jobs_preempted")),
                _fmt(row.get("job_preempt_rate")),
            ]
        )
        + " |\n"
    )
    pt_row = (
        "| "
        + " | ".join(
            [
                _fmt(row.get("C")),
                _fmt(row.get("preempt_run_with_pin")),
                _fmt(row.get("preempt_run_with_pin_share")),
                _fmt(row.get("preempt_run_after_unpin")),
                _fmt(row.get("preempt_run_after_unpin_share")),
                _fmt(row.get("preempt_pinned_with_pin")),
                _fmt(row.get("preempt_pinned_with_pin_share")),
            ]
        )
        + " |\n"
    )

    lines = report_md.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    for header, row_line in [
        (pt_header, pt_row),
        (pr_header, pr_row),
        (lat_header, lat_row),
        (thr_header, thr_row),
        (ctx_header, ctx_row),
    ]:
        lines = _insert_table_row(lines, header=header, row_line=row_line)
    report_md.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=None, help="Path to sweep TSV (default: newest outputs/hle_continuum_restart_per_c_*.tsv)")
    ap.add_argument("--report-md", default=None, help="Path to output markdown report")
    ap.add_argument("--repo-root", default=None, help="Path to compare repo root (contains hle-bench-rollout/ and outputs/)")
    ap.add_argument("--interval-sec", type=float, default=30.0)
    ap.add_argument("--window-start-sec", type=float, default=600.0)
    ap.add_argument("--window-end-sec", type=float, default=7800.0)
    ap.add_argument("--once", action="store_true", help="Run one scan then exit")
    args = ap.parse_args()

    # NOTE: oss-ToolOrchestra is a symlink to ../ToolOrchestra, so __file__.resolve()
    # may point outside the compare repo. Prefer cwd, and allow explicit override.
    if args.repo_root:
        repo_root = Path(args.repo_root).expanduser().resolve()
    else:
        repo_root = Path.cwd().resolve()
        if not (repo_root / "hle-bench-rollout").exists():
            # Fallback: walk up from the *non-resolved* script path.
            script_path = Path(__file__).absolute()
            for p in script_path.parents:
                if (p / "hle-bench-rollout").exists() and (p / "outputs").exists():
                    repo_root = p
                    break

    if args.tsv:
        tsv = Path(args.tsv)
    else:
        cand = sorted((repo_root / "outputs").glob("hle_continuum_restart_per_c_*.tsv"), key=lambda p: p.stat().st_mtime)
        tsv = cand[-1] if cand else (repo_root / "outputs" / "hle_continuum_restart_per_c.tsv")

    if args.report_md:
        report_md = Path(args.report_md)
    else:
        report_md = repo_root / "hle-bench-rollout" / "hle_preexp_dp1_continuum_restart_per_c_report_10_130.md"

    python_bin = sys.executable or "python"

    _ensure_report(report_md)
    state = _load_state(report_md)
    processed: set[str] = set(state.get("processed_out_dirs") or [])

    while True:
        rows = _iter_tsv_rows(tsv)
        any_new = False

        for r in rows:
            out_dir_s = r.get("out_dir") or ""
            ts_path_s = r.get("scheduler_timestamps_path") or ""
            if not out_dir_s or not ts_path_s:
                continue
            if out_dir_s in processed:
                continue

            out_dir = Path(out_dir_s)
            ts_path = Path(ts_path_s)
            if not out_dir.exists() or not ts_path.exists():
                continue

            extracted = _extract_one(
                python_bin=python_bin,
                out_dir=out_dir,
                scheduler_timestamps=ts_path,
                window_start_sec=float(args.window_start_sec),
                window_end_sec=float(args.window_end_sec),
                repo_root=repo_root,
            )
            if extracted is None:
                continue

            _append_rows(report_md=report_md, repo_root=repo_root, row=extracted)
            processed.add(out_dir_s)
            any_new = True

        if any_new:
            state["processed_out_dirs"] = sorted(processed)
            _save_state(report_md, state)

        if args.once:
            break
        time.sleep(float(args.interval_sec))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
