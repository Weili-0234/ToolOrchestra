#!/usr/bin/env python3
"""
Watch a TR-new restart-per-C sweep TSV and append windowed metrics into a markdown table.

Sweep TSV format (tab-separated, header included):
  ts  C  trials_completed  trials_per_min  out_dir

For each completed out_dir, we extract (10–130min window):
  - throughput: trials/min + steps/sec
  - vLLM server metrics from prefix_cache_timeseries.csv (sampled every 2s)
  - per-request cached_tokens/prompt_tokens from orchestrator_usage.jsonl
  - GPU SM util mean from gpu_sm_util_timeseries.csv
  - TR-new router step profile means (prefill/decode/pause/tool_call/kv_hit_rate) aligned to the same window:
      join `tr_profiles/step_profiles.csv` with `[PROFILE] type=step_complete ...` markers in eval_driver.log.

Output markdown is append-only per (out_dir), and includes precise metric definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
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
                "# HLE PreExp: DP=1 ThunderReact-new (TR-new) (restart per C) (10–130min window)",
                "",
                "- Each C is a cold restart of the TR-new router + Orchestrator-8B vLLM backend (clears KV/prefix cache).",
                "- `t0` = first successful orchestrator response that logged usage (from `orchestrator_usage.jsonl`).",
                "- Window = `[t0+10min, t0+130min]` = `[600s, 7800s]` after `t0` (2 hours).",
                "- vLLM `/metrics` sampled every 2s into `prefix_cache_timeseries.csv` (backend metrics).",
                "- TR-new router profiling is enabled; `tr_profiles/step_profiles.csv` is joined with step timestamps from `eval_driver.log` to compute windowed means.",
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
                "- TR-new router step profile columns (windowed means, joined by `(program_id, step_id)`):",
                "  - `tr_prefill_s_mean`: mean of router-observed time from request arrival to first token (seconds).",
                "  - `tr_decode_s_mean`: mean of router-observed time from first token to last token (seconds).",
                "  - `tr_pause_s_mean`: mean of router-observed pause time (seconds) due to TR scheduling (0 if not paused).",
                "  - `tr_tool_call_s_mean`: mean of time between last token of the previous request and arrival of the next request (seconds); computed over matched rows with `step_id>0`.",
                "  - `tr_kv_hit_rate_mean`: mean of router-observed KV hit rate `cached_tokens/prompt_tokens` per request (0–1), excluding missing values.",
                "",
                "## Results",
                "",
                "### Run Context",
                "| C | t0_iso | window_start_iso | window_end_iso | out_dir |",
                "| --- | --- | --- | --- | --- |",
                "",
                "### Artifacts",
                "| C | ports_json | tr_step_profiles_csv | window_summary |",
                "| --- | --- | --- | --- |",
                "",
                "### Throughput & Cache",
                "| C | trials/min | steps/sec | server_hit_ratio_prefill_tokens | request_hit_avg | request_hit_token_weighted | kv_cache_usage_mean_perc | gpu_sm_util_mean |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                "",
                "### Latency & Server Preemption",
                "| C | e2e_s | decode_s | decode_over_e2e | delta_preemptions_mean_per_2s | preemptions_per_sec |",
                "| --- | --- | --- | --- | --- | --- |",
                "",
                "### TR-new Router Step Profile (windowed)",
                "| C | tr_steps_matched | tr_steps_missing_profile | tr_prefill_s_mean | tr_decode_s_mean | tr_pause_s_mean | tr_tool_call_s_mean | tr_kv_hit_rate_mean |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
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


_STEP_COMPLETE_RE = re.compile(r"\btype=step_complete\b")


def _iter_step_complete_events(eval_log: Path) -> List[Tuple[float, str, int]]:
    """
    Returns: list of (ts_unix, program_id, step_id)
    program_id is taken from log field job_id=... (eval uses job_id=program_id for compatibility).
    """
    if not eval_log.exists():
        return []
    out: List[Tuple[float, str, int]] = []
    for line in _read_text(eval_log).splitlines():
        if not _STEP_COMPLETE_RE.search(line):
            continue
        m_ts = re.search(r"\bts_unix=(?P<ts>\d+(?:\.\d+)?)\b", line)
        m_job = re.search(r"\bjob_id=(?P<jid>\S+)\b", line)
        m_step = re.search(r"\bstep=(?P<step>\d+)\b", line)
        if not (m_ts and m_job and m_step):
            continue
        try:
            ts = float(m_ts.group("ts"))
            step = int(m_step.group("step"))
        except Exception:
            continue
        jid = m_job.group("jid")
        out.append((ts, jid, step))
    return out


def _load_tr_step_profiles(step_profiles_csv: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    if not step_profiles_csv.exists():
        return {}
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with step_profiles_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = (row.get("program_id") or "").strip()
            if not pid:
                continue
            try:
                step_id = int(row.get("step_id") or 0)
            except Exception:
                continue
            def _f(k: str) -> Optional[float]:
                v = (row.get(k) or "").strip()
                if not v:
                    return None
                try:
                    return float(v)
                except Exception:
                    return None
            out[(pid, step_id)] = {
                "prefill_s": _f("prefill_s"),
                "decode_s": _f("decode_s"),
                "pause_s": _f("pause_s"),
                "tool_call_s": _f("tool_call_s"),
                "kv_hit_rate": _f("kv_hit_rate"),
            }
    return out


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _tr_profile_window_means(
    *,
    eval_log: Path,
    step_profiles_csv: Path,
    t_start: float,
    t_end: float,
) -> Dict[str, Any]:
    events = _iter_step_complete_events(eval_log)
    prof = _load_tr_step_profiles(step_profiles_csv)

    matched = 0
    missing = 0

    prefill: List[float] = []
    decode: List[float] = []
    pause: List[float] = []
    tool_call: List[float] = []
    kv_hit: List[float] = []

    for ts, pid, step_id in events:
        if ts < t_start or ts > t_end:
            continue
        rec = prof.get((pid, step_id))
        if rec is None:
            missing += 1
            continue
        matched += 1
        if isinstance(rec.get("prefill_s"), (int, float)):
            prefill.append(float(rec["prefill_s"]))
        if isinstance(rec.get("decode_s"), (int, float)):
            decode.append(float(rec["decode_s"]))
        if isinstance(rec.get("pause_s"), (int, float)):
            pause.append(float(rec["pause_s"]))
        if step_id > 0 and isinstance(rec.get("tool_call_s"), (int, float)):
            tool_call.append(float(rec["tool_call_s"]))
        if isinstance(rec.get("kv_hit_rate"), (int, float)):
            kv_hit.append(float(rec["kv_hit_rate"]))

    return {
        "tr_steps_matched": matched,
        "tr_steps_missing_profile": missing,
        "tr_prefill_s_mean": _mean(prefill),
        "tr_decode_s_mean": _mean(decode),
        "tr_pause_s_mean": _mean(pause),
        "tr_tool_call_s_mean": _mean(tool_call),
        "tr_kv_hit_rate_mean": _mean(kv_hit),
    }


def _extract_one(
    *,
    python_bin: str,
    out_dir: Path,
    window_start_sec: float,
    window_end_sec: float,
    repo_root: Path,
) -> Optional[Dict[str, Any]]:
    window_summary = out_dir / "window_summary.json"
    usage_jsonl = out_dir / "orchestrator_usage.jsonl"
    eval_log = out_dir / "eval_driver.log"
    gpu_csv = out_dir / "gpu_sm_util_timeseries.csv"
    ports_json = out_dir / "ports.json"
    step_profiles_csv = out_dir / "tr_profiles" / "step_profiles.csv"

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

    m = re.search(r"_c(?P<c>\d+)_", out_dir.name)
    c_val = int(m.group("c")) if m else 0

    tr_prof = _tr_profile_window_means(
        eval_log=eval_log,
        step_profiles_csv=step_profiles_csv,
        t_start=t_start,
        t_end=t_end,
    )

    return {
        "C": c_val,
        "t0_iso": ws.get("t0_first_request_iso"),
        "window_start_iso": win.get("window_start_iso"),
        "window_end_iso": win.get("window_end_iso"),
        "trials_per_min": (float(trials.get("trials_per_sec")) * 60.0)
        if isinstance(trials.get("trials_per_sec"), (int, float))
        else None,
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
        **tr_prof,
        "out_dir": str(out_dir),
        "ports_json": str(ports_json) if ports_json.exists() else "n/a",
        "tr_step_profiles_csv": str(step_profiles_csv) if step_profiles_csv.exists() else "n/a",
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
    insert_at = header_idx + 2  # after header row + separator row
    while insert_at < len(lines) and lines[insert_at].startswith("|") and lines[insert_at].count("|") >= 2:
        insert_at += 1
    return lines[:insert_at] + [row_line] + lines[insert_at:]


def _append_rows(*, report_md: Path, row: Dict[str, Any]) -> None:
    lines = _read_text(report_md).splitlines()
    c = int(row.get("C") or 0)

    lines = _insert_table_row(
        lines,
        header="| C | t0_iso | window_start_iso | window_end_iso | out_dir |",
        row_line=f"| {c} | {_fmt(row.get('t0_iso'))} | {_fmt(row.get('window_start_iso'))} | {_fmt(row.get('window_end_iso'))} | {row.get('out_dir')} |",
    )
    lines = _insert_table_row(
        lines,
        header="| C | ports_json | tr_step_profiles_csv | window_summary |",
        row_line=f"| {c} | {row.get('ports_json')} | {row.get('tr_step_profiles_csv')} | {row.get('window_summary')} |",
    )
    lines = _insert_table_row(
        lines,
        header="| C | trials/min | steps/sec | server_hit_ratio_prefill_tokens | request_hit_avg | request_hit_token_weighted | kv_cache_usage_mean_perc | gpu_sm_util_mean |",
        row_line="| "
        + " | ".join(
            [
                str(c),
                _fmt(row.get("trials_per_min")),
                _fmt(row.get("steps_per_sec")),
                _fmt(row.get("server_hit_ratio_prefill_tokens")),
                _fmt(row.get("request_hit_avg")),
                _fmt(row.get("request_hit_token_weighted")),
                _fmt(row.get("kv_cache_usage_mean_perc")),
                _fmt(row.get("gpu_sm_util_mean")),
            ]
        )
        + " |",
    )
    lines = _insert_table_row(
        lines,
        header="| C | e2e_s | decode_s | decode_over_e2e | delta_preemptions_mean_per_2s | preemptions_per_sec |",
        row_line="| "
        + " | ".join(
            [
                str(c),
                _fmt(row.get("e2e_s")),
                _fmt(row.get("decode_s")),
                _fmt(row.get("decode_over_e2e")),
                _fmt(row.get("delta_preemptions_mean_per_2s")),
                _fmt(row.get("preemptions_per_sec")),
            ]
        )
        + " |",
    )
    lines = _insert_table_row(
        lines,
        header="| C | tr_steps_matched | tr_steps_missing_profile | tr_prefill_s_mean | tr_decode_s_mean | tr_pause_s_mean | tr_tool_call_s_mean | tr_kv_hit_rate_mean |",
        row_line="| "
        + " | ".join(
            [
                str(c),
                _fmt(row.get("tr_steps_matched")),
                _fmt(row.get("tr_steps_missing_profile")),
                _fmt(row.get("tr_prefill_s_mean")),
                _fmt(row.get("tr_decode_s_mean")),
                _fmt(row.get("tr_pause_s_mean")),
                _fmt(row.get("tr_tool_call_s_mean")),
                _fmt(row.get("tr_kv_hit_rate_mean")),
            ]
        )
        + " |",
    )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=None, help="Path to sweep TSV (default: newest outputs/hle_trnew_restart_per_c_*.tsv)")
    ap.add_argument("--report-md", default=None, help="Path to output markdown report")
    ap.add_argument("--repo-root", default=None, help="Path to compare repo root (contains hle-bench-rollout/ and outputs/)")
    ap.add_argument("--interval-sec", type=float, default=30.0)
    ap.add_argument("--window-start-sec", type=float, default=600.0)
    ap.add_argument("--window-end-sec", type=float, default=7800.0)
    ap.add_argument("--once", action="store_true", help="Run one scan then exit")
    args = ap.parse_args()

    if args.repo_root:
        repo_root = Path(args.repo_root).expanduser().resolve()
    else:
        repo_root = Path.cwd().resolve()
        if not (repo_root / "hle-bench-rollout").exists():
            script_path = Path(__file__).absolute()
            for p in script_path.parents:
                if (p / "hle-bench-rollout").exists() and (p / "outputs").exists():
                    repo_root = p
                    break

    if args.tsv:
        tsv = Path(args.tsv)
    else:
        cand = sorted((repo_root / "outputs").glob("hle_trnew_restart_per_c_*.tsv"), key=lambda p: p.stat().st_mtime)
        tsv = cand[-1] if cand else (repo_root / "outputs" / "hle_trnew_restart_per_c.tsv")

    if args.report_md:
        report_md = Path(args.report_md)
    else:
        report_md = repo_root / "hle-bench-rollout" / "hle_preexp_dp1_trnew_restart_per_c_report_10_130.md"

    python_bin = sys.executable or "python"

    _ensure_report(report_md)
    state = _load_state(report_md)
    processed: set[str] = set(state.get("processed_out_dirs") or [])

    while True:
        rows = _iter_tsv_rows(tsv)
        any_new = False

        for r in rows:
            out_dir_s = r.get("out_dir") or ""
            if not out_dir_s:
                continue
            if out_dir_s in processed:
                continue

            out_dir = Path(out_dir_s)
            if not out_dir.exists():
                continue

            extracted = _extract_one(
                python_bin=python_bin,
                out_dir=out_dir,
                window_start_sec=float(args.window_start_sec),
                window_end_sec=float(args.window_end_sec),
                repo_root=repo_root,
            )
            if extracted is None:
                continue

            _append_rows(report_md=report_md, row=extracted)
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

