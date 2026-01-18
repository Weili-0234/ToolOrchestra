#!/usr/bin/env python3
"""
Watch 5090 TAU2Bench outputs for one method and append fixed-window metrics to markdown tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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


def _ensure_report(report_md: Path, method: str) -> None:
    if report_md.exists():
        return
    report_md.parent.mkdir(parents=True, exist_ok=True)
    title_map = {
        "baseline": "TAU2Bench Serving: Baseline vLLM",
        "tr": "TAU2Bench Serving: ThunderReact-new (TR-new)",
        "continuum": "TAU2Bench Serving: vLLM-Continuum",
    }
    title = title_map.get(method, f"TAU2Bench Serving: {method}")
    report_md.write_text(
        "\n".join(
            [
                f"# {title} (10–130min window)",
                "",
                "- `t0` = first orchestrator response timestamp in `orchestrator_usage.jsonl`.",
                "- Window = `[t0+10min, t0+130min]` = `[600s, 7800s]` after `t0`.",
                "- Rows are appended in completion order (based on `window_summary.json` mtime).",
                "",
                "## Throughput",
                "| C | rep | tasks/min | steps/sec | steps/min |",
                "| --- | --- | --- | --- | --- |",
                "",
                "## Cache & Utilization",
                "| C | rep | server_hit_ratio | request_hit_avg | request_hit_token_weighted | kv_usage_mean_perc | gpu_sm_util_mean |",
                "| --- | --- | --- | --- | --- | --- | --- |",
                "",
                "## Latency",
                "| C | rep | decode_s | e2e_s | decode_over_e2e |",
                "| --- | --- | --- | --- | --- |",
                "",
                "## Preemptions",
                "| C | rep | delta_preemptions_mean_2s | preemptions_total_window | out_dir |",
                "| --- | --- | --- | --- | --- |",
                "",
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


def _parse_c_rep(out_dir_name: str) -> Tuple[int, int]:
    m = re.search(r"_c(?P<c>\d+)_rep(?P<rep>\d+)_", out_dir_name)
    if not m:
        return 0, 0
    return int(m.group("c")), int(m.group("rep"))


def _extract_one(
    *,
    out_dir: Path,
    window_start_sec: float,
    window_end_sec: float,
) -> Optional[Dict[str, Any]]:
    # Guard against polluted runs where the TR router failed to start and eval traffic
    # hit a stale router (port already in use).
    tr_router_log = out_dir / "logs" / "tr_router.log"
    if tr_router_log.exists():
        txt = _read_text(tr_router_log)
        if "Errno 98" in txt or "address already in use" in txt:
            return None

    window_summary = out_dir / "window_summary.json"
    steps_summary = out_dir / "steps_summary.json"
    gpu_csv = out_dir / "gpu_sm_util_timeseries.csv"

    if not window_summary.exists():
        return None
    try:
        ws = _load_json(window_summary)
    except Exception:
        return None
    if not isinstance(ws, dict) or ws.get("ok") is not True:
        return None

    key = f"{int(window_start_sec)}-{int(window_end_sec)}"
    win = (ws.get("windows") or {}).get(key) or {}
    if not isinstance(win, dict):
        return None

    metrics = win.get("metrics") or {}
    if isinstance(metrics, dict) and metrics.get("available") is False:
        return None
    usage = win.get("response_usage") or {}
    trials = win.get("trials") or {}

    pre = (metrics.get("preemptions") or {}) if isinstance(metrics, dict) else {}
    lat = (metrics.get("latency_means_seconds") or {}) if isinstance(metrics, dict) else {}

    t0_unix = ws.get("t0_first_request_unix")
    if not isinstance(t0_unix, (int, float)):
        return None
    t_start = float(t0_unix) + float(window_start_sec)
    t_end = float(t0_unix) + float(window_end_sec)

    steps_per_sec = None
    if steps_summary.exists():
        try:
            ss = _load_json(steps_summary)
            steps_block = ss.get("steps") if isinstance(ss, dict) else None
            if isinstance(steps_block, dict):
                steps_per_sec = steps_block.get("steps_per_sec")
        except Exception:
            steps_per_sec = None

    tasks_per_min = None
    if isinstance(trials, dict) and isinstance(trials.get("trials_per_sec"), (int, float)):
        tasks_per_min = float(trials.get("trials_per_sec")) * 60.0

    c_val, rep_val = _parse_c_rep(out_dir.name)
    steps_per_min = (float(steps_per_sec) * 60.0) if isinstance(steps_per_sec, (int, float)) else None

    return {
        "C": c_val,
        "rep": rep_val,
        "tasks_per_min": tasks_per_min,
        "steps_per_sec": steps_per_sec,
        "steps_per_min": steps_per_min,
        "server_hit_ratio": metrics.get("hit_ratio") if isinstance(metrics, dict) else None,
        "request_hit_avg": usage.get("cached_tokens_ratio_request_avg") if isinstance(usage, dict) else None,
        "request_hit_token_weighted": usage.get("cached_tokens_ratio_token_weighted") if isinstance(usage, dict) else None,
        "kv_usage_mean_perc": metrics.get("kv_cache_usage_mean_perc") if isinstance(metrics, dict) else None,
        "gpu_sm_util_mean": _gpu_sm_mean(gpu_csv, t_start, t_end),
        "decode_s": lat.get("request_decode_time_seconds") if isinstance(lat, dict) else None,
        "e2e_s": lat.get("e2e_request_latency_seconds") if isinstance(lat, dict) else None,
        "decode_over_e2e": metrics.get("decode_over_e2e_ratio") if isinstance(metrics, dict) else None,
        "delta_preemptions_mean_2s": (pre.get("delta_preemptions_mean_per_interval") if isinstance(pre, dict) else None),
        "preemptions_total_window": (pre.get("sum_delta_preemptions") if isinstance(pre, dict) else None),
        "out_dir": str(out_dir),
    }


def _insert_table_row(lines: List[str], *, header: str, row_line: str) -> List[str]:
    header_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == header.strip():
            header_idx = i
            break
    if header_idx < 0:
        return lines + [row_line]
    insert_at = header_idx + 2
    return lines[:insert_at] + [row_line] + lines[insert_at:]


def _append_rows(report_md: Path, rec: Dict[str, Any]) -> None:
    lines = _read_text(report_md).splitlines()

    thr_header = "| C | rep | tasks/min | steps/sec | steps/min |"
    thr_row = f"| {rec['C']} | {rec['rep']} | {_fmt(rec.get('tasks_per_min'))} | {_fmt(rec.get('steps_per_sec'))} | {_fmt(rec.get('steps_per_min'))} |"
    lines = _insert_table_row(lines, header=thr_header, row_line=thr_row)

    cu_header = "| C | rep | server_hit_ratio | request_hit_avg | request_hit_token_weighted | kv_usage_mean_perc | gpu_sm_util_mean |"
    cu_row = (
        f"| {rec['C']} | {rec['rep']} | {_fmt(rec.get('server_hit_ratio'))} | {_fmt(rec.get('request_hit_avg'))} | "
        f"{_fmt(rec.get('request_hit_token_weighted'))} | {_fmt(rec.get('kv_usage_mean_perc'))} | {_fmt(rec.get('gpu_sm_util_mean'))} |"
    )
    lines = _insert_table_row(lines, header=cu_header, row_line=cu_row)

    lat_header = "| C | rep | decode_s | e2e_s | decode_over_e2e |"
    lat_row = f"| {rec['C']} | {rec['rep']} | {_fmt(rec.get('decode_s'))} | {_fmt(rec.get('e2e_s'))} | {_fmt(rec.get('decode_over_e2e'))} |"
    lines = _insert_table_row(lines, header=lat_header, row_line=lat_row)

    pre_header = "| C | rep | delta_preemptions_mean_2s | preemptions_total_window | out_dir |"
    pre_row = (
        f"| {rec['C']} | {rec['rep']} | {_fmt(rec.get('delta_preemptions_mean_2s'))} | "
        f"{_fmt(rec.get('preemptions_total_window'))} | `{rec.get('out_dir')}` |"
    )
    lines = _insert_table_row(lines, header=pre_header, row_line=pre_row)

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _discover_out_dirs(repo_dir: Path, method: str, since_unix: float) -> List[Path]:
    out_root = repo_dir / "outputs"
    if not out_root.exists():
        return []
    out_dirs = []
    prefix = f"tau2_serving_5090_{method}_"
    for p in out_root.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith(prefix):
            continue
        try:
            st = p.stat()
        except Exception:
            continue
        if st.st_mtime < since_unix:
            continue
        out_dirs.append(p)
    out_dirs.sort(key=lambda x: x.stat().st_mtime)
    return out_dirs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["baseline", "continuum", "tr"])
    ap.add_argument("--report-md", required=True)
    ap.add_argument("--repo-dir", default=None)
    ap.add_argument("--interval-sec", type=float, default=60.0)
    ap.add_argument("--since-unix", type=float, default=None)
    ap.add_argument("--window-start-sec", type=float, default=600.0)
    ap.add_argument("--window-end-sec", type=float, default=7800.0)
    args = ap.parse_args()

    method = args.method
    report_md = Path(args.report_md)
    repo_dir = Path(args.repo_dir).resolve() if args.repo_dir else Path(__file__).resolve().parents[3]

    since_unix = float(args.since_unix) if args.since_unix is not None else (time.time() - 3600.0)
    window_start_sec = float(args.window_start_sec)
    window_end_sec = float(args.window_end_sec)

    _ensure_report(report_md, method)
    state = _load_state(report_md)
    processed = set(map(str, state.get("processed_out_dirs", [])))

    while True:
        out_dirs = _discover_out_dirs(repo_dir, method, since_unix)
        updated = False
        for out_dir in out_dirs:
            out_dir_s = str(out_dir)
            if out_dir_s in processed:
                continue
            rec = _extract_one(out_dir=out_dir, window_start_sec=window_start_sec, window_end_sec=window_end_sec)
            if rec is None:
                continue
            _append_rows(report_md, rec)
            processed.add(out_dir_s)
            state["processed_out_dirs"] = sorted(processed)
            _save_state(report_md, state)
            updated = True
        if updated:
            print(f"[watch] appended new rows to {report_md}", flush=True)
        time.sleep(float(args.interval_sec))


if __name__ == "__main__":
    raise SystemExit(main())

