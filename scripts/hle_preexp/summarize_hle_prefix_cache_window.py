#!/usr/bin/env python3
"""
Summarize HLE DP=1 throughput + prefix-cache hit-rate in fixed time windows.

Window definition (relative to t0 = first orchestrator response that logged usage):
  t0 is taken from orchestrator_usage.jsonl (TOOL_ORCH_USAGE_LOG_PATH).

Inputs:
  1) prefix_cache_timeseries.csv from scripts/hle_preexp/kv_prefix_cache_hit_sampler.sh
  2) orchestrator_usage.jsonl written by TOOL_ORCH_USAGE_LOG_PATH (LLM_CALL.py)
  3) eval_driver.log (optional): count HLE trials completed in-window via [HLE_TRIAL_COMPLETE] markers

Outputs JSON with:
  - t0
  - per-window:
    - /metrics-based token-weighted hit ratio (Δhits/Δqueries)
    - /metrics-based preemption severity (Δnum_preemptions per interval + per-sec)
    - response-usage-based cached_tokens/prompt_tokens ratio (token-weighted + request-avg)
    - trials/sec in the same window (if eval log provided)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_usage_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _first_request_ts(usage_recs: List[Dict[str, Any]]) -> Optional[float]:
    t0: Optional[float] = None
    for r in usage_recs:
        try:
            ts = float(r.get("ts_unix"))
            pt = float(r.get("prompt_tokens"))
            ct = float(r.get("cached_tokens"))
        except Exception:
            continue
        if pt <= 0:
            continue
        if ct < 0:
            continue
        if t0 is None or ts < t0:
            t0 = ts
    return t0


def _summarize_metrics_csv(path: Path, t_start: float, t_end: float) -> Dict[str, Any]:
    if not path.exists():
        return {"available": False}

    sum_hits = 0.0
    sum_queries = 0.0
    n_hit = 0

    kv_usage_sum = 0.0
    kv_usage_n = 0
    num_running_sum = 0.0
    num_running_n = 0
    num_waiting_sum = 0.0
    num_waiting_n = 0

    # Preemptions: per-interval deltas (already sampled as delta_preemptions).
    sum_preemptions = 0.0
    preempt_n = 0

    hist_bases = [
        "time_to_first_token_seconds",
        "inter_token_latency_seconds",
        "e2e_request_latency_seconds",
        "request_prefill_time_seconds",
        "request_decode_time_seconds",
    ]
    hist_sum: Dict[str, float] = {b: 0.0 for b in hist_bases}
    hist_cnt: Dict[str, float] = {b: 0.0 for b in hist_bases}

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = float(row["ts_unix"])
            except Exception:
                continue
            if ts < t_start or ts > t_end:
                continue

            # Gauges
            try:
                kv = float(row.get("kv_cache_usage_perc", "nan"))
                if kv == kv:
                    kv_usage_sum += kv
                    kv_usage_n += 1
            except Exception:
                pass
            try:
                nr = float(row.get("num_requests_running", "nan"))
                if nr == nr:
                    num_running_sum += nr
                    num_running_n += 1
            except Exception:
                pass
            try:
                nw = float(row.get("num_requests_waiting", "nan"))
                if nw == nw:
                    num_waiting_sum += nw
                    num_waiting_n += 1
            except Exception:
                pass

            # Histogram deltas (sum/count) for window-weighted mean
            for base in hist_bases:
                try:
                    dsum = float(row.get(f"delta_{base}_sum", "nan"))
                    dcnt = float(row.get(f"delta_{base}_count", "nan"))
                except Exception:
                    continue
                if dsum != dsum or dcnt != dcnt:
                    continue
                if dcnt <= 0 or dsum < 0:
                    continue
                hist_sum[base] += dsum
                hist_cnt[base] += dcnt

            # Token-level hit ratio counters
            try:
                dh = float(row.get("delta_hits", "nan"))
                dq = float(row.get("delta_queries", "nan"))
            except Exception:
                dh = dq = float("nan")
            if dq == dq and dq > 0 and dh == dh and dh >= 0:
                sum_hits += dh
                sum_queries += dq
                n_hit += 1

            # Preemptions deltas
            try:
                dp = float(row.get("delta_preemptions", "nan"))
            except Exception:
                dp = float("nan")
            if dp == dp:
                sum_preemptions += dp
                preempt_n += 1

    ratio = (sum_hits / sum_queries) if sum_queries > 0 else None
    kv_usage_mean = (kv_usage_sum / kv_usage_n) if kv_usage_n > 0 else None
    num_running_mean = (num_running_sum / num_running_n) if num_running_n > 0 else None
    num_waiting_mean = (num_waiting_sum / num_waiting_n) if num_waiting_n > 0 else None

    latency_means: Dict[str, Optional[float]] = {}
    for base in hist_bases:
        latency_means[base] = (hist_sum[base] / hist_cnt[base]) if hist_cnt[base] > 0 else None

    decode_over_e2e = None
    de = latency_means.get("request_decode_time_seconds")
    ee = latency_means.get("e2e_request_latency_seconds")
    if de is not None and ee is not None and ee > 0:
        decode_over_e2e = de / ee

    window_dur = max(0.0, t_end - t_start)
    if preempt_n > 0 and window_dur > 0:
        preemptions_per_sec = sum_preemptions / window_dur
        preemptions_delta_mean = sum_preemptions / preempt_n
        sum_preemptions_out: Optional[float] = sum_preemptions
    else:
        preemptions_per_sec = None
        preemptions_delta_mean = None
        sum_preemptions_out = None

    return {
        "available": True,
        "samples_hit": n_hit,
        "sum_hits_tokens": sum_hits,
        "sum_queries_tokens": sum_queries,
        "hit_ratio": ratio,
        "kv_cache_usage_mean_perc": kv_usage_mean,
        "num_requests_running_mean": num_running_mean,
        "num_requests_waiting_mean": num_waiting_mean,
        "latency_means_seconds": latency_means,
        "decode_over_e2e_ratio": decode_over_e2e,
        "preemptions": {
            "samples": preempt_n,
            "sum_delta_preemptions": sum_preemptions_out,
            "delta_preemptions_mean_per_interval": preemptions_delta_mean,
            "preemptions_per_sec": preemptions_per_sec,
        },
    }


def _summarize_usage_recs(usage_recs: List[Dict[str, Any]], t_start: float, t_end: float) -> Dict[str, Any]:
    sum_cached = 0.0
    sum_prompt = 0.0
    n = 0
    ratios: List[float] = []
    for r in usage_recs:
        try:
            ts = float(r.get("ts_unix"))
            pt = float(r.get("prompt_tokens"))
            ct = float(r.get("cached_tokens"))
        except Exception:
            continue
        if ts < t_start or ts > t_end:
            continue
        if pt <= 0:
            continue
        if ct < 0:
            continue
        sum_cached += ct
        sum_prompt += pt
        n += 1
        ratios.append(ct / pt)
    ratio_token_weighted = (sum_cached / sum_prompt) if sum_prompt > 0 else None
    ratio_request_avg = (sum(ratios) / len(ratios)) if ratios else None
    return {
        "available": True,
        "responses": n,
        "sum_cached_tokens": sum_cached,
        "sum_prompt_tokens": sum_prompt,
        "cached_tokens_ratio_token_weighted": ratio_token_weighted,
        "cached_tokens_ratio_request_avg": ratio_request_avg,
    }


_TS_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:\.\d+)?"),
    re.compile(r"\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:\.\d+)?\]"),
]


def _parse_line_ts_unix(line: str) -> Optional[float]:
    m = re.search(r"\bts_unix=(?P<ts>\d+(?:\.\d+)?)\b", line)
    if m:
        try:
            return float(m.group("ts"))
        except Exception:
            pass
    m = re.search(r"\bts_iso=(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?\b", line)
    if m:
        try:
            d = dt.datetime.fromisoformat(m.group("ts"))
            return d.timestamp()
        except Exception:
            pass
    for pat in _TS_PATTERNS:
        m = pat.search(line)
        if not m:
            continue
        s = m.group("ts")
        try:
            d = dt.datetime.fromisoformat(s)
            return d.timestamp()
        except Exception:
            continue
    return None


def _summarize_trials_per_sec(eval_log: Path, t_start: float, t_end: float) -> Dict[str, Any]:
    if not eval_log.exists():
        return {"available": False}
    count = 0
    parsed = 0
    ts_first: Optional[float] = None
    ts_last: Optional[float] = None
    for line in eval_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "HLE_TRIAL_COMPLETE" not in line:
            continue
        ts = _parse_line_ts_unix(line)
        if ts is None:
            continue
        parsed += 1
        if t_start <= ts <= t_end:
            count += 1
            if ts_first is None or ts < ts_first:
                ts_first = ts
            if ts_last is None or ts > ts_last:
                ts_last = ts
    window_dur = max(0.0, t_end - t_start)
    trials_per_sec = (count / window_dur) if window_dur > 0 else None
    return {
        "available": True,
        "marker_lines_parsed": parsed,
        "trials_completed_in_window": count,
        "window_duration_sec": window_dur,
        "trials_per_sec": trials_per_sec,
        "first_marker_unix": ts_first,
        "last_marker_unix": ts_last,
    }


def _candidate_eval_logs(primary: Path) -> List[Path]:
    candidates: List[Path] = []
    seen: set[str] = set()
    for p in [
        primary,
        primary.parent / "hle_profile.log",
        primary.parent / "eval_global.log",
        primary.parent / "hle_global.log",
        primary.parent / "tau2_global.log",
    ]:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            candidates.append(p)
    return candidates


def _summarize_trials_with_fallback(eval_log: Path, t_start: float, t_end: float) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    for p in _candidate_eval_logs(eval_log):
        cur = _summarize_trials_per_sec(p, t_start, t_end)
        if not cur.get("available"):
            continue
        cur["source_log"] = str(p)
        if best is None:
            best = cur
        if (cur.get("marker_lines_parsed") or 0) > 0:
            return cur
    return best or {"available": False}


def _iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).astimezone().replace(microsecond=0).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", required=True)
    ap.add_argument("--usage-jsonl", required=True)
    ap.add_argument("--eval-log", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--start-offset-sec", type=float, default=600.0)
    ap.add_argument("--end-offset-sec", type=float, default=7800.0)
    ap.add_argument(
        "--windows",
        type=str,
        default="",
        help="Comma-separated windows as start:end seconds (e.g. 600:2400,600:7800). Default uses 600:7800 only.",
    )
    args = ap.parse_args()

    metrics_csv = Path(args.metrics_csv)
    usage_jsonl = Path(args.usage_jsonl)
    eval_log = Path(args.eval_log) if args.eval_log else None

    usage = _read_usage_jsonl(usage_jsonl)
    t0 = _first_request_ts(usage)
    if t0 is None:
        out = {"ok": False, "reason": "no_t0", "usage_jsonl": str(usage_jsonl)}
    else:
        windows_arg = (args.windows or "").strip()
        if windows_arg:
            windows = []
            for part in windows_arg.split(","):
                part = part.strip()
                if not part:
                    continue
                a, b = part.split(":")
                windows.append((float(a), float(b)))
        else:
            windows = [(float(args.start_offset_sec), float(args.end_offset_sec))]

        out: Dict[str, Any] = {
            "ok": True,
            "t0_first_request_unix": t0,
            "t0_first_request_iso": _iso(t0),
            "windows": {},
        }
        for start_off, end_off in windows:
            t_start = t0 + start_off
            t_end = t0 + end_off
            key = f"{int(start_off)}-{int(end_off)}"
            out["windows"][key] = {
                "window_start_unix": t_start,
                "window_end_unix": t_end,
                "window_start_iso": _iso(t_start),
                "window_end_iso": _iso(t_end),
                "metrics": _summarize_metrics_csv(metrics_csv, t_start, t_end),
                "response_usage": _summarize_usage_recs(usage, t_start, t_end),
                "trials": _summarize_trials_with_fallback(eval_log, t_start, t_end) if eval_log else {"available": False},
            }

    blob = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out_json:
        Path(args.out_json).write_text(blob + "\n", encoding="utf-8")
    else:
        print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
