#!/usr/bin/env python3
"""
Summarize rollout-style tau2-bench evaluation:
- tasks done / unit time (from [TAU2_TASK_COMPLETE] markers in eval_<domain>.log)
- steps done / unit time (from [PROFILE] type=step_complete in tau2_<domain>.log)

This is intentionally lightweight and robust to partial runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Any


TS_PREFIX_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
TASK_MARKER_RE = re.compile(r"\[TAU2_TASK_COMPLETE\]\s+(.+)$")
TASK_TS_UNIX_RE = re.compile(r"\bts_unix=(\d+)\b")
PROFILE_TS_RE = re.compile(r"^\[PROFILE\]\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\b")


@dataclass
class DomainSummary:
    domain: str
    runtime_s: float
    tasks_ok: int
    tasks_error: int
    tasks_skipped: int
    tasks_total_markers: int
    step_complete_events: int

    @property
    def tasks_per_min(self) -> float:
        if self.runtime_s <= 0:
            return 0.0
        return self.tasks_ok / (self.runtime_s / 60.0)

    @property
    def steps_per_s(self) -> float:
        if self.runtime_s <= 0:
            return 0.0
        return self.step_complete_events / self.runtime_s


def _parse_driver_runtime_s(eval_log_path: Path) -> float:
    """
    eval_<domain>.log contains both run_local.py timestamped lines and child output.
    We compute runtime using the first/last run_local timestamped lines.
    """
    first: Optional[datetime] = None
    last: Optional[datetime] = None
    try:
        with eval_log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = TS_PREFIX_RE.match(line)
                if not m:
                    continue
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                if first is None:
                    first = ts
                last = ts
    except Exception:
        return 0.0
    if first is None or last is None:
        return 0.0
    return max(0.0, (last - first).total_seconds())


def _parse_task_markers(eval_log_path: Path) -> Tuple[int, int, int, int]:
    ok = err = skipped = total = 0
    try:
        with eval_log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = TASK_MARKER_RE.search(line)
                if not m:
                    continue
                total += 1
                kv = m.group(1)
                # marker looks like: domain=... trial=... task_id=... status=ok ...
                status_m = re.search(r"\bstatus=(\w+)\b", kv)
                status = status_m.group(1) if status_m else "unknown"
                if status == "ok":
                    ok += 1
                elif status == "error":
                    err += 1
                elif status == "skipped":
                    skipped += 1
    except Exception:
        pass
    return ok, err, skipped, total


def _count_step_complete(profile_log_path: Path) -> int:
    n = 0
    try:
        with profile_log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "[PROFILE]" not in line:
                    continue
                if "type=step_complete" in line:
                    n += 1
    except Exception:
        pass
    return n


def _detect_global_logs(log_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (eval_log, tau2_log) for global mode if present."""
    eval_global = log_dir / "eval_global.log"
    tau2_global = log_dir / "tau2_global.log"
    if eval_global.exists() and tau2_global.exists():
        return eval_global, tau2_global
    return None, None


def _read_kv_timeseries_csv(kv_csv_path: Path) -> Dict[str, Any]:
    """
    Read KV-cache timeseries CSV emitted by collect_kv_cache_timeseries.sh.

    Expected columns:
      ts_iso,ts_unix,port,metric,value

    Returns a dict with:
      - ports: List[str]
      - by_ts: Dict[int, Dict[str, float]]  # normalized to fraction [0,1]
      - unit: "fraction" | "percent" | "unknown"
      - max_raw_value: float | None
    """
    by_ts: Dict[int, Dict[str, float]] = {}
    ports_set: set[str] = set()
    max_raw: Optional[float] = None

    with kv_csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row:
                continue
            ts_unix_s = (row.get("ts_unix") or "").strip()
            port = (row.get("port") or "").strip()
            val_s = (row.get("value") or "").strip()
            if not ts_unix_s or not port:
                continue
            try:
                ts_unix = int(float(ts_unix_s))
            except Exception:
                continue
            try:
                val = float(val_s)
            except Exception:
                val = float("nan")

            ports_set.add(port)
            by_ts.setdefault(ts_unix, {})[port] = val
            if not math.isnan(val):
                max_raw = val if max_raw is None else max(max_raw, val)

    ports: List[str]
    try:
        ports = sorted(ports_set, key=lambda p: int(p))
    except Exception:
        ports = sorted(ports_set)

    # Detect unit: vLLM metric name says _perc; many setups emit 0-100, but some can be 0-1.
    if max_raw is None:
        unit = "unknown"
        scale = 1.0
    elif max_raw <= 1.0:
        unit = "fraction"
        scale = 1.0
    else:
        unit = "percent"
        scale = 100.0

    # Normalize values to fraction.
    for ts, per_port in by_ts.items():
        for port, v in list(per_port.items()):
            if math.isnan(v):
                continue
            per_port[port] = v / scale

    return {"ports": ports, "by_ts": by_ts, "unit": unit, "max_raw_value": max_raw}


def _compute_active_window_from_kv(
    kv: Dict[str, Any],
    *,
    threshold_fraction: float,
) -> Optional[Dict[str, Any]]:
    """
    Active window definition:
      - For each backend/port, its active window is:
          first time kv_cache_usage > threshold_fraction
          to last time kv_cache_usage > threshold_fraction
        (non-contiguous allowed; we only use first/last).
      - For DP>1, the experiment active window is the INTERSECTION of all per-port windows:
          start = max(per_port_start)
          end   = min(per_port_end)
    """
    ports: List[str] = kv["ports"]
    by_ts: Dict[int, Dict[str, float]] = kv["by_ts"]
    if not ports or not by_ts:
        return None

    per_port: Dict[str, Dict[str, Any]] = {}
    for p in ports:
        above: List[int] = []
        for ts in sorted(by_ts.keys()):
            row = by_ts.get(ts, {})
            v = row.get(p)
            if v is None or math.isnan(v):
                continue
            # Strictly greater-than per requirement (>50%).
            if float(v) > float(threshold_fraction):
                above.append(ts)
        if not above:
            return None
        per_port[p] = {
            "start_ts_unix": above[0],
            "end_ts_unix": above[-1],
            "samples_above": len(above),
        }

    start_ts = max(int(v["start_ts_unix"]) for v in per_port.values())
    end_ts = min(int(v["end_ts_unix"]) for v in per_port.values())
    if end_ts <= start_ts:
        return None

    return {
        "start_ts_unix": start_ts,
        "end_ts_unix": end_ts,
        "duration_s": max(0, end_ts - start_ts),
        "samples_total": len(by_ts),
        "ports": ports,
        "per_port": per_port,
        "definition": "intersection_of_per_port_first_last_above_threshold",
        "threshold_op": ">",
    }


def _count_expert_success_calls(profile_log_path: Path) -> int:
    """Count successful expert requests (PROFILE events: type=expert_call)."""
    n = 0
    try:
        with profile_log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "[PROFILE]" not in line:
                    continue
                if "type=expert_call" in line:
                    n += 1
    except Exception:
        return 0
    return n


def _count_expert_timeouts(profile_log_path: Path) -> int:
    # tau2 logs this exact prefix for expert failures.
    needle = "Expert LLM call failed permanently: LLMTimeoutError"
    n = 0
    try:
        with profile_log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if needle in line:
                    n += 1
    except Exception:
        return 0
    return n


def _iter_step_complete_ts_unix(profile_log_path: Path) -> Iterable[float]:
    """Yield epoch seconds (float) for each step_complete PROFILE event."""
    try:
        with profile_log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "type=step_complete" not in line:
                    continue
                m = PROFILE_TS_RE.match(line)
                if not m:
                    continue
                try:
                    dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
                except Exception:
                    continue
                # dt is naive; interpret in local timezone (consistent with date +%s used by kv sampler).
                yield dt.timestamp()
    except Exception:
        return


def _count_step_complete_in_window(profile_log_path: Path, start_ts: int, end_ts: int) -> int:
    n = 0
    for ts in _iter_step_complete_ts_unix(profile_log_path):
        if start_ts <= ts <= end_ts:
            n += 1
    return n


def _parse_task_markers_in_window(eval_log_path: Path, start_ts: int, end_ts: int) -> Dict[str, int]:
    """
    Count [TAU2_TASK_COMPLETE] markers whose embedded ts_unix falls inside [start_ts, end_ts].
    Returns counts for total/ok/error/skipped.
    """
    out = {"total": 0, "ok": 0, "error": 0, "skipped": 0, "unknown": 0, "missing_ts": 0}
    try:
        with eval_log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "[TAU2_TASK_COMPLETE]" not in line:
                    continue
                ts_m = TASK_TS_UNIX_RE.search(line)
                if not ts_m:
                    out["missing_ts"] += 1
                    continue
                try:
                    ts = int(ts_m.group(1))
                except Exception:
                    out["missing_ts"] += 1
                    continue
                if ts < start_ts or ts > end_ts:
                    continue
                out["total"] += 1
                status_m = re.search(r"\bstatus=(\w+)\b", line)
                status = status_m.group(1) if status_m else "unknown"
                if status in out:
                    out[status] += 1
                else:
                    out["unknown"] += 1
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True, help="Directory containing eval_<domain>.log and tau2_<domain>.log")
    ap.add_argument("--domains", nargs="+", default=["retail", "telecom", "airline"])
    ap.add_argument("--out-json", required=True, help="Output JSON path")
    ap.add_argument("--kv-csv", default=None, help="KV cache timeseries CSV path (optional). If set, compute active-window metrics.")
    ap.add_argument("--kv-threshold", type=float, default=0.50, help="Active-window KV threshold as fraction (default: 0.50 == 50%).")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    summaries: List[DomainSummary] = []

    # Global mode (cross-domain scheduler) uses unified log files.
    eval_global, tau2_global = _detect_global_logs(log_dir)
    if eval_global is not None and tau2_global is not None:
        runtime_s = _parse_driver_runtime_s(eval_global)
        tasks_ok, tasks_err, tasks_skipped, tasks_total = _parse_task_markers(eval_global)
        step_events = _count_step_complete(tau2_global)
        summaries.append(
            DomainSummary(
                domain="global",
                runtime_s=runtime_s,
                tasks_ok=tasks_ok,
                tasks_error=tasks_err,
                tasks_skipped=tasks_skipped,
                tasks_total_markers=tasks_total,
                step_complete_events=step_events,
            )
        )
    else:
        # Legacy per-domain logs.
        for domain in args.domains:
            eval_log = log_dir / f"eval_{domain}.log"
            tau2_log = log_dir / f"tau2_{domain}.log"

            runtime_s = _parse_driver_runtime_s(eval_log) if eval_log.exists() else 0.0
            tasks_ok, tasks_err, tasks_skipped, tasks_total = _parse_task_markers(eval_log) if eval_log.exists() else (0, 0, 0, 0)
            step_events = _count_step_complete(tau2_log) if tau2_log.exists() else 0

            summaries.append(
                DomainSummary(
                    domain=domain,
                    runtime_s=runtime_s,
                    tasks_ok=tasks_ok,
                    tasks_error=tasks_err,
                    tasks_skipped=tasks_skipped,
                    tasks_total_markers=tasks_total,
                    step_complete_events=step_events,
                )
            )

    total_runtime_s = sum(s.runtime_s for s in summaries)
    total_tasks_ok = sum(s.tasks_ok for s in summaries)
    total_steps = sum(s.step_complete_events for s in summaries)

    active_window: Optional[Dict[str, Any]] = None
    active_window_metrics: Optional[Dict[str, Any]] = None

    expert_requests_total: Optional[int] = None
    expert_timeouts_total: Optional[int] = None
    expert_timeout_ratio: Optional[float] = None

    # Expert timeout ratio is computed across the FULL setting runtime (not active window).
    if tau2_global is not None and tau2_global.exists():
        expert_success_calls = _count_expert_success_calls(tau2_global)
        expert_timeouts_total = _count_expert_timeouts(tau2_global)
        expert_requests_total = expert_success_calls + expert_timeouts_total
        if expert_requests_total > 0:
            expert_timeout_ratio = float(expert_timeouts_total) / float(expert_requests_total)
        else:
            expert_timeout_ratio = 0.0

    if args.kv_csv:
        kv_path = Path(args.kv_csv)
        if kv_path.exists():
            kv = _read_kv_timeseries_csv(kv_path)
            active_window = _compute_active_window_from_kv(kv, threshold_fraction=float(args.kv_threshold))
            if active_window is not None:
                # Use global logs when present; otherwise try to aggregate from per-domain logs.
                if eval_global is not None and tau2_global is not None:
                    steps_in_window = _count_step_complete_in_window(tau2_global, active_window["start_ts_unix"], active_window["end_ts_unix"])
                    trials_counts = _parse_task_markers_in_window(eval_global, active_window["start_ts_unix"], active_window["end_ts_unix"])
                else:
                    steps_in_window = 0
                    for domain in args.domains:
                        tau2_log = log_dir / f"tau2_{domain}.log"
                        if tau2_log.exists():
                            steps_in_window += _count_step_complete_in_window(tau2_log, active_window["start_ts_unix"], active_window["end_ts_unix"])
                    # Task markers in legacy mode may not have ts_unix; count only if present.
                    trials_counts = {"total": 0, "ok": 0, "error": 0, "skipped": 0, "unknown": 0, "missing_ts": 0}
                    for domain in args.domains:
                        eval_log = log_dir / f"eval_{domain}.log"
                        if eval_log.exists():
                            c = _parse_task_markers_in_window(eval_log, active_window["start_ts_unix"], active_window["end_ts_unix"])
                            for k, v in c.items():
                                trials_counts[k] = trials_counts.get(k, 0) + v

                duration_s = float(active_window.get("duration_s", 0) or 0)
                steps_per_s = (steps_in_window / duration_s) if duration_s > 0 else 0.0
                tasks_per_s = (trials_counts["total"] / duration_s) if duration_s > 0 else 0.0
                tasks_ok_per_s = (trials_counts["ok"] / duration_s) if duration_s > 0 else 0.0

                active_window_metrics = {
                    "steps": steps_in_window,
                    "steps_per_s": steps_per_s,
                    "trials_total": trials_counts["total"],
                    "trials_ok": trials_counts.get("ok", 0),
                    "trials_error": trials_counts.get("error", 0),
                    "trials_skipped": trials_counts.get("skipped", 0),
                    "tasks_per_s": tasks_per_s,
                    "tasks_ok_per_s": tasks_ok_per_s,
                    "trials_missing_ts": trials_counts.get("missing_ts", 0),
                }

    out: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(),
        "log_dir": str(log_dir),
        "domains": [s.domain for s in summaries],
        "active_window": (
            None
            if active_window is None
            else {
                **active_window,
                "kv_csv": str(Path(args.kv_csv)) if args.kv_csv else None,
                "kv_threshold_fraction": float(args.kv_threshold),
            }
        ),
        "active_window_metrics": active_window_metrics,
        "totals": {
            "runtime_s": total_runtime_s,
            "tasks_ok": total_tasks_ok,
            "tasks_total_markers": sum(s.tasks_total_markers for s in summaries),
            "steps": total_steps,
            "tasks_per_s": (total_tasks_ok / total_runtime_s) if total_runtime_s > 0 else 0.0,
            "steps_per_s": (total_steps / total_runtime_s) if total_runtime_s > 0 else 0.0,
        },
        "expert": (
            None
            if expert_requests_total is None or expert_timeouts_total is None or expert_timeout_ratio is None
            else {
                "requests_total": int(expert_requests_total),
                "timeouts_total": int(expert_timeouts_total),
                "timeout_ratio": float(expert_timeout_ratio),
                "timeout_threshold_s": 600.0,
                "definition": "timeouts / total_call_expert_requests (full setting)",
            }
        ),
        "per_domain": [
            {
                "domain": s.domain,
                "runtime_s": s.runtime_s,
                "tasks_ok": s.tasks_ok,
                "tasks_error": s.tasks_error,
                "tasks_skipped": s.tasks_skipped,
                "tasks_total_markers": s.tasks_total_markers,
                "tasks_per_s": (s.tasks_ok / s.runtime_s) if s.runtime_s > 0 else 0.0,
                "step_complete_events": s.step_complete_events,
                "steps_per_s": s.steps_per_s,
            }
            for s in summaries
        ],
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
