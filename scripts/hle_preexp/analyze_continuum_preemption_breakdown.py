#!/usr/bin/env python3
"""
Analyze Continuum preemption behavior from scheduler_timestamps within a fixed window.

Inputs:
  - scheduler_timestamps (JSON): {job_id: [event_dict, ...], ...}
    Written by vllm-continuum Continuum_Recorder to $RUN_OUTPUT_DIR/scheduler_timestamps.
  - orchestrator_usage.jsonl: used to define t0 and thus the measurement window.

Window definition (tau2-style):
  - t0 = first orchestrator response that logged usage (from orchestrator_usage.jsonl)
  - window = [t0 + start_offset_sec, t0 + end_offset_sec]

Preemption event types (user-defined):
  1) Running + With Pin:
     The request is evicted while it is running, and the previous step's KV cache
     was still pinned at the arrival of this request.
  2) Running + After Unpin:
     The request is evicted while it is running, and the previous step's KV cache
     had already been unpinned before the arrival of this request.
  3) Pinned + With Pin:
     A pinned KV cache entry (after a request finished) is evicted during its pin TTL.

Output JSON includes:
  - Total Requests (arrivals)
  - # of Preemptions (events)
  - Preemption Rate (events/request)  [main]
  - # of Requests Preempted + Request Preempt Rate
  - # of Jobs Preempted + Job Preempt Rate
  - Breakdown of preemption events into the 3 types above (counts + shares)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).astimezone().replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


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


def _t0_from_usage(usage: List[Dict[str, Any]]) -> Optional[float]:
    t0: Optional[float] = None
    for r in usage:
        try:
            ts = float(r.get("ts_unix"))
            pt = float(r.get("prompt_tokens"))
            ct = float(r.get("cached_tokens"))
        except Exception:
            continue
        if pt <= 0 or ct < 0:
            continue
        if t0 is None or ts < t0:
            t0 = ts
    return t0


@dataclass
class RequestRec:
    arrival_time: float
    start_times: List[float] = field(default_factory=list)  # waiting_to_running / evicted_to_running
    departure_time: Optional[float] = None
    pinned_time: Optional[float] = None
    unpinned_time: Optional[float] = None
    preempt_times: List[float] = field(default_factory=list)  # Request_evicted_from_running_queue_time


def _find_last(reqs: List[RequestRec], pred) -> Optional[RequestRec]:
    for r in reversed(reqs):
        if pred(r):
            return r
    return None


def _parse_job_requests(events: Any) -> List[RequestRec]:
    if not isinstance(events, list):
        return []

    reqs: List[RequestRec] = []

    for ev in events:
        if not isinstance(ev, dict):
            continue

        if "Request_arrival_time" in ev:
            try:
                t = float(ev["Request_arrival_time"])
            except Exception:
                continue
            reqs.append(RequestRec(arrival_time=t))
            continue

        if "waiting_to_running" in ev or "evicted_to_running" in ev:
            if not reqs:
                continue
            key = "waiting_to_running" if "waiting_to_running" in ev else "evicted_to_running"
            try:
                t = float(ev[key])
            except Exception:
                continue
            reqs[-1].start_times.append(t)
            continue

        if "Request_departure_time" in ev:
            try:
                t = float(ev["Request_departure_time"])
            except Exception:
                continue
            cur = _find_last(reqs, lambda r: r.departure_time is None)
            if cur is None and reqs:
                cur = reqs[-1]
            if cur is not None:
                cur.departure_time = t
            continue

        if "pinned_time" in ev:
            try:
                t = float(ev["pinned_time"])
            except Exception:
                continue
            cur = _find_last(reqs, lambda r: r.departure_time is not None and r.pinned_time is None)
            if cur is not None:
                cur.pinned_time = t
            continue

        if "unpinned_time" in ev:
            try:
                t = float(ev["unpinned_time"])
            except Exception:
                continue
            cur = _find_last(reqs, lambda r: r.pinned_time is not None and r.unpinned_time is None)
            if cur is not None:
                cur.unpinned_time = t
            continue

        if "Request_evicted_from_running_queue_time" in ev:
            try:
                t = float(ev["Request_evicted_from_running_queue_time"])
            except Exception:
                continue
            if not reqs:
                continue
            reqs[-1].preempt_times.append(t)
            continue

    return reqs


def _is_prev_kv_pinned_at_arrival(prev: RequestRec, cur: RequestRec) -> bool:
    if prev.pinned_time is None:
        return False
    if prev.unpinned_time is None:
        return True
    return prev.unpinned_time >= cur.arrival_time


def _classify_preempt_event(
    reqs: List[RequestRec],
    req_index: int,
    t_preempt: float,
) -> str:
    """
    Returns:
      - "running_with_pin"
      - "running_after_unpin"
      - "pinned_with_pin"
      - "unknown"
    """
    cur = reqs[req_index]

    # Pinned + With Pin: eviction happens after the request finished, while its pin is active.
    if cur.departure_time is not None and t_preempt > cur.departure_time:
        if cur.pinned_time is None:
            return "unknown"
        if t_preempt < cur.pinned_time:
            return "unknown"
        if cur.unpinned_time is None or t_preempt <= cur.unpinned_time:
            return "pinned_with_pin"
        return "unknown"

    # Running: use previous request's pin status at arrival.
    if req_index <= 0:
        return "running_after_unpin"
    prev = reqs[req_index - 1]
    if _is_prev_kv_pinned_at_arrival(prev, cur):
        return "running_with_pin"
    return "running_after_unpin"


def analyze(
    *,
    timestamps_path: Path,
    usage_jsonl: Path,
    start_offset_sec: float,
    end_offset_sec: float,
) -> Dict[str, Any]:
    usage = _read_usage_jsonl(usage_jsonl)
    t0 = _t0_from_usage(usage)
    if t0 is None:
        return {"ok": False, "reason": "no_t0", "usage_jsonl": str(usage_jsonl)}

    w_start = float(t0) + float(start_offset_sec)
    w_end = float(t0) + float(end_offset_sec)

    data = _read_json(timestamps_path)
    if not isinstance(data, dict):
        return {"ok": False, "reason": "bad_scheduler_timestamps_format", "timestamps": str(timestamps_path)}

    def in_window(ts: float) -> bool:
        return w_start <= ts <= w_end

    total_requests = 0
    preempt_events_total = 0
    preempted_requests = 0

    jobs_with_arrival_in_window: set[str] = set()
    jobs_with_preempt_in_window: set[str] = set()

    by_type_events = {
        "running_with_pin": 0,
        "running_after_unpin": 0,
        "pinned_with_pin": 0,
        "unknown": 0,
    }

    for job_id, events in data.items():
        if not isinstance(job_id, str) or job_id == "null":
            continue

        reqs = _parse_job_requests(events)
        if not reqs:
            continue

        # Identify "active" jobs for the denominator set.
        has_arrival_in_window = any(in_window(r.arrival_time) for r in reqs)
        if has_arrival_in_window:
            jobs_with_arrival_in_window.add(job_id)

        if not has_arrival_in_window:
            continue

        # Requests / request-preempt.
        for req in reqs:
            if in_window(req.arrival_time):
                total_requests += 1
                if any(in_window(t) for t in req.preempt_times):
                    preempted_requests += 1

        # Preemption events (classify).
        for idx, req in enumerate(reqs):
            for t_preempt in req.preempt_times:
                if not in_window(t_preempt):
                    continue
                preempt_events_total += 1
                jobs_with_preempt_in_window.add(job_id)
                kind = _classify_preempt_event(reqs, idx, t_preempt)
                by_type_events[kind] = by_type_events.get(kind, 0) + 1

    total_jobs = len(jobs_with_arrival_in_window)
    jobs_preempted = len(jobs_with_arrival_in_window.intersection(jobs_with_preempt_in_window))

    preemption_rate_events_per_request = (preempt_events_total / total_requests) if total_requests > 0 else None
    request_preempt_rate = (preempted_requests / total_requests) if total_requests > 0 else None
    job_preempt_rate = (jobs_preempted / total_jobs) if total_jobs > 0 else None

    def share(x: int) -> Optional[float]:
        return (x / preempt_events_total) if preempt_events_total > 0 else None

    return {
        "ok": True,
        "timestamps": str(timestamps_path),
        "usage_jsonl": str(usage_jsonl),
        "t0_unix": float(t0),
        "t0_iso": _iso(float(t0)),
        "window": {
            "start_offset_sec": float(start_offset_sec),
            "end_offset_sec": float(end_offset_sec),
            "window_start_unix": w_start,
            "window_end_unix": w_end,
            "window_start_iso": _iso(w_start),
            "window_end_iso": _iso(w_end),
        },
        "totals": {
            "total_requests": total_requests,
            "total_jobs": total_jobs,
            "preempt_events_total": preempt_events_total,
            "preempted_requests": preempted_requests,
            "jobs_preempted": jobs_preempted,
            "preemption_rate_events_per_request": preemption_rate_events_per_request,
            "request_preempt_rate": request_preempt_rate,
            "job_preempt_rate": job_preempt_rate,
        },
        "preemption_event_breakdown": {
            "running_with_pin": {
                "events": by_type_events["running_with_pin"],
                "share": share(by_type_events["running_with_pin"]),
            },
            "running_after_unpin": {
                "events": by_type_events["running_after_unpin"],
                "share": share(by_type_events["running_after_unpin"]),
            },
            "pinned_with_pin": {
                "events": by_type_events["pinned_with_pin"],
                "share": share(by_type_events["pinned_with_pin"]),
            },
            "unknown": {
                "events": by_type_events["unknown"],
                "share": share(by_type_events["unknown"]),
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamps", required=True, help="Path to scheduler_timestamps (JSON)")
    ap.add_argument("--usage-jsonl", required=True, help="Path to orchestrator_usage.jsonl (defines t0)")
    ap.add_argument("--start-offset-sec", type=float, default=600.0)
    ap.add_argument("--end-offset-sec", type=float, default=7800.0)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    out = analyze(
        timestamps_path=Path(args.timestamps),
        usage_jsonl=Path(args.usage_jsonl),
        start_offset_sec=float(args.start_offset_sec),
        end_offset_sec=float(args.end_offset_sec),
    )

    blob = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out_json:
        Path(args.out_json).write_text(blob + "\n", encoding="utf-8")
    else:
        print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

