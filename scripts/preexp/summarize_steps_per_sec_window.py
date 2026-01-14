#!/usr/bin/env python3
"""
Compute average steps/sec in a fixed window relative to t0 (first successful orchestrator response).

Definition:
  - step = one "[PROFILE] ... type=step_complete ..." line in eval_driver.log
  - t0 = first record in orchestrator_usage.jsonl (ts_unix)
  - window = [t0 + start_offset_sec, t0 + end_offset_sec]

Outputs JSON with:
  - step_complete_count_in_window
  - steps_per_sec (count / window_duration_sec)
  - markers parsed info (optional sanity)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


def _t0_from_usage_jsonl(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict) and "ts_unix" in rec:
            try:
                return float(rec["ts_unix"])
            except Exception:
                continue
    return None


def _count_step_complete(eval_log: Path, t_start: float, t_end: float) -> Dict[str, Any]:
    if not eval_log.exists():
        return {"available": False}
    # Example:
    # [PROFILE] 2026-01-07 18:48:29.507 task=... type=step_complete step=0 ...
    ts_pat = re.compile(r"\[PROFILE\]\s+(?P<ts_iso>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")
    count = 0
    parsed = 0
    for line in eval_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "type=step_complete" not in line:
            continue
        m = re.search(r"\bts_unix=(\d+(?:\.\d+)?)\b", line)
        ts = None
        if m:
            try:
                ts = float(m.group(1))
            except Exception:
                ts = None
        if ts is None:
            m2 = ts_pat.search(line)
            if not m2:
                continue
            # Parse iso without timezone as localtime; consistent with our other tools.
            import datetime as dt

            try:
                ts = dt.datetime.fromisoformat(m2.group("ts_iso")).timestamp()
            except Exception:
                continue
        parsed += 1
        if t_start <= ts <= t_end:
            count += 1
    dur = max(0.0, t_end - t_start)
    sps = (count / dur) if dur > 0 else None
    return {
        "available": True,
        "step_complete_lines_parsed": parsed,
        "step_complete_in_window": count,
        "window_duration_sec": dur,
        "steps_per_sec": sps,
    }


def _count_step_complete_with_fallback(eval_log: Path, t_start: float, t_end: float) -> Dict[str, Any]:
    # Prefer parsing the provided file, but fall back to common sibling logs
    # in case eval_driver.log was truncated.
    candidates = [eval_log, eval_log.parent / "tau2_global.log", eval_log.parent / "eval_global.log"]
    best: Optional[Dict[str, Any]] = None
    for p in candidates:
        if not p.exists():
            continue
        cur = _count_step_complete(p, t_start, t_end)
        cur["source_log"] = str(p)
        if best is None:
            best = cur
        if (cur.get("step_complete_lines_parsed") or 0) > 0:
            return cur
    return best or {"available": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage-jsonl", required=True)
    ap.add_argument("--eval-log", required=True)
    ap.add_argument("--start-offset-sec", type=float, default=600.0)
    ap.add_argument("--end-offset-sec", type=float, default=4200.0)
    args = ap.parse_args()

    usage = Path(args.usage_jsonl)
    ev = Path(args.eval_log)
    t0 = _t0_from_usage_jsonl(usage)
    if t0 is None:
        print(json.dumps({"ok": False, "reason": "no_t0", "usage_jsonl": str(usage)}))
        return 0
    t_start = t0 + float(args.start_offset_sec)
    t_end = t0 + float(args.end_offset_sec)
    out: Dict[str, Any] = {
        "ok": True,
        "t0_unix": t0,
        "t_start_unix": t_start,
        "t_end_unix": t_end,
        "steps": _count_step_complete_with_fallback(ev, t_start, t_end),
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
