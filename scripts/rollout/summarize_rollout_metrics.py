#!/usr/bin/env python3
"""
Summarize rollout-style tau2-bench evaluation:
- tasks done / unit time (from [TAU2_TASK_COMPLETE] markers in eval_<domain>.log)
- steps done / unit time (from [PROFILE] type=step_complete in tau2_<domain>.log)

This is intentionally lightweight and robust to partial runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TS_PREFIX_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
TASK_MARKER_RE = re.compile(r"\[TAU2_TASK_COMPLETE\]\s+(.+)$")


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True, help="Directory containing eval_<domain>.log and tau2_<domain>.log")
    ap.add_argument("--domains", nargs="+", default=["retail", "telecom", "airline"])
    ap.add_argument("--out-json", required=True, help="Output JSON path")
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

    out: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(),
        "log_dir": str(log_dir),
        "domains": [s.domain for s in summaries],
        "totals": {
            "runtime_s": total_runtime_s,
            "tasks_ok": total_tasks_ok,
            "steps": total_steps,
            "tasks_per_min": (total_tasks_ok / (total_runtime_s / 60.0)) if total_runtime_s > 0 else 0.0,
            "steps_per_s": (total_steps / total_runtime_s) if total_runtime_s > 0 else 0.0,
        },
        "per_domain": [
            {
                "domain": s.domain,
                "runtime_s": s.runtime_s,
                "tasks_ok": s.tasks_ok,
                "tasks_error": s.tasks_error,
                "tasks_skipped": s.tasks_skipped,
                "tasks_total_markers": s.tasks_total_markers,
                "tasks_per_min": s.tasks_per_min,
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


