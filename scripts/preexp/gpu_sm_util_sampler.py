#!/usr/bin/env python3
"""
Lightweight GPU SM utilization sampler (NVML-based).

Writes a CSV time series suitable for later windowed averaging:
  ts_iso,ts_unix,gpu_index,sm_util,mem_util,mem_used_mib,mem_total_mib

Prefer NVML via pynvml (nvidia-ml-py). If NVML is unavailable, fall back to
`nvidia-smi --query-gpu=...` parsing.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple


def _iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).astimezone().replace(microsecond=0).isoformat()


def _try_nvml(gpu_index: int) -> Optional[Tuple[int, int, int, int]]:
    try:
        import pynvml  # type: ignore
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mib = int(mem.used / (1024 * 1024))
        total_mib = int(mem.total / (1024 * 1024))
        return int(util.gpu), int(util.memory), used_mib, total_mib
    except Exception:
        return None


def _smi_once(gpu_index: int) -> Optional[Tuple[int, int, int, int]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(gpu_index),
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception:
        return None
    if not out:
        return None
    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 4:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--gpu-index", type=int, required=True)
    ap.add_argument("--interval-sec", type=float, default=5.0)
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_iso", "ts_unix", "gpu_index", "sm_util", "mem_util", "mem_used_mib", "mem_total_mib"])
        f.flush()
        while True:
            now = time.time()
            iso = _iso(now)
            gpu = int(args.gpu_index)
            vals = _try_nvml(gpu)
            if vals is None:
                vals = _smi_once(gpu)
            if vals is None:
                sm = memu = used = total = "nan"
            else:
                sm, memu, used, total = vals
            w.writerow([iso, f"{now:.3f}", gpu, sm, memu, used, total])
            f.flush()
            time.sleep(float(args.interval_sec))


if __name__ == "__main__":
    raise SystemExit(main())

