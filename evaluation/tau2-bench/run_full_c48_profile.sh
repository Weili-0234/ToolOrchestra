#!/usr/bin/env bash
set -euo pipefail

# Full tau2-bench background eval (c48) + auto analysis.
# - Creates timestamped logs/ + outputs/ directories
# - Launches run_local.py with PROFILE logging (so vLLM prefill/decode metrics exist)
# - Writes a status line every 30 minutes to monitor.out
# - After completion, runs analyze_timing_from_tau2_log.py in base conda to produce:
#   - <prefix>_stats.json
#   - exactly 7 PNGs (one per metric)
#   - analysis.out (tables printed to stdout)

ROOT="/workspace/ToolOrchestra/evaluation/tau2-bench"
cd "$ROOT"

ts="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/full_c48_profile_${ts}"
OUT_DIR="$ROOT/outputs/full_c48_profile_${ts}"
mkdir -p "$LOG_DIR" "$OUT_DIR"

echo "$LOG_DIR" > "$ROOT/logs/last_full_c48_profile_dir.txt"

{
  echo "[runner] start_ts=$ts"
  echo "[runner] log_dir=$LOG_DIR"
  echo "[runner] out_dir=$OUT_DIR"
} | tee -a "$LOG_DIR/runner.out"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm1
export REPO_PATH="/workspace/ToolOrchestra"
source /workspace/ToolOrchestra/setup_envs.sh

export PYTHONUNBUFFERED=1

if pgrep -af "vllm serve|python .*run_local\\.py|python -m tau2\\.cli" >/dev/null 2>&1; then
  echo "[runner] ERROR: found existing eval/vLLM processes; stop them before launching." | tee -a "$LOG_DIR/runner.out"
  pgrep -af "vllm serve|python .*run_local\\.py|python -m tau2\\.cli" | tee -a "$LOG_DIR/runner.out" || true
  exit 1
fi

cmd=(
  python run_local.py
  --agent-model "$CKPT_DIR"
  --domains retail telecom airline
  --num-trials 1
  --max-concurrency 48
  --num-servers 1
  --log-level PROFILE
  --output-dir "$OUT_DIR"
  --log-dir "$LOG_DIR"
)

echo "[runner] launching: ${cmd[*]}" | tee -a "$LOG_DIR/runner.out"

nohup "${cmd[@]}" > "$LOG_DIR/driver.out" 2>&1 &
PID="$!"
echo "$PID" > "$LOG_DIR/driver.pid"
echo "[runner] pid=$PID" | tee -a "$LOG_DIR/runner.out"

strip_ansi() {
  # Remove ANSI color codes for clean monitor output.
  sed -r 's/\x1b\[[0-9;]*m//g'
}

monitor_once() {
  local now
  now="$(date +%Y-%m-%dT%H:%M:%S%z)"
  local eta
  eta="$(grep -a "\[OVERALL_ETA\]" "$LOG_DIR/driver.out" 2>/dev/null | tail -n 1 | strip_ansi || true)"
  local done
  # With `set -o pipefail`, grep returns exit=1 on 0 matches; avoid duplicating "0" by forcing success.
  done="$(grep -a -c "\[TAU2_TASK_COMPLETE\]" "$LOG_DIR/driver.out" 2>/dev/null || true)"
  echo "[$now] done_markers=$done ${eta:-}" >> "$LOG_DIR/monitor.out"
}

# First status line quickly, then every 30 minutes
monitor_once || true
while kill -0 "$PID" >/dev/null 2>&1; do
  sleep 1800
  monitor_once || true
done

wait "$PID" || true
echo "[runner] run_local finished (pid=$PID). starting analysis..." | tee -a "$LOG_DIR/runner.out"

# Post-run analysis in base conda (tables -> analysis.out, 7 PNGs + JSON into LOG_DIR)
source /root/miniconda3/etc/profile.d/conda.sh
conda run -n base python "$ROOT/analyze_timing_from_tau2_log.py" \
  --log-dir "$LOG_DIR" \
  --out-dir "$LOG_DIR" \
  --out-prefix full_c48_profile \
  --bins 10 \
  > "$LOG_DIR/analysis.out" 2>&1

echo "[runner] analysis complete. outputs:" | tee -a "$LOG_DIR/runner.out"
ls -lah "$LOG_DIR" | tee -a "$LOG_DIR/runner.out"


