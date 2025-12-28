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

kill_by_pattern() {
  local pat="$1"
  local desc="$2"
  mapfile -t pids < <(pgrep -f "$pat" 2>/dev/null || true)
  if (( ${#pids[@]} == 0 )); then
    return 0
  fi
  echo "[runner] stopping ${desc}: pattern=\"$pat\" pids=${pids[*]}" | tee -a "$LOG_DIR/runner.out"
  kill "${pids[@]}" >/dev/null 2>&1 || true
  sleep 5
  mapfile -t pids2 < <(pgrep -f "$pat" 2>/dev/null || true)
  if (( ${#pids2[@]} > 0 )); then
    echo "[runner] force-killing ${desc}: pids=${pids2[*]}" | tee -a "$LOG_DIR/runner.out"
    kill -9 "${pids2[@]}" >/dev/null 2>&1 || true
  fi
}

echo "[runner] restarting servers (always fresh): stopping any existing vLLM / tau2 / retrieval processes..." | tee -a "$LOG_DIR/runner.out"
kill_by_pattern "python .*run_local\\.py" "run_local.py"
kill_by_pattern "python -m tau2\\.cli" "tau2.cli"
kill_by_pattern "vllm serve" "vLLM"
kill_by_pattern "python .*retrieval_hle\\.py" "retrieval_hle.py"

# Start retrieval service on GPU1 (optional for tau2-bench; keeps behavior aligned with HLE runs).
# Logs are stored under this run's LOG_DIR by design.
RETRIEVAL_OUT="$LOG_DIR/retrieval.out"
RETRIEVAL_ERR="$LOG_DIR/retrieval.err"
RETRIEVAL_PID_FILE="$LOG_DIR/retrieval.pid"
RETRIEVAL_CACHE_DIR="$LOG_DIR/retrieval_cache"
mkdir -p "$RETRIEVAL_CACHE_DIR"

echo "[runner] starting retrieval server on GPU1 (port=1401)..." | tee -a "$LOG_DIR/runner.out"
nohup bash -lc "set -euo pipefail; source /root/miniconda3/etc/profile.d/conda.sh; conda activate retriever; export CUDA_VISIBLE_DEVICES=1; export INDEX_DIR=\"${INDEX_DIR:-/workspace/dataset/multi-train/index}\"; export HF_HOME=\"${HF_HOME:-/workspace/cache/huggingface}\"; cd /workspace/ToolOrchestra/evaluation; exec python /workspace/ToolOrchestra/evaluation/retrieval_hle.py --port 1401 --new_cache_dir \"$RETRIEVAL_CACHE_DIR\" --example_id_file /workspace/ToolOrchestra/evaluation/examples.json --tavily_key \"${TAVILY_KEY:-}\"" \
  >"$RETRIEVAL_OUT" 2>"$RETRIEVAL_ERR" &
RETRIEVAL_PID="$!"
echo "$RETRIEVAL_PID" > "$RETRIEVAL_PID_FILE"

echo "[runner] retrieval.pid=$RETRIEVAL_PID" | tee -a "$LOG_DIR/runner.out"

cleanup() {
  # Best-effort cleanup so a cancelled run doesn't leave GPU1 retrieval running.
  if [[ -f "$RETRIEVAL_PID_FILE" ]]; then
    local rp
    rp="$(cat "$RETRIEVAL_PID_FILE" 2>/dev/null || true)"
    if [[ -n "${rp:-}" ]] && kill -0 "$rp" >/dev/null 2>&1; then
      echo "[runner] cleanup: stopping retrieval (pid=$rp)..." | tee -a "$LOG_DIR/runner.out"
      kill "$rp" >/dev/null 2>&1 || true
      sleep 2
      if kill -0 "$rp" >/dev/null 2>&1; then
        echo "[runner] cleanup: force-killing retrieval (pid=$rp)..." | tee -a "$LOG_DIR/runner.out"
        kill -9 "$rp" >/dev/null 2>&1 || true
      fi
    fi
  fi
}
trap cleanup EXIT INT TERM

for i in $(seq 1 240); do
  if curl -sSf "http://127.0.0.1:1401/health" >/dev/null 2>&1; then
    echo "[runner] retrieval ready" | tee -a "$LOG_DIR/runner.out"
    break
  fi
  sleep 1
done
if ! curl -sSf "http://127.0.0.1:1401/health" >/dev/null 2>&1; then
  echo "[runner] ERROR: retrieval did not become ready; tail logs:" | tee -a "$LOG_DIR/runner.out"
  tail -n 80 "$RETRIEVAL_OUT" | tee -a "$LOG_DIR/runner.out" || true
  tail -n 120 "$RETRIEVAL_ERR" | tee -a "$LOG_DIR/runner.out" || true
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
  --model-config-path "$LOG_DIR/model_config_local.json"
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

# Stop retrieval server for this run (so the next run always restarts GPU1).
if [[ -f "$RETRIEVAL_PID_FILE" ]]; then
  rp="$(cat "$RETRIEVAL_PID_FILE" 2>/dev/null || true)"
  if [[ -n "${rp:-}" ]]; then
    echo "[runner] stopping retrieval (pid=$rp)..." | tee -a "$LOG_DIR/runner.out"
    kill "$rp" >/dev/null 2>&1 || true
    sleep 3
    if kill -0 "$rp" >/dev/null 2>&1; then
      echo "[runner] force-killing retrieval (pid=$rp)..." | tee -a "$LOG_DIR/runner.out"
      kill -9 "$rp" >/dev/null 2>&1 || true
    fi
  fi
fi

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


