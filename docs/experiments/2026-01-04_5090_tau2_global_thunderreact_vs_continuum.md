## Goal

Reproduce the 2026-01-04 inference experiments on a **single RunPod RTX 5090** comparing:

- **ThunderReact** (router above vLLM) vs
- **Continuum** (vLLM `--scheduling-policy continuum`)

on **tau2-bench** in **global** scheduling mode across:

- **Domains**: `retail telecom airline`
- **Concurrency**: `32, 64, 128`
- **Experts**: remote H100 expert LLMs (standard vLLM; accessed via SSH tunnels)

This memo records **prerequisites**, **exact launch commands**, **what scripts are used**, and **the analysis code** (checked into `scripts/analysis/active_window/`) so others can reproduce the same results after open-sourcing.

---

## Repo + environments

- Repo root: `/workspace/ToolOrchestra`
- Required conda envs (must exist on the 5090 machine):
  - `vllm1` (baseline vLLM==0.9.2; used for ThunderReact runs)
  - `vllm-continuum` (vllm-continuum fork; used for Continuum runs)

The launcher scripts (`scripts/5090/launch_{thunderreact,continuum}.sh`) activate the correct env internally.

**Orchestrator checkpoint**:

- `CKPT_DIR` must point to `Nemotron-Orchestrator-8B` checkpoint.
- Recommended: source `setup_envs.sh` to set `CKPT_DIR` and related paths.

```bash
set +u
source /workspace/ToolOrchestra/setup_envs.sh
set -u
echo "CKPT_DIR=${CKPT_DIR}"
```

Note: `set +u` is used because `setup_envs.sh` may reference optional API key env vars that aren’t always set.

---

## Prerequisites (H100 experts + SSH tunnels)

### Expert LLMs

Experts are deployed on a separate H100 cluster using standard vLLM (no scheduler mods). We treat them as OpenAI-compatible endpoints.

Ports (localhost on 5090 after tunneling):

- `openai/gpt-oss-20b`: `1910-1913`
- `Qwen/Qwen3-32B-FP8`: `1904-1905`
- `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`: `1920-1921`

### SSH tunnels

Run the port-forward tunnel script in a persistent tmux session before running experiments:

```bash
cd /workspace/ToolOrchestra
bash scripts/5090/tunnel_experts_via_head.sh
```

Validate tunnels:

```bash
ss -lntp | egrep ':(1910|1911|1912|1913|1904|1905|1920|1921) '
curl -sS --max-time 5 http://127.0.0.1:1910/health
curl -sS --max-time 5 http://127.0.0.1:1904/health
curl -sS --max-time 5 http://127.0.0.1:1920/health
```

We also versioned a preflight check script:

```bash
bash /workspace/ToolOrchestra/scripts/5090/pre_experiment_check.sh
```

---

## Experiment ID + output layout

We store console logs and the sequential-run directories under:

```bash
export EXPERIMENT_ID="5090_thunderreact_continuum_$(date +%Y%m%d_%H%M%S)"
export OUTPUT_BASE="/workspace/ToolOrchestra/outputs/${EXPERIMENT_ID}"
mkdir -p "${OUTPUT_BASE}"
echo "${EXPERIMENT_ID}" > "${OUTPUT_BASE}/experiment_id.txt"
```

### Run directories

Each launcher creates a timestamped run directory under `/workspace/ToolOrchestra/outputs/`:

- ThunderReact: `outputs/thunderreact_c{C}_YYYYMMDD_HHMMSS/`
- Continuum: `outputs/continuum_c{C}_YYYYMMDD_HHMMSS/`

For convenience, we used `scripts/5090/run_remaining_matrix.sh` to **move** the latest run directory into:

- `${OUTPUT_BASE}/thunderreact_c32`, `${OUTPUT_BASE}/thunderreact_c64`, `${OUTPUT_BASE}/thunderreact_c128`
- `${OUTPUT_BASE}/continuum_c32`, `${OUTPUT_BASE}/continuum_c64`, `${OUTPUT_BASE}/continuum_c128`

In the 2026-01-04 run, the final directories used for analysis were:

- TR runs: under `${OUTPUT_BASE}/thunderreact_c{32,64,128}`
- CT runs (timestamped): under `/workspace/ToolOrchestra/outputs/continuum_c32_20260104_191004`,
  `/workspace/ToolOrchestra/outputs/continuum_c64_20260104_205216`,
  `/workspace/ToolOrchestra/outputs/continuum_c128_20260104_223636`

---

## Launching experiments (exact commands)

### Common env

All runs used:

```bash
export DOMAINS="retail telecom airline"
export LOG_LEVEL="PROFILE"
```

### ThunderReact C=32 (manual launch)

```bash
bash /workspace/ToolOrchestra/scripts/5090/pre_experiment_check.sh

cd /workspace/ToolOrchestra
set +u
source /workspace/ToolOrchestra/setup_envs.sh >/dev/null
set -u

export DOMAINS="retail telecom airline"
export LOG_LEVEL="PROFILE"

bash scripts/5090/launch_thunderreact.sh 32 2>&1 | tee "${OUTPUT_BASE}/tr32_console.log"
```

What this script does internally (see `scripts/5090/launch_thunderreact.sh`):

- starts vLLM backend: `vllm serve $CKPT_DIR --port 8100 ...`
- waits for `/health` and `/v1/models`
- starts ThunderReact router: `python multinode_router.py` on `ROUTER_PORT` (default 8000)
- starts KV sampler: `scripts/rollout/collect_kv_cache_timeseries.sh 127.0.0.1 <csv> 5 8100`
- runs evaluation:

```bash
cd evaluation/tau2-bench
python run_oss.py \
  --agent-model "${CKPT_DIR}" \
  --skip-server-start \
  --schedule-mode global \
  --model-config-path "${MODEL_CONFIG_PATH}" \
  --domains ${DOMAINS} \
  --max-concurrency 32 \
  --log-level "${LOG_LEVEL}" \
  --output-dir "${OUTPUT_DIR}/outputs" \
  --log-dir "${LOG_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/driver.log"
```

### Remaining matrix (sequential runner)

After TR32 existed, we ran the remaining 5 experiments sequentially:

```bash
chmod +x /workspace/ToolOrchestra/scripts/5090/run_remaining_matrix.sh

(
  export DOMAINS="retail telecom airline"
  export LOG_LEVEL="PROFILE"
  export FORCE_RERUN="0"
  /workspace/ToolOrchestra/scripts/5090/run_remaining_matrix.sh "${OUTPUT_BASE}" 2>&1 | tee "${OUTPUT_BASE}/runner_remaining.log"
)
```

This runner:

- waits for idle GPU/ports between runs
- launches TR64 → TR128 → CT32 → CT64 → CT128
- moves latest timestamped output dir into `${OUTPUT_BASE}/{thunderreact,continuum}_c{...}`

### Manual Continuum launches (if you prefer not to use the runner)

Example CT64:

```bash
cd /workspace/ToolOrchestra
set +u
source /workspace/ToolOrchestra/setup_envs.sh >/dev/null
set -u

export DOMAINS="retail telecom airline"
export LOG_LEVEL="PROFILE"

nohup bash scripts/5090/launch_continuum.sh 64 > "${OUTPUT_BASE}/continuum_c64_console.log" 2>&1 &
```

The Continuum launcher (see `scripts/5090/launch_continuum.sh`) internally:

- starts continuum vLLM:

```bash
vllm serve "${CKPT_DIR}" --port 1900 --scheduling-policy continuum ...
```

- waits for `/health` + `/v1/models`
- starts KV sampler against port 1900
- runs the same `run_oss.py` command with `--max-concurrency 64`
- on exit, gracefully terminates vLLM to ensure `scheduler_timestamps` flushes to disk

---

## What we logged and where

Per run directory:

- `driver.log`: full run log (includes `[PROFILE]` lines)
- `logs/`: contains vLLM log and tau2 logs
- `kv_cache_timeseries.csv`: sampled every 5s from `/metrics`
- Continuum only: `scheduler_timestamps`

---

## Profiling / metrics definitions used in analysis

We set `LOG_LEVEL=PROFILE`, which makes tau2 driver emit `[PROFILE] ...` records into `driver.log`.

We define:

- **tasks done** = count of `FINISHED SIMULATION:` lines in `driver.log`
  - (This matches “tasks done”, not “tasks started”.)
- **steps done** = count of `[PROFILE] ... type=step_complete ...` lines in `driver.log`
- **KV cache usage** = `vllm:kv_cache_usage_perc` sampled into `kv_cache_timeseries.csv`

### Active window definition (to avoid long-tail domination)

We use an “active window” based on KV cache usage:

> **window_start** = first timestamp where `kv_cache_usage_perc > 70%`  
> **window_end** = last timestamp where `kv_cache_usage_perc > 70%`

This is robust to runs that get stuck and spend hours at ~0% KV usage.

---

## Saved analysis code (checked into repo)

All scripts are under:

`/workspace/ToolOrchestra/scripts/analysis/active_window/`

### Generate the two throughput tables (tasks/min and steps/s)

```bash
cd /workspace/ToolOrchestra

python3 scripts/analysis/active_window/compare_tr_ct_active_window.py \
  --threshold 70 \
  --tr32 "${OUTPUT_BASE}/thunderreact_c32" \
  --tr64 "${OUTPUT_BASE}/thunderreact_c64" \
  --tr128 "${OUTPUT_BASE}/thunderreact_c128" \
  --ct32 "/workspace/ToolOrchestra/outputs/continuum_c32_20260104_191004" \
  --ct64 "/workspace/ToolOrchestra/outputs/continuum_c64_20260104_205216" \
  --ct128 "/workspace/ToolOrchestra/outputs/continuum_c128_20260104_223636"
```

### For each run: how much work happened outside the active window

```bash
python3 scripts/analysis/active_window/window_outside_breakdown.py --threshold 70 --name TR32 --run-dir "${OUTPUT_BASE}/thunderreact_c32"
python3 scripts/analysis/active_window/window_outside_breakdown.py --threshold 70 --name TR64 --run-dir "${OUTPUT_BASE}/thunderreact_c64"
python3 scripts/analysis/active_window/window_outside_breakdown.py --threshold 70 --name CT32 --run-dir "/workspace/ToolOrchestra/outputs/continuum_c32_20260104_191004"
python3 scripts/analysis/active_window/window_outside_breakdown.py --threshold 70 --name CT64 --run-dir "/workspace/ToolOrchestra/outputs/continuum_c64_20260104_205216"
python3 scripts/analysis/active_window/window_outside_breakdown.py --threshold 70 --name CT128 --run-dir "/workspace/ToolOrchestra/outputs/continuum_c128_20260104_223636"
```

### TR128 stuck diagnosis

```bash
python3 scripts/analysis/active_window/tr128_stuck_report.py \
  --run-dir "${OUTPUT_BASE}/thunderreact_c128" \
  --expected-total 278
```

---

## Notes on anomalies observed (2026-01-04 run)

- **TR128** and **CT128** each had a small number of tasks that didn’t complete due to **context length exceeded** causing **infinite retries** in the driver.
  - We manually terminated those runs after verifying outputs were written.
  - This is a workload tail issue (some tasks generate extremely long dialogues).

---

## Artifact manifest (markdown tables)

This section pins the **exact artifacts** used to produce the reported numbers, so others can verify they are operating on the same outputs.

**How it was generated**:

```bash
cd /workspace/ToolOrchestra
python3 scripts/analysis/active_window/generate_artifact_manifest.py --help
```

The manifest below was generated with:

```bash
cd /workspace/ToolOrchestra
python3 scripts/analysis/active_window/generate_artifact_manifest.py \
  --threshold 70 \
  --experiment-id 5090_thunderreact_continuum_20260104_043636 \
  --output-base /workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636 \
  --tr32 /workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/thunderreact_c32 \
  --tr64 /workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/thunderreact_c64 \
  --tr128 /workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/thunderreact_c128 \
  --ct32 /workspace/ToolOrchestra/outputs/continuum_c32_20260104_191004 \
  --ct64 /workspace/ToolOrchestra/outputs/continuum_c64_20260104_205216 \
  --ct128 /workspace/ToolOrchestra/outputs/continuum_c128_20260104_223636 \
  --tr32-console /workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/tr32_console_rerun.log \
  --tr64-console /workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/runner_remaining_v5.log \
  --tr128-console /workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/runner_remaining_v5.log \
  --ct32-console /workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/continuum_c32_console_v2.log \
  --ct64-console /workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/continuum_c64_console.log \
  --ct128-console /workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/continuum_c128_console.log
```

### Artifact manifest

#### Repo + system

| Item | Value |
| :-- | :-- |
| experiment_id | `5090_thunderreact_continuum_20260104_043636` |
| output_base | `/workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636` |
| git_head | `66c569740143d8dc5dfd32bedf3167507b720dca` |
| git_dirty_files | `18` |
| uname | `Linux e930a9df7df1 6.8.0-60-generic #63~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue Apr 22 19:00:15 UTC 2 x86_64 x86_64 x86_64 GNU/Linux` |
| nvidia_smi_head | `NVIDIA-SMI 570.144  Driver Version: 570.144  CUDA Version: 12.8` |

#### Runs (paths + active-window throughput)

Active window definition: **first KV>70% → last KV>70%** (see `scripts/analysis/active_window/README.md`).

| Scenario | Scheduler | Run dir | Console log | Global summary (tail) | Active window (start→end) | tasks/min | steps/s |
| :-- | :-- | :-- | :-- | :-- | :-- | --: | --: |
| C=32 | ThunderReact | `/workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/thunderreact_c32` | `/workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/tr32_console_rerun.log` | `[2026-01-04 06:57:36] Global evaluation complete: ok=278 error=0 total=278 elapsed_s=5350.3 tasks_per_min=3.12` | `2026-01-04 05:33:20` → `2026-01-04 06:36:28` | 3.72 | 2.97 |
| C=64 | ThunderReact | `/workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/thunderreact_c64` | `/workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/runner_remaining_v5.log` | `[2026-01-04 08:40:39] Global evaluation complete: ok=277 error=1 total=278 elapsed_s=5352.0 tasks_per_min=3.11` | `2026-01-04 07:13:41` → `2026-01-04 08:13:48` | 4.04 | 3.37 |
| C=128 | ThunderReact | `/workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/thunderreact_c128` | `/workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/runner_remaining_v5.log` | *(no `Global evaluation complete` line; run manually terminated due to infinite retry on context length exceeded)* | `2026-01-04 08:44:06` → `2026-01-04 10:06:07` | 3.22 | 2.52 |
| C=32 | Continuum | `/workspace/ToolOrchestra/outputs/continuum_c32_20260104_191004` | `/workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/continuum_c32_console_v2.log` | `[2026-01-04 20:38:32] Global evaluation complete: ok=277 error=1 total=278 elapsed_s=5140.8 tasks_per_min=3.23` | `2026-01-04 19:19:05` → `2026-01-04 20:25:37` | 3.52 | 2.83 |
| C=64 | Continuum | `/workspace/ToolOrchestra/outputs/continuum_c64_20260104_205216` | `/workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/continuum_c64_console.log` | `[2026-01-04 22:32:23] Global evaluation complete: ok=239 error=39 total=278 elapsed_s=5772.6 tasks_per_min=2.48` | `2026-01-04 20:58:07` → `2026-01-04 22:30:56` | 2.49 | 1.89 |
| C=128 | Continuum | `/workspace/ToolOrchestra/outputs/continuum_c128_20260104_223636` | `/workspace/ToolOrchestra/outputs/5090_thunderreact_continuum_20260104_043636/continuum_c128_console.log` | *(no `Global evaluation complete` line; run manually terminated due to infinite retry on context length exceeded)* | `2026-01-04 22:42:10` → `2026-01-05 00:19:56` | 2.39 | 1.97 |

#### Task result files (from `outputs/all_domains/`)

These counts are based on files written by tau2-bench under `outputs/all_domains/`:
`{domain}__{task_id}__trial0.json`.

| Scenario | Scheduler | retail | telecom | airline | total |
| :-- | :-- | --: | --: | --: | --: |
| C=32 | ThunderReact | 114 | 114 | 50 | 278 |
| C=64 | ThunderReact | 114 | 114 | 50 | 278 |
| C=128 | ThunderReact | 114 | 112 | 50 | 276 |
| C=32 | Continuum | 114 | 114 | 50 | 278 |
| C=64 | Continuum | 114 | 114 | 50 | 278 |
| C=128 | Continuum | 114 | 113 | 50 | 277 |

#### Key file hashes (sha256)

| Scenario | Scheduler | File | Bytes | Lines | sha256 |
| :-- | :-- | :-- | --: | --: | :-- |
| C=32 | ThunderReact | `driver.log` | 26169361 | 266286 | `09bb32767960efcf053711b55b554689526ac728657e6d43aa832bf32e9da3b2` |
| C=32 | ThunderReact | `kv_cache_timeseries.csv` | 340229 | 2129 | `e95e72e689ef09dea8f663f1993ffadf0c3c1396d987579e024b4835d9461a1c` |
| C=32 | ThunderReact | `model_config_5090_thunderreact.json` | 874 | 26 | `7cdc713b5aa3aeb8746656a9be3792e155f93ad2f09c68e16aa75d8057ddadc0` |
| C=32 | ThunderReact | `router_backends.json` | 121 | 1 | `983e8b59061b2295b37de72860c95fc2e79e3f32ddb8ecce4c097152efaaef5b` |
| C=32 | ThunderReact | `router_programs.json` | 2 | 1 | `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a` |
| C=64 | ThunderReact | `driver.log` | 25545079 | 261705 | `1fa399f674038a9602e10f2e50691e193b6c74a4d30f1065f22b5d0e4801240c` |
| C=64 | ThunderReact | `kv_cache_timeseries.csv` | 341076 | 2131 | `c9f60df365ce49bf2686e5083ca1001fc4cd3ab6cfbb276a33069ba738d43e02` |
| C=64 | ThunderReact | `model_config_5090_thunderreact.json` | 874 | 26 | `62d406701044884ffc235653abd4be7a7be6cc185159bbd89143c91fa8e5875d` |
| C=64 | ThunderReact | `router_backends.json` | 121 | 1 | `983e8b59061b2295b37de72860c95fc2e79e3f32ddb8ecce4c097152efaaef5b` |
| C=64 | ThunderReact | `router_programs.json` | 2 | 1 | `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a` |
| C=128 | ThunderReact | `driver.log` | 39539850 | 390465 | `495e223a77938203cf1e9d2f1a38ee4f8720241e3b53412fb2c4eebd4ba7dcae` |
| C=128 | ThunderReact | `kv_cache_timeseries.csv` | 2393424 | 14723 | `aa36c00469578101772cd65c0ea2a492557ebc1346f8520da47db10894ebdd83` |
| C=128 | ThunderReact | `model_config_5090_thunderreact.json` | 875 | 26 | `e8b0f0a7671c319eb8c74faa955c39b9aaed114da7dfe385aba5a2007ade54d9` |
| C=32 | Continuum | `driver.log` | 24782949 | 256720 | `2bc1634b940505908955aecd7e3a8bee47a53b32fc8ebe042a4549e6fab67ee2` |
| C=32 | Continuum | `kv_cache_timeseries.csv` | 326429 | 2057 | `da3b29f1de3eda5ec665d93f1130f20d6699b6895e162b11cdc4d5e35e3f6f8c` |
| C=32 | Continuum | `scheduler_timestamps` | 1292162 | 58078 | `b33b2e2f10d196cccbad90d6e6287c88e167c6d13077825d9c307c889b7a1793` |
| C=32 | Continuum | `model_config_5090_continuum.json` | 868 | 26 | `85d04be915e670474cf0240449e7ba01f14024f1a12d8a2f47480a71db5afa84` |
| C=64 | Continuum | `driver.log` | 21870168 | 244271 | `603692de7fb1333b8f955258c6f8fa60b3060cfe6eca817da6e7b83474ea1fbc` |
| C=64 | Continuum | `kv_cache_timeseries.csv` | 362860 | 2290 | `a8054c0b08cd3cdb04b9ab83853df0a9cc6bacb8db63acf9200ca96316f754d8` |
| C=64 | Continuum | `scheduler_timestamps` | 2117385 | 91447 | `b594632bc531847872df55c42e417ae41efadd5b5fa2b56b1e8547a143d03da6` |
| C=64 | Continuum | `model_config_5090_continuum.json` | 868 | 26 | `b79d0a11758f38b381f4f26e8e715f53ed30e636751d2768c2ce6ce7cea98cf3` |
| C=128 | Continuum | `driver.log` | 23698991 | 289945 | `efd4acb616cce2ebab32f0ec48b45b426489ec0a13f2a88c5db33392c28d2775` |
| C=128 | Continuum | `kv_cache_timeseries.csv` | 484501 | 3053 | `5aaae0930b0d584c55e2899aacd72985275dd343527b949119cb1f3f1790eb97` |
| C=128 | Continuum | `scheduler_timestamps` | 2236017 | 96907 | `23b8899694c60c548bcccbac002c5f0a02428d3f037d3f0ec7c754b348debe45` |
| C=128 | Continuum | `model_config_5090_continuum.json` | 869 | 26 | `dc03c37b0d719bc97feee1e5f8ac8335c253f96d459ea22ce47df49608ff0db8` |


