#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SLURM version for running tau2-bench with OSS expert models:
# - expert-1: openai/gpt-oss-120b (replaces gpt-5)
# - expert-2: openai/gpt-oss-20b (replaces gpt-5-mini)
# - expert-3: Qwen/Qwen3-Next-80B-A3B-Instruct (replaces Qwen/Qwen3-32B)
#
# All expert models use speculative decoding for faster inference:
# - GPT-OSS-120B: EAGLE3 with nvidia/gpt-oss-120b-Eagle3-v2
# - GPT-OSS-20B: EAGLE3 with RedHatAI/gpt-oss-20b-speculator.eagle3
# - Qwen3-Next-80B: MTP (Multi-Token Prediction)

import os
import sys
import json
import time
import signal
import argparse
import subprocess
import selectors
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

def log(msg: str):
    """Print timestamped log message"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ============================================================================
# Model Configuration
# ============================================================================

# Expert model mappings (OSS replacements for proprietary models)
OSS_EXPERT_MAPPING = {
    "expert-1": "openai/gpt-oss-120b",      # replaces gpt-5
    "expert-2": "openai/gpt-oss-20b",       # replaces gpt-5-mini
    "expert-3": "Qwen/Qwen3-Next-80B-A3B-Instruct",  # replaces Qwen/Qwen3-32B
}

# Model deployment specifications
MODEL_SPECS = {
    "openai/gpt-oss-120b": {
        "tensor_parallel_size": 4,  # TP4 for lower latency on 120B
        "speculative_config": json.dumps({
            "model": "nvidia/gpt-oss-120b-Eagle3-v2",
            "num_speculative_tokens": 3,
            "method": "eagle3",
            "draft_tensor_parallel_size": 1
        }),
        "extra_args": [
            "--async-scheduling",
            "--no-enable-prefix-caching",
            "--max-cudagraph-capture-size", "2048",
            "--max-num-batched-tokens", "8192",
            "--stream-interval", "20",
        ],
        "gpu_memory_utilization": 0.95,
        "num_instances": 1,  # One instance with TP4
    },
    "openai/gpt-oss-20b": {
        "tensor_parallel_size": 1,  # TP1, small enough for single GPU
        "speculative_config": json.dumps({
            "model": "RedHatAI/gpt-oss-20b-speculator.eagle3",
            "num_speculative_tokens": 3,
            "method": "eagle3"
        }),
        "extra_args": [
            "--async-scheduling",
            "--no-enable-prefix-caching",
        ],
        "gpu_memory_utilization": 0.95,
        "num_instances": 4,  # 4 DP instances for high throughput
    },
    "Qwen/Qwen3-Next-80B-A3B-Instruct": {
        "tensor_parallel_size": 4,  # TP4 required for 80B
        "speculative_config": json.dumps({
            "method": "qwen3_next_mtp",
            "num_speculative_tokens": 2
        }),
        "extra_args": [
            "--tokenizer-mode", "auto",
            "--no-enable-chunked-prefill",
            "--compilation_config.pass_config.enable_fi_allreduce_fusion", "true",
            "--compilation_config.pass_config.enable_noop", "true",
        ],
        "gpu_memory_utilization": 0.95,
        "num_instances": 1,  # One instance with TP4
    },
}

# ============================================================================
# SLURM Script Templates
# ============================================================================

SLURM_HEADER = """#!/bin/bash

#SBATCH --partition batch
#SBATCH --time 04:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node={gpus}
#SBATCH --job-name {job_name}
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err

set -x

# Print node IP for service discovery
hostname -i

# Setup environment - must source conda.sh first for non-interactive shells
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm1

# Verify vllm is available
which vllm || echo "ERROR: vllm not found in PATH"

echo "HF_HOME: $HF_HOME"
echo "CKPT_DIR: $CKPT_DIR"
echo "Job: {job_name}"
echo "Node: $(hostname)"

# Set cache directories
CACHE_BASE="${{USER_PATH:-$HOME}}/cache/vllm"
mkdir -p "$CACHE_BASE"

# Check GPU status before launching vLLM servers
echo "=== GPU Status at Job Start ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv
echo "=== End GPU Status ==="
"""

def generate_agent_script(job_name: str, ckpt_dir: str, num_instances: int = 4, start_port: int = 1900) -> str:
    """Generate SLURM script for agent (Orchestrator-8B) servers"""
    script = SLURM_HEADER.format(gpus=8, job_name=job_name)

    for i in range(num_instances):
        port = start_port + i
        script += f"""
# Agent server {i+1}
export VLLM_CACHE_ROOT="$CACHE_BASE/{job_name}_agent_{i}"
CUDA_VISIBLE_DEVICES={i} vllm serve {ckpt_dir} \\
    --enable-auto-tool-choice \\
    --tool-call-parser hermes \\
    --port {port} &
sleep 30
"""

    return script


def generate_expert_script(
    job_name: str,
    model_name: str,
    start_gpu: int,
    start_port: int,
    instance_id: int = 0
) -> str:
    """Generate vLLM serve command for an expert model"""
    spec = MODEL_SPECS[model_name]
    tp_size = spec["tensor_parallel_size"]

    # Calculate GPU range
    if tp_size > 1:
        gpu_ids = ",".join(str(start_gpu + i) for i in range(tp_size))
    else:
        gpu_ids = str(start_gpu)

    cmd = f"""
# Expert: {model_name} (instance {instance_id})
export VLLM_CACHE_ROOT="$CACHE_BASE/{job_name}_{model_name.replace('/', '_')}_{instance_id}"
export CUDA_VISIBLE_DEVICES={gpu_ids}
vllm serve {model_name} \\
    --enable-auto-tool-choice \\
    --tool-call-parser hermes \\
    --port {start_port} \\
    --tensor-parallel-size {tp_size} \\
    --gpu-memory-utilization {spec['gpu_memory_utilization']} \\
    --speculative-config '{spec['speculative_config']}' \\
"""

    # Add extra args
    for arg in spec.get("extra_args", []):
        cmd += f"    {arg} \\\n"

    cmd += "    &\nsleep 60\n"
    return cmd


def generate_node1_script(job_name: str, ckpt_dir: str) -> str:
    """
    Generate SLURM script for Node 1 (Agent + Expert-3):
    - GPU 0-3: Orchestrator-8B (4 DP instances)
    - GPU 4-7: Qwen3-Next-80B-A3B-Instruct (TP4)
    """
    script = SLURM_HEADER.format(gpus=8, job_name=job_name)

    # Agent servers (GPU 0-3, ports 1900-1903) - 4 DP instances
    for i in range(4):
        script += f"""
# Agent server {i+1} (Orchestrator-8B)
export VLLM_CACHE_ROOT="$CACHE_BASE/{job_name}_agent_{i}"
export CUDA_VISIBLE_DEVICES={i}
vllm serve {ckpt_dir} \\
    --enable-auto-tool-choice \\
    --tool-call-parser hermes \\
    --port {1900 + i} \\
    --gpu-memory-utilization 0.95 &
sleep 30
"""

    # Qwen3-Next-80B (GPU 4-7, TP4, port 1904)
    script += generate_expert_script(
        job_name=job_name,
        model_name="Qwen/Qwen3-Next-80B-A3B-Instruct",
        start_gpu=4,
        start_port=1904,
        instance_id=0
    )

    script += "\n# Keep job running\nsleep 15000\n"
    return script


def generate_node2_script(job_name: str) -> str:
    """
    Generate SLURM script for Node 2 (Expert-1 + Expert-2):
    - GPU 0-3: GPT-OSS-120B (TP4)
    - GPU 4-7: GPT-OSS-20B (4 DP instances)
    """
    script = SLURM_HEADER.format(gpus=8, job_name=job_name)

    # GPT-OSS-120B (GPU 0-3, TP4, port 1910)
    script += generate_expert_script(
        job_name=job_name,
        model_name="openai/gpt-oss-120b",
        start_gpu=0,
        start_port=1910,
        instance_id=0
    )

    # GPT-OSS-20B (GPU 4-7, 4 DP instances, ports 1914-1917)
    for i in range(4):
        script += generate_expert_script(
            job_name=job_name,
            model_name="openai/gpt-oss-20b",
            start_gpu=4 + i,
            start_port=1914 + i,
            instance_id=i
        )

    script += "\n# Keep job running\nsleep 15000\n"
    return script


# ============================================================================
# SLURM Job Management
# ============================================================================

def get_jobs() -> List[Dict]:
    """Get list of current SLURM jobs for this user"""
    try:
        result = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', '')],
            timeout=60,
            capture_output=True,
            text=True
        )
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        jobs = []
        for line in lines:
            parts = [p for p in line.split(' ') if p]
            if len(parts) >= 6:
                # Parse running time
                running_time = parts[5]
                total_time = 0
                time_parts = running_time.split(':')
                if '-' in time_parts[0]:
                    total_time = 3600  # More than an hour
                elif len(time_parts) == 2:
                    total_time = int(time_parts[0]) * 60 + int(time_parts[1])
                elif len(time_parts) == 3:
                    total_time = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])

                jobs.append({
                    'id': parts[0],
                    'name': parts[2],
                    'status': parts[4],
                    'total_time': total_time,
                    'reason': parts[-1] if len(parts) > 7 else ''
                })
        return jobs
    except Exception as e:
        log(f"Error getting jobs: {e}")
        return []


def wait_for_job_ready(job_name: str, min_time: int = 600, timeout: int = 1800) -> Optional[str]:
    """Wait for a SLURM job to be running for at least min_time seconds"""
    start = time.time()
    while time.time() - start < timeout:
        jobs = get_jobs()
        for job in jobs:
            if job['name'] == job_name and job['status'].lower() == 'r':
                if job['total_time'] >= min_time:
                    # Read IP from output file
                    out_file = f"{job_name}.out"
                    if os.path.isfile(out_file):
                        with open(out_file) as f:
                            lines = f.readlines()
                        if lines:
                            return lines[0].strip()
                else:
                    remaining = min_time - job['total_time']
                    log(f"Job {job_name} running for {job['total_time']}s, waiting {remaining}s more...")
        time.sleep(30)
    return None


def submit_job(script_path: str) -> bool:
    """Submit a SLURM job"""
    try:
        result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
        if result.returncode == 0:
            log(f"Submitted job: {script_path}")
            return True
        else:
            log(f"Failed to submit {script_path}: {result.stderr}")
            return False
    except Exception as e:
        log(f"Error submitting job: {e}")
        return False


def cancel_job(job_id: str):
    """Cancel a SLURM job"""
    os.system(f"scancel {job_id}")


# ============================================================================
# Model Config Generation
# ============================================================================

def generate_model_config(
    node1_ip: str,
    node2_ip: str,
    ckpt_dir: str,
    output_path: str = "model_config_oss.json"
) -> Dict:
    """
    Generate model configuration for tau2-bench with OSS models.

    Node 1: Agent (ports 1900-1903) + Qwen3-Next-80B (port 1904)
    Node 2: GPT-OSS-120B (port 1910) + GPT-OSS-20B (ports 1914-1917)
    """

    config = {
        # Agent model (Orchestrator-8B) - Node 1, GPU 0-3
        ckpt_dir: [
            {"ip_addr": node1_ip, "port": "1900"},
            {"ip_addr": node1_ip, "port": "1901"},
            {"ip_addr": node1_ip, "port": "1902"},
            {"ip_addr": node1_ip, "port": "1903"},
        ],
        # Expert-3: Qwen3-Next-80B - Node 1, GPU 4-7 (TP4)
        "Qwen/Qwen3-Next-80B-A3B-Instruct": [
            {"ip_addr": node1_ip, "port": "1904"},
        ],
        # Expert-1: GPT-OSS-120B - Node 2, GPU 0-3 (TP4)
        "openai/gpt-oss-120b": [
            {"ip_addr": node2_ip, "port": "1910"},
        ],
        # Expert-2: GPT-OSS-20B - Node 2, GPU 4-7 (4 DP instances)
        "openai/gpt-oss-20b": [
            {"ip_addr": node2_ip, "port": "1914"},
            {"ip_addr": node2_ip, "port": "1915"},
            {"ip_addr": node2_ip, "port": "1916"},
            {"ip_addr": node2_ip, "port": "1917"},
        ],
        "vllm_model_config_path": output_path,
        # OSS expert mapping for llm_utils.py
        "oss_expert_mapping": OSS_EXPERT_MAPPING,
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    log(f"Model configuration written to {output_path}")
    log(f"  Node 1 ({node1_ip}): Agent (1900-1903), Qwen3-Next-80B (1904)")
    log(f"  Node 2 ({node2_ip}): GPT-OSS-120B (1910), GPT-OSS-20B (1914-1917)")
    return config


# ============================================================================
# Evaluation Runner
# ============================================================================

def run_evaluation(
    domain: str,
    agent_llm: str,
    user_llm: str,
    task_path: str,
    output_file: str,
    model_config_path: str,
    max_steps: int = 200,
    num_trials: int = 1,
    use_model_tool: bool = True,
    max_concurrency: int = 10,
    num_tasks: Optional[int] = None,
    max_errors: int = 10,
    seed: int = 300,
    log_level: str = "PROFILE",  # Enable profiling by default
    log_dir: str = "logs",
    overall_state: Optional[Dict[str, Any]] = None
) -> int:
    """Run tau2-bench evaluation for a specific domain"""
    log(f"========== Starting evaluation: {domain.upper()} ==========")

    cmd = [
        sys.executable, "-m", "tau2.cli",
        "--domain", domain,
        "--agent-llm", agent_llm,
        "--user-llm", user_llm,
        "--num-trials", str(num_trials),
        "--task_path", task_path,
        "--max-steps", str(max_steps),
        "--output_file", output_file,
        "--model_config_path", model_config_path,
        "--max-concurrency", str(max_concurrency),
        "--max-errors", str(max_errors),
        "--seed", str(seed),
        "--log-level", log_level,
    ]

    if num_tasks is not None:
        cmd += ["--num-tasks", str(num_tasks)]

    if use_model_tool:
        cmd.append("--use_model_tool")

    log(f"Running command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    os.makedirs(log_dir, exist_ok=True)
    env["TAU2_LOG_FILE"] = os.path.join(log_dir, f"tau2_{domain}.log")

    eval_log_path = os.path.join(log_dir, f"eval_{domain}.log")

    def _fmt_hms(seconds: float) -> str:
        if seconds < 0:
            seconds = 0
        s = int(seconds)
        h = s // 3600
        m = (s % 3600) // 60
        s = s % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _print_overall_eta() -> None:
        if not overall_state:
            return
        total = int(overall_state.get("total") or 0)
        done = int(overall_state.get("done") or 0)
        start_ts = float(overall_state.get("start_ts") or time.time())
        now_ts = time.time()
        elapsed_s = max(0.0, now_ts - start_ts)
        if total <= 0 or done <= 0:
            return
        remaining = max(0, total - done)
        per_task = elapsed_s / done
        eta_s = per_task * remaining
        print(
            "\033[31m"
            f"[OVERALL_ETA] done={done}/{total} elapsed={_fmt_hms(elapsed_s)} eta={_fmt_hms(eta_s)}"
            "\033[0m",
            flush=True,
        )

    with open(eval_log_path, "a", encoding="utf-8", buffering=1) as eval_log:
        process = subprocess.Popen(
            cmd,
            cwd=os.getcwd(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        log(f"Process started with PID: {process.pid}")

        selector = selectors.DefaultSelector()
        if process.stdout is not None:
            selector.register(process.stdout, selectors.EVENT_READ, data="STDOUT")
        if process.stderr is not None:
            selector.register(process.stderr, selectors.EVENT_READ, data="STDERR")

        retcode: Optional[int] = None
        last_output_ts = time.time()
        last_heartbeat_ts = last_output_ts

        while True:
            retcode = process.poll()

            events = selector.select(timeout=0.2)
            for key, _mask in events:
                stream_name = key.data
                fileobj = key.fileobj
                try:
                    line = fileobj.readline()
                except Exception as e:
                    log(f"WARNING: Failed reading {stream_name}: {e}")
                    try:
                        selector.unregister(fileobj)
                    except Exception:
                        pass
                    continue

                if line:
                    last_output_ts = time.time()

                    try:
                        eval_log.write(f"[tau2/cli.py {stream_name}] {line}")
                        eval_log.flush()
                    except Exception:
                        pass

                    raw = line.rstrip("\n")
                    if raw.startswith("[TAU2_TASK_COMPLETE]") and overall_state is not None:
                        overall_state["done"] = int(overall_state.get("done") or 0) + 1
                        _print_overall_eta()

                    # Print PROFILE lines in cyan for visibility
                    if "[PROFILE]" in raw:
                        print(f"\033[36m[tau2/cli.py {stream_name}] {raw}\033[0m", flush=True)
                    else:
                        print(f"[tau2/cli.py {stream_name}] {raw}", flush=True)
                else:
                    try:
                        selector.unregister(fileobj)
                    except Exception:
                        pass

            now = time.time()
            if retcode is None and (now - last_heartbeat_ts) > 30 and (now - last_output_ts) > 30:
                last_heartbeat_ts = now
                log(f"{domain.upper()} still running (pid={process.pid}); no output for {int(now - last_output_ts)}s...")

            if retcode is not None and not selector.get_map():
                break

        if retcode == 0:
            log(f"========== Finished {domain.upper()} evaluation successfully ==========")
        else:
            log(f"========== {domain.upper()} evaluation failed with code {retcode} ==========")

        return retcode


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run tau2-bench evaluation on SLURM with OSS expert models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
OSS Expert Model Mapping:
  expert-1: openai/gpt-oss-120b (EAGLE3 speculative decoding)
  expert-2: openai/gpt-oss-20b (EAGLE3 speculative decoding)
  expert-3: Qwen/Qwen3-Next-80B-A3B-Instruct (MTP speculative decoding)

GPU Allocation (2 nodes, 16 GPU total):
  Node 1 (8 GPU) - Agent + Expert-3:
    - GPU 0-3: Orchestrator-8B (4 DP instances, ports 1900-1903)
    - GPU 4-7: Qwen3-Next-80B-A3B (TP4 + MTP, port 1904)

  Node 2 (8 GPU) - Expert-1 + Expert-2:
    - GPU 0-3: GPT-OSS-120B (TP4 + EAGLE3, port 1910)
    - GPU 4-7: GPT-OSS-20B (4 DP instances + EAGLE3, ports 1914-1917)

Example usage:
  python run_oss.py
  python run_oss.py --domains retail telecom
  python run_oss.py --num-tasks 10  # Quick test
        """
    )

    # Model configuration
    parser.add_argument(
        "--agent-model",
        type=str,
        default=os.getenv("CKPT_DIR", ""),
        help="Path to the agent model checkpoint (default: $CKPT_DIR)"
    )
    parser.add_argument(
        "--user-llm",
        type=str,
        default="gpt-5",
        help="LLM for user simulation (default: gpt-5, requires OPENAI_API_KEY)"
    )

    # Job configuration
    parser.add_argument(
        "--job-prefix",
        type=str,
        default="oss_eval",
        help="Prefix for SLURM job names (default: oss_eval)"
    )
    parser.add_argument(
        "--server-wait-time",
        type=int,
        default=600,
        help="Seconds to wait for servers to be ready (default: 600)"
    )

    # Evaluation configuration
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=["retail", "telecom", "airline"],
        choices=["mock", "retail", "telecom", "airline"],
        help="Domains to evaluate (default: retail telecom airline)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per task (default: 200)"
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials per task (default: 1)"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Run only first N tasks (for testing)"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum concurrent tasks (default: 10)"
    )
    parser.add_argument(
        "--use_model_tool",
        action="store_true",
        default=True,
        help="Enable model-tool behavior (call_expert)"
    )
    parser.add_argument(
        "--no-use-model-tool",
        dest="use_model_tool",
        action="store_false",
        help="Disable model-tool behavior"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="PROFILE",
        choices=["DEBUG", "PROFILE", "INFO", "WARNING", "ERROR"],
        help="Log level (default: PROFILE for timing data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_oss",
        help="Output directory (default: outputs_oss)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs_oss",
        help="Log directory (default: logs_oss)"
    )
    parser.add_argument(
        "--model-config-path",
        type=str,
        default="model_config_oss.json",
        help="Model config file path (default: model_config_oss.json)"
    )

    # Skip options
    parser.add_argument(
        "--skip-server-start",
        action="store_true",
        help="Skip starting servers (use existing)"
    )

    args = parser.parse_args()

    # Validate
    if not args.agent_model:
        log("ERROR: Agent model path not specified. Set --agent-model or $CKPT_DIR")
        sys.exit(1)

    repo_path = os.getenv("REPO_PATH")
    if not repo_path:
        repo_path = str(Path(__file__).parent.parent.parent.absolute())
        log(f"REPO_PATH not set, inferring: {repo_path}")
        os.environ["REPO_PATH"] = repo_path

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    node1_ip = None
    node2_ip = None

    if not args.skip_server_start:
        # Generate and submit SLURM jobs
        job1_name = f"{args.job_prefix}_node1"
        job2_name = f"{args.job_prefix}_node2"

        # Check for existing jobs and cancel if needed
        jobs = get_jobs()
        for job in jobs:
            if job['name'].startswith(args.job_prefix):
                log(f"Cancelling existing job: {job['name']} ({job['id']})")
                cancel_job(job['id'])
        time.sleep(5)

        # Generate scripts
        log("Generating SLURM scripts...")

        script1 = generate_node1_script(job1_name, args.agent_model)
        script1_path = f"{job1_name}.sh"
        with open(script1_path, 'w') as f:
            f.write(script1)
        log(f"Generated: {script1_path}")

        script2 = generate_node2_script(job2_name)
        script2_path = f"{job2_name}.sh"
        with open(script2_path, 'w') as f:
            f.write(script2)
        log(f"Generated: {script2_path}")

        # Clean up old output files
        for name in [job1_name, job2_name]:
            for ext in ['.out', '.err']:
                path = f"{name}{ext}"
                if os.path.exists(path):
                    os.remove(path)

        # Submit jobs
        log("Submitting SLURM jobs...")
        if not submit_job(script1_path):
            sys.exit(1)
        if not submit_job(script2_path):
            sys.exit(1)

        # Wait for jobs to be ready
        log(f"Waiting for servers to be ready (min {args.server_wait_time}s)...")

        node1_ip = wait_for_job_ready(job1_name, min_time=args.server_wait_time)
        if not node1_ip:
            log("ERROR: Node 1 (agent + expert-1/2) failed to start")
            sys.exit(1)
        log(f"Node 1 ready: {node1_ip}")

        node2_ip = wait_for_job_ready(job2_name, min_time=args.server_wait_time)
        if not node2_ip:
            log("ERROR: Node 2 (expert-3) failed to start")
            sys.exit(1)
        log(f"Node 2 ready: {node2_ip}")

        # Generate model config
        generate_model_config(
            node1_ip=node1_ip,
            node2_ip=node2_ip,
            ckpt_dir=args.agent_model,
            output_path=args.model_config_path
        )
    else:
        log("Skipping server startup (--skip-server-start)")
        if not os.path.exists(args.model_config_path):
            log(f"ERROR: Model config not found: {args.model_config_path}")
            sys.exit(1)

    # Run evaluations
    task_paths = {
        "mock": os.path.join(repo_path, "data/tau2/domains/mock/tasks.json"),
        "retail": os.path.join(repo_path, "data/tau2/domains/retail/tasks.json"),
        "telecom": os.path.join(repo_path, "data/tau2/domains/telecom/tasks.json"),
        "airline": os.path.join(repo_path, "data/tau2/domains/airline/original_tasks.json"),
    }

    # Verify task files
    for domain in args.domains:
        if not os.path.exists(task_paths[domain]):
            log(f"ERROR: Task file not found: {task_paths[domain]}")
            sys.exit(1)

    # Count total tasks
    def _count_tasks(path: str) -> int:
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return len(data)
            if isinstance(data, dict) and "tasks" in data:
                return len(data["tasks"])
        except Exception:
            pass
        return 0

    overall_total = 0
    for domain in args.domains:
        n = _count_tasks(task_paths[domain])
        if args.num_tasks is not None:
            n = min(n, args.num_tasks)
        overall_total += n * args.num_trials

    overall_state = {
        "total": overall_total,
        "done": 0,
        "start_ts": time.time(),
    }

    results = {}
    for domain in args.domains:
        output_file = os.path.join(args.output_dir, f"{domain}.json")

        returncode = run_evaluation(
            domain=domain,
            agent_llm=args.agent_model,
            user_llm=args.user_llm,
            task_path=task_paths[domain],
            output_file=output_file,
            model_config_path=args.model_config_path,
            max_steps=args.max_steps,
            num_trials=args.num_trials,
            use_model_tool=args.use_model_tool,
            max_concurrency=args.max_concurrency,
            num_tasks=args.num_tasks,
            log_level=args.log_level,
            log_dir=args.log_dir,
            overall_state=overall_state,
        )

        results[domain] = "SUCCESS" if returncode == 0 else "FAILED"

    # Print summary
    log("\n" + "=" * 60)
    log("EVALUATION SUMMARY (OSS Expert Models)")
    log("=" * 60)
    log(f"Expert-1: openai/gpt-oss-120b (EAGLE3)")
    log(f"Expert-2: openai/gpt-oss-20b (EAGLE3)")
    log(f"Expert-3: Qwen/Qwen3-Next-80B-A3B-Instruct (MTP)")
    log("-" * 60)
    for domain, status in results.items():
        log(f"{domain.upper()}: {status}")
    log("=" * 60)
    log(f"Output directory: {args.output_dir}")
    log(f"Log directory: {args.log_dir}")
    log(f"Profile logs: {args.log_dir}/tau2_*.log")


if __name__ == "__main__":
    main()
