#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Local version of run.py for running tau2-bench evaluation without SLURM
# This script starts vLLM servers for the agent model locally and runs evaluations

import os
import sys
import json
import time
import signal
import argparse
import subprocess
import requests
import selectors
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

def log(msg: str):
    """Print timestamped log message"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

class VLLMServer:
    """Manages a single vLLM server instance"""
    def __init__(self, model_path: str, port: int, gpu_id: int, tool_parser: str = "hermes"):
        self.model_path = model_path
        self.port = port
        self.gpu_id = gpu_id
        self.tool_parser = tool_parser
        self.process: Optional[subprocess.Popen] = None
        self.ip_addr = "127.0.0.1"
        self.log_file = None
        self.stdout_file = None
        self.stderr_file = None

    def start(self, log_dir: str = "logs"):
        """Start the vLLM server"""
        os.makedirs(log_dir, exist_ok=True)

        # Create log files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stdout_path = os.path.join(log_dir, f"vllm_port_{self.port}_{timestamp}.out")
        stderr_path = os.path.join(log_dir, f"vllm_port_{self.port}_{timestamp}.err")

        self.stdout_file = open(stdout_path, 'w')
        self.stderr_file = open(stderr_path, 'w')

        cmd = [
            "vllm", "serve", self.model_path,
            "--enable-auto-tool-choice",
            "--tool-call-parser", self.tool_parser,
            "--port", str(self.port),
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        log(f"Starting vLLM server: {self.model_path}")
        log(f"  GPU: {self.gpu_id}, Port: {self.port}")
        log(f"  Logs: {stdout_path}, {stderr_path}")

        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self.stdout_file,
            stderr=self.stderr_file,
            text=True
        )

    def is_ready(self, timeout: int = 5) -> bool:
        """Check if the server is ready to accept requests"""
        try:
            log(f"Attempting connection to http://{self.ip_addr}:{self.port}/health")
            response = requests.get(
                f"http://{self.ip_addr}:{self.port}/health",
                timeout=timeout
            )
            log(f"Received response status: {response.status_code}")
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            log(f"Connection failed: {str(e)}")
            return False

    def wait_until_ready(self, max_wait: int = 600, check_interval: int = 10):
        """Wait until the server is ready"""
        log(f"Waiting for vLLM server on port {self.port} to be ready...")
        start_time = time.time()

        while time.time() - start_time < max_wait:
            if self.process and self.process.poll() is not None:
                log(f"✗ Server process on port {self.port} exited unexpectedly with return code {self.process.returncode}")
                # Print last few lines of stderr if available
                if self.stderr_file:
                    self.stderr_file.flush()
                return False

            log(f"Checking server status on port {self.port} (elapsed: {int(time.time() - start_time)}s)...")
            if self.is_ready():
                elapsed = int(time.time() - start_time)
                log(f"✓ Server on port {self.port} is ready (took {elapsed}s)")
                return True

            time.sleep(check_interval)

        log(f"✗ Server on port {self.port} failed to start within {max_wait}s")
        return False

    def stop(self):
        """Stop the vLLM server"""
        if self.process:
            log(f"Stopping vLLM server on port {self.port}...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log(f"Force killing server on port {self.port}...")
                self.process.kill()
                self.process.wait()

        if self.stdout_file:
            self.stdout_file.close()
        if self.stderr_file:
            self.stderr_file.close()

    def to_config(self) -> Dict[str, str]:
        """Return server configuration for model_config.json"""
        return {
            "ip_addr": self.ip_addr,
            "port": str(self.port)
        }

class VLLMServerManager:
    """Manages multiple vLLM servers"""
    def __init__(self):
        self.servers: List[VLLMServer] = []

    def add_server(self, server: VLLMServer):
        """Add a server to manage"""
        self.servers.append(server)

    def start_all(self, log_dir: str = "logs", stagger_delay: int = 60):
        """Start all servers with staggered delays"""
        log(f"Starting {len(self.servers)} vLLM server(s)...")

        for i, server in enumerate(self.servers):
            server.start(log_dir=log_dir)
            if i < len(self.servers) - 1:  # Don't sleep after the last server
                log(f"Waiting {stagger_delay}s before starting next server...")
                time.sleep(stagger_delay)

        log("All servers started, waiting for them to be ready...")

    def wait_all_ready(self, max_wait: int = 600) -> bool:
        """Wait for all servers to be ready"""
        all_ready = True
        for server in self.servers:
            if not server.wait_until_ready(max_wait=max_wait):
                all_ready = False
        return all_ready

    def stop_all(self):
        """Stop all servers"""
        log("Stopping all vLLM servers...")
        for server in self.servers:
            server.stop()
        log("All servers stopped")

    def generate_model_config(self, agent_model_path: str, output_path: str):
        """Generate model_config.json for tau2-bench"""
        config = {
            agent_model_path: [server.to_config() for server in self.servers],
            'vllm_model_config_path': output_path
        }

        # Write config file
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        log(f"Model configuration written to {output_path}")
        log(f"Config: {json.dumps(config, indent=2)}")

        return config

def get_available_gpus() -> int:
    """Get number of available GPUs"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '-L'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return len(result.stdout.strip().split('\n'))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return 0

def run_evaluation(domain: str, agent_llm: str, user_llm: str,
                   task_path: str, output_file: str, model_config_path: str,
                   max_steps: int = 200, num_trials: int = 1,
                   use_model_tool: bool = False, max_concurrency: int = 10,
                   num_tasks: Optional[int] = None, max_errors: int = 10,
                   seed: int = 300, log_level: str = "ERROR",
                   log_dir: str = "logs",
                   overall_state: Optional[Dict[str, Any]] = None):
    """Run tau2-bench evaluation for a specific domain"""
    log(f"========== Starting evaluation: {domain.upper()} ==========")

    # Use the current interpreter to ensure we're running inside the active conda env (e.g. vllm1)
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
    log(f"Working directory: {os.getcwd()}")
    log(f"Checking task path: {task_path} (exists: {os.path.exists(task_path)})")
    
    # Capture output to print progress
    env = os.environ.copy()
    # Ensure child python prints are flushed promptly (avoids "looks hung" due to buffering)
    env["PYTHONUNBUFFERED"] = "1"

    # Stream structured tau2 logging to a per-domain file (append) so logs survive cancellation.
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
        # Red ANSI line for terminal
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
        
        # Read output in real-time without blocking on readline() when the child is quiet.
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

                    # Append to per-domain eval log (streaming, survives cancellation)
                    try:
                        eval_log.write(f"[tau2/cli.py {stream_name}] {line}")
                        eval_log.flush()
                    except Exception:
                        # Best-effort: never break the main flow due to log I/O.
                        pass

                    # Detect task-complete marker from child to update overall ETA.
                    raw = line.rstrip("\n")
                    if raw.startswith("[TAU2_TASK_COMPLETE]") and overall_state is not None:
                        overall_state["done"] = int(overall_state.get("done") or 0) + 1
                        _print_overall_eta()

                    print(f"[tau2/cli.py {stream_name}] {line.rstrip()}", flush=True)
                else:
                    # EOF on this stream
                    try:
                        selector.unregister(fileobj)
                    except Exception:
                        pass

            # Heartbeat so users can distinguish "no output yet" from "stuck"
            now = time.time()
            if retcode is None and (now - last_heartbeat_ts) > 30 and (now - last_output_ts) > 30:
                last_heartbeat_ts = now
                log(f"{domain.upper()} still running (pid={process.pid}); no output for {int(now - last_output_ts)}s...")

            # Exit when process ended and we've drained both pipes
            if retcode is not None and not selector.get_map():
                break

        if retcode == 0:
            log(f"========== Finished {domain.upper()} evaluation successfully ==========")
        else:
            log(f"========== {domain.upper()} evaluation failed with code {retcode} ==========")

        return retcode

def main():
    parser = argparse.ArgumentParser(
        description="Run tau2-bench evaluation locally without SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic usage (starts 4 agent servers)
  python run_local.py --agent-model /path/to/model

  # Custom number of servers
  python run_local.py --agent-model /path/to/model --num-servers 2

  # Evaluate only specific domains
  python run_local.py --agent-model /path/to/model --domains retail telecom

  # Use existing servers (don't start new ones)
  python run_local.py --agent-model /path/to/model --skip-server-start
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
        help="LLM to use for user simulation (default: gpt-5, requires OPENAI_API_KEY). Use gpt-5 to match tau2-bench baseline."
    )

    # Server configuration
    parser.add_argument(
        "--num-servers",
        type=int,
        default=4,
        help="Number of agent model servers to start (default: 4)"
    )
    parser.add_argument(
        "--start-port",
        type=int,
        default=1900,
        help="Starting port for servers (default: 1900)"
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific GPU IDs to use (default: use 0,1,2,... sequentially)"
    )
    parser.add_argument(
        "--stagger-delay",
        type=int,
        default=60,
        help="Delay in seconds between starting servers (default: 60)"
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=600,
        help="Maximum seconds to wait for server startup (default: 600)"
    )

    # Evaluation configuration
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=["retail", "telecom", "airline"],
        choices=["mock", "retail", "telecom", "airline"],
        help="Domains to evaluate (default: retail telecom airline). Only these domains have task files available."
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
        help="Run only the first N tasks (smoke test). If unset, runs all tasks."
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum number of concurrent tasks (default: 10). For debugging with a single local vLLM server, use 1."
    )
    parser.add_argument(
        "--use_model_tool",
        action="store_true",
        default=True,
        help="Enable model-tool behavior (e.g., call_expert) during agent rollout. Default: enabled to match run.py baseline."
    )
    parser.add_argument(
        "--no-use-model-tool",
        dest="use_model_tool",
        action="store_false",
        help="Disable model-tool behavior (call_expert will not be available)."
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="Maximum number of tool errors allowed in a row (default: 10). Matches tau2-bench baseline."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=300,
        help="Random seed for reproducibility (default: 300). Matches tau2-bench baseline."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "PROFILE", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for tau2 simulation (default: INFO). PROFILE enables tool/LLM call timing."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs)"
    )
    parser.add_argument(
        "--model-config-path",
        type=str,
        default="model_config_local.json",
        help="Path to save model configuration (default: model_config_local.json)"
    )

    # Other options
    parser.add_argument(
        "--skip-server-start",
        action="store_true",
        help="Skip starting vLLM servers (use existing servers)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for server logs (default: logs)"
    )

    args = parser.parse_args()

    # Validate environment
    if not args.agent_model:
        log("ERROR: Agent model path not specified. Set --agent-model or $CKPT_DIR")
        sys.exit(1)

    if not os.path.exists(args.agent_model):
        log(f"ERROR: Agent model path does not exist: {args.agent_model}")
        sys.exit(1)

    # Get repo path
    repo_path = os.getenv("REPO_PATH")
    if not repo_path:
        # Try to infer from current location
        repo_path = str(Path(__file__).parent.parent.parent.absolute())
        log(f"REPO_PATH not set, inferring from script location: {repo_path}")
        os.environ["REPO_PATH"] = repo_path
    else:
        log(f"Using REPO_PATH: {repo_path}")

    # Check GPU availability
    num_gpus = get_available_gpus()
    log(f"Detected {num_gpus} available GPU(s)")

    if num_gpus == 0:
        log("ERROR: No GPUs detected. vLLM requires at least one GPU.")
        sys.exit(1)

    if not args.skip_server_start and num_gpus < args.num_servers:
        log(f"WARNING: Requested {args.num_servers} servers but only {num_gpus} GPUs available")
        log(f"Will only start {num_gpus} server(s)")
        args.num_servers = num_gpus

    # Determine GPU IDs to use
    if args.gpu_ids:
        gpu_ids = args.gpu_ids
        if len(gpu_ids) < args.num_servers:
            log(f"ERROR: Specified {len(gpu_ids)} GPU IDs but requested {args.num_servers} servers")
            sys.exit(1)
    else:
        gpu_ids = list(range(num_gpus))

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Initialize server manager
    manager = VLLMServerManager()

    if not args.skip_server_start:
        # Add agent model servers
        for i in range(args.num_servers):
            server = VLLMServer(
                model_path=args.agent_model,
                port=args.start_port + i,
                gpu_id=gpu_ids[i],
                tool_parser="hermes"
            )
            manager.add_server(server)

        # Signal handler for graceful shutdown
        def signal_handler(sig, frame):
            log("\nReceived interrupt signal. Shutting down...")
            manager.stop_all()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start all servers
        try:
            manager.start_all(log_dir=args.log_dir, stagger_delay=args.stagger_delay)

            # Wait for all servers to be ready
            if not manager.wait_all_ready(max_wait=args.server_timeout):
                log("ERROR: Some servers failed to start")
                log(f"Check logs in {args.log_dir}/ for details")
                manager.stop_all()
                sys.exit(1)

            # Generate model configuration
            manager.generate_model_config(args.agent_model, args.model_config_path)

        except Exception as e:
            log(f"ERROR: Failed to start servers: {e}")
            import traceback
            traceback.print_exc()
            manager.stop_all()
            sys.exit(1)
    else:
        log("Skipping server startup (--skip-server-start)")
        if not os.path.exists(args.model_config_path):
            log(f"ERROR: Model config file not found: {args.model_config_path}")
            sys.exit(1)

    # Run evaluations
    try:
        task_paths = {
            "mock": os.path.join(repo_path, "data/tau2/domains/mock/tasks.json"),
            "retail": os.path.join(repo_path, "data/tau2/domains/retail/tasks.json"),
            "telecom": os.path.join(repo_path, "data/tau2/domains/telecom/tasks.json"),
            "airline": os.path.join(repo_path, "data/tau2/domains/airline/original_tasks.json"),
        }

        # Verify task files exist
        for domain in args.domains:
            if not os.path.exists(task_paths[domain]):
                log(f"ERROR: Task file not found: {task_paths[domain]}")
                manager.stop_all()
                sys.exit(1)

        # Compute overall total tasks across requested domains (for red overall ETA).
        def _count_tasks_in_file(path: str) -> int:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return len(data)
                if isinstance(data, dict) and "tasks" in data and isinstance(data["tasks"], list):
                    return len(data["tasks"])
            except Exception:
                pass
            return 0

        overall_total = 0
        for domain in args.domains:
            n = _count_tasks_in_file(task_paths[domain])
            if args.num_tasks is not None:
                n = min(n, args.num_tasks)
            overall_total += n * args.num_trials

        overall_state: Dict[str, Any] = {
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
                max_errors=args.max_errors,
                seed=args.seed,
                log_level=args.log_level,
                log_dir=args.log_dir,
                overall_state=overall_state,
            )

            results[domain] = "SUCCESS" if returncode == 0 else "FAILED"

        # Print summary
        log("\n" + "="*60)
        log("EVALUATION SUMMARY")
        log("="*60)
        for domain, status in results.items():
            log(f"{domain.upper()}: {status}")
        log("="*60)

    finally:
        # Cleanup
        if not args.skip_server_start:
            manager.stop_all()

if __name__ == "__main__":
    main()
