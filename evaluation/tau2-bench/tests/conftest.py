from typing import Callable

import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.registry import registry
from tau2.run import get_tasks


@pytest.fixture
def domain_name():
    return "mock"


@pytest.fixture
def get_environment() -> Callable[[], Environment]:
    return registry.get_env_constructor("mock")


@pytest.fixture
def base_task() -> Task:
    return get_tasks("mock", task_ids=["create_task_1"])[0]


@pytest.fixture
def task_with_env_assertions() -> Task:
    return get_tasks("mock", task_ids=["create_task_1_with_env_assertions"])[0]


@pytest.fixture
def task_with_message_history() -> Task:
    return get_tasks("mock", task_ids=["update_task_with_message_history"])[0]


@pytest.fixture
def task_with_initialization_data() -> Task:
    return get_tasks("mock", task_ids=["update_task_with_initialization_data"])[0]


@pytest.fixture
def task_with_initialization_actions() -> Task:
    return get_tasks("mock", task_ids=["update_task_with_initialization_actions"])[0]


@pytest.fixture
def task_with_history_and_env_assertions() -> Task:
    return get_tasks("mock", task_ids=["update_task_with_history_and_env_assertions"])[
        0
    ]


@pytest.fixture
def task_with_action_checks() -> Task:
    return get_tasks("mock", task_ids=["impossible_task_1"])[0]


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def mock_router_url(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set ROUTER_URL environment variable for tests."""
    url = "http://localhost:8000"
    monkeypatch.setenv("ROUTER_URL", url)
    return url


@pytest.fixture
def agent_with_llm_args(get_environment: Callable[[], Environment]):
    """Agent fixture with llm_args dict for router testing."""
    # Import inside fixture so unit-only test collection doesn't fail if optional deps change.
    from tau2.agent.llm_agent import LLMAgent

    environment = get_environment()
    return LLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm="gpt-3.5-turbo",
        llm_args={"temperature": 0.0},
        domain="mock",
    )


@pytest.fixture
def router_process() -> str:
    """Start a local ThunderReact router + a dummy vLLM backend for integration tests."""

    class _DummyBackendHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            # Silence http.server logging in tests
            return

        def do_GET(self):  # noqa: N802
            if self.path == "/metrics":
                body = "\n".join(
                    [
                        "# HELP vllm:kv_cache_usage_perc Dummy metric",
                        "# TYPE vllm:kv_cache_usage_perc gauge",
                        "vllm:kv_cache_usage_perc 0.1",
                        "vllm:num_requests_running 0",
                        "vllm:num_requests_waiting 0",
                        "",
                    ]
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):  # noqa: N802
            if self.path == "/v1/chat/completions":
                # Minimal OpenAI-compatible response body
                body = {
                    "id": "dummy",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "dummy",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                payload = __import__("json").dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(404)
            self.end_headers()

    backend_port = _get_free_port()
    router_port = _get_free_port()

    backend = HTTPServer(("127.0.0.1", backend_port), _DummyBackendHandler)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()

    router_url = f"http://127.0.0.1:{router_port}"

    repo_root = Path(__file__).resolve().parents[3]  # oss-ToolOrchestra/
    router_path = repo_root / "multinode_router.py"

    env = os.environ.copy()
    env["VLLM_BACKENDS"] = f"http://127.0.0.1:{backend_port}"
    env["ROUTER_HOST"] = "127.0.0.1"
    env["ROUTER_PORT"] = str(router_port)
    env.setdefault("ROUTER_LOG_LEVEL", "warning")

    proc = subprocess.Popen(
        [sys.executable, str(router_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Wait for router to become healthy
        import urllib.request

        deadline = time.time() + 15.0
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{router_url}/health", timeout=1.0) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                time.sleep(0.2)
        else:
            stdout = proc.stdout.read() if proc.stdout else ""
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(
                f"Router did not become healthy at {router_url}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )

        yield router_url
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        backend.shutdown()
        backend.server_close()
