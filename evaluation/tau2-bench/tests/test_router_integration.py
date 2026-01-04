import os
from unittest.mock import Mock, patch

import pytest
import requests

from tau2.orchestrator.orchestrator import Orchestrator


def _make_orchestrator(domain: str = "mock") -> Orchestrator:
    task = Mock()
    task.id = "task_123"
    task.initial_state = None
    return Orchestrator(
        domain=domain,
        agent=Mock(),
        user=Mock(),
        environment=Mock(),
        task=task,
    )


def test_orchestrator_job_id_format():
    """Verify job_id follows format: tau2:{domain}:{task_id}:{uuid8}"""
    orchestrator = _make_orchestrator(domain="mock")
    assert orchestrator.job_id.startswith("tau2:mock:")
    parts = orchestrator.job_id.split(":")
    assert len(parts) == 4
    assert parts[0] == "tau2"
    assert parts[1] == "mock"
    assert parts[2] == "task_123"
    assert len(parts[3]) == 8  # uuid hex[:8]


def test_set_llm_router_args_sets_fields():
    """Verify _set_llm_router_args sets job_id and is_last_step on llm_args"""

    class MockTarget:
        llm_args = {}

    target = MockTarget()
    Orchestrator._set_llm_router_args(target, "test-job-123", False)
    assert target.llm_args["job_id"] == "test-job-123"
    assert target.llm_args["is_last_step"] is False


def test_set_llm_router_args_no_llm_args():
    """Verify _set_llm_router_args handles targets without llm_args gracefully"""

    class MockTarget:
        pass

    target = MockTarget()
    # Should not raise
    Orchestrator._set_llm_router_args(target, "test-job", True)


@patch("tau2.orchestrator.orchestrator.requests.post")
def test_release_router_job_calls_router(mock_post):
    """Verify _release_router_job makes POST to /programs/release"""
    mock_post.return_value.raise_for_status = Mock()

    with patch.dict(os.environ, {"ROUTER_URL": "http://localhost:8000"}):
        orchestrator = _make_orchestrator(domain="mock")
        orchestrator._release_router_job()

    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert "/programs/release" in call_args[0][0]
    assert call_args[1]["json"]["job_id"] == orchestrator.job_id


@patch("tau2.orchestrator.orchestrator.requests.post")
def test_release_router_job_idempotent(mock_post):
    """Verify _release_router_job only calls once (router_released flag)"""
    mock_post.return_value.raise_for_status = Mock()

    with patch.dict(os.environ, {"ROUTER_URL": "http://localhost:8000"}):
        orchestrator = _make_orchestrator(domain="mock")
        orchestrator._release_router_job()
        orchestrator._release_router_job()  # Second call

    assert mock_post.call_count == 1


def test_release_router_job_no_router_url():
    """Verify _release_router_job does nothing when ROUTER_URL not set"""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("ROUTER_URL", None)
        orchestrator = _make_orchestrator(domain="mock")
        orchestrator._release_router_job()  # Should not raise
        assert not orchestrator.router_released


@pytest.mark.integration
@pytest.mark.router
def test_router_tracks_program_after_chat_completion(router_process: str):
    """Verify router /programs lists job_id after a chat completion request."""
    router_url = router_process
    job_id = "tau2:mock:task_123:deadbeef"

    resp = requests.post(
        f"{router_url}/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
            "job_id": job_id,
            "is_last_step": False,
        },
        timeout=5.0,
    )
    resp.raise_for_status()

    programs = requests.get(f"{router_url}/programs", timeout=5.0).json()
    assert job_id in programs


@pytest.mark.integration
@pytest.mark.router
def test_orchestrator_releases_on_completion(router_process: str):
    """Verify job is released from router after orchestrator._release_router_job() runs."""
    router_url = router_process
    job_id = "tau2:mock:task_123:cafebabe"

    # Create a program in router by issuing a request with this job_id.
    resp = requests.post(
        f"{router_url}/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "hi"}],
            "job_id": job_id,
            "is_last_step": False,
        },
        timeout=5.0,
    )
    resp.raise_for_status()
    programs = requests.get(f"{router_url}/programs", timeout=5.0).json()
    assert job_id in programs

    # Call orchestrator release against the real router.
    with patch.dict(os.environ, {"ROUTER_URL": router_url}):
        orchestrator = _make_orchestrator(domain="mock")
        orchestrator.job_id = job_id
        orchestrator.router_released = False
        orchestrator._release_router_job()

    programs = requests.get(f"{router_url}/programs", timeout=5.0).json()
    assert job_id not in programs


