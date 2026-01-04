import os
from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest

from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.simulation import TerminationReason
from tau2.orchestrator.orchestrator import Orchestrator, Role


@dataclass
class _DummyState:
    messages: list


class _DummyEnv:
    def set_state(self, **_kwargs):
        return None

    def get_info(self):
        return Mock()

    def sync_tools(self):
        return None


class _DummyUser:
    def __init__(self):
        self.llm_args: dict = {}

    def get_init_state(self, message_history=None):
        return _DummyState(messages=list(message_history or []))

    def set_seed(self, _seed: int):
        return None

    def generate_next_message(self, _message, state: _DummyState):
        msg = UserMessage(role="user", content="hi")
        return msg, state


class _DummyAgent:
    def __init__(self, *, stop_on_first: bool = False):
        self.llm_args: dict = {}
        self._stop_on_first = stop_on_first

    def get_init_state(self, message_history=None):
        return _DummyState(messages=list(message_history or []))

    def set_seed(self, _seed: int):
        return None

    def is_stop(self, _message: AssistantMessage) -> bool:
        return bool(self._stop_on_first)

    def generate_next_message(self, _message, state: _DummyState):
        msg = AssistantMessage(role="assistant", content="ok")
        return msg, state


def test_orchestrator_step_sets_router_args_before_agent_call():
    """Verify _set_llm_router_args is applied before user/agent LLM calls."""
    env = _DummyEnv()
    user = _DummyUser()
    agent = _DummyAgent(stop_on_first=False)

    task = Mock()
    task.id = "task_123"
    task.initial_state = None

    orchestrator = Orchestrator(
        domain="mock",
        agent=agent,
        user=user,
        environment=env,
        task=task,
        max_steps=5,
    )
    orchestrator.initialize()

    seen = {}

    def _user_generate_next_message(message, state):
        seen["user_job_id"] = user.llm_args.get("job_id")
        seen["user_is_last_step"] = user.llm_args.get("is_last_step")
        return UserMessage(role="user", content="hi"), state

    def _agent_generate_next_message(message, state):
        seen["agent_job_id"] = agent.llm_args.get("job_id")
        seen["agent_is_last_step"] = agent.llm_args.get("is_last_step")
        return AssistantMessage(role="assistant", content="ok"), state

    user.generate_next_message = _user_generate_next_message
    agent.generate_next_message = _agent_generate_next_message

    # Step 0: AGENT -> USER (user.generate_next_message)
    orchestrator.step()
    assert orchestrator.from_role == Role.USER
    assert orchestrator.to_role == Role.AGENT
    assert seen["user_job_id"] == orchestrator.job_id
    assert seen["user_is_last_step"] is False

    # Step 1: USER -> AGENT (agent.generate_next_message)
    orchestrator.step()
    assert orchestrator.from_role == Role.AGENT
    assert orchestrator.to_role in [Role.USER, Role.ENV]
    assert seen["agent_job_id"] == orchestrator.job_id
    assert seen["agent_is_last_step"] is False


@pytest.mark.parametrize(
    "stop_on_first,max_steps,expected_reason",
    [
        (True, 10, TerminationReason.AGENT_STOP),
        (False, 1, TerminationReason.MAX_STEPS),
    ],
)
@patch("tau2.orchestrator.orchestrator.requests.post")
def test_orchestrator_run_releases_on_termination_reasons(
    mock_post,
    stop_on_first: bool,
    max_steps: int,
    expected_reason: TerminationReason,
):
    """Verify router release is called regardless of termination reason."""
    mock_post.return_value.raise_for_status = Mock()

    env = _DummyEnv()
    user = _DummyUser()
    agent = _DummyAgent(stop_on_first=stop_on_first)

    task = Mock()
    task.id = "task_123"
    task.initial_state = None

    with patch.dict(os.environ, {"ROUTER_URL": "http://localhost:8000"}):
        orchestrator = Orchestrator(
            domain="mock",
            agent=agent,
            user=user,
            environment=env,
            task=task,
            max_steps=max_steps,
        )
        simulation_run = orchestrator.run()

    assert simulation_run is not None
    assert orchestrator.termination_reason == expected_reason
    assert mock_post.called


