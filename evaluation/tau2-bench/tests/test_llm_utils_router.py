import pytest
from unittest.mock import patch

from tau2.data_model.message import Message, SystemMessage, UserMessage
from tau2.utils.llm_utils import generate


def _messages() -> list[Message]:
    return [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(role="user", content="Hello!"),
    ]


def test_generate_builds_extra_body_with_job_id():
    """Verify generate() passes extra_body containing job_id/is_last_step to get_llm_response."""
    with patch("tau2.utils.llm_utils.get_llm_response") as mock_llm:
        mock_llm.side_effect = RuntimeError("sentinel")

        with pytest.raises(RuntimeError, match="sentinel"):
            generate(
                model="gpt-4o-mini",
                messages=_messages(),
                role="assistant",
                domain="mock",
                job_id="test-job-123",
                is_last_step=False,
            )

        call_kwargs = mock_llm.call_args.kwargs
        extra_body = call_kwargs.get("extra_body")
        assert extra_body is not None
        assert extra_body["job_id"] == "test-job-123"
        assert extra_body["is_last_step"] is False


def test_generate_extra_body_includes_last_func_call():
    """Verify generate() includes last_func_call in extra_body when provided."""
    with patch("tau2.utils.llm_utils.get_llm_response") as mock_llm:
        mock_llm.side_effect = RuntimeError("sentinel")

        with pytest.raises(RuntimeError, match="sentinel"):
            generate(
                model="gpt-4o-mini",
                messages=_messages(),
                role="assistant",
                domain="mock",
                job_id="test-job-123",
                last_func_call="get_order_details",
            )

        call_kwargs = mock_llm.call_args.kwargs
        extra_body = call_kwargs.get("extra_body")
        assert extra_body is not None
        assert extra_body["job_id"] == "test-job-123"
        assert extra_body["last_func_call"] == "get_order_details"


def test_generate_no_extra_body_without_job_id():
    """Verify generate() passes None for extra_body when no router/scheduler fields are provided."""
    with patch("tau2.utils.llm_utils.get_llm_response") as mock_llm:
        mock_llm.side_effect = RuntimeError("sentinel")

        with pytest.raises(RuntimeError, match="sentinel"):
            generate(
                model="gpt-4o-mini",
                messages=_messages(),
                role="assistant",
                domain="mock",
            )

        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs.get("extra_body") is None


