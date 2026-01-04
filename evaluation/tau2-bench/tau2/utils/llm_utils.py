import json
import re
import os
import copy
import time
from typing import Any, Optional
import pickle
import litellm
import uuid
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger
from transformers import AutoTokenizer
from tau2.utils.logging_config import (
    get_tau2_logger,
    log_profile_event,
    log_user_judge_event,
    PROFILE,
    Timer,
)
from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
import random
from tau2.environment.tool import Tool
import sys
REPO_PATH = os.getenv("REPO_PATH")
sys.path.append(REPO_PATH)
from LLM_CALL import get_llm_response

# litellm._turn_on_debug()

TOOL_PRICING = {
    "gpt-5": {
        "input_tokens_per_million": 1.25/10000000,
        "output_tokens_per_million": 10/1000000
    },
    "o3-mini": {
        "input_tokens_per_million": 1.1/10000000,
        "output_tokens_per_million": 4.4/1000000
    },
    "gpt-5-mini": {
        "input_tokens_per_million": 0.25/1000000,
        "output_tokens_per_million": 2/1000000
    },
    "Qwen/Qwen3-32B": {
        "input_tokens_per_million": 0.8/1000000,
        "output_tokens_per_million": 0.8/1000000
    },
    "Qwen/Qwen2.5-Coder-32B-Instruct": {
        "input_tokens_per_million": 0.8/1000000,
        "output_tokens_per_million": 0.8/1000000
    },
    "Qwen/Qwen2.5-Math-72B-Instruct": {
        "input_tokens_per_million": 0.9/1000000,
        "output_tokens_per_million": 0.9/1000000
    },
    "Qwen/Qwen2.5-Math-7B-Instruct": {
        "input_tokens_per_million": 0.2/1000000,
        "output_tokens_per_million": 0.2/1000000
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "input_tokens_per_million": 0.9/1000000,
        "output_tokens_per_million": 0.9/1000000
    },
    "Qwen/Qwen3-8B": {
        "input_tokens_per_million": 0.2/1000000,
        "output_tokens_per_million": 0.2/1000000
    },
    "claude-4.1-opus": {
        "input_tokens_per_million": 15/1000000,
        "output_tokens_per_million": 75/1000000
    },
    "claude-4.1-sonnet": {
        "input_tokens_per_million": 3/1000000,
        "output_tokens_per_million": 15/1000000
    },
    # OSS Expert Models
    "openai/gpt-oss-120b": {
        "input_tokens_per_million": 0.6/1000000,  # Estimated OSS pricing
        "output_tokens_per_million": 0.6/1000000
    },
    "openai/gpt-oss-20b": {
        "input_tokens_per_million": 0.2/1000000,
        "output_tokens_per_million": 0.2/1000000
    },
    "Qwen/Qwen3-Next-80B-A3B-Instruct": {
        "input_tokens_per_million": 0.5/1000000,
        "output_tokens_per_million": 0.5/1000000
    },
    "Qwen/Qwen3-32B-FP8": {
        "input_tokens_per_million": 0.8/1000000,
        "output_tokens_per_million": 0.8/1000000
    },
    "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8": {
        "input_tokens_per_million": 0.5/1000000,
        "output_tokens_per_million": 0.5/1000000
    },
    "code_interpreter_per_second": 0.0000083,
    "tavily": {
        "search": 0.01,
        "extract": 0.002
    },
}

_DEFAULT_PRICING = {
    "input_tokens_per_million": 0.0,
    "output_tokens_per_million": 0.0,
}

MODEL_TYPE = "Qwen/Qwen3-8B"

POLICY_STRINGS = [
    """You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions. To transfer, first make a tool call to transfer_to_human_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user.
""",
    """You should escalate to a human agent if and only if the request cannot be handled within the scope of your actions. To escalate, use the tool call transfer_to_human_agents

You should try your best to resolve the issue before escalating the user to a human agent.
""",
    """You should try your best to resolve the issue for the user before transferring the user to a human agent.""",
    """Make sure you try all the possible ways to resolve the user's issue before transferring to a human agent.
""",
    """Make sure you try all the relevant resolution steps before transferring the user to a human agent.
""",
    """Transfer to human agent
- Transfer to a human agent only if:
  - the user explicitly asks for a human agent, or
  - the request cannot be handled within this policy and available tools (for example, authentication cannot be completed because the user cannot provide an email).
- To transfer: first call transfer_to_human_agents with a concise summary of the user’s issue, then send the message: YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.""",
    """Transfer to human agents
- Transfer only if:
  - the user explicitly asks for a human agent, or
  - the request cannot be handled with the available policy and tools.
- To transfer: first call transfer_to_human_agents with a concise summary; then send the message: YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.
""",
    """You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions, or if the user explicitly asks for a human agent. To transfer, first make a tool call to transfer_to_human_agents with a brief summary, and then send the message: YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.
""",
    """- If the user does not know their booking_id, the agent cannot locate it via tools and should transfer to a human agent.
""",
    """- If the user does not know their booking_id, the agent cannot locate it via tools and should transfer to a human agent.
""",
    """- If the show is completed, do not cancel; transfer to a human agent.
""",
    """## Compensation

- The tools do not support compensation or ex gratia certificates. If the user requests compensation, transfer to a human agent.
""",
    """- If a request cannot be fulfilled with available tools (e.g., locating a booking without booking_id, changing showtime, applying promotions), transfer to a human agent following the transfer procedure.""",
    """You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions. To transfer, first make a tool call to transfer_to_human_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user.
""",
    """## Transfer to Human Agent

Transfer the user to a human agent if:
- The request cannot be handled within the scope of your actions or tools.
- The user explicitly asks for a human agent.

To transfer, first make a tool call to transfer_to_human_agents with a brief summary, and then send: 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.'""",
    """Transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions. To transfer, first make a tool call to transfer_to_human_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user.
""",
    """## Human agent transfer

Transfer only if:
- The user explicitly asks for a human agent, or
- The request cannot be handled within the scope of these tools and policies.

To transfer:
- First, call the tool to transfer_to_human_agents with a concise summary.
- Then send: 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.'""",
    """You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions. If a transfer is needed, inform the user with the message: YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.
""",
    """Out-of-scope examples (transfer to a human agent if requested):
- Changes to student profile data (e.g., editing name, program, or address) beyond what tools support.
- Advisor assignments or financial aid adjustments beyond what tools support.
- Any requests requiring procedures or systems not represented by the available tools.""",
    """You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions. To transfer, first make a tool call to transfer_to_human_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user.
""",
    """- If the requested change is not supported by available tools, transfer to a human agent.
""",
    """- If the policy is unclear or the user’s situation is not covered, transfer to a human agent.
""",
    """- If the user requests actions not supported (e.g., changing package on an existing booking, changing the number of travelers), or if package policies are unclear, transfer to a human agent:
  - First call transfer_to_human_agents with a concise summary.
  - Then send: 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.'""",
  """You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions. To transfer, first make a tool call to transfer_to_human_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user.
""",
    """- If a request is outside the scope of available tools or violates this policy, deny the request or transfer to a human agent if necessary (using the transfer procedure above).""",
    """- If a request cannot be handled with the available tools and policies, transfer to a human agent:
  - First call transfer_to_human_agents with a concise summary.
  - Then send: YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.""",
]

EXPERT_POLICT = """You should transfer the user to an expert if you are not confident about how to reply or which tool to use. To transfer, first make a tool call to call_expert, and then choose an expert based on the task difficulty, i.e. strong expert on tricky task, and weaker expert on simpler task. Think carefully about whether you are confident to meet user expectation, or call an appropriate expert based on both expert cost and performance."""
# EXPERT_POLICT = ""

if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

litellm.drop_params = True

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    # logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()

from datetime import datetime
import string
def generate_random_string(length):
    """Generates a random string of specified length using alphanumeric characters."""
    characters = string.ascii_letters + string.digits + '~!@#$%^&*()-=_+[]'
    return ''.join(random.choice(characters) for _ in range(length))


ALLOW_SONNET_THINKING = False

# if not ALLOW_SONNET_THINKING:
#     logger.warning("Sonnet thinking is disabled")

extra_tool = {
    "type": "function",
    "function": {
      "name": "call_expert",
      "description": "Call the expert such that the user request can be better solved compared to existing functions",
      "parameters": {
        "properties": {
          "expert": {
            "description": "The expert. Choices: ['expert-1', 'expert-2', 'expert-3']. expert-1 exhibits strong functional calling abilities and performs excellent in most domains (math, physics, social science, etc.), but could lack domain knowledge in some cases. expert-2 presents reasonable solutions in most tasks, but could get stuck in complex reasoning and specific domain knowledge. expert-3 demonstrates moderate performance: it can parse complex prompts, performs some mathematical derivations, call functions, yet it sometimes misreads details, mixes concepts. The table below shows the pricing and latency of each model:\nModel | price per million input tokens | price per million output tokens | average latency\nexpert-1 | $1.25 | $10 | 96s\nexpert-2 | $0.25 | $2 | 27s\nexpert-3 | $0.8 | $0.8 | 11s",
            "title": "Expression",
            "type": "string"
          }
        },
        "required": [
          "expert"
        ],
        "title": "parameters",
        "type": "object"
      }
    }
  }
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
def cut_middle_turns(tokenizer,messages,max_length):
    exec_count = 0
    while exec_count<10:
        try:
            exec_count += 1
            messages_str = ''
            start_identifier = generate_random_string(15)
            end_identifier = generate_random_string(15)
            assert not start_identifier in str(messages) and not end_identifier in str(messages) and start_identifier!=end_identifier
            for mid,m in enumerate(messages):
                messages_str += f"{m}{start_identifier}{mid}{end_identifier}"
            token_ids = tokenizer(str(messages_str))['input_ids']
            if len(token_ids)<=max_length:
                return messages
            p1_tokens = tokenizer.batch_decode(token_ids[:max_length//2])
            p1 = ''.join(p1_tokens)
            p1_idx = int(p1.split(start_identifier)[-1].split(end_identifier)[0])
            p2_tokens = tokenizer.batch_decode(token_ids[-max_length//2:])
            p2 = ''.join(p2_tokens)
            p2_idx = int(p2.split(end_identifier)[0].split(start_identifier)[-1])
            return messages[:p1_idx+1]+messages[p2_idx:]
        except Exception as cut_error:
            formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('cut_error_message.json','w') as f:
                json.dump({
                    'messages': messages,
                    'max_length': max_length,
                    'current_time': str(formatted_time)
                },f,indent=2)
    raise ValueError(f'cut_middle_turns error')

def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.error(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.usage
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def to_litellm_messages(messages: list[Message],model,use_model_tool,domain,role,enable_thinking=True) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.

    Args:
        enable_thinking: If False, disable thinking mode for Qwen3 models by adding /no_think.
                        This is used for expert models to get direct responses without CoT.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            content = message.content
            # For Qwen3 models with thinking disabled, append /no_think to user messages
            # Per Qwen3 documentation: https://huggingface.co/Qwen/Qwen3-32B
            if not enable_thinking and 'qwen3' in model.lower():
                content = content + " /no_think"
            litellm_messages.append({"role": "user", "content": content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            if 'qwen' in model.lower() or 'train' in model.lower() or 'huggingface' in model.lower():
                cur_content =  message.content
                if use_model_tool:
                    for s in POLICY_STRINGS:
                        cur_content = cur_content.replace(s,EXPERT_POLICT)
                # Only add thinking instructions if enable_thinking is True
                if enable_thinking:
                    litellm_messages.append({"role": "system", "content": cur_content+'  You are dedicated to provide the best service. Wrap thinking process between <think> </think>, message between <message> </message> and the tool call between <tool_call> </tool_call> .'})
                else:
                    # For expert Qwen3 models, keep output format but no thinking
                    litellm_messages.append({"role": "system", "content": cur_content+'  You are dedicated to provide the best service. Wrap message between <message> </message> and the tool call between <tool_call> </tool_call> .'})
            else:
                litellm_messages.append({"role": "system", "content": message.content})
    if role=='assistant' and 'gpt-5' in model.lower():
        litellm_messages[-1]['content'] += '\n\nWait, the information I provide may not be correct, please ask again.'
    return litellm_messages


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    debug = False,
    role=None,
    cur_transfer_dir=None,
    use_model_tool=False,
    model_config_path=None,
    domain=None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    # Backwards compatibility: some call sites historically didn't pass role.
    if role is None:
        role = "assistant"
    if role != "user" and role != "assistant" and role != "evaluator":
        raise ValueError(f"unknown role {role}")

    # Router / scheduler fields (ThunderReact / Continuum)
    job_id = kwargs.pop("job_id", None)
    is_last_step = kwargs.pop("is_last_step", False)
    last_func_call = kwargs.pop("last_func_call", None)
    extra_body: dict[str, Any] = {}
    if job_id:
        extra_body["job_id"] = job_id
        extra_body["is_last_step"] = bool(is_last_step)
    if last_func_call is not None:
        extra_body["last_func_call"] = last_func_call
    
    tau2_logger = get_tau2_logger()
    llm_timer = Timer()
    llm_timer.__enter__()
    # Main ToolOrchestra (local vLLM 8B) inference wall time, recorded separately from total duration.
    toolorchestra_vllm_infer_ms: Optional[float] = None
    toolorchestra_vllm_prefill_ms: Optional[float] = None
    toolorchestra_vllm_decode_ms: Optional[float] = None
    toolorchestra_vllm_prefill_len: Optional[int] = None
    toolorchestra_vllm_decode_len: Optional[int] = None
    
    tau2_logger.debug(f"LLM generate: role={role} model={model} num_messages={len(messages)} use_model_tool={use_model_tool}")
    
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}
    if role == "assistant" and not domain:
        domain = "unknown"
    litellm_messages = to_litellm_messages(messages,model=model,use_model_tool=use_model_tool,domain=domain,role=role)
    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = "auto"
    original_tools = copy.deepcopy(tools)
    start_time = time.time()
    cost = 0

    # Streaming profiling toggles:
    #
    # - We used streaming mainly to estimate TTFT/prefill and decode time for vLLM models.
    # - However, streaming can introduce compatibility issues across OSS backends (e.g. fields
    #   emitted under reasoning_content vs content), and increases complexity.
    #
    # Policy:
    # - Expert LLM calls: streaming is OFF by default.
    # - Orchestrator-8B calls: streaming is OFF by default, but can be enabled via env.
    #
    # Enable by setting:
    #   - TAU2_TOOLORCH_STREAM_PROFILE=1   (orchestrator only)
    #   - TAU2_EXPERT_STREAM_PROFILE=1     (experts; default 0)
    _TOOLORCH_STREAM_PROFILE = os.getenv("TAU2_TOOLORCH_STREAM_PROFILE", "0") == "1"
    _EXPERT_STREAM_PROFILE = os.getenv("TAU2_EXPERT_STREAM_PROFILE", "0") == "1"

    def _extract_message_text(msg: Any) -> str:
        """
        Best-effort extraction of assistant text from OpenAI-compatible responses.
        Some OSS models served by vLLM may populate `reasoning_content`/`reasoning`
        while leaving `content` as None.
        """
        val = getattr(msg, "content", None)
        if val is None or val == "" or val == []:
            val = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        # Handle potential list-of-parts content formats.
        if isinstance(val, list):
            parts: list[str] = []
            for p in val:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    parts.append(p["text"])
                else:
                    parts.append(str(p))
            val = "".join(parts)
        if val is None:
            return ""
        return str(val)
    
    # Determine call type for logging
    call_type = "unknown"
    if 'qwen' in model.lower() or 'huggingface' in model.lower() or 'llama' in model.lower() or 'nemotron' in model.lower() or 'orchestrator' in model.lower():
        call_type = "vllm"
    elif 'claude' in model.lower():
        call_type = "claude"
    elif model in ['o3','o3-mini','gpt-4o','o3-high','gpt-5','gpt-5-mini','gpt-4.1','gpt-4o-mini']:
        call_type = "openai"
    else:
        call_type = "openai"  # default to openai for other models
    if role=='assistant' and ('qwen' in model.lower() or 'huggingface' in model.lower() or 'llama' in model.lower() or 'nemotron' in model.lower() or 'orchestrator' in model.lower()):
        # Local model path: always route through local vLLM using model_config_path.
        # Previously, 'nemotron' models were hard-coded to 'nv/dev' (Together API),
        # which makes local runs hang/idly wait and never hit vLLM.
        with open(model_config_path) as f:
            model_config = json.load(f)[model]
        config_idx = random.randint(0, len(model_config)-1)
        if use_model_tool:
            updated_tools = []
            for t in tools:
                if t['function']['name']!='transfer_to_human_agents':
                    updated_tools.append(t)
            updated_tools += [extra_tool]
        else:
            updated_tools = tools
        tools_length = len(tokenizer(str(updated_tools))['input_ids'])
        updated_messages = cut_middle_turns(tokenizer=tokenizer,messages=litellm_messages,max_length=23000-tools_length)
        # Time only the main ToolOrchestra vLLM model inference (Nemotron-Orchestrator-8B).
        # If log-level is PROFILE/DEBUG, enable streaming profiling to capture TTFT (prefill) + decode time.
        is_toolorchestra_main_vllm = ("orchestrator" in model.lower() and "8b" in model.lower())
        # Default OFF; enable explicitly via TAU2_TOOLORCH_STREAM_PROFILE=1.
        stream_profile = bool(is_toolorchestra_main_vllm and _TOOLORCH_STREAM_PROFILE and tau2_logger.isEnabledFor(PROFILE))
        if is_toolorchestra_main_vllm:
            _infer_timer = Timer()
            _infer_timer.__enter__()
            try:
                response = get_llm_response(
                    model=model,
                    messages=updated_messages,
                    tools=updated_tools,
                    return_raw_response=True,
                    temperature=1,
                    model_config=model_config,
                    model_config_path=model_config_path,
                    model_config_idx=config_idx,
                    model_type='vllm',
                    max_length=8000,
                    tau2_stream_profile=stream_profile,
                    extra_body=extra_body if extra_body else None,
                )
            finally:
                _infer_timer.__exit__(None, None, None)
            # Default: python-side wall time; if streaming profiling is enabled, prefer request-derived timings.
            toolorchestra_vllm_infer_ms = _infer_timer.duration_ms
            if not isinstance(response, str):
                toolorchestra_vllm_infer_ms = getattr(response, "tau2_vllm_infer_ms", toolorchestra_vllm_infer_ms)
                toolorchestra_vllm_prefill_ms = getattr(response, "tau2_vllm_prefill_ms", None)
                toolorchestra_vllm_decode_ms = getattr(response, "tau2_vllm_decode_ms", None)
                # Lengths: prompt/completion tokens
                toolorchestra_vllm_prefill_len = getattr(response, "tau2_vllm_prompt_tokens", None)
                toolorchestra_vllm_decode_len = getattr(response, "tau2_vllm_completion_tokens", None)
        else:
            response = get_llm_response(
                model=model,
                messages=updated_messages,
                tools=updated_tools,
                return_raw_response=True,
                temperature=1,
                model_config=model_config,
                model_config_path=model_config_path,
                model_config_idx=config_idx,
                model_type='vllm',
                max_length=8000,
                extra_body=extra_body if extra_body else None,
            )
        mode_to_call = None
        tool_calls = []
        input_tokens = 0
        output_tokens = 0
        expert_input_tokens = 0
        expert_output_tokens = 0
        raw_response = None
        raw_tool_calls = None
        if isinstance(response,str):
            response_content = "Wait a minute, I will take it very soon"
        else:
            msg = response.choices[0].message
            raw_response = _extract_message_text(msg)
            if raw_response:
                response_content = raw_response.split('</think>')[-1].strip()
            else:
                response_content = ''
                raw_response = ''
            if response.choices[0].message.tool_calls:
                raw_tool_calls = [{'name': one_tool_call.function.name,'arguments': json.loads(one_tool_call.function.arguments)} for one_tool_call in response.choices[0].message.tool_calls]
            else:
                raw_tool_calls = []
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            pricing = TOOL_PRICING.get(MODEL_TYPE, _DEFAULT_PRICING)
            cost += (
                response.usage.prompt_tokens * pricing["input_tokens_per_million"]
                + response.usage.completion_tokens * pricing["output_tokens_per_million"]
            )
        if response_content is not None and isinstance(response_content,str) and len(response_content)>5 and '<message>' in response_content and '</message>' in response_content:
            response_content = response_content.split('<message>')[-1].split('</message>')[0]
        if not isinstance(response,str) and response.choices[0].message.tool_calls:
            for one_tool_call in response.choices[0].message.tool_calls:
                one_tool_call_arguments = json.loads(one_tool_call.function.arguments)
                if one_tool_call.function.name=='call_expert':
                    if 'expert' in one_tool_call_arguments:
                        # Check if OSS expert mapping is configured in model_config
                        oss_mapping = None
                        if model_config_path:
                            try:
                                with open(model_config_path) as f:
                                    cfg = json.load(f)
                                oss_mapping = cfg.get('oss_expert_mapping', None)
                            except Exception:
                                pass

                        expert_choice = one_tool_call_arguments['expert']
                        if oss_mapping and expert_choice in oss_mapping:
                            # Use OSS model mapping
                            mode_to_call = oss_mapping[expert_choice]
                        else:
                            # Default proprietary model mapping
                            if expert_choice == 'expert-1':
                                mode_to_call = 'gpt-5'
                            elif expert_choice == 'expert-2':
                                mode_to_call = 'gpt-5-mini'
                            elif expert_choice == 'expert-3':
                                mode_to_call = 'Qwen/Qwen3-32B'
                tool_calls.append({
                        'name': one_tool_call.function.name,
                        'arguments': one_tool_call_arguments
                    })
        expert_model = mode_to_call
        if mode_to_call:
            # Time the expert call separately
            expert_timer = Timer()
            expert_timer.__enter__()

            # Expert profiling variables
            expert_prefill_ms: Optional[float] = None
            expert_decode_ms: Optional[float] = None
            expert_prefill_len: Optional[int] = None
            expert_decode_len: Optional[int] = None

            # For Qwen3 expert models, disable thinking mode to get direct responses
            # Per Qwen3 documentation: use /no_think to disable CoT
            expert_enable_thinking = not ('qwen3' in mode_to_call.lower())
            llm_messages = to_litellm_messages(messages,model=mode_to_call,use_model_tool=False,domain=domain,role=role,enable_thinking=expert_enable_thinking)

            # Determine expert call type based on model
            is_openai_api = 'gpt-5' in mode_to_call and 'gpt-oss' not in mode_to_call.lower()
            is_vllm_oss = 'gpt-oss' in mode_to_call.lower() or 'qwen3' in mode_to_call.lower()
            expert_call_type = "openai" if is_openai_api else "vllm"

            if is_openai_api:
                # Proprietary OpenAI models (gpt-5, gpt-5-mini)
                response = get_llm_response(model=mode_to_call,messages=llm_messages,tools=original_tools,return_raw_response=True,max_length=40000,extra_body=extra_body if extra_body else None)
            elif is_vllm_oss:
                # OSS models via local vLLM (gpt-oss-*, qwen3-*, qwen3-next-*)
                # Default OFF; enable explicitly via TAU2_EXPERT_STREAM_PROFILE=1.
                # (Most evaluations don't require TTFT breakdown for experts, and streaming can be flaky.)
                expert_stream_profile = bool(_EXPERT_STREAM_PROFILE and tau2_logger.isEnabledFor(PROFILE))
                with open(model_config_path) as f:
                    model_config = json.load(f)[mode_to_call]
                tools_length = len(tokenizer(str(original_tools))['input_ids'])
                cut_messages = cut_middle_turns(tokenizer=tokenizer,messages=litellm_messages,max_length=23000-tools_length)
                response = get_llm_response(
                    model=mode_to_call,
                    messages=cut_messages,
                    tools=original_tools,
                    return_raw_response=True,
                    model_config=model_config,
                    model_config_path=model_config_path,
                    model_config_idx=config_idx,
                    model_type='vllm',
                    max_length=8000,
                    tau2_stream_profile=expert_stream_profile,
                    extra_body=extra_body if extra_body else None,
                )
                # Extract streaming profiling data if available
                if not isinstance(response, str):
                    expert_prefill_ms = getattr(response, "tau2_vllm_prefill_ms", None)
                    expert_decode_ms = getattr(response, "tau2_vllm_decode_ms", None)
                    expert_prefill_len = getattr(response, "tau2_vllm_prompt_tokens", None)
                    expert_decode_len = getattr(response, "tau2_vllm_completion_tokens", None)
            else:
                raise ValueError(f'Model {mode_to_call} is not supported')

            expert_timer.__exit__(None, None, None)

            # Log expert call with prefill/decode breakdown for vLLM models
            log_profile_event(
                "expert_call",
                model=mode_to_call,
                call_type=expert_call_type,
                duration_ms=expert_timer.duration_ms,
                prefill_ms=expert_prefill_ms,
                decode_ms=expert_decode_ms,
                input_tokens=expert_prefill_len,
                output_tokens=expert_decode_len,
            )
            
            if isinstance(response,str):
                response_content = "Wait a minute, I will take it very soon"
            else:
                response_content = _extract_message_text(response.choices[0].message)
                expert_input_tokens = response.usage.prompt_tokens
                expert_output_tokens = response.usage.completion_tokens
                pricing = TOOL_PRICING.get(mode_to_call, _DEFAULT_PRICING)
                cost += (
                    response.usage.prompt_tokens * pricing["input_tokens_per_million"]
                    + response.usage.completion_tokens * pricing["output_tokens_per_million"]
                )
            tool_calls = []
            if not isinstance(response,str) and response.choices[0].message.tool_calls:
                for one_tool_call in response.choices[0].message.tool_calls:
                    tool_calls.append({
                        'name': one_tool_call.function.name,
                        'arguments': json.loads(one_tool_call.function.arguments)
                    })
        if not response_content and not tool_calls:
            response_content = "Wait a minute, I will take it very soon"
        response = {
            'content': response_content,
            'tool_calls': tool_calls,
        }
    elif 'claude' in model.lower():
        response = get_llm_response(model=model,messages=litellm_messages,tools=tools,return_raw_response=True,max_length=40000,extra_body=extra_body if extra_body else None)
        response = get_llm_response(model=model,messages=litellm_messages,tools=tools,return_raw_response=True,max_length=40000,extra_body=extra_body if extra_body else None)
        tool_calls = []
        response_content = ""
        input_tokens = 0
        output_tokens = 0
        if not isinstance(response,str):
            for return_result in response['content']:
                if return_result['type']=='text':
                    response_content = return_result['text']
                elif return_result['type']=='tool_use':
                    tool_calls.append({
                        'name': return_result['name'],
                        'arguments': return_result['input']
                    })
                else:
                    raise ValueError(f"Unknown type: {return_result['type']}")
            input_tokens = response['usage']['input_tokens']
            output_tokens = response['usage']['output_tokens']
            pricing = TOOL_PRICING.get(model, _DEFAULT_PRICING)
            cost += (
                input_tokens * pricing["input_tokens_per_million"]
                + output_tokens * pricing["output_tokens_per_million"]
            )
        if not response_content and not tool_calls:
            response_content = "Wait a minute, I will take it very soon"
        response = {
            'content': response_content,
            'tool_calls': tool_calls,
        }
    else:
        try:
            if os.getenv("TAU2_TRACE", "0") == "1":
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] llm_utils.generate: calling get_llm_response role={role} model={model}",
                    flush=True,
                )
        except Exception:
            pass
        response = get_llm_response(
            model=model,
            messages=litellm_messages,
            tools=tools,
            return_raw_response=True,
            max_length=40000,
            extra_body=extra_body if extra_body else None,
        )
        tool_calls = []
        if not isinstance(response,str) and response.choices[0].message.tool_calls:
            for one_tool_call in response.choices[0].message.tool_calls:
                tool_calls.append({
                    'name': one_tool_call.function.name,
                    'arguments': json.loads(one_tool_call.function.arguments)
                })
        input_tokens = 0
        output_tokens = 0
        if isinstance(response,str) or not response:
            response_content = "Wait a minute, I will take it very soon"
        else:
            response_content = _extract_message_text(response.choices[0].message)
            if role=='assistant' and isinstance(response_content,str):
                response_content = response_content.split('</think>')[-1]
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            if role == "assistant":
                pricing = TOOL_PRICING.get(model, _DEFAULT_PRICING)
                cost += (
                    response.usage.prompt_tokens * pricing["input_tokens_per_million"]
                    + response.usage.completion_tokens * pricing["output_tokens_per_million"]
                )
        if not response_content and not tool_calls:
            response_content = "Wait a minute, I will take it very soon"
        response = {
            'content': response_content,
            'tool_calls': tool_calls,
        }
    cost = 0
    usage = {
        "completion_tokens": 0,
        "prompt_tokens": 0,
    }
    content = response['content']
    tool_calls = []
    if response['tool_calls']:
        assert isinstance(response['tool_calls'],list)
        for one_tool_call in response['tool_calls']:
            my_uuid = uuid.uuid4()
            uuid_string1 = str(my_uuid)
            tool_calls.append(ToolCall(
                    id=f"{uuid_string1}",
                    name=one_tool_call['name'],
                    arguments=one_tool_call['arguments'],
                ))
            if len(tool_calls)>5:
                break
    tool_calls = tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response,
    )
    
    # Log LLM call completion with timing
    llm_timer.__exit__(None, None, None)
    
    # Log based on role
    if role == 'assistant':
        log_profile_event(
            "llm_call",
            model=model,
            call_type=call_type,
            duration_ms=llm_timer.duration_ms,
            vllm_infer_ms=toolorchestra_vllm_infer_ms,
            prefill_ms=toolorchestra_vllm_prefill_ms,
            decode_ms=toolorchestra_vllm_decode_ms,
            input_tokens=toolorchestra_vllm_prefill_len,
            output_tokens=toolorchestra_vllm_decode_len,
            has_tool_calls=bool(tool_calls),
        )
    elif role in ('user', 'evaluator'):
        event_type = "user_sim" if role == 'user' else "evaluator"
        log_user_judge_event(
            event_type,
            model=model,
            call_type=call_type,
            duration_ms=llm_timer.duration_ms,
        )
    
    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage
