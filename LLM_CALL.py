# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import openai
from openai import AzureOpenAI
import requests
import time
import os
import json
import requests
import subprocess
from openai import OpenAI
import random
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional
import uuid
import traceback
from datetime import datetime

KEYS_DIR = 'keys'
if not os.path.isdir(KEYS_DIR):
    os.makedirs(KEYS_DIR,exist_ok=True)

def _llm_log_path() -> str:
    # If unset, we won't write any JSONL logs (avoid surprising file writes).
    return os.getenv("TOOL_ORCH_LLM_LOG_PATH", "")

def _llm_log(event: dict) -> None:
    """
    Best-effort JSONL logging. Never raises.
    """
    try:
        path = _llm_log_path()
        if not path:
            return
        event = dict(event)
        event.setdefault("ts", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        # Never break the main flow due to logging
        pass

def convert_openai_tools_to_claude(openai_tools: list) -> list:
    claude_tools = []
    for tool in openai_tools:
        if tool.get("type") != "function":
            raise ValueError(f"Unsupported tool type: {tool.get('type')}")
        
        fn = tool["function"]
        claude_tools.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}})
        })
    return claude_tools

def normalize_messages_for_tools(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Detects and corrects common Chat Completions tool-message issues:
      1) In assistant messages, each entry in `tool_calls` must have:
         {
           "id": "...",
           "type": "function",
           "function": {"name": "<fn_name>", "arguments": "<json string>"}
         }
         - Moves top-level `name` / `arguments` into `function`.
         - Ensures `type == "function"`.
         - JSON-serializes non-string `arguments`.

      2) In tool messages:
         - Ensures `content` is a string; JSON-serializes if dict/list.
         - Ensures `tool_call_id` exists. If missing, tries to pair with the
           most recent unmatched assistant tool_call ID (by order).

      3) Removes illegal extra fields at `tool_calls` top level.

    Returns:
        (fixed_messages, issues)
        - fixed_messages: deep-copied, corrected messages list
        - issues: human-readable list of detected/corrected problems
    """
    fixed = deepcopy(messages)
    issues = []

    # Build a set of valid function names from `tools` (optional validation)
    valid_fn_names = set()
    if tools:
        for t in tools:
            try:
                if t.get("type") == "function":
                    fn = t.get("function", {})
                    name = fn.get("name")
                    if isinstance(name, str):
                        valid_fn_names.add(name)
            except Exception:
                pass

    # Track assistant tool_calls -> to match subsequent tool results
    pending_tool_call_ids = []

    # First pass: fix assistant tool_calls and record pending IDs
    for i, msg in enumerate(fixed):
        role = msg.get("role")
        if role == "assistant" and isinstance(msg.get("tool_calls"), list):
            for j, tc in enumerate(msg["tool_calls"]):
                # Ensure container objects exist
                if not isinstance(tc, dict):
                    issues.append(f"[assistant#{i}] tool_calls[{j}] is not an object; replaced with empty object.")
                    msg["tool_calls"][j] = tc = {}

                # Move name/arguments into function
                fn_obj = tc.get("function") or {}
                moved = False

                if "name" in tc:
                    fn_obj["name"] = tc.pop("name")
                    moved = True
                    issues.append(f"[assistant#{i}] tool_calls[{j}]: moved top-level 'name' into 'function.name'.")

                if "arguments" in tc:
                    fn_obj["arguments"] = tc.pop("arguments")
                    moved = True
                    issues.append(f"[assistant#{i}] tool_calls[{j}]: moved top-level 'arguments' into 'function.arguments'.")

                # Ensure function object present
                if "function" not in tc:
                    tc["function"] = fn_obj if fn_obj else {}
                elif moved:
                    tc["function"].update(fn_obj)

                # Ensure type is "function"
                if tc.get("type") != "function":
                    tc["type"] = "function"
                    issues.append(f"[assistant#{i}] tool_calls[{j}]: set 'type' to 'function'.")

                # Ensure arguments is a string
                if "arguments" in tc["function"]:
                    args_val = tc["function"]["arguments"]
                    if not isinstance(args_val, str):
                        try:
                            tc["function"]["arguments"] = json.dumps(args_val, ensure_ascii=False)
                            issues.append(f"[assistant#{i}] tool_calls[{j}]: JSON-serialized non-string 'function.arguments'.")
                        except Exception:
                            tc["function"]["arguments"] = "{}"
                            issues.append(f"[assistant#{i}] tool_calls[{j}]: failed to serialize arguments; defaulted to '{{}}'.")

                else:
                    # Provide default empty JSON object
                    tc["function"]["arguments"] = "{}"
                    issues.append(f"[assistant#{i}] tool_calls[{j}]: added default empty 'function.arguments'.")

                # Validate function name if possible
                fn_name = tc.get("function", {}).get("name")
                if isinstance(fn_name, str):
                    if valid_fn_names and fn_name not in valid_fn_names:
                        issues.append(f"[assistant#{i}] tool_calls[{j}]: unknown function '{fn_name}' (not in tools).")
                else:
                    issues.append(f"[assistant#{i}] tool_calls[{j}]: missing 'function.name'.")

                # Track pending tool_call_id for pairing
                tc_id = tc.get("id")
                if isinstance(tc_id, str):
                    pending_tool_call_ids.append(tc_id)
                else:
                    # If missing id, synthesize a stable one
                    tc_id = f"call_{i}_{j}"
                    tc["id"] = tc_id
                    pending_tool_call_ids.append(tc_id)
                    issues.append(f"[assistant#{i}] tool_calls[{j}]: synthesized missing 'id' -> '{tc_id}'.")

                # Remove illegal top-level keys except allowed
                allowed = {"id", "type", "function"}
                extraneous = [k for k in list(tc.keys()) if k not in allowed]
                for k in extraneous:
                    tc.pop(k, None)
                    issues.append(f"[assistant#{i}] tool_calls[{j}]: removed unsupported top-level field '{k}'.")

    # Second pass: fix tool messages (pair to pending assistant calls)
    # We'll consume from the front of pending_tool_call_ids in order.
    for i, msg in enumerate(fixed):
        if msg.get("role") == "tool":
            # tool_call_id
            if not msg.get("tool_call_id"):
                if pending_tool_call_ids:
                    inferred = pending_tool_call_ids.pop(0)
                    msg["tool_call_id"] = inferred
                    issues.append(f"[tool#{i}]: added missing 'tool_call_id' -> '{inferred}'.")
                else:
                    issues.append(f"[tool#{i}]: missing 'tool_call_id' and none could be inferred.")

            # content must be string
            content = msg.get("content")
            if not isinstance(content, str):
                try:
                    msg["content"] = json.dumps(content, ensure_ascii=False)
                    issues.append(f"[tool#{i}]: JSON-serialized non-string 'content'.")
                except Exception:
                    msg["content"] = ""
                    issues.append(f"[tool#{i}]: failed to serialize content; set to empty string.")

            # Remove fields illegal for tool role (defensive)
            for bad in ("name", "type", "function"):
                if bad in msg:
                    msg.pop(bad, None)
                    issues.append(f"[tool#{i}]: removed illegal field '{bad}'.")

        # If someone mistakenly returned a tool result as role='assistant' with tool_call_id,
        # quietly convert it to role='tool' (optional but handy).
        if msg.get("role") == "assistant" and "tool_call_id" in msg:
            msg["role"] = "tool"
            issues.append(f"[assistant#{i}]: message had 'tool_call_id'; converted role to 'tool'.")

    return fixed, issues

def convert_openai_messages_to_claude(openai_messages):
    claude_messages = []
    for m in openai_messages:
        if "tool_calls" in m:
            m['content'] += '\n\n'+str(m["tool_calls"])
            m.pop("tool_calls")
            claude_messages.append(m)
        elif m['role']=='tool':
            claude_messages.append({
                "role": 'user',
                "content": "Tool call result: "+m['content']
            })
        else:
            claude_messages.append(m)
    return claude_messages

def get_openai_token(p_token_url, p_client_id, p_client_secret, p_scope, **kwargs):
    try:
        with open(os.path.join(KEYS_DIR,f'openai_key.json')) as f:
            key = json.load(f)
        if time.time()<key['expire_at']:
            return key["access_token"]
    except:
        pass
    
    response = requests.post(
        p_token_url,
        data={"grant_type": "client_credentials", "client_id": p_client_id,
                "client_secret": p_client_secret, "scope": p_scope}
    )
    response.raise_for_status()
    token = response.json()

    with open(os.path.join(KEYS_DIR,f'openai_key.json'),'w') as f:
        json.dump({
            "access_token": token["access_token"],
            'expire_at': time.time()+900
        },f,indent=2)
    os.chmod(str(os.path.join(KEYS_DIR,f'openai_key.json')), 0o777)

    return token["access_token"]

def get_claude_token():
    try:
        with open(os.path.join(KEYS_DIR,'claude_key.json')) as f:
            key = json.load(f)
        if time.time()<key['expire_at']:
            return key["access_token"]
    except:
        pass

    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    command = f"""curl -s --location 'https://5kbfxgaqc3xgz8nhid1x1r8cfestoypn-trofuum-oc.ssa.nvidia.com/token' --header 'Content-Type: application/x-www-form-urlencoded' --header "Authorization: Basic $(echo -n {client_id}:{client_secret} | base64 -w0)" --data-urlencode 'grant_type=client_credentials' --data-urlencode 'scope=awsanthropic-readwrite azureopenai-readwrite' | jq -r '.access_token'"""
    result = subprocess.check_output(command, shell=True, text=True).strip()

    with open(os.path.join(KEYS_DIR,'claude_key.json'),'w') as f:
        json.dump({
            "access_token": result,
            'expire_at': time.time()+900
        },f,indent=2)
    os.chmod(str(os.path.join(KEYS_DIR,'claude_key.json')), 0o777)


    return result


def get_openai_client(model):
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return OpenAI(api_key=api_key)

    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("CLIENT_ID/CLIENT_SECRET are required when OPENAI_API_KEY is not set.")
    token_url = "https://prod.api.nvidia.com/oauth/api/v1/ssa/default/token"
    scope = "azureopenai-readwrite"
    token = get_openai_token(token_url, client_id, client_secret, scope)
    openai.api_type = "azure"
    openai.api_base = "https://prod.api.nvidia.com/llm/v1/azure/"
    openai.api_version = "2025-04-01-preview"
    openai.api_key = token
    client = AzureOpenAI(
        api_key=token,
        api_version="2025-04-01-preview",
        azure_endpoint="https://prod.api.nvidia.com/llm/v1/azure/"
    )
    return client

def get_llm_response(model,messages,temperature=1.0,return_raw_response=False,tools=None,show_messages=False,model_type=None,max_length=1024,model_config=None,model_config_idx=0,model_config_path=None,payload=None,**kwargs):
    if isinstance(messages,str):
        messages = [{'role': 'user','content': messages}]
    # Compatibility aliases: some older configs/tests still reference legacy model names.
    MODEL_ALIASES = {
        "gpt-3.5-turbo": "gpt-4o-mini",
        "gpt-4-turbo": "gpt-4o-mini",
    }
    model = MODEL_ALIASES.get(model, model)
    req_id = str(uuid.uuid4())
    _llm_log({
        "event": "enter",
        "req_id": req_id,
        "model": model,
        "model_type": model_type,
        "max_length": max_length,
        "temperature": temperature,
        "has_tools": bool(tools),
        "num_messages": len(messages) if isinstance(messages, list) else None,
    })
    # Optional: enable streaming for local vLLM calls so we can measure time-to-first-token (prefill)
    # and decode time. Only used by tau2 profiling.
    tau2_stream_profile = bool(kwargs.pop("tau2_stream_profile", False))
    # Together AI (OpenAI-compatible) hosted models.
    #
    # IMPORTANT:
    # - For Together **dedicated endpoints**, the correct `model` is typically the endpoint **Name**
    #   (e.g. "HK123/Qwen2.5-Math-72B-Instruct-1d5fc83c"), not the endpoint ID ("endpoint-...").
    # - We keep "endpoint-..." support as a legacy/back-compat path because some workspaces/scripts
    #   still reference endpoint IDs.
    if isinstance(model, str) and model.startswith("endpoint-"):
        answer = ''
        # Defensive copy; also strip any tool_calls fields that Together may not accept in messages.
        updated_messages = []
        for m in messages:
            if isinstance(m, dict) and 'tool_calls' in m:
                m = dict(m)
                m['content'] = (m.get('content') or '') + str(m.get('tool_calls'))
                m.pop('tool_calls', None)
            updated_messages.append(m)
        while answer == '':
            try:
                oss_client = OpenAI(
                    base_url="https://api.together.xyz/v1",
                    api_key=os.getenv("TOGETHER_API_KEY"),
                )
                if tools:
                    chat_completion = oss_client.chat.completions.create(
                        model=model,
                        messages=updated_messages,
                        temperature=temperature,
                        top_p=0.7,
                        max_tokens=max_length,
                        tools=tools,
                    )
                else:
                    chat_completion = oss_client.chat.completions.create(
                        model=model,
                        messages=updated_messages,
                        temperature=temperature,
                        top_p=0.7,
                        max_tokens=max_length,
                    )
                if return_raw_response:
                    answer = chat_completion
                else:
                    answer = chat_completion.choices[0].message.content
            except Exception as error:
                print(
                    f"Error calling Together (legacy endpoint-id model) req_id={req_id}: {error}. "
                    f"Note: Together often requires the endpoint Name (e.g. 'HK123/...') as `model`.",
                    flush=True,
                )
                time.sleep(60)
        return answer
    if model in ['o3','o3-mini','gpt-4o','o3-high','gpt-5','gpt-5-mini','gpt-4.1','gpt-4o-mini']:
        if max_length==1024:
            max_length = 40000
        if model in ['gpt-4.1','gpt-4o-mini']:
            max_length = 8000
        # Safety clamp: prevent infinite retry due to invalid max_tokens.
        # Observed: gpt-4o rejects >16384 completion tokens.
        OPENAI_MAX_COMPLETION_TOKENS = {
            "gpt-4o": 16384,
            "gpt-4o-mini": 16384,
            "gpt-4.1": 8000,
        }
        # Default to a safe ceiling to avoid invalid_request loops on newer models.
        max_length = min(max_length, OPENAI_MAX_COMPLETION_TOKENS.get(model, 16384))
        openai_client = get_openai_client(model=model)
        answer = ''
        while answer=='':
            try:
                print(f"DEBUG: Calling OpenAI model={model} req_id={req_id}", flush=True)
                _llm_log({
                    "event": "request",
                    "req_id": req_id,
                    "backend": "openai",
                    "model": model,
                    "max_length": max_length,
                })
                t0 = time.time()
                chat_completion = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    tools=tools,
                    max_completion_tokens=max_length
                )
                dur = time.time() - t0
                if return_raw_response:
                    answer = chat_completion
                else:
                    answer = chat_completion.choices[0].message.content
                print(f"DEBUG: OpenAI request successful req_id={req_id}", flush=True)
                _llm_log({
                    "event": "response",
                    "req_id": req_id,
                    "backend": "openai",
                    "model": model,
                    "duration_s": round(dur, 4),
                })
            except Exception as error:
                print(f"Error calling OpenAI req_id={req_id}: {error}", flush=True)
                _llm_log({
                    "event": "error",
                    "req_id": req_id,
                    "backend": "openai",
                    "model": model,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                })
                # If this is a hard invalid-request error (e.g., max_tokens too large),
                # don't retry forever. Surface it to the caller.
                if "max_tokens is too large" in str(error) or "invalid_value" in str(error):
                    raise
                time.sleep(5)
        return answer
    # Together OpenAI-compatible API (used by local HLE eval for OSS expert calls).
    # Prefer using this path with model strings like:
    # - Serverless: "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    # - Dedicated:  "HK123/Qwen2.5-Math-72B-Instruct-1d5fc83c"
    elif model_type in {'nv/dev', 'together'}:
        answer = ''
        updated_messages = []
        for m in messages:
            if 'tool_calls' in m:
                m['content'] += str(m['tool_calls'])
                m.pop('tool_calls')
            updated_messages.append(m)
        while answer=='':
            try:
                oss_client = OpenAI(
                    base_url = "https://api.together.xyz/v1",
                    api_key = os.getenv("TOGETHER_API_KEY")
                )
                if tools:
                    chat_completion = oss_client.chat.completions.create(
                        model=model, 
                        messages=updated_messages,
                        temperature=temperature,
                        top_p=0.7,
                        max_tokens=max_length,
                        tools=tools
                    )
                else:
                    chat_completion = oss_client.chat.completions.create(
                        model=model, 
                        messages=updated_messages,
                        temperature=temperature,
                        top_p=0.7,
                        max_tokens=max_length,
                    )
                if return_raw_response:
                    answer = chat_completion
                else:
                    answer = chat_completion.choices[0].message.content
            except Exception as error:
                time.sleep(60)
        return answer
    elif 'qwen' in model.lower() or model_type=='vllm':
        # Check if we should use Nebius API for Qwen3-32B
        # Only use Nebius if model_config is not provided (i.e., not using local vLLM)
        nebius_api_key = os.getenv("NEBIUS_API_KEY")
        use_nebius = nebius_api_key and 'qwen3-32b' in model.lower() and not model_config

        if use_nebius:
            # Use Nebius API for Qwen3-32B
            answer = ''
            while answer=='':
                try:
                    nebius_client = OpenAI(
                        base_url="https://api.tokenfactory.nebius.com/v1/",
                        api_key=nebius_api_key
                    )
                    # Map model name to Nebius model name
                    nebius_model = "Qwen/Qwen3-32B-fast"

                    chat_completion = nebius_client.chat.completions.create(
                        model=nebius_model,
                        messages=messages,
                        max_tokens=max_length,
                        temperature=temperature,
                        tools=tools
                    )
                    if return_raw_response:
                        answer = chat_completion
                    else:
                        answer = chat_completion.choices[0].message.content
                except Exception as error:
                    print('Error calling Nebius API:',error)
                    time.sleep(60)
            return answer
        else:
            # Use local vLLM server
            answer = ''
            while answer=='':
                config_idx = random.choice(range(len(model_config)))
                ip_addr = model_config[config_idx]["ip_addr"]
                port = model_config[config_idx]["port"]
                try:
                    print(f"DEBUG: Calling vLLM at http://{ip_addr}:{port}/v1/chat/completions (model={model}) req_id={req_id}", flush=True)
                    _llm_log({
                        "event": "request",
                        "req_id": req_id,
                        "backend": "vllm",
                        "model": model,
                        "base_url": f"http://{ip_addr}:{port}/v1",
                        "max_length": max_length,
                        "stream": bool(tau2_stream_profile),
                    })
                    req_start = time.time()
                    vllm_client = OpenAI(
                        api_key="EMPTY",
                        base_url=f"http://{ip_addr}:{port}/v1",
                        timeout=600.0
                    )
                    if tau2_stream_profile:
                        # Streaming mode: measure TTFT (prefill) and decode time.
                        first_token_ts = None
                        content_parts = []
                        tool_calls_by_index = {}
                        final_usage = None

                        def _mark_first_token():
                            nonlocal first_token_ts
                            if first_token_ts is None:
                                first_token_ts = time.time()

                        # Ask for usage in the final chunk if supported by the server/client.
                        extra_body = kwargs.get("extra_body")
                        stream_create_kwargs = {
                            "model": model,
                            "messages": messages,
                            "max_tokens": max_length,
                            "temperature": temperature,
                            "tools": tools,
                            "stream": True,
                        }
                        if extra_body:
                            stream_create_kwargs["extra_body"] = extra_body
                        try:
                            stream = vllm_client.chat.completions.create(
                                **stream_create_kwargs,
                                stream_options={"include_usage": True},
                            )
                        except Exception:
                            stream = vllm_client.chat.completions.create(**stream_create_kwargs)

                        for chunk in stream:
                            try:
                                if getattr(chunk, "usage", None) is not None:
                                    final_usage = chunk.usage
                            except Exception:
                                pass

                            choices = getattr(chunk, "choices", None)
                            if not choices:
                                continue
                            choice0 = choices[0]
                            delta = getattr(choice0, "delta", None)
                            if delta is None:
                                continue

                            delta_content = getattr(delta, "content", None)
                            if delta_content:
                                _mark_first_token()
                                content_parts.append(delta_content)

                            # Some OSS models (e.g., gpt-oss-*) served by vLLM stream tokens under
                            # `delta.reasoning_content` / `delta.reasoning` while leaving `delta.content` empty.
                            # For tau2-bench we treat these as "content" so callers don't see empty strings.
                            delta_reasoning_content = getattr(delta, "reasoning_content", None)
                            delta_reasoning = getattr(delta, "reasoning", None)
                            if delta_reasoning_content:
                                _mark_first_token()
                                content_parts.append(delta_reasoning_content)
                            elif delta_reasoning:
                                _mark_first_token()
                                content_parts.append(delta_reasoning)

                            delta_tool_calls = getattr(delta, "tool_calls", None)
                            if delta_tool_calls:
                                _mark_first_token()
                                for tc in delta_tool_calls:
                                    idx = getattr(tc, "index", None)
                                    if idx is None:
                                        # Fallback: append ordering
                                        idx = len(tool_calls_by_index)
                                    entry = tool_calls_by_index.setdefault(
                                        idx,
                                        {
                                            "id": None,
                                            "type": "function",
                                            "function": {"name": None, "arguments": ""},
                                        },
                                    )
                                    tc_id = getattr(tc, "id", None)
                                    if tc_id:
                                        entry["id"] = tc_id
                                    fn = getattr(tc, "function", None)
                                    if fn is not None:
                                        fn_name = getattr(fn, "name", None)
                                        if fn_name:
                                            entry["function"]["name"] = fn_name
                                        fn_args = getattr(fn, "arguments", None)
                                        if fn_args:
                                            entry["function"]["arguments"] += fn_args

                        end_ts = time.time()
                        infer_ms = (end_ts - req_start) * 1000.0
                        if first_token_ts is None:
                            prefill_ms = infer_ms
                        else:
                            prefill_ms = max(0.0, (first_token_ts - req_start) * 1000.0)
                        decode_ms = max(0.0, infer_ms - prefill_ms)

                        # Prefer usage if present; otherwise keep as 0 to preserve compatibility.
                        prompt_tokens = getattr(final_usage, "prompt_tokens", 0) if final_usage is not None else 0
                        completion_tokens = getattr(final_usage, "completion_tokens", 0) if final_usage is not None else 0

                        class _Fn:
                            def __init__(self, name, arguments):
                                self.name = name
                                self.arguments = arguments

                        class _ToolCall:
                            def __init__(self, id, function):
                                self.id = id
                                self.type = "function"
                                self.function = function

                        class _Msg:
                            def __init__(self, content, tool_calls):
                                self.content = content
                                self.tool_calls = tool_calls

                        class _Choice:
                            def __init__(self, message):
                                self.message = message

                        class _Usage:
                            def __init__(self, prompt_tokens, completion_tokens):
                                self.prompt_tokens = int(prompt_tokens or 0)
                                self.completion_tokens = int(completion_tokens or 0)

                        class _ChatCompletion:
                            def __init__(self, choices, usage):
                                self.choices = choices
                                self.usage = usage

                        tool_calls = []
                        for idx in sorted(tool_calls_by_index.keys()):
                            e = tool_calls_by_index[idx]
                            tc_id = e.get("id") or f"call_{req_id}_{idx}"
                            fn_name = (e.get("function") or {}).get("name") or ""
                            fn_args = (e.get("function") or {}).get("arguments") or ""
                            tool_calls.append(_ToolCall(tc_id, _Fn(fn_name, fn_args)))

                        content = "".join(content_parts)
                        msg = _Msg(content=content, tool_calls=tool_calls or None)
                        chat_completion = _ChatCompletion(choices=[_Choice(msg)], usage=_Usage(prompt_tokens, completion_tokens))

                        # Attach timing/length fields for tau2 profiling
                        chat_completion.tau2_vllm_infer_ms = infer_ms
                        chat_completion.tau2_vllm_prefill_ms = prefill_ms
                        chat_completion.tau2_vllm_decode_ms = decode_ms
                        chat_completion.tau2_vllm_prompt_tokens = int(prompt_tokens or 0)
                        chat_completion.tau2_vllm_completion_tokens = int(completion_tokens or 0)

                        print(
                            f"DEBUG: vLLM stream complete (took {infer_ms/1000.0:.2f}s, ttft {prefill_ms/1000.0:.2f}s) req_id={req_id}",
                            flush=True,
                        )
                        _llm_log({
                            "event": "response",
                            "req_id": req_id,
                            "backend": "vllm",
                            "model": model,
                            "base_url": f"http://{ip_addr}:{port}/v1",
                            "duration_s": round(infer_ms / 1000.0, 4),
                            "prefill_s": round(prefill_ms / 1000.0, 4),
                            "decode_s": round(decode_ms / 1000.0, 4),
                            "prompt_tokens": int(prompt_tokens or 0),
                            "completion_tokens": int(completion_tokens or 0),
                        })

                        if return_raw_response:
                            answer = chat_completion
                        else:
                            # Avoid infinite retry loops if the model responds with tool_calls but empty content.
                            if content:
                                answer = content
                            else:
                                answer = json.dumps(
                                    [
                                        {
                                            "id": tc.id,
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        }
                                        for tc in (tool_calls or [])
                                    ],
                                    ensure_ascii=False,
                                )
                    else:
                        extra_body = kwargs.get("extra_body")
                        create_kwargs = {
                            "model": model,
                            "messages": messages,
                            "max_tokens": max_length,
                            "temperature": temperature,
                            "tools": tools,
                        }
                        if extra_body:
                            create_kwargs["extra_body"] = extra_body
                        chat_completion = vllm_client.chat.completions.create(**create_kwargs)
                        req_dur = time.time() - req_start
                        print(f"DEBUG: vLLM request successful (took {req_dur:.2f}s) req_id={req_id}", flush=True)
                        _llm_log({
                            "event": "response",
                            "req_id": req_id,
                            "backend": "vllm",
                            "model": model,
                            "base_url": f"http://{ip_addr}:{port}/v1",
                            "duration_s": round(req_dur, 4),
                        })
                        
                        if return_raw_response:
                            answer = chat_completion
                        else:
                            answer = chat_completion.choices[0].message.content
                except Exception as error:
                    print(f'Error calling vLLM: {error}', flush=True)
                    traceback.print_exc()
                    _llm_log({
                        "event": "error",
                        "req_id": req_id,
                        "backend": "vllm",
                        "model": model,
                        "base_url": f"http://{ip_addr}:{port}/v1",
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    })
                    if os.path.isfile(str(model_config_path)):
                        # print(f"call {model} error, load {model_config_path}")
                        with open(model_config_path) as f:
                            update_model_configs = json.load(f)
                        model_config = update_model_configs[model]
                    print("Retrying in 5 seconds...", flush=True)
                    time.sleep(5)
            return answer
    elif 'claude' in model.lower():
        access_token = get_claude_token()
        if 'opus' in model:
            endpoint = f"https://prod.api.nvidia.com/llm/v1/aws/model/us.anthropic.claude-opus-4-20250514-v1:0/invoke"
        elif 'sonnet' in model:
            endpoint = f"https://prod.api.nvidia.com/llm/v1/aws/model/us.anthropic.claude-sonnet-4-20250514-v1:0/invoke"
        if not payload:
            updated_messages = []
            system_message = 'You are a good assistant'
            for m in messages:
                if m['role'] == 'system':
                    system_message = m['content']
                else:
                    updated_messages.append(m)
            if not tools:
                payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": updated_messages,
                    "temperature": temperature,
                    "top_p": 1.0,
                    "max_tokens": 4096,
                    'system': system_message,
                }
            else:
                claude_tools = convert_openai_tools_to_claude(tools)
                payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": updated_messages,
                    "temperature": temperature,
                    "top_p": 1.0,
                    "max_tokens": 4096,
                    'system': system_message,
                    'tools': claude_tools
                }

        payload['messages'] = convert_openai_messages_to_claude(payload['messages'])
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        answer = ''
        while answer=='':
            try:
                print(f"DEBUG: Calling Claude model={model}", flush=True)
                response = requests.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                if return_raw_response:
                    answer = response.json()
                else:
                    answer = response.json()['content'][0]['text']
                print(f"DEBUG: Claude request successful", flush=True)
            except Exception as error:
                print(f"Error calling Claude: {error}", flush=True)
                time.sleep(5)
        return answer

