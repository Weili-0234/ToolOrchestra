# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test script to verify Together AI API connectivity with all open-source models
# used in ToolOrchestra codebase.

import os
from openai import OpenAI

# Setup Together AI client (same interface as in codebase)
client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.getenv("TOGETHER_API_KEY")
)

# All open-source models used in the codebase
MODELS_TO_TEST = [
    # Used as fallback in eval_hle.py, eval_frames.py, eval_hle_basic.py, generation_quick3.py
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",

    # Used in MODEL_MAPPING (vLLM primarily, but good to test availability)
    "Qwen/Qwen3-32B",
    "Qwen/Qwen2.5-Math-72B-Instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct",

    # Used in tau2-bench as expert-3
    # "Qwen/Qwen3-32B",  # Already included above
]

TEST_MESSAGE = [{"role": "user", "content": "Say 'Hello, Together AI test successful!' in exactly 5 words."}]


def test_model(model_name: str) -> dict:
    """Test a single model with Together AI API"""
    result = {
        "model": model_name,
        "status": "unknown",
        "response": None,
        "error": None
    }

    try:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")

        response = client.chat.completions.create(
            model=model_name,
            messages=TEST_MESSAGE,
            temperature=0.2,
            top_p=0.7,
            max_tokens=50,
        )

        content = response.choices[0].message.content
        result["status"] = "success"
        result["response"] = content
        print(f"✅ SUCCESS")
        print(f"   Response: {content[:100]}..." if len(content) > 100 else f"   Response: {content}")
        print(f"   Usage: {response.usage.prompt_tokens} prompt + {response.usage.completion_tokens} completion tokens")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        print(f"❌ FAILED")
        print(f"   Error: {str(e)[:200]}")

    return result


def test_model_with_tools(model_name: str) -> dict:
    """Test a model with tool calling (as used in codebase)"""
    result = {
        "model": model_name,
        "test_type": "tool_calling",
        "status": "unknown",
        "response": None,
        "error": None
    }

    # Sample tool definition (similar to codebase)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    try:
        print(f"\n{'='*60}")
        print(f"Testing with tools: {model_name}")
        print(f"{'='*60}")

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=100,
            tools=tools
        )

        message = response.choices[0].message
        result["status"] = "success"

        if message.tool_calls:
            result["response"] = f"Tool call: {message.tool_calls[0].function.name}({message.tool_calls[0].function.arguments})"
            print(f"✅ SUCCESS (tool call detected)")
            print(f"   Tool: {message.tool_calls[0].function.name}")
            print(f"   Args: {message.tool_calls[0].function.arguments}")
        else:
            result["response"] = message.content
            print(f"✅ SUCCESS (no tool call, text response)")
            print(f"   Response: {message.content[:100] if message.content else 'None'}")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        print(f"❌ FAILED")
        print(f"   Error: {str(e)[:200]}")

    return result


def main():
    print("\n" + "="*70)
    print("Together AI API Test for ToolOrchestra")
    print("="*70)

    # Check API key
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("\n❌ ERROR: TOGETHER_API_KEY environment variable not set!")
        print("   Please run: export TOGETHER_API_KEY='your-api-key'")
        return

    print(f"\n✅ TOGETHER_API_KEY found (length: {len(api_key)} chars)")
    print(f"   Base URL: https://api.together.xyz/v1")

    # Test basic chat completion for each model
    print("\n" + "="*70)
    print("PART 1: Basic Chat Completion Tests")
    print("="*70)

    results = []
    for model in MODELS_TO_TEST:
        result = test_model(model)
        results.append(result)

    # Test tool calling for models that support it
    print("\n" + "="*70)
    print("PART 2: Tool Calling Tests (models used as fallback)")
    print("="*70)

    tool_models = [
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ]

    tool_results = []
    for model in tool_models:
        result = test_model_with_tools(model)
        tool_results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\nBasic Chat Completion:")
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"   {success_count}/{len(results)} models passed")

    for r in results:
        status_icon = "✅" if r["status"] == "success" else "❌"
        print(f"   {status_icon} {r['model']}")

    print("\nTool Calling:")
    tool_success_count = sum(1 for r in tool_results if r["status"] == "success")
    print(f"   {tool_success_count}/{len(tool_results)} models passed")

    for r in tool_results:
        status_icon = "✅" if r["status"] == "success" else "❌"
        print(f"   {status_icon} {r['model']}")

    # Final verdict
    all_passed = all(r["status"] == "success" for r in results + tool_results)
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Together AI integration is ready.")
    else:
        print("⚠️  SOME TESTS FAILED. Please check the errors above.")
        print("\nPossible issues:")
        print("   - Model name might be different on Together AI")
        print("   - Model might not be available on Together AI")
        print("   - API key might have insufficient permissions")
        print("\nCheck Together AI's model list: https://docs.together.ai/docs/chat-models")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
