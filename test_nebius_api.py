#!/usr/bin/env python3
"""
Test script for Nebius API integration with Qwen3-32B

This script tests whether the Nebius API is properly configured and working.
"""

import os
import sys
from openai import OpenAI

def test_nebius_api():
    """Test Nebius API connection and basic functionality"""

    print("="*60)
    print("Testing Nebius API Integration for Qwen3-32B")
    print("="*60)

    # Check if NEBIUS_API_KEY is set
    nebius_api_key = os.getenv("NEBIUS_API_KEY")

    if not nebius_api_key:
        print("\n❌ ERROR: NEBIUS_API_KEY environment variable is not set")
        print("\nPlease set it in setup_envs.sh or export it:")
        print('  export NEBIUS_API_KEY="your_api_key_here"')
        print('  source setup_envs.sh')
        sys.exit(1)

    print(f"\n✓ NEBIUS_API_KEY is set: {nebius_api_key[:20]}...")

    # Initialize Nebius client
    print("\n[1/3] Initializing Nebius client...")
    try:
        client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=nebius_api_key
        )
        print("  ✓ Client initialized successfully")
    except Exception as e:
        print(f"  ❌ Failed to initialize client: {e}")
        sys.exit(1)

    # Test basic chat completion
    print("\n[2/3] Testing basic chat completion...")
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3-32B-fast",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Say 'Hello from Nebius!' in exactly those words."
                }
            ],
            max_tokens=50,
            temperature=0.1
        )

        content = response.choices[0].message.content
        print(f"  ✓ Received response: {content}")

        # Check usage statistics
        if hasattr(response, 'usage'):
            print(f"  ✓ Token usage: {response.usage.prompt_tokens} prompt + "
                  f"{response.usage.completion_tokens} completion = "
                  f"{response.usage.total_tokens} total")

    except Exception as e:
        print(f"  ❌ Failed to get completion: {e}")
        sys.exit(1)

    # Test tool calling (if supported)
    print("\n[3/3] Testing tool calling capability...")
    try:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        response = client.chat.completions.create(
            model="Qwen/Qwen3-32B-fast",
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in San Francisco?"
                }
            ],
            tools=tools,
            max_tokens=100
        )

        if response.choices[0].message.tool_calls:
            print("  ✓ Tool calling is supported")
            print(f"  ✓ Tool called: {response.choices[0].message.tool_calls[0].function.name}")
        else:
            print("  ⚠️  No tool calls detected (may not be supported or need different prompt)")

    except Exception as e:
        print(f"  ⚠️  Tool calling test failed: {e}")
        print("  (This is optional, basic completion still works)")

    # Summary
    print("\n" + "="*60)
    print("✅ All tests passed! Nebius API is working correctly.")
    print("="*60)
    print("\nYou can now run τ²-Bench evaluation with Nebius API:")
    print("  cd evaluation/tau2-bench/")
    print("  python run_local.py --agent-model $CKPT_DIR")
    print("\nQwen3-32B calls will automatically use Nebius API.")
    print("="*60)

def test_llm_call_integration():
    """Test integration with LLM_CALL.py"""

    print("\n" + "="*60)
    print("Testing LLM_CALL.py Integration")
    print("="*60)

    # Add repo path to sys.path
    repo_path = os.getenv("REPO_PATH")
    if not repo_path:
        print("\n⚠️  REPO_PATH not set, cannot test LLM_CALL.py integration")
        print("  This is OK if you only want to test Nebius API directly")
        return

    sys.path.insert(0, repo_path)

    try:
        from LLM_CALL import get_llm_response
        print("\n✓ Successfully imported get_llm_response from LLM_CALL.py")

        # Test with Qwen3-32B
        print("\nTesting get_llm_response with Qwen3-32B...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Integration test successful!' in exactly those words."}
        ]

        response = get_llm_response(
            model="Qwen/Qwen3-32B",
            messages=messages,
            temperature=0.1,
            max_length=50,
            model_config=None,  # Will use Nebius API if NEBIUS_API_KEY is set
            model_type='vllm'
        )

        print(f"✓ Response: {response}")
        print("\n✅ LLM_CALL.py integration is working correctly!")

    except ImportError as e:
        print(f"\n⚠️  Could not import LLM_CALL.py: {e}")
        print("  Make sure REPO_PATH is set correctly")
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Nebius API integration")
    parser.add_argument(
        "--skip-integration",
        action="store_true",
        help="Skip LLM_CALL.py integration test"
    )
    args = parser.parse_args()

    # Test basic Nebius API
    test_nebius_api()

    # Test integration with LLM_CALL.py
    if not args.skip_integration:
        test_llm_call_integration()
