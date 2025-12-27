import os
from openai import OpenAI

# 设置 Together AI client
client = OpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)

# ID:             endpoint-686c7075-c556-421d-80ef-8a14ec577f2f
# Name:           HK123/Qwen/Qwen2.5-Math-7B-Instruct-edb7ac78
# Display Name:   Qwen/Qwen2.5-Math-7B-Instruct-edb7ac78
# Hardware:       1x_nvidia_h100_80gb_sxm
# Autoscaling:    Min=1, Max=1
# Model:          Qwen/Qwen2.5-Math-7B-Instruct
# Type:           dedicated
# Owner:          HK123
# State:          STARTED
# Created:        2025-12-25 02:56:34.224000+00:00

# ⚠️ Together dedicated endpoints should be called using the endpoint **Name** as `model`.
# (The endpoint ID "endpoint-..." may not work depending on Together configuration.)
MODEL_NAME = "HK123/Qwen/Qwen2.5-Math-7B-Instruct-edb7ac78"

# 一个有挑战性的数学问题
math_problem = """
Solve the following problem step by step:

A water tank can be filled by pipe A in 4 hours and by pipe B in 6 hours. 
Pipe C can empty the full tank in 8 hours. If all three pipes are opened 
simultaneously when the tank is empty, how long will it take to fill the tank?

Please show your complete reasoning process.
"""

print("=" * 60)
print(f"Testing Model Name: {MODEL_NAME}")
print(f"Model: Qwen/Qwen2.5-Math-7B-Instruct")
print("=" * 60)
print(f"\nQuestion:\n{math_problem}")
print("=" * 60)

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert mathematics tutor. Solve problems step-by-step with clear explanations."},
            {"role": "user", "content": math_problem},
        ],
        temperature=0.3,  # Lower temperature for more precise math reasoning
        max_tokens=2048,
    )
    
    print("\n✅ Endpoint Response:\n")
    print(response.choices[0].message.content)
    print("\n" + "=" * 60)
    print("✅ Test successful! Endpoint is working properly.")
    
    # 显示一些额外信息
    if hasattr(response, 'usage'):
        print(f"\nToken Usage:")
        print(f"  - Prompt tokens: {response.usage.prompt_tokens}")
        print(f"  - Completion tokens: {response.usage.completion_tokens}")
        print(f"  - Total tokens: {response.usage.total_tokens}")
    
except Exception as e:
    print(f"\n❌ Error calling endpoint: {e}")
    print(f"\nError type: {type(e).__name__}")
    import traceback
    traceback.print_exc()