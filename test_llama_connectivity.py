#!/usr/bin/env python3
"""Minimal test to verify Llama-3.3-70B-Instruct connectivity via OpenAI API."""

import sys
sys.path.insert(0, "/Data2/v-parksungho/Meta-ARE")

from are.simulation.agents.llm.litellm.litellm_engine import LiteLLMEngine, LiteLLMModelConfig

MODEL_NAME = "Llama-3.3-70B-Instruct"
PROVIDER = "openai"

print(f"Testing model: {MODEL_NAME} via provider: {PROVIDER}")
print("=" * 60)

config = LiteLLMModelConfig(
    model_name=MODEL_NAME,
    provider=PROVIDER,
)
engine = LiteLLMEngine(model_config=config)

messages = [{"role": "user", "content": "Say hello in one sentence."}]

try:
    response, metadata = engine.chat_completion(messages)
    print(f"Response: {response}")
    print(f"Metadata: {metadata}")
    print("\n✅ Llama-3.3-70B-Instruct connectivity test PASSED")
except Exception as e:
    print(f"\n❌ Connectivity test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
