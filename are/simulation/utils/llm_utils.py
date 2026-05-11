# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging

from are.simulation.agents.llm.llm_engine import ModelConfig

# DEFAULT_MODEL = "meta-llama/llama3-70b-instruct"
# DEFAULT_PROVIDER = "huggingface"
DEFAULT_MODEL = "gpt-5-20250807"
DEFAULT_PROVIDER = "openai"
DEFAULT_APP_AGENT = "default_app_agent"

logger = logging.getLogger(__name__)


def build_llm(model_config: ModelConfig | None = None):
    """
    Build an LLM engine based on the provided model configuration.

    :param model_config: Configuration for the LLM model
    :type model_config: ModelConfig | None
    :return: An LLM engine instance
    :raises Exception: If the model fails to load
    """
    try:
        if model_config is None:
            raise ValueError("model_config must be provided")

        from are.simulation.agents.llm.litellm.litellm_engine import (
            LiteLLMEngine,
            LiteLLMModelConfig,
        )

        return LiteLLMEngine(
            model_config=LiteLLMModelConfig(
                model_name=f"{model_config.model_name}", provider="huggingface"
            )
        )
    except Exception as e:
        model_name = model_config.model_name if model_config else ""
        provider = model_config.provider if model_config else ""
        raise Exception(f"Failed to load model '{model_name}' from '{provider}':  {e}")
