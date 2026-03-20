# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
import re
import time
import traceback
from typing import Any

from litellm import completion
from litellm.exceptions import APIError, AuthenticationError
from litellm.types.utils import Choices, ModelResponse
from pydantic import BaseModel

from are.simulation.agents.llm.llm_engine import LLMEngine, LLMEngineException

# TODO: Litellm should be agnostic to the agent or model. Remove this dependency.
from are.simulation.agents.llm.types import MessageRole
from are.simulation.agents.multimodal import Attachment

from are.simulation.agents.llm.litellm.cloudgpt_aoai import get_openai_token_provider
from openai import AzureOpenAI

logger = logging.getLogger(__name__)

# TODO: Instead of tool, llama has ipython support. We should use that instead of user role.
role_conversions = {"tool-response": "user", "tool-call": "assistant"}


class LiteLLMModelConfig(BaseModel):
    model_name: str
    provider: str
    endpoint: str | None = None
    api_key: str | None = None


class LiteLLMEngine(LLMEngine):
    """
    A class that extends the LLMEngine to provide a specific implementation for the Litellm model.
    Attributes:
        model_config (ModelConfig): The configuration for the model.
    """

    def __init__(self, model_config: LiteLLMModelConfig):
        super().__init__(model_config.model_name)

        self.model_config = model_config

        self.mock_response = None
        if model_config.provider == "mock":
            self.mock_response = """Thought: Good choice, this is a mock, so I can't do anything. Let's return the result.
Action:
{
  "action": "_mock",
  "action_input": "Mock result"
}<end_action>
"""
        self._client = None
        if self.model_config.provider == "azure":
            token_provider = get_openai_token_provider(use_azure_cli=True)
            token_provider()

            self._api_version = "2025-04-01-preview"
            self._base_url = "https://cloudgpt-openai.azure-api.net/openai/"

            self._client = AzureOpenAI(
                base_url=self._base_url,
                api_version=self._api_version,
                azure_ad_token_provider=token_provider
            )            

    def _convert_message_to_litellm_format(
        self, message: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert a message to LiteLLM format, handling both text and multimodal content."""
        role = MessageRole(message["role"]).value
        role = role_conversions.get(role, role)

        # Handle attachments if present
        attachments: list[Attachment] | None = message.get("attachments")
        content = message.get("content", "")

        if attachments and len(attachments) > 0:
            # Create multimodal content with both text and images
            content_list = []

            # Add text content if present
            if content:
                content_list.append({"type": "text", "text": content})

            # Add image attachments
            for attachment in attachments:
                if attachment.mime.startswith("image/"):
                    content_list.append(attachment.to_openai_json())
                else:
                    logger.warning(
                        f"Unsupported attachment mime type: {attachment.mime}"
                    )

            return {"role": role, "content": content_list}
        else:
            # Text-only message
            return {"role": role, "content": content}

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        stop_sequences=[],
        **kwargs,
    ) -> tuple[str, dict | None]:
        try:
            # Convert messages to LiteLLM format with multimodal support
            converted_messages = []
            for message in messages:
                converted_message = self._convert_message_to_litellm_format(message)
                converted_messages.append(converted_message)

            provider = (
                self.model_config.provider
                if self.model_config.provider != "local"
                else None
            )

            t0 = time.time()

            if self.model_config.provider == "azure":
                assert self._client is not None, "CloudGPT client not initialized."
                response = completion(
                    model=self.model_config.model_name,
                    custom_llm_provider=provider,
                    messages=converted_messages,
                    api_base=self._base_url,
                    api_version=self._api_version,
                    client=self._client,
                    mock_response=self.mock_response,
                )
            else:
                response = completion(
                    model=self.model_config.model_name,
                    custom_llm_provider=provider,
                    messages=converted_messages,
                    api_base=self.model_config.endpoint,
                    api_key=self.model_config.api_key,
                    mock_response=self.mock_response,
                )

            completion_duration = time.time() - t0

            assert type(response) is ModelResponse
            assert len(response.choices) >= 1
            assert type(response.choices[0]) is Choices

            res = response.choices[0].message.content
            assert res is not None

            res = re.sub(
                r'\[\[(True|False)\]\]|(True|False)',
                lambda m: m.group(0) if m.group(1) else m.group(2).lower(),
                res,
            )
            for stop_token in stop_sequences:
                res = res.split(stop_token)[0]

            # Extract token usage from the LLM response
            metadata: dict[str, int | float] = {
                "completion_duration": completion_duration,
            }
            usage = getattr(response, "usage", None)
            if usage is not None:
                metadata["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
                metadata["completion_tokens"] = getattr(usage, "completion_tokens", 0) or 0
                metadata["total_tokens"] = getattr(usage, "total_tokens", 0) or 0
                # reasoning_tokens may be a top-level attr (litellm) or nested
                reasoning = getattr(usage, "reasoning_tokens", None)
                if reasoning is None:
                    details = getattr(usage, "completion_tokens_details", None)
                    if details is not None:
                        reasoning = (
                            details.get("reasoning_tokens", 0)
                            if isinstance(details, dict)
                            else getattr(details, "reasoning_tokens", 0)
                        )
                metadata["reasoning_tokens"] = reasoning or 0

            return res, metadata
        except (AuthenticationError, APIError) as e:
            raise LLMEngineException("Auth error in litellm.") from e
