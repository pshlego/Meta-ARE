# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from are.simulation.agents.llm.models import ALL_MODELS
from are.simulation.types import SimulatedGenerationTimeConfig


class LLMEngineConfig(BaseModel):
    model_name: str = Field(
        default=ALL_MODELS[0] if ALL_MODELS else "",
        examples=ALL_MODELS,
    )
    provider: str | None = None
    endpoint: str | None = None
    reasoning_effort: str | None = None  # "minimal", "low", "medium", "high", or "xhigh"


class ARESimulationBaseAgentConfig(BaseModel):
    llm_engine_config: LLMEngineConfig = Field(default_factory=LLMEngineConfig)
    simulated_generation_time_config: SimulatedGenerationTimeConfig | None = Field(
        default=None
    )
    use_custom_logger: bool = Field(default=True)


class ARESimulationReactBaseAgentConfig(ARESimulationBaseAgentConfig):
    system_prompt: str = Field(default="")
    max_iterations: int = Field(default=80)


class RunnableARESimulationAgentConfig(ABC):
    # Handle common config fields.
    @abstractmethod
    def get_agent_name(self) -> str | None:
        pass

    @abstractmethod
    def get_base_agent_config(self) -> ARESimulationBaseAgentConfig:
        pass

    # Handle model dump and schema.
    @abstractmethod
    def get_model_dump(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_model_json_schema(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def validate_model(
        self, agent_config_dict: dict[str, Any]
    ) -> "RunnableARESimulationAgentConfig":
        pass


class MainAgentConfig(BaseModel):
    agent_name: str = Field(default="")
    max_turns: int | None = Field(default=None)

    def get_agent_name(self) -> str | None:
        return self.agent_name

    def get_model_dump(self) -> dict[str, Any]:
        return self.model_dump()

    def get_model_json_schema(self) -> dict[str, Any]:
        return self.model_json_schema()


class ARESimulationReactAgentConfig(MainAgentConfig, RunnableARESimulationAgentConfig):
    # Exact subclass type declaration is required for pydantic deserialization.
    base_agent_config: ARESimulationReactBaseAgentConfig = Field(
        default_factory=ARESimulationReactBaseAgentConfig
    )

    def get_base_agent_config(self) -> ARESimulationBaseAgentConfig:
        return self.base_agent_config

    def validate_model(
        self, agent_config_dict: dict[str, Any]
    ) -> RunnableARESimulationAgentConfig:
        return type(self).model_validate(agent_config_dict)


class ARESimulationReactAppAgentConfig(
    MainAgentConfig, ARESimulationReactBaseAgentConfig
):
    max_iterations: int = Field(default=40)
