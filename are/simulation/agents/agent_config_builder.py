# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from are.simulation.agents.are_simulation_agent_config import (
    ARESimulationReactAgentConfig,
    ARESimulationReactAppAgentConfig,
    ARESimulationReactBaseAgentConfig,
    RunnableARESimulationAgentConfig,
)
from are.simulation.agents.default_agent.prompts import (
    DEFAULT_ARE_SIMULATION_APP_AGENT_REACT_JSON_SYSTEM_PROMPT,
    DEFAULT_ARE_SIMULATION_REACT_JSON_SYSTEM_PROMPT,
)
from are.simulation.agents.default_agent.prompts.system_prompt import (
    ARE_SIMULATION_ENVIRONMENT_INSTRUCTIONS,
    GENERAL_SYSTEM_PROMPT_TEMPLATE,
    REACT_LOOP_JSON_SYSTEM_PROMPT,
    SYSTEM_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


class AbstractAgentConfigBuilder(ABC):
    """
    Abstract class for building agent configs.
    """

    @abstractmethod
    def build(
        self,
        agent_name: str,
        scenario_id: str | None = None,
        hints_file_path: str | None = None,
    ) -> Any:
        """
        Method to build a config.
        :param agent_name: Name of the agent that affects the config type.
        :param scenario_id: Scenario ID for per-task hint lookup.
        :param hints_file_path: Path to JSON hints file mapping scenario_id -> hint text.
        :returns: An instance of the config.
        """


def _build_system_prompt_with_hints(
    scenario_id: str, hints_file_path: str
) -> str | None:
    """Load hints from JSON file and build a system prompt with injected hints.

    Returns the formatted system prompt string, or None if no hint found for this scenario.
    """
    try:
        with open(hints_file_path, "r") as f:
            hints = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load hints file {hints_file_path}: {e}")
        return None

    hint_text = hints.get(scenario_id)
    if not hint_text:
        return None

    return SYSTEM_PROMPT_TEMPLATE.format(
        general_instructions=GENERAL_SYSTEM_PROMPT_TEMPLATE,
        agent_instructions=REACT_LOOP_JSON_SYSTEM_PROMPT.format(
            json_agent_hints=hint_text,
        ),
        environment_instructions=ARE_SIMULATION_ENVIRONMENT_INSTRUCTIONS.format(
            environment_hints="",
        ),
    )


class AgentConfigBuilder(AbstractAgentConfigBuilder):
    def build(
        self,
        agent_name: str,
        scenario_id: str | None = None,
        hints_file_path: str | None = None,
    ) -> RunnableARESimulationAgentConfig:
        match agent_name:
            case "default":
                # Try to inject per-task hints if both scenario_id and hints_file are provided
                system_prompt = None
                if scenario_id and hints_file_path:
                    system_prompt = _build_system_prompt_with_hints(
                        scenario_id, hints_file_path
                    )
                if system_prompt is None:
                    system_prompt = str(
                        DEFAULT_ARE_SIMULATION_REACT_JSON_SYSTEM_PROMPT
                    )

                return ARESimulationReactAgentConfig(
                    agent_name=agent_name,
                    base_agent_config=ARESimulationReactBaseAgentConfig(
                        system_prompt=system_prompt,
                        max_iterations=80,
                    ),
                )

            case _:
                raise ValueError(f"Agent {agent_name} not found")


class AppAgentConfigBuilder(AbstractAgentConfigBuilder):
    def build(
        self,
        agent_name: str,
    ) -> ARESimulationReactAppAgentConfig:
        match agent_name:
            case "default_app_agent":
                return ARESimulationReactAppAgentConfig(
                    agent_name=agent_name,
                    system_prompt=str(
                        DEFAULT_ARE_SIMULATION_APP_AGENT_REACT_JSON_SYSTEM_PROMPT
                    ),
                    max_iterations=80,
                )
            case _:
                raise ValueError(f"Sub agent {agent_name} not found")
