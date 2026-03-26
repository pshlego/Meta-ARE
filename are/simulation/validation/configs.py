# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from dataclasses import dataclass, field
from typing import Any, Callable

from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
from are.simulation.validation.constants import (
    TOOL_ARG_CHECKER_TYPE_REGISTRY,
    TOOL_EVALUATION_CRITERIA_REGISTRY,
    TOOL_SOFT_CHECKER_TYPE_REGISTRY,
    CheckerType,
    SoftCheckerType,
    ToolArgCheckerTypeRegistry,
    ToolCriteriaRegistry,
    ToolSoftCheckerTypeRegistry,
)
from are.simulation.validation.prompts import (
    IN_CONTEXT_JUDGE_SYSTEM_PROMPT_TEMPLATE,
    TIME_SYSTEM_PROMPT_TEMPLATE,
)

# Default judge configuration
DEFAULT_JUDGE_MODEL = "meta-llama/Meta-Llama-3.3-70B-Instruct"
DEFAULT_JUDGE_PROVIDER = "huggingface"


def create_judge_engine(
    judge_engine_config: LLMEngineConfig | None = None,
):
    """Create a judge engine with the specified configuration."""
    if judge_engine_config is None:
        judge_engine_config = LLMEngineConfig(
            model_name=DEFAULT_JUDGE_MODEL,
            provider=None,
            endpoint=None,
        )

    # Use LiteLLM for all other cases (external or internal with overrides)
    from are.simulation.agents.llm.litellm.litellm_engine import (
        LiteLLMEngine,
        LiteLLMModelConfig,
    )

    final_provider = (
        judge_engine_config.provider or DEFAULT_JUDGE_PROVIDER or "huggingface"
    )

    judge_config = LiteLLMModelConfig(
        model_name=judge_engine_config.model_name,
        provider=final_provider,
        endpoint=judge_engine_config.endpoint,
        reasoning_effort=judge_engine_config.reasoning_effort,
    )

    return LiteLLMEngine(model_config=judge_config)


@dataclass
class ToolCheckerParam:
    # Parameter for the tool checker of the hard judge
    arg_name: str
    checker_type: CheckerType
    tool_name: str
    checker_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseToolJudgeConfig:
    tool_name: str
    arg_to_checker_type: dict[
        str, CheckerType
    ]  # the list of args to check and the type of checker to use for each arg
    tracer: Callable | None = None


@dataclass
class HardToolJudgeConfig(BaseToolJudgeConfig):
    event_id_to_checker_params: dict[str, list[ToolCheckerParam]] | None = None


@dataclass
class SoftToolJudgeConfig(BaseToolJudgeConfig):
    engine: Callable = field(default_factory=create_judge_engine)
    # Soft checker
    soft_checker_types: list[SoftCheckerType] = field(
        default_factory=lambda: [SoftCheckerType.content_checker]
    )

    def __post_init__(self):
        if len(self.soft_checker_types) == 0:
            self.soft_checker_types = [SoftCheckerType.content_checker]


@dataclass
class MildToolJudgeConfig(BaseToolJudgeConfig):
    engine: Callable = field(default_factory=create_judge_engine)
    soft_checker_types: list[SoftCheckerType] = field(default_factory=list)
    # Scripted checkers related config
    event_id_to_checker_params: dict[str, list[ToolCheckerParam]] | None = None


@dataclass
class BaseEventJudgeConfig:
    tracer: Callable | None = None


@dataclass
class EnvUserEventJudgeConfig(BaseEventJudgeConfig):
    pass


@dataclass
class AgentEventJudgeConfig(BaseEventJudgeConfig):
    # Time related config
    check_time_threshold_seconds: float = 1.0
    pre_event_tolerance_seconds: float = 10.0
    post_event_tolerance_seconds: float = 25.0
    # Set to True to skip check_time() enforcement (timing bypass for experiments)
    disable_time_check: bool = False
    # Tool related config
    per_tool_arg_to_checker_type: ToolArgCheckerTypeRegistry = field(
        default_factory=lambda: TOOL_ARG_CHECKER_TYPE_REGISTRY
    )
    per_tool_soft_checker_types: ToolSoftCheckerTypeRegistry = field(
        default_factory=lambda: TOOL_SOFT_CHECKER_TYPE_REGISTRY
    )
    engine: Callable = field(default_factory=create_judge_engine)
    # Scripted checkers related config
    event_id_to_checker_params: dict[str, list[ToolCheckerParam]] | None = None


@dataclass
class BaseJudgeConfig:
    tracer: Callable | None = None


@dataclass
class GraphPerEventJudgeConfig(BaseJudgeConfig):
    # Time related config
    check_time_threshold_seconds: float = 1.0
    pre_event_tolerance_seconds: float = 10.0
    post_event_tolerance_seconds: float = 25.0
    # Set to True to skip check_time() enforcement (timing bypass for experiments)
    disable_time_check: bool = False
    # Tool related config
    per_tool_arg_to_checker_type: ToolArgCheckerTypeRegistry = field(
        default_factory=lambda: TOOL_ARG_CHECKER_TYPE_REGISTRY
    )
    engine: Callable = field(default_factory=create_judge_engine)
    per_tool_soft_checker_types: ToolSoftCheckerTypeRegistry = field(
        default_factory=lambda: TOOL_SOFT_CHECKER_TYPE_REGISTRY
    )
    # Scripted checkers related config
    # If this field is not `None`, the soft judge will not be used.
    event_id_to_checker_params: dict[str, list[ToolCheckerParam]] | None = None
    # Preliminary check
    extra_send_message_to_user_allowed: int = 1


@dataclass
class ScriptedGraphPerEventJudgeConfig(GraphPerEventJudgeConfig):
    """
    Config for the scripted graph per event judge.
    Scripted judge is a judge where the soft judge is deactivated and instead scripted checks will be performed by the hard judge.
    The `event_id_to_checker_params` field is used to specify the scripted checks to perform.
    """

    # Change default such that soft judge is not used.
    event_id_to_checker_params: dict[str, list[ToolCheckerParam]] | None = field(
        default_factory=dict
    )

    def __post_init__(self):
        if self.event_id_to_checker_params is None:
            raise ValueError(
                "event_id_to_checker_params must be specified for ScriptedGraphPerEventJudgeConfig"
            )


@dataclass
class InContextJudgeConfig(BaseJudgeConfig):
    # Time related config
    check_time_threshold_seconds: float = 1.0
    pre_event_tolerance_seconds: float = 10.0
    post_event_tolerance_seconds: float = 25.0
    # Set to True to skip check_time() enforcement (timing bypass for experiments)
    disable_time_check: bool = False
    time_system_prompt_template: str = TIME_SYSTEM_PROMPT_TEMPLATE
    # Tool related config
    per_tool_evaluation_criteria: ToolCriteriaRegistry = field(
        default_factory=lambda: TOOL_EVALUATION_CRITERIA_REGISTRY
    )
    tool_to_selected_args: ToolArgCheckerTypeRegistry = (
        field(  # Will not use the checker type but only arg names
            default_factory=lambda: TOOL_ARG_CHECKER_TYPE_REGISTRY
        )
    )
    engine: Callable = field(default_factory=create_judge_engine)
    system_prompt_template: str = IN_CONTEXT_JUDGE_SYSTEM_PROMPT_TEMPLATE
