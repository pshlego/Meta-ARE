# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
from typing import Any

from are.simulation.types import CompletedEvent, EventTimeComparator, EventType
from are.simulation.validation.base import EventJudge
from are.simulation.validation.configs import (
    AgentEventJudgeConfig,
    EnvUserEventJudgeConfig,
    MildToolJudgeConfig,
)
from are.simulation.validation.tool_judge import MildToolJudge
from are.simulation.validation.utils.scenario_utils import CompletedOracleEvent
from are.simulation.validation.utils.trace_utils import injected_traceable

logger: logging.Logger = logging.getLogger(__name__)


class EnvUserEventJudge(EventJudge):
    """
    A judge that compares a pair of environment/user events from the agent log and the oracle agent log.
    The two events match if their event ids is the same.
    """

    def __init__(self, event_type: EventType, config: EnvUserEventJudgeConfig) -> None:
        super().__init__(config, event_type, "env-user")

    @injected_traceable(trace_type="eq_checker", tags=["judge"])
    def eq_checker(self, x_agent: Any, x_oracle: Any, **kwargs) -> bool:
        return x_agent == x_oracle

    def compare(
        self, agent_event: CompletedEvent, oracle_event: CompletedOracleEvent, **kwargs
    ) -> bool | None:
        # Only compare the event id
        return self.eq_checker(
            agent_event.event_id, oracle_event.event_id, arg_name="event_id"
        )


class AgentEventJudge(EventJudge):
    """
    A judge that compares a pair of agent events from the agent log and the oracle agent log.
    """

    def __init__(self, config: AgentEventJudgeConfig) -> None:
        super().__init__(config, EventType.AGENT, "agent")
        self.config = config
        # Per tool judge
        self.tool_judges = {}
        for tool_name in self.config.per_tool_arg_to_checker_type.keys():
            # Arg to checker type
            arg_to_checker_type = self.config.per_tool_arg_to_checker_type[tool_name]
            # Event id to checker params
            event_id_to_checker_params = None
            if self.config.event_id_to_checker_params is not None:
                event_id_to_checker_params = {
                    event_id: [
                        checker_param
                        for checker_param in checker_params
                        if checker_param.tool_name == tool_name
                    ]
                    for event_id, checker_params in self.config.event_id_to_checker_params.items()
                    if any(
                        checker_param.tool_name == tool_name
                        for checker_param in checker_params
                    )
                }
            # Soft checker types
            soft_checker_types = (
                self.config.per_tool_soft_checker_types[tool_name]
                if tool_name in self.config.per_tool_soft_checker_types
                else []
            )
            self.tool_judges[tool_name] = MildToolJudge(
                MildToolJudgeConfig(
                    tool_name=tool_name,
                    arg_to_checker_type=arg_to_checker_type,
                    engine=self.config.engine,
                    event_id_to_checker_params=event_id_to_checker_params,
                    soft_checker_types=soft_checker_types,
                    tracer=self.tracer,
                )
            )

    @injected_traceable(trace_type="event_time_checker", tags=["judge"])
    def event_time_checker(
        self,
        agent_event_time: float,
        oracle_event_time: float,
        pre_event_tolerance_seconds: float = 5.0,
        post_event_tolerance_seconds: float = 20.0,
        event_time_comparator: str | None = None,
    ) -> bool:
        """
        Checks if the agent event time is within the allowed tolerance range compared to the oracle event time.

        Args:
            agent_event_time (float): The time of the agent event (relative or absolute)
            oracle_event_time (float): The time of the oracle event (relative or absolute).
            pre_event_tolerance_seconds (float): The allowed time in seconds before the oracle event time.
            post_event_tolerance_seconds (float): The allowed time in seconds after the oracle event time.
            event_time_comparator (str | None): The type of comparison to perform between the agent and oracle event times. The arg type is str instead of EventTimeComparator for better readability in the tracer.

        Returns:
            bool: True if the agent event time is within the allowed tolerance range, False otherwise.
        """

        if (
            event_time_comparator is None
            or event_time_comparator == EventTimeComparator.EQUAL.value
        ):
            return (
                agent_event_time <= oracle_event_time + post_event_tolerance_seconds
                and agent_event_time >= oracle_event_time - pre_event_tolerance_seconds
            )
        elif event_time_comparator == EventTimeComparator.LESS_THAN.value:
            return agent_event_time <= oracle_event_time + post_event_tolerance_seconds
        elif event_time_comparator == EventTimeComparator.GREATER_THAN.value:
            return agent_event_time >= oracle_event_time - pre_event_tolerance_seconds
        else:
            raise ValueError(
                f"Event time comparator {event_time_comparator} is not valid"
            )

    def check_time(
        self,
        agent_event: CompletedEvent,
        oracle_event: CompletedOracleEvent,
        max_parent_oracle_event_time: float,
        max_parent_agent_event_time: float,
    ) -> bool:
        assert agent_event.event_time is not None, "Agent event time cannot be None"
        comparator = (
            oracle_event.event_time_comparator.value
            if oracle_event.event_time_comparator
            else None
        )
        if oracle_event.absolute_event_time is not None:
            return self.event_time_checker(
                agent_event_time=agent_event.event_time,
                oracle_event_time=oracle_event.absolute_event_time,
                pre_event_tolerance_seconds=self.config.pre_event_tolerance_seconds,
                post_event_tolerance_seconds=self.config.post_event_tolerance_seconds,
                event_time_comparator=comparator,
            )
        agent_event_time = agent_event.event_time
        agent_event_relative_time = agent_event_time - max_parent_agent_event_time
        oracle_event_time = oracle_event.event_time
        assert oracle_event_time is not None, "Oracle event time cannot be None"
        oracle_event_relative_time = oracle_event_time - max_parent_oracle_event_time
        if (
            oracle_event_relative_time > self.config.check_time_threshold_seconds
            or oracle_event.event_time_comparator is not None
        ):
            return self.event_time_checker(
                agent_event_time=agent_event_relative_time,
                oracle_event_time=oracle_event_relative_time,
                pre_event_tolerance_seconds=self.config.pre_event_tolerance_seconds,
                post_event_tolerance_seconds=self.config.post_event_tolerance_seconds,
                event_time_comparator=comparator,
            )
        return True

    def compare(
        self, agent_event: CompletedEvent, oracle_event: CompletedOracleEvent, **kwargs
    ) -> bool | None:
        oracle_tool_name = oracle_event.tool_name
        agent_tool_name = agent_event.tool_name
        logger.info(f"Comparing {oracle_tool_name} to {agent_tool_name}")
        # First check time (skipped when disable_time_check is True)
        if not self.config.disable_time_check and not self.check_time(
            agent_event=agent_event,
            oracle_event=oracle_event,
            max_parent_oracle_event_time=kwargs.get(
                "max_parent_oracle_event_time", 0.0
            ),
            max_parent_agent_event_time=kwargs.get("max_parent_agent_event_time", 0.0),
        ):
            return False
        # Check tool call (action)
        assert oracle_tool_name in self.tool_judges, (
            f"Tool {oracle_tool_name} not found in tool judges"
        )
        return self.tool_judges[oracle_tool_name](agent_event, oracle_event, **kwargs)
