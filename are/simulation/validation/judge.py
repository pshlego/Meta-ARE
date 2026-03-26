# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
from collections import Counter
from typing import Any, cast

from are.simulation.scenarios.scenario import Scenario
from are.simulation.scenarios.scenario_imported_from_json.benchmark_scenario import (
    BenchmarkScenarioImportedFromJson,
)
from are.simulation.scenarios.utils.personalization.utils import jinja_format
from are.simulation.types import AbstractEnvironment, CompletedEvent, EventType
from are.simulation.validation.base import BaseJudge
from are.simulation.validation.configs import (
    AgentEventJudgeConfig,
    EnvUserEventJudgeConfig,
    GraphPerEventJudgeConfig,
    InContextJudgeConfig,
)
from are.simulation.validation.event_judge import AgentEventJudge, EnvUserEventJudge
from are.simulation.validation.judge_states import (
    GraphPerEventJudgeState,
    InContextJudgeState,
)
from are.simulation.validation.judgment import (
    EnvOracleMatchingFailure,
    EventComparisonFailure,
    EventComparisonFailureType,
    Judgment,
    OracleEventMatchingFailure,
    ToolCallCountsFailure,
)
from are.simulation.validation.utils.event_utils import (
    AgentEventFilter,
    BulletEventDescription,
    EnvAgentEventFilter,
    resolve_arg_placeholders,
)
from are.simulation.validation.utils.scenario_utils import (
    CompletedOracleEvent,
    extract_agent_events,
    extract_oracle_events,
    extract_tasks,
    extract_user_details,
    run_oracle_mode,
)
from are.simulation.validation.utils.trace_utils import injected_traceable

logger: logging.Logger = logging.getLogger(__name__)


class InContextJudge(BaseJudge):
    """
    A baseline judge that put all the agent and oracle events in the context of a model.
    """

    def __init__(self, config: InContextJudgeConfig):
        super().__init__(config)
        # Config
        self.config = config
        # Engine
        self.engine = self.config.engine
        # Event filter
        self.event_filter = AgentEventFilter()
        # Add time evaluation criteria
        time_evaluation_criterion = jinja_format(
            config.time_system_prompt_template,
            check_time_threshold_seconds=config.check_time_threshold_seconds,
            pre_event_tolerance_seconds=config.pre_event_tolerance_seconds,
            post_event_tolerance_seconds=config.post_event_tolerance_seconds,
        )
        # System prompt
        criteria = [
            k + ":\n" + v for k, v in config.per_tool_evaluation_criteria.items()
        ]
        criteria.append(time_evaluation_criterion)
        criteria = "\n\n".join(criteria)
        self.system_prompt = jinja_format(
            config.system_prompt_template,
            evaluation_criteria=criteria,
        )
        # Tool to event description
        self.tool_to_event_description = {
            k: BulletEventDescription(
                tool_name=k,
                action_selected_args=[kk for kk in v.keys()],
            )
            for k, v in self.config.tool_to_selected_args.items()
        }
        # State
        self.state = InContextJudgeState()

    def initialize_state(self, scenario: Scenario):
        if not isinstance(scenario, BenchmarkScenarioImportedFromJson):
            # If the scenario is not a benchmark scenario, run it in oracle mode
            # to get the oracle events
            logger.warning("Run scenario in oracle mode")
            run_oracle_mode(scenario)
        # Get oracle events and graphs
        nb_turns = scenario.nb_turns  # type: ignore
        assert nb_turns is not None, "Number of turns not found in scenario"
        turn_to_oracle_events = []
        turn_to_oracle_graph = []
        for turn_idx in range(nb_turns):
            oracle_events, oracle_graph = extract_oracle_events(
                scenario, self.event_filter, turn_idx
            )
            turn_to_oracle_events.append(oracle_events)
            turn_to_oracle_graph.append(oracle_graph)
        # Scenario data
        assert scenario.start_time is not None, "Start time not found in scenario"
        # Initialize state
        self.state = InContextJudgeState(
            initialized=True,
            nb_turns=nb_turns,
            turn_idx=-1,
            scenario_start_time=scenario.start_time,
            scenario_tasks=extract_tasks(scenario),
            user_details=extract_user_details(scenario),
            turn_to_agent_events=[],
            turn_to_oracle_events=turn_to_oracle_events,
            turn_to_oracle_graph=turn_to_oracle_graph,
        )

    def update_state(self, env: AbstractEnvironment):
        # Check if the state is initialized
        if not self.state.initialized:
            raise ValueError("State not initialized")
        # Update turn
        self.state.turn_idx += 1
        # Update agent events
        agent_events = extract_agent_events(env, self.event_filter, self.state.turn_idx)
        self.state.turn_to_agent_events.append(agent_events)

    def get_oracle_graph_str(self, event_id: str) -> str:
        graph_str = f"Tool call id: {event_id}"
        parents = self.state.current_turn_oracle_graph.get(event_id, [])
        parents = ", ".join(parents)
        graph_str += "\nParent tool call ids: {parents}\n"
        return graph_str

    def list_events_str(
        self, events: list[CompletedEvent], oracle_events: bool = False
    ) -> str:
        # Get event tool names
        event_tool_names = [e.tool_name for e in events]
        # Get description of the events
        event_descriptions = [
            self.tool_to_event_description[tool_name](event)
            for tool_name, event in zip(event_tool_names, events)
        ]
        # If oracle event add the parents ids
        if oracle_events:
            event_descriptions = [
                self.get_oracle_graph_str(e.event_id) + d
                for d, e in zip(event_descriptions, events)
            ]

        return "\n\n".join(event_descriptions)

    def build_user_prompt(self, agent_events, oracle_events) -> str:
        user_prompt = ""
        # Agent events
        user_prompt += "Agent actions:\n\n"
        user_prompt += self.list_events_str(agent_events)
        # Orace events
        user_prompt += "\n\nOracle actions:\n\n"
        user_prompt += self.list_events_str(oracle_events, oracle_events=True)
        # Task
        task = ""
        if bool(task):
            user_prompt += "\n\nTask: " + task
        # Previous tasks
        previous_task = ""
        if bool(task):
            user_prompt += "\n\nPrevious task: " + previous_task
        # User name
        user_name = ""
        if bool(user_name):
            user_prompt += "\n\nUser name: " + user_name
        return user_prompt

    @injected_traceable(trace_type="judge", tags=["judge"], log_input_args=False)
    def inner_call(self, env: AbstractEnvironment) -> bool | None:
        # Update state
        self.update_state(env)
        # Extract agent events
        agent_events = self.state.current_turn_agent_events
        # Extract oracle events
        oracle_events = self.state.current_turn_oracle_events
        # Construct the user prompt for the judge
        user_prompt = self.build_user_prompt(
            agent_events,
            oracle_events,
        )
        # Messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # Call the engine
        response, _ = self.engine(messages, additional_trace_tags=["judge"])
        # Parse the response
        if response is None:
            return None
        if "[[success]]" in response:
            return True
        elif "[[failure]]" in response:
            return False
        return None

    def __call__(self, env: AbstractEnvironment) -> Judgment:
        success = bool(self.inner_call(env))
        self.state.last_turn_success = success
        return Judgment(
            success=success,
            agent_event_id_to_oracle_event_id=self.state.agent_id_to_oracle_id,
        )


class GraphPerEventJudge(BaseJudge):
    """
    Verifies the correctness of agent events by comparing them with oracle events.
    The judge performs a preliminary check to ensure that the tool call counts are the same in both the oracle and agent events.
    If this check passes, it orders the oracle events in topological order based on their dependencies.
    Then in attempts to match each one with an agent event using two types of tool judges:
    - A hard tool judge for specific arguments
    - A soft tool judge (LLM-based) for other arguments.
    Once a match is found, the judge verifies the causality by ensuring that all parent events of the oracle event have already been matched with previous agent events.
    If all oracle events are successfully matched, the judge returns a success.
    """

    def __init__(self, config: GraphPerEventJudgeConfig | None = None):
        # Config
        self.config = config or GraphPerEventJudgeConfig()
        # Init tracer
        super().__init__(self.config)
        assert self.config.extra_send_message_to_user_allowed >= 0
        # Env judges
        self.event_judges = {
            EventType.ENV: EnvUserEventJudge(
                EventType.ENV, EnvUserEventJudgeConfig(tracer=self.tracer)
            ),
            EventType.USER: EnvUserEventJudge(
                EventType.USER, EnvUserEventJudgeConfig(tracer=self.tracer)
            ),
            EventType.AGENT: AgentEventJudge(
                AgentEventJudgeConfig(
                    check_time_threshold_seconds=self.config.check_time_threshold_seconds,
                    pre_event_tolerance_seconds=self.config.pre_event_tolerance_seconds,
                    post_event_tolerance_seconds=self.config.post_event_tolerance_seconds,
                    disable_time_check=self.config.disable_time_check,
                    per_tool_arg_to_checker_type=self.config.per_tool_arg_to_checker_type,
                    per_tool_soft_checker_types=self.config.per_tool_soft_checker_types,
                    engine=self.config.engine,
                    event_id_to_checker_params=self.config.event_id_to_checker_params,
                    tracer=self.tracer,
                )
            ),
        }
        # Event filter
        self.event_filter = EnvAgentEventFilter()
        # State
        self.state = GraphPerEventJudgeState()

    def get_judge_kwargs(self, oracle_event: CompletedOracleEvent) -> dict[str, Any]:
        judge_kwargs = {}
        # Get tasks
        judge_kwargs["tasks"] = self.state.scenario_tasks[
            : (self.state.oracle_event_id_to_turn_idx[oracle_event.event_id] + 1)
        ]
        # User details
        user_details = self.state.user_details
        judge_kwargs["user_details"] = user_details
        # Tolerance list
        user_name = None
        if user_details is not None:
            user_name = f"{user_details.first_name} {user_details.last_name}"
        judge_kwargs["tolerance_list_str"] = (
            [user_name] if user_name is not None else []
        )
        # Compute max oracle parent agent event time
        parent_ids = self.state.current_turn_oracle_graph[oracle_event.event_id]
        oracle_event_times_by_id = {
            e.event_id: e.event_time for e in self.state.current_turn_oracle_events
        }
        parent_oracle_event_times = [
            oracle_event_times_by_id[parent_id] for parent_id in parent_ids
        ] + [self.state.scenario_start_time]
        assert all(t is not None for t in parent_oracle_event_times), (
            "Oracle event time is None"
        )
        max_parent_oracle_event_time = max(parent_oracle_event_times)  # type: ignore
        argmax_parent_oracle_event_time = [
            i
            for i, t in enumerate(parent_oracle_event_times)
            if t >= max_parent_oracle_event_time
        ]
        judge_kwargs["max_parent_oracle_event_time"] = max_parent_oracle_event_time
        # Compute max agent parent event time
        # It is the max of the parent agent event times for parents that are matched to
        # a parent oracle event with a maximal event time
        parent_agent_event_times = [
            self.state.agent_events[
                self.state.oracle_id_to_agent_idx[parent_id]
            ].event_time
            for parent_id in parent_ids
        ] + [self.state.scenario_start_time]
        parent_agent_event_times = [
            parent_agent_event_times[i] for i in argmax_parent_oracle_event_time
        ]
        assert all(t is not None for t in parent_agent_event_times), (
            "Agent event time is None"
        )
        max_parent_agent_event_time = max(parent_agent_event_times)  # type: ignore
        judge_kwargs["max_parent_agent_event_time"] = max_parent_agent_event_time
        # Start time
        judge_kwargs["start_time"] = self.state.scenario_start_time
        return judge_kwargs

    def initialize_state(self, scenario: Scenario):
        if not isinstance(scenario, BenchmarkScenarioImportedFromJson):
            # If the scenario is not a benchmark scenario, run it in oracle mode
            # to get the oracle events
            logger.warning("Run scenario in oracle mode")
            run_oracle_mode(scenario)
        # Get oracle events and graphs
        nb_turns = scenario.nb_turns  # type: ignore
        assert nb_turns is not None, "Number of turns not found in scenario"
        turn_to_agent_events = []
        turn_to_oracle_events = []
        turn_to_oracle_graph = []
        for turn_idx in range(nb_turns):
            oracle_events, oracle_graph = extract_oracle_events(
                scenario, self.event_filter, turn_idx
            )
            turn_to_oracle_events.append(oracle_events)
            turn_to_oracle_graph.append(oracle_graph)
        # Scenario data
        assert scenario.start_time is not None, "Start time not found in scenario"
        # Initialize state
        self.state = GraphPerEventJudgeState(
            initialized=True,
            nb_turns=nb_turns,
            turn_idx=-1,
            scenario_start_time=scenario.start_time,
            scenario_tasks=extract_tasks(scenario),
            user_details=extract_user_details(scenario),
            turn_to_agent_events=turn_to_agent_events,
            turn_to_oracle_events=turn_to_oracle_events,
            turn_to_oracle_graph=turn_to_oracle_graph,
            oracle_event_id_to_turn_idx=scenario.event_id_to_turn_idx,  # type: ignore
        )

    def update_state(self, env: AbstractEnvironment):
        # Check if the state is initialized
        if not self.state.initialized:
            raise ValueError("State not initialized")
        # Update turn
        self.state.turn_idx += 1
        # Update agent events
        agent_events = extract_agent_events(env, self.event_filter, self.state.turn_idx)
        self.state.turn_to_agent_events.append(agent_events)

    @injected_traceable(trace_type="check_tool_call_counts", tags=["judge"])
    def check_tool_call_counts(
        self,
        agent_counter: Counter,
        agent_aui_count: int,
        oracle_counter: Counter,
        oracle_aui_count: int,
        extra_send_message_to_user_allowed: int = 0,
    ) -> Judgment:
        if (
            agent_counter == oracle_counter
            and oracle_aui_count
            <= agent_aui_count
            <= oracle_aui_count + extra_send_message_to_user_allowed
        ):
            return Judgment(success=True)
        return Judgment(
            success=False,
            failure=ToolCallCountsFailure(
                agent_counter=agent_counter,
                agent_aui_count=agent_aui_count,
                oracle_counter=oracle_counter,
                oracle_aui_count=oracle_aui_count,
                extra_send_message_to_user_allowed=extra_send_message_to_user_allowed,
            ),
        )

    def preliminary_checks(
        self,
        agent_events: list[CompletedEvent],
        oracle_events: list[CompletedOracleEvent],
    ) -> Judgment:
        """
        Check if the agent call the same tools the same number of times as the oracle
        except for sending a message to the user where extra calls may be allowed
        """

        def get_count(events: list[CompletedEvent]):
            aui_name = "AgentUserInterface__send_message_to_user"
            counter = Counter([e.tool_name for e in events])
            aui_count = counter[aui_name]
            counter[aui_name] = 0
            return counter, aui_count

        agent_counter, agent_aui_count = get_count(agent_events)
        oracle_counter, oracle_aui_count = get_count(oracle_events)  # type: ignore
        return self.check_tool_call_counts(
            agent_counter,
            agent_aui_count,
            oracle_counter,
            oracle_aui_count,
            self.config.extra_send_message_to_user_allowed,
        )

    def _match_env_oracle_event(
        self,
        oracle_event: CompletedOracleEvent,
    ) -> Judgment:
        """
        Matching an env or user event. It is easy we only need to check if the event id matches.
        """
        matched = False
        for agent_event in self.state.current_turn_agent_events:
            matched = self.event_judges[oracle_event.event_type](
                agent_event, oracle_event
            )
            if matched:
                self.state.add_match(
                    agent_idx=self.state.agent_events.index(agent_event),
                    oracle_id=oracle_event.event_id,
                )
                logger.info(
                    f"JUDGE ACCEPT: Oracle event matched with {agent_event.event_id}"
                )
                return Judgment(success=True)
        logger.error(
            "JUDGE REJECT: Env oracle event not matched this is not normal!!!!"
        )
        return Judgment(
            success=False,
            failure=EnvOracleMatchingFailure(
                oracle_event_id=oracle_event.event_id,
            ),
        )

    def _match_agent_oracle_event(
        self,
        oracle_event: CompletedOracleEvent,
    ) -> Judgment:
        """
        Match an agent oracle event. This is the most complex case.
        """
        # Get the tool name
        oracle_tool_name = oracle_event.tool_name  # type: ignore
        # If there is no tool judge for this tool impossible to judge
        if oracle_tool_name not in self.event_judges[EventType.AGENT].tool_judges:
            raise ValueError(f"Tool {oracle_tool_name} not supported")
        # Search for a matching agent event
        matched = False
        failures = []
        for i, agent_event in enumerate(self.state.agent_events):
            # Compare agent event to oracle event
            agent_tool_name = agent_event.tool_name
            # If agent event not already matched to another oracle event call the tool judge
            if i not in self.state.agent_idx_to_oracle_id:
                judge_kwargs = self.get_judge_kwargs(oracle_event)
                matched = self.event_judges[EventType.AGENT](
                    agent_event, oracle_event, **judge_kwargs
                )
                if not matched and oracle_tool_name == agent_tool_name:
                    failures.append(
                        EventComparisonFailure(
                            oracle_tool_name=oracle_tool_name,
                            oracle_event_id=oracle_event.event_id,
                            agent_tool_name=agent_tool_name,
                            agent_event_id=agent_event.event_id,
                            failure_type=EventComparisonFailureType.TOOL_JUDGE_REJECT,
                        )
                    )
            else:
                logger.info(
                    "JUDGE REJECT: Agent event already matched to another oracle event"
                )
                failures.append(
                    EventComparisonFailure(
                        oracle_tool_name=oracle_tool_name,
                        oracle_event_id=oracle_event.event_id,
                        agent_tool_name=agent_tool_name,
                        agent_event_id=agent_event.event_id,
                        failure_type=EventComparisonFailureType.ALREADY_MATCHED,
                    )
                )
            # Check causality if matched
            if matched:
                for parent_id in self.state.current_turn_oracle_graph[
                    oracle_event.event_id
                ]:
                    if (
                        parent_id not in self.state.oracle_id_to_agent_idx
                        or self.state.oracle_id_to_agent_idx[parent_id] >= i
                    ):
                        logger.info("JUDGE REJECT: Causality violation")
                        failures.append(
                            EventComparisonFailure(
                                oracle_tool_name=oracle_tool_name,
                                oracle_event_id=oracle_event.event_id,
                                agent_tool_name=agent_tool_name,
                                agent_event_id=agent_event.event_id,
                                failure_type=EventComparisonFailureType.CAUSALITY,
                            )
                        )
                        matched = False
            logger.info(f"Matched: {matched}")
            # If matched, update matching events
            if matched:
                self.state.add_match(agent_idx=i, oracle_id=oracle_event.event_id)
                return Judgment(success=True)
        # No match found
        logger.info(f"JUDGE REJECT: cannot match {oracle_tool_name} oracle event")
        oracle_args = oracle_event.get_args()
        return Judgment(
            success=False,
            failure=OracleEventMatchingFailure(
                oracle_tool_name=oracle_tool_name,
                oracle_tool_args={
                    k: str(v) for k, v in oracle_args.items() if k != "self"
                },
                comparison_failures=failures,
            ),
        )

    def match_oracle_event(self, oracle_event: CompletedOracleEvent) -> Judgment:
        """
        Match an oracle event with an agent event.
        """

        trace_type = (
            f"matching_{oracle_event.event_type.name}_tool_{oracle_event.tool_name}"
        )

        @injected_traceable(trace_type=trace_type, tags=["judge"], log_input_args=False)
        def inner_call(self, oracle_event: CompletedOracleEvent) -> Judgment:
            if oracle_event.event_type in [EventType.ENV, EventType.USER]:
                # If the oracle event is an env or user event
                return self._match_env_oracle_event(oracle_event)
            elif oracle_event.event_type == EventType.AGENT:
                # If the oracle event is an agent event
                return self._match_agent_oracle_event(oracle_event)
            else:
                raise ValueError(f"Event type {oracle_event.event_type} not supported")

        return inner_call(self, oracle_event)

    @injected_traceable(trace_type="judge", tags=["judge"], log_input_args=False)
    def __call__(self, env: AbstractEnvironment) -> Judgment:
        judgment = self.inner_call(env)
        judgment.agent_event_id_to_oracle_event_id = self.state.agent_id_to_oracle_id
        judgment.failure = str(judgment.failure)  # Only for logging purpose
        if not judgment.success:
            logger.info(f"JUDGE REJECT: {judgment.failure}")
        # Update state
        self.state.last_turn_success = bool(judgment.success)
        self.state.last_turn_rationale = str(judgment.failure)
        return judgment

    def inner_call(self, env: AbstractEnvironment) -> Judgment:
        # Update state
        self.update_state(env)
        # Extract agent events
        agent_events = self.state.current_turn_agent_events
        # Extract oracle events
        oracle_events = self.state.current_turn_oracle_events
        # Prints
        logger.info(
            "\nFOUND AGENT EVENTS:\n" + str([e.tool_name for e in agent_events])
        )
        logger.info(
            "\nFOUND ORACLE EVENTS:\n" + str([e.tool_name for e in oracle_events])
        )
        # Preliminary checks
        preliminary_checks = self.preliminary_checks(agent_events, oracle_events)
        if not preliminary_checks.success:
            logger.info("JUDGE REJECT: Agent events do not match oracle events")
            return preliminary_checks

        for oracle_event in oracle_events:
            # Resolve arg placeholders in the event.
            oracle_event = resolve_arg_placeholders(
                event=oracle_event,
                event_id_to_target_event_idx=self.state.oracle_id_to_agent_idx,
                target_events=self.state.agent_events,
            )
            # Try to match the oracle event
            oracle_event = cast(CompletedOracleEvent, oracle_event)
            matched = self.match_oracle_event(oracle_event)
            # If unable to judge, return False
            if not matched.success:
                return Judgment(
                    success=False,
                    failure=matched.failure,
                )
        # All oracle events matched (up to some extra message to the user allowed for the agent)
        return Judgment(success=True)
