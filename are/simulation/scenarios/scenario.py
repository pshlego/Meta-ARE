# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import datetime
import json
import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Type, TypeVar, cast

from are.simulation.apps import INTERNAL_APPS
from are.simulation.apps.app import App
from are.simulation.scenarios.utils.scenario_expander import (
    EnvEventsConfig,
    EnvEventsExpander,
)
from are.simulation.scenarios.utils.turn_conditions import condition_from_name
from are.simulation.scenarios.validation_result import ScenarioValidationResult
from are.simulation.tool_utils import AppTool
from are.simulation.types import (
    AbstractEnvironment,
    AbstractEvent,
    ActionDescription,
    CapabilityTag,
    CompletedEvent,
    ConditionCheckEvent,
    EnvironmentState,
    Event,
    EventTimeComparator,
    EventType,
    Hint,
    HintType,
    OracleEvent,
    ScenarioGUIConfig,
    ToolAugmentationConfig,
)
from are.simulation.utils import EnumEncoder

logger = logging.getLogger(__name__)


SPECIAL_RULE_FOR_SEND_MESSAGE_TO_USER = True
ORACLE_EVENT_DEPENDENCY_REQUIRED = True
SPECIAL_RULE_FOR_SEND_MESSAGE_EVENTS = True
ENVIRONMENT_EVENT_DEPENDENCY_REQUIRED = True
EVENT_TIME_VALIDATION_REQUIRED = True


class AutoDataclass(type):
    def __new__(cls, name, bases, namespace):
        # Create the new class
        new_cls = super().__new__(cls, name, bases, namespace)
        # Apply the dataclass decorator to the new class
        try:
            new_cls = dataclass(new_cls)  # type: ignore
        except TypeError as e:
            logger.error(
                f"Error applying dataclass decorator to {name}/{namespace}: {e}"
            )
            raise e
        return new_cls


class ScenarioStatus(Enum):
    Draft = "Draft"
    AwaitingReview = "AwaitingReview"
    Valid = "Valid"
    Invalid = "Invalid"
    Abandoned = "Abandoned"
    Safe = "safe"
    Violating = "violating"


T = TypeVar("T", bound=App)


class Scenario(metaclass=AutoDataclass):
    # Scenario internals
    _initialized: bool = field(default=False)

    is_benchmark_ready: bool = field(
        default=False
    )  # Is the scenario ready for benchmarking
    events: list[AbstractEvent] = field(default_factory=list)
    apps: list[App] | None = field(default=None)
    tags: tuple[CapabilityTag, ...] = field(
        default_factory=tuple
    )  # Tags to describe the scenario

    scenario_id: str = field(default="")
    seed: int = field(default=0)
    nb_turns: int | None = field(default=None)
    run_number: int | None = field(
        default=None
    )  # Run number for multiple runs of the same scenario

    config: str | None = field(default=None)
    has_a2a_augmentation: bool = field(default=False)

    # Annotation

    status: ScenarioStatus = field(default=ScenarioStatus.Draft)
    comment: str | None = field(default=None)
    annotation_id: str | None = field(default=None)
    hints: list[Hint] | None = field(default=None)

    additional_system_prompt: str | None = field(default=None)

    start_time: float | None = field(
        default_factory=lambda: datetime.datetime.now().timestamp()
    )
    duration: float | None = field(default=None)
    queue_based_loop: bool = field(default=False)
    time_increment_in_seconds: int = field(default=1)
    working_dir: str = field(default="")

    # A preserved copy of apps in their initial state.
    _initial_apps: dict[str, Any] | None = field(default=None)

    # Provides configuration to augment the tools. E.g. change the probability of failure
    tool_augmentation_config: ToolAugmentationConfig | None = field(default=None)
    # Provides configuration to augment the scenario with random ENV events
    env_events_config: EnvEventsConfig | None = field(default=None)
    # GUI specific configuration
    gui_config: ScenarioGUIConfig | None = field(default=None)

    # Augmentation data for tools.
    augmentation_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.events is None:
            self.events = []
        if self.apps is None:
            self.apps = []
        if self.hints is None:
            self.hints = []

        # Copy the class's scenario_id to the instance if the instance's scenario_id is empty
        if not self.scenario_id and hasattr(self.__class__, "scenario_id"):
            self.scenario_id = getattr(self.__class__, "scenario_id")

        assert self.scenario_id is not None, "scenario_id is mandatory."

    def initialize(self, *args, **kwargs) -> None:
        if self._initialized:
            return

        # Initialize apps with the context
        self.init_and_populate_apps(*args, **kwargs)

        # Set the seed for each app
        if self.apps is not None:
            for app in self.apps:
                app.set_seed(self.seed)

        self.apply_augmentation_configs()

        # Preserve the initial state of the apps.
        self._initial_apps = {
            app.name: {
                "class_name": app.__class__.__name__,
                "serialized_state": json.dumps(app.get_state(), cls=EnumEncoder),
            }
            for app in self.apps or []
        }

        self.build_events_flow()

        if self.env_events_config is not None:
            expander = EnvEventsExpander(env_events_config=self.env_events_config)
            expander.add_env_events_to_scenario(scenario=self)

        self._initialized = True

    def soft_reset(self):
        for app in self.apps or []:
            failure_probability = app.failure_probability
            name = app.name
            app.reset()
            app.name = name
            if self._initial_apps and app.name in self._initial_apps:
                app.load_state(
                    json.loads(self._initial_apps[app.name]["serialized_state"])
                )
            if failure_probability is not None:
                app.set_failure_probability(failure_probability)

        self.apply_augmentation_configs()

        if self.events:
            for event in self.events:
                if event.event_relative_time is not None:
                    event.event_time = None

        # Set the seed for each app
        if self.apps is not None:
            for app in self.apps:
                app.set_seed(self.seed)

    def reset_apps(self, new_apps):
        logger.warning(
            f"Hard resetting scenario apps to {new_apps}. This will erase any apps previously registered to the env."
        )
        self.apps = new_apps

    def init_and_populate_apps(self, *args, **kwargs) -> None:
        """
        Initialize the apps that will be used in the Scenario.
        """

    def build_events_flow(self) -> None:
        """
        Core logic of the scenario, this is where the scenario is built.
        Where events are scheduled, event triggers are defined, as well as any element of the task.
        By default, this function is empty, and should be overridden by the scenario if any extra logic is needed.
        """

    def get_app(self, app_name: str) -> App:
        """
        Get the app with the given name
        """
        for app in self.apps or []:
            if app.name == app_name:
                return app
        raise ValueError(f"App {app_name} not found in scenario.")

    def get_typed_app(self, app_type: Type[T], app_name: str | None = None) -> T:
        """
        Get the app with the given type and optional name.
        If name is not provided, it will be inferred from the app type.
        """
        name = app_name or app_type.__name__
        for app in self.apps or []:
            if isinstance(app, app_type) and app.name == name:
                return cast(T, app)
        raise ValueError(
            f"App {name} of type {app_type.__name__} not found in scenario."
        )

    def get_tools_by_app(self) -> dict[str, list[AppTool]]:
        """
        Get for each app, the list of tools it has.
        """
        apps_to_skip = set(app.name for app in INTERNAL_APPS)
        return {
            app.name: app.get_tools()
            for app in self.apps or []
            if app.name not in apps_to_skip
        }

    def get_tools(self) -> list[AppTool]:
        """
        Get the entire list of tools from all the apps.
        """
        return [
            item for sublist in self.get_tools_by_app().values() for item in sublist
        ]

    def get_user_tools_by_app(self) -> dict[str, list[AppTool]]:
        """
        Get for each app, the list of tools it has that are accessible to the User.
        """
        return {app.name: app.get_user_tools() for app in self.apps or []}

    def get_user_tools(self) -> list[AppTool]:
        """
        Get the entire list of User tools from all the apps.
        """
        return [
            item
            for sublist in self.get_user_tools_by_app().values()
            for item in sublist
        ]

    def is_send_message_to_user(self, event: AbstractEvent) -> bool:
        """
        Check if the event is a send message to user event
        """
        return (
            isinstance(event, OracleEvent)
            and event.action_desc
            and event.action_desc.app == "AgentUserInterface"
            and event.action_desc.function == "send_message_to_user"
        ) or (
            isinstance(event, CompletedEvent)
            and event.event_type == EventType.AGENT
            and event.action
            and event.action.function_name == "send_message_to_user"
            and event.action.class_name == "AgentUserInterface"
        )

    def build_event_id_to_turn_idx(self):
        """
        Build a dictionary to store the turn of each event
        The turn of an event is the number of event send_message_to_user among its ancestors
        The dictionary and the number of turns are stored to the scenario
        """
        if self.events is None:
            raise ValueError("Events not found")
        visited_events = set()
        # Initialize a dictionary to store the turn of each event
        event_id_to_turn_idx = defaultdict(lambda: 0)
        # Initialize a queue with event without dependencies
        queue = deque([e for e in self.events if not e.dependencies])
        # Track the beginning of each turn
        turn_beginnings = {}

        # Perform BFS on events
        while queue:
            event = queue.popleft()
            # If the event has not been visited yet
            if event.event_id not in visited_events:
                event_id_to_turn_idx[event.event_id] = 0
                # Mark it as visited
                visited_events.add(event.event_id)
                if event.dependencies:
                    event_id_to_turn_idx[event.event_id] = max(
                        [event_id_to_turn_idx[e.event_id] for e in event.dependencies],
                    )
                    # Check if there is a send message to user event among the dependencies
                    if any(
                        self.is_send_message_to_user(e)
                        for e in event.dependencies  # type: ignore
                    ):
                        event_id_to_turn_idx[event.event_id] += 1
                # If this event starts a new turn, add it to turn_beginnings
                turn_number = event_id_to_turn_idx[event.event_id]
                if turn_number not in turn_beginnings:
                    turn_beginnings[turn_number] = event.event_id
                # Add the successors to the queue
                queue.extend(
                    e for e in event.successors if e.event_id not in visited_events
                )
        # Convert the defaultdict back to a regular dictionary
        event_id_to_turn_idx = dict(event_id_to_turn_idx)

        return event_id_to_turn_idx

    def get_env_tools_by_app(self) -> dict[str, list[AppTool]]:
        """
        Get for each app, the list of tools it has that are accessible to the Environment.
        """
        return {app.name: app.get_env_tools() for app in self.apps or []}

    def get_env_tools(self) -> list[AppTool]:
        """
        Get the entire list of Env tools from all the apps.
        """
        return [
            item for sublist in self.get_env_tools_by_app().values() for item in sublist
        ]

    def get_data_tools_by_app(self) -> dict[str, list[AppTool]]:
        """
        Get for each app, the list of tools it has that are accessible to the Data.
        """
        return {app.name: app.get_data_tools() for app in self.apps or []}

    def get_data_tools(self) -> list[AppTool]:
        """
        Get the entire list of Data tools from all the apps.
        """
        return [
            item
            for sublist in self.get_data_tools_by_app().values()
            for item in sublist
        ]

    def _add_hint(self, event_id: str, hint_type: HintType):
        """
        Add a placeholder hint to the scenario when import trace doesn't provided the hint content.
        """
        if not self.hints:
            self.hints = []
        if event_id not in [hint.associated_event_id for hint in self.hints]:
            self.hints.append(
                Hint(
                    **{
                        "hint_type": HintType(hint_type),
                        "content": "",
                        "associated_event_id": event_id,
                    }
                )
            )

    def edit_hint_content(self, event_id: str, content: str):
        """
        Edit the content of a hint.
        """
        for hint in self.hints or []:
            if hint.associated_event_id == event_id:
                hint.content = content
                return
        raise ValueError(f"Hint with event_id {event_id} not found.")

    def _delete_hint(self, event_id: str):
        """
        Delete a hint from the scenario.
        """
        self.hints = [
            hint for hint in self.hints or [] if hint.associated_event_id != event_id
        ]

    def validate(self, env: AbstractEnvironment) -> ScenarioValidationResult:
        """
        Validate the state of the environment after the scenario has been executed.
        """
        try:
            env.final_validation_checks()
        except Exception as e:
            return ScenarioValidationResult(success=False, exception=e)

        success = env.state != EnvironmentState.FAILED
        return ScenarioValidationResult(success=success)

    def set_duration(self, duration: float | None):
        """
        Set the duration of the scenario.
        """
        self.duration = duration

    def set_time_increment(self, time_increment_in_seconds: int):
        """
        Set the time increment of the scenario.
        """
        self.time_increment_in_seconds = time_increment_in_seconds

    def delete_completed_events(self) -> None:
        self.events.clear()
        if self.hints:
            self.hints.clear()

    def delete_event(self, event_id: str) -> None:
        # only delete the event itself, not the chain of events that follow it
        # go through all events and remove any item in successor or descendant that matches the events_to_delete
        for event in self.events:
            event.successors = [
                successor
                for successor in event.successors
                if successor.event_id != event_id
            ]
            event.dependencies = [
                dependency
                for dependency in event.dependencies
                if dependency.event_id != event_id
            ]
        # Remove any matching event_id node
        self.events = [e for e in self.events if e.event_id != event_id]
        self._delete_hint(event_id)

    def _is_event_send_message_to_agent(self, event: AbstractEvent) -> bool:
        """
        Checks if the current event is sending a message to an agent.
        :param The current event object.
        :return: True if the current event is sending a message to an agent, False otherwise.
        """
        return (
            isinstance(event, Event)
            and event.function_name() == "send_message_to_agent"
        )

    def _is_event_send_message_to_user(self, event: AbstractEvent) -> bool:
        """
        Checks if the predecessor event is sending a message to a user.
        :param The predecessor event object.
        :return: True if the predecessor event is sending a message to a user, False otherwise.
        """
        return (
            isinstance(event, OracleEvent)
            and event.action_desc is not None
            and event.action_desc.function == "send_message_to_user"
        )

    def validate_predecessors(
        self,
        predecessor_event_ids: list[str],
        function_name: str,
        event_type: EventType,
        events: list[AbstractEvent],
    ):
        if (
            ORACLE_EVENT_DEPENDENCY_REQUIRED
            and not predecessor_event_ids
            and event_type == EventType.AGENT
        ):
            raise ValueError("Agent events must have at least one predecessor event.")

        if ENVIRONMENT_EVENT_DEPENDENCY_REQUIRED and event_type == EventType.ENV:
            if len(predecessor_event_ids) != 1:
                raise ValueError("Env events must have only one predecessor event")

            if len(predecessor_event_ids) == 1:
                predecessor = next(
                    (e for e in events if e.event_id == predecessor_event_ids[0]),
                    None,
                )
                if predecessor is None:
                    raise ValueError(
                        f"Predecessor event with id '{predecessor_event_ids[0]}' not found."
                    )
                if (
                    not self._is_event_send_message_to_user(predecessor)
                    and not predecessor.event_type == EventType.ENV
                    and not predecessor.event_type == EventType.USER
                    and not predecessor.event_type == EventType.CONDITION
                ):
                    raise ValueError(
                        "Env events can only have a send_message_to_agent or user/env event as a predecessor event."
                    )

        for predecessor_event_id in predecessor_event_ids:
            predecessor = next(
                (e for e in events if e.event_id == predecessor_event_id),
                None,
            )
            if predecessor is None:
                raise ValueError(
                    f"Predecessor event with id '{predecessor_event_id}' not found."
                )
            if SPECIAL_RULE_FOR_SEND_MESSAGE_TO_USER:
                if self._is_event_send_message_to_user(predecessor) and not (
                    event_type == EventType.ENV
                    or function_name == "send_message_to_agent"
                ):
                    raise ValueError(
                        f"The {function_name} event is not allowed to link after send_message_to_user, "
                        "only send_message_to_agent or Env event is allowed."
                    )

    def filter_connected_events(
        self, predecessor_event_ids: list[str], events: list[AbstractEvent]
    ) -> list[AbstractEvent]:
        """
        Filters out all events connected with the given predecessor_event_ids,
        including the predecessor events themselves, their successors and dependencies.
        :param predecessor_event_ids: List of event IDs to start filtering from.
        :param events: List of all events.
        :return: List of events excluding those connected with the predecessor_event_ids.
        """
        # Create a dictionary for quick lookup of events by their ID
        event_dict: dict[str, AbstractEvent] = {
            event.event_id: event for event in events
        }

        # Set to keep track of all connected event IDs
        connected_event_ids = set(predecessor_event_ids)

        # Queue for breadth-first search (BFS) to find all connected events
        queue = list(predecessor_event_ids)

        while queue:
            current_event_id = queue.pop(0)
            current_event = event_dict.get(current_event_id)

            if current_event:
                # Add successors to the queue if they are not already processed
                for successor in current_event.successors:
                    if successor.event_id not in connected_event_ids:
                        connected_event_ids.add(successor.event_id)
                        queue.append(successor.event_id)

                # Add dependencies to the queue if they are not already processed
                for dependency in current_event.dependencies:
                    if dependency.event_id not in connected_event_ids:
                        connected_event_ids.add(dependency.event_id)
                        queue.append(dependency.event_id)
        # Filter out the connected events from the original list
        filtered_events = [
            event for event in events if event.event_id not in connected_event_ids
        ]

        return filtered_events

    def validate_events_dag_aui_single_branch(
        self,
        predecessor_event_ids: list[str],
        function_name: str,
        events: list[AbstractEvent],
        event_id: str | None = None,
    ):
        if not SPECIAL_RULE_FOR_SEND_MESSAGE_EVENTS:
            return

        # In multi-turn scenarios (those containing CONDITION events),
        # each turn legitimately has its own AUI events. Skip the
        # single-branch check when CONDITION events are present.
        if any(
            isinstance(event, ConditionCheckEvent) for event in events
        ):
            return

        # All the AUI events should be in a single branch of the events graph
        if (
            function_name == "send_message_to_agent"
            or function_name == "send_message_to_user"
        ):
            if len(predecessor_event_ids) > 0:
                events = self.filter_connected_events(predecessor_event_ids, events)
            # Filter out the single event note from edit use case
            if event_id is not None:
                events = self.filter_connected_events([event_id], events)
            for event in events:
                if self._is_event_send_message_to_agent(
                    event
                ) or self._is_event_send_message_to_user(event):
                    raise ValueError(
                        "Only one branch of the events graph should contain send_message_to_agent or send_message_to_user events. "
                        "Found multiple branches with send_message_to_agent or send_message_to_user events."
                    )

    def accumulate_times_from_event(
        self,
        events: list[AbstractEvent],
        event_id: str | None = None,
        new_event_time: float | None = None,
        new_event_relative_time: float | None = None,
    ) -> dict[str, float]:
        accumulated_times = {event.event_id: 0.0 for event in events}

        # Process events in the order they appear in events
        for event in events:
            max_predecessor_time = max(
                (
                    accumulated_times.get(pred.event_id, 0)
                    for pred in event.dependencies
                ),
                default=self.start_time or 0,
            )

            if event_id is not None and event.event_id == event_id:
                if new_event_time is not None:
                    if new_event_time < max_predecessor_time:
                        raise ValueError(
                            f"Event {event.event_id} has an absolute time of {new_event_time}, which is less than the maximum predecessor time of {max_predecessor_time}."
                        )
                    accumulated_times[event.event_id] = new_event_time
                else:
                    accumulated_times[event.event_id] = max_predecessor_time + (
                        new_event_relative_time or 0
                    )
            else:
                if event.event_time is not None:
                    if event.event_time < max_predecessor_time:
                        raise ValueError(
                            f"Event {event.event_id} has an absolute time of {event.event_time}, which is less than the maximum predecessor time of {max_predecessor_time}."
                        )
                    accumulated_times[event.event_id] = event.event_time
                else:
                    accumulated_times[event.event_id] = max_predecessor_time + (
                        event.event_relative_time or 0
                    )

        return accumulated_times

    def validate_events_dag_message_to_user_time(
        self,
        predecessor_event_ids: list[str],
        function_name: str,
        new_event_relative_time: float | None = None,
        new_event_time: float | None = None,
        event_id: str | None = None,
    ):
        if not EVENT_TIME_VALIDATION_REQUIRED:
            return

        if not predecessor_event_ids:
            return  # Skip if there are no predecessor_event_ids

        events = self.events
        events_map = {e.event_id: e for e in events}

        event_id_to_turn_idx = self.build_event_id_to_turn_idx()

        # Find the turn number for the given predecessor_event_ids
        predecessor_turns = [
            event_id_to_turn_idx[pred_id] for pred_id in predecessor_event_ids
        ]
        if not predecessor_turns:
            raise ValueError("No valid predecessor turns found.")

        # Ensure all predecessor turns are the same
        if len(set(predecessor_turns)) != 1:
            raise ValueError("Predecessor events do not belong to the same turn.")

        # Get the turn index for the predecessors
        turn_index = predecessor_turns[0]
        events_in_turn = [
            events_map[event_id]
            for event_id, turn in event_id_to_turn_idx.items()
            if turn == turn_index
        ]

        # If all events in turn have relative time of 0 or 1, skip validation (time is not important)
        if events_in_turn and all(
            (event.event_relative_time == 1 or event.event_relative_time == 0)
            for event in events_in_turn
        ):
            return

        # Step 3: Data Storage and Accumulation
        accumulated_times = self.accumulate_times_from_event(
            events_in_turn, event_id, new_event_time, new_event_relative_time
        )

        # Step 4: Evaluate New Event
        max_predecessor_time = max(
            accumulated_times.get(pred_id, 0) for pred_id in predecessor_event_ids
        )
        if function_name == "send_message_to_user":
            max_time_value = max(accumulated_times.values())

            if new_event_relative_time:
                # Check if the new event's relative time exceeds the maximum time value
                if max_predecessor_time + new_event_relative_time < max_time_value:
                    raise ValueError(
                        f"send_message_to_user event's time ({max_predecessor_time + new_event_relative_time}) should be the maximum of the turn ({max_time_value})."
                    )

            elif new_event_time:
                # Check if the new event's time exceeds the maximum time value
                if new_event_time < max_time_value:
                    raise ValueError(
                        f"send_message_to_user event's time ({new_event_time}) should be the maximum of the turn ({max_time_value})."
                    )

        else:
            send_message_to_user_event = next(
                (
                    event
                    for event in events_in_turn
                    if self.is_send_message_to_user(event)
                ),
                None,
            )

            if (
                send_message_to_user_event is not None
                and send_message_to_user_event.event_id not in predecessor_event_ids
            ):
                send_message_to_user_event_time = accumulated_times.get(
                    send_message_to_user_event.event_id, 0
                )

                if new_event_relative_time:
                    # Check if the new event's relative time exceeds the maximum time value
                    if (
                        send_message_to_user_event_time
                        < max_predecessor_time + new_event_relative_time
                    ):
                        raise ValueError(
                            f"New event's time ({max_predecessor_time + new_event_relative_time}) exceeds send_message_to_user event's time ({send_message_to_user_event_time})."
                        )

                elif new_event_time:
                    # Check if the new event's time exceeds the maximum time value
                    if send_message_to_user_event_time < new_event_time:
                        raise ValueError(
                            f"New event's time ({new_event_time}) exceeds send_message_to_user event's time ({send_message_to_user_event_time})."
                        )

    def _setup_event_dependencies(
        self,
        new_event: AbstractEvent,
        predecessor_event_ids: list[str],
    ) -> None:
        for predecessor_event_id in predecessor_event_ids:
            predecessor_event = next(
                (e for e in self.events if e.event_id == predecessor_event_id),
                None,
            )
            assert predecessor_event is not None, "couldn't find predecessor"
            new_event.dependencies.append(predecessor_event)
            predecessor_event.successors.append(new_event)

    def add_event(
        self,
        app_name: str,
        function_name: str,
        parameters: dict[str, Any],
        predecessor_event_ids: list[str],
        event_type: EventType,
        event_id: str | None = None,
        event_relative_time: float | None = None,
        event_time: float | None = None,
        event_time_comparator: EventTimeComparator | None = None,
    ) -> AbstractEvent:
        if event_type not in [
            EventType.AGENT,
            EventType.ENV,
            EventType.USER,
            EventType.CONDITION,
        ]:
            raise ValueError("event_type must be one of AGENT, ENV, USER, CONDITION")

        # Validation
        self.validate_predecessors(
            predecessor_event_ids, function_name, event_type, self.events
        )
        self.validate_events_dag_aui_single_branch(
            predecessor_event_ids, function_name, self.events
        )
        self.validate_events_dag_message_to_user_time(
            predecessor_event_ids,
            function_name,
            event_relative_time,
            event_time,
        )

        new_event = self._create_event(
            app_name,
            function_name,
            parameters,
            event_type,
            event_id,
            event_relative_time,
            event_time,
            event_time_comparator,
        )

        self._setup_event_dependencies(new_event, predecessor_event_ids)

        self.events.append(new_event)

        # Hints
        if function_name == "send_message_to_agent":
            self._add_hint(new_event.event_id, HintType.TASK_HINT)
        if new_event.event_type == EventType.ENV:
            self._add_hint(new_event.event_id, HintType.ENVIRONMENT_HINT)

        return new_event

    def edit_event(
        self,
        app_name: str,
        function_name: str,
        parameters: dict[str, Any],
        event_id: str,
        event_type: EventType,
        predecessor_event_ids: list[str],
        event_relative_time: float | None = None,
        event_time: float | None = None,
        event_time_comparator: EventTimeComparator | None = None,
    ) -> AbstractEvent:
        if event_type not in [EventType.AGENT, EventType.ENV, EventType.USER]:
            raise ValueError("event_type must be one of AGENT, ENV, USER")

        current_event = next((e for e in self.events if e.event_id == event_id), None)
        if current_event is None:
            raise ValueError(f"Current event with id '{event_id}' not found.")

        # Validation
        self.validate_predecessors(
            predecessor_event_ids, function_name, event_type, self.events
        )
        self.validate_events_dag_aui_single_branch(
            predecessor_event_ids, function_name, self.events, event_id
        )
        self.validate_events_dag_message_to_user_time(
            predecessor_event_ids,
            function_name,
            event_relative_time,
            event_time,
            event_id,
        )

        new_event = self._create_event(
            app_name,
            function_name,
            parameters,
            event_type,
            None,
            event_relative_time,
            event_time,
            event_time_comparator,
        )

        if new_event.event_type == current_event.event_type:
            # Only keep the event_id if it's the same type
            new_event.event_id = current_event.event_id

        new_event.successors = current_event.successors
        for successor in current_event.successors:
            successor.dependencies.remove(current_event)
            successor.dependencies.append(new_event)
        for dependency in current_event.dependencies:
            dependency.successors.remove(current_event)

        self._setup_event_dependencies(new_event, predecessor_event_ids)

        self.events.remove(current_event)
        self.events.append(new_event)

        # Hints
        is_current_event_send_message_to_agent = self._is_event_send_message_to_agent(
            current_event
        )
        if (
            is_current_event_send_message_to_agent
            and function_name != "send_message_to_agent"
        ):
            self._delete_hint(current_event.event_id)
        elif (
            not is_current_event_send_message_to_agent
            and function_name == "send_message_to_agent"
        ):
            self._add_hint(new_event.event_id, HintType.TASK_HINT)
        if current_event.event_type == EventType.ENV:
            self._delete_hint(current_event.event_id)
        if new_event.event_type == EventType.ENV:
            self._add_hint(new_event.event_id, HintType.ENVIRONMENT_HINT)

        return new_event

    def process_events(self, events):
        graph = defaultdict(list)
        in_degree = defaultdict(int)

        # Build a set of all event IDs that actually exist
        all_event_ids = {event.event_id for event in events}

        # Build a mapping from condition base IDs (without CHECK_N suffix)
        # to their actual event IDs. During runtime, ConditionCheckEvent
        # increments _internal_check_count which changes the ID from e.g.
        # "condition_turn_1-CHECK_0" to "condition_turn_1-CHECK_649".
        # Dependent events may reference the runtime ID, so we need to
        # normalize them back to the exported base ID.
        condition_base_to_id: dict[str, str] = {}
        for event in events:
            if event.event_type == "CONDITION":
                base_id = re.sub(r"-CHECK_\d+$", "", event.event_id)
                condition_base_to_id[base_id] = event.event_id

        def _normalize_dep(dep: str) -> str:
            """Normalize a dependency ID, resolving runtime condition IDs."""
            if dep in all_event_ids:
                return dep
            base = re.sub(r"-CHECK_\d+$", "", dep)
            if base in condition_base_to_id:
                return condition_base_to_id[base]
            return dep

        for event in events:
            event_id = event.event_id
            dependencies = [_normalize_dep(d) for d in event.dependencies]
            # Update the event's dependencies with normalized values
            event.dependencies = dependencies
            in_degree[event_id] = in_degree.get(event_id, 0)

            for dep in dependencies:
                graph[dep].append(event_id)
                in_degree[event_id] += 1

        # Queue to manage the processing order of events (initialized with events that have no dependencies)
        event_queue = deque(
            [event for event in events if in_degree[event.event_id] == 0]
        )

        while event_queue:
            event_data = event_queue.popleft()
            event_id = event_data.event_id
            app_name = ""
            function_name = ""
            parameters = {}
            if event_data.action:
                app_name = event_data.action.app
                function_name = event_data.action.function
                parameters = (
                    {
                        arg.name: {"value": arg.value, "type": arg.value_type}
                        for arg in event_data.action.args
                    }
                    if event_data.action.args
                    else {}
                )

            # Get the predecessor event IDs if any
            predecessor_event_ids = [
                dependency for dependency in event_data.dependencies
            ]

            event_type = getattr(EventType, event_data.event_type)
            event_relative_time = event_data.event_relative_time
            event_time = event_data.event_time if event_relative_time is None else None

            # Handle event_time_comparator for ExportedOracleEvent
            event_time_comparator_value = getattr(
                event_data, "event_time_comparator", None
            )
            try:
                event_time_comparator = (
                    EventTimeComparator(event_time_comparator_value)
                    if event_time_comparator_value
                    else None
                )
            except ValueError:
                logger.warning(
                    f"Invalid event_time_comparator value: {event_time_comparator_value}"
                )
                event_time_comparator = None

            # Create a new event
            self.add_event(
                app_name=app_name,
                function_name=function_name,
                parameters=parameters,
                predecessor_event_ids=predecessor_event_ids,
                event_type=event_type,
                event_id=event_id,
                event_relative_time=event_relative_time,
                event_time=event_time,
                event_time_comparator=event_time_comparator,
            )

            # Enqueue events that are dependent on this event
            for dependent_event_id in graph[event_id]:
                in_degree[dependent_event_id] -= 1
                if in_degree[dependent_event_id] == 0:
                    dependent_event = next(
                        e for e in events if e.event_id == dependent_event_id
                    )
                    event_queue.append(dependent_event)

        if len(self.events) != len(events):
            raise ValueError(
                f"An error occurred while processing the events, please check the logs (got {len(self.events)} events, expected {len(events)} events)."
            )

    @staticmethod
    def _has_event_reference_pattern(s):
        pattern = r"^\{\{.*\}\}$"
        return bool(re.match(pattern, s))

    @staticmethod
    def _type_allows_none(type_str: str) -> bool:
        # Check for Optional[T].
        if re.match(r"^Optional\[(.*)\]$", type_str):
            return True
        pipeSeparatedTypes = [
            # Split by the pipe symbol on the top level only and then trim.
            m.strip()
            for m in re.split(r"(?![^\[\]\(\)\<\>]*[\]\)\>\]])\|", type_str)
        ]
        # Check for T | None.
        return len(pipeSeparatedTypes) > 1 and "None" in pipeSeparatedTypes

    @staticmethod
    def _parse_parameter_value(value: str, type_str: str) -> Any:
        if type_str is None and value is None:
            return None

        if value is None or value == "":
            if Scenario._type_allows_none(type_str):
                return None
            raise ValueError(f"Non-optional type {type_str} cannot be None or empty.")

        if Scenario._has_event_reference_pattern(value):
            # We defer the conversion of argument value, it will be replaced with the return value of the event
            return value

        # In case of a Union (e.g. `str | int`) consider the first type only.
        # Check for Optional[T] first for backwards compatibility.
        match = re.match(r"Optional\[(.*)\]|\b(\w+)\b", type_str)
        if match:
            base_type = match.group(1) or match.group(2)
        else:
            raise ValueError(f"Invalid type format: {type_str}.")

        type_converters = {
            "int": int,
            "str": str,
            "float": float,
            "bool": lambda x: x.lower() in ["true", "1", "t", "y", "yes"],
            "list": json.loads,
            "dict": json.loads,
        }

        try:
            return type_converters[base_type](value)
        except KeyError:
            raise ValueError(f"Unsupported type {base_type}.")
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error converting to JSON value {value} to {base_type}: {e}."
            )
        except Exception as e:
            raise ValueError(f"Error converting value {value} to {base_type}: {e}.")

    @staticmethod
    def _create_oracle_event_factory(
        app_name: str, function_name: str, kwargs: dict[str, Any]
    ):
        def make_oracle_event(env: AbstractEnvironment) -> AbstractEvent:
            app = env.get_app(app_name)
            if app is None:
                raise ValueError(f"App '{app_name}' not found in scenario.")
            app_function = getattr(app, function_name)
            if app_function is None:
                raise ValueError(
                    f"Function '{function_name}' not found in app '{app_name}'."
                )
            return app_function(**kwargs)

        return make_oracle_event

    def _create_event(
        self,
        app_name: str,
        function_name: str,
        parameters: dict[str, Any],
        event_type: EventType,
        event_id: str | None = None,
        event_relative_time: float | None = None,
        event_time: float | None = None,
        event_time_comparator: EventTimeComparator | None = None,
    ) -> AbstractEvent:
        if event_relative_time is not None and event_relative_time < 0:
            raise ValueError("event_relative_time must be non-negative.")

        if event_time is not None and event_time < 0:
            raise ValueError("event_time must be non-negative.")

        if event_relative_time is not None and event_time is not None:
            raise ValueError(
                "event_relative_time and event_time cannot both be specified."
            )

        if event_relative_time is None and event_time is None:
            event_relative_time = 0.0

        kwargs = (
            {
                key: Scenario._parse_parameter_value(param["value"], param["type"])
                for key, param in parameters.items()
            }
            if parameters
            else {}
        )

        event = None

        if event_type == EventType.AGENT:
            action_desc_args = (
                [
                    {"name": k, "value": v["value"], "value_type": v["type"]}
                    for k, v in parameters.items()
                ]
                if parameters is not None
                else []
            )

            event = OracleEvent(
                make_event=Scenario._create_oracle_event_factory(
                    app_name, function_name, kwargs
                ),
                event_type=event_type,
                event_time_comparator=event_time_comparator,
                action_desc=ActionDescription(
                    app=app_name,
                    function=function_name,
                    args=action_desc_args,
                ),
            )

        elif event_type == EventType.ENV or event_type == EventType.USER:
            if event_time_comparator is not None:
                raise ValueError(
                    f"event_time_comparator is only supported for events of type '{EventType.AGENT}'."
                )

            app = self.get_app(app_name)
            if app is None:
                raise ValueError(f"App '{app_name}' not found.")

            app_function = getattr(app, function_name)
            if app_function is None:
                raise ValueError(
                    f"Function '{function_name}' not found in app '{app_name}'."
                )

            event = Event.from_function(
                function=app_function, event_type=event_type, **kwargs
            )

        elif event_type == EventType.CONDITION:
            # Only support condition for turn now
            execution_metadata = getattr(self, "execution_metadata", None)
            if execution_metadata is None:
                raise ValueError("execution_metadata is required for CONDITION events")
            event = ConditionCheckEvent.from_condition(
                condition=condition_from_name(
                    function_name,
                    execution_metadata,
                ),
                every_tick=1,
                timeout=None,
            )

        else:
            raise ValueError(f"Unsupported event type '{event_type}'.")

        # Force the event ID if it is provided
        if event_id is not None and event is not None:
            event.event_id = event_id

        if event is None:
            raise ValueError("Failed to create event.")

        if event_relative_time is not None:
            event.event_relative_time = event_relative_time

        if event_time is not None:
            event.event_time = event_time

        return event

    def patch_oracle_user_message_order(self) -> None:
        """
        Patches the event dependencies to ensure send_message_to_user events are executed last in each turn.

        This method groups events by turns and ensures that send_message_to_user events depend on all other
        events with maximum accumulated time in the same turn. This guarantees that user messages are sent
        after all other operations in a turn are completed, maintaining proper execution order in oracle mode.
        """
        events = self.events
        events_map = {e.event_id: e for e in events}

        event_id_to_turn_idx = self.build_event_id_to_turn_idx()

        # Compute accumulated_times before splitting by turns
        accumulated_times = self.accumulate_times_from_event(events)

        # Group events by turn using the values of the build_event_id_to_turn_idx dict
        turns_to_events = defaultdict(list)
        for event_id, turn_idx in event_id_to_turn_idx.items():
            if event_id in events_map:
                turns_to_events[turn_idx].append(events_map[event_id])

        # Apply the same logic to each turn
        for turn_idx, events_in_turn in turns_to_events.items():
            send_message_to_user_event = next(
                (
                    event
                    for event in events_in_turn
                    if self.is_send_message_to_user(event)
                ),
                None,
            )

            if send_message_to_user_event is None:
                continue

            # Find the maximum accumulated time for events in this turn
            turn_accumulated_times = {
                event.event_id: accumulated_times[event.event_id]
                for event in events_in_turn
            }
            max_time = (
                max(turn_accumulated_times.values()) if turn_accumulated_times else -1
            )

            # Find all events in this turn that have the maximum accumulated time
            max_time_events = [
                event
                for event in events_in_turn
                if accumulated_times[event.event_id] == max_time
            ]

            # Filter out the send_message_to_user_event from max_time_events to get events it should depend on
            events_to_depend_on = [
                event
                for event in max_time_events
                if event != send_message_to_user_event
            ]

            # Make send_message_to_user_event depend on all other events with maximum time in this turn
            for event in events_to_depend_on:
                # Check if send_message_to_user_event already depends on this event
                if event not in send_message_to_user_event.dependencies:
                    # Add dependency: send_message_to_user_event depends on this event
                    send_message_to_user_event.dependencies.append(event)
                    event.successors.append(send_message_to_user_event)
                    logger.debug(
                        f"Patched oracle mode: send_message_to_user event in turn {turn_idx} now depends on event {event.event_id} with maximum accumulated time {max_time}"
                    )

    def apply_augmentation_configs(self):
        if self.tool_augmentation_config is not None and self.apps is not None:
            for app in self.apps:
                app.set_failure_probability(
                    self.tool_augmentation_config.tool_failure_probability
                )

            if self.augmentation_data is not None:
                name_map = self.augmentation_data.get("tool_names_mapping", {})
                desc_map = self.augmentation_data.get("tool_descriptions_mapping", {})
                apps_to_filter = ["AgentUserInterface", "SystemApp"]
                filtered_apps = [
                    app for app in self.apps if app.name not in apps_to_filter
                ]

                for app in filtered_apps:
                    for tool in app.get_tools():
                        if self.tool_augmentation_config.apply_tool_name_augmentation:
                            tool._public_name = name_map.get(tool.name, tool.name)

                        if self.tool_augmentation_config.apply_tool_description_augmentation:
                            tool._public_description = desc_map.get(
                                tool.name, tool.function_description
                            )
