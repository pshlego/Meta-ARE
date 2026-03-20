# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import inspect
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

from are.simulation.types import CompletedEvent
from are.simulation.validation.base import ToolJudge
from are.simulation.validation.configs import (
    CheckerType,
    HardToolJudgeConfig,
    MildToolJudgeConfig,
    SoftCheckerType,
    SoftToolJudgeConfig,
)
from are.simulation.validation.utils.llm_utils import (
    build_llm_checkers,
    build_subtask_extractor,
)
from are.simulation.validation.utils.misc import (
    extract_text_between_tags,
    normalize_arg,
    normalize_str,
)
from are.simulation.validation.utils.scenario_utils import CompletedOracleEvent
from are.simulation.validation.utils.trace_utils import injected_traceable

logger: logging.Logger = logging.getLogger(__name__)


class HardToolJudge(ToolJudge):
    """
    A judge that performs a scripted check on some action args to compare an agent and oracle event representing a tool call.
    """

    def __init__(self, config: HardToolJudgeConfig):
        super().__init__(config, "hard")
        self.config = config
        self.checkers = {
            CheckerType.eq_checker.value: self.eq_checker,
            CheckerType.unordered_list_checker.value: self.unordered_list_checker,
            CheckerType.datetime_checker.value: self.datetime_checker,
            CheckerType.list_attendees_checker.value: self.list_attendees_checker,
            CheckerType.phone_number_checker.value: self.phone_number_checker,
            CheckerType.eq_str_strip_checker.value: self.eq_str_strip_checker,
            CheckerType.contain_any_checker.value: self.contain_any_checker,
            CheckerType.contain_all_checker.value: self.contain_all_checker,
            CheckerType.path_checker.value: self.path_checker,
            CheckerType.unordered_path_list_checker.value: self.unordered_path_list_checker,
        }
        # Collect checker args names
        self.checker_to_args_names = {
            checker_name: list(inspect.signature(checker_fn).parameters.keys())
            for checker_name, checker_fn in self.checkers.items()
        }

    @injected_traceable(trace_type="eq_checker", tags=["judge"])
    def eq_checker(self, x_agent: Any, x_oracle: Any, **kwargs) -> bool:
        return x_agent == x_oracle

    @injected_traceable(trace_type="unordered_list_checker", tags=["judge"])
    def unordered_list_checker(
        self, x_agent: list[Any] | None, x_oracle: list[Any] | None, **kwargs
    ) -> bool:
        if x_agent is None:
            return x_oracle is None or len(x_oracle) == 0
        if x_oracle is None:
            return x_agent is None or len(x_agent) == 0
        return set(x_agent) == set(x_oracle)

    @injected_traceable(trace_type="path_checker", tags=["judge"])
    def path_checker(self, x_agent: str | None, x_oracle: str | None, **kwargs) -> bool:
        if x_agent is None or x_oracle is None:
            return x_agent == x_oracle
        normalized_agent = os.path.normpath(x_agent).lstrip("/")
        normalized_oracle = os.path.normpath(x_oracle).lstrip("/")
        return normalized_agent == normalized_oracle

    @injected_traceable(trace_type="unordered_path_list_checker", tags=["judge"])
    def unordered_path_list_checker(
        self, x_agent: list[str] | None, x_oracle: list[str] | None, **kwargs
    ) -> bool:
        if x_agent is None:
            return x_oracle is None or len(x_oracle) == 0
        if x_oracle is None:
            return x_agent is None or len(x_agent) == 0
        normalized_agent_paths = {
            os.path.normpath(path).lstrip("/") for path in x_agent
        }
        normalized_oracle_paths = {
            os.path.normpath(path).lstrip("/") for path in x_oracle
        }
        return normalized_agent_paths == normalized_oracle_paths

    def list_attendees_checker(
        self,
        x_agent: list[str] | None,
        x_oracle: list[str] | None,
        tolerance_list_str: list[str] | None = None,
        **kwargs,
    ) -> bool:
        if tolerance_list_str is None:
            tolerance_list_str = []
        _tolerance_list = [normalize_str(x) for x in tolerance_list_str]
        # If the oracle list is empty or contains only tolerance strings, return True
        if x_oracle is None or len(x_oracle) == 0:
            return True
        if all(
            normalize_str(x_oracle[i]) in _tolerance_list for i in range(len(x_oracle))
        ):
            return True
        # Otherwise call unordered list checker with tolerance list
        return self.unordered_str_list_with_tolerance_checker(
            x_agent, x_oracle, _tolerance_list
        )

    @injected_traceable(
        trace_type="unordered_str_list_with_tolerance_checker", tags=["judge"]
    )
    def unordered_str_list_with_tolerance_checker(
        self,
        x_agent: list[str] | None,
        x_oracle: list[str] | None,
        tolerance_list_str: list[str] | None = None,
        **kwargs,
    ) -> bool:
        if tolerance_list_str is None:
            tolerance_list_str = []
        _x_agent = [normalize_str(x) for x in x_agent] if x_agent is not None else []
        _x_oracle = [normalize_str(x) for x in x_oracle] if x_oracle is not None else []
        # Remove elements from tolerance list
        _x_agent = [x for x in _x_agent if x not in tolerance_list_str]
        _x_oracle = [x for x in _x_oracle if x not in tolerance_list_str]
        # Compare sets
        return set(_x_agent) == set(_x_oracle)

    @injected_traceable(trace_type="datetime_checker", tags=["judge"])
    def datetime_checker(
        self, x_agent: str | None, x_oracle: str | None, **kwargs
    ) -> bool:
        if x_agent is None or x_oracle is None:
            return x_agent == x_oracle
        _x_agent = datetime.strptime(x_agent, "%Y-%m-%d %H:%M:%S")
        _x_oracle = datetime.strptime(x_oracle, "%Y-%m-%d %H:%M:%S")
        return _x_agent == _x_oracle

    @injected_traceable(trace_type="eq_str_strip_checker", tags=["judge"])
    def eq_str_strip_checker(
        self, x_agent: str | None, x_oracle: str | None, **kwargs
    ) -> bool:
        _x_agent = (
            x_agent.strip() if bool(x_agent) else ""
        )  # Are we sure we want strip here?
        _x_oracle = x_oracle.strip() if bool(x_oracle) else ""
        return _x_agent == _x_oracle

    @injected_traceable(trace_type="phone_number_checker", tags=["judge"])
    def phone_number_checker(
        self, x_agent: str | None, x_oracle: str | None, **kwargs
    ) -> bool:
        if x_agent is None or x_oracle is None:
            return x_agent is None and x_oracle is None
        # Remove any non-digit characters from the input strings
        _x_agent = "".join(char for char in x_agent if char.isdigit())
        _x_oracle = "".join(char for char in x_oracle if char.isdigit())
        # Compare the cleaned phone numbers
        return _x_agent == _x_oracle

    @injected_traceable(trace_type="contain_any_checker", tags=["judge"])
    def contain_any_checker(self, x_agent: str, targets: list[str], **kwargs) -> bool:
        return any(x_oracle.lower() in x_agent.lower() for x_oracle in targets)

    @injected_traceable(trace_type="contain_any_checker", tags=["judge"])
    def contain_all_checker(self, x_agent: str, targets: list[str], **kwargs) -> bool:
        return all(x_oracle.lower() in x_agent.lower() for x_oracle in targets)

    def compare(
        self, agent_event: CompletedEvent, oracle_event: CompletedOracleEvent, **kwargs
    ) -> bool:
        # Get args
        agent_args = agent_event.get_args()
        oracle_args = oracle_event.get_args()
        # Hard checks
        for arg_name, check_type in self.config.arg_to_checker_type.items():
            if check_type.is_hard():
                checker_fn = self.checkers[check_type.value]  # type: ignore
                args_names = self.checker_to_args_names[check_type.value]  # type: ignore
                # This is only for logging purposes
                checker_args = {k: v for k, v in kwargs.items() if k in args_names}
                checker_args["arg_name"] = arg_name
                # Call the checker
                if not checker_fn(
                    x_agent=agent_args[arg_name],
                    x_oracle=oracle_args[arg_name],
                    **checker_args,
                ):
                    return False
        # Scripted checks TODO: unify hard and scripted checks
        if (
            self.config.event_id_to_checker_params
            and oracle_event.event_id in self.config.event_id_to_checker_params
        ):
            for params in self.config.event_id_to_checker_params[oracle_event.event_id]:
                if params.checker_type.is_hard():
                    if not self.checkers[params.checker_type.value](  # type: ignore
                        agent_args[params.arg_name],
                        oracle_args[params.arg_name],
                        **params.checker_args,
                    ):
                        return False
                elif params.checker_type.is_scripted():
                    if not self.checkers[params.checker_type.value](  # type: ignore
                        agent_args[params.arg_name],
                        **params.checker_args,
                    ):
                        return False
        return True


class SoftToolJudge(ToolJudge):
    """
    A soft judge that compares some action args of an agent and oracle event representing a tool call with an llm.
    """

    def __init__(self, config: SoftToolJudgeConfig):
        super().__init__(config, "soft")
        self.config = config
        # Engine
        self.engine = self.config.engine
        # No arg to check
        self.selected_action_args = [
            name
            for name, check_type in self.config.arg_to_checker_type.items()
            if check_type == CheckerType.llm_checker
        ]
        self.no_arg_to_check = len(self.selected_action_args) == 0

        # Subtask extractor
        self.subtask_extractor = build_subtask_extractor(
            engine=self.config.engine, tool_name=self.tool_name
        )
        # LLM checkers
        self.llm_checkers = build_llm_checkers(engine=self.config.engine)
        # Soft checkers
        self.soft_checkers = {
            SoftCheckerType.content_checker.value: self.content_checker,
            SoftCheckerType.sanity_checker.value: self.sanity_checker,
            SoftCheckerType.signature_checker.value: self.signature_checker,
            SoftCheckerType.placeholder_checker.value: self.placeholder_checker,
            SoftCheckerType.cab_checker.value: self.cab_checker,
            SoftCheckerType.email_checker.value: self.email_checker,
            SoftCheckerType.message_checker.value: self.message_checker,
            SoftCheckerType.user_message_checker.value: self.user_message_checker,
            SoftCheckerType.event_checker.value: self.event_checker,
            SoftCheckerType.tone_checker.value: self.tone_checker,
        }
        # Checker to args names
        self.checker_to_args_names = {
            checker_name: list(inspect.signature(checker_fn).parameters.keys())
            for checker_name, checker_fn in self.soft_checkers.items()
        }
        # Flag if we need to extract subtask
        self.need_subtask = any(c.need_subtask for c in self.config.soft_checker_types)

    def describe_action_args(self, args: dict[str, Any]) -> str:
        return_string = ""
        for k, v in args.items():
            return_string += f"{k}: {v} \n"
        return return_string.strip()

    @injected_traceable(trace_type="equality_checker", tags=["judge"])
    def equality_checker(
        self,
        agent_args: dict[str, Any],
        oracle_args: dict[str, Any],
        **kwargs,
    ) -> bool:
        # Check args
        for arg_name in oracle_args.keys():
            if normalize_arg(agent_args[arg_name]) != normalize_arg(
                oracle_args[arg_name]
            ):
                return False
        return True

    @injected_traceable(trace_type="placeholder_checker", tags=["judge"])
    def placeholder_checker(
        self,
        agent_args: dict[str, Any],
        **kwargs,
    ) -> bool:
        agent_args_str = " ".join([v for v in agent_args.values()])
        placeholders = [
            "[User's Name]",
            "[User Name]",
            "[User]",
            "[Your Name]",
            "[My Name]",
            "Best regards,\nYour Name",
            "Best,\nYour Name",
        ]
        if any(p.lower() in agent_args_str.lower() for p in placeholders):
            return False
        return True

    @injected_traceable(trace_type="extract_subtask", tags=["judge"])
    def extract_subtask(self, oracle_action_call: str, task: str) -> str:
        task = task.strip()
        if len(task) == 0:
            return ""
        # Extract subtask
        subtask = self.subtask_extractor(
            user_prompt_args={
                "tool_name": self.tool_name,
                "oracle_action_call": oracle_action_call,
                "task": task,
            }
        )
        # Extract subtask
        if subtask is None:
            return ""
        subtask = extract_text_between_tags(subtask, "subtask")
        return subtask[0].strip() if len(subtask) > 0 else ""

    @injected_traceable(trace_type="content_checker", tags=["judge"])
    def content_checker(
        self,
        agent_args: dict[str, Any],
        oracle_args: dict[str, Any],
        today_date: str,
        user_address: str,
        subtask: str,
        **kwargs,
    ) -> bool | None:
        # Call the soft checker
        return self.llm_checkers[SoftCheckerType.content_checker.value](
            user_prompt_args={
                "agent_action_call": self.describe_action_args(agent_args),
                "oracle_action_call": self.describe_action_args(oracle_args),
                "task": subtask,
                "tool_name": self.tool_name,
                "today_date": today_date,
                "user_address": user_address,
            }
        )

    @injected_traceable(trace_type="signature_checker", tags=["judge"])
    def signature_checker(
        self,
        agent_args: dict[str, Any],
        user_name: str,
        **kwargs,
    ) -> bool | None:
        # Call the soft checker
        return self.llm_checkers[SoftCheckerType.signature_checker.value](
            user_prompt_args={
                "agent_action_call": self.describe_action_args(agent_args),
                "user_name": user_name,
            }
        )

    @injected_traceable(trace_type="sanity_checker", tags=["judge"])
    def sanity_checker(
        self,
        agent_args: dict[str, Any],
        task: str = "",
        previous_task: str = "",
        **kwargs,
    ) -> bool | None:
        # Check for numerical values first
        agent_action_call = self.describe_action_args(agent_args)

        def is_valid_numerical_values_format(s):
            # Define the regular expression pattern
            pattern = r"^content: \d+(\.\d+)?$"

            # Use re.match to check if the string matches the pattern
            return re.match(pattern, s) is not None

        if is_valid_numerical_values_format(agent_action_call):
            return True
        # Call the soft checker
        return self.llm_checkers[SoftCheckerType.sanity_checker.value](
            user_prompt_args={
                "agent_action_call": agent_action_call,
                "task": "\n".join([previous_task, task]),
            }
        )

    @injected_traceable(trace_type="cab_checker", tags=["judge"])
    def cab_checker(
        self,
        agent_args: dict[str, Any],
        oracle_args: dict[str, Any],
        user_address: str,
        **kwargs,
    ) -> bool | None:
        # Call the soft checker
        return self.llm_checkers[SoftCheckerType.cab_checker.value](
            user_prompt_args={
                "agent_action_call": self.describe_action_args(agent_args),
                "oracle_action_call": self.describe_action_args(oracle_args),
                "user_address": user_address,
            }
        )

    @injected_traceable(trace_type="email_checker", tags=["judge"])
    def email_checker(
        self,
        agent_args: dict[str, Any],
        oracle_args: dict[str, Any],
        today_date: str,
        **kwargs,
    ) -> bool | None:
        # Call the soft checker
        return self.llm_checkers[SoftCheckerType.email_checker.value](
            user_prompt_args={
                "agent_action_call": self.describe_action_args(agent_args),
                "oracle_action_call": self.describe_action_args(oracle_args),
                "today_date": today_date,
            }
        )

    @injected_traceable(trace_type="message_checker", tags=["judge"])
    def message_checker(
        self,
        agent_args: dict[str, Any],
        oracle_args: dict[str, Any],
        today_date: str,
        **kwargs,
    ) -> bool | None:
        # Call the soft checker
        return self.llm_checkers[SoftCheckerType.message_checker.value](
            user_prompt_args={
                "agent_action_call": self.describe_action_args(agent_args),
                "oracle_action_call": self.describe_action_args(oracle_args),
                "today_date": today_date,
            }
        )

    @injected_traceable(trace_type="event_checker", tags=["judge"])
    def event_checker(
        self,
        agent_args: dict[str, Any],
        oracle_args: dict[str, Any],
        user_address: str,
        subtask: str,
        **kwargs,
    ) -> bool | None:
        # Call the soft checker
        return self.llm_checkers[SoftCheckerType.event_checker.value](
            user_prompt_args={
                "agent_action_call": self.describe_action_args(agent_args),
                "oracle_action_call": self.describe_action_args(oracle_args),
                "user_address": user_address,
                "task": subtask,
            }
        )

    @injected_traceable(trace_type="user_message_checker", tags=["judge"])
    def user_message_checker(
        self,
        agent_args: dict[str, Any],
        oracle_args: dict[str, Any],
        subtask: str,
        **kwargs,
    ) -> bool | None:
        # Call the soft checker
        return self.llm_checkers[SoftCheckerType.user_message_checker.value](
            user_prompt_args={
                "agent_action_call": self.describe_action_args(agent_args),
                "oracle_action_call": self.describe_action_args(oracle_args),
                "task": subtask,
            }
        )

    @injected_traceable(trace_type="tone_checker", tags=["judge"])
    def tone_checker(
        self,
        agent_args: dict[str, Any],
        **kwargs,
    ) -> bool | None:
        # Call the soft checker
        return self.llm_checkers[SoftCheckerType.tone_checker.value](
            user_prompt_args={
                "agent_action_call": self.describe_action_args(agent_args),
            }
        )

    def get_checker_kwargs(
        self,
        kwargs: dict[str, Any],
        oracle_event: CompletedOracleEvent,
        oracle_args: dict[str, Any],
    ) -> dict[str, Any]:
        # Get the task
        tasks = kwargs.get("tasks", [""])
        # Soft checker kwargs
        today_date = ""
        if oracle_event.event_time is not None:
            today_date = datetime.fromtimestamp(
                oracle_event.event_time, tz=timezone.utc
            ).strftime("%Y-%m-%d %A")
        # Subtask
        subtask = (
            self.extract_subtask(
                oracle_action_call=self.describe_action_args(oracle_args),
                task="\n".join(tasks),
            )
            if self.need_subtask
            else ""
        )
        # User name
        user_details = kwargs.get("user_details")
        user_name = (
            f"{user_details.first_name} {user_details.last_name}"
            if user_details
            else ""
        )
        # User address
        user_address = f"{user_details.address}" if user_details else ""

        # Tasks
        return {
            "task": tasks[-1],
            "previous_task": "/n".join(tasks[:-1]) if len(tasks) > 1 else "",
            "user_name": user_name,
            "user_address": user_address,
            "today_date": today_date,
            "oracle_args": oracle_args,
            "subtask": subtask,
        }

    def compare(
        self,
        agent_event: CompletedEvent,
        oracle_event: CompletedOracleEvent,
        **kwargs,
    ) -> bool | None:
        # If no args to check, then return True
        if self.no_arg_to_check:
            return True
        # Get the args
        oracle_args = oracle_event.get_args()
        agent_args = agent_event.get_args()
        selected_action_args = [
            arg for arg in self.selected_action_args if bool(oracle_args[arg])
        ]
        oracle_args = {
            k: v for k, v in oracle_args.items() if k in selected_action_args
        }
        agent_args = {k: v for k, v in agent_args.items() if k in selected_action_args}
        # Check equality
        if self.equality_checker(
            agent_args=agent_args,
            oracle_args=oracle_args,
        ):
            return True
        # Get checker kwargs
        checker_kwargs = self.get_checker_kwargs(
            kwargs=kwargs,
            oracle_event=oracle_event,
            oracle_args=oracle_args,
        )
        # Apply soft checkers
        for checker in self.config.soft_checker_types:
            checker_fn = self.soft_checkers[checker.value]
            # This is only for logging purposes
            _checker_kwargs = {
                k: v
                for k, v in checker_kwargs.items()
                if k in self.checker_to_args_names[checker.value]
            }
            # Call the checker
            result = checker_fn(
                agent_args=agent_args,
                **_checker_kwargs,
            )
            if result is None:
                logger.warning(
                    f"Soft checker '{checker.value}' returned None (inconclusive) "
                    f"for tool '{self.tool_name}' — treating as FAIL"
                )
                return False
            if not result:
                logger.info(
                    f"Soft checker '{checker.value}' returned False "
                    f"for tool '{self.tool_name}'"
                )
                return False
        return True


class MildToolJudge(ToolJudge):
    """
    A mild judge that combines a hard and soft judge to compare an agent and oracle event representing a tool call.
    If first call the hard judge and if it passes, then call the soft judge.
    """

    def __init__(self, config: MildToolJudgeConfig):
        super().__init__(config, "mild")
        # Config
        self.config = config
        # Hard judge
        self.hard_judge = HardToolJudge(
            HardToolJudgeConfig(
                tool_name=config.tool_name,
                arg_to_checker_type=config.arg_to_checker_type,
                event_id_to_checker_params=config.event_id_to_checker_params,
                tracer=self.tracer,
            )
        )
        # Soft judge
        self.soft_judge = SoftToolJudge(
            SoftToolJudgeConfig(
                tool_name=config.tool_name,
                arg_to_checker_type=config.arg_to_checker_type,
                soft_checker_types=config.soft_checker_types,
                engine=config.engine,
                tracer=self.tracer,
            )
        )
        # If scripted checkers are provided we do not use the soft judge
        if self.config.event_id_to_checker_params is not None:
            self.soft_judge.no_arg_to_check = True

    def compare(
        self, agent_event: CompletedEvent, oracle_event: CompletedOracleEvent, **kwargs
    ) -> bool | None:
        # TODO: change system prompt when hard comparison fails to call soft judge
        hard_comparison = self.hard_judge(agent_event, oracle_event, **kwargs)
        if not hard_comparison or self.soft_judge.no_arg_to_check:
            return hard_comparison
        return self.soft_judge(agent_event, oracle_event, **kwargs)
