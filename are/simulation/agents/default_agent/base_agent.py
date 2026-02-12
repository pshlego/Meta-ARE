# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


# Copied-pasted and customized from https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py
import random
import time
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from os import getenv
from threading import Event
from typing import Callable, Sequence, Type, TypeVar, Union

from are.simulation.agents.agent_log import (
    BaseAgentLog,
    ErrorLog,
    LLMInputLog,
    LLMOutputThoughtActionLog,
    ObservationLog,
    StepLog,
    StopLog,
    SubagentLog,
    SystemPromptLog,
    TaskLog,
    ThoughtLog,
)
from are.simulation.agents.are_simulation_agent import AgentStoppedException
from are.simulation.agents.decoding_types import DecodingSchema
from are.simulation.agents.default_agent.tools.action_executor import (
    BaseActionExecutor,
    ParsedAction,
)
from are.simulation.agents.default_agent.tools.json_action_executor import (
    JsonActionExecutor,
)
from are.simulation.agents.default_agent.utils.logging_utils import (
    get_default_logger,
    get_parent_logger,
)
from are.simulation.agents.llm.exceptions import (
    EmptyResponseException,
    PromptTooLongException,
)
from are.simulation.agents.llm.types import MessageRole, MMObservation
from are.simulation.agents.multimodal import Attachment, attachments_to_pil
from are.simulation.exceptions import FatalError, InvalidActionAgentError, LoggedError
from are.simulation.notification_system import BaseNotificationSystem
from are.simulation.time_manager import TimeManager
from are.simulation.tool_box import DEFAULT_TOOL_DESCRIPTION_TEMPLATE, Toolbox
from are.simulation.tools import SystemPrompt, Tool
from are.simulation.types import SimulatedGenerationTimeConfig


def to_text(input: list[dict[str, str]] | dict[str, str] | str) -> str:
    if isinstance(input, list):
        return "\n".join([m["content"] for m in input])
    elif isinstance(input, dict):
        return input["content"]
    else:
        return input


class RunningState(Enum):
    RUNNING = "running"
    PAUSED = "paused"
    TERMINATED = "terminated"
    FAILED = "failed"


DEFAULT_STEP_2_ROLE = OrderedDict(
    [
        ("system_prompt", MessageRole.SYSTEM),
        ("task", MessageRole.USER),
        ("llm_output", MessageRole.ASSISTANT),
        ("facts", MessageRole.USER),
        ("plan", MessageRole.USER),
        ("tool_call", MessageRole.TOOL_CALL),
        ("rationale", MessageRole.ASSISTANT),
        ("observation", MessageRole.TOOL_RESPONSE),
        ("error", MessageRole.TOOL_RESPONSE),
        ("agent_user_interface", MessageRole.USER),
        ("environment_notifications", MessageRole.USER),
        ("hint", MessageRole.USER),
        ("task_reminder", MessageRole.USER),
    ]
)


DEFAULT_STEP_2_MESSAGE = OrderedDict(
    [
        ("system_prompt", "{content}\n"),
        ("task", "[TASK]: \n{content}\n"),
        ("llm_output", "{content}\n"),
        ("facts", "[FACTS LIST]:\n{content}\n"),
        ("plan", "[PLAN]:\n{content}\n"),
        ("tool_call", "[STEP {i} TOOL CALL]:\n{content}\n"),
        ("rationale", "{content}\n"),
        ("observation", "[OUTPUT OF STEP {i}] Observation:\n***\n{content}\n***\n"),
        (
            "error",
            "[OUTPUT OF STEP {i}] ERROR:\n***\n{content}\n***\n\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n",
        ),
        ("agent_user_interface", "User messages updates:\n***\n{content}\n***\n"),
        (
            "environment_notifications",
            "Environment notifications updates:\n***\n{content}\n***\n",
        ),
        ("hint", "[HINT]:\n{content}\n"),
        ("task_reminder", "[TASK REMINDER]:\n{content}\n"),
    ]
)

# Reserved keyword for MM observation at initialization.
DEFAULT_TASK_OBSERVATION = "task_image"


def convert_plan_fact_messages_to_user(content: str) -> str:
    # changing of content
    if "I still need to solve the task I was given:" in content:
        content = content.replace(
            "I still need to solve the task I was given:",
            "You still need to solve the task you were given:",
        )

    # Changing of role type
    if "Here are the facts that I know so far" in content:
        content = content.replace(
            "Here are the facts that I know so far",
            "Here are the facts that you know so far",
        )
    elif "Here is the plan of action that I will follow to solve the task" in content:
        content = content.replace(
            "Here is the plan of action that I will follow to solve the task",
            "Here is the plan of action that you will follow to solve the task",
        )
        content += "\n Now pursue your plan"
    elif "Here is the updated list of the facts that I know" in content:
        content = "Following this result:\n\n" + content.replace(
            "Here is the updated list of the facts that I know",
            "Here is the updated list of the facts that you know",
        )
    elif "Here is my new/updated plan of action to solve the task:" in content:
        content = content.replace(
            "Here is my new/updated plan of action to solve the task:",
            "Here is your new/updated plan of action to solve the task:",
        )
        content += "\n Now pursue your plan"

    return content


def format_message(
    message_dict,
    message_type: str,
    content: str,
    i: int | None = None,
    timestamp: float | None = None,
) -> str:
    template = message_dict.get(message_type)
    if template is None:
        raise ValueError(f"Unknown message type: {message_type}")

    if message_type in ["facts", "plan"]:
        content = convert_plan_fact_messages_to_user(content)

    format_args = {"content": content}
    if "{i}" in template:
        if i is None:
            raise ValueError(f"Message type '{message_type}' requires 'i' parameter")
        format_args["i"] = str(i)

    if "{timestamp}" in template:
        if timestamp is None:
            raise ValueError(
                f"Message type '{message_type}' requires 'timestamp' parameter"
            )
        format_args["timestamp"] = datetime.fromtimestamp(
            timestamp, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")

    return template.format(**format_args)


def default_termination_condition(agent) -> bool:
    return agent.iterations >= agent.max_iterations


@dataclass
class ConditionalStep:
    condition: Callable[["BaseAgent"], bool] | None
    function: Callable[["BaseAgent"], None]
    name: str


@dataclass
class TerminationStep(ConditionalStep):
    function: Callable[["BaseAgent"], None] = lambda _: None
    name: str = "termination"


default_termination_step = TerminationStep(
    condition=default_termination_condition,
)


def update_system_prompt_with_tools(
    system_prompts: dict[str, str | SystemPrompt], tools: list[Tool]
) -> dict[str, str]:
    system_prompt = system_prompts["system_prompt"]
    if isinstance(system_prompt, SystemPrompt):
        system_prompt = str(system_prompt)
    toolbox = Toolbox(tools=tools)
    tool_descriptions = toolbox.show_tool_descriptions(
        DEFAULT_TOOL_DESCRIPTION_TEMPLATE
    )
    prompt = system_prompt.replace("<<tool_descriptions>>", tool_descriptions)
    if "<<tool_names>>" in prompt:
        tool_names = [f"'{tool_name}'" for tool_name in toolbox.tools.keys()]
        prompt = prompt.replace("<<tool_names>>", ", ".join(tool_names))
    system_prompts["system_prompt"] = prompt
    return system_prompts


def update_system_prompt_with_authorized_imports(
    system_prompts: dict[str, str], authorized_imports: list[str]
) -> dict[str, str]:
    system_prompt = system_prompts["system_prompt"]
    if isinstance(system_prompt, SystemPrompt):
        system_prompt = str(system_prompt)
    if "<<authorized_imports>>" in system_prompt:
        prompt = system_prompt.replace(
            "<<authorized_imports>>", ", ".join(authorized_imports)
        )
        system_prompts["system_prompt"] = prompt

    return system_prompts


FORCE_RETRY_PROMPT = """
Thought: ERROR
Action:
{
  "action": "final_answer",
  "action_input": {"answer": "Search failed because of an unexcepted error. Please try again."}
}<end_action>
"""


class BaseAgent:
    """
    This agent is a ReAct loop on steroids.
    """

    def __init__(
        self,
        llm_engine: Callable,
        system_prompts: dict[str, str] = {},
        tools: dict[str, Tool] = {},
        update_system_prompt_tools: Callable[
            [dict[str, str], list[Tool]], dict[str, str]
        ] = update_system_prompt_with_tools,
        conditional_pre_steps: list[ConditionalStep] | None = None,
        conditional_post_steps: list[ConditionalStep] | None = None,
        termination_step: TerminationStep = default_termination_step,
        role_dict: OrderedDict = DEFAULT_STEP_2_ROLE,
        message_dict: OrderedDict = DEFAULT_STEP_2_MESSAGE,
        action_executor: BaseActionExecutor | None = None,
        max_iterations: int = 10,
        total_iterations: int = 50,
        shuffle_tools: bool = False,
        shuffle_authorized_imports: bool = False,
        thought_token: str | None = None,
        action_token: str | None = None,
        retry_llm_call_on_error: bool = True,
        time_manager: TimeManager | None = None,
        log_callback: Callable[[BaseAgentLog], None] | None = None,
        handle_prompt_too_long: bool = False,
        simulated_generation_time_config: SimulatedGenerationTimeConfig | None = None,
        use_custom_logger: bool = True,
    ):
        """
        :param tools: list[Tool] - List of tools available to the agent.
        :param tool_description_template: str - Template for describing tools to LLM prompt.
        :param llm_engine: Callable - Function to call LLM.
        :param system_prompts: dict[str, str] - System prompts for different steps of the agent.
        :param post_step_functions: dict[str, Callable] - Functions to call after each step of the agent.
        :param termination_methods: dict[str, Callable] - Methods to handle the end of the agent.
        :param step_conditions: dict[str, Callable] - Conditions to check before each step of the agent.
        :param role_dict: dict[str, str] - Dictionary mapping roles to their names.
        :param tool_parser: Callable - Function to parse tool calls.
        :param max_iterations: int - Maximum self.planning_counter before termination (excludes errors)
        :param total_iterations: int - Maximum self.iterations before termination (includes errors)
        :param shuffle_tools: bool - Whether to shuffle the list of tools and authorized imports.
        :param time_manager: TimeManager - Time manager for the agent.
        :param log_callback: Callable[[BaseAgentLog], None] - Callback function to log agent actions.
        :param simulated_generation_time_config: SimulatedGenerationTimeConfig - Configuration for simulated generation time.
        """
        self.name = "base_agent"
        # Here we store a reference to the parent Agent to build the log hierarchy, if the current agent is a sub-agent
        self.parent: BaseAgent | None = None
        self.llm_engine = llm_engine
        self.time_manager = time_manager if time_manager is not None else TimeManager()
        self.log_callback = log_callback

        self.tools = tools
        self.update_system_prompt_tools = update_system_prompt_tools
        self.init_system_prompts = system_prompts

        self.conditional_pre_steps = (
            conditional_pre_steps if conditional_pre_steps is not None else []
        )
        self.conditional_post_steps = (
            conditional_post_steps if conditional_post_steps is not None else []
        )

        self.termination_step = termination_step

        self.role_dict = role_dict
        self.message_dict = message_dict
        assert action_executor, "you need to specify an action_executor"
        self.action_executor = action_executor

        self.max_iterations = max_iterations
        self.total_iterations = total_iterations
        self.action_token = (
            self.action_executor.action_token if action_token is None else action_token
        )
        self.thought_token = (
            self.action_executor.thought_token
            if thought_token is None
            else thought_token
        )
        regex_schema_pattern: str = getenv("BASE_AGENT_REGEX_SCHEMA", "")
        regex_schema: str | None = (
            regex_schema_pattern.format(self.action_token)
            if regex_schema_pattern
            else None
        )
        self.decoding_schema: DecodingSchema | None = (
            DecodingSchema(type="regex", decoding_schema=regex_schema)
            if regex_schema
            else None
        )
        invalid_format_retries: str | None = getenv("BASE_AGENT_INVALID_FORMAT_RETRIES")
        self.invalid_format_retries = int(invalid_format_retries or 10)
        self.logs: list[BaseAgentLog] = []
        self.sub_start_logs = None

        # Use the use_custom_logger parameter directly
        self.logger = (
            get_default_logger(__name__)
            if use_custom_logger
            else get_parent_logger(__name__)
        )

        # Placeholders for the Meta Agents Research Environments notification system that is set when running a Scenario
        self.notification_system: BaseNotificationSystem | None = None

        # Data Augmentation parameters
        self.shuffle_tools = shuffle_tools
        self.shuffle_authorized_imports = shuffle_authorized_imports
        self.shuffle_seed = time.time()

        self.initialized = False
        self.stop_event = Event()

        # Placeholders for pause/resume functionality
        self.pause_env: Callable[[], None] | None = None
        self.resume_env: Callable[[float], None] | None = None
        self.simulated_generation_time_config = simulated_generation_time_config
        self.simulation_generation_time_rng = None

        # We keep here the mapping between the original tool name and the variant generated by data augmentation
        # Because we need this for termination conditions for example, or to check if a particular tool was called
        # Knowing we don't exactly know the variant name generated only the original name
        self.original_tool_name_to_variant = {}

        self.retry_llm_call_on_error = retry_llm_call_on_error

        self.custom_state = {}
        self.task = ""
        self.attachments = []

        self.handle_prompt_too_long = handle_prompt_too_long

        # Create Agent hex ID for logging
        self.agent_id = str(uuid.uuid4().hex)

    T = TypeVar("T", bound=BaseAgentLog)

    def get_last_log_of_type(
        self, log_type: Type[T], break_on: Sequence[Type[BaseAgentLog]] = [TaskLog]
    ) -> T | None:
        # Normally we only care about the last log of a specific type in the current agent turn, which are broken by user tasks, so we break on TaskLog
        for log in self.logs[::-1]:
            if isinstance(log, log_type):
                return log

            for break_type in break_on:
                if isinstance(log, break_type):
                    return None

        return None

    def _most_recent_attachments(self) -> set[Attachment]:
        """
        Get the most recent attachments from the logs, at most 5. Since LLM call supports at most 5 attachments.
        :return: List[Attachment] - List of attachments.
        """
        # We always include task attachments, no matter how many.
        attachments = set(
            attachment
            for log in self.logs
            if isinstance(log, TaskLog)
            for attachment in (log.attachments or [])
        )
        MAX_SIZE = 5
        # Include attachments from the last observation logs, until max size is reached.
        remaining = MAX_SIZE - len(attachments)
        if remaining <= 0:
            self.logger.debug("Too many attachments, only including task attachments.")
        for log in self.logs[::-1]:
            if isinstance(log, ObservationLog) and remaining > 0 and log.attachments:
                attachments |= set(log.attachments)
                remaining -= len(log.attachments)
            if remaining <= 0:
                break
        return attachments

    def build_history_from_logs(
        self, exclude_log_types: list[str] = []
    ) -> list[dict[str, str | list[Attachment]]]:
        """
        Build the history of messages from the logs, ensuring a specific order of steps.
        :param exclude_log_types: List of log types to exclude from the history.
        :return: List[Dict[str, str]] - List of messages.
        """
        # The order is already defined by the OrderedDict
        step_order = list(self.role_dict.keys())
        history = []
        id_output_step = 0

        valid_attachments = self._most_recent_attachments()
        for i, log in enumerate(self.logs):
            step_messages = defaultdict(list)
            role = log.get_type()
            timestamp = log.timestamp
            if role in ["observation", "error"]:
                id_output_step += 1

            if isinstance(log, ErrorLog) and log.error == "MaxIterationsAgentError":
                continue

            if (
                isinstance(log, ErrorLog)
                and log.error == "PromptTooLongException"
                and self.handle_prompt_too_long
            ):
                attachments_for_llm = None
                prev_observation_log = None
                for prev_log in self.logs[i::-1]:
                    if isinstance(prev_log, ObservationLog):
                        prev_observation_log = prev_log
                        break
                if prev_observation_log is not None:
                    content = repr(prev_observation_log.content)
                    trunc_content = content[:100] + "\n[...]\n" + content[-100:]
                    self.logger.debug(
                        f"Prompt too long: Removing the observation from the previous step because it possibly flooded the context. Here is the truncated observation (first 100 + [...] + last 100 chars):\n{trunc_content}"
                    )
                    exception = (
                        log.exception
                        + f"\nObservation was removed because it possibly flooded the context. Truncated observation (first 100 + [...] + last 100 chars):\n{trunc_content}"
                    )
                    history.pop()
                    content_for_llm = f"Error: {log.error}\nException: {exception}\nCategory: {log.category}"  # so that we don't modify the log each time we call build_history_from_logs
                else:
                    content_for_llm = log.get_content_for_llm()
            elif isinstance(log, TaskLog):
                content_for_llm = log.get_content_for_llm_no_attachment()
                if not content_for_llm:
                    # If there is no content, we don't want to include the task in the history
                    continue
                attachments_for_llm = log.get_attachments_for_llm()
            elif isinstance(log, ObservationLog):
                # Include attachments only if this log is in our selected attachment_logs
                content_for_llm = log.get_content_for_llm_no_attachment()
                attachments_for_llm = [
                    attachment
                    for attachment in log.get_attachments_for_llm() or []
                    if attachment in valid_attachments
                ]
            else:
                content_for_llm = log.get_content_for_llm()
                attachments_for_llm = None

            if (
                role not in self.role_dict
                or role in exclude_log_types
                or (
                    (content_for_llm is None or content_for_llm == "")
                    and attachments_for_llm is None
                )
            ):
                continue

            # some providers (like fireworks-ai) do not support attachments, so if the attachments is an empty list, we set it to None
            if not attachments_for_llm:
                attachments_for_llm = None

            content = format_message(
                message_dict=self.message_dict,
                message_type=role,
                content=content_for_llm or "",
                i=id_output_step,
                timestamp=timestamp,
            )
            step_messages[role].append(
                {
                    "role": self.role_dict[role],
                    "content": content,
                    "attachments": attachments_for_llm,
                }
            )
            for step in step_order:
                if step in step_messages:
                    history.extend(step_messages[step])
        return history

    def seed_observation(self, observation: MMObservation):
        self.action_executor.inject_state({DEFAULT_TASK_OBSERVATION: observation})

    def init_tools(self):
        tool_values = [tool for tool in self.tools.values()]

        if self.shuffle_tools:
            self.logger.debug(f"Shuffling tools with seed {self.shuffle_seed}")
            shuffle_tools_rng = random.Random(self.shuffle_seed)
            shuffle_tools_rng.shuffle(tool_values)

        self.action_executor.update_tools(self.tools)
        return tool_values

    def is_initialized(self) -> bool:
        return self.initialized

    def initialize(self, attachments: list[Attachment] | None = None, **kwargs) -> None:
        """
        Initialize the agent for a given task.
        :param kwargs: dict[str, Any] - Additional arguments for the agent.
        """
        self.initialized = True
        self.logs = []
        self.iterations = 0
        self.planning_counter = (
            0  # used by the agent (may not be incremented at every step)
        )

        tool_values = self.init_tools()
        self.system_prompts = self.update_system_prompt_tools(
            self.init_system_prompts,
            tool_values,
        )

        # TODO: Check if still relevant / up-to-date
        self.system_prompt = "\n\n".join(
            prompt for prompt in self.init_system_prompts.values()
        )
        self.append_agent_log(
            SystemPromptLog(
                content=self.system_prompt,
                timestamp=self.make_timestamp(),
                agent_id=self.agent_id,
            )
        )

        self.logger.debug("System prompt is as follows:")
        self.logger.debug(self.system_prompt)

        # Here we update the tools in the action executor
        # to take into account any modifications to the tools that happened after the init
        if hasattr(self.action_executor, "llm_engine"):
            self.action_executor.llm_engine = self.llm_engine

        # Reload the agent state if logs are provided
        start_logs = kwargs.pop("start_logs", [])
        if start_logs:
            self.replay(start_logs)

        # Include additional image PILs directly into state stack.
        if attachments is not None:
            images = attachments_to_pil(attachments)
            self.action_executor.inject_state(
                {f"image_{i}": image for i, image in enumerate(images)}
            )
            self.logger.debug(
                f"======== Injecting images into states for {len(images)} images ========"
            )
            self.logger.debug(f"New Keys {','.join(self.action_executor.state.keys())}")

    def step(self):
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        The errors are raised here, they are caught and logged in the run() method.
        """
        # 1. Build the history messages from the logs to prompt the LLM
        agent_memory = self.build_history_from_logs(
            exclude_log_types=["tool_call", "rationale", "action"]
        )
        prompt = agent_memory.copy()

        self.append_agent_log(
            LLMInputLog(
                content=prompt, timestamp=self.make_timestamp(), agent_id=self.agent_id
            )
        )

        # 2. Call the model to generate the next rationale and action
        # We don't put empty string as a default value, because it can cause an infinite loop if Metagen returns an empty string
        # For example when the input prompt is too long
        if self.simulated_generation_time_config is not None:
            # We pause the environment for the generation of a thought/action
            if self.pause_env is None:
                raise ValueError("pause_env is not set")
            self.pause_env()

        format_try_count: int = 0
        llm_output = None
        metadata = None

        if "ERROR" in prompt[-1]["content"] and "candidates" in prompt[-1]["content"]:
            self.logger.warning("Server error. Retrying...")
            llm_output = FORCE_RETRY_PROMPT
        else:
            while llm_output is None or (
                self.retry_llm_call_on_error
                and self.action_token not in llm_output
                and self.thought_token not in llm_output
            ):
                if llm_output is not None:
                    self.logger.warning(
                        f"LLM did not return a valid output: {llm_output}.\nRetrying..."
                    )
                    self.log_error(
                        InvalidActionAgentError(
                            f"The LLM output was not formatted correctly: {llm_output}"
                        )
                    )
                llm_response = self.llm_engine(
                    prompt,
                    stop_sequences=["<end_action>", "Observation:"],
                    additional_trace_tags=["action"],
                    schema=self.decoding_schema,
                )
                if isinstance(llm_response, tuple) and len(llm_response) == 2:
                    llm_output, metadata = llm_response
                else:
                    llm_output = llm_response
                    metadata = {}

                format_try_count += 1
                # This is a failsafe from infinite loop issues that can happen when the input prompt is too long
                if format_try_count > self.invalid_format_retries:
                    break

        self.logger.debug("===== Output message of the LLM: =====")
        self.logger.debug(llm_output)

        if metadata is None:
            self.logger.debug(
                "LLM Engine did not return any metadata. This will impact the time management of the agent."
            )
            metadata = {}

        completion_duration = metadata.get("completion_duration", 0)

        # Resume the environment after the generation of a thought/action if needed
        if self.simulated_generation_time_config is not None:
            if self.resume_env is None:
                raise ValueError("resume_env is not set")

            offset = get_offset_from_time_config_mode(
                time_config=self.simulated_generation_time_config,
                completion_duration=completion_duration,
            )
            self.logger.debug(f"Resuming environment with {offset} offset")
            self.resume_env(offset)

        # DO NOT REMOVE THIS LINE, IT BREAKS REACT LOOP
        self.append_agent_log(
            LLMOutputThoughtActionLog(
                content=llm_output,
                timestamp=self.make_timestamp(),
                agent_id=self.agent_id,
                prompt_tokens=metadata.get("prompt_tokens", 0),
                completion_tokens=metadata.get("completion_tokens", 0),
                total_tokens=metadata.get("total_tokens", 0),
                reasoning_tokens=metadata.get("reasoning_tokens", 0),
                completion_duration=metadata.get("completion_duration", 0),
            )
        )

        if format_try_count > self.invalid_format_retries:
            raise InvalidActionAgentError(
                f"LLM did not return a valid output after {format_try_count} iterations: {llm_output}"
            )

        if llm_output is None or (
            self.action_token not in llm_output and "Thought:" not in llm_output
        ):
            raise InvalidActionAgentError(
                f"The LLM output was not formatted correctly: {llm_output} - did not contain {self.action_token} or {self.thought_token}"
            )

        # 3. Extract the rationale and the action from the LLM output
        try:
            agent_action = self.action_executor.extract_action(
                llm_output=llm_output, split_token=self.action_token
            )
        except Exception as e:
            self.logger.error(f"Error while extracting action: {e}")
            self.logger.debug(f"LLM output: {llm_output}")
            raise e

        self.append_agent_log(
            ThoughtLog(
                content=agent_action.rationale,
                timestamp=self.make_timestamp(),
                agent_id=self.agent_id,
            )
        )

        if agent_action.action is not None:
            # Parse the action
            parsed_action = self.action_executor.parse_action(agent_action)
            self.action_executor.execute_parsed_action(
                parsed_action,
                self.append_agent_log,
                self.make_timestamp,
                self.agent_id,
            )
        else:
            self.logger.warning(f"No action found in LLM output {llm_output}")

    def get_agent_logs(self) -> list[BaseAgentLog]:
        return self.logs.copy()

    def append_subagent_log(self, subagent_log: BaseAgentLog, group_id: str) -> None:
        for log in self.logs:
            if isinstance(log, SubagentLog) and log.group_id == group_id:
                log.children.append(subagent_log)
                return
        self.append_agent_log(
            SubagentLog(
                group_id=group_id,
                children=[subagent_log],
                name=self.name,
                timestamp=self.make_timestamp(),
                agent_id=self.agent_id,
            )
        )

    def append_agent_log(self, log: BaseAgentLog) -> None:
        self.logs.append(log)
        if self.log_callback is not None:
            self.log_callback(log)

    def stop(self) -> None:
        if not self.stop_event.is_set():
            self.logger.warning(f"Stopping agent {self.name}")
        self.stop_event.set()

    def execute_agent_loop(self) -> str | None | MMObservation:
        while (
            self.termination_step.condition is not None
            and not self.termination_step.condition(self)
        ):
            try:
                self.logger.debug(f"Starting iteration {self.iterations}...")

                self.append_agent_log(
                    StepLog(
                        iteration=self.iterations,
                        timestamp=self.make_timestamp(),
                        agent_id=self.agent_id,
                    )
                )

                if self.stop_event.is_set():
                    raise AgentStoppedException("Agent stopped.")

                # Execute a pre_step() function if it exists
                for conditional_step in self.conditional_pre_steps:
                    if conditional_step.condition is None or conditional_step.condition(
                        self
                    ):
                        conditional_step.function(self)

                if self.stop_event.is_set():
                    raise AgentStoppedException("Agent stopped.")

                # Execute the step()
                self.step()

                if self.stop_event.is_set():
                    raise AgentStoppedException("Agent stopped.")

                # Execute a post_step() function if it exists (polling the Meta Agents Research Environments notifications for example)
                for conditional_step in self.conditional_post_steps:
                    if conditional_step.condition is None or conditional_step.condition(
                        self
                    ):
                        conditional_step.function(self)

            except AgentStoppedException as e:
                self.logger.warning("Agent stopped.")
                self.append_agent_log(
                    StopLog(
                        content=f"Agent stopped - {e}",
                        timestamp=self.make_timestamp(),
                        agent_id=self.agent_id,
                    )
                )
                break
            except PromptTooLongException as e:
                if not self.handle_prompt_too_long:
                    # Raise handle_prompt_too_long exception
                    raise e
                else:
                    # Handle the error and add truncated content in build_history_from_logs()
                    self.log_error(e)
            except FatalError as e:
                self.logger.warning(f"Fatal error - {e}")
                self.log_error(e)
                self.custom_state["running_state"] = RunningState.FAILED
                break
            except Exception as e:
                self.log_error(e)
            finally:
                if (
                    self.simulated_generation_time_config is not None
                    and self.resume_env is not None
                ):
                    # Resume the environment in case it was paused and not resumed because of an exception

                    self.resume_env(0.0)  # Resume without advancing time
                self.iterations += 1
                self.planning_counter += 1

        # We have reached a termination condition, execute the termination method
        if self.termination_step.function is not None and not self.stop_event.is_set():
            return self.termination_step.function(self)

    def run(
        self,
        task: str,
        reset: bool = True,
        attachments: list[Attachment] | None = None,
        **kwargs,
    ) -> Union[str, MMObservation] | None:
        """
        Run the agent on a given task.
        :param task: str - Task to solve.
        :param reset: bool - Whether to reset the agent before running.
        :param kwargs: Dict[str, Any] - Additional arguments for the agent.
        :return: Any - Result of the agent (depends on the termination_methods)
        """
        self.custom_state["running_state"] = RunningState.RUNNING

        if reset:
            self.initialize(attachments=attachments, **kwargs)

        self.task = task
        self.attachments = attachments
        self.append_agent_log(
            TaskLog(
                content=self.task,
                attachments=self.attachments,
                timestamp=self.make_timestamp(),
                agent_id=self.agent_id,
            )
        )

        self.logger.debug(f"======== New task for {self.name} ========")
        self.logger.debug(self.task)
        self.logger.debug(
            f"With number of attachments: {len(self.attachments if self.attachments else [])}"
        )

        ret = self.execute_agent_loop()

        return ret

    def log_error(self, e: Exception) -> None:
        """
        Add an error to the last agent log.
        """
        self.logger.warning(f"Agent Error: {str(e)}")
        self.logger.debug(str(e), exc_info=e)
        if isinstance(e, LoggedError):
            category = e.category
        elif isinstance(e, PromptTooLongException) or isinstance(
            e, EmptyResponseException
        ):
            category = type(e).__name__
        else:
            category = "UnhandledError"
        self.append_agent_log(
            ErrorLog(
                error=str(type(e).__name__),
                exception=str(e),
                category=category,
                agent=self.name,
                timestamp=self.make_timestamp(),
                agent_id=self.agent_id,
            )
        )

    def make_timestamp(self) -> float:
        """
        Make a timestamp for the current time.
        """
        return self.time_manager.time() if self.time_manager else time.time()

    def replay(self, start_logs: list[BaseAgentLog]) -> None:
        """
        Reload the state of the agent at a given starting point.
        """
        if len(start_logs) == 0:
            self.logger.debug("Trying to replay an empty log")
            return

        self.logs = start_logs.copy()

        iterations = [log.iteration for log in start_logs if isinstance(log, StepLog)]
        if len(iterations) == 0:
            self.iterations = 0
        else:
            self.iterations = max(iterations) + 1
        for log in reversed(start_logs):
            if isinstance(log, TaskLog):
                self.task = log.content
                self.attachments = log.attachments
                break

        self.planning_counter = 0

        self.initialized = True

    def get_original_tool_name(self, tool_name: str) -> str:
        """
        Get the original tool name from the variant name.
        """
        return self.original_tool_name_to_variant.get(tool_name, tool_name)

    def send_message_to_user(self, content: str) -> None:
        if "AgentUserInterface__send_message_to_user" in self.tools:
            action_executor = self.action_executor
            parsed_action = None

            if type(action_executor) is JsonActionExecutor:
                parsed_action = ParsedAction(
                    action_name="AgentUserInterface__send_message_to_user",
                    tool_name="AgentUserInterface__send_message_to_user",
                    arguments={
                        "content": content,
                    },
                    app_name="AgentUserInterface",
                )

            if parsed_action is None:
                raise NotImplementedError(
                    f"Unknown action executor type: {type(action_executor)}"
                )

            action_executor.execute_parsed_action(
                parsed_action=parsed_action,
                append_agent_log=self.append_agent_log,
                make_timestamp=self.make_timestamp,
                agent_id=self.agent_id,
            )
        else:
            raise ValueError(
                "AgentUserInterface__send_message_to_user is not in agent tools"
            )


def get_offset_from_time_config_mode(
    time_config: SimulatedGenerationTimeConfig,
    completion_duration: float,
) -> float:
    """
    Determine the time to advance based on the mode
    """
    if time_config.mode == "fixed" and time_config.seconds is not None:
        offset = time_config.seconds

    elif time_config.mode == "measured":
        offset = completion_duration

    else:
        raise ValueError(f"Invalid simulated_generation_time_config: {time_config}")

    return offset
