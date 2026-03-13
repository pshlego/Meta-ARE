# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import base64
import json
import logging
import os
import shutil
from functools import lru_cache
from mimetypes import guess_type
from threading import Thread
from typing import Any, Callable

from are.simulation.agents.agent_builder import AbstractAgentBuilder
from are.simulation.agents.agent_config_builder import AbstractAgentConfigBuilder
from are.simulation.agents.agent_log import (
    BaseAgentLog,
    LLMOutputFactsLog,
    LLMOutputPlanLog,
    LLMOutputThoughtActionLog,
    ObservationLog,
    SystemPromptLog,
    TaskLog,
)
from are.simulation.agents.are_simulation_agent import RunnableARESimulationAgent
from are.simulation.agents.are_simulation_agent_config import (
    RunnableARESimulationAgentConfig,
)
from are.simulation.agents.multimodal import Attachment
from are.simulation.apps import AgentUserInterface, Files, SandboxLocalFileSystem
from are.simulation.config import ARE_SIMULATION_SANDBOX_PATH
from are.simulation.data_handler.exporter import JsonScenarioExporter
from are.simulation.data_handler.importer import JsonScenarioImporter
from are.simulation.environment import Environment, EnvironmentConfig
from are.simulation.gui.server.api_errors import APIScenarioNotFoundError
from are.simulation.gui.server.graphql.types import (
    AnnotationForGraphQL,
    BugReportForGraphQL,
)
from are.simulation.gui.server.utils import (
    file_contents_img,
    file_contents_pdf,
    file_contents_txt,
    validate_attachment_sizes,
)
from are.simulation.notification_system import BaseNotificationSystem
from are.simulation.replay import (
    find_log_position,
    find_previous_step_position,
    replay_logs,
)
from are.simulation.scenarios.scenario import Scenario
from are.simulation.types import (
    CapabilityTag,
    CompletedEvent,
    EnvironmentState,
    EnvironmentType,
    EventTimeComparator,
    EventType,
    disable_events,
)
from are.simulation.utils.huggingface import (
    get_scenario_from_huggingface,
    parse_huggingface_url,
)

# There is no open implementation for db manager.
DatabaseManagerType = None

logger = logging.getLogger(__name__)


class ARESimulationGui:
    def __init__(
        self,
        session_id: str,
        scenario_id: str | None,
        agent_config_builder: AbstractAgentConfigBuilder,
        agent_builder: AbstractAgentBuilder,
        scenario_args: dict[str, Any] = {},
        default_agent_name: str | None = None,
        default_model_name: str | None = None,
        default_provider: str | None = None,
        default_endpoint: str | None = None,
        annotator_name: str | None = None,
        notification_system_builder: Callable[[], BaseNotificationSystem] | None = None,
        db_manager: DatabaseManagerType = None,  # type: ignore[reportInvalidTypeForm]
        dataset_path: str | None = None,
    ):
        self.session_id: str = session_id
        self.are_simulation_agent: RunnableARESimulationAgent | None = None  # type: ignore[reportGeneralTypeIssues]
        self.scenario_args: dict[str, Any] = scenario_args
        self.agent_config_builder: AbstractAgentConfigBuilder = agent_config_builder
        self.agent_builder: AbstractAgentBuilder = agent_builder
        self.agent_name: str | None = default_agent_name

        self.default_model_name = default_model_name
        self.default_provider = default_provider
        self.default_endpoint = default_endpoint
        self.dataset_path: str | None = dataset_path

        self.agent_config: RunnableARESimulationAgentConfig | None = (
            None
            if not default_agent_name
            else self.agent_config_builder.build(default_agent_name)
        )
        self.apply_agent_config_defaults()

        self.env: Environment | None = None
        self.annotator_name = annotator_name
        self.notification_system_builder = notification_system_builder

        self.tmpdir: str = os.path.join(
            ARE_SIMULATION_SANDBOX_PATH, self.session_id, ""
        )
        if not os.path.exists(self.tmpdir):
            os.makedirs(self.tmpdir, exist_ok=True)
            logger.info(f"Created session's temporary directory: {self.tmpdir}")

        # Import it inplace, since it has side effects.
        from are.simulation.gui.server.scenarios import GUI_SCENARIOS

        # This is used to store events that are sent before the environment is initialized
        self.log_queue = []
        default_scenario_id = (
            scenario_id if scenario_id else next(iter(GUI_SCENARIOS.keys()))
        )
        self.set_scenario(default_scenario_id)

    def __del__(self):
        if os.path.exists(self.tmpdir):
            logger.info(f"Removing session's temporary directory: {self.tmpdir}")
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def get_agent_name(self) -> str | None:
        return (
            self.agent_config.get_agent_name()
            if self.agent_config is not None
            else None
        )

    def get_model_name(self) -> str | None:
        return (
            self.agent_config.get_base_agent_config().llm_engine_config.model_name
            if self.agent_config is not None
            else None
        )

    def replay_turn(self, log_id: str, task: str | None = None):
        world_logs = self.get_world_logs()
        new_world_logs = []

        for log in world_logs:
            if log.id == log_id:
                break
            new_world_logs.append(log)

        self.replay(initial_world_logs=new_world_logs, task=task, mock_responses=None)

    def replay_modified_output(self, log_id: str, content: str):
        world_logs = self.get_world_logs()

        editable_types = (
            LLMOutputThoughtActionLog,
            LLMOutputFactsLog,
            LLMOutputPlanLog,
            ObservationLog,
            SystemPromptLog,
        )
        log_id_pos = find_log_position(world_logs, log_id, editable_types)

        if log_id_pos == -1:
            raise ValueError(f"Log with id {log_id} not found.")

        modified_log = world_logs[log_id_pos]

        if isinstance(modified_log, SystemPromptLog):
            if log_id_pos + 1 >= len(world_logs) or not isinstance(
                world_logs[log_id_pos + 1], TaskLog
            ):
                raise ValueError("Expected TaskLog to follow the system prompt.")

            new_world_logs = world_logs[: log_id_pos + 2]
            modified_log.content = content
            mock_responses = None

        elif isinstance(modified_log, ObservationLog):
            new_world_logs = world_logs[: log_id_pos + 1]
            modified_log.content = content
            mock_responses = None

        else:
            prev_step_pos = find_previous_step_position(world_logs, log_id_pos)
            new_world_logs = world_logs[:prev_step_pos] if prev_step_pos != -1 else []

            llm_output_types = (
                LLMOutputThoughtActionLog,
                LLMOutputFactsLog,
                LLMOutputPlanLog,
            )

            mock_responses = [
                content if log.id == log_id else log.content
                for log in world_logs[prev_step_pos : log_id_pos + 1]
                if isinstance(log, llm_output_types)
            ]

        self.replay(
            initial_world_logs=new_world_logs,
            task=None,
            mock_responses=mock_responses,
        )

    def replay(
        self,
        initial_world_logs: list[BaseAgentLog],
        task: str | None = None,
        mock_responses: list[str] | None = None,
    ):
        self.stop()
        self.load()
        self.play(
            initial_world_logs=initial_world_logs,
            task=task,
            mock_responses=mock_responses,
        )

    def get_world_logs(self) -> list[BaseAgentLog]:
        return self.env.get_world_logs() if self.env is not None else []

    def set_scenario(
        self, scenario_id: str, initial_world_logs: list[BaseAgentLog] | None = None
    ) -> None:
        # Check if this is a HuggingFace URL
        if scenario_id.startswith("hf://datasets/"):
            hf_params = parse_huggingface_url(scenario_id)
            if hf_params is None:
                raise APIScenarioNotFoundError(
                    f"Invalid HuggingFace URL format: '{scenario_id}'. Expected format: hf://datasets/dataset_name/config/split/scenario_id"
                )

            # Import scenario from HuggingFace
            success = self.import_from_huggingface(
                dataset_name=hf_params["dataset_name"],
                dataset_config=hf_params["config"],
                dataset_split=hf_params["split"],
                scenario_id=hf_params["scenario_id"],
            )

            if not success:
                raise APIScenarioNotFoundError(
                    f"Failed to load scenario '{hf_params['scenario_id']}' from HuggingFace dataset '{hf_params['dataset_name']}/{hf_params['config']}/{hf_params['split']}'"
                )

            if initial_world_logs is not None:
                self.play(initial_world_logs)
            return

        # Import it inplace, since it has side effects.
        from are.simulation.gui.server.scenarios import GUI_SCENARIOS

        # Handle regular scenario IDs from GUI_SCENARIOS
        if scenario_id not in GUI_SCENARIOS:
            raise APIScenarioNotFoundError(
                f"Scenario with id '{scenario_id}' does not exist."
            )
        self.stop()
        self.scenario: Scenario = GUI_SCENARIOS[scenario_id]()
        self.scenario_args = {}  # we don't support scenario args from the UI yet
        self.scenario.initialize(sandbox_dir=self.tmpdir)
        self._post_load()

        if initial_world_logs is not None:
            self.play(initial_world_logs)

    def set_agent_name(self, agent_id: str | None) -> None:
        self.agent_name = agent_id
        self.agent_config = (
            None
            if not self.agent_name
            else self.agent_config_builder.build(self.agent_name)
        )
        self.apply_agent_config_defaults()
        self._configure_environment()
        logger.info(f"Agent set to {self.agent_name}")

    def apply_agent_config_defaults(self) -> None:
        # Adjust the agent config with default values if specified.
        if self.agent_config is None:
            return
        if self.default_model_name is not None:
            self.agent_config.get_base_agent_config().llm_engine_config.model_name = (
                self.default_model_name
            )
        if self.default_provider is not None:
            self.agent_config.get_base_agent_config().llm_engine_config.provider = (
                self.default_provider
            )
        if self.default_endpoint is not None:
            self.agent_config.get_base_agent_config().llm_engine_config.endpoint = (
                self.default_endpoint
            )

    def get_agent_config_with_schema(self) -> dict[str, Any] | None:
        return (
            {
                "schema": self.agent_config.get_model_json_schema(),
                "value": self.agent_config.get_model_dump(),
            }
            if self.agent_config is not None
            else None
        )

    def set_agent_config(self, agent_config_dict: dict[str, Any] | None) -> None:
        self.agent_config = (
            None
            if agent_config_dict is None or self.agent_config is None
            # Construct agent config from JSON.
            else self.agent_config.validate_model(agent_config_dict)
        )
        self._configure_environment()
        agent_name = (
            None if self.agent_config is None else self.agent_config.get_agent_name()
        )
        logger.info(f"Agent config for {agent_name} updated.")

    def set_annotator_name(self, name: str | None) -> None:
        self.annotator_name = name

    def set_override_phone_number(self, override_phone_number: str | None) -> None:
        pass

    def get_state(self) -> dict[str, Any]:
        if self.env is None:
            raise ValueError("Environment is not initialized.")
        return self.env.get_state()

    def pause(self):
        if self.env is not None:
            self.env.pause()

    def play(
        self,
        initial_world_logs: list[BaseAgentLog] | None = None,
        task: str | None = None,
        mock_responses: list[str] | None = None,
    ):
        if self.env is None:
            raise ValueError("Environment is not initialized.")

        if self.env.state == EnvironmentState.PAUSED:
            self.env.resume()
            return

        schedule_events = initial_world_logs is None
        self.env.run(self.scenario, wait_for_end=False, schedule_events=schedule_events)

        if initial_world_logs is not None:
            replay_logs(initial_world_logs, self.env)
            if task is not None:
                self.send_user_message_to_agent(task, None)

        self._start_agent(initial_world_logs, mock_responses)

    def load(self):
        self.scenario.soft_reset()
        self._post_load()

    def _post_load(
        self, completed_events: CompletedEvent | list[CompletedEvent] | None = None
    ) -> None:
        if self.notification_system_builder is None:
            logger.warning("No notification system builder provided - using default.")
            self.notification_system = None
        else:
            self.notification_system = self.notification_system_builder()
            logger.info(
                f"Using provided notification system builder {self.notification_system}."
            )

        config = EnvironmentConfig()
        if self.scenario.start_time and self.scenario.start_time > 0:
            config.start_time = self.scenario.start_time
        self.env = Environment(
            environment_type=EnvironmentType.GUI,
            notification_system=self.notification_system,
            config=config,
        )
        self._configure_environment()
        self.env.register_apps(self.scenario.apps if self.scenario.apps else [])

        if completed_events is not None:
            self.env.add_to_log(completed_events)

    def add_scenario_event(
        self,
        app_name: str,
        function_name: str,
        parameters: str,
        predecessor_event_ids: list[str],
        event_type: EventType,
        event_relative_time: float | None = None,
        event_time: float | None = None,
        event_time_comparator: EventTimeComparator | None = None,
    ) -> str:
        params = json.loads(parameters)

        event = self.scenario.add_event(
            app_name=app_name,
            function_name=function_name,
            parameters=params,
            predecessor_event_ids=predecessor_event_ids,
            event_type=event_type,
            event_id=None,
            event_relative_time=event_relative_time,
            event_time=event_time,
            event_time_comparator=event_time_comparator,
        )

        return event.event_id

    def edit_scenario_event(
        self,
        app_name: str,
        function_name: str,
        parameters: str,
        event_id: str,
        event_type: EventType,
        predecessor_event_ids: list[str],
        event_relative_time: float | None = None,
        event_time: float | None = None,
        event_time_comparator: EventTimeComparator | None = None,
    ) -> str:
        params = json.loads(parameters)

        event = self.scenario.edit_event(
            app_name=app_name,
            function_name=function_name,
            parameters=params,
            event_id=event_id,
            event_type=event_type,
            predecessor_event_ids=predecessor_event_ids,
            event_relative_time=event_relative_time,
            event_time=event_time,
            event_time_comparator=event_time_comparator,
        )

        return event.event_id

    def edit_scenario_duration(
        self,
        duration: float | None,
    ) -> None:
        if self.scenario is None:
            logger.error("Cannot edit scenario duration: scenario is None.")
            return
        self.scenario.set_duration(duration)
        logger.info(f"Scenario Duration set to {duration} seconds.")

    def edit_scenario_time_increment(self, time_increment_in_seconds: int) -> None:
        if self.scenario is None:
            logger.error("Cannot edit time increment: scenario is None.")
            return
        self.scenario.set_time_increment(time_increment_in_seconds)
        logger.info(
            f"Scenario Time increment set to {time_increment_in_seconds} seconds."
        )

    def delete_scenario_event(
        self,
        event_id: str,
    ) -> None:
        if self.scenario is None:
            logger.error("Cannot delete scenario event: scenario is None.")
            return
        self.scenario.delete_event(event_id)

    def delete_all_scenario_events(self) -> None:
        if self.scenario is None:
            logger.error("Cannot delete all scenario events: scenario is None.")
            return
        self.scenario.delete_completed_events()
        assert self.env is not None, "Environment not found."
        self.env.delete_all_completed_events()

    def edit_scenario_event_hint_content(self, event_id: str, hints: str) -> None:
        if self.scenario is None:
            logger.error("Cannot edit scenario event hint content: scenario is None.")
            return
        self.scenario.edit_hint_content(event_id, hints)

    def stop(self):
        if self.env is not None:
            self.env.stop()
        if self.are_simulation_agent is not None:
            self.are_simulation_agent.stop()

    def send_user_message_to_agent(
        self, message: str, attachments: list[tuple[str, str]] | None
    ) -> str:
        assert self.env is not None, "Environment not found."
        aui = self.env.apps.get(AgentUserInterface.__name__, None)
        if aui is None:
            return ""

        message_attachments = []
        base64_utf8_encoded_attachment_contents: list[dict[str, Any]] = []

        if attachments:
            # Server-side file validation
            validate_attachment_sizes(attachments)
            fs = self.env.apps.get(
                SandboxLocalFileSystem.__name__, None
            ) or self.env.apps.get(Files.__name__, None)

            if not fs:
                raise ValueError("Filesystem not found, but attachments are provided.")
            with disable_events():
                fs.makedirs("Attachments", exist_ok=True)
                for index, (file_name, base64_data) in enumerate(attachments):
                    file_path = "Attachments/" + file_name
                    message_attachments.append(file_path)
                    attachment: dict[str, Any] = {}
                    with fs.open(file_path, "wb") as f:
                        if base64_data:
                            file_data = base64.b64decode(base64_data)
                            f.write(file_data)
                            mime_type, _ = guess_type(file_name)
                            attachment = Attachment(
                                base64_data=base64_data.encode(),
                                mime=mime_type,
                                name=file_name,
                            ).model_dump()
                        else:
                            logger.error(f"No base64 data for attachment {file_name}.")
                    base64_utf8_encoded_attachment_contents.append(attachment)
                    message += f"\n\nAttachment {index}: {file_path}"

        message_id = aui.send_message_to_agent(
            message, message_attachments, base64_utf8_encoded_attachment_contents
        )
        self.env.resume()

        return message_id

    def send_conversation_message(
        self, messaging_app_name: str, conversation_id: str, sender: str, message: str
    ) -> bool:
        assert self.env is not None, "Environment not found."
        messaging_app = self.env.apps.get(messaging_app_name, None)
        if messaging_app is not None:
            messaging_app.add_message(conversation_id, sender, message)
            return True
        return False

    def report_bug(
        self,
        scenario_id: str | None,
        reporter: str,
        context: str | None,
    ) -> str | None:
        raise NotImplementedError("Bug reporting is not implemented.")

    def export_trace(
        self,
        scenario_id: str,
        validation_decision: str | None,
        annotation_id: str | None,
        annotator_name: str | None,
        comment: str | None = None,
        tags: list[CapabilityTag] | None = None,
    ) -> str | None:
        env: Environment | None = self.env
        try:
            if not env:
                raise RuntimeError("No environment to export.")

            if tags:
                self.scenario.tags = tuple(set(list(self.scenario.tags) + tags))

            scenario_exporter = JsonScenarioExporter()

            return scenario_exporter.export_to_json(
                env=env,
                scenario=self.scenario,
                scenario_id=scenario_id or "unknown_scenario",
                model_id=self.get_model_name(),
                agent_id=self.get_agent_name(),
                validation_decision=validation_decision,
                annotation_id=annotation_id,
                annotator_name=annotator_name,
                comment=comment,
                world_logs=self.get_world_logs(),
            )
        except Exception as e:
            logger.exception(f"Failed to export trace: {e}.")
            return None

    def export_annotated_trace(
        self,
        scenario_id: str,
        category: str | None,
        is_violating: bool | None = None,
        has_ars_violation: bool | None = None,
        has_crs_violation: bool | None = None,
        is_false_refusal: bool | None = None,
        annotation_id: str | None = None,
        annotator_name: str | None = None,
        group_id: str | None = None,
        attack_category: str | None = None,
        references: str | None = None,
        additional_notes: str | None = None,
    ) -> str | None:
        env: Environment | None = self.env
        try:
            if not env:
                raise RuntimeError("No environment to export.")

            apps_state: dict[str, Any] | None = None
            scenario_exporter = JsonScenarioExporter()

            return scenario_exporter.export_to_json(
                env=env,
                scenario=self.scenario,
                scenario_id=scenario_id,
                model_id=self.get_model_name(),
                agent_id=self.get_agent_name(),
                category=category,
                is_violating=is_violating,
                has_ars_violation=has_ars_violation,
                has_crs_violation=has_crs_violation,
                is_false_refusal=is_false_refusal,
                annotation_id=annotation_id,
                annotator_name=annotator_name,
                apps_state=apps_state,
                group_id=group_id,
                attack_category=attack_category,
                references=references,
                additional_notes=additional_notes,
                world_logs=self.get_world_logs(),
            )
        except Exception as e:
            logger.exception(f"Failed to export trace: {e}.")
            return None

    def save_annotation(
        self,
        scenario_id: str,
        validation_decision: str | None,
        annotation_id: str | None,
        annotator_name: str | None,
        comment: str | None = None,
    ) -> bool:
        raise NotImplementedError("Annotation saving to DB is not implemented.")

    def get_file_content(self, filesystem_app_name: str, file_path: str) -> str:
        assert self.env is not None, "Environment not found"
        fs = self.env.apps.get(filesystem_app_name, None)
        assert fs is not None, f"Filesystem app {filesystem_app_name} not found."
        file_extension = file_path.split(".")[-1]
        if file_path.startswith("/"):
            # Remove the leading slash
            sandbox_path = file_path[1:]
        else:
            # Backward compatibility with old file paths including the sandbox prefix
            sandbox_path = "/".join(file_path.split("/")[1:])
        try:
            if file_extension == "pdf":
                return file_contents_pdf(fs, sandbox_path)
            elif file_extension in {"txt", "json", "csv"}:
                return file_contents_txt(fs, sandbox_path)
            elif file_extension in {
                "png",
                "jpg",
                "jpeg",
                "gif",
                "webp",
                "bmp",
            }:
                return file_contents_img(fs, sandbox_path)
            else:
                return "Unsupported file type for now: but you can use the download button!"
        except Exception as e:
            logger.exception(
                f"Failed to read file fs:{filesystem_app_name} = {file_path}.",
                exc_info=e,
            )
            return f"{e}"

    def download_file(self, filesystem_app_name: str, file_path: str) -> str:
        assert self.env is not None, "Environment not found."
        fs = self.env.apps.get(filesystem_app_name, None)
        assert fs is not None, f"Filesystem app {filesystem_app_name} not found."
        if file_path.startswith("/"):
            # Remove the leading slash
            sandbox_path = file_path[1:]
        else:
            # Backward compatibility with old file paths including the sandbox prefix
            sandbox_path = "/".join(file_path.split("/")[1:])
        try:
            binary = fs.get_file_content(sandbox_path)
            return base64.b64encode(binary).decode("utf-8")
        except Exception as e:
            logger.exception(
                f"Failed to read file fs:{filesystem_app_name} = {file_path}.",
                exc_info=e,
            )
            return f"{e}"

    def upload_file(
        self,
        filesystem_app_name: str,
        file_name: str,
        file_content: str,
        destination_path: str,
    ) -> bool:
        """
        Upload a file to the filesystem.

        :param filesystem_app_name: The name of the filesystem app
        :param file_name: The name of the file to upload
        :param file_content: The base64-encoded content of the file
        :param destination_path: The destination path where the file should be saved

        :return: True if the upload was successful, False otherwise
        """
        assert self.env is not None, "Environment not found."
        fs = self.env.apps.get(filesystem_app_name, None)
        assert fs is not None, f"Filesystem app {filesystem_app_name} not found."

        try:
            # Decode the base64 content
            file_data = base64.b64decode(file_content)

            # Ensure destination path starts with /
            if not destination_path.startswith("/"):
                destination_path = "/" + destination_path

            # Remove leading slash for filesystem operations
            if destination_path.startswith("/"):
                sandbox_destination = destination_path[1:]
            else:
                sandbox_destination = destination_path

            # Create the full file path
            full_file_path = os.path.join(sandbox_destination, file_name)

            with disable_events():
                # Create the destination directory if it doesn't exist
                destination_dir = os.path.dirname(full_file_path)
                if destination_dir:
                    fs.makedirs(destination_dir, exist_ok=True)

                # Write the file
                with fs.open(full_file_path, "wb") as f:
                    f.write(file_data)

            logger.info(
                f"Successfully uploaded file {file_name} to {full_file_path} in filesystem {filesystem_app_name}"
            )
            return True

        except Exception as e:
            logger.exception(
                f"Failed to upload file {file_name} to {destination_path} in filesystem {filesystem_app_name}: {e}"
            )
            return False

    def import_trace(self, scenario_json: str, replay_logs: bool = True) -> bool:
        try:
            if self.env and self.env.is_running():
                raise Exception(
                    "Cannot import scenario while the environment is running."
                )

            scenario_importer = JsonScenarioImporter()
            scenario, completed_events, world_logs = scenario_importer.import_from_json(
                scenario_json
            )
            self.scenario = scenario
            self.scenario.initialize(sandbox_dir=self.tmpdir)

            if replay_logs:
                self._post_load(completed_events)
                if len(world_logs) > 0:
                    self.play(world_logs)
                    # Restore completed_events after play(), because env.run()
                    # calls delete_all_completed_events() which clears them.
                    if completed_events is not None:
                        self.env.add_to_log(completed_events)
            else:
                self._post_load()

            return True
        except Exception as e:
            logger.exception(f"Failed to import scenario: {e}.")
            return False

    def import_from_huggingface(
        self,
        dataset_name: str,
        dataset_config: str,
        dataset_split: str,
        scenario_id: str,
    ) -> bool:
        """
        Import a scenario from a Huggingface dataset.

        Args:
            dataset_name: The name of the Huggingface dataset
            dataset_config: The config name of the Huggingface dataset
            dataset_split: The split name of the Huggingface dataset
            scenario_id: The ID of the scenario to import

        Returns:
            True if the import was successful, False otherwise
        """
        try:
            if self.env and self.env.is_running():
                raise Exception(
                    "Cannot import scenario while the environment is running."
                )

            logger.info(
                f"Importing scenario {scenario_id} from Huggingface dataset {dataset_name}/{dataset_config}/{dataset_split}"
            )

            # Get scenario data from HuggingFace
            scenario_data = get_scenario_from_huggingface(
                dataset_name, dataset_config, dataset_split, scenario_id
            )

            if scenario_data is None:
                logger.error(
                    f"Failed to get scenario {scenario_id} from dataset {dataset_name}/{dataset_config}/{dataset_split}"
                )
                return False

            # Import the scenario
            scenario_importer = JsonScenarioImporter()
            scenario, completed_events, world_logs = scenario_importer.import_from_json(
                scenario_data
            )

            # Set up the scenario
            self.scenario = scenario
            self.scenario.initialize(sandbox_dir=self.tmpdir)
            self._post_load(completed_events)

            # Play the scenario if there are world logs
            if len(world_logs) > 0:
                self.play(world_logs)

            return True
        except Exception as e:
            logger.exception(f"Failed to import scenario from Huggingface: {e}")
            return False

    def _configure_environment(self) -> None:
        oracle_mode = self.agent_config is None
        assert self.env is not None, "Environment not found."
        self.env.oracle_mode = oracle_mode
        self.env.queue_based_loop = oracle_mode

    def _start_agent(
        self,
        initial_world_logs: list[BaseAgentLog] | None = None,
        mock_responses: list[str] | None = None,
    ) -> None:
        if self.agent_name is None:
            logger.info("No agent name specified, the agent will not be started.")
            return

        if self.agent_config is None:
            logger.error("Cannot _start_agent, Agent config is None.")
            return

        if self.env is None:
            logger.error("Cannot _start_agent, Environment is None.")
            return

        self.are_simulation_agent: RunnableARESimulationAgent = (
            self.agent_builder.build(self.agent_config, self.env, mock_responses)
        )

        logger.info(
            f"Running Scenario with agent of type '{type(self.are_simulation_agent).__name__}'."
        )

        def run_agent():
            self.are_simulation_agent.run_scenario(
                scenario=self.scenario,
                notification_system=self.env.notification_system,  # type: ignore
                initial_agent_logs=initial_world_logs,
            )

        agent_thread = Thread(target=run_agent, name="Agent")
        agent_thread.daemon = True
        agent_thread.start()

    def get_annotations_from_db(self) -> list[AnnotationForGraphQL]:
        raise NotImplementedError("Annotation fetching from DB is not implemented.")

    def get_bug_reports_from_db(
        self, with_blob=False, where_id=None, limit=20, offset=0
    ) -> list[BugReportForGraphQL]:
        raise NotImplementedError("Bug report fetching from DB is not implemented.")

    def import_from_db(self, scenario_id: str) -> bool:
        raise NotImplementedError("Importing from DB is not implemented.")

    def save_annotated_trace_to_db(
        self,
        scenario_id: str,
        category: str | None = None,
        is_violating: bool | None = None,
        has_ars_violation: bool | None = None,
        has_crs_violation: bool | None = None,
        is_false_refusal: bool | None = None,
        annotation_id: str | None = None,
        annotator_name: str | None = None,
        group_id: str | None = None,
        attack_category: str | None = None,
        references: str | None = None,
        additional_notes: str | None = None,
    ) -> str | None:
        raise NotImplementedError("Integration is not implemented.")

    def _validate_dataset_path(
        self, capability: str | None, filename: str | None = None
    ) -> str | None:
        """
        Validate and resolve a path within the dataset directory to prevent path traversal attacks.

        Args:
            capability: The capability (subfolder) name, or None for files directly in dataset path
            filename: Optional filename to include in the path

        Returns:
            The validated absolute path if safe, None if validation fails
        """
        if self.dataset_path is None:
            logger.error("No dataset path configured.")
            return None

        try:
            # Start with the base dataset path (resolve to absolute path)
            base_path = os.path.abspath(self.dataset_path)

            # Build the target path using normal path construction (potentially unsafe)
            if capability is None:
                target_path = base_path
            else:
                target_path = os.path.join(base_path, capability)

            # Add filename if provided
            if filename is not None:
                target_path = os.path.join(target_path, filename)

            # Resolve the final absolute path and check if it's within the allowed directory
            resolved_path = os.path.abspath(target_path)

            # Ensure the resolved path is within the dataset directory
            # Use os.path.commonpath to handle edge cases properly
            try:
                common_path = os.path.commonpath([base_path, resolved_path])
                if common_path != base_path:
                    logger.error(
                        f"Path traversal attempt detected: {resolved_path} is outside {base_path}"
                    )
                    return None
            except ValueError:
                # os.path.commonpath raises ValueError if paths are on different drives (Windows)
                logger.error(
                    f"Path traversal attempt detected: {resolved_path} is outside {base_path}"
                )
                return None

            return resolved_path

        except Exception as e:
            logger.exception(f"Failed to validate dataset path: {e}")
            return None

    @staticmethod
    @lru_cache(maxsize=128)
    def _get_capabilities_cached(dataset_path: str, mtime: float) -> list[str | None]:
        """
        Static cached method to get capabilities. Uses mtime as part of cache key for invalidation.
        """
        try:
            capabilities = []
            has_json_files = False

            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    capabilities.append(item)
                elif item.lower().endswith(".json"):
                    has_json_files = True

            # If no subdirectories but JSON files exist directly in dataset path,
            # return None to indicate root-level files
            if not capabilities and has_json_files:
                return [None]
            else:
                capabilities.sort()  # Sort alphabetically for consistent ordering
                return capabilities
        except Exception as e:
            logger.exception(f"Failed to list dataset capabilities: {e}")
            return []

    def get_local_json_dataset_capabilities(self) -> list[str]:
        """
        Get the list of capabilities (subfolders) in the local JSON dataset directory.
        If no subdirectories exist, returns an empty list.

        Returns:
            A list of capability names (subfolder names)
        """
        if self.dataset_path is None:
            logger.warning("No dataset path configured.")
            return []

        if not os.path.exists(self.dataset_path):
            logger.warning(f"Dataset path does not exist: {self.dataset_path}")
            return []

        # Use modification time as part of cache key for automatic invalidation
        mtime = os.path.getmtime(self.dataset_path)
        capabilities = self._get_capabilities_cached(self.dataset_path, mtime)
        # Filter out None values to match return type
        return [cap for cap in capabilities if cap is not None]

    @staticmethod
    @lru_cache(maxsize=128)
    def _get_files_cached(capability_path: str, mtime: float) -> list[str]:
        """
        Static cached method to get JSON files. Uses mtime as part of cache key for invalidation.
        """
        try:
            json_files = []
            for item in os.listdir(capability_path):
                if item.lower().endswith(".json"):
                    json_files.append(item)

            json_files.sort()  # Sort alphabetically for consistent ordering
            return json_files
        except Exception as e:
            logger.exception(f"Failed to list JSON files in {capability_path}: {e}")
            return []

    def get_local_json_dataset_files(self, capability: str | None) -> list[str]:
        """
        Get the list of JSON files in a specific capability folder or directly in the dataset path.

        Args:
            capability: The capability (subfolder) name, or None for files directly in dataset path

        Returns:
            A list of JSON filenames in the capability folder or dataset root
        """
        if self.dataset_path is None:
            logger.warning("No dataset path configured.")
            return []

        # Validate the capability path to prevent path traversal
        capability_path = self._validate_dataset_path(capability)
        if capability_path is None:
            logger.error(f"Invalid or unsafe capability path: {capability}")
            return []

        if not os.path.exists(capability_path):
            logger.warning(f"Capability path does not exist: {capability_path}")
            return []

        if not os.path.isdir(capability_path):
            logger.warning(f"Capability path is not a directory: {capability_path}")
            return []

        # Use modification time as part of cache key for automatic invalidation
        mtime = os.path.getmtime(capability_path)
        return self._get_files_cached(capability_path, mtime)

    def import_from_local_json_dataset(
        self, capability: str | None, filename: str, replay_logs: bool = True
    ) -> bool:
        """
        Import a scenario from a local JSON dataset file.

        Args:
            capability: The capability (subfolder) name, or None for files directly in dataset path
            filename: The JSON filename to import
            replay_logs: Whether to replay the logs after importing

        Returns:
            True if the import was successful, False otherwise
        """
        if self.dataset_path is None:
            logger.error("No dataset path configured.")
            return False

        try:
            if self.env and self.env.is_running():
                raise Exception(
                    "Cannot import scenario while the environment is running."
                )

            # Validate the file path to prevent path traversal attacks
            file_path = self._validate_dataset_path(capability, filename)
            if file_path is None:
                logger.error(
                    f"Invalid or unsafe file path: capability={capability}, filename={filename}"
                )
                return False

            if not os.path.exists(file_path):
                logger.error(f"Dataset file does not exist: {file_path}")
                return False

            if not os.path.isfile(file_path):
                logger.error(f"Dataset path is not a file: {file_path}")
                return False

            logger.info(f"Importing scenario from local dataset file: {file_path}")

            # Read the JSON file content
            with open(file_path, "r", encoding="utf-8") as f:
                scenario_json = f.read()

            # Import the scenario using the same pattern as import_from_huggingface
            scenario_importer = JsonScenarioImporter()
            scenario, completed_events, world_logs = scenario_importer.import_from_json(
                scenario_json
            )

            # Set up the scenario
            self.scenario = scenario
            self.scenario.initialize(sandbox_dir=self.tmpdir)
            self._post_load(completed_events)

            # Play the scenario if there are world logs and replay_logs is True
            if replay_logs and len(world_logs) > 0:
                self.play(world_logs)

            return True

        except Exception as e:
            logger.exception(f"Failed to import scenario from local dataset: {e}")
            return False
