# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import json
import logging

from are.simulation.agents.are_simulation_agent import BaseAgentLog
from are.simulation.apps import SystemApp
from are.simulation.data_handler.models import (
    TRACE_V1_VERSION,
    ExportedHuggingFaceMetadata,
    ExportedTrace,
    ExportedTraceBase,
)
from are.simulation.scenarios.scenario import Scenario, ScenarioStatus
from are.simulation.scenarios.scenario_imported_from_json.benchmark_scenario import (
    BenchmarkScenarioImportedFromJson,
)
from are.simulation.scenarios.scenario_imported_from_json.scenario import (
    ScenarioImportedFromJson,
)
from are.simulation.types import (
    CapabilityTag,
    CompletedEvent,
    ExecutionMetadata,
    Hint,
    HintType,
)

logger = logging.getLogger(__name__)


def _importer() -> type[ExportedTraceBase]:
    return ExportedTrace


class JsonScenarioImporter:
    """
    JSON Scenario Importer.
    """

    SUPPORTED_VERSIONS = [TRACE_V1_VERSION]

    @staticmethod
    def map_action(action_data, app_name_to_class):
        args = (
            {
                arg["name"]: Scenario._parse_parameter_value(
                    arg["value"], arg["value_type"]
                )
                for arg in action_data["args"]
            }
            if action_data["args"] is not None
            else {}
        )

        return {
            "action_id": action_data["action_id"],
            "class_name": (
                app_name_to_class[action_data["app"]]
                if action_data["app"] is not None
                else "ConditionCheckAction"
            ),
            "args": args,
            "function_name": action_data["function"],
            "operation_type": action_data["operation_type"],
        }

    @staticmethod
    def map_event_metadata(metadata_data):
        return {
            "exception": metadata_data["exception"],
            "exception_stack_trace": metadata_data["exception_stack_trace"],
            "return_value": metadata_data["return_value"],
            "return_value_type": metadata_data["return_value_type"],
        }

    @staticmethod
    def map_event(input_json, app_name_to_class):
        mapped_json = {
            "event_id": input_json["event_id"],
            "event_time": input_json["event_time"],
            "event_type": input_json["event_type"],
            "dependencies": input_json["dependencies"],
        }
        metadata = input_json.get("metadata", None)
        if metadata is not None:
            mapped_json["metadata"] = JsonScenarioImporter.map_event_metadata(metadata)
        action = input_json.get("action", None)
        if action is not None:
            mapped_json["action"] = JsonScenarioImporter.map_action(
                action, app_name_to_class
            )
        return mapped_json

    def import_from_json(
        self,
        json_str: str | bytes,
        apps_to_skip: list[str] | None = None,
        apps_to_keep: list[str] | None = None,
        load_completed_events: bool = True,
    ) -> tuple[ScenarioImportedFromJson, list[CompletedEvent], list[BaseAgentLog]]:
        """
        Imports a scenario and associated data from a JSON string or bytes.

        :param json_str: The JSON data representing the scenario.
        :param apps_to_skip: A list of app names to skip during import. If None, no apps are skipped.
        :param apps_to_keep: A list of app names to keep during import. If None, all apps are kept unless skipped.
        :param load_completed_events: Whether to load completed events from the JSON. Defaults to True.
        :returns: A tuple containing:
            - The imported scenario object.
            - A list of completed events.
            - A list of base agent logs.
        """

        data = _importer().model_validate_json(json_str)

        if data.version not in self.SUPPORTED_VERSIONS:
            raise Exception(f"Unsupported version: {data.version}")

        scenario_metadata = data.metadata  # type: ignore
        events = data.events

        app_name_to_class = {app.name: app.class_name for app in data.apps}

        app_name_to_class["SystemApp"] = SystemApp.__name__
        from are.simulation.apps.agent_user_interface import AgentUserInterface

        app_name_to_class["AgentUserInterface"] = AgentUserInterface.__name__
        completed_events = (
            [
                CompletedEvent.from_dict(
                    self.map_event(completed_event.model_dump(), app_name_to_class)
                )
                for completed_event in data.completed_events
            ]
            if load_completed_events
            else []
        )

        scenario = ScenarioImportedFromJson()

        scenario.scenario_id = scenario_metadata.definition.scenario_id

        if scenario_metadata.definition.seed is not None:
            scenario.seed = scenario_metadata.definition.seed

        # Handle run_number if present in the metadata
        if (
            hasattr(scenario_metadata.definition, "run_number")
            and scenario_metadata.definition.run_number is not None
        ):
            scenario.run_number = scenario_metadata.definition.run_number

        if scenario_metadata.definition.start_time is not None:
            scenario.start_time = scenario_metadata.definition.start_time

        if scenario_metadata.definition.duration is not None:
            scenario.duration = scenario_metadata.definition.duration

        if scenario_metadata.definition.time_increment_in_seconds is not None:
            scenario.time_increment_in_seconds = (
                scenario_metadata.definition.time_increment_in_seconds
            )

        scenario.hints = [
            Hint(
                hint_type=HintType(hint.hint_type),
                content=hint.content,
                associated_event_id=hint.associated_event_id,
            )
            for hint in scenario_metadata.definition.hints
        ]

        if hasattr(scenario_metadata.definition, "config"):
            scenario.config = scenario_metadata.definition.config
        if hasattr(scenario_metadata.definition, "has_a2a_augmentation"):
            scenario.has_a2a_augmentation = (
                scenario_metadata.definition.has_a2a_augmentation
            )
        if hasattr(scenario_metadata.definition, "has_tool_augmentation"):
            scenario.has_tool_augmentation = (
                scenario_metadata.definition.has_tool_augmentation
            )
        if hasattr(scenario_metadata.definition, "has_env_events_augmentation"):
            scenario.has_env_events_augmentation = (
                scenario_metadata.definition.has_env_events_augmentation
            )
        if hasattr(scenario_metadata.definition, "has_exception"):
            scenario.has_exception = scenario_metadata.definition.has_exception
        if hasattr(scenario_metadata.definition, "exception_type"):
            scenario.exception_type = scenario_metadata.definition.exception_type
        if hasattr(scenario_metadata.definition, "exception_message"):
            scenario.exception_message = scenario_metadata.definition.exception_message

        if scenario_metadata.definition.tags:
            try:
                scenario.tags = tuple(
                    CapabilityTag[tag] for tag in scenario_metadata.definition.tags
                )
            except KeyError:
                logger.error(
                    f"Invalid tags '{scenario_metadata.definition.tags}' found in scenario metadata."
                )

        if (
            scenario_metadata.annotation
            and scenario_metadata.annotation.validation_decision
        ):
            status = scenario_metadata.annotation.validation_decision
            if status in ScenarioStatus.__members__:
                scenario.status = ScenarioStatus[status]
            elif status is not None and status != "":
                logger.error(f"Invalid status '{status}' found in scenario metadata.")

        if scenario_metadata.annotation and scenario_metadata.annotation.comment:
            reason = scenario_metadata.annotation.comment
            if reason is not None and reason != "":
                scenario.comment = reason

        if scenario_metadata.annotation and scenario_metadata.annotation.annotation_id:
            annotation_id = scenario_metadata.annotation.annotation_id
            if annotation_id is not None and annotation_id != "":
                scenario.annotation_id = annotation_id

        if scenario_metadata.execution is not None:
            scenario.execution_metadata = scenario_metadata.execution.to_metadata()
        else:
            # If execution metadata is missing but there are CONDITION events,
            # provide a default empty ExecutionMetadata so the scenario can initialize.
            has_condition_events = any(
                e.event_type == "CONDITION" for e in events
            )
            if has_condition_events:
                scenario.execution_metadata = ExecutionMetadata(
                    has_placeholder_conflicts=False,
                    placeholders=[],
                )

        scenario.serialized_apps = data.apps
        scenario.apps_to_skip = apps_to_skip
        scenario.apps_to_keep = apps_to_keep
        scenario.serialized_events = events
        scenario.augmentation_data = data.augmentation  # type: ignore

        world_logs = []

        for agent_log_str in data.world_logs:
            agent_log_dict = json.loads(agent_log_str)
            # Ensure agent_id is present for backward compatibility
            if "agent_id" not in agent_log_dict:
                agent_log_dict["agent_id"] = "unknown"
            world_logs.append(BaseAgentLog.from_dict(agent_log_dict))

        return scenario, completed_events, world_logs

    def _fetch_apps_from_huggingface(
        self, hf_metadata: ExportedHuggingFaceMetadata, scenario_id: str
    ) -> dict:
        """
        Fetch apps from HuggingFace dataset when they're missing from the imported trace.

        :param hf_metadata: HuggingFace metadata containing dataset, split, and revision
        :param scenario_id: ID of the scenario to fetch
        :return: Dictionary of apps from the original scenario
        :raises ValueError: If apps cannot be retrieved from HuggingFace
        """
        from datasets import IterableDatasetDict, load_dataset

        try:
            logger.info(
                f"Fetching apps from HuggingFace dataset: {hf_metadata.dataset}/{hf_metadata.split}"
            )
            ds = load_dataset(
                hf_metadata.dataset, revision=hf_metadata.revision, streaming=True
            )

            assert isinstance(ds, IterableDatasetDict), (
                f"Looks like the {hf_metadata.dataset} is not split by capabilities."
            )

            # Find the scenario with matching ID in the dataset
            for row in ds[hf_metadata.split]:
                if row.get("scenario_id") == scenario_id:
                    # Extract apps from the data
                    original_data = json.loads(row["data"])
                    if "apps" in original_data and original_data["apps"]:
                        logger.info(
                            f"Successfully retrieved apps for scenario {scenario_id}"
                        )
                        return original_data["apps"]

            raise ValueError(
                f"Could not find scenario {scenario_id} in dataset {hf_metadata.dataset}/{hf_metadata.split}"
            )
        except Exception as e:
            raise ValueError(f"Error fetching apps from HuggingFace: {str(e)}") from e

    def import_from_json_to_benchmark(
        self,
        json_str: str | bytes,
        apps_to_skip: list[str] | None = None,
        apps_to_keep: list[str] | None = None,
        load_completed_events: bool = True,
    ) -> tuple[
        BenchmarkScenarioImportedFromJson, list[CompletedEvent], list[BaseAgentLog]
    ]:
        _scenario, completed_events, world_logs = self.import_from_json(
            json_str=json_str,
            apps_to_skip=apps_to_skip,
            apps_to_keep=apps_to_keep,
            load_completed_events=load_completed_events,
        )
        scenario = BenchmarkScenarioImportedFromJson()

        scenario.scenario_id = _scenario.scenario_id
        scenario.run_number = _scenario.run_number
        scenario.seed = _scenario.seed
        scenario.start_time = _scenario.start_time
        scenario.duration = _scenario.duration
        scenario.time_increment_in_seconds = _scenario.time_increment_in_seconds
        scenario.hints = _scenario.hints
        scenario.tags = _scenario.tags
        scenario.execution_metadata = _scenario.execution_metadata

        # Check if we need to fetch apps from HuggingFace
        # If apps are empty and HuggingFace metadata is available, try to fetch apps
        if not _scenario.serialized_apps:
            # Get the metadata from the already parsed data
            data = _importer().model_validate_json(json_str)
            scenario_metadata = data.metadata  # type: ignore

            if scenario_metadata.definition.hf_metadata:
                hf_metadata = scenario_metadata.definition.hf_metadata

                # Store HuggingFace metadata in the scenario
                scenario.hf_metadata = hf_metadata

                # Try to fetch apps from HuggingFace
                fetched_apps = self._fetch_apps_from_huggingface(
                    hf_metadata, scenario.scenario_id
                )
                _scenario.serialized_apps = fetched_apps
                logger.info(
                    f"Using apps from HuggingFace for scenario {scenario.scenario_id}"
                )

        scenario.serialized_apps = _scenario.serialized_apps
        scenario.apps_to_skip = _scenario.apps_to_skip
        scenario.apps_to_keep = _scenario.apps_to_keep
        scenario.serialized_events = _scenario.serialized_events
        scenario.augmentation_data = _scenario.augmentation_data

        return scenario, completed_events, world_logs
