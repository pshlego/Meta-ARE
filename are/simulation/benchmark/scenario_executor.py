# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging

from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
from are.simulation.benchmark.scenario_loader import setup_scenarios_iterator
from are.simulation.multi_scenario_runner import MultiScenarioRunner
from are.simulation.scenarios.config import (
    DEFAULT_SCENARIO_TIMEOUT,
    MultiScenarioRunnerConfig,
)
from are.simulation.scenarios.scenario_imported_from_json.benchmark_scenario import (
    BenchmarkScenarioImportedFromJson,
)
from are.simulation.scenarios.scenario_imported_from_json.utils import (
    preprocess_scenario_from_config,
)
from are.simulation.scenarios.utils.scenario_expander import EnvEventsConfig
from are.simulation.scenarios.validation_result import MultiScenarioValidationResult
from are.simulation.types import CompletedEvent, ToolAugmentationConfig
from are.simulation.utils.countable_iterator import CountableIterator
from are.simulation.validation.configs import DEFAULT_JUDGE_MODEL

ORACLE_EVENT_CLASS_NAME = "OracleEvent"

logger: logging.Logger = logging.getLogger(__name__)


def multiply_scenarios_iterator(
    scenarios_iterator: CountableIterator[
        tuple[BenchmarkScenarioImportedFromJson, list[CompletedEvent] | None]
    ],
    num_runs: int,
) -> CountableIterator[
    tuple[BenchmarkScenarioImportedFromJson, list[CompletedEvent] | None]
]:
    """Multiply the scenarios iterator to run each scenario N times.

    This function creates multiple copies of each scenario with different run numbers
    to improve variance in the results.

    :param scenarios_iterator: Iterator of scenarios and completed events
    :param num_runs: Number of times to run each scenario
    :return: Iterator with each scenario repeated num_runs times
    :rtype: CountableIterator[tuple[BenchmarkScenarioImportedFromJson, list[CompletedEvent]]]
    """
    import copy

    def iterator():
        # First, collect all scenarios to avoid iterator exhaustion
        scenarios_list = list(scenarios_iterator)

        for scenario, completed_events in scenarios_list:
            for run_num in range(num_runs):
                # Create a deep copy of the scenario for each run
                scenario_copy = copy.deepcopy(scenario)

                # Set the run number in the scenario structure
                scenario_copy.run_number = run_num + 1

                # Also copy completed events if they exist
                completed_events_copy = (
                    copy.deepcopy(completed_events) if completed_events else None
                )

                yield scenario_copy, completed_events_copy

    # Calculate new total count
    new_total_count = (
        scenarios_iterator.total_count * num_runs
        if scenarios_iterator.total_count is not None
        else None
    )

    return CountableIterator(iterator(), new_total_count)


def preprocess_scenarios_iterator(
    scenarios_iterator: CountableIterator[
        tuple[BenchmarkScenarioImportedFromJson, list[CompletedEvent] | None]
    ],
    config: MultiScenarioRunnerConfig,
) -> CountableIterator[
    tuple[BenchmarkScenarioImportedFromJson, list[CompletedEvent] | None]
]:
    """Preprocess scenarios after they've been loaded.

    This function applies necessary preprocessing to scenarios based on the run mode.
    In normal mode, it applies judge configuration and validation settings.
    In oracle mode, it initializes the scenario and patches the user message order.

    :param scenarios_iterator: Iterator of loaded scenarios and completed events
    :param oracle: Whether to run in oracle mode where oracle events are executed
    :param offline_validation: Whether to run in offline validation mode using completed events
    :return: Iterator of preprocessed scenarios and completed events
    :rtype: CountableIterator[tuple[BenchmarkScenarioImportedFromJson, list[CompletedEvent]]]
    """

    # Create an iterator function to wrap the generator
    def iterator():
        for scenario, completed_events in scenarios_iterator:
            try:
                # Preprocess the scenario
                preprocess_scenario_from_config(
                    scenario=scenario,
                    config=config,
                )
                yield scenario, completed_events
            except Exception as e:
                logger.error(
                    f"Failed to preprocess scenario {scenario.scenario_id}: {e}"
                )
                # Continue with the next scenario instead of crashing
                continue

    # Return a new CountableIterator with the same total count
    return CountableIterator(iterator(), scenarios_iterator.total_count)


def serialize_scenarios_iterator(
    scenarios_iterator: CountableIterator[
        tuple[BenchmarkScenarioImportedFromJson, list[CompletedEvent] | None]
    ],
) -> CountableIterator[str]:
    """Serialize scenarios after they've been loaded for future pickle in the case of a multi-process run."""
    from are.simulation.data_handler.exporter import JsonScenarioExporter
    from are.simulation.environment import Environment, EnvironmentConfig

    def iterator():
        for scenario, completed_events in scenarios_iterator:
            try:
                # Create fake environment for export
                env = Environment(
                    EnvironmentConfig(
                        oracle_mode=True,
                        queue_based_loop=True,
                        start_time=scenario.start_time,
                    )
                )
                if completed_events is not None:
                    # Set completed events in the environment
                    env.event_log = env.event_log.from_list_view(completed_events)
                # Convert the scenario back to a string
                scenario.initialize()
                scenario_str = JsonScenarioExporter().export_to_json(
                    env=env,
                    scenario=scenario,
                    scenario_id=scenario.scenario_id,
                    runner_config=None,
                )
                yield scenario_str
            except Exception as e:
                logger.error(
                    f"Failed to serialize scenario {scenario.scenario_id}: {e}"
                )
                # Continue with the next scenario instead of crashing
                continue

    # Return a new CountableIterator with the same total count
    return CountableIterator(iterator(), scenarios_iterator.total_count)


def run_scenarios(
    config: MultiScenarioRunnerConfig,
    scenarios_iterator: CountableIterator[
        tuple[BenchmarkScenarioImportedFromJson, list[CompletedEvent] | None] | str
    ],
    progress_description: str | None = None,
) -> MultiScenarioValidationResult:
    """Run the scenarios and return the results.

    This function executes the scenarios using the MultiScenarioRunner.

    :param config: Configuration for the MultiScenarioRunner
    :param scenarios_iterator: Iterator of preprocessed scenarios and completed events
    :param progress_description: Optional description for the progress bar
    :return: The validation result object
    :rtype: MultiScenarioValidationResult
    """
    multi_scenario_runner = MultiScenarioRunner()

    logger.info("Starting.")
    result = multi_scenario_runner.run_with_events(
        config,
        scenarios_iterator,
        progress_description,
    )

    return result


def run_dataset(
    model: str,
    dataset_path: str | None = None,
    hf: str | None = None,
    config: str | None = None,
    split: str | None = None,
    hf_revision: str | None = None,
    limit: int | None = None,
    model_provider: str | None = None,
    endpoint: str | None = None,
    azure_config_dir: str | None = None,
    agent: str | None = None,
    oracle: bool = False,
    offline_validation: bool = False,
    output_dir: str | None = None,
    trace_dump_format: str = "hf",
    max_concurrent_scenarios: int | None = None,
    executor_type: str = "thread",
    enable_caching: bool = False,
    scenario_timeout: int = DEFAULT_SCENARIO_TIMEOUT,
    a2a_app_prop: float = 0,
    a2a_app_agent: str = "",
    a2a_model: str | None = None,
    a2a_model_provider: str | None = None,
    a2a_endpoint: str | None = None,
    use_custom_logger: bool = False,
    simulated_generation_time_mode: str = "measured",
    tool_augmentation_config: ToolAugmentationConfig | None = None,
    env_events_config: EnvEventsConfig | None = None,
    num_runs: int = 3,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_provider: str | None = None,
    judge_endpoint: str | None = None,
    judge_azure_config_dir: str | None = None,
    reasoning_effort: str | None = None,
    log_level: str = "INFO",
    phase_name: str | None = None,
    enable_checkpoints: bool = False,
    checkpoint_verbose: bool = False,
    hints_file_path: str | None = None,
    hook_config_path: str | None = None,
    **kwargs,
) -> MultiScenarioValidationResult:
    """Run a dataset of scenarios with the specified configuration.

    This is the main entry point for running benchmark scenarios. It handles the complete
    workflow of setting up scenarios, preprocessing them, configuring the runner, and
    executing the scenarios.

    :param model: Model name to use for the agent
    :param dataset_path: Path to local dataset directory or JSONL file (mutually exclusive with hf)
    :param hf: HuggingFace dataset name (mutually exclusive with dataset_path)
    :param config: Dataset config (subset) name
    :param split: Dataset split name (e.g., "test", "validation", "train")
    :param hf_revision: Revision of the HuggingFace dataset
    :param limit: Maximum number of scenarios to load
    :param model_provider: Provider of the model
    :param endpoint: URL of the endpoint to contact for running the agent's model
    :param agent: Agent to use for running the scenarios
    :param oracle: Whether to run in oracle mode
    :param offline_validation: Whether to run in offline validation mode
    :param output_dir: Directory to dump the scenario states and logs
    :param trace_dump_format: Format to dump the traces in (e.g., "json", "hf")
    :param hf_upload: Dataset name to upload the traces to HuggingFace
    :param hf_private: Whether to upload the dataset as private
    :param max_concurrent_scenarios: Maximum number of concurrent scenarios to run
    :param enable_caching: Enable caching of results
    :param scenario_timeout: Timeout for each scenario in seconds
    :param a2a_app_prop: Fraction of available Apps to run in Agent2Agent mode
    :param a2a_app_agent: Agent used for App agent instances
    :param a2a_model: Model used for App agent instances
    :param a2a_model_provider: Provider of the App agent model
    :param a2a_endpoint: URL of the endpoint for App agent models
    :param use_custom_logger: Whether to use a custom logger
    :param simulated_generation_time_mode: Mode for simulating generation time
    :param tool_augmentation_config: Configuration for tool augmentation
    :param env_events_config: Configuration for environment events augmentation
    :param num_runs: Number of times to run each scenario (default: 3)
    :return: The validation result object
    :rtype: MultiScenarioValidationResult
    """
    setup_kwargs = {}

    scenarios_iterator = setup_scenarios_iterator(
        dataset_path=dataset_path,
        dataset_config=config,
        dataset_split=split,
        hf=hf,
        hf_revision=hf_revision,
        load_completed_events=offline_validation,
        limit=limit,
        **setup_kwargs,
    )

    # Create checkpoint configuration if enabled
    checkpoint_config = None
    if enable_checkpoints:
        from pathlib import Path

        from are.simulation.checkpoint import CheckpointConfig

        # Checkpoints stored in subdirectory of output_dir
        checkpoint_dir = Path(output_dir) / "checkpoints" if output_dir else Path("./checkpoints")

        checkpoint_config = CheckpointConfig(
            enabled=True,
            checkpoint_dir=checkpoint_dir,
            verbose_logging=checkpoint_verbose,
        )

        logger.info(f"Checkpointing enabled: storing checkpoints in {checkpoint_dir}")

    runner_config = MultiScenarioRunnerConfig(
        model=model,
        model_provider=model_provider,
        agent=agent,
        oracle=oracle,
        export=True,
        output_dir=output_dir,
        trace_dump_format=trace_dump_format,
        endpoint=endpoint,
        azure_config_dir=azure_config_dir,
        max_concurrent_scenarios=max_concurrent_scenarios,
        executor_type=executor_type,
        enable_caching=enable_caching,
        judge_only=offline_validation,
        timeout_seconds=scenario_timeout,
        a2a_app_prop=a2a_app_prop,
        a2a_app_agent=a2a_app_agent,
        a2a_model=a2a_model,
        a2a_model_provider=a2a_model_provider,
        a2a_endpoint=a2a_endpoint,
        use_custom_logger=use_custom_logger,
        simulated_generation_time_mode=simulated_generation_time_mode,
        tool_augmentation_config=tool_augmentation_config,
        env_events_config=env_events_config,
        judge_engine_config=LLMEngineConfig(
            model_name=judge_model,
            provider=judge_provider,
            endpoint=judge_endpoint,
            azure_config_dir=judge_azure_config_dir,
        ),
        reasoning_effort=reasoning_effort,
        log_level=log_level,
        checkpoint_config=checkpoint_config,
        hints_file_path=hints_file_path,
        hook_config_path=hook_config_path,
    )

    # Set base metadata on all scenarios
    def set_scenario_metadata():
        for scenario, completed_events in scenarios_iterator:
            # Set base metadata - config should be set when scenario is loaded/created
            if not hasattr(scenario, "config") or scenario.config is None:
                scenario.config = config
            if not hasattr(scenario, "has_a2a_augmentation"):
                scenario.has_a2a_augmentation = False
            yield scenario, completed_events

    scenarios_with_metadata = CountableIterator(
        set_scenario_metadata(), scenarios_iterator.total_count
    )

    # Multiply scenarios iterator to run each scenario num_runs times
    if num_runs > 1:
        logger.info(f"Running each scenario {num_runs} times to improve variance")
        multiplied_scenarios_iterator = multiply_scenarios_iterator(
            scenarios_with_metadata, num_runs
        )
    else:
        multiplied_scenarios_iterator = scenarios_with_metadata

    if executor_type == "process":
        final_scenarios_iterator = serialize_scenarios_iterator(
            multiplied_scenarios_iterator
        )
    else:
        final_scenarios_iterator = preprocess_scenarios_iterator(
            multiplied_scenarios_iterator,
            runner_config,
        )

    # Create a descriptive name for the progress bar
    progress_description = None
    if config and phase_name:
        if phase_name == "standard":
            progress_description = f"Running {config.title()} scenarios"
        elif phase_name == "agent2agent":
            progress_description = (
                f"Running {config.title()} scenarios with Agent2Agent"
            )
        elif phase_name == "noise":
            progress_description = f"Running {config.title()} scenarios with Noise"
        else:
            progress_description = f"Running {config.title()} scenarios ({phase_name})"
    elif config:
        progress_description = f"Running {config.title()} scenarios"

    return run_scenarios(runner_config, final_scenarios_iterator, progress_description)
