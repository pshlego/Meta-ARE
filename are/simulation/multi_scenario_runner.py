# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import concurrent.futures
import itertools
import json
import logging
import os
import signal
import sys
import tempfile
import time

from tqdm import tqdm

from are.simulation.agents.agent_builder import AbstractAgentBuilder
from are.simulation.agents.agent_config_builder import AbstractAgentConfigBuilder
from are.simulation.environment import Scenario
from are.simulation.scenario_runner import ScenarioRunner
from are.simulation.scenarios.config import (
    MultiScenarioRunnerConfig,
    ScenarioRunnerConfig,
)
from are.simulation.scenarios.scenario_imported_from_json.utils import (
    load_and_preprocess_scenario_str,
)
from are.simulation.scenarios.validation_result import (
    MultiScenarioValidationResult,
    ScenarioValidationResult,
)
from are.simulation.types import CompletedEvent
from are.simulation.utils.countable_iterator import CountableIterator
from are.simulation.utils.streaming_utils import stream_pool

logger = logging.getLogger(__name__)


class ScenarioTimeoutError(Exception):
    pass


def extract_scenario_id_and_run_number(
    scenario_with_events: tuple[Scenario, list[CompletedEvent] | None] | str,
) -> tuple[str, int | None]:
    """
    Extract the scenario_id and run_number from the input, which can be either:
    - A tuple of (Scenario, list of CompletedEvent or None)
    - A JSON string containing scenario metadata

    :return: A tuple of (scenario_id, run_number), where scenario_id is a string and run_number is an int or None.
    """
    if isinstance(scenario_with_events, str):
        scenario_data = json.loads(scenario_with_events)
        definition = scenario_data.get("metadata", {}).get("definition", {})
        scenario_id = definition.get("scenario_id")
        run_number = definition.get("run_number", None)
        return scenario_id, int(run_number) if run_number is not None else None
    scenario, _ = scenario_with_events
    return scenario.scenario_id, getattr(scenario, "run_number", None)


def _export_benchmark_result_jsonl(
    result: MultiScenarioValidationResult,
    output_dir: str,
) -> None:
    """
    Export the benchmark result to a JSON file with detailed run information.
    """
    from are.simulation.benchmark.hf_upload_utils import get_scenario_result_info

    benchmark_result_file = os.path.join(output_dir, "output.jsonl")
    with open(benchmark_result_file, "w", encoding="utf-8") as f:
        for scenario_key, scenario_result in result.scenario_results.items():
            # Deconstruct the tuple key to get scenario_id and run_number
            scenario_id, run_number = scenario_key

            # Get score and status using shared function
            score, status = get_scenario_result_info(scenario_result)

            # Determine trace_id - handle cases where export_path might be None
            trace_id = (
                scenario_result.export_path if scenario_result.export_path else None
            )

            # Build comprehensive metadata
            detailed_metadata = {
                "scenario_id": scenario_id,
                "run_number": run_number,
                "status": status,
                "has_exception": scenario_result.exception is not None,
                "exception_type": (
                    type(scenario_result.exception).__name__
                    if scenario_result.exception
                    else None
                ),
                "exception_message": (
                    str(scenario_result.exception)
                    if scenario_result.exception
                    else None
                ),
                "rationale": scenario_result.rationale,
            }

            # Remove None values from metadata to keep it clean
            detailed_metadata = {
                k: v for k, v in detailed_metadata.items() if v is not None
            }

            json_obj = {
                "task_id": scenario_id,
                "trace_id": trace_id,
                "score": score,
                "metadata": detailed_metadata,
            }
            json.dump(json_obj, f)
            f.write("\n")
    logger.info(f"Exported benchmark result to {benchmark_result_file}")


def _create_scenario_runner_config(
    config: MultiScenarioRunnerConfig, scenario: Scenario
) -> ScenarioRunnerConfig:
    """Create a ScenarioRunnerConfig from a MultiScenarioRunnerConfig and a scenario."""
    runner_config = ScenarioRunnerConfig(
        model=config.model,
        model_provider=config.model_provider,
        endpoint=config.endpoint,
        agent=config.agent,
        scenario_creation_params=config.scenario_creation_params,
        scenario_initialization_params=config.scenario_initialization_params,
        oracle=config.oracle,
        export=config.export,
        wait_for_user_input_timeout=config.wait_for_user_input_timeout,
        trace_dump_format=config.trace_dump_format,
        output_dir=config.output_dir,
        judge_only=config.judge_only,
        a2a_app_prop=config.a2a_app_prop,
        a2a_app_agent=config.a2a_app_agent,
        a2a_model=config.a2a_model,
        simulated_generation_time_mode=config.simulated_generation_time_mode,
        use_custom_logger=config.use_custom_logger,
        tool_augmentation_config=config.tool_augmentation_config,
        env_events_config=config.env_events_config,
        judge_engine_config=config.judge_engine_config,
        hints_file_path=config.hints_file_path,
    )
    if scenario.nb_turns is not None:
        runner_config.max_turns = scenario.nb_turns
    return runner_config


def process_scenario(
    scenario_with_events: tuple[Scenario, list[CompletedEvent] | None] | str,
    config: MultiScenarioRunnerConfig,
    agent_config_builder: AbstractAgentConfigBuilder | None,
    agent_builder: AbstractAgentBuilder | None,
) -> ScenarioValidationResult:
    """Process a single scenario with its completed events.

    :param scenario_with_events: Tuple containing the scenario and its completed events
    :param config: The MultiScenarioRunnerConfig for this run
    :param agent_config_builder: The agent config builder
    :param agent_builder: The agent builder
    :return: The result of running the scenario
    """
    # Re-establish logging configuration in worker thread
    from are.simulation.cli.utils import suppress_noisy_loggers
    from are.simulation.logging_config import configure_logging

    # Convert log level string to logging constant
    numeric_level = getattr(logging, config.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO  # Fallback to INFO if invalid level

    # Re-configure logging with tqdm support
    configure_logging(level=numeric_level, use_tqdm=config.executor_type != "process")
    suppress_noisy_loggers()

    # Load and preprocess the scenario if needed
    if isinstance(scenario_with_events, str):
        scenario, completed_events = load_and_preprocess_scenario_str(
            config, scenario_with_events
        )
    else:
        scenario, completed_events = scenario_with_events
    scenario_id = scenario.scenario_id

    runner_config = _create_scenario_runner_config(config, scenario)

    # Set the scenario ID and run number for the current thread
    from are.simulation.logging_config import set_logger_scenario_id

    run_number = getattr(scenario, "run_number", None)
    set_logger_scenario_id(scenario_id, run_number)

    # Create scenario runner - use checkpointable version if checkpoint config provided
    if config.checkpoint_config is not None and config.checkpoint_config.enabled:
        from are.simulation.checkpoint import CheckpointableScenarioRunner, CheckpointManager
        
        checkpoint_manager = CheckpointManager(config.checkpoint_config)
        scenario_runner = CheckpointableScenarioRunner(
            checkpoint_config=config.checkpoint_config,
            checkpoint_manager=checkpoint_manager,
            agent_config_builder=agent_config_builder,
            agent_builder=agent_builder,
        )
        # Pass scenario source info for checkpoint metadata
        scenario_source_path = getattr(scenario, '_checkpoint_source_path', None)
        if scenario_source_path is not None:
            scenario_runner._scenario_source_path = scenario_source_path
            scenario_runner._scenario_source_type = "json_file"
        else:
            from are.simulation.scenarios.scenario_imported_from_json.benchmark_scenario import (
                BenchmarkScenarioImportedFromJson,
            )
            if isinstance(scenario, BenchmarkScenarioImportedFromJson):
                scenario_runner._scenario_source_type = "json_file"
        logger.info(f"Checkpointing enabled for scenario {scenario_id}")
    else:
        scenario_runner = ScenarioRunner(
            agent_config_builder=agent_config_builder,
            agent_builder=agent_builder,
        )

    try:
        return maybe_run_scenario(
            scenario_runner,
            runner_config,
            scenario,
            completed_events,
            enable_caching=config.enable_caching,
        )
    except Exception as e:
        logger.error(f"Scenario {scenario_id} failed with exception: {e}")
        raise e


def maybe_run_scenario(
    scenario_runner: ScenarioRunner,
    runner_config: ScenarioRunnerConfig,
    scenario: Scenario,
    completed_events: list[CompletedEvent] | None,
    enable_caching: bool = True,
) -> ScenarioValidationResult:
    """Run a scenario with caching support and timeout handling."""

    # Import caching functions
    if enable_caching:
        from are.simulation.scenarios.utils.caching import maybe_load_cached_result

        # Check if we have a cached result
        cached_result = maybe_load_cached_result(runner_config, scenario)
        if cached_result is not None:
            log_msg = f"Found cached result, skipping scenario {scenario.scenario_id}"
            run_number = getattr(scenario, "run_number", None)
            if run_number is not None:
                log_msg += f", run: {run_number}"
            logger.warning(log_msg)
            return cached_result

    # No cached result found (or caching disabled), run the scenario
    result = scenario_runner.run(runner_config, scenario, completed_events)

    # Cache the result for future runs if caching is enabled
    if enable_caching:
        from are.simulation.scenarios.utils.caching import write_cached_result

        write_cached_result(runner_config, scenario, result)

    return result


class MultiScenarioRunner:
    def __init__(
        self,
        agent_config_builder: AbstractAgentConfigBuilder | None = None,
        agent_builder: AbstractAgentBuilder | None = None,
    ):
        self.agent_config_builder = agent_config_builder
        self.agent_builder = agent_builder
        self._interrupted = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers to handle Ctrl+C gracefully."""

        def signal_handler(signum, frame):
            logger.info("Received interrupt signal. Stopping scenario execution...")
            self._interrupted = True
            # Re-raise KeyboardInterrupt to stop the execution
            raise KeyboardInterrupt("Execution interrupted by user")

        signal.signal(signal.SIGINT, signal_handler)

    def run(
        self,
        config: MultiScenarioRunnerConfig,
        scenarios: list[Scenario],
    ) -> MultiScenarioValidationResult:
        assert len(scenarios) > 0, "No scenarios provided"
        return self.run_with_events(
            config,
            itertools.zip_longest(scenarios, [], fillvalue=None),  # type: ignore
        )

    def run_with_events(
        self,
        config: MultiScenarioRunnerConfig,
        scenarios_with_events: CountableIterator[
            tuple[Scenario, list[CompletedEvent] | None] | str
        ],
        progress_description: str | None = None,
    ):
        multi_scenario_validation_result = MultiScenarioValidationResult(
            run_config=config
        )
        if config.output_dir is None:
            config.output_dir = tempfile.gettempdir()
        if os.path.isabs(config.output_dir):
            config.output_dir = os.path.abspath(config.output_dir)

        os.makedirs(config.output_dir, exist_ok=True)

        processed_scenarios = 0
        start_time = time.time()

        # Ensure max_workers is an integer (default to CPU count if None)
        max_workers = config.max_concurrent_scenarios
        if max_workers is None:
            import multiprocessing

            max_workers = multiprocessing.cpu_count()

        # Log execution mode
        if max_workers == 1:
            logger.info("Running scenarios sequentially (max_concurrent_scenarios=1)")
        else:
            logger.info(f"Running scenarios in parallel with {max_workers} workers")

        # Process scenarios using stream_process (handles both sequential and parallel cases)
        try:
            with stream_pool(
                scenarios_with_events,
                process_scenario,
                max_workers=max_workers,
                timeout_seconds=config.timeout_seconds,
                executor_type=config.executor_type,
                config=config,
                agent_config_builder=self.agent_config_builder,
                agent_builder=self.agent_builder,
            ) as stream:
                # Try to get the total count from the CountableIterator
                total = None
                try:
                    total = len(scenarios_with_events)
                except (TypeError, AttributeError):
                    # If len() raises an exception, the iterator doesn't have a known length
                    pass

                # Create tqdm progress bar with initial postfix
                desc = (
                    progress_description
                    if progress_description
                    else "Running scenarios"
                )
                progress_bar = tqdm(
                    stream,
                    desc=desc,
                    position=1,
                    leave=True,
                    total=total,
                    mininterval=0.1,  # Update at least every 0.1 seconds
                    maxinterval=1.0,  # Don't wait more than 1 second between updates
                    smoothing=0.3,  # Smooth out rate estimates
                    dynamic_ncols=True,  # Automatically resize to terminal width
                    file=sys.stdout,  # Ensure output goes to stdout
                )
                progress_bar.set_postfix({"Success": "0.0%"})

                # Iterate through results as they complete
                for scenario_with_events, result, error in progress_bar:
                    if self._interrupted:
                        logger.info("Execution interrupted, stopping...")
                        break

                    scenario_id, run_number = extract_scenario_id_and_run_number(
                        scenario_with_events
                    )

                    # Ensure we have a valid result
                    if error:
                        if isinstance(
                            error, (TimeoutError, concurrent.futures.TimeoutError)
                        ):
                            logger.error(
                                f"Scenario {scenario_id} timed out after {config.timeout_seconds} seconds"
                            )
                            result = ScenarioValidationResult(
                                success=False,
                                exception=ScenarioTimeoutError(
                                    f"Scenario {scenario_id} timed out after {config.timeout_seconds} seconds"
                                ),
                                duration=(
                                    float(config.timeout_seconds)
                                    if config.timeout_seconds
                                    else 0.0
                                ),
                            )
                        else:
                            logger.error(
                                f"Scenario {scenario_id} failed with exception: {error}"
                            )
                            result = ScenarioValidationResult(
                                success=False,
                                exception=error,
                                duration=None,
                            )

                    # Make sure result is not None before processing
                    if result is None:
                        result = ScenarioValidationResult(
                            success=False,
                            exception=Exception(
                                f"Unknown error occurred for scenario {scenario_id}"
                            ),
                            duration=None,
                        )

                    processed_scenarios += 1
                    multi_scenario_validation_result.add_result(
                        result, scenario_id, run_number
                    )

                    # Update progress bar with success percentage
                    success_rate = multi_scenario_validation_result.success_rate()
                    progress_bar.set_postfix({"Success": f"{success_rate:.1f}%"})
                    # Update progress bar with success percentage
                    success_rate = multi_scenario_validation_result.success_rate()
                    progress_bar.set_postfix({"Success": f"{success_rate:.1f}%"})

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping scenario execution...")
            self._interrupted = True
            raise

        assert processed_scenarios > 0, "No scenarios processed"
        # Add duration to the results
        multi_scenario_validation_result.duration = time.time() - start_time

        _export_benchmark_result_jsonl(
            multi_scenario_validation_result,
            config.output_dir,
        )

        return multi_scenario_validation_result
