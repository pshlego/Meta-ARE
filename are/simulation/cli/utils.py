# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


"""
Shared utility functions for Meta Agents Research Environments CLI tools.

This module provides common utility functions that can be imported and used by all CLI tools
to reduce code duplication and ensure consistent behavior.
"""

import json
import logging
import os
import signal
import sys
import threading
from typing import Any, Callable, Type

import click

from are.simulation.logging_config import configure_logging
from are.simulation.multi_scenario_runner import (
    MultiScenarioRunner,
    MultiScenarioRunnerConfig,
    MultiScenarioValidationResult,
)
from are.simulation.scenarios.config import ScenarioRunnerConfig
from are.simulation.scenarios.scenario import Scenario

# Set up logger for this module
logger = logging.getLogger(__name__)


def setup_logging(level: str, use_tqdm: bool = False) -> None:
    """
    Set up logging configuration with the specified level.

    :param level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :type level: str
    :param use_tqdm: Whether to use tqdm-compatible logging (for progress bars)
    :type use_tqdm: bool
    :raises ValueError: If the log level is invalid
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # Configure logging with the specified level
    configure_logging(level=numeric_level, use_tqdm=use_tqdm)


def parse_json_safely(json_str: str, param_name: str) -> dict[str, Any]:
    """
    Parse a JSON string safely with consistent error handling.

    :param json_str: JSON string to parse
    :type json_str: str
    :param param_name: Name of the parameter for error messages
    :type param_name: str
    :returns: Parsed JSON as dictionary
    :rtype: dict[str, Any]
    :raises click.UsageError: If JSON parsing fails
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise click.UsageError(f"Invalid JSON string for {param_name}: {e}")


def validate_mutually_exclusive_params(**params) -> None:
    """
    Validate that mutually exclusive parameters are not specified together.

    :param params: Dictionary of parameter names and their values
    :type params: dict
    :raises click.UsageError: If mutually exclusive parameters are specified
    """
    # Filter out None values
    specified_params = {
        name: value for name, value in params.items() if value is not None
    }

    if len(specified_params) > 1:
        param_names = list(specified_params.keys())
        raise click.UsageError(
            f"Cannot specify multiple mutually exclusive parameters: {', '.join(param_names)}"
        )
    elif len(specified_params) == 0:
        param_names = list(params.keys())
        raise click.UsageError(
            f"Must specify at least one of: {', '.join(param_names)}"
        )


def setup_signal_handlers(cleanup_callback: Callable[[], None]) -> None:
    """
    Set up cross-platform signal handlers for graceful shutdown.

    :param cleanup_callback: Function to call for cleanup on shutdown
    :type cleanup_callback: Callable[[], None]
    """
    shutdown_event = threading.Event()

    def signal_handler(sig=None, frame=None):
        """Handle shutdown signals gracefully."""
        print("\nShutting down gracefully...")
        cleanup_callback()
        shutdown_event.set()
        sys.exit(0)

    def setup_handlers():
        """Set up signal handlers with error handling."""
        try:
            # Handle SIGINT (Ctrl+C)
            signal.signal(signal.SIGINT, signal_handler)

            # Handle SIGTERM for graceful shutdown
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, signal_handler)

            # On Windows, also handle SIGBREAK (Ctrl+Break)
            if sys.platform == "win32":
                if hasattr(signal, "SIGBREAK"):
                    signal.signal(signal.SIGBREAK, signal_handler)

        except (OSError, ValueError) as e:
            # Some signals might not be available on all platforms
            logging.warning(f"Could not set up signal handler: {e}")

    setup_handlers()

    # On Windows, set up keyboard interrupt monitor thread
    if sys.platform == "win32":

        def keyboard_interrupt_monitor():
            """Monitor for keyboard interrupts on Windows."""
            try:
                while not shutdown_event.is_set():
                    # Check for keyboard interrupt every 100ms
                    shutdown_event.wait(0.1)
            except KeyboardInterrupt:
                signal_handler()

        monitor_thread = threading.Thread(
            target=keyboard_interrupt_monitor, daemon=True
        )
        monitor_thread.start()


def create_noise_configs():
    """
    Create noise augmentation configurations if needed.

    Returns:
        Tuple of (tool_augmentation_config, env_events_config)
    """
    from are.simulation.scenarios.utils.scenario_expander import EnvEventsConfig
    from are.simulation.types import ToolAugmentationConfig

    logging.info(
        "Noise augmentation enabled - creating tool augmentation and environment events configs"
    )

    tool_augmentation_config = ToolAugmentationConfig()
    env_events_config = EnvEventsConfig(
        num_env_events_per_minute=10,
        env_events_seed=0,
    )

    return tool_augmentation_config, env_events_config


def handle_parameter_conflicts(
    unified_param: Any, legacy_param: Any, unified_name: str, legacy_name: str
) -> Any:
    """
    Handle conflicts between unified and legacy parameters.

    Args:
        unified_param: Value of the unified parameter
        legacy_param: Value of the legacy parameter
        unified_name: Name of the unified parameter
        legacy_name: Name of the legacy parameter

    Returns:
        The parameter value to use

    Raises:
        click.UsageError: If both parameters are specified
    """
    if unified_param is not None and legacy_param is not None:
        raise click.UsageError(
            f"Cannot specify both {unified_name} and {legacy_name}. Use {unified_name} instead."
        )

    return unified_param if unified_param is not None else legacy_param


def suppress_noisy_loggers() -> None:
    """Suppress noisy loggers that are commonly used in Meta Agents Research Environments CLIs."""
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("are.simulation.validation.judge").setLevel(logging.WARNING)
    logging.getLogger("are.simulation.agents.default_agent").setLevel(logging.WARNING)
    logging.getLogger("are.simulation.environment").setLevel(logging.WARNING)


# Scenario Loading utils
def _initialize_loaded_scenario(
    scenario: Scenario, config: ScenarioRunnerConfig
) -> Scenario:
    """Apply additional initialization parameters to an already loaded scenario."""
    try:
        scenario_initialization_params_dict = json.loads(
            config.scenario_initialization_params
        )
    except json.JSONDecodeError:
        raise ValueError(
            f"Invalid JSON string for scenario initialization params: "
            f"{config.scenario_initialization_params}"
        )

    scenario.initialize(**scenario_initialization_params_dict)
    return scenario


def _create_and_initialize_scenario(
    scenario_type: Type[Scenario],
    config: ScenarioRunnerConfig,
    creation_kwargs: dict,
    init_kwargs: dict,
) -> Scenario:
    try:
        scenario_initialization_params_dict = json.loads(
            config.scenario_initialization_params
        )
    except json.JSONDecodeError:
        raise ValueError(
            f"Invalid JSON string for scenario initialization params: {config.scenario_initialization_params}"
        )
    try:
        scenario_creation_params_dict = json.loads(config.scenario_creation_params)
    except json.JSONDecodeError:
        raise ValueError(
            f"Invalid JSON string for scenario creation params: {config.scenario_creation_params}"
        )

    if len(creation_kwargs) > 0:
        scenario_creation_params_dict.update(creation_kwargs)
    if len(init_kwargs) > 0:
        scenario_initialization_params_dict.update(init_kwargs)
    scenario = scenario_type(**scenario_creation_params_dict)
    scenario.initialize(**scenario_initialization_params_dict)
    return scenario


def run_scenarios_by_id(
    config: MultiScenarioRunnerConfig,
    scenario_ids: list[str],
) -> MultiScenarioValidationResult:
    from are.simulation.scenarios.utils.constants import ALL_SCENARIOS

    if not scenario_ids:
        raise ValueError("No scenarios provided.")

    try:
        creation_kwargs = json.loads(config.scenario_multi_creation_params)
    except json.JSONDecodeError:
        raise ValueError(
            f"Invalid JSON string for scenario multi initialization params: {config.scenario_multi_initialization_params}"
        )

    try:
        init_kwargs = json.loads(config.scenario_multi_initialization_params)
    except json.JSONDecodeError:
        raise ValueError(
            f"Invalid JSON string for scenario multi initialization params: {config.scenario_multi_initialization_params}"
        )

    scenarios = []
    for id in scenario_ids:
        if id not in ALL_SCENARIOS:
            raise ValueError(f"Scenario {id} not found")
        try:
            scenario_type = ALL_SCENARIOS[id]

            for creation_kwarg in creation_kwargs:
                for init_kwarg in init_kwargs:
                    scenario = _create_and_initialize_scenario(
                        scenario_type, config, creation_kwarg, init_kwarg
                    )

                    # INJECT CONFIGS HERE - after creation but before running
                    if config.tool_augmentation_config is not None:
                        scenario.tool_augmentation_config = (
                            config.tool_augmentation_config
                        )
                    if config.env_events_config is not None:
                        scenario.env_events_config = config.env_events_config

                    scenarios.append(scenario)
            logger.info(f"Loaded scenario {id}")
        except Exception as e:
            logger.error(f"Failed to load scenario {id}: {e}")
            raise ValueError(f"Failed to load scenario {id}: {e}") from e

    runner = MultiScenarioRunner()
    return runner.run(config, scenarios)


def run_scenarios_by_json_files(
    config: MultiScenarioRunnerConfig,
    json_file_paths: list[str],
) -> MultiScenarioValidationResult:
    from are.simulation.scenarios.utils.load_utils import load_scenario

    if not json_file_paths:
        raise ValueError("No JSON files provided.")

    scenarios = []
    for json_path in json_file_paths:
        try:
            loaded_scenario = load_scenario(json_path)

            # INJECT CONFIGS HERE - before initialization
            if config.tool_augmentation_config is not None:
                loaded_scenario.tool_augmentation_config = (
                    config.tool_augmentation_config
                )
            if config.env_events_config is not None:
                loaded_scenario.env_events_config = config.env_events_config

            initialized_scenario = _initialize_loaded_scenario(loaded_scenario, config)
            # Store the source file path for checkpoint metadata
            initialized_scenario._checkpoint_source_path = os.path.abspath(json_path)
            scenarios.append(initialized_scenario)
            logger.info(f"Loaded scenario from {json_path}")
        except Exception as e:
            logger.error(f"Failed to load scenario from {json_path}: {e}")
            raise ValueError(f"Failed to load scenario from {json_path}: {e}") from e

    runner = MultiScenarioRunner()
    return runner.run(config, scenarios)


def run_scenarios_by_huggingface_urls(
    config: MultiScenarioRunnerConfig,
    hf_urls: list[str],
) -> MultiScenarioValidationResult:
    """
    Run scenarios from HuggingFace dataset URLs.

    :param config: Multi-scenario runner configuration
    :param hf_urls: List of HuggingFace URLs in format hf://datasets/dataset_name/config/split/scenario_id
    :return: Multi-scenario validation result
    :raises ValueError: If URLs are invalid or scenarios cannot be loaded
    """
    from are.simulation.utils.huggingface import (
        load_scenario_from_huggingface_for_cli,
        parse_huggingface_url,
    )

    if not hf_urls:
        raise ValueError("No HuggingFace URLs provided.")

    scenarios = []
    for hf_url in hf_urls:
        try:
            # Parse the HuggingFace URL
            hf_params = parse_huggingface_url(hf_url)
            if hf_params is None:
                raise ValueError(f"Invalid HuggingFace URL format: {hf_url}")

            logger.info(f"Loading scenario from HuggingFace: {hf_url}")

            # Use the common HuggingFace loader
            scenario, completed_events = load_scenario_from_huggingface_for_cli(
                dataset_name=hf_params["dataset_name"],
                dataset_config=hf_params["config"],
                dataset_split=hf_params["split"],
                scenario_id=hf_params["scenario_id"],
            )

            if scenario is None:
                raise ValueError(
                    f"Scenario '{hf_params['scenario_id']}' not found in dataset {hf_params['dataset_name']}/{hf_params['config']}/{hf_params['split']}"
                )

            # INJECT CONFIGS HERE - before initialization
            if config.tool_augmentation_config is not None:
                scenario.tool_augmentation_config = config.tool_augmentation_config
            if config.env_events_config is not None:
                scenario.env_events_config = config.env_events_config

            initialized_scenario = _initialize_loaded_scenario(scenario, config)
            scenarios.append(initialized_scenario)
            logger.info(f"Loaded scenario {hf_params['scenario_id']} from {hf_url}")

        except Exception as e:
            logger.error(f"Failed to load scenario from {hf_url}: {e}")
            raise ValueError(f"Failed to load scenario from {hf_url}: {e}") from e

    runner = MultiScenarioRunner()
    return runner.run(config, scenarios)
