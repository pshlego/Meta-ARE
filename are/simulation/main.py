# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import tempfile

import click

from are.simulation.cli.shared_params import (
    common_options,
    json_config_options,
    output_config_options,
    runtime_config_options,
)
from are.simulation.cli.utils import (
    create_noise_configs,
    run_scenarios_by_huggingface_urls,
    run_scenarios_by_id,
    run_scenarios_by_json_files,
    setup_logging,
)
from are.simulation.scenarios.config import MultiScenarioRunnerConfig
from are.simulation.utils.huggingface import parse_huggingface_url


def validate_main_scenario_sources(**scenario_params):
    """
    Validate scenario source parameters for main.py and return the active ones.

    This function handles the validation and mapping of scenario parameters
    specific to main.py (scenario_id, json_file, and hf_url).

    :param scenario_params: Dictionary containing scenario parameters
    :type scenario_params: dict
    :returns: Tuple of (scenario_ids, scenario_files, hf_urls) with the active parameters
    :rtype: tuple[list[str] | None, list[str] | None, list[str] | None]
    :raises click.UsageError: If validation fails
    """
    # Extract unified parameters
    scenario_id = scenario_params.get("scenario_id")
    scenario_file = scenario_params.get("scenario_file")
    hf_url = scenario_params.get("hf_url")

    # Extract legacy parameters
    json_file = scenario_params.get("json_file")

    # Handle parameter mapping from unified to legacy parameters
    final_scenario_id = scenario_id
    final_scenario_file = json_file

    # Map unified parameters to legacy ones if needed
    if scenario_file and not json_file:
        final_scenario_file = scenario_file
    elif scenario_file and json_file:
        raise click.UsageError(
            "Cannot specify both --scenario-file and --json_file. Use --scenario-file instead."
        )

    # Validate HuggingFace URLs if provided
    if hf_url:
        for url in hf_url:
            hf_params = parse_huggingface_url(url)
            if hf_params is None:
                raise click.UsageError(
                    f"Invalid HuggingFace URL format: '{url}'. Expected format: hf://datasets/dataset_name/config/split/scenario_id"
                )

    # Count how many sources are specified
    sources_specified = sum(
        [bool(final_scenario_id), bool(final_scenario_file), bool(hf_url)]
    )

    # Validate that exactly one source is specified
    if sources_specified == 0:
        raise click.UsageError(
            "Must specify one of: --scenario-id/--scenario_id, --scenario-file/--json_file, or --hf-url"
        )
    elif sources_specified > 1:
        raise click.UsageError(
            "Cannot specify multiple scenario sources at the same time. Choose one of: --scenario-id, --scenario-file, or --hf-url"
        )

    return final_scenario_id, final_scenario_file, hf_url


@click.command()
@common_options()
@runtime_config_options()
@json_config_options()
# Main-specific scenario parameters
@click.option(
    "-s",
    "--scenario-id",
    required=False,
    multiple=True,
    help="Scenarios to run from registry (can be specified multiple times)",
)
@click.option(
    "--scenario-file",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=False,
    multiple=True,
    help="JSON scenario files to run (can be specified multiple times)",
)
@click.option(
    "--hf-url",
    required=False,
    multiple=True,
    help="HuggingFace dataset URLs in format: hf://datasets/dataset_name/config/split/scenario_id (can be specified multiple times)",
)
# Legacy scenario parameters (for backward compatibility)
@click.option(
    "--scenario_id",
    required=False,
    multiple=True,
    help="Scenarios to run (deprecated, use --scenario-id instead)",
    hidden=True,
)
@click.option(
    "-j",
    "--json_file",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=False,
    multiple=True,
    help="JSON scenario files to run (deprecated, use --scenario-file instead)",
    hidden=True,
)
@output_config_options()
@click.option(
    "-e",
    "--export",
    is_flag=True,
    default=False,
    help="Export the trace to a JSON file.",
)
@click.option(
    "-w",
    "--wait-for-user-input-timeout",
    type=float,
    default=None,
    help="Timeout for user inputs in seconds (no timeout by default).",
)
@click.option(
    "--list-scenarios",
    is_flag=True,
    default=False,
    help="List all available scenarios and exit.",
)
# Checkpoint options
@click.option(
    "--enable-checkpoints",
    is_flag=True,
    default=False,
    help="Enable checkpoint capture during scenario execution (captures before and after each agent tool call).",
)
@click.option(
    "--checkpoint-verbose",
    is_flag=True,
    default=False,
    help="Enable verbose logging for checkpoint operations.",
)
@click.option(
    "--hints-file",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=False,
    default=None,
    help="JSON file mapping scenario_id to hint text for system prompt injection.",
)
def main(
    model: str,
    provider: str | None = None,
    endpoint: str | None = None,
    agent: str | None = None,
    log_level: str = "INFO",
    oracle: bool = False,
    simulated_generation_time_mode: str = "measured",
    noise: bool = False,
    max_concurrent_scenarios: int | None = None,
    kwargs: str = "{}",
    multi_kwargs: str = "[{}]",
    scenario_kwargs: str = "{}",
    multi_scenario_kwargs: str = "[{}]",
    # Main-specific scenario parameters
    scenario_id: list[str] | None = None,
    scenario_file: list[str] | None = None,
    hf_url: list[str] | None = None,
    # Legacy scenario parameters (for backward compatibility)
    json_file: list[str] | None = None,
    output_dir: str | None = None,
    export: bool = False,
    wait_for_user_input_timeout: float | None = None,
    list_scenarios: bool = False,
    # Checkpoint parameters
    enable_checkpoints: bool = False,
    checkpoint_verbose: bool = False,
    # Hints file
    hints_file: str | None = None,
):
    """
    Main entry point for the Meta Agents Research Environments scenario runner CLI.

    This function processes command line arguments and runs scenarios using the
    MultiScenarioRunner. It supports running scenarios by ID from the registry
    or by providing JSON scenario files.
    """
    # Set up logging
    setup_logging(log_level)

    # Handle list scenarios flag
    if list_scenarios:
        from are.simulation.scenarios.utils.registry import registry

        scenarios = registry.get_all_scenarios()
        if not scenarios:
            click.echo("No scenarios are currently registered.")
        else:
            for scenario_key in sorted(scenarios.keys()):
                click.echo(scenario_key)
        return

    # Validate scenario sources and get the active ones
    final_scenario_id, final_json_file, final_hf_url = validate_main_scenario_sources(
        scenario_id=scenario_id,
        scenario_file=scenario_file,
        hf_url=hf_url,
        json_file=json_file,
    )

    # Create noise configs if --noise flag is enabled
    tool_augmentation_config = None
    env_events_config = None
    if noise:
        tool_augmentation_config, env_events_config = create_noise_configs()

    # If output_dir is None, create a temporary directory for it
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="are_simulation_output_")

    # Create checkpoint configuration if enabled
    checkpoint_config = None
    if enable_checkpoints:
        from pathlib import Path
        from are.simulation.checkpoint import CheckpointConfig
        
        # Checkpoints stored in subdirectory of output_dir
        checkpoint_dir = Path(output_dir) / "checkpoints"
        
        checkpoint_config = CheckpointConfig(
            enabled=True,
            checkpoint_dir=checkpoint_dir,
            verbose_logging=checkpoint_verbose,
        )
        
        click.echo(f"Checkpointing enabled: storing checkpoints in {checkpoint_dir}")
        click.echo(f"Checkpoints will be created before and after each agent tool call")

    # Runner config
    runner_config = MultiScenarioRunnerConfig(
        model=model,
        model_provider=provider,
        agent=agent,
        scenario_creation_params=scenario_kwargs,
        scenario_multi_creation_params=multi_scenario_kwargs,
        scenario_initialization_params=kwargs,
        scenario_multi_initialization_params=multi_kwargs,
        oracle=oracle,
        export=export,
        wait_for_user_input_timeout=wait_for_user_input_timeout,
        output_dir=output_dir,
        endpoint=endpoint,
        max_concurrent_scenarios=max_concurrent_scenarios,
        simulated_generation_time_mode=simulated_generation_time_mode,
        tool_augmentation_config=tool_augmentation_config,
        env_events_config=env_events_config,
        enable_caching=False,
        checkpoint_config=checkpoint_config,
        hints_file_path=hints_file,
    )

    # Run scenarios by ID
    if final_scenario_id:
        run_scenarios_by_id(runner_config, final_scenario_id)
        return

    # Run scenarios by JSON file
    if final_json_file:
        run_scenarios_by_json_files(runner_config, final_json_file)
        return

    # Run scenarios by HuggingFace URL
    if final_hf_url:
        run_scenarios_by_huggingface_urls(runner_config, final_hf_url)
        return


if __name__ == "__main__":
    main()
