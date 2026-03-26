# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import json
import logging
import logging.config
import os
from pathlib import Path

import click

from are.simulation.agents.agent_builder import AppAgentBuilder
from are.simulation.benchmark.gaia2_submission import handle_gaia2_run
from are.simulation.benchmark.report_stats import (
    combine_results_to_dataframe,
    generate_json_stats_report,
    generate_validation_report,
)
from are.simulation.benchmark.scenario_executor import (
    DEFAULT_SCENARIO_TIMEOUT,
    run_dataset,
)
from are.simulation.cli.shared_params import (
    common_options,
    output_config_options,
    runtime_config_options,
)
from are.simulation.cli.utils import (
    create_noise_configs,
    handle_parameter_conflicts,
    setup_logging,
    suppress_noisy_loggers,
)
from are.simulation.config import PROVIDERS
from are.simulation.utils import DEFAULT_APP_AGENT
from are.simulation.validation.configs import DEFAULT_JUDGE_MODEL

logger: logging.Logger = logging.getLogger(__name__)


def internal_options():
    def decorator(func):
        return func

    return decorator


# Special config names
DEMO_CONFIG = "demo"
MINI_CONFIG = "mini"


def generate_and_save_reports(
    results: dict,
    model: str,
    provider: str | None,
    output_dir: str | None = None,
    num_runs: int = 3,
) -> None:
    """Generate validation and JSON reports from benchmark results.

    :param results: Dictionary of results (either simple config->result mapping or tuple keys)
    :type results: dict
    :param model: Model name used for the benchmark
    :type model: str
    :param provider: Provider name (or None)
    :type provider: str | None
    :param output_dir: Output directory for saving JSON report (optional)
    :type output_dir: str | None
    :param num_runs: Number of runs per scenario for display
    :type num_runs: int
    """
    if not results:
        logger.warning("No results to report.")
        return

    # Convert to DataFrame and generate report
    df = combine_results_to_dataframe(results)

    if not df.is_empty():
        # Generate text report
        report = generate_validation_report(df, model, provider or "unknown", num_runs)
        logger.info("\n" + report)

        # Generate and save JSON report
        json_report = generate_json_stats_report(df, model, provider or "unknown")

        # Save JSON report to file
        if output_dir:
            json_output_path = os.path.join(output_dir, "benchmark_stats.json")
        else:
            json_output_path = "benchmark_stats.json"

        # Ensure output directory exists
        Path(json_output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(json_report, f, indent=2)

        logger.info(f"JSON stats report saved to: {json_output_path}")
    else:
        logger.warning("No results available for reporting.")


def validate_benchmark_scenario_sources(**scenario_params):
    """
    Validate scenario source parameters for benchmark CLI and return the active ones.

    This function handles the validation and mapping of scenario parameters
    specific to benchmark CLI (dataset and HuggingFace parameters).

    :param scenario_params: Dictionary containing scenario parameters
    :type scenario_params: dict
    :return: Dictionary with validated and mapped scenario parameters
    :rtype: dict
    :raises click.UsageError: If invalid parameter combinations are provided
    """
    # Extract parameters
    dataset = scenario_params.get("dataset")
    hf_dataset = scenario_params.get("hf_dataset")
    hf_config = scenario_params.get("hf_config")
    hf_split = scenario_params.get("hf_split")
    hf_revision = scenario_params.get("hf_revision")
    dataset_path = scenario_params.get("dataset_path")
    hf = scenario_params.get("hf")
    config = scenario_params.get("config")
    split = scenario_params.get("split")

    # Handle parameter conflicts between unified and legacy parameters
    final_dataset_path = handle_parameter_conflicts(
        dataset, dataset_path, "--dataset", "--dataset_path"
    )
    final_hf = handle_parameter_conflicts(hf_dataset, hf, "--hf-dataset", "--hf")
    final_config = handle_parameter_conflicts(
        hf_config, config, "--hf-config", "--config"
    )
    final_split = handle_parameter_conflicts(hf_split, split, "--hf-split", "--split")
    final_hf_revision = (
        hf_revision  # This parameter is the same in both unified and legacy
    )

    # Validate that exactly one dataset source is specified
    if not final_dataset_path and not final_hf:
        raise click.UsageError(
            "Must specify either --dataset/--dataset_path or --hf-dataset/--hf"
        )
    elif final_dataset_path and final_hf:
        raise click.UsageError(
            "Cannot specify both local dataset and HuggingFace dataset parameters at the same time"
        )

    return {
        "dataset_path": final_dataset_path,
        "hf": final_hf,
        "config": final_config,
        "split": final_split,
        "hf_revision": final_hf_revision,
    }


@click.command()
@click.argument(
    "command",
    type=click.Choice(["run", "judge", "gaia2-run"]),
    required=True,
)
@common_options()
@runtime_config_options()
# Benchmark-specific scenario parameters
@click.option(
    "-d",
    "--dataset",
    type=str,
    required=False,
    help="Dataset directory containing scenarios as JSON files, or JSONL file listing scenarios",
)
@click.option(
    "--hf-dataset",
    type=str,
    required=False,
    help="HuggingFace dataset path",
)
@click.option(
    "--hf-config",
    type=str,
    required=False,
    help="Dataset config (subset) name for HuggingFace datasets",
)
@click.option(
    "--hf-split",
    type=str,
    required=False,
    help="Dataset split name (e.g., 'test', 'validation', 'train')",
)
@click.option(
    "--hf-revision",
    type=str,
    required=False,
    help="HuggingFace dataset revision",
)
@click.option(
    "-l",
    "--limit",
    type=int,
    required=False,
    help="Limit the number of scenarios to run per config",
)
@click.option(
    "--enable_caching",
    is_flag=True,
    default=False,
    help="Enable caching of results.",
)
@click.option(
    "--executor_type",
    type=click.Choice(["thread", "process"]),
    default="process",
    help="Type of executor to use for running scenarios.",
)
# Legacy scenario parameters (for backward compatibility)
@click.option(
    "--dataset_path",
    type=str,
    required=False,
    help="Dataset directory or JSONL file (deprecated, use --dataset instead)",
    hidden=True,
)
@click.option(
    "--hf",
    type=str,
    required=False,
    help="HuggingFace dataset path (deprecated, use --hf-dataset instead)",
    hidden=True,
)
@click.option(
    "--config",
    type=str,
    required=False,
    help="Dataset config (subset) name",
)
@click.option(
    "--split",
    type=str,
    required=False,
    help="Dataset split name (e.g., 'test', 'validation', 'train')",
)
@output_config_options()
@internal_options()
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
@click.option(
    "--trace_dump_format",
    default="both",
    type=click.Choice(["hf", "lite", "both"]),
    help="Format in which to dump traces to JSON. 'hf' for HuggingFace format, 'lite' for lightweight format, 'both' for dual export. Must include 'hf' for upload to HuggingFace.",
)
@click.option(
    "--hf_upload",
    type=str,
    default=None,
    help="Dataset name to upload the traces to HuggingFace. If not specified, the traces are not uploaded.",
)
@click.option(
    "--hf_public",
    type=bool,
    default=False,
    help="If true, the dataset is uploaded as a public dataset. If false, the dataset is uploaded as a private dataset.",
)
@click.option(
    "--scenario_timeout",
    type=int,
    default=DEFAULT_SCENARIO_TIMEOUT,
    help=f"Timeout for each scenario in seconds. Defaults to {DEFAULT_SCENARIO_TIMEOUT} seconds.",
)
@click.option(
    "--a2a_app_prop",
    type=click.FloatRange(0.0, 1.0),
    default=0,
    help="When set to >0, turns on Agent2Agent mode, spinning up independent agents for a `a2a_app_prop` fraction of available Apps per scenario.",
)
@click.option(
    "--a2a_app_agent",
    type=click.Choice(AppAgentBuilder().list_agents()),
    required=False,
    default=DEFAULT_APP_AGENT,
    help="[Agent2Agent mode] Agent used for App agent instances.",
)
@click.option(
    "--a2a_model",
    type=str,
    required=False,
    default=None,
    help="[Agent2Agent mode] Model used for App agent instances.",
)
@click.option(
    "--a2a_model_provider",
    type=click.Choice(PROVIDERS),
    required=False,
    default=None,
    help="[Agent2Agent mode] Provider of the App agent model",
)
@click.option(
    "--a2a_endpoint",
    type=str,
    required=False,
    help="[Agent2Agent mode] URL of the endpoint to contact for running App agent models",
)
@click.option(
    "--num_runs",
    type=int,
    default=3,
    help="Number of times to run each scenario to improve variance. Defaults to 3.",
)
@click.option(
    "--reasoning_effort",
    type=click.Choice(["minimal", "low", "medium", "high"]),
    default=None,
    help="Reasoning effort level for the agent model (e.g., 'low', 'medium', 'high'). Only applicable to reasoning-capable models.",
)
@click.option(
    "--judge_model",
    type=str,
    default=DEFAULT_JUDGE_MODEL,
    help="Model to use for the judge system. Use a capable model for best evaluation quality.",
)
@click.option(
    "--judge_provider",
    type=click.Choice(PROVIDERS),
    default=None,
    help="Provider for the judge model. If not specified, uses the same provider as the main model.",
)
@click.option(
    "--judge_endpoint",
    type=str,
    default=None,
    help="URL of the endpoint for the judge model. Optional for custom endpoints.",
)
@click.pass_context
def main(
    ctx: click.Context,
    command: str,
    model: str,
    provider: str | None = None,
    endpoint: str | None = None,
    agent: str | None = None,
    log_level: str = "INFO",
    oracle: bool = False,
    simulated_generation_time_mode: str = "measured",
    noise: bool = False,
    max_concurrent_scenarios: int | None = None,
    executor_type: str = "thread",
    enable_caching: bool = False,
    # Benchmark-specific scenario parameters
    dataset: str | None = None,
    hf_dataset: str | None = None,
    hf_config: str | None = None,
    hf_split: str | None = None,
    hf_revision: str | None = None,
    limit: int | None = None,
    # Legacy scenario parameters (for backward compatibility)
    dataset_path: str | None = None,
    hf: str | None = None,
    config: str | None = None,
    split: str | None = None,
    output_dir: str | None = None,
    trace_dump_format: str = "both",
    hf_upload: str | None = None,
    hf_public: bool = False,
    scenario_timeout: int = DEFAULT_SCENARIO_TIMEOUT,
    a2a_app_prop: float = 0,
    a2a_app_agent: str = "",
    a2a_model: str | None = None,
    a2a_model_provider: str | None = None,
    a2a_endpoint: str | None = None,
    num_runs: int = 3,
    reasoning_effort: str | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_provider: str | None = None,
    judge_endpoint: str | None = None,
    # Checkpoint parameters
    enable_checkpoints: bool = False,
    checkpoint_verbose: bool = False,
    # Hints file
    hints_file: str | None = None,
    **kwargs,
):
    """
    Main entry point for the Meta Agents Research Environments benchmark CLI.

    This function processes command line arguments and runs the benchmark with the specified
    configuration. It supports three main commands: "run" for executing scenarios, "judge"
    for offline validation of scenarios, and "gaia2-run" for complete GAIA2 evaluation.
    """
    # Set up logging with tqdm support
    setup_logging(log_level, use_tqdm=True)
    suppress_noisy_loggers()

    # Handle judge mode num_runs validation
    if command == "judge":
        # Check if num_runs was explicitly provided
        if (
            "num_runs" in ctx.params
            and ctx.get_parameter_source("num_runs")
            != click.core.ParameterSource.DEFAULT
        ):
            raise click.UsageError(
                "In judge mode, num_runs should not be explicitly specified. It is automatically set to 1."
            )
        # Set num_runs to 1 for judge mode
        num_runs = 1

    # Validate and map benchmark scenario parameters
    scenario_params = validate_benchmark_scenario_sources(
        dataset=dataset,
        hf_dataset=hf_dataset,
        hf_config=hf_config,
        hf_split=hf_split,
        hf_revision=hf_revision,
        dataset_path=dataset_path,
        hf=hf,
        config=config,
        split=split,
    )

    final_dataset_path = scenario_params["dataset_path"]
    final_hf = scenario_params["hf"]
    final_config = scenario_params["config"]
    final_split = scenario_params["split"]
    final_hf_revision = scenario_params["hf_revision"]

    assert final_hf is not None or final_dataset_path is not None, (
        "At least one of dataset_path (-d) or HuggingFace datasets (--hf) must be specified."
    )

    assert not (final_hf is not None and final_dataset_path is not None), (
        "Only one of dataset_path (-d) or HuggingFace datasets (--hf) can be specified, not both."
    )

    if final_hf and not final_split:
        logger.warning(
            "No split specified for HuggingFace dataset. Defaulting to 'validation'."
        )
        final_split = "validation"

    # Set default judge_provider to match model_provider if not specified
    if judge_provider is None:
        judge_provider = provider

    # Handle gaia2-run command
    if command == "gaia2-run":
        if trace_dump_format == "lite" and hf_upload is not None:
            raise click.UsageError(
                "Only HuggingFace format is supported for uploading to HuggingFace."
            )

        if hf_upload is not None and final_hf is None:
            raise click.UsageError(
                "We only support uploading to HuggingFace (--hf_upload) if the input dataset is a HuggingFace dataset (--hf-dataset/--hf)."
            )

        assert agent is None or agent == "default", (
            "For gaia2-run, only the default agent is supported."
        )

        gaia2_kwargs = {}

        gaia2_results = handle_gaia2_run(
            model=model,
            dataset_path=final_dataset_path,
            hf=final_hf,
            split=final_split,
            hf_revision=final_hf_revision,
            limit=limit,
            model_provider=provider,
            endpoint=endpoint,
            oracle=oracle,
            output_dir=output_dir,
            hf_upload=hf_upload,
            hf_public=hf_public,
            max_concurrent_scenarios=max_concurrent_scenarios,
            executor_type=executor_type,
            enable_caching=enable_caching,
            scenario_timeout=scenario_timeout,
            a2a_app_agent=a2a_app_agent,
            a2a_model=a2a_model,
            a2a_model_provider=a2a_model_provider,
            a2a_endpoint=a2a_endpoint,
            simulated_generation_time_mode=simulated_generation_time_mode,
            judge_model=judge_model,
            judge_provider=judge_provider,
            judge_endpoint=judge_endpoint,
            log_level=log_level,
            **gaia2_kwargs,
        )

        # Generate summary using polars-based reporting for gaia2-run
        generate_and_save_reports(gaia2_results, model, provider, output_dir, num_runs)

        logger.info("All Done.")
        return gaia2_results

    if hf_upload is not None:
        assert final_hf is not None, (
            "We only support uploading to HuggingFace (--hf_upload) if the input dataset is a HuggingFace dataset (--hf)."
        )

    offline_validation = command == "judge"

    # Create noise configs if --noise flag is enabled
    tool_augmentation_config = None
    env_events_config = None
    if noise:
        tool_augmentation_config, env_events_config = create_noise_configs()

    # Initialize configs list
    configs = []

    if final_config is None:
        from are.simulation.types import CapabilityTag

        configs = [
            capability.value.lower()
            for capability in CapabilityTag.gaia2_capabilities()
        ]

        # In judge mode with HuggingFace dataset, only use configs that are available in the dataset
        if offline_validation and final_hf:
            try:
                from datasets import get_dataset_config_names

                available_configs = get_dataset_config_names(final_hf)
                # Filter configs to only include those available in the dataset, do not process demo config
                filtered_configs = [
                    c for c in configs if c in available_configs and c != DEMO_CONFIG
                ]
                if filtered_configs:
                    configs = filtered_configs
                    logger.info(
                        f"Judge mode: Filtered to available configs in dataset {final_hf}: "
                        + ", ".join(configs)
                    )
                else:
                    logger.warning(
                        f"Judge mode: No GAIA2 capability configs found in dataset {final_hf}. "
                        f"Available configs: {available_configs}"
                    )
            except Exception as e:
                logger.warning(
                    f"Judge mode: Failed to get available configs from dataset {final_hf}: {e}. "
                    "Using all GAIA2 configs."
                )

        if not offline_validation or not final_hf:
            logger.info(
                "No config specified. Using the following configs: "
                + ", ".join(configs)
            )
    else:
        configs = [final_config]

    results = {}
    failed_configs = []
    for config in configs:
        try:
            run_dataset_kwargs = {}

            result = run_dataset(
                model=model,
                dataset_path=final_dataset_path,
                hf=final_hf,
                config=config,
                split=final_split,
                hf_revision=final_hf_revision,
                limit=limit,
                model_provider=provider,
                endpoint=endpoint,
                agent=agent,
                oracle=oracle,
                offline_validation=offline_validation,
                output_dir=output_dir,
                trace_dump_format=trace_dump_format,
                max_concurrent_scenarios=max_concurrent_scenarios,
                enable_caching=enable_caching,
                executor_type=executor_type,
                scenario_timeout=scenario_timeout,
                a2a_app_prop=a2a_app_prop,
                a2a_app_agent=a2a_app_agent,
                a2a_model=a2a_model,
                a2a_model_provider=a2a_model_provider,
                a2a_endpoint=a2a_endpoint,
                simulated_generation_time_mode=simulated_generation_time_mode,
                tool_augmentation_config=tool_augmentation_config,
                env_events_config=env_events_config,
                num_runs=num_runs,
                judge_model=judge_model,
                judge_provider=judge_provider,
                judge_endpoint=judge_endpoint,
                reasoning_effort=reasoning_effort,
                log_level=log_level,  # Pass log level to run_dataset
                enable_checkpoints=enable_checkpoints,
                checkpoint_verbose=checkpoint_verbose,
                hints_file_path=hints_file,
                **run_dataset_kwargs,
            )
            results[config] = result
            logger.info(f"Successfully completed config '{config}'")
        except Exception as e:
            logger.error(f"Config '{config}' failed with error: {e}", exc_info=e)
            failed_configs.append(config)
            # Continue with next config

    # Generate summary using polars-based reporting
    if results:
        # Transform results to the format expected by polars functions
        # Map config names to tuples with metadata
        transformed_results = {}
        for config, result in results.items():
            # Create a tuple key with metadata for polars functions
            # (phase_name, config, a2a_app_prop, has_tool_augmentation, has_env_events)
            result_key = (
                "standard",  # Default phase name for CLI runs
                config,
                a2a_app_prop,
                tool_augmentation_config is not None,
                env_events_config is not None,
            )
            transformed_results[result_key] = result

        generate_and_save_reports(transformed_results, model, provider, output_dir)
    else:
        logger.warning("No results to report.")

    # Report failed configs
    if failed_configs:
        logger.warning(
            f"The following {len(failed_configs)} config(s) failed entirely "
            "and were skipped:"
        )
        for config in failed_configs:
            logger.warning(f"  - {config}")
        logger.warning("Check the logs above for specific error details.")

    # Summary
    total_configs = len(configs)
    successful_configs = len(results)
    failed_config_count = len(failed_configs)

    logger.info("Benchmark run summary:")
    logger.info(f"  Total configs attempted: {total_configs}")
    logger.info(f"  Successful configs: {successful_configs}")
    logger.info(f"  Failed configs: {failed_config_count}")

    if successful_configs > 0:
        logger.info("All Done.")
    else:
        logger.error("All configs failed. No results generated.")

    return results


if __name__ == "__main__":
    main()
