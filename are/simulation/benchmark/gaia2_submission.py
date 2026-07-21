# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
import os
from pathlib import Path

from are.simulation.benchmark.hf_upload_utils import (
    generate_submission_summary,
    upload_consolidated_results_to_hf,
)
from are.simulation.benchmark.scenario_executor import run_dataset
from are.simulation.scenarios.utils.scenario_expander import EnvEventsConfig
from are.simulation.types import ToolAugmentationConfig

logger: logging.Logger = logging.getLogger(__name__)

NUM_RUNS = 3


def handle_gaia2_run(
    model: str,
    dataset_path: str | None = None,
    hf: str | None = None,
    split: str | None = None,
    hf_revision: str | None = None,
    limit: int | None = None,
    model_provider: str | None = None,
    endpoint: str | None = None,
    azure_config_dir: str | None = None,
    oracle: bool = False,
    output_dir: str | None = None,
    hf_upload: str | None = None,
    hf_public: bool = False,
    max_concurrent_scenarios: int | None = None,
    executor_type: str = "thread",
    enable_caching: bool = False,
    scenario_timeout: int = 300,
    a2a_app_agent: str = "",
    a2a_model: str | None = None,
    a2a_model_provider: str | None = None,
    a2a_endpoint: str | None = None,
    simulated_generation_time_mode: str = "measured",
    judge_model: str = "meta-llama/Meta-Llama-3.3-70B-Instruct",
    judge_provider: str | None = None,
    judge_endpoint: str | None = None,
    judge_azure_config_dir: str | None = None,
    log_level: str = "INFO",
    **kwargs,
):
    """Handle the gaia2-submit command for complete evaluation and submission.

    This function automatically runs all required configurations for GAIA2
    leaderboard submission:
    - Standard configs: execution, search, adaptability, time, ambiguity
    - Agent2Agent scenarios: All configs with --a2a_app_prop 1
    - Noise scenarios: All configs with --noise

    Args:
        model: Model name to use for the agent
        dataset_path: Path to local dataset directory or JSONL file
        hf: HuggingFace dataset name
        split: Dataset split name (defaults to 'test' for submissions)
        hf_revision: Revision of the HuggingFace dataset
        limit: Maximum number of scenarios to load
        model_provider: Provider of the model
        endpoint: URL of the endpoint for the model
        oracle: Whether to run in oracle mode
        output_dir: Directory to dump the scenario states and logs
        hf_upload: Dataset name to upload the traces to HuggingFace
        hf_public: Whether to upload the dataset as public
        max_concurrent_scenarios: Maximum number of concurrent scenarios
        executor_type: Type of executor to use for running scenarios
        enable_caching: Enable caching of results
        scenario_timeout: Timeout for each scenario in seconds
        a2a_app_agent: Agent used for App agent instances
        a2a_model: Model used for App agent instances
        a2a_model_provider: Provider of the App agent model
        a2a_endpoint: URL of the endpoint for App agent models
        simulated_generation_time_mode: Mode for simulating generation time
        judge_model: Model to use for the judge system
        judge_provider: Provider for the judge model
        judge_endpoint: URL of the endpoint for the judge model

    Returns:
        Dictionary mapping run types and configs to results
    """
    from are.simulation.types import CapabilityTag

    # Set defaults for submission
    if split is None:
        split = "validation"
        logger.info(
            "No split specified for gaia2-submit. Using 'validation' split for "
            "leaderboard submission."
        )

    if output_dir is None:
        output_dir = "./gaia2_results"
        logger.info(f"No output directory specified. Using default: {output_dir}")

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all GAIA2 capability configs for standard phase
    standard_configs = [
        capability.value.lower() for capability in CapabilityTag.gaia2_capabilities()
    ]

    logger.info("Starting GAIA2 submission pipeline...")
    logger.info(f"Standard phase will run configs: {', '.join(standard_configs)}")
    logger.info("Agent2Agent and Noise phases will run mini config only")
    logger.info("This includes: standard runs, agent2agent runs, and noise runs")

    all_results = {}
    failed_configs = []
    total_scenarios_processed = 0

    # Define phase configurations with their specific parameters and configs
    phase_configs = [
        {
            "phase_name": "standard",
            "configs": standard_configs,
            "a2a_app_prop": 0,
            "tool_augmentation_config": None,
            "env_events_config": None,
            "num_runs": NUM_RUNS,  # Force 3 runs for variance analysis
        },
        {
            "phase_name": "agent2agent",
            "configs": ["mini"],  # Use mini config for agent2agent augmentation
            "a2a_app_prop": 1.0,
            "tool_augmentation_config": None,
            "env_events_config": None,
            "num_runs": NUM_RUNS,
        },
        {
            "phase_name": "noise",
            "configs": ["mini"],  # Use mini config for noise augmentation
            "a2a_app_prop": 0,
            "tool_augmentation_config": ToolAugmentationConfig(),
            "env_events_config": EnvEventsConfig(
                num_env_events_per_minute=10,
                env_events_seed=0,
            ),
            "num_runs": NUM_RUNS,
        },
    ]

    # Execute all phases using the unified function
    for phase_config in phase_configs:
        phase_name = phase_config["phase_name"]
        logger.info(f"=== Phase: Running {phase_name} configurations ===")

        phase_dir = os.path.join(output_dir, phase_name)
        Path(phase_dir).mkdir(parents=True, exist_ok=True)

        for config in phase_config["configs"]:
            # Check if we've already hit the limit
            if limit is not None and total_scenarios_processed >= limit:
                logger.info(f"Reached scenario limit ({limit}). Stopping execution.")
                logger.info(f"Total scenarios processed: {total_scenarios_processed}")
                break

            config_output_dir = os.path.join(phase_dir, config)
            logger.info(f"Running {phase_name} config: {config}")

            # Calculate remaining limit for this run
            remaining_limit = None
            if limit is not None:
                remaining_limit = limit - total_scenarios_processed
                if remaining_limit <= 0:
                    break

            try:
                run_dataset_kwargs = {}

                result = run_dataset(
                    model=model,
                    dataset_path=dataset_path,
                    hf=hf,
                    config=config,
                    split=split,
                    hf_revision=hf_revision,
                    limit=remaining_limit,
                    model_provider=model_provider,
                    endpoint=endpoint,
                    azure_config_dir=azure_config_dir,
                    agent="default",  # for gaia2 submission, we always use default agent
                    oracle=oracle,
                    offline_validation=False,
                    output_dir=config_output_dir,
                    trace_dump_format="both",  # Force HF format for submissions
                    max_concurrent_scenarios=max_concurrent_scenarios,
                    executor_type=executor_type,
                    enable_caching=enable_caching,
                    scenario_timeout=scenario_timeout,
                    a2a_app_prop=phase_config["a2a_app_prop"],
                    a2a_app_agent=a2a_app_agent,
                    a2a_model=a2a_model,
                    a2a_model_provider=a2a_model_provider,
                    a2a_endpoint=a2a_endpoint,
                    simulated_generation_time_mode=simulated_generation_time_mode,
                    tool_augmentation_config=phase_config["tool_augmentation_config"],
                    env_events_config=phase_config["env_events_config"],
                    num_runs=phase_config["num_runs"],
                    judge_model=judge_model,
                    judge_provider=judge_provider,
                    judge_endpoint=judge_endpoint,
                    judge_azure_config_dir=judge_azure_config_dir,
                    log_level=log_level,
                    phase_name=phase_name,
                    **run_dataset_kwargs,
                )

                # Count scenarios processed in this run
                scenarios_in_this_run = len(result.scenario_results)
                total_scenarios_processed += scenarios_in_this_run

                # Create a flattened, hashable key with all phase info
                result_key = (
                    phase_name,
                    config,
                    phase_config["a2a_app_prop"],
                    phase_config["tool_augmentation_config"] is not None,
                    phase_config["env_events_config"] is not None,
                )
                all_results[result_key] = result
                logger.info(f"Completed {phase_name} config: {config}.")
            except Exception as e:
                logger.error(
                    f"Phase '{phase_name}' config '{config}' failed with error: {e}"
                )
                failed_configs.append((phase_name, config))
                # Continue with next config

        # Break out of outer loop if limit reached
        if limit is not None and total_scenarios_processed >= limit:
            break

    # Report failed configs
    if failed_configs:
        logger.warning(
            f"The following {len(failed_configs)} phase/config(s) failed entirely "
            "and were skipped:"
        )
        for phase_name, config in failed_configs:
            logger.warning(f"  - {phase_name}/{config}")
        logger.warning("Check the logs above for specific error details.")

    # Summary
    total_phase_configs = sum(
        len(phase_config["configs"]) for phase_config in phase_configs
    )
    successful_configs = len(all_results)
    failed_config_count = len(failed_configs)

    logger.info("GAIA2 submission summary:")
    logger.info(f"  Total phase/configs attempted: {total_phase_configs}")
    logger.info(f"  Successful phase/configs: {successful_configs}")
    logger.info(f"  Failed phase/configs: {failed_config_count}")
    logger.info(f"  Total scenarios processed: {total_scenarios_processed}")

    if successful_configs == 0:
        logger.error("All phase/configs failed. No results generated.")
        return all_results

    # =====================================================

    # Upload to HuggingFace if requested
    if hf_upload:
        logger.info("=== Uploading consolidated results to HuggingFace ===")
        upload_consolidated_results_to_hf(
            all_results,
            hf_upload,
            model,
            model_provider or "unknown",
            hf_public,
            hf or "local",
            split or "validation",
        )
        logger.info(f"Traces uploaded to HuggingFace dataset: {hf_upload}")
    else:
        # Generate local submission summary if not uploading
        generate_submission_summary(
            output_dir,
            all_results,
            model,
            model_provider or "unknown",
            NUM_RUNS,
        )

    return all_results
