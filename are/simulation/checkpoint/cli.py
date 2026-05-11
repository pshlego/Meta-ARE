"""
CLI command for checkpoint replay.

This module provides CLI commands for listing, inspecting, and replaying
from checkpoints created during ARE scenario execution.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from are.simulation.cli.utils import setup_logging
from are.simulation.checkpoint import (
    CheckpointReplayRunner,
    ScenarioMetaLoader,
    get_checkpoint_info,
    list_checkpoints,
)

logger = logging.getLogger(__name__)


@click.group()
def checkpoint():
    """Checkpoint management commands for ARE scenarios."""
    pass


@checkpoint.command("list")
@click.argument("checkpoint_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging level.",
)
def list_cmd(checkpoint_dir: str, log_level: str):
    """
    List all available checkpoints in a directory.
    
    CHECKPOINT_DIR: Directory containing checkpoints (e.g., test_results/checkpoints)
    """
    setup_logging(log_level)
    
    checkpoint_path = Path(checkpoint_dir)
    
    # Check for scenario_meta.json
    meta_loader = ScenarioMetaLoader(checkpoint_path, verbose=False)
    if meta_loader.has_scenario_meta():
        try:
            meta = meta_loader.load_scenario_meta()
            click.echo(f"Scenario: {meta.get('scenario_id', 'unknown')}")
            runner_config = meta.get('runner_config', {})
            if runner_config:
                click.echo(f"  Model: {runner_config.get('model', 'unknown')}")
                click.echo(f"  Agent: {runner_config.get('agent', 'unknown')}")
                click.echo(f"  Max Turns: {runner_config.get('max_turns', 'unknown')}")
            click.echo("")
        except Exception as e:
            click.echo(f"Warning: Could not load scenario_meta.json: {e}", err=True)
    
    # List checkpoints
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        click.echo("No checkpoints found.")
        return
    
    click.echo(f"Found {len(checkpoints)} checkpoint(s):\n")
    
    for cp_info in checkpoints:
        if "error" in cp_info:
            click.echo(f"  ❌ {cp_info['checkpoint_id']}: {cp_info['error']}")
        else:
            click.echo(f"  📁 {cp_info['checkpoint_id']}")
            click.echo(f"     Type: {cp_info.get('checkpoint_type', 'unknown')}")
            click.echo(f"     Created: {cp_info.get('created_at', 'unknown')}")
            click.echo(f"     Agent Logs: {cp_info.get('event_sequence', 0)}")
            click.echo("")


@checkpoint.command("info")
@click.argument("checkpoint_path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Output as JSON.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging level.",
)
def info_cmd(checkpoint_path: str, as_json: bool, log_level: str):
    """
    Show detailed information about a specific checkpoint.
    
    CHECKPOINT_PATH: Path to the checkpoint directory 
                     (e.g., test_results/checkpoints/event_1_after_tool_call)
    """
    setup_logging(log_level)
    
    try:
        info = get_checkpoint_info(checkpoint_path)
        
        if as_json:
            click.echo(json.dumps(info, indent=2, default=str))
        else:
            click.echo(f"Checkpoint: {info['checkpoint_id']}")
            click.echo(f"  Path: {info['checkpoint_path']}")
            click.echo(f"  Scenario ID: {info['scenario_id']}")
            click.echo(f"  Type: {info['checkpoint_type']}")
            click.echo(f"  Created: {info['created_at']}")
            click.echo(f"  Event Sequence: {info['event_sequence']}")
            click.echo(f"  Checkpoint Sequence: {info['checkpoint_sequence']}")
            click.echo("")
            click.echo("State Counts:")
            click.echo(f"  Agent Logs: {info['agent_logs_count']}")
            click.echo(f"  Past Events: {info['past_events_count']}")
            click.echo(f"  World Logs: {info['world_logs_count']}")
            click.echo(f"  Apps: {info['apps_count']}")
            click.echo("")
            
            # Scenario metadata info
            if info.get('has_scenario_meta'):
                click.echo("Scenario Metadata: ✅ Available")
                if info.get('runner_config_available'):
                    click.echo(f"  Model: {info.get('runner_config_model', 'unknown')}")
                    click.echo(f"  Agent: {info.get('runner_config_agent', 'unknown')}")
                    click.echo(f"  Max Turns: {info.get('runner_config_max_turns', 'unknown')}")
                click.echo(f"  Has Creation Params: {info.get('has_creation_params', False)}")
                click.echo(f"  Has Initialization Params: {info.get('has_initialization_params', False)}")
            else:
                click.echo("Scenario Metadata: ❌ Not available")
                
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@checkpoint.command("replay")
@click.argument("checkpoint_path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-s",
    "--scenario-id",
    required=False,
    help="Scenario ID to replay (defaults to scenario from checkpoint metadata).",
)
@click.option(
    "-m",
    "--model",
    required=False,
    help="Override model from checkpoint metadata.",
)
@click.option(
    "-p",
    "--provider",
    required=False,
    help="Override model provider from checkpoint metadata.",
)
@click.option(
    "-a",
    "--agent",
    required=False,
    help="Override agent from checkpoint metadata.",
)
@click.option(
    "--max-turns",
    type=int,
    required=False,
    help="Override max turns from checkpoint metadata. WARNING: Setting this higher than the original may cause the agent to hang waiting for messages that never arrive. Use the original max_turns value unless you understand the scenario's multi-turn behavior.",
)
@click.option(
    "-e",
    "--export",
    is_flag=True,
    default=None,
    help="Export the trace to a JSON file.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False),
    required=False,
    help="Override output directory (defaults to sibling of checkpoint dir).",
)
@click.option(
    "--scenario-file",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="Path to scenario JSON file (required for benchmark scenarios not in the registry).",
)
@click.option(
    "--judge-model",
    required=False,
    help="Model to use for the judge engine (for benchmark scenario validation).",
)
@click.option(
    "--judge-provider",
    required=False,
    help="Model provider for the judge engine (for benchmark scenario validation).",
)
@click.option(
    "--max-duration",
    type=int,
    required=False,
    help="Override max simulation duration for replay (seconds). Limits how long "
         "the environment loop runs, useful when replaying from a mid-task checkpoint.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging level.",
)
def replay_cmd(
    checkpoint_path: str,
    scenario_id: Optional[str],
    model: Optional[str],
    provider: Optional[str],
    agent: Optional[str],
    max_turns: Optional[int],
    export: Optional[bool],
    output_dir: Optional[str],
    scenario_file: Optional[str],
    judge_model: Optional[str],
    judge_provider: Optional[str],
    max_duration: Optional[int],
    verbose: bool,
    log_level: str,
):
    """
    Replay execution from a checkpoint.
    
    CHECKPOINT_PATH: Path to the checkpoint directory
                     (e.g., test_results/checkpoints/event_1_after_tool_call)
    
    The command will load the checkpoint and continue execution from that state.
    By default, it uses the scenario configuration saved in scenario_meta.json.
    You can override specific settings using command line options.
    
    Examples:
    
        # Replay using saved config
        are-checkpoint replay test_results/checkpoints/event_1_after_tool_call
        
        # Replay with different model
        are-checkpoint replay test_results/checkpoints/event_1_after_tool_call -m gpt-4
        
        # Replay with more turns
        are-checkpoint replay test_results/checkpoints/event_1_after_tool_call --max-turns 20
        
        # Replay a benchmark scenario (loaded from JSON file)
        are-checkpoint replay test_results/checkpoints/event_1_before_tool_call \\
            --scenario-file datasets_local/gaia2/search/validation/scenario.json \\
            --judge-model gpt-4 --judge-provider azure
    """
    setup_logging(log_level)
    
    try:
        # Create replay runner
        runner = CheckpointReplayRunner(checkpoint_path, verbose=verbose)
        
        # Load scenario
        scenario = _load_scenario_for_replay(
            runner, scenario_id, scenario_file=scenario_file,
            judge_model=judge_model, judge_provider=judge_provider,
        )
        
        # Load or create config
        config = _prepare_config_for_replay(
            runner=runner,
            model=model,
            provider=provider,
            agent=agent,
            max_turns=max_turns,
            export=export,
            output_dir=output_dir,
            judge_model=judge_model,
            judge_provider=judge_provider,
        )
        
        click.echo(f"Replaying from checkpoint: {checkpoint_path}")
        click.echo(f"  Scenario: {scenario.scenario_id}")
        click.echo(f"  Model: {config.model}")
        click.echo(f"  Agent: {config.agent}")
        click.echo(f"  Max Turns: {config.max_turns}")
        click.echo("")
        
        # Run replay
        result, result_output_dir = runner.run(
            scenario, config, max_duration=max_duration,
        )
        
        # Report results
        click.echo("")
        if result.success:
            click.echo(f"✅ Replay completed successfully!")
        elif result.success is False:
            click.echo(f"❌ Replay completed with validation failure.")
        else:
            click.echo(f"⚠️ Replay completed (validation inconclusive).")
        
        if result.rationale:
            click.echo(f"   Rationale: {result.rationale}")
        
        if result.exception:
            click.echo(f"   Exception: {result.exception}")
        
        click.echo(f"   Output directory: {result_output_dir}")
        
        if result.export_path:
            click.echo(f"   Trace exported to: {result.export_path}")
        
        # Exit with appropriate code
        if result.success is False:
            sys.exit(1)
        elif result.exception:
            sys.exit(2)
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _load_scenario_for_replay(
    runner: CheckpointReplayRunner, 
    scenario_id: Optional[str],
    scenario_file: Optional[str] = None,
    judge_model: Optional[str] = None,
    judge_provider: Optional[str] = None,
) -> "Scenario":
    """
    Load the scenario for replay.
    
    Supports both registry scenarios (from are-run) and benchmark scenarios
    loaded from JSON files (from are-benchmark).
    
    Args:
        runner: The checkpoint replay runner
        scenario_id: Optional scenario ID override
        scenario_file: Path to scenario JSON file (for benchmark scenarios)
        judge_model: Model for judge engine (for benchmark scenario preprocessing)
        judge_provider: Provider for judge engine (for benchmark scenario preprocessing)
    """
    from are.simulation.scenarios.utils.constants import ALL_SCENARIOS
    
    # Determine scenario ID
    if scenario_id is None:
        # Try to get from checkpoint metadata
        if runner.has_scenario_meta():
            meta = runner.get_scenario_meta()
            scenario_id = meta.get("scenario_id")
        
        if scenario_id is None:
            # Try to get from checkpoint metadata file
            checkpoint_data = runner.loader.load_checkpoint(runner.checkpoint_id)
            metadata = checkpoint_data.get("metadata", {})
            scenario_id = metadata.get("scenario_id")
    
    if scenario_id is None:
        raise click.UsageError(
            "Could not determine scenario ID. Please specify --scenario-id."
        )
    
    # Check scenario source type from metadata
    scenario_source_type = None
    scenario_source_path_from_meta = None
    if runner.has_scenario_meta():
        meta = runner.get_scenario_meta()
        scenario_source_type = meta.get("scenario_source_type")
        scenario_source_path_from_meta = meta.get("scenario_source_path")
    
    # Determine the effective scenario file path
    effective_scenario_file = scenario_file or scenario_source_path_from_meta
    
    # Strategy 1: Try loading from registry (for are-run scenarios)
    if scenario_id in ALL_SCENARIOS and scenario_source_type != "json_file":
        scenario_class = ALL_SCENARIOS[scenario_id]
        scenario = scenario_class()
        
        # Initialize scenario - try to use params from metadata
        init_params = {}
        if runner.has_scenario_meta():
            meta_loader = runner.scenario_meta_loader
            init_params_str = meta_loader.get_scenario_initialization_params()
            
            # Parse initialization params if they're a string
            if isinstance(init_params_str, str):
                try:
                    init_params = json.loads(init_params_str)
                except (json.JSONDecodeError, TypeError):
                    init_params = {}
            elif isinstance(init_params_str, dict):
                init_params = init_params_str
        
        scenario.initialize(**init_params)
        return scenario
    
    # Strategy 2: Load from JSON file (for are-benchmark scenarios)
    if effective_scenario_file is not None:
        return _load_benchmark_scenario_from_file(
            runner, effective_scenario_file, judge_model, judge_provider
        )
    
    # Strategy 3: Scenario not found anywhere
    if scenario_source_type == "json_file":
        raise click.UsageError(
            f"Scenario '{scenario_id}' is a benchmark scenario loaded from a JSON file. "
            f"Please provide the scenario file path with --scenario-file."
        )
    
    raise click.UsageError(
        f"Scenario '{scenario_id}' not found in registry. "
        f"If this is a benchmark scenario, use --scenario-file to specify the JSON file path."
    )


def _load_benchmark_scenario_from_file(
    runner: CheckpointReplayRunner,
    scenario_file: str,
    judge_model: Optional[str] = None,
    judge_provider: Optional[str] = None,
) -> "Scenario":
    """
    Load a benchmark scenario from a JSON file and preprocess it.
    
    This handles benchmark scenarios that were created via are-benchmark
    and are not in the scenario registry.
    
    Args:
        runner: The checkpoint replay runner
        scenario_file: Path to the scenario JSON file
        judge_model: Model for judge engine
        judge_provider: Provider for judge engine
    """
    from are.simulation.scenarios.scenario_imported_from_json.utils import (
        preprocess_scenario_from_config,
    )
    from are.simulation.data_handler.importer import JsonScenarioImporter
    from are.simulation.scenarios.config import ScenarioRunnerConfig
    from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
    
    scenario_path = Path(scenario_file)
    if not scenario_path.exists():
        raise click.UsageError(f"Scenario file not found: {scenario_file}")
    
    # Read the JSON file
    with open(scenario_path, 'r', encoding='utf-8') as f:
        scenario_str = f.read()
    
    # Import the scenario from JSON
    (
        scenario,
        _completed_events,
        _,
    ) = JsonScenarioImporter().import_from_json_to_benchmark(
        scenario_str, load_completed_events=True
    )
    
    # Get runner config from metadata for preprocessing
    config = runner.get_runner_config_from_meta()
    if config is None:
        config = ScenarioRunnerConfig(
            model="unknown",
            model_provider="openai",
            agent="default",
        )
    
    # Set up judge engine config for preprocessing
    # Priority: CLI flags > metadata config > no judge
    if judge_model is not None:
        config.judge_engine_config = LLMEngineConfig(
            model_name=judge_model,
            provider=judge_provider or "openai",
        )
    elif config.judge_engine_config is None:
        # Try to get from metadata runner_config
        if runner.has_scenario_meta():
            meta = runner.get_scenario_meta()
            rc = meta.get("runner_config", {})
            jec = rc.get("judge_engine_config")
            if jec and isinstance(jec, dict):
                config.judge_engine_config = LLMEngineConfig(
                    model_name=jec.get("model_name", ""),
                    provider=jec.get("provider", "openai"),
                    endpoint=jec.get("endpoint"),
                )
    
    if config.judge_engine_config is None:
        click.echo(
            "Warning: No judge engine config available. "
            "Benchmark scenario validation may not work correctly. "
            "Use --judge-model and --judge-provider to specify.",
            err=True,
        )
    
    # Preprocess scenario (sets up judge, runs oracle, initializes turns)
    preprocess_scenario_from_config(scenario=scenario, config=config)
    
    click.echo(f"  Loaded benchmark scenario from: {scenario_file}")
    
    return scenario


def _prepare_config_for_replay(
    runner: CheckpointReplayRunner,
    model: Optional[str],
    provider: Optional[str],
    agent: Optional[str],
    max_turns: Optional[int],
    export: Optional[bool],
    output_dir: Optional[str],
    judge_model: Optional[str] = None,
    judge_provider: Optional[str] = None,
) -> "ScenarioRunnerConfig":
    """Prepare configuration for replay, merging metadata with overrides."""
    from are.simulation.scenarios.config import ScenarioRunnerConfig
    from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
    
    # Start with config from metadata
    config = runner.get_runner_config_from_meta()
    
    if config is None:
        # No metadata available, create minimal config
        if model is None:
            raise click.UsageError(
                "No scenario metadata found. Please specify --model."
            )
        if agent is None:
            raise click.UsageError(
                "No scenario metadata found. Please specify --agent."
            )
        
        config = ScenarioRunnerConfig(
            model=model,
            model_provider=provider or "openai",
            agent=agent,
            max_turns=max_turns or 10,
            export=export if export is not None else True,
            output_dir=output_dir or str(runner.base_output_dir),
        )
    else:
        # Apply overrides
        if model is not None:
            config.model = model
        if provider is not None:
            config.model_provider = provider
        if agent is not None:
            config.agent = agent
        if max_turns is not None:
            config.max_turns = max_turns
        if export is not None:
            config.export = export
        if output_dir is not None:
            config.output_dir = output_dir
    
    # Apply judge engine config overrides
    if judge_model is not None:
        config.judge_engine_config = LLMEngineConfig(
            model_name=judge_model,
            provider=judge_provider or "openai",
        )
    
    return config


# Main entry point
def main():
    """Main entry point for the checkpoint CLI."""
    checkpoint()


if __name__ == "__main__":
    main()
