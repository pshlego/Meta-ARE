"""
Checkpoint replay system for restoring and resuming from saved checkpoints.

This module provides functionality to load checkpoints and restore the ARE
system state, enabling replay from any saved checkpoint point.

Key Classes:
- CheckpointLoader: Loads checkpoint data from storage
- CheckpointReplayRunner: Loads checkpoint and continues execution
- ScenarioMetaLoader: Loads scenario metadata from scenario_meta.json
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from are.simulation.agents.agent_builder import AgentBuilder
from are.simulation.agents.agent_config_builder import AgentConfigBuilder
from are.simulation.agents.agent_log import BaseAgentLog, ToolCallLog, ObservationLog
from are.simulation.agents.are_simulation_agent import RunnableARESimulationAgent
from are.simulation.agents.default_agent.are_simulation_main import ARESimulationAgent
from are.simulation.agents.are_simulation_agent_config import (
    LLMEngineConfig,
    MainAgentConfig,
    RunnableARESimulationAgentConfig,
)
from are.simulation.agents.default_agent.tools.action_executor import ParsedAction
from are.simulation.agents.default_agent.tools.json_action_executor import (
    JsonActionExecutor,
    get_observation_log,
)
from are.simulation.agents.llm.types import MMObservation
from are.simulation.data_handler.exporter import JsonScenarioExporter
from are.simulation.environment import Environment, EnvironmentConfig
from are.simulation.notification_system import VerboseNotificationSystem
from are.simulation.scenarios import Scenario
from are.simulation.scenarios.config import ScenarioRunnerConfig
from are.simulation.scenarios.scenario import ScenarioStatus, ScenarioValidationResult
from are.simulation.types import (
    CompletedEvent,
    ConditionCheckEvent,
    EnvironmentType,
    EventLog,
    SimulatedGenerationTimeConfig,
)

from .config import CheckpointConfig
from .storage import CheckpointStorage
from .metadata import CheckpointMetadata
from .manager import CheckpointManager
from .wrappers import CheckpointableEnvironment

logger = logging.getLogger(__name__)


class CheckpointReplayError(Exception):
    """Exception raised when checkpoint replay fails."""
    pass


class ScenarioMetaLoader:
    """
    Loads scenario metadata from scenario_meta.json.
    
    This class provides access to the ScenarioRunnerConfig and scenario
    parameters saved during checkpointing, enabling automatic replay
    configuration.
    """
    
    def __init__(self, checkpoint_dir: Path, verbose: bool = False):
        """
        Initialize scenario meta loader.
        
        Args:
            checkpoint_dir: Directory containing checkpoints and scenario_meta.json
            verbose: Enable detailed logging
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self._meta_cache: Optional[Dict[str, Any]] = None
    
    def has_scenario_meta(self) -> bool:
        """Check if scenario_meta.json exists."""
        return (self.checkpoint_dir / "scenario_meta.json").exists()
    
    def load_scenario_meta(self) -> Dict[str, Any]:
        """
        Load scenario metadata from scenario_meta.json.
        
        Returns:
            Dictionary with scenario metadata
            
        Raises:
            CheckpointReplayError: If scenario_meta.json cannot be loaded
        """
        if self._meta_cache is not None:
            return self._meta_cache
        
        meta_file = self.checkpoint_dir / "scenario_meta.json"
        if not meta_file.exists():
            raise CheckpointReplayError(
                f"scenario_meta.json not found in {self.checkpoint_dir}"
            )
        
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                self._meta_cache = json.load(f)
            
            if self.verbose:
                self.logger.info(f"Loaded scenario metadata from {meta_file}")
            
            # At this point, _meta_cache is guaranteed to be set
            assert self._meta_cache is not None
            return self._meta_cache
        except Exception as e:
            raise CheckpointReplayError(f"Failed to load scenario_meta.json: {e}")
    
    def get_scenario_id(self) -> str:
        """Get scenario ID from metadata."""
        meta = self.load_scenario_meta()
        return meta.get("scenario_id", "unknown")
    
    def get_runner_config_dict(self) -> Optional[Dict[str, Any]]:
        """
        Get runner config as dictionary.
        
        Returns:
            Runner config dictionary or None if not available
        """
        meta = self.load_scenario_meta()
        return meta.get("runner_config")
    
    def get_runner_config(self) -> Optional[ScenarioRunnerConfig]:
        """
        Reconstruct ScenarioRunnerConfig from metadata.
        
        Returns:
            ScenarioRunnerConfig object or None if not available
        """
        config_dict = self.get_runner_config_dict()
        if config_dict is None:
            return None
        
        try:
            # Use Pydantic to reconstruct the config
            return ScenarioRunnerConfig(**config_dict)
        except Exception as e:
            self.logger.warning(f"Failed to reconstruct ScenarioRunnerConfig: {e}")
            return None
    
    def get_scenario_creation_params(self) -> Optional[Dict[str, Any]]:
        """Get scenario creation parameters."""
        meta = self.load_scenario_meta()
        return meta.get("scenario_creation_params")
    
    def get_scenario_initialization_params(self) -> Optional[Dict[str, Any]]:
        """Get scenario initialization parameters."""
        meta = self.load_scenario_meta()
        return meta.get("scenario_initialization_params")


class CheckpointLoader:
    """
    Loads and validates checkpoint data from storage.
    """
    
    def __init__(self, checkpoint_dir: Path, verbose: bool = False):
        """
        Initialize checkpoint loader.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            verbose: Enable detailed logging
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.verbose = verbose
        self.storage = CheckpointStorage(checkpoint_dir, verbose)
        self.logger = logging.getLogger(__name__)
    
    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Load a checkpoint from storage.
        
        Args:
            checkpoint_id: Checkpoint identifier (directory name)
            
        Returns:
            Complete checkpoint data dictionary
            
        Raises:
            CheckpointReplayError: If checkpoint cannot be loaded
        """
        # For our simplified structure, scenario_id is not used in paths
        checkpoint_data = self.storage.load_checkpoint("", checkpoint_id)
        
        if checkpoint_data is None:
            raise CheckpointReplayError(
                f"Checkpoint {checkpoint_id} not found in {self.checkpoint_dir}"
            )
        
        # Validate checkpoint structure
        required_keys = ["metadata", "environment", "agent"]
        for key in required_keys:
            if key not in checkpoint_data:
                raise CheckpointReplayError(
                    f"Invalid checkpoint: missing required key '{key}'"
                )
        
        return checkpoint_data
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        return self.storage.list_scenario_checkpoints("")
    
    def get_checkpoint_metadata(self, checkpoint_id: str) -> CheckpointMetadata:
        """
        Load checkpoint metadata.
        
        Args:
            checkpoint_id: Checkpoint identifier
            
        Returns:
            CheckpointMetadata object
        """
        checkpoint_data = self.load_checkpoint(checkpoint_id)
        return CheckpointMetadata.from_dict(checkpoint_data["metadata"])


class CheckpointReplayRunner:
    """
    Runner that loads a checkpoint and continues execution from that state.
    
    This class handles:
    1. Loading checkpoint data
    2. Loading scenario metadata from scenario_meta.json
    3. Reconstructing agent logs for replay
    4. Setting up environment and scenario
    5. Continuing agent execution from checkpoint state
    6. Exporting results to a new directory
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        verbose: bool = False,
    ):
        """
        Initialize replay runner.
        
        Args:
            checkpoint_path: Path to checkpoint directory (e.g., "test_results/checkpoints/event_1_after_tool_call")
            verbose: Enable detailed logging
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Validate checkpoint path
        if not self.checkpoint_path.exists():
            raise CheckpointReplayError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        # The checkpoint_dir is the parent of the checkpoint folder
        # e.g., "test_results/checkpoints" for "test_results/checkpoints/event_1_after_tool_call"
        self.checkpoint_dir = self.checkpoint_path.parent
        self.checkpoint_id = self.checkpoint_path.name
        
        # Output dir is sibling to checkpoints dir
        # e.g., "test_results" for "test_results/checkpoints"
        self.base_output_dir = self.checkpoint_dir.parent
        
        # Initialize loaders
        self.loader = CheckpointLoader(self.checkpoint_dir, verbose)
        self.scenario_meta_loader = ScenarioMetaLoader(self.checkpoint_dir, verbose)
        
        # Agent builders
        self.agent_config_builder = AgentConfigBuilder()
        self.agent_builder = AgentBuilder()
        
        # Checkpoint manager for continual run (will be created in run())
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self._current_agent = None  # Agent reference for checkpoint state capture
    
    def has_scenario_meta(self) -> bool:
        """Check if scenario metadata is available."""
        return self.scenario_meta_loader.has_scenario_meta()
    
    def get_scenario_meta(self) -> Dict[str, Any]:
        """
        Get scenario metadata.
        
        Returns:
            Dictionary with scenario_id, runner_config, creation_params, initialization_params
        """
        return self.scenario_meta_loader.load_scenario_meta()
    
    def get_runner_config_from_meta(self) -> Optional[ScenarioRunnerConfig]:
        """
        Get ScenarioRunnerConfig from scenario metadata.
        
        Returns:
            ScenarioRunnerConfig object or None if not available
        """
        return self.scenario_meta_loader.get_runner_config()
    
    def _create_output_dir(self) -> Path:
        """Create timestamped output directory for continual run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.base_output_dir / f"continual_run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _copy_scenario_meta(self, checkpoint_dir: Path) -> None:
        """
        Copy scenario_meta.json from original checkpoint dir to continual run checkpoint dir.
        
        This ensures the continual run checkpoints have access to the same scenario metadata.
        """
        import shutil
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        source_meta = self.checkpoint_dir / "scenario_meta.json"
        dest_meta = checkpoint_dir / "scenario_meta.json"
        
        if source_meta.exists():
            try:
                shutil.copy2(source_meta, dest_meta)
                self.logger.info(f"Copied scenario_meta.json to {checkpoint_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to copy scenario_meta.json: {e}")
    
    def _reconstruct_agent_logs(self, agent_data: Dict[str, Any]) -> List[BaseAgentLog]:
        """
        Reconstruct BaseAgentLog objects from checkpoint data.
        
        Args:
            agent_data: Agent snapshot data from checkpoint
            
        Returns:
            List of BaseAgentLog objects
        """
        logs_data = agent_data.get("logs", [])
        if not logs_data:
            self.logger.warning("No agent logs found in checkpoint")
            return []
        
        reconstructed_logs = []
        for log_data in logs_data:
            try:
                # Use BaseAgentLog.from_dict() to reconstruct
                log = BaseAgentLog.from_dict(log_data)
                reconstructed_logs.append(log)
            except Exception as e:
                self.logger.warning(f"Failed to reconstruct agent log: {e}")
                continue
        
        if self.verbose:
            self.logger.info(f"Reconstructed {len(reconstructed_logs)} agent logs from checkpoint")
        
        return reconstructed_logs
    
    def _reconstruct_world_logs(self, env_data: Dict[str, Any]) -> List[BaseAgentLog]:
        """
        Reconstruct world_logs from environment checkpoint data.
        
        Args:
            env_data: Environment snapshot data from checkpoint
            
        Returns:
            List of BaseAgentLog objects for world_logs
        """
        world_logs_data = env_data.get("world_logs", [])
        if not world_logs_data:
            return []
        
        reconstructed_logs = []
        for log_data in world_logs_data:
            try:
                log = BaseAgentLog.from_dict(log_data)
                reconstructed_logs.append(log)
            except Exception as e:
                self.logger.warning(f"Failed to reconstruct world log: {e}")
                continue
        
        return reconstructed_logs
    
    def _reconstruct_completed_events(self, env_data: Dict[str, Any]) -> List[CompletedEvent]:
        """
        Reconstruct CompletedEvent objects from checkpoint data.
        
        Args:
            env_data: Environment snapshot data
            
        Returns:
            List of CompletedEvent objects
        """
        event_log_data = env_data.get("event_log", {})
        past_events_data = event_log_data.get("past_events", [])
        
        if not past_events_data:
            return []
        
        completed_events = []
        for event_data in past_events_data:
            try:
                event = CompletedEvent.from_dict(event_data)
                completed_events.append(event)
            except Exception as e:
                self.logger.warning(f"Failed to reconstruct completed event: {e}")
                continue
        
        return completed_events
    
    def _restore_apps_state(self, env: Environment, apps_data: Dict[str, Any]) -> None:
        """
        Restore app states from checkpoint data.
        
        Args:
            env: Environment with registered apps
            apps_data: Apps snapshot data from checkpoint
        """
        if not apps_data:
            return
        
        restored_count = 0
        for app_name, app_state_data in apps_data.items():
            if app_name.startswith("_"):  # Skip metadata like _protocol_mappings
                continue
            
            try:
                if hasattr(env, 'apps') and app_name in env.apps:
                    app_instance = env.apps[app_name]
                    
                    # Check if app has load_state method
                    if hasattr(app_instance, 'load_state'):
                        state_data = app_state_data.get("state")
                        if state_data and app_state_data.get("has_state"):
                            app_instance.load_state(state_data)
                            restored_count += 1
                            if self.verbose:
                                self.logger.debug(f"Restored state for app: {app_name}")
            except Exception as e:
                self.logger.warning(f"Failed to restore state for app {app_name}: {e}")
        
        if self.verbose:
            self.logger.info(f"Restored state for {restored_count} apps")

    def _schedule_unfired_events(
        self,
        env: Environment,
        scenario: Scenario,
        completed_events: List[CompletedEvent],
    ) -> int:
        """Schedule scenario events that were not yet completed at checkpoint time.

        During replay, ``schedule_events=False`` prevents ALL events from being
        added to the event queue.  This is correct for events that already
        completed before the checkpoint, but future events — particularly
        ``ConditionCheckEvent`` turn triggers and their ENV-event successors
        needed by adaptability tasks — must still be scheduled so they can fire
        at the right simulation time.

        ``ConditionCheckEvent`` instances are always scheduled regardless of
        whether a prior CHECK_N appears in the completed events, because they
        are polling events that must keep running until their condition is met.
        The fresh event from preprocessing carries the correct condition
        function and successor chain (including ENV events for adaptability
        tasks).

        To prevent duplicate firing of events that already completed before the
        checkpoint (e.g. an ENV event whose condition was already satisfied),
        we seed the event queue's ``already_scheduled`` set with the completed
        event IDs.  The queue's built-in dedup then rejects any successor that
        was already executed.

        Only root events (those with no unresolved dependencies) are scheduled
        here.  Events with dependencies are triggered automatically when their
        parent completes via ``process_event()`` → ``get_successors()``.
        """
        completed_ids = {e.event_id for e in completed_events}

        # Seed the event queue's dedup set so that already-completed events
        # (e.g. ENV events that fired before the checkpoint) are not
        # re-enqueued when a ConditionCheckEvent re-fires and returns them
        # as successors.
        env.event_queue.already_scheduled.update(completed_ids)

        scheduled = 0

        for event in scenario.events:
            is_condition_check = isinstance(event, ConditionCheckEvent)

            # ConditionCheckEvents are polling events that must be re-started
            # even if a prior CHECK_N variant is in the completed event log.
            # For all other event types, skip if already completed.
            if not is_condition_check and event.event_id in completed_ids:
                continue

            # For ConditionCheckEvents, skip if ALL successors have already
            # completed — the condition already fired and its effects are done.
            if is_condition_check and event.successors:
                all_successors_done = all(
                    s.event_id in completed_ids for s in event.successors
                )
                if all_successors_done:
                    self.logger.debug(
                        f"Skipping ConditionCheckEvent {event.event_id}: "
                        f"all successors already completed"
                    )
                    continue

            # Only schedule root events (no unresolved dependencies).
            if len(event.dependencies) == 0:
                # Ensure the event has an absolute time before enqueueing.
                # compute_absolute_time handles both cases: events with
                # event_relative_time (time offset from start/dependency) and
                # events without (defaults to start_time).
                if event.event_time is None:
                    event.compute_absolute_time(env.start_time)
                env.event_queue.put(event)
                scheduled += 1
                self.logger.debug(
                    f"Scheduled unfired event: {event.event_id} "
                    f"(type={type(event).__name__}, time={event.event_time})"
                )

        if scheduled > 0:
            self.logger.info(
                f"Scheduled {scheduled} unfired scenario events for replay"
            )
        return scheduled

    def run(
        self,
        scenario: Scenario,
        config: Optional[ScenarioRunnerConfig] = None,
        enable_checkpoints: bool = True,
        max_duration: Optional[int] = None,
    ) -> Tuple[ScenarioValidationResult, Path]:
        """
        Load checkpoint and continue execution.
        
        Args:
            scenario: The scenario to continue (must match checkpoint scenario)
            config: Runner configuration. If None, will try to load from scenario_meta.json
            enable_checkpoints: Whether to capture checkpoints during continual run
            
        Returns:
            Tuple of (validation result, output directory path)
            
        Raises:
            CheckpointReplayError: If config is None and no scenario_meta.json exists
        """
        start_time = time.time()
        
        # Load config from scenario_meta.json if not provided
        if config is None:
            self.logger.info("No config provided, attempting to load from scenario_meta.json")
            config = self.get_runner_config_from_meta()
            if config is None:
                raise CheckpointReplayError(
                    "Config not provided and could not be loaded from scenario_meta.json"
                )
            self.logger.info("Loaded config from scenario_meta.json")
        
        # Create output directory
        output_dir = self._create_output_dir()
        self.logger.info(f"Continual run output will be saved to: {output_dir}")
        
        # Load checkpoint
        self.logger.info(f"Loading checkpoint from: {self.checkpoint_path}")
        checkpoint_data = self.loader.load_checkpoint(self.checkpoint_id)
        metadata = CheckpointMetadata.from_dict(checkpoint_data["metadata"])
        
        self.logger.info(
            f"Checkpoint info: scenario={metadata.scenario_id}, "
            f"type={metadata.checkpoint_type}, time={metadata.simulation_time}"
        )
        
        # Get starting checkpoint sequence from loaded checkpoint
        starting_event_sequence = metadata.event_sequence_number or 0
        starting_checkpoint_sequence = metadata.checkpoint_sequence or 0
        
        # Set up checkpoint manager for continual run
        if enable_checkpoints:
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_config = CheckpointConfig(
                enabled=True,
                checkpoint_dir=checkpoint_dir,
                verbose_logging=self.verbose,
            )
            self.checkpoint_manager = CheckpointManager(checkpoint_config)
            # Continue sequence from where we left off
            self.checkpoint_manager.event_sequence_counter = starting_event_sequence
            self.checkpoint_manager.checkpoint_sequence_counter = starting_checkpoint_sequence
            self.logger.info(
                f"Checkpointing enabled for continual run, starting from event_{starting_event_sequence + 1}"
            )
            
            # Copy scenario_meta.json to continual run checkpoint dir
            self._copy_scenario_meta(checkpoint_dir)
        
        # Reconstruct agent logs
        agent_data = checkpoint_data.get("agent", {})
        initial_agent_logs = self._reconstruct_agent_logs(agent_data)
        
        # Reconstruct environment state
        env_data = checkpoint_data.get("environment", {})
        world_logs = self._reconstruct_world_logs(env_data)
        completed_events = self._reconstruct_completed_events(env_data)
        
        # Get configuration from checkpoint
        checkpoint_env_config = env_data.get("configuration", {})
        
        # Create environment config - use checkpoint config where available
        env_config = EnvironmentConfig(
            oracle_mode=checkpoint_env_config.get("oracle_mode", config.oracle),
            queue_based_loop=checkpoint_env_config.get("queue_based_loop", config.oracle),
            wait_for_user_input_timeout=checkpoint_env_config.get(
                "wait_for_user_input_timeout", 
                config.wait_for_user_input_timeout
            ),
            dump_dir=str(output_dir) if config.oracle else None,
            time_increment_in_seconds=checkpoint_env_config.get(
                "time_increment_in_seconds",
                scenario.time_increment_in_seconds
            ),
            exit_when_no_events=config.agent is None,
            start_time=env_data.get("start_time", 0),
            verbose=env_data.get("verbose", False),
        )
        
        # Create environment - use CheckpointableEnvironment if checkpointing is enabled
        notification_sys = VerboseNotificationSystem()
        if enable_checkpoints and self.checkpoint_manager:
            env = CheckpointableEnvironment(
                config=env_config,
                checkpoint_manager=self.checkpoint_manager,
                notification_system=notification_sys,
            )
            env.environment_type = EnvironmentType.CLI
        else:
            env = Environment(
                environment_type=EnvironmentType.CLI,
                config=env_config,
                notification_system=notification_sys,
            )
        
        # Mark as replaying initially
        env.is_replaying = True
        
        # Run scenario to set up apps, but do NOT re-schedule scenario events
        # (they are restored from the checkpoint's event_log to avoid duplication)
        env.run(scenario, wait_for_end=False, schedule_events=False)

        # Cap the simulation duration when a max_duration override is provided.
        # This prevents the environment loop from running for the full scenario
        # duration (e.g. 1800s) when replaying from a mid-task checkpoint.
        if max_duration is not None:
            env.duration = max_duration
            self.logger.info(
                f"Replay duration capped to {max_duration}s "
                f"(scenario duration: {scenario.duration}s)"
            )

        # Enable checkpointing after scenario setup
        if enable_checkpoints and self.checkpoint_manager and isinstance(env, CheckpointableEnvironment):
            env.enable_checkpointing(self.checkpoint_manager, scenario.scenario_id)
            # Set up agent reference callback
            env._get_agent_for_checkpoint = lambda: self._current_agent
        
        # Restore world_logs
        env.world_logs = world_logs
        
        # Restore event log with past events
        if completed_events:
            env.event_log = EventLog.from_list_view(completed_events)
            if self.verbose:
                self.logger.info(f"Restored {len(completed_events)} past events")
        
        # Restore apps state
        apps_data = checkpoint_data.get("apps", {})
        self._restore_apps_state(env, apps_data)

        # Schedule scenario events that haven't fired yet at checkpoint time.
        # This is critical for adaptability tasks where ConditionCheckEvent
        # triggers and their ENV event successors must fire during replay.
        self._schedule_unfired_events(env, scenario, completed_events)

        # Set replaying to False before continuing execution
        env.is_replaying = False
        
        # Run the agent with restored logs
        try:
            if config.agent is None:
                self.logger.info("Running without Agent (continuing from checkpoint)")
                env.join()
                validation_result = scenario.validate(env)
            else:
                validation_result = self._run_with_agent(
                    scenario=scenario,
                    env=env,
                    config=config,
                    initial_agent_logs=initial_agent_logs,
                    checkpoint_type=metadata.checkpoint_type,
                )
        except Exception as exception:
            self.logger.exception(f"Failed to continue execution: {exception}")
            validation_result = ScenarioValidationResult(
                success=None, exception=exception
            )
        
        run_duration = time.time() - start_time
        
        # Export trace
        if config.export:
            export_path = self._export_trace(
                env=env,
                scenario=scenario,
                config=config,
                validation_result=validation_result,
                run_duration=run_duration,
                output_dir=output_dir,
            )
            validation_result.export_path = export_path
        
        env.stop()
        
        self.logger.info(
            f"{'✅' if validation_result.success else '❌'} "
            f"Continual run completed: {validation_result}"
        )
        
        return validation_result, output_dir
    
    def _run_with_agent(
        self,
        scenario: Scenario,
        env: Environment,
        config: ScenarioRunnerConfig,
        initial_agent_logs: List[BaseAgentLog],
        checkpoint_type: str = "after_tool_call",
    ) -> ScenarioValidationResult:
        """
        Continue running with agent from checkpoint state.
        
        Args:
            scenario: Scenario to run
            env: Restored environment
            config: Runner configuration
            initial_agent_logs: Reconstructed agent logs from checkpoint
            checkpoint_type: Type of checkpoint being replayed ("before_tool_call" or "after_tool_call")
            
        Returns:
            Validation result
        """
        if config.agent is None:
            raise CheckpointReplayError("Agent name required for replay with agent")
        
        agent_config: RunnableARESimulationAgentConfig = (
            self.agent_config_builder.build(agent_name=config.agent)
        )
        
        agent_config.get_base_agent_config().use_custom_logger = config.use_custom_logger
        agent_config.get_base_agent_config().llm_engine_config = LLMEngineConfig(
            model_name=config.model,
            provider=config.model_provider,
            endpoint=config.endpoint,
        )
        
        simulated_generation_time_config = SimulatedGenerationTimeConfig(
            mode=config.simulated_generation_time_mode  # type: ignore
        )
        agent_config.get_base_agent_config().simulated_generation_time_config = (
            simulated_generation_time_config
        )
        
        if isinstance(agent_config, MainAgentConfig) and config.max_turns is not None:
            agent_config.max_turns = config.max_turns
        
        are_simulation_agent: RunnableARESimulationAgent = self.agent_builder.build(
            agent_config=agent_config, env=env
        )
        
        # Store agent reference for checkpoint state capture
        self._current_agent = are_simulation_agent
        
        # Set up agent checkpoint callback for proper checkpoint timing
        # (checkpoints after ObservationLog is added, not when tool call event is logged)
        if isinstance(env, CheckpointableEnvironment):
            env.setup_agent_checkpoint_callback(are_simulation_agent)
            self.logger.info("Agent checkpoint callback configured for proper checkpoint timing")
        
        self.logger.info(f"Continuing with Agent {config.agent} from checkpoint")
        self.logger.info(f"Replaying {len(initial_agent_logs)} agent logs before continuing")
        
        if checkpoint_type == "before_tool_call":
            # Special handling: the checkpoint was taken before the tool executed.
            # We need to execute the pending tool call before starting new iterations.
            if not isinstance(are_simulation_agent, ARESimulationAgent):
                raise CheckpointReplayError(
                    "Replay from before_tool_call requires ARESimulationAgent, "
                    f"got {type(are_simulation_agent).__name__}"
                )
            result = self._run_from_before_tool_call(
                are_simulation_agent=are_simulation_agent,
                scenario=scenario,
                env=env,
                initial_agent_logs=initial_agent_logs,
            )
        else:
            # Normal replay: checkpoint was taken after tool call completed.
            # The agent logs include the observation, so we can resume normally.
            result = are_simulation_agent.run_scenario(
                scenario=scenario,
                notification_system=env.notification_system,
                initial_agent_logs=initial_agent_logs,
            )
        
        output = result.output
        self.logger.info(f"Agent Output: {output}")
        self.logger.info("Validating...")
        
        validation_result = scenario.validate(env)
        self.logger.info(f"Validation {validation_result} EnvState={env.state}")
        
        return validation_result
    
    def _run_from_before_tool_call(
        self,
        are_simulation_agent: ARESimulationAgent,
        scenario: Scenario,
        env: Environment,
        initial_agent_logs: List[BaseAgentLog],
    ):
        """
        Handle replay from a before_tool_call checkpoint.
        
        When replaying from before_tool_call, the agent logs end with a ToolCallLog
        but the tool hasn't been executed yet. This method:
        1. Sets up the agent (tools, system prompt, replays logs)
        2. Executes the pending tool call
        3. Continues with the normal agent loop
        
        Args:
            are_simulation_agent: The agent to run
            scenario: Scenario being replayed
            env: Restored environment
            initial_agent_logs: Agent logs from checkpoint (ending with ToolCallLog)
            
        Returns:
            AgentExecutionResult
        """
        from are.simulation.agents.agent_execution_result import AgentExecutionResult
        
        # Step 1: Prepare the agent (same as run_scenario does internally)
        are_simulation_agent.prepare_are_simulation_run(
            scenario=scenario,
            notification_system=env.notification_system,
            initial_agent_logs=initial_agent_logs,
        )
        
        # Step 2: Execute the pending tool call
        self._execute_pending_tool_call(are_simulation_agent, env)
        
        # Step 3: Continue with agent loop
        # initial_agent_logs being truthy tells agent_loop to call execute_agent_loop()
        # to finish the current turn before entering the normal turn loop.
        max_turns = are_simulation_agent.max_turns
        if scenario.nb_turns is not None:
            max_turns = scenario.nb_turns
        
        result = are_simulation_agent.agent_loop(
            max_turns=max_turns,
            initial_agent_logs=initial_agent_logs,
        )
        
        return AgentExecutionResult(output=result)
    
    def _execute_pending_tool_call(
        self,
        are_simulation_agent: ARESimulationAgent,
        env: Environment,
    ) -> None:
        """
        Execute the pending tool call from a before_tool_call checkpoint.
        
        The agent logs contain a ToolCallLog at the end (the agent decided to call
        a tool but it hasn't executed yet). This method:
        1. Extracts the ToolCallLog from the agent's replayed logs
        2. Builds a ParsedAction from it
        3. Executes the tool call via the action executor
        4. Appends the ObservationLog (which triggers the after_tool_call checkpoint)
        
        Args:
            are_simulation_agent: Agent with replayed logs
            env: Environment (may be CheckpointableEnvironment)
        """
        react_agent = are_simulation_agent.react_agent
        
        # Find the last ToolCallLog in the agent's logs
        pending_tool_call = None
        for log in reversed(react_agent.logs):
            if isinstance(log, ToolCallLog):
                pending_tool_call = log
                break
        
        if pending_tool_call is None:
            self.logger.warning(
                "No pending ToolCallLog found in agent logs for before_tool_call replay"
            )
            return
        
        tool_name = pending_tool_call.tool_name
        arguments = pending_tool_call.tool_arguments
        
        self.logger.info(f"Executing pending tool call from checkpoint: {tool_name}")
        
        # Set _before_checkpoint_created flag so the after_tool_call checkpoint
        # uses the same event sequence number as the before_tool_call
        if isinstance(env, CheckpointableEnvironment):
            env._before_checkpoint_created = True
        
        # Ensure tools are available in the action executor
        react_agent.init_tools()
        
        # Build ParsedAction from the ToolCallLog
        app_name, action_name = (
            tool_name.split("__") if "__" in tool_name
            else (tool_name, None)
        )
        
        parsed_action = ParsedAction(
            tool_name=tool_name,
            app_name=app_name,
            arguments=arguments if arguments else {},
            action_name=action_name,
        )
        
        # Execute the tool call (this runs through the environment, creating the AGENT event)
        self.logger.info(f"Calling tool: '{tool_name}' with arguments: {arguments}")
        action_executor = react_agent.action_executor
        if not isinstance(action_executor, JsonActionExecutor):
            raise CheckpointReplayError(
                "Replay from before_tool_call requires JsonActionExecutor, "
                f"got {type(action_executor).__name__}"
            )
        observation = action_executor.execute_tool_call(
            parsed_action, react_agent.append_agent_log, react_agent.make_timestamp
        )
        
        # Append the observation log via append_agent_log
        # This triggers the log_callback → checkpoint callback → after_tool_call checkpoint
        if isinstance(observation, MMObservation):
            react_agent.append_agent_log(
                get_observation_log(
                    react_agent.make_timestamp(),
                    observation.content,
                    react_agent.agent_id,
                    observation.attachments,
                )
            )
        else:
            react_agent.append_agent_log(
                get_observation_log(
                    react_agent.make_timestamp(),
                    str(observation),
                    react_agent.agent_id,
                )
            )
        
        self.logger.info(f"Pending tool call executed successfully: {tool_name}")
    
    def _export_trace(
        self,
        env: Environment,
        scenario: Scenario,
        config: ScenarioRunnerConfig,
        validation_result: ScenarioValidationResult,
        run_duration: float,
        output_dir: Path,
    ) -> Optional[str]:
        """Export trace to output directory."""
        validation_decision = (
            ScenarioStatus.Valid.value
            if validation_result.success
            else ScenarioStatus.Invalid.value
        )
        validation_rationale = validation_result.rationale
        
        scenario_exporter = JsonScenarioExporter()
        has_hf_metadata = getattr(scenario, "hf_metadata", None) is not None
        
        success, export_path = scenario_exporter.export_to_json_file(
            env,
            scenario,
            config.model,
            config.agent,
            validation_decision,
            validation_rationale,
            run_duration=run_duration,
            output_dir=str(output_dir),
            export_apps=not has_hf_metadata,
            trace_dump_format=config.trace_dump_format,
            scenario_exception=validation_result.exception,
            runner_config=config,
        )
        
        if success:
            self.logger.info(f"Trace exported to {export_path}")
        else:
            self.logger.error("Failed to export trace")
        
        return export_path


# Convenience functions

def replay_from_checkpoint(
    checkpoint_path: str,
    scenario: Scenario,
    config: Optional[ScenarioRunnerConfig] = None,
    verbose: bool = False,
    max_duration: Optional[int] = None,
) -> Tuple[ScenarioValidationResult, Path]:
    """
    Convenience function to replay from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        scenario: Scenario to continue
        config: Runner configuration. If None, will try to load from scenario_meta.json
        verbose: Enable detailed logging
        max_duration: Override max simulation duration for replay (seconds)

    Returns:
        Tuple of (validation result, output directory path)
    """
    runner = CheckpointReplayRunner(checkpoint_path, verbose=verbose)
    return runner.run(scenario, config, max_duration=max_duration)


def list_checkpoints(checkpoint_dir: str) -> List[Dict[str, Any]]:
    """
    List available checkpoints in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        List of checkpoint information dictionaries
    """
    loader = CheckpointLoader(Path(checkpoint_dir), verbose=False)
    checkpoint_ids = loader.list_checkpoints()
    
    checkpoints_info = []
    for checkpoint_id in checkpoint_ids:
        try:
            metadata = loader.get_checkpoint_metadata(checkpoint_id)
            checkpoints_info.append({
                "checkpoint_id": checkpoint_id,
                "scenario_id": metadata.scenario_id,
                "checkpoint_type": metadata.checkpoint_type,
                "simulation_time": metadata.simulation_time,
                "created_at": metadata.created_at,
                "event_sequence": metadata.event_sequence_number,
                "checkpoint_sequence": metadata.checkpoint_sequence,
            })
        except Exception as e:
            checkpoints_info.append({
                "checkpoint_id": checkpoint_id,
                "error": str(e)
            })
    
    return checkpoints_info


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Get information about a specific checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        Dictionary with checkpoint information including scenario metadata if available
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise CheckpointReplayError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    loader = CheckpointLoader(path.parent, verbose=False)
    checkpoint_data = loader.load_checkpoint(path.name)
    metadata = CheckpointMetadata.from_dict(checkpoint_data["metadata"])
    
    agent_data = checkpoint_data.get("agent", {})
    env_data = checkpoint_data.get("environment", {})
    apps_data = checkpoint_data.get("apps", {})
    
    info = {
        "checkpoint_id": path.name,
        "checkpoint_path": str(path),
        "scenario_id": metadata.scenario_id,
        "checkpoint_type": metadata.checkpoint_type,
        "simulation_time": metadata.simulation_time,
        "created_at": metadata.created_at,
        "event_sequence": metadata.event_sequence_number,
        "checkpoint_sequence": metadata.checkpoint_sequence,
        "agent_logs_count": len(agent_data.get("logs", [])),
        "past_events_count": len(env_data.get("event_log", {}).get("past_events", [])),
        "world_logs_count": len(env_data.get("world_logs", [])),
        "apps_count": len([k for k in apps_data.keys() if not k.startswith("_")]),
    }
    
    # Add scenario metadata info if available
    scenario_meta_loader = ScenarioMetaLoader(path.parent, verbose=False)
    if scenario_meta_loader.has_scenario_meta():
        try:
            scenario_meta = scenario_meta_loader.load_scenario_meta()
            info["has_scenario_meta"] = True
            info["scenario_meta_created_at"] = scenario_meta.get("created_at")
            
            runner_config = scenario_meta.get("runner_config")
            if runner_config:
                info["runner_config_available"] = True
                info["runner_config_model"] = runner_config.get("model")
                info["runner_config_agent"] = runner_config.get("agent")
                info["runner_config_max_turns"] = runner_config.get("max_turns")
            else:
                info["runner_config_available"] = False
            
            info["has_creation_params"] = scenario_meta.get("scenario_creation_params") is not None
            info["has_initialization_params"] = scenario_meta.get("scenario_initialization_params") is not None
        except Exception as e:
            info["has_scenario_meta"] = True
            info["scenario_meta_error"] = str(e)
    else:
        info["has_scenario_meta"] = False
    
    return info
