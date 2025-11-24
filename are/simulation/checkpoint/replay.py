"""
Checkpoint replay system for restoring and resuming from saved checkpoints.

This module provides functionality to load checkpoints and restore the ARE
system state, enabling replay from any saved checkpoint point.

Note on Agent State Restoration:
    Agent state (logs, iterations, custom_state) is captured in checkpoints but
    restoration must be done at the ScenarioRunner level since:
    - Environment doesn't create agents
    - Agents are built by ScenarioRunner
    - Agent logs are passed via initial_agent_logs parameter to agent.run_scenario()
    
    To replay with agent state:
    1. Load checkpoint with CheckpointLoader
    2. Extract agent state: checkpoint_data["agent"]
    3. Restore agent logs: AgentStateRestorer.get_initial_agent_logs_for_replay()
    4. Create agent and pass logs to run_scenario(initial_agent_logs=restored_logs)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from are.simulation.environment import Environment, EnvironmentConfig, EnvironmentType
from are.simulation.types import EventLog, EventQueue, AbstractEvent, CompletedEvent
from are.simulation.notification_system import BaseNotificationSystem

from .config import CheckpointConfig
from .storage import CheckpointStorage
from .metadata import CheckpointMetadata

logger = logging.getLogger(__name__)


class CheckpointReplayError(Exception):
    """Exception raised when checkpoint replay fails."""
    pass


class CheckpointLoader:
    """
    Loads and validates checkpoint data from storage.
    """
    
    def __init__(self, checkpoint_config: CheckpointConfig):
        """
        Initialize checkpoint loader.
        
        Args:
            checkpoint_config: Checkpoint system configuration
        """
        self.config = checkpoint_config
        self.storage = CheckpointStorage(
            checkpoint_config.checkpoint_dir,
            checkpoint_config.verbose_logging
        )
        self.logger = logging.getLogger(__name__)
    
    def load_checkpoint(self, scenario_id: str, checkpoint_id: str) -> Dict[str, Any]:
        """
        Load a checkpoint from storage.
        
        Args:
            scenario_id: Scenario identifier
            checkpoint_id: Checkpoint identifier
            
        Returns:
            Complete checkpoint data dictionary
            
        Raises:
            CheckpointReplayError: If checkpoint cannot be loaded
        """
        checkpoint_data = self.storage.load_checkpoint(scenario_id, checkpoint_id)
        
        if checkpoint_data is None:
            raise CheckpointReplayError(
                f"Checkpoint {checkpoint_id} not found for scenario {scenario_id}"
            )
        
        # Validate checkpoint structure
        required_keys = ["metadata", "environment"]
        for key in required_keys:
            if key not in checkpoint_data:
                raise CheckpointReplayError(
                    f"Invalid checkpoint: missing required key '{key}'"
                )
        
        return checkpoint_data
    
    def list_checkpoints(self, scenario_id: str) -> List[str]:
        """List all available checkpoints for a scenario."""
        return self.storage.list_scenario_checkpoints(scenario_id)
    
    def get_checkpoint_metadata(self, scenario_id: str, checkpoint_id: str) -> CheckpointMetadata:
        """
        Load checkpoint metadata.
        
        Args:
            scenario_id: Scenario identifier
            checkpoint_id: Checkpoint identifier
            
        Returns:
            CheckpointMetadata object
        """
        checkpoint_data = self.load_checkpoint(scenario_id, checkpoint_id)
        return CheckpointMetadata.from_dict(checkpoint_data["metadata"])


class EnvironmentRestorer:
    """
    Restores Environment state from checkpoint data.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize environment restorer.
        
        Args:
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
    
    def restore_environment(
        self,
        checkpoint_data: Dict[str, Any],
        environment_type: EnvironmentType = EnvironmentType.UNKNOWN,
        notification_system: Optional[BaseNotificationSystem] = None
    ) -> Environment:
        """
        Restore an Environment from checkpoint data.
        
        Args:
            checkpoint_data: Complete checkpoint data
            environment_type: Type of environment to create
            notification_system: Optional notification system
            
        Returns:
            Restored Environment instance
        """
        env_state = checkpoint_data.get("environment", {})
        
        # Create environment configuration from checkpoint
        config = self._restore_environment_config(env_state)
        
        # Create new environment
        env = Environment(
            config=config,
            environment_type=environment_type,
            notification_system=notification_system
        )
        
        # Restore time state
        self._restore_time_state(env, env_state)
        
        # Restore event log
        self._restore_event_log(env, env_state)
        
        # Restore event queue
        self._restore_event_queue(env, env_state)
        
        # Restore runtime state
        self._restore_runtime_state(env, env_state)
        
        # Restore apps state
        self._restore_apps_state(env, checkpoint_data.get("apps", {}))
        
        if self.verbose:
            self.logger.info(
                f"Restored environment to time {env.current_time} "
                f"with {len(env.event_log.list_view())} past events and "
                f"{len(env.event_queue.list_view())} queued events"
            )
        
        return env
    
    def _restore_environment_config(self, env_state: Dict[str, Any]) -> EnvironmentConfig:
        """Restore environment configuration from checkpoint."""
        config_data = env_state.get("configuration", {})
        
        return EnvironmentConfig(
            oracle_mode=config_data.get("oracle_mode", False),
            queue_based_loop=config_data.get("queue_based_loop", True),
            time_increment_in_seconds=config_data.get("time_increment_in_seconds", 1),
            duration=config_data.get("duration", None),
            exit_when_no_events=config_data.get("exit_when_no_events", False),
            wait_for_user_input_timeout=config_data.get("wait_for_user_input_timeout", None),
            start_time=env_state.get("start_time", 0),
            verbose=env_state.get("verbose", False),
        )
    
    def _restore_time_state(self, env: Environment, env_state: Dict[str, Any]) -> None:
        """Restore time management state."""
        current_time = env_state.get("current_time", 0)
        start_time = env_state.get("start_time", 0)
        tick_count = env_state.get("tick_count", 0)
        
        # Reset time manager to checkpoint time
        if hasattr(env, 'time_manager'):
            env.time_manager.reset(start_time=start_time)
            # Note: Time manager will need to be advanced through normal tick operations
            # Direct time manipulation may not be supported by all TimeManager implementations
        
        env.current_time = current_time
        env.start_time = start_time
        env.tick_count = tick_count
        
        if self.verbose:
            self.logger.debug(f"Restored time state: current_time={current_time}, tick_count={tick_count}")
    
    def _restore_event_log(self, env: Environment, env_state: Dict[str, Any]) -> None:
        """Restore event log with past events."""
        event_log_data = env_state.get("event_log", {})
        past_events_data = event_log_data.get("past_events", [])
        
        if not past_events_data:
            if self.verbose:
                self.logger.debug("No past events to restore")
            return
        
        # Reconstruct CompletedEvent objects from serialized data
        past_events = []
        for event_data in past_events_data:
            try:
                # Use CompletedEvent.from_dict() if available
                if hasattr(CompletedEvent, 'from_dict'):
                    event = CompletedEvent.from_dict(event_data)
                    past_events.append(event)
                else:
                    self.logger.warning("CompletedEvent.from_dict() not available, skipping event restoration")
                    break
            except Exception as e:
                self.logger.warning(f"Failed to restore past event: {e}")
                continue
        
        # Replace event log with restored events
        if past_events:
            env.event_log = EventLog.from_list_view(past_events)
            if self.verbose:
                self.logger.info(f"Restored {len(past_events)} past events")
    
    def _restore_event_queue(self, env: Environment, env_state: Dict[str, Any]) -> None:
        """Restore event queue with future events."""
        event_queue_data = env_state.get("event_queue", {})
        future_events_data = event_queue_data.get("future_events", [])
        already_scheduled = set(event_queue_data.get("already_scheduled", []))
        
        if not future_events_data:
            if self.verbose:
                self.logger.debug("No future events to restore")
            return
        
        # Reconstruct Event objects from serialized data
        future_events = []
        for event_data in future_events_data:
            try:
                # Determine event type and reconstruct
                # This requires the Event classes to have from_dict() methods
                event_type = event_data.get("event_type") or event_data.get("type")
                
                # Import event types dynamically
                from are.simulation.types import Event
                
                if hasattr(Event, 'from_dict'):
                    event = Event.from_dict(event_data)
                    future_events.append(event)
                else:
                    self.logger.warning("Event.from_dict() not available, skipping event queue restoration")
                    break
            except Exception as e:
                self.logger.warning(f"Failed to restore future event: {e}")
                continue
        
        # Replace event queue with restored events
        if future_events:
            new_queue = EventQueue()
            for event in future_events:
                new_queue.put(event)
            new_queue.already_scheduled = already_scheduled
            env.event_queue = new_queue
            
            if self.verbose:
                self.logger.info(f"Restored {len(future_events)} future events")
    
    def _restore_runtime_state(self, env: Environment, env_state: Dict[str, Any]) -> None:
        """Restore runtime state flags."""
        # Restore state enum
        state_str = env_state.get("state", "SETUP")
        # Keep environment in SETUP state for now, as we'll start it manually
        # env.state = EnvironmentState[state_str] if state_str in EnvironmentState.__members__ else EnvironmentState.SETUP
        
        # Restore other runtime state
        env.is_replaying = True  # Mark as replaying
        env.waiting_for_notification = env_state.get("waiting_for_notification", False)
        env.world_logs = env_state.get("world_logs", [])
        
        if "graphql_cache" in env_state:
            env.graphql_cache = env_state["graphql_cache"]
        
        if self.verbose:
            self.logger.debug(f"Restored runtime state: is_replaying=True")
    
    def _restore_apps_state(self, env: Environment, apps_state: Dict[str, Any]) -> None:
        """
        Restore apps state.
        
        Note: This is a placeholder. Full app restoration requires:
        1. Apps to be registered in the environment
        2. Apps to implement load_state() method
        3. Proper initialization order
        """
        if not apps_state:
            if self.verbose:
                self.logger.debug("No apps state to restore")
            return
        
        restored_count = 0
        for app_name, app_data in apps_state.items():
            if app_name.startswith("_"):  # Skip metadata keys like _protocol_mappings
                continue
            
            try:
                # Check if app exists in environment
                if hasattr(env, 'apps') and app_name in env.apps:
                    app_instance = env.apps[app_name]
                    
                    # Check if app has load_state method
                    if hasattr(app_instance, 'load_state') and app_data.get("has_state"):
                        state_data = app_data.get("state")
                        if state_data:
                            app_instance.load_state(state_data)
                            restored_count += 1
                            if self.verbose:
                                self.logger.debug(f"Restored state for app: {app_name}")
                else:
                    if self.verbose:
                        self.logger.warning(f"App {app_name} not found in environment, skipping state restoration")
            except Exception as e:
                self.logger.warning(f"Failed to restore state for app {app_name}: {e}")
        
        if self.verbose:
            self.logger.info(f"Restored state for {restored_count} apps")


class CheckpointReplayer:
    """
    High-level interface for loading and replaying from checkpoints.
    """
    
    def __init__(self, checkpoint_config: CheckpointConfig):
        """
        Initialize checkpoint replayer.
        
        Args:
            checkpoint_config: Checkpoint system configuration
        """
        self.config = checkpoint_config
        self.loader = CheckpointLoader(checkpoint_config)
        self.restorer = EnvironmentRestorer(verbose=checkpoint_config.verbose_logging)
        self.logger = logging.getLogger(__name__)
    
    def replay_from_checkpoint(
        self,
        scenario_id: str,
        checkpoint_id: str,
        environment_type: EnvironmentType = EnvironmentType.UNKNOWN,
        notification_system: Optional[BaseNotificationSystem] = None
    ) -> Environment:
        """
        Load a checkpoint and restore the environment to that state.
        
        Args:
            scenario_id: Scenario identifier
            checkpoint_id: Checkpoint identifier
            environment_type: Type of environment to create
            notification_system: Optional notification system
            
        Returns:
            Restored Environment ready to continue execution
            
        Raises:
            CheckpointReplayError: If replay fails
        """
        self.logger.info(f"Loading checkpoint {checkpoint_id} for scenario {scenario_id}")
        
        # Load checkpoint data
        checkpoint_data = self.loader.load_checkpoint(scenario_id, checkpoint_id)
        metadata = CheckpointMetadata.from_dict(checkpoint_data["metadata"])
        
        self.logger.info(
            f"Checkpoint metadata: type={metadata.checkpoint_type}, "
            f"time={metadata.simulation_time}, sequence={metadata.checkpoint_sequence}"
        )
        
        # Restore environment
        env = self.restorer.restore_environment(
            checkpoint_data,
            environment_type=environment_type,
            notification_system=notification_system
        )
        
        self.logger.info(f"Successfully restored environment from checkpoint {checkpoint_id}")
        
        return env
    
    def list_available_checkpoints(self, scenario_id: str) -> List[Dict[str, Any]]:
        """
        List all available checkpoints for a scenario with metadata.
        
        Args:
            scenario_id: Scenario identifier
            
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoint_ids = self.loader.list_checkpoints(scenario_id)
        
        checkpoints_info = []
        for checkpoint_id in checkpoint_ids:
            try:
                metadata = self.loader.get_checkpoint_metadata(scenario_id, checkpoint_id)
                checkpoints_info.append({
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_type": metadata.checkpoint_type,
                    "simulation_time": metadata.simulation_time,
                    "created_at": metadata.created_at,
                    "event_sequence": metadata.event_sequence_number,
                    "checkpoint_sequence": metadata.checkpoint_sequence,
                })
            except Exception as e:
                self.logger.warning(f"Failed to load metadata for {checkpoint_id}: {e}")
                checkpoints_info.append({
                    "checkpoint_id": checkpoint_id,
                    "error": str(e)
                })
        
        return checkpoints_info


# Convenience functions for quick replay
def replay_from_checkpoint(
    checkpoint_dir: Path,
    scenario_id: str,
    checkpoint_id: str,
    verbose: bool = False
) -> Environment:
    """
    Convenience function to replay from a checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        scenario_id: Scenario identifier
        checkpoint_id: Checkpoint identifier
        verbose: Enable detailed logging
        
    Returns:
        Restored Environment
    """
    config = CheckpointConfig(
        enabled=True,
        checkpoint_dir=checkpoint_dir,
        verbose_logging=verbose
    )
    
    replayer = CheckpointReplayer(config)
    return replayer.replay_from_checkpoint(scenario_id, checkpoint_id)


def list_checkpoints(checkpoint_dir: Path, scenario_id: str) -> List[Dict[str, Any]]:
    """
    Convenience function to list available checkpoints.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        scenario_id: Scenario identifier
        
    Returns:
        List of checkpoint information
    """
    config = CheckpointConfig(
        enabled=True,
        checkpoint_dir=checkpoint_dir,
        verbose_logging=False
    )
    
    replayer = CheckpointReplayer(config)
    return replayer.list_available_checkpoints(scenario_id)
