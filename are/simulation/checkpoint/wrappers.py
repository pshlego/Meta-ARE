"""
Checkpoint-enabled wrapper classes for key ARE components.

This module provides wrapper classes that add checkpoint functionality
to the existing ARE components while maintaining minimal invasive design
by inheriting and overriding specific methods.

Checkpoint Timing:
    Two checkpoints are created around each agent tool call:
    
    1. BEFORE tool call: Created when ToolCallLog is added to agent logs.
       Captures the agent's decision (which tool, what arguments) before execution.
    
    2. AFTER tool call: Created when ObservationLog is added to agent logs.
       Captures the complete state including the tool result (observation).
    
    The checkpoint flow is:
    1. Agent calls tool (LLMOutputThoughtActionLog added to agent logs)
    2. ToolCallLog is added to agent logs      <- BEFORE_TOOL_CALL CHECKPOINT
    3. Tool executes
    4. AGENT event added to environment log (env.add_to_log)
    5. ObservationLog is added to agent logs    <- AFTER_TOOL_CALL CHECKPOINT
    6. Agent continues to next step
    
    Both checkpoints for the same tool call share the same event sequence number
    (e.g., event_1_before_tool_call and event_1_after_tool_call).
"""

from typing import Dict, Any, Optional, Callable
import logging
import time

# Import ARE core classes
from are.simulation.environment import Environment, EnvironmentConfig
from are.simulation.scenario_runner import ScenarioRunner

# Import the checkpoint components
from .config import CheckpointConfig
from .manager import CheckpointManager
from .agent_state import capture_agent_state

logger = logging.getLogger(__name__)


class CheckpointableEnvironment(Environment):
    """
    Environment subclass that adds checkpoint functionality for agent tool calls.
    
    This class inherits from Environment and overrides add_to_log to create
    checkpoints after each agent tool call (AGENT events).
    """
    
    def __init__(
        self,
        config: Optional[EnvironmentConfig] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        **kwargs
    ):
        """
        Initialize checkpoint-enabled environment.
        
        Args:
            config: Environment configuration
            checkpoint_manager: Optional CheckpointManager for checkpointing
            **kwargs: Additional arguments passed to parent Environment
        """
        super().__init__(config=config, **kwargs)
        self.checkpoint_manager = checkpoint_manager
        self._checkpoint_enabled = False
        self._scenario_id = None
        self._get_agent_for_checkpoint: Optional[Callable] = None  # Callable that returns agent instance
    
    def enable_checkpointing(self, checkpoint_manager: CheckpointManager, scenario_id: str):
        """Enable checkpointing for this environment."""
        self.checkpoint_manager = checkpoint_manager
        self._scenario_id = scenario_id
        self._checkpoint_enabled = True
        self._pending_checkpoint_event: Optional[Dict[str, Any]] = None  # Track pending checkpoint
        self._before_checkpoint_created = False  # Track if before_tool_call was created for current event
        logger.info(f"Checkpointing enabled for scenario {scenario_id}")
    
    def track_agent_event_for_checkpoint(self, event) -> None:
        """
        Track an AGENT event for pending checkpoint creation.
        
        This is called from add_to_log when an AGENT event is added. The actual
        checkpoint will be created when ObservationLog is added to agent logs
        (via setup_agent_checkpoint_callback).
        
        Args:
            event: The AGENT event to track
        """
        if not self._checkpoint_enabled or not self.checkpoint_manager:
            return
            
        try:
            event_id = getattr(event, 'event_id', str(id(event)))
            
            # Get tool/action info from the event
            action_info = {}
            if hasattr(event, 'action'):
                action = event.action
                if hasattr(action, 'function_name'):
                    action_info['function_name'] = action.function_name
                if hasattr(action, 'app'):
                    app = action.app
                    if app is not None:
                        if hasattr(app, 'name'):
                            action_info['app_name'] = app.name
                        elif hasattr(app, 'app_name'):
                            action_info['app_name'] = app.app_name()
            
            # Store pending checkpoint info (will be created when observation is added)
            self._pending_checkpoint_event = {
                "event_type": "AGENT",
                "event_id": event_id,
                "source": "add_to_log",
                **action_info
            }
            logger.debug(f"Tracked AGENT event for pending checkpoint: {event_id}")
        except Exception as e:
            logger.warning(f"Failed to track AGENT event: {e}", exc_info=True)
    
    def add_to_log(self, events):
        """
        Override add_to_log to track AGENT events for checkpoint creation.
        
        When an AGENT event is added, we store its info for the pending checkpoint.
        The actual checkpoint will be created when ObservationLog is added to agent logs
        (via the callback set up in setup_agent_checkpoint_callback).
        """
        from are.simulation.types import EventType
        
        # First call the parent to actually add the events
        super().add_to_log(events)
        
        # Track AGENT events for pending checkpoint
        if not self._checkpoint_enabled or not self.checkpoint_manager:
            return
        
        # Normalize to list
        if not isinstance(events, list):
            events = [events]
        
        for event in events:
            event_type = getattr(event, 'event_type', None)
            if event_type == EventType.AGENT:
                self.track_agent_event_for_checkpoint(event)
    
    def setup_agent_checkpoint_callback(self, agent) -> None:
        """
        Set up a callback on the agent to create checkpoints around tool calls.
        
        Two checkpoint types are supported:
        - before_tool_call: Created when ToolCallLog is added (before tool executes)
        - after_tool_call: Created when ObservationLog is added (after tool completes)
        
        Args:
            agent: The agent instance to set up callback on
        """
        from are.simulation.agents.agent_log import ObservationLog, ToolCallLog
        
        if not self._checkpoint_enabled or not self.checkpoint_manager:
            logger.debug("Checkpointing not enabled, skipping agent callback setup")
            return
        
        # Get the base agent that has the log_callback
        base_agent = None
        if hasattr(agent, 'react_agent'):
            base_agent = agent.react_agent
        elif hasattr(agent, 'logs'):
            base_agent = agent
        
        if base_agent is None:
            logger.warning("Could not find base agent for checkpoint callback")
            return
        
        # Store original callback (if any)
        original_callback = getattr(base_agent, 'log_callback', None)
        
        # Get which checkpoint types are enabled
        config = self.checkpoint_manager.config
        before_enabled = config.should_create_checkpoint_type("before_tool_call")
        after_enabled = config.should_create_checkpoint_type("after_tool_call")
        
        def checkpoint_log_callback(log):
            """Callback that triggers checkpoints around tool calls."""
            # Call original callback first if exists
            if original_callback is not None:
                original_callback(log)
            
            # BEFORE TOOL CALL: checkpoint when ToolCallLog is added
            if before_enabled and isinstance(log, ToolCallLog):
                try:
                    self._before_checkpoint_created = True
                    # Build event info from the ToolCallLog
                    event_info = {
                        "event_type": "AGENT",
                        "event_id": f"tool_call_{log.tool_name}",
                        "source": "agent_log_callback",
                        "tool_name": log.tool_name,
                        "tool_arguments": str(log.tool_arguments) if log.tool_arguments else "",
                    }
                    logger.info(f"Creating before_tool_call checkpoint for: {log.tool_name}")
                    self._create_checkpoint(
                        checkpoint_type="before_tool_call",
                        event_info=event_info,
                        increment_event_sequence=True,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create before_tool_call checkpoint: {e}", exc_info=True)
            
            # AFTER TOOL CALL: checkpoint when ObservationLog is added and we have a pending event
            if after_enabled and isinstance(log, ObservationLog) and self._pending_checkpoint_event is not None:
                try:
                    event_info = self._pending_checkpoint_event
                    self._pending_checkpoint_event = None  # Clear pending
                    
                    # If before_tool_call was already created for this event,
                    # don't increment event sequence (same tool call event)
                    increment = not self._before_checkpoint_created
                    
                    logger.info(f"Creating after_tool_call checkpoint for: {event_info.get('event_id', 'unknown')}")
                    self._create_checkpoint(
                        checkpoint_type="after_tool_call",
                        event_info=event_info,
                        increment_event_sequence=increment,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create after_tool_call checkpoint: {e}", exc_info=True)
                finally:
                    self._before_checkpoint_created = False
        
        # Set the callback on the base agent
        base_agent.log_callback = checkpoint_log_callback
        logger.info(f"Agent checkpoint callback configured (before={before_enabled}, after={after_enabled})")
    
    def _create_checkpoint(self, checkpoint_type: str, event_info: Dict[str, Any], 
                          related_checkpoint_id: Optional[str] = None,
                          increment_event_sequence: bool = True) -> Optional[str]:
        """Create a checkpoint with comprehensive state capture."""
        try:
            if not self._scenario_id or not self.checkpoint_manager:
                logger.warning("Cannot create checkpoint without scenario_id or checkpoint_manager")
                return None
                
            return self.checkpoint_manager.create_event_checkpoint(
                scenario_id=self._scenario_id,
                checkpoint_type=checkpoint_type,
                simulation_time=time.time(),
                trigger_event_data=event_info,
                system_state={
                    "environment": self._capture_environment_state(),
                    "agent": self._capture_agent_state(),
                    "apps": self._capture_apps_state()
                },
                increment_event_sequence=increment_event_sequence,
            )
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return None
    
    def _capture_environment_state(self) -> Dict[str, Any]:
        """Capture complete environment state for replay."""
        try:
            state_dict = {
                # Time management
                "current_time": self.current_time,
                "start_time": getattr(self, 'start_time', 0),
                "tick_count": getattr(self, 'tick_count', 0),
                
                # Environment state
                "state": str(getattr(self, 'state', 'unknown')),
                "is_replaying": getattr(self, 'is_replaying', False),
                
                # Event system - CRITICAL for replay
                "event_log": self._capture_event_log(),
                "event_queue": self._capture_event_queue(),
                
                # Configuration
                "configuration": {
                    "oracle_mode": getattr(self, 'oracle_mode', False),
                    "queue_based_loop": getattr(self, 'queue_based_loop', True),
                    "time_increment_in_seconds": getattr(self, 'time_increment_in_seconds', 1),
                    "duration": getattr(self, 'duration', None),
                    "exit_when_no_events": getattr(self, 'exit_when_no_events', False),
                    "wait_for_user_input_timeout": getattr(self, 'wait_for_user_input_timeout', None),
                },
                
                # Runtime state
                "verbose": getattr(self, 'verbose', False),
                "waiting_for_notification": getattr(self, 'waiting_for_notification', False),
                
                # World logs - needed for environment state restoration
                "world_logs": self._capture_world_logs(),
            }
            
            # Add time manager state if available
            if hasattr(self, 'time_manager'):
                state_dict["time_manager_state"] = self._capture_time_manager_state()
            
            return state_dict
        except Exception as e:
            logger.warning(f"Failed to capture environment state: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def _capture_event_log(self) -> Dict[str, Any]:
        """Capture event log state."""
        try:
            if hasattr(self, 'event_log') and self.event_log:
                if hasattr(self.event_log, 'to_dict'):
                    return self.event_log.to_dict()
                elif hasattr(self.event_log, 'list_view'):
                    return {
                        "past_events": [event.to_dict() for event in self.event_log.list_view()]
                    }
            return {"past_events": []}
        except Exception as e:
            logger.warning(f"Failed to capture event log: {e}")
            return {"error": str(e), "past_events": []}
    
    def _capture_world_logs(self) -> list:
        """
        Capture world_logs with proper log_type field.
        
        World logs contain BaseAgentLog subclass instances. We need to serialize
        them with their log_type to enable proper reconstruction during replay.
        """
        try:
            world_logs = getattr(self, 'world_logs', [])
            if not world_logs:
                return []
            
            serialized_logs = []
            for log in world_logs:
                try:
                    # Use the log's to_dict method if available
                    if hasattr(log, 'to_dict'):
                        log_dict = log.to_dict()
                    else:
                        # Fallback to dataclass asdict
                        from dataclasses import asdict, is_dataclass
                        if is_dataclass(log):
                            log_dict = asdict(log)
                        else:
                            log_dict = {"content": str(log)}
                    
                    # Ensure log_type is included
                    if 'log_type' not in log_dict and hasattr(log, 'get_type'):
                        log_dict['log_type'] = log.get_type()
                    
                    serialized_logs.append(log_dict)
                except Exception as e:
                    logger.warning(f"Failed to serialize world log: {e}")
                    continue
            
            return serialized_logs
        except Exception as e:
            logger.warning(f"Failed to capture world_logs: {e}")
            return []
    
    def _capture_event_queue(self) -> Dict[str, Any]:
        """Capture event queue state."""
        try:
            if hasattr(self, 'event_queue') and self.event_queue:
                if hasattr(self.event_queue, 'to_dict'):
                    return self.event_queue.to_dict()
                elif hasattr(self.event_queue, 'list_view'):
                    future_events = self.event_queue.list_view()
                    return {
                        "future_events": [event.to_dict() for event in future_events],
                        "already_scheduled": list(getattr(self.event_queue, 'already_scheduled', set()))
                    }
            return {"future_events": [], "already_scheduled": []}
        except Exception as e:
            logger.warning(f"Failed to capture event queue: {e}")
            return {"error": str(e), "future_events": [], "already_scheduled": []}
    
    def _capture_time_manager_state(self) -> Dict[str, Any]:
        """Capture time manager state."""
        try:
            tm = self.time_manager
            return {
                "current_time": tm.time() if hasattr(tm, 'time') else 0,
                "start_time": getattr(tm, 'start_time', 0),
            }
        except Exception as e:
            logger.warning(f"Failed to capture time manager state: {e}")
            return {"error": str(e)}
    
    def _capture_agent_state(self) -> Dict[str, Any]:
        """
        Capture agent state including logs (chat history).
        
        Uses agent reference provided by ScenarioRunner to capture the agent's
        logs which contain the full conversation history needed for replay.
        """
        if not self._get_agent_for_checkpoint:
            return {
                "error": "No agent reference available",
                "note": "Agent state capture requires ScenarioRunner to provide agent reference."
            }
        
        try:
            agent = self._get_agent_for_checkpoint()
            if agent:
                # Use the agent_state module to capture state
                return capture_agent_state(agent)
            else:
                return {
                    "note": "Agent not yet built at checkpoint time.",
                    "logs": [],
                    "logs_count": 0
                }
        except Exception as e:
            logger.warning(f"Failed to capture agent state: {e}")
            return {"error": str(e), "logs": [], "logs_count": 0}
    
    def _capture_apps_state(self) -> Dict[str, Any]:
        """Capture complete apps state for replay."""
        try:
            apps_state = {}
            
            if hasattr(self, 'apps') and self.apps:
                for app_name, app_instance in self.apps.items():
                    app_checkpoint = {
                        "app_type": type(app_instance).__name__,
                        "app_class_module": type(app_instance).__module__,
                        "timestamp": time.time(),
                        "app_name": app_name,
                    }
                    
                    # Capture app configuration
                    if hasattr(app_instance, 'name'):
                        app_checkpoint["name"] = app_instance.name
                    if hasattr(app_instance, 'seed'):
                        app_checkpoint["seed"] = app_instance.seed
                    if hasattr(app_instance, 'failure_probability'):
                        app_checkpoint["failure_probability"] = app_instance.failure_probability
                    
                    # Try to capture full app state via get_state()
                    if hasattr(app_instance, 'get_state'):
                        try:
                            state = app_instance.get_state()
                            if state is not None:
                                if isinstance(state, dict):
                                    app_checkpoint["state"] = state
                                    app_checkpoint["has_state"] = True
                                else:
                                    app_checkpoint["raw_state"] = str(state)
                                    app_checkpoint["has_state"] = True
                            else:
                                app_checkpoint["has_state"] = False
                        except Exception as e:
                            app_checkpoint["state_error"] = str(e)
                            app_checkpoint["has_state"] = False
                    else:
                        app_checkpoint["has_state"] = False
                    
                    # Check if app has load_state for replay capability
                    app_checkpoint["has_load_state"] = hasattr(app_instance, 'load_state')
                    
                    apps_state[app_name] = app_checkpoint
            
            # Also capture protocol connections if available
            if hasattr(self, 'protocol_to_app') and self.protocol_to_app:
                apps_state["_protocol_mappings"] = {
                    str(protocol): app.name if hasattr(app, 'name') else str(type(app).__name__)
                    for protocol, app in self.protocol_to_app.items()
                }
            
            return apps_state
        except Exception as e:
            logger.warning(f"Failed to capture apps state: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def _safe_get_queue_size(self) -> int:
        """Safely get event queue size."""
        try:
            if hasattr(self, 'event_queue'):
                queue = self.event_queue
                if hasattr(queue, 'list_view'):
                    return len(queue.list_view())
                elif hasattr(queue, '__len__'):
                    return len(queue)
            return 0
        except Exception:
            return 0
    
    def _safe_get_log_size(self) -> int:
        """Safely get event log size."""
        try:
            if hasattr(self, 'event_log'):
                log = self.event_log
                if hasattr(log, 'list_view'):
                    return len(log.list_view())
                elif hasattr(log, '__len__'):
                    return len(log)
            return 0
        except Exception:
            return 0


class CheckpointableScenarioRunner(ScenarioRunner):
    """
    ScenarioRunner subclass that integrates checkpoint functionality.
    """
    
    def __init__(self, checkpoint_config: Optional[CheckpointConfig] = None, 
                 checkpoint_manager: Optional[CheckpointManager] = None, **kwargs):
        """
        Initialize checkpoint-enabled scenario runner.
        
        Args:
            checkpoint_config: Checkpoint configuration
            checkpoint_manager: Optional CheckpointManager for checkpointing
            **kwargs: Additional arguments passed to parent ScenarioRunner
        """
        super().__init__(**kwargs)
        self.checkpoint_config = checkpoint_config
        self.checkpoint_manager = checkpoint_manager
    
    def enable_checkpointing(self, checkpoint_config: CheckpointConfig,
                           checkpoint_manager: Optional[CheckpointManager] = None):
        """Enable checkpointing for scenario execution."""
        self.checkpoint_config = checkpoint_config
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(checkpoint_config)
        logger.info("Checkpointing enabled for ScenarioRunner")
    
    def _set_scenario_config_for_checkpointing(self, config, scenario) -> None:
        """
        Pass scenario runner config to checkpoint manager for saving scenario_meta.json.
        
        Args:
            config: ScenarioRunnerConfig
            scenario: Scenario instance
        """
        if not self.checkpoint_manager:
            return
        
        # Extract scenario creation/initialization params from config if available
        scenario_creation_params = getattr(config, 'scenario_creation_params', None)
        scenario_initialization_params = getattr(config, 'scenario_initialization_params', None)
        
        # Also try to get params from scenario directly
        if scenario_creation_params is None:
            scenario_creation_params = getattr(scenario, 'creation_params', None)
        if scenario_initialization_params is None:
            scenario_initialization_params = getattr(scenario, 'initialization_params', None)
        
        # Detect scenario source type and path
        scenario_source_path = getattr(self, '_scenario_source_path', None)
        scenario_source_type = getattr(self, '_scenario_source_type', None)
        
        # Try to get source path from the scenario object itself
        if scenario_source_path is None and not isinstance(scenario, str):
            scenario_source_path = getattr(scenario, '_checkpoint_source_path', None)
        
        if scenario_source_type is None:
            # Auto-detect: check if this is a benchmark scenario (imported from JSON)
            if isinstance(scenario, str):
                # Scenario passed as JSON string — it's a benchmark/JSON scenario
                scenario_source_type = "json_file"
            elif scenario_source_path is not None:
                # Has a source file path — it was loaded from a JSON file
                scenario_source_type = "json_file"
            else:
                from are.simulation.scenarios.scenario_imported_from_json.benchmark_scenario import (
                    BenchmarkScenarioImportedFromJson,
                )
                if isinstance(scenario, BenchmarkScenarioImportedFromJson):
                    scenario_source_type = "json_file"
                else:
                    scenario_source_type = "registry"
        
        self.checkpoint_manager.set_scenario_config(
            runner_config=config,
            scenario_creation_params=scenario_creation_params,
            scenario_initialization_params=scenario_initialization_params,
            scenario_source_path=scenario_source_path,
            scenario_source_type=scenario_source_type,
        )
        logger.debug("Scenario config passed to checkpoint manager")
    
    def run(self, config, scenario, completed_events=None):
        """
        Override run to integrate checkpointing.
        """
        if not self.checkpoint_manager:
            # No checkpointing enabled, use original behavior
            return super().run(config, scenario, completed_events)
        
        # Get scenario ID for checkpointing
        scenario_id = getattr(scenario, 'scenario_id', 'unknown_scenario')
        
        # Pass scenario config to checkpoint manager for scenario_meta.json
        self._set_scenario_config_for_checkpointing(config, scenario)
        
        # Set up monkey-patching BEFORE calling super().run()
        original_env_class = None
        original_agent_builder_build = None
        original_checkpointable_init = CheckpointableEnvironment.__init__
        self._current_agent = None
        
        try:
            # Monkey-patch Environment class
            from are.simulation.environment import Environment as OriginalEnvironment
            original_env_class = OriginalEnvironment
            
            runner_self = self  # Capture for closures
            
            def checkpointable_init_with_auto_enable(self, *args, **kwargs):
                """Modified __init__ that automatically enables checkpointing."""
                # Call original init
                original_checkpointable_init(self, *args, **kwargs)
                
                # Auto-enable checkpointing after init
                if runner_self.checkpoint_manager:
                    scenario_id_val = getattr(scenario, 'scenario_id', 'unknown')
                    self.enable_checkpointing(
                        runner_self.checkpoint_manager, 
                        scenario_id_val
                    )
                    # Pass callback to get agent reference
                    self._get_agent_for_checkpoint = lambda: runner_self._current_agent
                    logger.info(f"CheckpointableEnvironment auto-enabled for scenario {scenario_id_val}")
            
            # Temporarily patch CheckpointableEnvironment's init
            CheckpointableEnvironment.__init__ = checkpointable_init_with_auto_enable
            
            # Monkey-patch agent builder to capture agent reference
            original_agent_builder_build = self.agent_builder.build
            
            def build_with_capture(*args, **kwargs):
                """Wrapper that captures agent reference for checkpoint state capture."""
                logger.info(f"build_with_capture called - capturing agent")
                agent = original_agent_builder_build(*args, **kwargs)
                runner_self._current_agent = agent
                logger.info(f"Captured agent reference: {type(agent).__name__}")
                
                # Get the environment from kwargs (if passed) or from agent
                env = kwargs.get('env', None)
                if env is None and hasattr(agent, 'env'):
                    env = agent.env
                
                # Set up agent checkpoint callback for proper checkpoint timing
                if env is not None and isinstance(env, CheckpointableEnvironment):
                    env.setup_agent_checkpoint_callback(agent)
                    logger.info("Agent checkpoint callback configured for proper checkpoint timing")
                
                return agent
            
            # Apply monkey patches
            # We need to patch BOTH the module attribute AND the scenario_runner's local import
            import are.simulation.environment
            import are.simulation.scenario_runner
            
            are.simulation.environment.Environment = CheckpointableEnvironment
            are.simulation.scenario_runner.Environment = CheckpointableEnvironment
            self.agent_builder.build = build_with_capture
            
            logger.info(f"Checkpointing enabled for scenario {scenario_id}")
            
            # Call parent's run() which will now use CheckpointableEnvironment
            result = super().run(config, scenario, completed_events)
            
        except Exception as e:
            logger.exception(f"Scenario execution failed: {e}")
            raise
        finally:
            # Restore original classes and methods
            if original_env_class:
                import are.simulation.environment
                import are.simulation.scenario_runner
                are.simulation.environment.Environment = original_env_class
                are.simulation.scenario_runner.Environment = original_env_class
            if original_agent_builder_build:
                self.agent_builder.build = original_agent_builder_build
            # Restore original CheckpointableEnvironment init
            CheckpointableEnvironment.__init__ = original_checkpointable_init
            self._current_agent = None
        
        return result



def create_checkpointable_runner(checkpoint_config: CheckpointConfig,
                                checkpoint_manager: Optional[CheckpointManager] = None,
                                **kwargs):
    """
    Factory function to create a checkpoint-enabled ScenarioRunner.
    """
    manager = checkpoint_manager or CheckpointManager(checkpoint_config)
    runner = CheckpointableScenarioRunner(
        checkpoint_config=checkpoint_config,
        checkpoint_manager=manager,
        **kwargs
    )
    return runner


# Convenience functions for integration
def create_checkpointable_environment(checkpoint_manager: CheckpointManager, 
                                     scenario_id: str,
                                     config: Optional[EnvironmentConfig] = None,
                                     **kwargs):
    """Create a checkpoint-enabled environment."""
    env = CheckpointableEnvironment(config=config, checkpoint_manager=checkpoint_manager, **kwargs)
    env.enable_checkpointing(checkpoint_manager, scenario_id)
    return env


def create_checkpointable_runner_from_config(checkpoint_config: CheckpointConfig, **kwargs):
    """Create a checkpoint-enabled scenario runner from configuration."""
    return create_checkpointable_runner(checkpoint_config, **kwargs)
