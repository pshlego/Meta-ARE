"""
Checkpoint-enabled wrapper classes for key ARE components.

This module provides wrapper classes that add checkpoint functionality
to the existing ARE components while maintaining minimal invasive design
by inheriting and overriding specific methods.
"""

from typing import Dict, Any, Optional, List, Union, Callable
import logging
import time
from pathlib import Path

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
    Environment subclass that adds event-based checkpoint functionality.
    
    This class inherits from Environment and overrides process_event to create
    before/after checkpoints at critical event boundaries during scenario execution.
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
        logger.info(f"Checkpointing enabled for scenario {scenario_id}")
    
    def process_event(self, event):
        """
        Override process_event to add checkpoint creation at event boundaries.
        """
        logger.debug(f"process_event called - checkpoint_enabled={self._checkpoint_enabled}, has_manager={self.checkpoint_manager is not None}")
        
        if not self._checkpoint_enabled or not self.checkpoint_manager:
            logger.debug("Checkpointing not enabled or no manager, calling super().process_event()")
            return super().process_event(event)
        
        # Determine if this event should trigger checkpoints
        event_type = self._get_event_type_str(event)
        should_checkpoint = self._should_checkpoint_event(event_type, event)
        
        logger.info(f"Processing event type={event_type}, should_checkpoint={should_checkpoint}")
        
        checkpoint_before_id = None
        if should_checkpoint:
            # Create checkpoint BEFORE event execution
            try:
                logger.info(f"Creating before-event checkpoint for event_type={event_type}")
                checkpoint_before_id = self._create_checkpoint(
                    checkpoint_type="before",
                    event_info={
                        "event_type": event_type,
                        "event_id": getattr(event, 'event_id', str(id(event))),
                        "event_time": getattr(event, 'event_time', self.current_time),
                    }
                )
                logger.info(f"Created before-event checkpoint: {checkpoint_before_id}")
            except Exception as e:
                logger.warning(f"Failed to create before-event checkpoint: {e}", exc_info=True)
        
        # Execute the original event processing
        execution_success = False
        execution_error = None
        successors = []
        
        try:
            successors = super().process_event(event)
            execution_success = True
        except Exception as e:
            execution_success = False
            execution_error = str(e)
            raise
        finally:
            if should_checkpoint:
                # Create checkpoint AFTER event execution
                try:
                    logger.info(f"Creating after-event checkpoint for event_type={event_type}")
                    self._create_checkpoint(
                        checkpoint_type="after",
                        event_info={
                            "event_type": event_type,
                            "event_id": getattr(event, 'event_id', str(id(event))),
                            "execution_success": execution_success,
                            "execution_error": execution_error,
                            "successor_count": len(successors) if successors else 0
                        },
                        related_checkpoint_id=checkpoint_before_id
                    )
                    logger.info(f"Created after-event checkpoint successfully")
                except Exception as e:
                    logger.warning(f"Failed to create after-event checkpoint: {e}", exc_info=True)
        
        return successors
    
    def _should_checkpoint_event(self, event_type: str, event) -> bool:
        """Determine if an event should trigger checkpoint creation."""
        if not self.checkpoint_manager:
            return False
        
        # Check if this event type is configured for checkpointing
        config = self.checkpoint_manager.config
        if hasattr(config, 'checkpoint_on_events') and config.checkpoint_on_events:
            return event_type in config.checkpoint_on_events
        
        # Default: checkpoint critical events
        return event_type in ['ENV', 'AGENT', 'VALIDATION', 'STOP']
    
    def _get_event_type_str(self, event) -> str:
        """Extract event type as string."""
        if hasattr(event, 'event_type'):
            event_type = event.event_type
            # Handle EventType enum: extract just the name part
            if hasattr(event_type, 'name'):
                return event_type.name  # Returns 'ENV', 'AGENT', etc.
            else:
                # Convert to string and extract the enum value name
                event_type_str = str(event_type)
                if '.' in event_type_str:
                    return event_type_str.split('.')[-1]  # "EventType.ENV" -> "ENV"
                return event_type_str
        
        # Fallback: classify by class name
        class_name = event.__class__.__name__
        if 'Stop' in class_name:
            return 'STOP'
        elif 'Validation' in class_name:
            return 'VALIDATION'
        elif 'Agent' in class_name:
            return 'AGENT'
        else:
            return 'ENV'
    
    def _create_checkpoint(self, checkpoint_type: str, event_info: Dict[str, Any], 
                          related_checkpoint_id: Optional[str] = None) -> Optional[str]:
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
                }
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
                "events_to_schedule": self._capture_events_to_schedule(),
                
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
                "world_logs": getattr(self, 'world_logs', []),
                
                # Additional state for replay
                "graphql_cache": getattr(self, 'graphql_cache', {}),
                "dump_dir": str(getattr(self, 'dump_dir', None)) if getattr(self, 'dump_dir', None) else None,
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
    
    def _capture_events_to_schedule(self) -> list:
        """Capture events_to_schedule list."""
        try:
            if hasattr(self, 'events_to_schedule') and self.events_to_schedule:
                return [event.to_dict() if hasattr(event, 'to_dict') else str(event) 
                        for event in self.events_to_schedule]
            return []
        except Exception as e:
            logger.warning(f"Failed to capture events_to_schedule: {e}")
            return []
    
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
        Capture complete agent state including logs and history.
        
        Uses agent reference provided by ScenarioRunner via _get_agent_for_checkpoint
        callback to capture comprehensive agent state including logs, iterations,
        and internal state.
        """
        try:
            # Capture basic agent-related environment state
            agent_state = {
                "timestamp": time.time(),
                "status": "captured",
                "capture_method": "agent_reference",
            }
            
            # Try to capture agent logs if add_event_to_agent_log callback exists
            if hasattr(self, 'add_event_to_agent_log') and self.add_event_to_agent_log:
                agent_state["has_agent_log_callback"] = True
            else:
                agent_state["has_agent_log_callback"] = False
            
            # Capture agent action validators state
            if hasattr(self, 'agent_action_validators'):
                agent_state["agent_action_validators_count"] = len(self.agent_action_validators)
                agent_state["agent_action_validators"] = [
                    {
                        "validator_type": type(validator).__name__,
                        "validator_class": f"{type(validator).__module__}.{type(validator).__name__}"
                    }
                    for validator in self.agent_action_validators
                ]
            
            # Capture full agent state if agent reference is available
            if self._get_agent_for_checkpoint:
                try:
                    agent = self._get_agent_for_checkpoint()
                    if agent:
                        # Use the agent_state module to capture comprehensive agent state
                        agent_state.update(capture_agent_state(agent))
                        agent_state["full_agent_state_captured"] = True
                        logger.debug(f"Captured full agent state with {agent_state.get('logs_count', 0)} logs")
                    else:
                        agent_state["full_agent_state_captured"] = False
                        agent_state["note"] = "Agent reference returned None"
                except Exception as e:
                    logger.warning(f"Failed to capture agent state via reference: {e}")
                    agent_state["full_agent_state_captured"] = False
                    agent_state["capture_error"] = str(e)
            else:
                agent_state["full_agent_state_captured"] = False
                agent_state["note"] = (
                    "No agent reference available. "
                    "Agent state capture requires ScenarioRunner to provide agent reference."
            )
            
            return agent_state
        except Exception as e:
            logger.warning(f"Failed to capture agent state: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
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
    
    def run(self, config, scenario, completed_events=None):
        """
        Override run to integrate checkpointing.
        """
        if not self.checkpoint_manager:
            # No checkpointing enabled, use original behavior
            return super().run(config, scenario, completed_events)
        
        # Get scenario ID for checkpointing
        scenario_id = getattr(scenario, 'scenario_id', 'unknown_scenario')
        
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
                """Wrapper that captures agent reference after building."""
                agent = original_agent_builder_build(*args, **kwargs)
                runner_self._current_agent = agent
                logger.debug(f"Captured agent reference: {type(agent).__name__}")
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
