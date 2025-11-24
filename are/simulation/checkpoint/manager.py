"""
Central checkpoint management system.

This module provides the main CheckpointManager class that coordinates
event-based checkpoint operations, bringing together configuration, storage,
classification, and metadata management.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

from .config import CheckpointConfig
from .storage import CheckpointStorage
from .metadata import CheckpointMetadata
from .classifier import EventCheckpointClassifier


class CheckpointManager:
    """
    Central manager for event-based checkpoint operations.
    
    This class coordinates the creation, storage, and retrieval of event-based
    checkpoints, serving as the main interface between the ARE system and the
    checkpoint subsystem.
    """
    
    def __init__(self, config: CheckpointConfig):
        """
        Initialize the checkpoint manager with configuration.
        
        Args:
            config: Checkpoint system configuration
        """
        self.config = config
        self.storage = CheckpointStorage(config.checkpoint_dir, config.verbose_logging)
        self.classifier = EventCheckpointClassifier(
            checkpoint_events=config.checkpoint_on_events,
            verbose=config.verbose_logging
        )
        self.logger = logging.getLogger(__name__)
        
        # Event tracking counters
        self.checkpoint_sequence_counter = 0
        self.event_sequence_counter = 0
    
    def create_event_checkpoint(self, scenario_id: str, checkpoint_type: str, 
                              simulation_time: float, trigger_event_data: Dict[str, Any],
                              system_state: Dict[str, Any]) -> Optional[str]:
        """
        Create a checkpoint triggered by an event boundary.
        
        Args:
            scenario_id: Identifier for the scenario
            checkpoint_type: Type of checkpoint (before_env, after_agent, etc.)
            simulation_time: Current simulation time
            trigger_event_data: Data about the triggering event
            system_state: Complete system state to checkpoint
            
        Returns:
            Checkpoint ID if successful, None otherwise
        """
        if not self.config.enabled:
            return None
        
        # Generate checkpoint ID
        self.checkpoint_sequence_counter += 1
        self.event_sequence_counter += 1
        checkpoint_id = f"event_{self.event_sequence_counter}_{checkpoint_type}"
        
        # Create metadata
        metadata = CheckpointMetadata.create_event_boundary(
            checkpoint_id=checkpoint_id,
            scenario_id=scenario_id,
            simulation_time=simulation_time,
            checkpoint_type=checkpoint_type,
            trigger_event_id=trigger_event_data.get("event_id", "unknown"),
            trigger_event_type=trigger_event_data.get("event_type", "unknown"),
            event_sequence_number=self.event_sequence_counter,
            checkpoint_sequence=self.checkpoint_sequence_counter
        )
        
        # Build complete checkpoint data
        checkpoint_data = {
            "metadata": metadata.to_dict(),
            "trigger_event": trigger_event_data,
            "environment": system_state.get("environment", {}),
            "agent": system_state.get("agent", {}),
            "apps": system_state.get("apps", {}),
            "execution_context": system_state.get("execution_context", {})
        }
        
        # Add completed event data if this is an after-event checkpoint
        if "completed_event" in system_state:
            checkpoint_data["completed_event"] = system_state["completed_event"]
        
        # Save checkpoint
        success = self.storage.save_checkpoint(
            scenario_id, 
            checkpoint_id, 
            checkpoint_data
        )
        
        if success:
            if self.config.verbose_logging:
                self.logger.info(f"Created event checkpoint: {checkpoint_id}")
            return checkpoint_id
        else:
            self.logger.error(f"Failed to create checkpoint: {checkpoint_id}")
            return None
    
    def should_create_checkpoint(self, event_type: str, execution_phase: str,
                               event_id: Optional[str] = None,
                               execution_success: bool = True,
                               state_changed: bool = True) -> bool:
        """
        Check if a checkpoint should be created for an event.
        
        Args:
            event_type: Type of event (ENV, AGENT, VALIDATION, etc.)
            execution_phase: Whether this is before or after execution
            event_id: Optional specific event identifier
            execution_success: Whether execution was successful
            state_changed: Whether execution changed system state
            
        Returns:
            True if a checkpoint should be created
        """
        if not self.config.enabled:
            return False
        
        # Use classifier to make the decision
        analysis = self.classifier.analyze_event_for_checkpoint(
            event_type=event_type,
            event_id=event_id,
            execution_phase=execution_phase,
            execution_success=execution_success,
            state_changed=state_changed,
            is_first_event=(self.event_sequence_counter == 0 and execution_phase == "before")
        )
        
        return analysis["should_checkpoint"]
    
    def handle_event_start(self, scenario_id: str, event_type: str, event_id: str, 
                          event_data: Dict[str, Any], system_state: Dict[str, Any]) -> Optional[str]:
        """
        Handle the start of an event execution (before-event checkpoint).
        
        Args:
            scenario_id: Identifier for the scenario
            event_type: Type of event starting
            event_id: Unique identifier for the event
            event_data: Data about the event
            system_state: Current system state
            
        Returns:
            Checkpoint ID if created, None otherwise
        """
        # Increment event counter for new events
        if self.config.checkpoint_on_events and event_type in self.config.checkpoint_on_events:
            self.event_sequence_counter += 1
        
        # Check if we should create a before-event checkpoint
        if not self.should_create_checkpoint(event_type, "before", event_id):
            return None
        
        # Prepare trigger event data
        trigger_event_data = {
            "event_id": event_id,
            "event_type": event_type,
            "event_time": system_state.get("simulation_time", 0.0),
            **event_data
        }
        
        # Create before-event checkpoint
        checkpoint_type = f"before_{event_type.lower()}"
        return self.create_event_checkpoint(
            scenario_id=scenario_id,
            checkpoint_type=checkpoint_type,
            simulation_time=system_state.get("simulation_time", 0.0),
            trigger_event_data=trigger_event_data,
            system_state=system_state
        )
    
    def handle_event_completion(self, scenario_id: str, event_type: str, event_id: str,
                              completed_event_data: Dict[str, Any],
                              system_state: Dict[str, Any],
                              execution_success: bool = True) -> Optional[str]:
        """
        Handle the completion of an event execution (after-event checkpoint).
        
        Args:
            scenario_id: Identifier for the scenario
            event_type: Type of event that completed
            event_id: Unique identifier for the event
            completed_event_data: Data about the completed event
            system_state: System state after execution
            execution_success: Whether execution was successful
            
        Returns:
            Checkpoint ID if created, None otherwise
        """
        # Check if we should create an after-event checkpoint
        if not self.should_create_checkpoint(event_type, "after", event_id, execution_success):
            return None
        
        # Prepare trigger event data
        trigger_event_data = {
            "event_id": event_id,
            "event_type": event_type,
            "event_time": system_state.get("simulation_time", 0.0),
        }
        
        # Add completed event data to system state
        system_state_with_completion = system_state.copy()
        system_state_with_completion["completed_event"] = completed_event_data
        
        # Create after-event checkpoint
        checkpoint_type = f"after_{event_type.lower()}"
        return self.create_event_checkpoint(
            scenario_id=scenario_id,
            checkpoint_type=checkpoint_type,
            simulation_time=system_state.get("simulation_time", 0.0),
            trigger_event_data=trigger_event_data,
            system_state=system_state_with_completion
        )
    
    def get_checkpoint(self, scenario_id: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific checkpoint."""
        return self.storage.load_checkpoint(scenario_id, checkpoint_id)
    
    def list_scenario_checkpoints(self, scenario_id: str) -> List[str]:
        """List all checkpoints for a scenario."""
        return self.storage.list_scenario_checkpoints(scenario_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get checkpoint manager statistics."""
        base_stats = self.classifier.get_statistics()
        
        return {
            **base_stats,
            "checkpoint_sequence": self.checkpoint_sequence_counter,
            "event_sequence": self.event_sequence_counter,
            "storage_base_dir": str(self.config.checkpoint_dir),
            "config_enabled": self.config.enabled
        }