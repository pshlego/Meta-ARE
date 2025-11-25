"""
Central checkpoint management system.

This module provides the main CheckpointManager class that coordinates
checkpoint operations, bringing together configuration, storage, and metadata.

Note: Simplified to focus on creating checkpoints after agent tool calls.
The classification logic is now handled directly in wrappers.py.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

from .config import CheckpointConfig
from .storage import CheckpointStorage, CheckpointJSONEncoder
from .metadata import CheckpointMetadata


class CheckpointManager:
    """
    Central manager for checkpoint operations.
    
    This class coordinates the creation, storage, and retrieval of checkpoints,
    serving as the main interface between the ARE system and the checkpoint subsystem.
    """
    
    def __init__(self, config: CheckpointConfig):
        """
        Initialize the checkpoint manager with configuration.
        
        Args:
            config: Checkpoint system configuration
        """
        self.config = config
        self.storage = CheckpointStorage(config.checkpoint_dir, config.verbose_logging)
        self.logger = logging.getLogger(__name__)
        
        # Event tracking counters
        self.checkpoint_sequence_counter = 0
        self.event_sequence_counter = 0
        
        # Scenario metadata tracking
        self._scenario_meta_saved = False
        self._scenario_runner_config: Optional[Any] = None
        self._scenario_creation_params: Optional[Dict[str, Any]] = None
        self._scenario_initialization_params: Optional[Dict[str, Any]] = None
    
    def set_scenario_config(
        self, 
        runner_config: Any,  # ScenarioRunnerConfig
        scenario_creation_params: Optional[Dict[str, Any]] = None,
        scenario_initialization_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set scenario runner configuration for saving to scenario_meta.json.
        
        This should be called before checkpoints are created so that the
        scenario metadata can be saved with the first checkpoint.
        
        Args:
            runner_config: ScenarioRunnerConfig (Pydantic model)
            scenario_creation_params: Parameters used to create the scenario
            scenario_initialization_params: Parameters used to initialize the scenario
        """
        self._scenario_runner_config = runner_config
        self._scenario_creation_params = scenario_creation_params
        self._scenario_initialization_params = scenario_initialization_params
        self.logger.debug("Scenario config set for checkpointing")
    
    def _save_scenario_meta_if_needed(self, scenario_id: str) -> None:
        """
        Save scenario_meta.json if it hasn't been saved yet.
        
        Called automatically when creating the first checkpoint.
        """
        if self._scenario_meta_saved:
            return
        
        try:
            scenario_meta = {
                "scenario_id": scenario_id,
                "checkpoint_dir": str(self.config.checkpoint_dir),
                "created_at": time.time(),
            }
            
            # Add runner config if available
            if self._scenario_runner_config is not None:
                # Try to serialize config using model_dump() (Pydantic v2) or dict() (v1)
                if hasattr(self._scenario_runner_config, 'model_dump'):
                    config_dict = self._scenario_runner_config.model_dump(
                        exclude_none=False, 
                        mode='json'
                    )
                elif hasattr(self._scenario_runner_config, 'dict'):
                    config_dict = self._scenario_runner_config.dict()
                else:
                    config_dict = dict(self._scenario_runner_config)
                scenario_meta["runner_config"] = config_dict
            
            # Add scenario creation/initialization params
            if self._scenario_creation_params:
                scenario_meta["scenario_creation_params"] = self._scenario_creation_params
            
            if self._scenario_initialization_params:
                scenario_meta["scenario_initialization_params"] = self._scenario_initialization_params
            
            # Save to scenario_meta.json in checkpoint_dir
            meta_file = self.config.checkpoint_dir / "scenario_meta.json"
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(scenario_meta, f, indent=2, ensure_ascii=False, cls=CheckpointJSONEncoder)
            
            self._scenario_meta_saved = True
            self.logger.info(f"Saved scenario metadata to {meta_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save scenario metadata: {e}", exc_info=True)
    
    def create_event_checkpoint(self, scenario_id: str, checkpoint_type: str, 
                              simulation_time: float, trigger_event_data: Dict[str, Any],
                              system_state: Dict[str, Any]) -> Optional[str]:
        """
        Create a checkpoint triggered by an event.
        
        Args:
            scenario_id: Identifier for the scenario
            checkpoint_type: Type of checkpoint (e.g., "after_tool_call")
            simulation_time: Current simulation time
            trigger_event_data: Data about the triggering event
            system_state: Complete system state to checkpoint
            
        Returns:
            Checkpoint ID if successful, None otherwise
        """
        if not self.config.enabled:
            return None
        
        # Save scenario_meta.json on first checkpoint
        self._save_scenario_meta_if_needed(scenario_id)
        
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
            "apps": system_state.get("apps", {})
        }
        
        # Add completed event data if present
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
                self.logger.info(f"Created checkpoint: {checkpoint_id}")
            return checkpoint_id
        else:
            self.logger.error(f"Failed to create checkpoint: {checkpoint_id}")
            return None
    
    def get_checkpoint(self, scenario_id: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific checkpoint."""
        return self.storage.load_checkpoint(scenario_id, checkpoint_id)
    
    def list_scenario_checkpoints(self, scenario_id: str) -> List[str]:
        """List all checkpoints for a scenario."""
        return self.storage.list_scenario_checkpoints(scenario_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get checkpoint manager statistics."""
        return {
            "checkpoint_sequence": self.checkpoint_sequence_counter,
            "event_sequence": self.event_sequence_counter,
            "storage_base_dir": str(self.config.checkpoint_dir),
            "config_enabled": self.config.enabled,
            "checkpoint_events": list(self.config.checkpoint_on_events) if self.config.checkpoint_on_events else []
        }
