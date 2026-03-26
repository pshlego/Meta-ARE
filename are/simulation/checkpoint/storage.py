"""
JSON-based checkpoint storage system.

This module provides the storage backend for saving and loading checkpoint data
as JSON files, following the minimal invasive design principle.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from dataclasses import is_dataclass, asdict

from .metadata import CheckpointMetadata


class CheckpointJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles special types in checkpoint data.
    
    Handles:
    - Dataclasses (ActionLog, BaseAgentLog, etc.)
    - Path objects
    - Other non-serializable types with fallback to string representation
    """
    
    def default(self, o: Any) -> Any:
        # Handle dataclasses (ActionLog, BaseAgentLog, etc.)
        if is_dataclass(o) and not isinstance(o, type):
            try:
                return asdict(o)  # type: ignore
            except Exception:
                # Fallback: convert to dict manually
                return {
                    '__type__': type(o).__name__,
                    '__module__': type(o).__module__,
                    'data': {k: self._safe_value(v) for k, v in o.__dict__.items()}
                }
        
        # Handle Path objects
        if isinstance(o, Path):
            return str(o)
        
        # Handle sets
        if isinstance(o, set):
            return list(o)
        
        # Handle bytes
        if isinstance(o, bytes):
            try:
                return o.decode('utf-8')
            except Exception:
                return o.hex()
        
        # Fallback for other types
        try:
            return super().default(o)
        except TypeError:
            # Last resort: convert to string
            return {
                '__type__': type(o).__name__,
                '__repr__': repr(o),
                '__str__': str(o)
            }
    
    def _safe_value(self, value: Any) -> Any:
        """Safely convert a value to JSON-serializable form."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._safe_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._safe_value(v) for k, v in value.items()}
        elif is_dataclass(value) and not isinstance(value, type):
            try:
                return asdict(value)  # type: ignore
            except Exception:
                return str(value)
        elif isinstance(value, Path):
            return str(value)
        elif isinstance(value, set):
            return list(value)
        else:
            return str(value)


class CheckpointStorage:
    """
    JSON-based storage system for checkpoint data.
    
    Handles the actual file operations for saving and loading checkpoint data,
    maintaining a clear directory structure for easy navigation and debugging.
    """
    
    def __init__(self, base_checkpoint_dir: Path, verbose: bool = False):
        """
        Initialize storage system with base directory.
        
        Args:
            base_checkpoint_dir: Root directory for all checkpoint storage
            verbose: Enable detailed logging for storage operations
        """
        self.base_dir = Path(base_checkpoint_dir)
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_scenario_dir(self, scenario_id: str) -> Path:
        """Get the directory path for a specific scenario's checkpoints."""
        return self.base_dir / scenario_id
    
    def get_checkpoint_dir(self, scenario_id: str, checkpoint_id: str) -> Path:
        """Get the directory path for a specific checkpoint."""
        return self.base_dir / scenario_id / checkpoint_id
    
    def save_checkpoint(self, scenario_id: str, checkpoint_id: str, 
                       checkpoint_data: Dict[str, Any]) -> bool:
        """
        Save a complete checkpoint to disk.
        
        Args:
            scenario_id: ID of the scenario this checkpoint belongs to
            checkpoint_id: Unique identifier for this checkpoint
            checkpoint_data: Complete checkpoint data dictionary
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            checkpoint_dir = self.get_checkpoint_dir(scenario_id, checkpoint_id)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            if self.verbose:
                self.logger.info(f"Saving checkpoint {checkpoint_id} for scenario {scenario_id}")
            
            # Save each component of the checkpoint as separate JSON files
            self._save_json_file(checkpoint_dir / "metadata.json", checkpoint_data.get("metadata", {}))
            
            if "trigger_event" in checkpoint_data:
                self._save_json_file(checkpoint_dir / "trigger_event.json", checkpoint_data["trigger_event"])
            
            if "completed_event" in checkpoint_data:
                self._save_json_file(checkpoint_dir / "completed_event.json", checkpoint_data["completed_event"])
            
            if "environment" in checkpoint_data:
                self._save_json_file(checkpoint_dir / "environment_snapshot.json", checkpoint_data["environment"])
            
            if "agent" in checkpoint_data:
                self._save_json_file(checkpoint_dir / "agent_snapshot.json", checkpoint_data["agent"])
            
            if "apps" in checkpoint_data:
                apps_dir = checkpoint_dir / "apps_snapshot"
                apps_dir.mkdir(exist_ok=True)
                for app_name, app_state in checkpoint_data["apps"].items():
                    self._save_json_file(apps_dir / f"{app_name}_state.json", app_state)
            
            if self.verbose:
                self.logger.info(f"Successfully saved checkpoint {checkpoint_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
            return False
    
    def load_checkpoint(self, scenario_id: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a complete checkpoint from disk.
        
        Args:
            scenario_id: ID of the scenario
            checkpoint_id: ID of the checkpoint to load
            
        Returns:
            Complete checkpoint data dictionary, or None if not found
        """
        try:
            checkpoint_dir = self.get_checkpoint_dir(scenario_id, checkpoint_id)
            
            if not checkpoint_dir.exists():
                if self.verbose:
                    self.logger.warning(f"Checkpoint {checkpoint_id} not found for scenario {scenario_id}")
                return None
            
            if self.verbose:
                self.logger.info(f"Loading checkpoint {checkpoint_id} for scenario {scenario_id}")
            
            checkpoint_data = {}
            
            # Load each component
            metadata_file = checkpoint_dir / "metadata.json"
            if metadata_file.exists():
                checkpoint_data["metadata"] = self._load_json_file(metadata_file)
            
            trigger_event_file = checkpoint_dir / "trigger_event.json"
            if trigger_event_file.exists():
                checkpoint_data["trigger_event"] = self._load_json_file(trigger_event_file)
            
            completed_event_file = checkpoint_dir / "completed_event.json"
            if completed_event_file.exists():
                checkpoint_data["completed_event"] = self._load_json_file(completed_event_file)
            
            env_snapshot_file = checkpoint_dir / "environment_snapshot.json"
            if env_snapshot_file.exists():
                checkpoint_data["environment"] = self._load_json_file(env_snapshot_file)
            
            agent_snapshot_file = checkpoint_dir / "agent_snapshot.json"
            if agent_snapshot_file.exists():
                checkpoint_data["agent"] = self._load_json_file(agent_snapshot_file)
            
            # Load app states
            apps_dir = checkpoint_dir / "apps_snapshot"
            if apps_dir.exists():
                checkpoint_data["apps"] = {}
                for app_file in apps_dir.glob("*_state.json"):
                    app_name = app_file.stem.replace("_state", "")
                    checkpoint_data["apps"][app_name] = self._load_json_file(app_file)
            
            if self.verbose:
                self.logger.info(f"Successfully loaded checkpoint {checkpoint_id}")
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    def list_scenario_checkpoints(self, scenario_id: str) -> List[str]:
        """List all checkpoint IDs for a scenario."""
        scenario_dir = self.get_scenario_dir(scenario_id)
        if not scenario_dir.exists():
            return []
        
        checkpoints = []
        for item in scenario_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Verify it's a valid checkpoint by checking for metadata
                if (item / "metadata.json").exists():
                    checkpoints.append(item.name)
        
        return sorted(checkpoints)
    
    def delete_checkpoint(self, scenario_id: str, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint."""
        try:
            checkpoint_dir = self.get_checkpoint_dir(scenario_id, checkpoint_id)
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
                

                
                if self.verbose:
                    self.logger.info(f"Deleted checkpoint {checkpoint_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    def delete_scenario_checkpoints(self, scenario_id: str) -> bool:
        """Delete all checkpoints for a scenario."""
        try:
            scenario_dir = self.get_scenario_dir(scenario_id)
            if scenario_dir.exists():
                shutil.rmtree(scenario_dir)
                if self.verbose:
                    self.logger.info(f"Deleted all checkpoints for scenario {scenario_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete scenario checkpoints: {e}")
            return False
    

    
    def _save_json_file(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Save data to a JSON file with proper formatting and custom encoding."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=CheckpointJSONEncoder)
    
    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load data from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
