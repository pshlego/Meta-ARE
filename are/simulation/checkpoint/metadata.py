"""
Checkpoint metadata structures for event-based checkpointing.

This module defines the metadata classes that capture checkpoint context
and timing information, following the minimal invasive design principle.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import time
from datetime import datetime


@dataclass
class CheckpointMetadata:
    """
    Metadata for a single checkpoint, capturing event context and timing.
    
    This class stores essential metadata about when and why a checkpoint
    was created. Detailed system state is captured separately in the checkpoint data.
    """
    
    # Basic checkpoint identification
    checkpoint_id: str
    scenario_id: str
    
    # Event timing information
    simulation_time: float          # Simulation time when checkpoint was created
    wall_clock_time: float         # Real-world timestamp
    
    # Event context
    checkpoint_type: str           # "before_env", "after_agent", "scenario_init", etc.
    trigger_event_id: Optional[str] = None    # ID of event that triggered this checkpoint
    trigger_event_type: Optional[str] = None  # Type of triggering event (ENV/AGENT/VALIDATION)
    
    # Checkpoint sequence information
    event_sequence_number: int = 0  # Sequential event number in scenario
    checkpoint_sequence: int = 0    # Sequential checkpoint number
    
    # System state summary
    agent_turn: int = 0                       # Current agent turn number
    events_completed_count: int = 0           # Total events processed so far
    events_pending_count: int = 0             # Events remaining in queue
    
    def __post_init__(self):
        """Set wall clock time if not provided."""
        if self.wall_clock_time == 0:
            self.wall_clock_time = time.time()
    
    @property
    def created_at(self) -> str:
        """Human-readable timestamp when checkpoint was created."""
        return datetime.fromtimestamp(self.wall_clock_time).strftime("%Y-%m-%d %H:%M:%S")
    
    @property
    def is_event_boundary_checkpoint(self) -> bool:
        """Check if this checkpoint was triggered by an event boundary."""
        return self.trigger_event_id is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            'checkpoint_id': self.checkpoint_id,
            'scenario_id': self.scenario_id,
            'simulation_time': self.simulation_time,
            'wall_clock_time': self.wall_clock_time,
            'created_at': self.created_at,
            'checkpoint_type': self.checkpoint_type,
            'trigger_event_id': self.trigger_event_id,
            'trigger_event_type': self.trigger_event_type,
            'event_sequence_number': self.event_sequence_number,
            'checkpoint_sequence': self.checkpoint_sequence,
            'agent_turn': self.agent_turn,
            'events_completed_count': self.events_completed_count,
            'events_pending_count': self.events_pending_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create metadata from dictionary (excluding computed properties)."""
        # Remove computed properties that shouldn't be in constructor
        data_copy = data.copy()
        data_copy.pop('created_at', None)  # This is a computed property
        
        return cls(**data_copy)
    
    @classmethod
    def create_event_boundary(
        cls, 
        checkpoint_id: str,
        scenario_id: str,
        simulation_time: float,
        checkpoint_type: str,
        trigger_event_id: str,
        trigger_event_type: str,
        event_sequence_number: int,
        checkpoint_sequence: int
    ) -> 'CheckpointMetadata':
        """Create metadata for an event boundary checkpoint."""
        return cls(
            checkpoint_id=checkpoint_id,
            scenario_id=scenario_id,
            simulation_time=simulation_time,
            wall_clock_time=time.time(),
            checkpoint_type=checkpoint_type,
            trigger_event_id=trigger_event_id,
            trigger_event_type=trigger_event_type,
            event_sequence_number=event_sequence_number,
            checkpoint_sequence=checkpoint_sequence
        )


