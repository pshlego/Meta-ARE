"""
Checkpoint system configuration.

This module defines the configuration class for the event-based checkpoint system,
following the minimal invasive design principle by keeping all checkpoint
configuration separate from existing ARE components.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set


@dataclass
class CheckpointConfig:
    """
    Configuration for the event-based checkpoint system with JSON storage.
    
    This class encapsulates core checkpoint settings to avoid
    contaminating existing ARE configuration classes.
    """
    
    # Basic checkpoint settings
    enabled: bool = False
    checkpoint_dir: Path = Path("./checkpoints")
    
    # Event-based checkpoint configuration
    checkpoint_on_events: Optional[Set[str]] = None  # {"ENV", "AGENT", "VALIDATION"}
    checkpoint_before_events: bool = True
    checkpoint_after_events: bool = True
    
    # Debug settings
    verbose_logging: bool = False
    
    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Ensure checkpoint directory is a Path object
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        
        # Set default events to checkpoint if none specified
        if self.checkpoint_on_events is None:
            self.checkpoint_on_events = {"ENV", "AGENT", "VALIDATION", "STOP"}
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def should_checkpoint_event(self, event_type: str) -> bool:
        """Check if this event type should trigger a checkpoint."""
        if not self.enabled:
            return False
        
        if self.checkpoint_on_events is None:
            return False
        
        return event_type in self.checkpoint_on_events
    
    def get_checkpoint_path(self, scenario_id: str) -> Path:
        """Get the checkpoint directory path for a specific scenario."""
        return self.checkpoint_dir / scenario_id
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        return {
            'enabled': self.enabled,
            'checkpoint_dir': str(self.checkpoint_dir),
            'checkpoint_on_events': list(self.checkpoint_on_events) if self.checkpoint_on_events else [],
            'checkpoint_before_events': self.checkpoint_before_events,
            'checkpoint_after_events': self.checkpoint_after_events,
            'verbose_logging': self.verbose_logging,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CheckpointConfig':
        """Create configuration from dictionary."""
        # Convert event list back to set
        if 'checkpoint_on_events' in data:
            data['checkpoint_on_events'] = set(data['checkpoint_on_events'])
        
        return cls(**data)
    
    @classmethod
    def create_default_event_based(cls, checkpoint_dir: str = "./checkpoints") -> 'CheckpointConfig':
        """Create a default event-based checkpoint configuration."""
        return cls(
            enabled=True,
            checkpoint_dir=Path(checkpoint_dir),
            checkpoint_on_events={"ENV", "AGENT", "VALIDATION", "STOP"},
            checkpoint_before_events=True,
            checkpoint_after_events=True,
            verbose_logging=True,
        )
    
    @classmethod
    def create_minimal_testing(cls, checkpoint_dir: str = "./test_checkpoints") -> 'CheckpointConfig':
        """Create a minimal configuration for testing purposes."""
        return cls(
            enabled=True,
            checkpoint_dir=Path(checkpoint_dir),
            checkpoint_on_events={"AGENT"},  # Only checkpoint agent events for testing
            checkpoint_before_events=True,
            checkpoint_after_events=False,   # Reduce checkpoint volume for testing
            verbose_logging=True,
        )