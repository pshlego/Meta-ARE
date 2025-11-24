"""
Checkpoint system for ARE agent benchmarking.

This package provides event-based checkpointing functionality for the ARE framework
following minimal invasive design principles. All checkpoint logic is contained
within this package to avoid contaminating existing code.

Key Components:
- CheckpointConfig: Configuration for checkpoint behavior
- CheckpointManager: Central coordinator for checkpoint operations
- CheckpointStorage: Filesystem storage backend for checkpoints
- EventCheckpointClassifier: Determines which events require checkpoints
- CheckpointableEnvironment: Environment subclass with checkpointing
- CheckpointableScenarioRunner: ScenarioRunner subclass with checkpointing
- CheckpointReplayer: Load and replay from saved checkpoints
"""

from .config import CheckpointConfig
from .manager import CheckpointManager
from .wrappers import (
    CheckpointableEnvironment,
    CheckpointableScenarioRunner,
    create_checkpointable_runner,
    create_checkpointable_environment,
)
from .replay import (
    CheckpointReplayer,
    CheckpointLoader,
    EnvironmentRestorer,
    CheckpointReplayError,
    replay_from_checkpoint,
    list_checkpoints,
)
from .agent_state import (
    AgentStateCapture,
    AgentStateRestorer,
    capture_agent_state,
    restore_agent_logs,
)

__all__ = [
    "CheckpointConfig",
    "CheckpointManager",
    "CheckpointableEnvironment",
    "CheckpointableScenarioRunner",
    "create_checkpointable_runner",
    "create_checkpointable_environment",
    "CheckpointReplayer",
    "CheckpointLoader",
    "EnvironmentRestorer",
    "CheckpointReplayError",
    "replay_from_checkpoint",
    "list_checkpoints",
    "AgentStateCapture",
    "AgentStateRestorer",
    "capture_agent_state",
    "restore_agent_logs",
]