"""
Checkpoint system for ARE agent benchmarking.

This package provides checkpointing functionality for the ARE framework.
Checkpoints are created after each agent tool call, capturing the complete
system state (environment, agent, apps) for debugging and analysis.

Key Components:
- CheckpointConfig: Configuration for checkpoint behavior
- CheckpointManager: Central coordinator for checkpoint operations
- CheckpointableEnvironment: Environment subclass with checkpointing
- CheckpointableScenarioRunner: ScenarioRunner subclass with checkpointing

Replay Components:
- CheckpointLoader: Load checkpoints from storage
- CheckpointReplayRunner: Load checkpoint and continue execution
- ScenarioMetaLoader: Load scenario metadata (ScenarioRunnerConfig) from scenario_meta.json
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
    CheckpointReplayRunner,
    CheckpointLoader,
    ScenarioMetaLoader,
    CheckpointReplayError,
    replay_from_checkpoint,
    list_checkpoints,
    get_checkpoint_info,
)
from .agent_state import capture_agent_state

__all__ = [
    # Core checkpointing
    "CheckpointConfig",
    "CheckpointManager",
    "CheckpointableEnvironment",
    "CheckpointableScenarioRunner",
    "create_checkpointable_runner",
    "create_checkpointable_environment",
    # Agent state capture
    "capture_agent_state",
    # Replay
    "CheckpointReplayRunner",
    "CheckpointLoader",
    "ScenarioMetaLoader",
    "CheckpointReplayError",
    "replay_from_checkpoint",
    "list_checkpoints",
    "get_checkpoint_info",
]