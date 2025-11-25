# ARE Checkpoint System

A comprehensive system for capturing and replaying system state during ARE scenario execution. The checkpoint system enables debugging, analysis, and continuation of agent execution from any saved checkpoint point.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Checkpoint Timing](#checkpoint-timing)
- [API Reference](#api-reference)
- [CLI Usage](#cli-usage)
- [Storage Format](#storage-format)
- [State Captured](#state-captured)
- [Replay from Checkpoint](#replay-from-checkpoint)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

The checkpoint system captures complete system state after each agent tool call, enabling:

- **Debugging**: Inspect state at any point in execution
- **Analysis**: Study agent behavior and decision-making
- **Replay**: Continue execution from any checkpoint
- **Reproducibility**: Re-run scenarios from specific states

### Key Features

- **Minimal Invasive Design**: All logic contained within `checkpoint/` package
- **Automatic Capture**: Checkpoints created after each agent tool call
- **Complete State**: Captures environment, agent logs, and app states
- **Replay Support**: Continue execution from any checkpoint
- **CLI Integration**: Easy-to-use command-line interface

---

## Quick Start

### Enable Checkpointing via CLI

```bash
# Basic checkpointing
are-run -s scenario_find_image_file -a default \
    --output_dir ./test_results \
    --enable-checkpoints

# With verbose logging
are-run -s scenario_find_image_file -a default \
    --output_dir ./test_results \
    --enable-checkpoints \
    --checkpoint-verbose
```

### Enable Checkpointing via Python API

```python
from pathlib import Path
from are.simulation.checkpoint import CheckpointConfig, CheckpointableScenarioRunner

# Create configuration
config = CheckpointConfig(
    enabled=True,
    checkpoint_dir=Path("./checkpoints"),
    verbose_logging=True
)

# Create runner with checkpointing
runner = CheckpointableScenarioRunner(
    checkpoint_config=config,
    # ... standard ScenarioRunner args ...
)

# Run - checkpoints created automatically after each agent tool call
result = runner.run(scenario_config, scenario)
```

### Replay from Checkpoint

```bash
# List available checkpoints
are-checkpoint list test_results/checkpoints

# View checkpoint details
are-checkpoint info test_results/checkpoints/event_1_after_tool_call

# Replay from checkpoint (uses saved config from scenario_meta.json)
are-checkpoint replay test_results/checkpoints/event_1_after_tool_call
```

---

## Architecture

### Module Structure

```
checkpoint/
├── __init__.py         # Public API exports
├── config.py           # CheckpointConfig dataclass
├── metadata.py         # CheckpointMetadata dataclass
├── storage.py          # JSON file storage (CheckpointStorage)
├── manager.py          # Central coordination (CheckpointManager)
├── agent_state.py      # Agent state capture (capture_agent_state)
├── wrappers.py         # Checkpointable wrappers (CheckpointableEnvironment, CheckpointableScenarioRunner)
├── replay.py           # Checkpoint replay (CheckpointReplayRunner, ScenarioMetaLoader)
├── cli.py              # CLI commands (are-checkpoint)
└── README.md           # This documentation
```

### Component Responsibilities

| Component | Purpose | Key Classes/Functions |
|-----------|---------|----------------------|
| `config.py` | Configuration | `CheckpointConfig` |
| `metadata.py` | Checkpoint metadata | `CheckpointMetadata` |
| `storage.py` | File I/O | `CheckpointStorage`, `CheckpointJSONEncoder` |
| `manager.py` | Coordination | `CheckpointManager` |
| `agent_state.py` | Agent state capture | `capture_agent_state()` |
| `wrappers.py` | Integration wrappers | `CheckpointableEnvironment`, `CheckpointableScenarioRunner` |
| `replay.py` | Checkpoint replay | `CheckpointReplayRunner`, `ScenarioMetaLoader` |
| `cli.py` | CLI commands | `list`, `info`, `replay` |

---

## Checkpoint Timing

### When Checkpoints Are Created

Checkpoints are created **after each agent tool call completes** (after the `ObservationLog` is added to agent logs). This ensures that:

1. The tool call has fully executed
2. The observation (tool result) is included in the checkpoint
3. The agent has complete context for replay

### Checkpoint Flow

```
1. Agent decides to call a tool
2. ToolCallLog added to agent logs
3. Tool executes
4. AGENT event added to environment log (env.add_to_log)
5. ObservationLog added to agent logs <- CHECKPOINT CREATED HERE
6. Agent processes observation and continues
```

### Implementation Details

The checkpoint is triggered via `setup_agent_checkpoint_callback()` which hooks into the agent's `log_callback`. When an `ObservationLog` is added, the checkpoint is created with the complete state including the tool result.

---

## API Reference

### CheckpointConfig

Configuration for the checkpoint system.

```python
from pathlib import Path
from are.simulation.checkpoint import CheckpointConfig

config = CheckpointConfig(
    enabled=True,                              # Master switch
    checkpoint_dir=Path("./checkpoints"),      # Storage directory
    checkpoint_on_events={"AGENT"},            # Event types to checkpoint
    verbose_logging=False                      # Detailed logging
)

# Convenience factory
config = CheckpointConfig.create_default(checkpoint_dir=Path("./checkpoints"))
```

### CheckpointManager

Central coordinator for checkpoint operations.

```python
from are.simulation.checkpoint import CheckpointManager, CheckpointConfig

config = CheckpointConfig.create_default(checkpoint_dir=Path("./checkpoints"))
manager = CheckpointManager(config)

# Set scenario config (for scenario_meta.json)
manager.set_scenario_config(
    runner_config=runner_config,
    scenario_creation_params={...},
    scenario_initialization_params={...}
)

# Create checkpoint
checkpoint_id = manager.create_event_checkpoint(
    scenario_id="my_scenario",
    checkpoint_type="after_tool_call",
    simulation_time=123.456,
    trigger_event_data={...},
    system_state={...}
)

# List checkpoints
checkpoints = manager.list_scenario_checkpoints(scenario_id)

# Load checkpoint
data = manager.get_checkpoint(scenario_id, checkpoint_id)

# Get statistics
stats = manager.get_statistics()
```

### CheckpointableScenarioRunner

ScenarioRunner with integrated checkpointing.

```python
from are.simulation.checkpoint import CheckpointConfig, CheckpointableScenarioRunner

config = CheckpointConfig.create_default(checkpoint_dir=Path("./checkpoints"))

runner = CheckpointableScenarioRunner(
    checkpoint_config=config,
    # ... standard ScenarioRunner args ...
)

# Run with automatic checkpointing
result = runner.run(scenario_config, scenario)
```

### CheckpointReplayRunner

Load checkpoint and continue execution.

```python
from are.simulation.checkpoint import CheckpointReplayRunner

runner = CheckpointReplayRunner(
    checkpoint_path="./checkpoints/event_1_after_tool_call",
    verbose=True
)

# Check if scenario metadata is available
if runner.has_scenario_meta():
    config = runner.get_runner_config_from_meta()

# Replay (config loaded from scenario_meta.json if not provided)
result, output_dir = runner.run(scenario, config=None, enable_checkpoints=True)
```

### ScenarioMetaLoader

Load scenario configuration from scenario_meta.json.

```python
from are.simulation.checkpoint import ScenarioMetaLoader
from pathlib import Path

loader = ScenarioMetaLoader(Path("./checkpoints"), verbose=True)

if loader.has_scenario_meta():
    meta = loader.load_scenario_meta()
    scenario_id = loader.get_scenario_id()
    runner_config = loader.get_runner_config()  # Returns ScenarioRunnerConfig
    creation_params = loader.get_scenario_creation_params()
    init_params = loader.get_scenario_initialization_params()
```

### Convenience Functions

```python
from are.simulation.checkpoint import (
    capture_agent_state,
    list_checkpoints,
    get_checkpoint_info,
    replay_from_checkpoint
)

# Capture agent state
agent_state = capture_agent_state(agent)

# List checkpoints in directory
checkpoints = list_checkpoints("./checkpoints")

# Get checkpoint details
info = get_checkpoint_info("./checkpoints/event_1_after_tool_call")

# Quick replay
result, output_dir = replay_from_checkpoint(
    checkpoint_path="./checkpoints/event_1_after_tool_call",
    scenario=scenario,
    verbose=True
)
```

---

## CLI Usage

### are-checkpoint list

List all checkpoints in a directory.

```bash
are-checkpoint list <checkpoint_dir>

# Example
are-checkpoint list test_results/checkpoints
```

Output:
```
Scenario: scenario_find_image_file
  Model: gpt-4
  Agent: default
  Max Turns: 10

Found 3 checkpoint(s):

  📁 event_1_after_tool_call
     Type: after_tool_call
     Created: 2024-01-15 10:23:45
     Agent Logs: 1

  📁 event_2_after_tool_call
     Type: after_tool_call
     Created: 2024-01-15 10:23:52
     Agent Logs: 2
```

### are-checkpoint info

Show detailed information about a checkpoint.

```bash
are-checkpoint info <checkpoint_path> [--json]

# Examples
are-checkpoint info test_results/checkpoints/event_1_after_tool_call
are-checkpoint info test_results/checkpoints/event_1_after_tool_call --json
```

Output:
```
Checkpoint: event_1_after_tool_call
  Path: test_results/checkpoints/event_1_after_tool_call
  Scenario ID: scenario_find_image_file
  Type: after_tool_call
  Created: 2024-01-15 10:23:45
  Event Sequence: 1
  Checkpoint Sequence: 1

State Counts:
  Agent Logs: 9
  Past Events: 5
  World Logs: 4
  Apps: 3

Scenario Metadata: ✅ Available
  Model: gpt-4
  Agent: default
  Max Turns: 10
```

### are-checkpoint replay

Replay execution from a checkpoint.

```bash
are-checkpoint replay <checkpoint_path> [OPTIONS]

# Options
-s, --scenario-id    Override scenario ID
-m, --model          Override model
-p, --provider       Override model provider  
-a, --agent          Override agent
--max-turns          Override max turns
-e, --export         Export trace to JSON
-o, --output-dir     Override output directory
-v, --verbose        Enable verbose logging
--log-level          Set logging level
```

**Examples:**

```bash
# Replay using saved config
are-checkpoint replay test_results/checkpoints/event_1_after_tool_call

# Replay with different model
are-checkpoint replay test_results/checkpoints/event_1_after_tool_call -m gpt-4

# Replay with export and verbose
are-checkpoint replay test_results/checkpoints/event_1_after_tool_call -e -v

# Replay with custom output directory
are-checkpoint replay test_results/checkpoints/event_1_after_tool_call -o ./replay_results
```

---

## Storage Format

### Directory Structure

```
checkpoints/
├── scenario_meta.json                    # Scenario configuration
├── event_1_after_tool_call/             # First checkpoint
│   └── checkpoint.json                   # Complete checkpoint data
├── event_2_after_tool_call/             # Second checkpoint
│   └── checkpoint.json
└── ...
```

### scenario_meta.json

Saved with the first checkpoint, contains configuration for auto-replay:

```json
{
  "scenario_id": "scenario_find_image_file",
  "checkpoint_dir": "./checkpoints",
  "created_at": 1705321425.123,
  "runner_config": {
    "model": "gpt-4",
    "model_provider": "azure",
    "agent": "default",
    "max_turns": 10,
    "oracle": false,
    "export": true,
    ...
  },
  "scenario_creation_params": {...},
  "scenario_initialization_params": {...}
}
```

### checkpoint.json

```json
{
  "metadata": {
    "checkpoint_id": "event_1_after_tool_call",
    "scenario_id": "scenario_find_image_file",
    "simulation_time": 123.456,
    "wall_clock_time": 1705321425.123,
    "created_at": "2024-01-15 10:23:45",
    "checkpoint_type": "after_tool_call",
    "trigger_event_id": "AGENT-123",
    "trigger_event_type": "AGENT",
    "event_sequence_number": 1,
    "checkpoint_sequence": 1
  },
  "trigger_event": {
    "event_type": "AGENT",
    "event_id": "AGENT-123",
    "function_name": "run",
    "app_name": "BashTool"
  },
  "environment": {
    "current_time": 123.456,
    "start_time": 0,
    "tick_count": 5,
    "state": "RUNNING",
    "event_log": {...},
    "event_queue": {...},
    "world_logs": [...],
    "configuration": {...}
  },
  "agent": {
    "agent_type": "ARESimulationAgent",
    "logs": [...],
    "logs_count": 9,
    "iterations": 1,
    "task": "Find the image file...",
    ...
  },
  "apps": {
    "FileManager": {...},
    "BashTool": {...},
    ...
  }
}
```

---

## State Captured

### Environment State

| Field | Description |
|-------|-------------|
| `current_time` | Simulation time |
| `start_time` | When simulation started |
| `tick_count` | Number of ticks elapsed |
| `state` | Environment state (RUNNING, STOPPED, etc.) |
| `event_log` | Past events (CompletedEvent objects) |
| `event_queue` | Future scheduled events |
| `world_logs` | BaseAgentLog objects for world events |
| `configuration` | Environment config (oracle_mode, etc.) |

### Agent State

| Field | Description |
|-------|-------------|
| `logs` | Complete agent logs (chat history) - **CRITICAL FOR REPLAY** |
| `logs_count` | Number of log entries |
| `iterations` | Agent iteration counter |
| `planning_counter` | Planning step counter |
| `task` | Current task description |
| `custom_state` | Any agent-specific state |
| `original_tool_name_to_variant` | Tool name mappings (for data augmentation) |
| `shuffle_seed` | Shuffle configuration seed |

### Apps State

For each registered app:

| Field | Description |
|-------|-------------|
| `app_type` | App class name |
| `state` | App state via `get_state()` method |
| `has_state` | Whether state was captured |
| `has_load_state` | Whether app supports state restoration |

---

## Replay from Checkpoint

### How Replay Works

1. Load checkpoint data
2. Load scenario_meta.json for configuration
3. Reconstruct agent logs from checkpoint
4. Reconstruct environment state (event_log, world_logs)
5. Restore app states
6. Continue agent execution with `initial_agent_logs`
7. Optionally create new checkpoints during replay

### Replay Output

Replay creates a timestamped output directory:

```
test_results/
├── checkpoints/                          # Original checkpoints
└── continual_run_20240115_102345/       # Replay output
    ├── checkpoints/                      # New checkpoints (if enabled)
    │   ├── scenario_meta.json           # Copied from original
    │   ├── event_2_after_tool_call/     # Continues numbering
    │   └── ...
    └── trace.json                        # Exported trace (if --export)
```

### Checkpoint Continuity

When replaying with checkpoints enabled:
- Sequence counters continue from the loaded checkpoint
- If loaded `event_1`, new checkpoints start at `event_2`
- scenario_meta.json is copied to the new checkpoint directory

---

## Best Practices

### Development

1. **Enable Verbose Logging**: Use `verbose_logging=True` or `--checkpoint-verbose` during development
2. **Inspect JSON Files**: Checkpoints are human-readable JSON - use `jq` for exploration
3. **Test Replay Early**: Verify replay works before long-running executions

### Production

1. **Monitor Disk Space**: Checkpoints can be large (1-50MB each)
2. **Regular Cleanup**: Delete old checkpoint directories after analysis
3. **Use Specific Checkpoint Events**: Configure `checkpoint_on_events` to reduce checkpoint frequency

### Debugging

1. **Compare Checkpoints**: Diff checkpoints from different runs
2. **Inspect Agent Logs**: Focus on `agent.logs` for execution history
3. **Check Event Sequence**: Verify event ordering in `event_log.past_events`

---

## Troubleshooting

### Common Issues

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| No checkpoints created | Checkpointing not enabled | Add `--enable-checkpoints` flag |
| Missing agent logs | Agent reference not captured | Verify using `CheckpointableScenarioRunner` |
| Large checkpoint files | Verbose app state | Consider implementing `get_state()` carefully |
| Replay fails | Missing scenario_meta.json | Ensure first checkpoint was created successfully |
| Wrong checkpoint state | Old timing (pre-fix) | Update to latest version, checkpoints now include ObservationLog |

### Debugging Commands

```bash
# List checkpoint directory structure
find test_results/checkpoints -name "*.json" | head -20

# Check checkpoint metadata
cat test_results/checkpoints/event_1_after_tool_call/checkpoint.json | jq '.metadata'

# Count agent logs in checkpoint
cat test_results/checkpoints/event_1_after_tool_call/checkpoint.json | jq '.agent.logs_count'

# View agent logs
cat test_results/checkpoints/event_1_after_tool_call/checkpoint.json | jq '.agent.logs'

# Check scenario metadata
cat test_results/checkpoints/scenario_meta.json | jq .
```

### Logging

Enable debug logging for detailed checkpoint operations:

```bash
are-run ... --enable-checkpoints --checkpoint-verbose --log_level DEBUG
```

Or in Python:

```python
import logging
logging.getLogger("are.simulation.checkpoint").setLevel(logging.DEBUG)
```

---

## Version History

### Current Version

- **Checkpoint Timing Fix**: Checkpoints now capture state AFTER `ObservationLog` is added
  - Ensures tool call results are included in checkpoint
  - Implemented via `setup_agent_checkpoint_callback()` instead of `add_to_log()` hook
- **Scenario Meta**: Saves `scenario_meta.json` for auto-configurable replay
- **Continual Run Checkpoints**: Replay creates new checkpoints continuing sequence numbering
- **CLI Commands**: `are-checkpoint list/info/replay` for checkpoint management
