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

The checkpoint system captures complete system state around each agent tool call, enabling:

- **Debugging**: Inspect state at any point in execution
- **Analysis**: Study agent behavior and decision-making
- **Replay**: Continue execution from any checkpoint
- **Reproducibility**: Re-run scenarios from specific states

### Key Features

- **Minimal Invasive Design**: All logic contained within `checkpoint/` package
- **Automatic Capture**: Checkpoints created before and after each agent tool call
- **Complete State**: Captures environment, agent logs, and app states
- **Configurable Types**: Enable/disable before_tool_call and after_tool_call independently
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
result = runner.run(config, scenario)
```

### Replay from Checkpoint

```bash
# List available checkpoints
are-checkpoint list test_results/checkpoints

# View checkpoint details
are-checkpoint info test_results/checkpoints/event_1_after_tool_call

# Replay from after_tool_call checkpoint (uses saved config from scenario_meta.json)
are-checkpoint replay test_results/checkpoints/event_1_after_tool_call

# Replay from before_tool_call checkpoint
# The pending tool call is executed first, then the agent continues normally
are-checkpoint replay test_results/checkpoints/event_1_before_tool_call
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
| `wrappers.py` | Integration wrappers | `CheckpointableEnvironment`, `CheckpointableScenarioRunner`, `create_checkpointable_runner()`, `create_checkpointable_environment()` |
| `replay.py` | Checkpoint replay | `CheckpointReplayRunner`, `CheckpointLoader`, `ScenarioMetaLoader`, `CheckpointReplayError` |
| `cli.py` | CLI commands | `list`, `info`, `replay` |

---

## Checkpoint Timing

### When Checkpoints Are Created

Two checkpoints are created around each agent tool call:

1. **before_tool_call**: Created when `ToolCallLog` is added to agent logs (before the tool executes). Captures the agent's decision — which tool to call and with what arguments.

2. **after_tool_call**: Created when `ObservationLog` is added to agent logs (after the tool completes). Captures the complete state including the tool result.

Both checkpoint types for the same tool call share the same event sequence number (e.g., `event_1_before_tool_call` and `event_1_after_tool_call`).

### Checkpoint Flow

```
1. Agent decides to call a tool
2. ToolCallLog added to agent logs     <- BEFORE_TOOL_CALL CHECKPOINT
3. Tool executes
4. AGENT event added to environment log (env.add_to_log)
5. ObservationLog added to agent logs   <- AFTER_TOOL_CALL CHECKPOINT
6. Agent processes observation and continues
```

### Configuration

By default, both checkpoint types are enabled. You can control which types are created:

```python
# Both (default)
config = CheckpointConfig(enabled=True, checkpoint_types={"before_tool_call", "after_tool_call"})

# Only before tool call
config = CheckpointConfig(enabled=True, checkpoint_types={"before_tool_call"})

# Only after tool call (legacy behavior)
config = CheckpointConfig(enabled=True, checkpoint_types={"after_tool_call"})
```

### Implementation Details

Both checkpoints are triggered via `setup_agent_checkpoint_callback()` which hooks into the agent's `log_callback`:
- When a `ToolCallLog` is added, a `before_tool_call` checkpoint is created
- When an `ObservationLog` is added, an `after_tool_call` checkpoint is created

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
    checkpoint_types={"before_tool_call", "after_tool_call"},  # Checkpoint types
    verbose_logging=False                      # Detailed logging
)

# Convenience factory (note: sets verbose_logging=True by default)
config = CheckpointConfig.create_default(checkpoint_dir="./checkpoints")

# Minimal testing configuration
config = CheckpointConfig.create_minimal_testing(checkpoint_dir="./test_checkpoints")
```

Additional methods:
- `should_checkpoint_event(event_type)` — check if an event type should trigger a checkpoint
- `should_create_checkpoint_type(checkpoint_type)` — check if a checkpoint type is enabled
- `get_checkpoint_path(scenario_id)` — get checkpoint directory for a scenario
- `to_dict()` / `from_dict(data)` — serialization support

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

config = CheckpointConfig.create_default(checkpoint_dir="./checkpoints")

runner = CheckpointableScenarioRunner(
    checkpoint_config=config,
    # ... standard ScenarioRunner args ...
)

# Run with automatic checkpointing
result = runner.run(config, scenario)
# config: ScenarioRunnerConfig, scenario: Scenario
# Optional: completed_events parameter for pre-loaded events
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

### CheckpointLoader

Load and validate checkpoint data from storage.

```python
from are.simulation.checkpoint import CheckpointLoader
from pathlib import Path

loader = CheckpointLoader(Path("./checkpoints"), verbose=True)

# List available checkpoints
checkpoint_ids = loader.list_checkpoints()

# Load checkpoint data
checkpoint_data = loader.load_checkpoint("event_1_after_tool_call")

# Get checkpoint metadata
metadata = loader.get_checkpoint_metadata("event_1_after_tool_call")
```

### Factory Functions

```python
from are.simulation.checkpoint import (
    create_checkpointable_runner,
    create_checkpointable_environment,
)

# Create a CheckpointableScenarioRunner from config
runner = create_checkpointable_runner(checkpoint_config=config, ...)

# Create a CheckpointableEnvironment from config
env = create_checkpointable_environment(checkpoint_config=config, ...)
```

### Convenience Functions

```python
from are.simulation.checkpoint import (
    capture_agent_state,
    list_checkpoints,
    get_checkpoint_info,
    replay_from_checkpoint,
    CheckpointReplayError,
)

# Capture agent state
agent_state = capture_agent_state(agent)

# List checkpoints in directory
checkpoints = list_checkpoints("./checkpoints")

# Get checkpoint details
info = get_checkpoint_info("./checkpoints/event_1_after_tool_call")

# Quick replay (config loaded from scenario_meta.json if not provided)
result, output_dir = replay_from_checkpoint(
    checkpoint_path="./checkpoints/event_1_after_tool_call",
    scenario=scenario,
    config=None,       # optional ScenarioRunnerConfig override
    verbose=True,
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
# Replay from after_tool_call checkpoint (normal)
are-checkpoint replay test_results/checkpoints/event_1_after_tool_call

# Replay from before_tool_call checkpoint
# Executes the pending tool call first, then continues the agent loop
are-checkpoint replay test_results/checkpoints/event_1_before_tool_call

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
├── event_1_before_tool_call/            # Before first tool call
│   ├── metadata.json
│   ├── trigger_event.json
│   ├── environment_snapshot.json
│   ├── agent_snapshot.json
│   └── apps_snapshot/
│       ├── AgentUserInterface_state.json
│       ├── SandboxLocalFileSystem_state.json
│       └── _protocol_mappings_state.json
├── event_1_after_tool_call/             # After first tool call
│   ├── metadata.json
│   ├── trigger_event.json
│   ├── environment_snapshot.json
│   ├── agent_snapshot.json
│   └── apps_snapshot/
│       └── ...
├── event_2_before_tool_call/            # Before second tool call
│   └── ...
├── event_2_after_tool_call/             # After second tool call
│   └── ...
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

### Checkpoint Files

Each checkpoint is stored as separate JSON files within its directory:

**metadata.json**
```json
{
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
}
```

**trigger_event.json**
```json
{
  "event_type": "AGENT",
  "event_id": "AGENT-123",
  "function_name": "run",
  "app_name": "BashTool"
}
```

**environment_snapshot.json** — full environment state (current_time, event_log, event_queue, world_logs, configuration, etc.)

**agent_snapshot.json** — agent state (logs, logs_count, iterations, task, custom_state, tool name mappings, etc.)

**apps_snapshot/** — directory with one `{AppName}_state.json` file per registered app

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
| `is_replaying` | Whether the environment is in replay mode |
| `verbose` | Verbose logging flag |
| `waiting_for_notification` | Notification wait state |
| `time_manager_state` | Time manager state (if available) |

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
| `shuffle_tools` | Whether tool order is shuffled |
| `agent_id` | Agent ID for log correlation |
| `agent_class` | Fully qualified agent class name |
| `timestamp` | When state was captured |

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

Replay behavior depends on the checkpoint type:

#### Replay from `after_tool_call` (normal)

1. Load checkpoint data
2. Load scenario_meta.json for configuration
3. Reconstruct agent logs from checkpoint
4. Reconstruct environment state (event_log, world_logs)
5. Restore app states
6. Continue agent execution with `initial_agent_logs` via `run_scenario()`
7. Optionally create new checkpoints during replay

#### Replay from `before_tool_call`

When replaying from a `before_tool_call` checkpoint the agent logs end with a `ToolCallLog` — the agent decided to call a tool but it hasn't been executed yet. The replay system handles this with a three-step process:

1. Load checkpoint data and restore state (same as above)
2. **Prepare** the agent: set up tools, system prompt, and replay logs via `prepare_are_simulation_run()`
3. **Execute the pending tool call**: extract the last `ToolCallLog`, build a `ParsedAction`, and run the tool through the action executor. The resulting `ObservationLog` is appended to agent logs, triggering the `after_tool_call` checkpoint (same event sequence number)
4. **Continue** with `agent_loop()` for subsequent LLM iterations

This ensures no gap in the checkpoint sequence — replaying from `event_1_before_tool_call` first produces `event_1_after_tool_call`, then continues normally.

### Replay Output

Replay creates a timestamped output directory:

**From `after_tool_call` checkpoint (e.g., `event_1_after_tool_call`):**
```
test_results/
├── checkpoints/                          # Original checkpoints
└── continual_run_20240115_102345/       # Replay output
    ├── checkpoints/                      # New checkpoints (if enabled)
    │   ├── scenario_meta.json           # Copied from original
    │   ├── event_2_before_tool_call/    # Continues numbering
    │   ├── event_2_after_tool_call/
    │   └── ...
    └── trace.json                        # Exported trace (if --export)
```

**From `before_tool_call` checkpoint (e.g., `event_1_before_tool_call`):**
```
test_results/
├── checkpoints/                          # Original checkpoints
└── continual_run_20240115_102345/       # Replay output
    ├── checkpoints/                      # New checkpoints (if enabled)
    │   ├── scenario_meta.json           # Copied from original
    │   ├── event_1_after_tool_call/     # Pending tool executed first
    │   ├── event_2_before_tool_call/    # Then normal continuation
    │   ├── event_2_after_tool_call/
    │   └── ...
    └── trace.json                        # Exported trace (if --export)
```

### Checkpoint Continuity

When replaying with checkpoints enabled:
- Sequence counters continue from the loaded checkpoint
- **From `after_tool_call`**: If loaded `event_1_after_tool_call` (event_seq=1, cp_seq=2), new checkpoints start at `event_2_before_tool_call` (event_seq=2, cp_seq=3)
- **From `before_tool_call`**: If loaded `event_1_before_tool_call` (event_seq=1, cp_seq=1), the pending tool executes first producing `event_1_after_tool_call` (event_seq=1, cp_seq=2), then continues with `event_2_*`
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
| Replay from before_tool_call skips pending tool | Old replay logic | Update to latest version with `_run_from_before_tool_call()` support |
| Replay from before_tool_call fails with type error | Non-default agent type | Replay from before_tool_call requires `ARESimulationAgent` and `JsonActionExecutor` |

### Debugging Commands

```bash
# List checkpoint directory structure
find test_results/checkpoints -name "*.json" | head -20

# Check checkpoint metadata
cat test_results/checkpoints/event_1_before_tool_call/metadata.json | jq .

# See what tool the agent decided to call (before execution)
cat test_results/checkpoints/event_1_before_tool_call/trigger_event.json | jq .

# Count agent logs in checkpoint
cat test_results/checkpoints/event_1_after_tool_call/agent_snapshot.json | jq '.logs_count'

# View agent logs
cat test_results/checkpoints/event_1_after_tool_call/agent_snapshot.json | jq '.logs'

# View environment state
cat test_results/checkpoints/event_1_after_tool_call/environment_snapshot.json | jq '.current_time'

# View trigger event
cat test_results/checkpoints/event_1_after_tool_call/trigger_event.json | jq .

# List app snapshots
ls test_results/checkpoints/event_1_after_tool_call/apps_snapshot/

# Check scenario metadata
cat test_results/checkpoints/scenario_meta.json | jq .

# Compare state before and after tool call
diff <(cat test_results/checkpoints/event_1_before_tool_call/agent_snapshot.json | jq '.logs_count') \
     <(cat test_results/checkpoints/event_1_after_tool_call/agent_snapshot.json | jq '.logs_count')
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

- **Dual Checkpoint Timing**: Two checkpoints per tool call
  - `before_tool_call`: Captures state when agent decides to call a tool (on `ToolCallLog`)
  - `after_tool_call`: Captures state after tool result is received (on `ObservationLog`)
  - Both share the same event sequence number (e.g., `event_1_before_tool_call`, `event_1_after_tool_call`)
  - Configurable via `checkpoint_types` in `CheckpointConfig`
- **Replay from both checkpoint types**:
  - `after_tool_call`: Normal replay — agent continues from where it left off
  - `before_tool_call`: Executes the pending tool call first (producing `after_tool_call` checkpoint), then continues the agent loop. Implemented via `_run_from_before_tool_call()` and `_execute_pending_tool_call()` in `replay.py`
- **Checkpoint Timing via `setup_agent_checkpoint_callback()`**: Hooks into agent's `log_callback`
- **Scenario Meta**: Saves `scenario_meta.json` for auto-configurable replay
- **Continual Run Checkpoints**: Replay creates new checkpoints continuing sequence numbering
- **CLI Commands**: `are-checkpoint list/info/replay` for checkpoint management
