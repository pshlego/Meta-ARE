# Checkpoint System Architecture

## Overview

The checkpoint system provides comprehensive state capture and replay capabilities for the ARE agent benchmarking framework. It follows a **minimal invasive design** principle, keeping all checkpoint logic contained within the `checkpoint/` package without modifying existing ARE code.

## Design Principles

1. **Minimal Invasiveness**: Uses inheritance and monkey-patching to integrate with ARE without modifying core files
2. **Event-Based Checkpointing**: Captures state at critical event boundaries (ENV, AGENT, VALIDATION, STOP)
3. **Complete State Capture**: Saves environment, agent, apps, and execution context
4. **JSON Storage**: Human-readable checkpoint format for debugging and analysis
5. **Thread-Safe**: Preserves ARE's asynchronous event-driven architecture

## Architecture Components

### Core Modules

```
checkpoint/
├── config.py           # Checkpoint configuration
├── metadata.py         # Checkpoint metadata structures
├── storage.py          # JSON-based file storage
├── classifier.py       # Event classification logic
├── manager.py          # Central coordination
├── agent_state.py      # Agent state capture/restore
├── wrappers.py         # Checkpointable wrapper classes
├── replay.py           # Checkpoint loading and replay
└── __init__.py         # Public API exports
```

### 1. Configuration (`config.py`)

**Purpose**: Define checkpoint behavior and settings

**Key Class**: `CheckpointConfig`
- `enabled`: Master switch for checkpointing
- `checkpoint_dir`: Base directory for storage
- `checkpoint_on_events`: Set of event types to checkpoint (ENV, AGENT, VALIDATION, STOP)
- `checkpoint_before_events`: Create checkpoints before event execution
- `checkpoint_after_events`: Create checkpoints after event execution
- `verbose_logging`: Detailed logging for debugging

**Factory Methods**:
- `create_default_event_based()`: Standard configuration
- `create_minimal_testing()`: Reduced checkpoint volume for testing

### 2. Metadata (`metadata.py`)

**Purpose**: Store checkpoint context and timing information

**Key Class**: `CheckpointMetadata`
- Checkpoint identification (ID, scenario ID)
- Timing information (simulation time, wall clock time)
- Event context (trigger event type, event ID)
- Sequence numbers (checkpoint sequence, event sequence)
- System state summary (agent turn, event counts)

**Methods**:
- `to_dict()` / `from_dict()`: JSON serialization
- `create_event_boundary()`: Factory for event checkpoints

### 3. Storage (`storage.py`)

**Purpose**: Handle filesystem operations for checkpoint data

**Key Class**: `CheckpointStorage`

**Directory Structure**:
```
checkpoints/
└── {scenario_id}/
    └── {checkpoint_id}/
        ├── metadata.json                  # Checkpoint metadata
        ├── trigger_event.json             # Triggering event data
        ├── completed_event.json           # Completion data (after-checkpoints)
        ├── environment_snapshot.json      # Full environment state
        ├── agent_snapshot.json            # Full agent state
        ├── execution_context.json         # Execution context
        └── apps_snapshot/                 # App states
            ├── app1_state.json
            └── app2_state.json
```

**Methods**:
- `save_checkpoint()`: Save complete checkpoint
- `load_checkpoint()`: Load checkpoint data
- `list_scenario_checkpoints()`: List available checkpoints
- `delete_checkpoint()` / `delete_scenario_checkpoints()`: Cleanup

### 4. Event Classification (`classifier.py`)

**Purpose**: Determine which events trigger checkpoints

**Key Class**: `EventCheckpointClassifier`

**Logic**:
- Critical events: ENV, AGENT, VALIDATION, STOP
- Before-event checkpoints: Capture state before execution
- After-event checkpoints: Capture state after successful/failed execution
- First event special handling: Always checkpoint first occurrence

**Methods**:
- `is_critical_event()`: Check if event type is critical
- `should_checkpoint_before_event()`: Decision logic for before-checkpoints
- `should_checkpoint_after_event()`: Decision logic for after-checkpoints
- `analyze_event_for_checkpoint()`: Comprehensive analysis with reasoning
- `get_statistics()`: Monitoring and optimization metrics

### 5. Checkpoint Manager (`manager.py`)

**Purpose**: Central coordinator for all checkpoint operations

**Key Class**: `CheckpointManager`

**Responsibilities**:
- Coordinate between config, storage, classifier, and metadata
- Generate checkpoint IDs and maintain sequence counters
- Create event-based checkpoints with complete system state
- Provide query interface for checkpoint retrieval

**Key Methods**:
- `create_event_checkpoint()`: Create checkpoint at event boundary
- `handle_event_start()`: Handle before-event checkpoint
- `handle_event_completion()`: Handle after-event checkpoint
- `should_create_checkpoint()`: Delegate to classifier
- `get_checkpoint()`: Retrieve specific checkpoint
- `list_scenario_checkpoints()`: List available checkpoints

### 6. Agent State Capture (`agent_state.py`)

**Purpose**: Capture and restore complete agent state

**Key Classes**: 
- `AgentStateCapture`: Serialize agent state
- `AgentStateRestorer`: Deserialize agent state

**Agent State Components**:
```python
{
    "agent_type": "BaseAgent",
    "agent_id": "agent_123",
    "logs": [...],                    # List of BaseAgentLog
    "logs_count": 42,
    "iterations": 10,                 # Current iteration
    "planning_counter": 5,            # Planning steps
    "custom_state": {},               # Custom agent state
    "task": "Task description",
    "attachments": [],                # Task attachments
    "action_executor_state": {...},   # Executor state
    "original_tool_name_to_variant": {},
    "initialized": true,
    "system_prompt": "...",
    "shuffle_seed": 42,
    "tools_info": {
        "tool_count": 15,
        "tool_names": [...]
    }
}
```

**Key Methods**:
- `capture_agent_state(agent)`: Comprehensive state capture
- `restore_agent_logs(checkpoint_data)`: Extract logs for replay
- `validate_agent_state()`: Verify checkpoint integrity

### 7. Checkpointable Wrappers (`wrappers.py`)

**Purpose**: Integrate checkpointing with ARE components

**Key Classes**:

#### `CheckpointableEnvironment(Environment)`
- Inherits from `Environment`
- Overrides `process_event()` to create checkpoints at event boundaries
- Captures complete environment state (event log, event queue, apps, time manager)
- Receives agent reference via callback for agent state capture

**State Capture**:
- `_capture_environment_state()`: Time, config, event system
- `_capture_agent_state()`: Agent logs, iterations, custom state
- `_capture_apps_state()`: App states via `get_state()` method
- `_capture_event_log()`: Past events
- `_capture_event_queue()`: Future events

#### `CheckpointableScenarioRunner(ScenarioRunner)`
- Inherits from `ScenarioRunner`
- Overrides `run()` to enable checkpointing
- Uses **monkey-patching** to inject checkpointing while preserving threading model

**Monkey-Patching Strategy**:
```python
# Patch 1: Environment class → CheckpointableEnvironment factory
Environment = create_checkpointable_env

# Patch 2: agent_builder.build() → wrapper that captures agent reference
agent_builder.build = build_with_capture

# Call parent's run() - preserves original threading:
# - env.run(scenario, wait_for_end=False) - background thread
# - agent.run_scenario() - main thread
# - Event-based communication maintained
super().run(config, scenario, completed_events)
```

**Why This Works**:
1. Environment runs in background thread (asynchronous)
2. Agent runs in main thread (synchronous)
3. They communicate via events (event queue)
4. Agent reference captured during `build()` and passed to environment
5. Environment can access agent state for checkpoints

### 8. Replay System (`replay.py`)

**Purpose**: Load and restore checkpoints for replay

**Key Classes**:

#### `CheckpointLoader`
- Load checkpoint data from storage
- Validate checkpoint integrity
- List available checkpoints

#### `EnvironmentRestorer`
- Restore environment state from checkpoint
- Reconstruct event log and event queue
- Restore apps state via `load_state()`
- Restore time manager state

#### `CheckpointReplayer`
- High-level replay orchestration
- Combine loader + restorer
- Provide replay interface

**Replay Flow**:
```python
# 1. Load checkpoint
loader = CheckpointLoader(config)
checkpoint_data = loader.load_checkpoint(scenario_id, checkpoint_id)

# 2. Restore environment
restorer = EnvironmentRestorer()
env = restorer.restore_environment(checkpoint_data)

# 3. Extract agent logs
agent_logs = restore_agent_logs(checkpoint_data)

# 4. Create agent with restored logs
agent = agent_builder.build(agent_config, env)
result = agent.run_scenario(
    scenario=scenario,
    initial_agent_logs=agent_logs  # Resume from checkpoint
)
```

## Integration Flow

### Checkpoint Creation Flow

```
1. User creates CheckpointableScenarioRunner with CheckpointConfig
   
2. Runner.run() calls _run_with_checkpointing()
   
3. Monkey-patch Environment → CheckpointableEnvironment factory
   Monkey-patch agent_builder.build() → capture agent reference
   
4. Call super().run() which follows original ARE threading:
   - env.run(scenario, wait_for_end=False) → background thread
   - agent_builder.build() → captures agent in self._current_agent
   - agent.run_scenario() → main thread
   - Agent ↔ Environment via events (async)
   
5. Environment receives agent via callback:
   env._get_agent_for_checkpoint() → returns captured agent
   
6. During execution, events trigger checkpoints:
   a) Environment.process_event() called
   b) Check if event is critical (ENV/AGENT/VALIDATION/STOP)
   c) Create before-checkpoint with current state
   d) Execute event
   e) Create after-checkpoint with updated state
   
7. Checkpoint creation:
   a) capture_environment_state() → event_log, event_queue, config
   b) capture_agent_state() → logs, iterations, custom_state
   c) capture_apps_state() → app.get_state() for each app
   d) Build checkpoint_data dictionary
   e) Save to JSON files via CheckpointStorage
```

### Checkpoint Replay Flow

```
1. Load checkpoint:
   checkpoint_data = CheckpointLoader.load_checkpoint(scenario_id, checkpoint_id)
   
2. Restore environment:
   - Create new Environment instance
   - Restore time_manager state
   - Restore event_log with past events
   - Restore event_queue with future events
   - Restore apps via app.load_state()
   - Restore configuration
   
3. Restore agent:
   - Extract agent_logs from checkpoint_data["agent"]["logs"]
   - Deserialize using BaseAgentLog.from_dict()
   - Create agent with restored configuration
   
4. Resume execution:
   - Pass initial_agent_logs to agent.run_scenario()
   - Agent continues from checkpoint state
   - Environment continues processing queued events
```

## State Capture Details

### Environment State
```json
{
  "current_time": 1234567890.0,
  "start_time": 0,
  "tick_count": 42,
  "state": "RUNNING",
  "event_log": {
    "past_events": [...]  // CompletedEvent.to_dict()
  },
  "event_queue": {
    "future_events": [...],  // Event.to_dict()
    "already_scheduled": ["evt1", "evt2"]
  },
  "events_to_schedule": [...],
  "configuration": {
    "oracle_mode": false,
    "queue_based_loop": true,
    "time_increment_in_seconds": 1
  },
  "time_manager_state": {...}
}
```

### Agent State
```json
{
  "agent_type": "BaseAgent",
  "logs": [
    {
      "log_type": "PlanningLog",
      "timestamp": 123.456,
      "content": {...}
    }
  ],
  "logs_count": 42,
  "iterations": 10,
  "planning_counter": 5,
  "custom_state": {},
  "task": "Complete the task",
  "attachments": [],
  "action_executor_state": {
    "executor_type": "ActionExecutor",
    "state_keys": ["key1", "key2"]
  }
}
```

### Apps State
```json
{
  "app_name": {
    "app_type": "FileApp",
    "has_state": true,
    "has_load_state": true,
    "state": {
      "files": {...},
      "current_directory": "/home/user"
    }
  }
}
```

## Usage Examples

### Basic Checkpointing

```python
from are.simulation.checkpoint import (
    CheckpointConfig,
    CheckpointableScenarioRunner,
)

# Configure checkpointing
config = CheckpointConfig.create_default_event_based(
    checkpoint_dir="./my_checkpoints"
)

# Create checkpointable runner
runner = CheckpointableScenarioRunner(
    checkpoint_config=config,
    # ... other ScenarioRunner args
)

# Run scenario with checkpointing
result = runner.run(scenario_config, scenario)

# Checkpoints are automatically created at event boundaries
```

### Checkpoint Replay

```python
from are.simulation.checkpoint import (
    CheckpointLoader,
    EnvironmentRestorer,
    restore_agent_logs,
)

# Load checkpoint
loader = CheckpointLoader(checkpoint_config)
checkpoint_data = loader.load_checkpoint(
    scenario_id="my_scenario",
    checkpoint_id="event_10_after_agent"
)

# Restore environment
restorer = EnvironmentRestorer()
env = restorer.restore_environment(checkpoint_data)

# Restore agent logs
agent_logs = restore_agent_logs(checkpoint_data)

# Create agent and resume
agent = agent_builder.build(agent_config, env)
result = agent.run_scenario(
    scenario=scenario,
    initial_agent_logs=agent_logs
)
```

### Custom Configuration

```python
# Minimal checkpointing for testing
config = CheckpointConfig(
    enabled=True,
    checkpoint_dir=Path("./test_checkpoints"),
    checkpoint_on_events={"AGENT"},  # Only agent events
    checkpoint_before_events=True,
    checkpoint_after_events=False,   # Reduce volume
    verbose_logging=True
)
```

## Key Design Decisions

### 1. Inheritance over Composition
- `CheckpointableEnvironment` inherits from `Environment`
- `CheckpointableScenarioRunner` inherits from `ScenarioRunner`
- Minimal code duplication, natural override pattern

### 2. Monkey-Patching for Integration
- Preserves original threading model
- No modifications to ARE core code
- Clean separation of concerns

### 3. Event-Based Checkpoints
- Captures state at meaningful boundaries
- Before/after pairs enable precise replay
- Critical events: ENV, AGENT, VALIDATION, STOP

### 4. JSON Storage
- Human-readable for debugging
- Easy to inspect and modify
- Standard tools for analysis

### 5. Comprehensive State Capture
- Environment: event_log, event_queue, time_manager, config
- Agent: logs, iterations, custom_state, action_executor
- Apps: get_state() / load_state() protocol

### 6. Agent Reference via Callback
- ScenarioRunner captures agent during build()
- Passes reference to Environment via closure
- Environment accesses agent for state capture
- Clean separation, no tight coupling

## Testing and Validation

### Verify Checkpoint Creation
```python
# Check checkpoint files exist
checkpoints = loader.list_checkpoints(scenario_id)
print(f"Created {len(checkpoints)} checkpoints")

# Inspect checkpoint content
checkpoint_data = loader.load_checkpoint(scenario_id, checkpoint_id)
print(f"Agent logs: {checkpoint_data['agent']['logs_count']}")
print(f"Environment state: {checkpoint_data['environment']['state']}")
```

### Verify State Capture
```python
# Check agent state completeness
agent_state = checkpoint_data['agent']
assert agent_state['full_agent_state_captured'] == True
assert agent_state['logs_count'] > 0

# Check environment state
env_state = checkpoint_data['environment']
assert 'event_log' in env_state
assert 'event_queue' in env_state
```

### Verify Replay
```python
# Load and replay
env = restorer.restore_environment(checkpoint_data)
agent_logs = restore_agent_logs(checkpoint_data)

# Verify agent logs restored
assert len(agent_logs) == checkpoint_data['agent']['logs_count']

# Resume and compare
result = agent.run_scenario(scenario, initial_agent_logs=agent_logs)
# Compare result with original execution
```

## Limitations and Future Work

### Current Limitations
1. Agent state requires BaseAgentLog.from_dict() implementation
2. App restoration requires apps to implement load_state()
3. Event deserialization requires Event.from_dict() support
4. No compression for large checkpoints
5. No checkpoint cleanup policies

### Future Enhancements
1. Checkpoint compression for large scenarios
2. Incremental checkpointing (only changed state)
3. Checkpoint versioning for backward compatibility
4. Automatic checkpoint cleanup based on policies
5. Checkpoint comparison and diff tools
6. Performance profiling integration
7. Distributed checkpoint storage support

## Troubleshooting

### Checkpoints Not Created
- Verify `config.enabled = True`
- Check event types in `checkpoint_on_events`
- Enable `verbose_logging` for detailed logs
- Verify CheckpointManager is passed to runner

### Agent State Not Captured
- Ensure ScenarioRunner uses CheckpointableScenarioRunner
- Verify agent reference callback is set
- Check agent has required attributes (logs, iterations)

### Replay Fails
- Verify checkpoint data completeness
- Check BaseAgentLog.from_dict() implementation
- Verify apps implement load_state()
- Check Event.from_dict() for event deserialization

### Large Checkpoint Files
- Reduce checkpoint frequency (fewer event types)
- Use `checkpoint_after_events=False` for testing
- Implement custom state filtering in apps
- Consider checkpoint cleanup policies
