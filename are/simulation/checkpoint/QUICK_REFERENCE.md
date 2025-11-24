# Checkpoint System Quick Reference

## Quick Start

### Enable Checkpointing
```python
from are.simulation.checkpoint import CheckpointConfig, CheckpointableScenarioRunner

# Create config
config = CheckpointConfig(
    enabled=True,
    checkpoint_dir=Path("./checkpoints"),
    checkpoint_on_events={"ENV", "AGENT", "VALIDATION", "STOP"},
    verbose_logging=True
)

# Create runner with checkpointing
runner = CheckpointableScenarioRunner(
    checkpoint_config=config,
    # ... standard ScenarioRunner args ...
)

# Run - checkpoints created automatically
result = runner.run(scenario_config, scenario)
```

### Load and Replay
```python
from are.simulation.checkpoint import CheckpointLoader, restore_agent_logs

# Load checkpoint
loader = CheckpointLoader(config)
checkpoint = loader.load_checkpoint(scenario_id, checkpoint_id)

# Restore agent logs
agent_logs = restore_agent_logs(checkpoint)

# Resume execution
agent = agent_builder.build(agent_config, env)
result = agent.run_scenario(scenario, initial_agent_logs=agent_logs)
```

## Key Components

| Component | Purpose | Key Methods |
|-----------|---------|-------------|
| `CheckpointConfig` | Configuration | `create_default_event_based()` |
| `CheckpointManager` | Coordination | `create_event_checkpoint()` |
| `CheckpointableScenarioRunner` | Integration | `run()`, `enable_checkpointing()` |
| `CheckpointLoader` | Loading | `load_checkpoint()`, `list_checkpoints()` |
| `AgentStateCapture` | Agent state | `capture_agent_state()` |
| `AgentStateRestorer` | Agent restore | `restore_agent_logs()` |

## Checkpoint Structure

```
checkpoints/
└── {scenario_id}/
    └── {checkpoint_id}/
        ├── metadata.json              # When, why, what
        ├── environment_snapshot.json  # Full env state
        ├── agent_snapshot.json        # Full agent state
        └── apps_snapshot/             # App states
```

## Event Types

- **ENV**: Environment events (app actions, system events)
- **AGENT**: Agent actions and decisions
- **VALIDATION**: Validation checkpoints
- **STOP**: Stop/termination events

## Configuration Presets

### Development (All Events)
```python
config = CheckpointConfig.create_default_event_based()
# Checkpoints: ENV, AGENT, VALIDATION, STOP
# Before + After events
```

### Testing (Minimal)
```python
config = CheckpointConfig.create_minimal_testing()
# Checkpoints: AGENT only
# Before events only
```

### Custom
```python
config = CheckpointConfig(
    enabled=True,
    checkpoint_dir=Path("./my_checkpoints"),
    checkpoint_on_events={"AGENT", "VALIDATION"},
    checkpoint_before_events=True,
    checkpoint_after_events=True,
    verbose_logging=False
)
```

## State Captured

### Environment
- Event log (past events)
- Event queue (future events)
- Time manager state
- Configuration
- Runtime state

### Agent
- Agent logs (complete history)
- Iteration counters
- Custom state
- Task and attachments
- Action executor state

### Apps
- App-specific state via `get_state()`
- Requires `load_state()` for restoration

## Common Patterns

### List Checkpoints
```python
checkpoints = loader.list_checkpoints(scenario_id)
for cp_id in checkpoints:
    metadata = loader.get_checkpoint_metadata(scenario_id, cp_id)
    print(f"{cp_id}: {metadata['checkpoint_type']} at {metadata['simulation_time']}")
```

### Inspect Checkpoint
```python
checkpoint = loader.load_checkpoint(scenario_id, checkpoint_id)
print(f"Agent iterations: {checkpoint['agent']['iterations']}")
print(f"Agent logs: {checkpoint['agent']['logs_count']}")
print(f"Events completed: {len(checkpoint['environment']['event_log']['past_events'])}")
```

### Conditional Checkpointing
```python
# Only checkpoint specific events
config = CheckpointConfig(
    enabled=True,
    checkpoint_on_events={"AGENT"},  # Agent actions only
    checkpoint_before_events=True,
    checkpoint_after_events=False,  # Skip after-checkpoints
)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No checkpoints created | Enable `verbose_logging=True` and check logs |
| Agent state incomplete | Verify CheckpointableScenarioRunner is used |
| Replay fails | Check agent implements required from_dict() methods |
| Large files | Reduce event types or disable after-checkpoints |

## Best Practices

1. **Start Minimal**: Use `create_minimal_testing()` for initial testing
2. **Enable Verbose Logging**: Use `verbose_logging=True` during development
3. **Checkpoint Critical Events**: Focus on AGENT and VALIDATION events
4. **Regular Cleanup**: Delete old checkpoints to save space
5. **Test Replay**: Always verify checkpoint → replay cycle works
6. **Inspect Files**: Checkpoints are JSON - easy to inspect with tools

## API Reference

### CheckpointConfig
```python
CheckpointConfig(
    enabled: bool = False,
    checkpoint_dir: Path = Path("./checkpoints"),
    checkpoint_on_events: Optional[Set[str]] = None,
    checkpoint_before_events: bool = True,
    checkpoint_after_events: bool = True,
    verbose_logging: bool = False
)
```

### CheckpointableScenarioRunner
```python
CheckpointableScenarioRunner(
    checkpoint_config: Optional[CheckpointConfig] = None,
    checkpoint_manager: Optional[CheckpointManager] = None,
    **kwargs  # Standard ScenarioRunner args
)
```

### CheckpointLoader
```python
loader = CheckpointLoader(checkpoint_config)
checkpoint_data = loader.load_checkpoint(scenario_id, checkpoint_id)
checkpoints = loader.list_checkpoints(scenario_id)
metadata = loader.get_checkpoint_metadata(scenario_id, checkpoint_id)
```

### Agent Restoration
```python
from are.simulation.checkpoint import restore_agent_logs

# Extract agent logs from checkpoint
agent_logs = restore_agent_logs(checkpoint_data)

# Pass to agent
result = agent.run_scenario(
    scenario=scenario,
    initial_agent_logs=agent_logs
)
```

## Examples

See `demo_agent_state.py` for complete working examples.
