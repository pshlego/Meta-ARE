# Using Checkpoints with are-run CLI

## Quick Start

Enable checkpointing with the `--enable-checkpoints` flag:

```bash
are-run -s scenario_find_image_file -a default \
    --output_dir ./test_results \
    --enable-checkpoints
```

Checkpoints will be saved in `./test_results/checkpoints/`

## Command Options

### Basic Checkpointing
```bash
are-run -s <scenario_id> -a <agent> \
    --output_dir <output_directory> \
    --enable-checkpoints
```

### Select Specific Events
```bash
are-run -s <scenario_id> -a <agent> \
    --output_dir <output_directory> \
    --enable-checkpoints \
    --checkpoint-events "AGENT,VALIDATION"
```

### Verbose Checkpoint Logging
```bash
are-run -s <scenario_id> -a <agent> \
    --output_dir <output_directory> \
    --enable-checkpoints \
    --checkpoint-verbose
```

## Checkpoint Options

| Option | Description | Default |
|--------|-------------|---------|
| `--enable-checkpoints` | Enable checkpoint capture | False (disabled) |
| `--checkpoint-events` | Comma-separated event types to checkpoint | `ENV,AGENT,VALIDATION,STOP` |
| `--checkpoint-verbose` | Enable detailed checkpoint logging | False |

## Event Types

- **ENV**: Environment events (app actions, system events)
- **AGENT**: Agent actions and decisions
- **VALIDATION**: Validation checkpoints
- **STOP**: Stop/termination events

## Checkpoint Storage

Checkpoints are stored as a subdirectory in the specified `--output_dir`:

```
test_results/
├── checkpoints/                        # Checkpoint root
│   └── scenario_find_image_file/      # Per-scenario directory
│       ├── event_1_before_env/        # First checkpoint
│       │   ├── metadata.json
│       │   ├── environment_snapshot.json
│       │   ├── agent_snapshot.json
│       │   └── apps_snapshot/
│       ├── event_2_after_agent/       # Second checkpoint
│       └── event_3_after_validation/  # Third checkpoint
└── ... (other output files)
```

## Examples

### Example 1: Checkpoint All Events (Default)
```bash
are-run -s scenario_find_image_file -a default \
    --output_dir ./results \
    --enable-checkpoints
```

Creates checkpoints for ENV, AGENT, VALIDATION, and STOP events.

### Example 2: Checkpoint Only Agent Events
```bash
are-run -s scenario_find_image_file -a default \
    --output_dir ./results \
    --enable-checkpoints \
    --checkpoint-events "AGENT"
```

Creates checkpoints only when agent acts (reduces checkpoint volume).

### Example 3: Full Debugging Mode
```bash
are-run -s scenario_find_image_file -a default \
    --output_dir ./results \
    --enable-checkpoints \
    --checkpoint-verbose \
    --log_level DEBUG
```

Maximum verbosity for debugging checkpoint issues.

### Example 4: Multiple Scenarios with Checkpoints
```bash
are-run -s scenario_1 -s scenario_2 -s scenario_3 \
    -a default \
    --output_dir ./batch_results \
    --enable-checkpoints \
    --checkpoint-events "AGENT,VALIDATION"
```

Each scenario gets its own checkpoint subdirectory.

## Checkpoint File Structure

### Metadata File (`metadata.json`)
```json
{
  "checkpoint_id": "event_5_after_agent",
  "scenario_id": "scenario_find_image_file",
  "simulation_time": 123.456,
  "checkpoint_type": "after_agent",
  "trigger_event_type": "AGENT",
  "checkpoint_sequence": 5,
  "events_completed_count": 10
}
```

### Environment Snapshot (`environment_snapshot.json`)
Contains:
- Event log (past events)
- Event queue (future events)
- Time manager state
- Configuration settings
- Runtime state

### Agent Snapshot (`agent_snapshot.json`)
Contains:
- Agent logs (complete execution history)
- Iteration counters
- Custom state
- Action executor state
- Task and attachments

### Apps Snapshot (`apps_snapshot/`)
Individual JSON files for each app's state.

## Inspecting Checkpoints

### List Checkpoints
```bash
python -c "
from are.simulation.checkpoint import CheckpointLoader, CheckpointConfig
from pathlib import Path

config = CheckpointConfig(checkpoint_dir=Path('./test_results/checkpoints'))
loader = CheckpointLoader(config)
checkpoints = loader.list_checkpoints('scenario_find_image_file')
print(f'Found {len(checkpoints)} checkpoints:')
for cp_id in checkpoints:
    print(f'  - {cp_id}')
"
```

### Inspect Checkpoint Content
```bash
python -c "
from are.simulation.checkpoint import CheckpointLoader, CheckpointConfig
from pathlib import Path
import json

config = CheckpointConfig(checkpoint_dir=Path('./test_results/checkpoints'))
loader = CheckpointLoader(config)
checkpoint = loader.load_checkpoint('scenario_find_image_file', 'event_5_after_agent')

print('Checkpoint Info:')
print(f\"  Type: {checkpoint['metadata']['checkpoint_type']}\")
print(f\"  Time: {checkpoint['metadata']['simulation_time']}\")
print(f\"  Agent logs: {checkpoint['agent']['logs_count']}\")
print(f\"  Agent iterations: {checkpoint['agent']['iterations']}\")
"
```

## Disk Space Considerations

Checkpoints can be large (several MB per checkpoint). To manage disk space:

1. **Selective Events**: Use `--checkpoint-events "AGENT"` to reduce volume
2. **Regular Cleanup**: Delete old checkpoint directories after analysis
3. **Monitor Space**: Check `du -sh test_results/checkpoints/`

### Typical Checkpoint Sizes
- Small scenario: 500KB - 2MB per checkpoint
- Medium scenario: 2MB - 10MB per checkpoint
- Large scenario: 10MB - 50MB per checkpoint

## Troubleshooting

### Checkpoints Not Created
1. Verify `--enable-checkpoints` flag is set
2. Check `--checkpoint-events` includes desired event types
3. Enable `--checkpoint-verbose` to see detailed logs
4. Check output directory permissions

### Large Checkpoint Files
1. Reduce event types: `--checkpoint-events "AGENT"`
2. Consider storage capacity before long runs
3. Implement cleanup scripts for old checkpoints

### Permission Errors
Ensure write permissions on output directory:
```bash
chmod 755 ./test_results
```

## Integration with Existing Workflows

### With Trace Export
```bash
are-run -s scenario_find_image_file -a default \
    --output_dir ./results \
    --export \
    --enable-checkpoints
```

Both traces and checkpoints will be saved.

### With Oracle Mode
```bash
are-run -s scenario_find_image_file -a default \
    --output_dir ./results \
    --oracle \
    --enable-checkpoints
```

Checkpoints work in oracle mode as well.

### With Noise Injection
```bash
are-run -s scenario_find_image_file -a default \
    --output_dir ./results \
    --noise \
    --enable-checkpoints
```

Captures state with noise augmentation active.

## Next Steps

After capturing checkpoints, you can:
1. **Analyze agent behavior**: Inspect agent logs at different stages
2. **Debug issues**: Examine state when problems occur
3. **Replay execution**: Load checkpoint and resume (see replay documentation)
4. **Compare runs**: Diff checkpoints from different executions

For replay functionality, see `CHECKPOINT_REPLAY_GUIDE.md`.
