# Historical Trace Replay Implementation Task

## Task Context

### What We Want to Achieve
Extend the ARE simulation system to support **historical trace replay with modification capabilities**. Currently, the system only supports modifying and replaying logs in **live sessions** where all agent, environment, and tool states are already in memory. We want to enable the same "modify-and-rerun-from-point" functionality for **historical execution traces** loaded from disk.

### Current System Capabilities

#### ✅ What Already Works (Live Sessions)
1. **Agent Log Editing UI**: 
   - Located in `are/simulation/gui/client/src/components/agent-logs/AgentLogDetailView.tsx`
   - Supports editing specific log types: `SYSTEM_PROMPT`, `LLM_OUTPUT_THOUGHT_ACTION`, `LLM_OUTPUT_FACTS`, `LLM_OUTPUT_PLAN`, `OBSERVATION`
   - Provides text editor with save/cancel functionality

2. **Live Replay Backend**:
   - `replay_modified_output(log_id, content)` in `are/simulation/gui/server/are_simulation_gui.py`
   - Uses `AmendAndReplayMutation` GraphQL mutation
   - Calls existing replay functions from `are/simulation/replay.py`

3. **Basic Replay Functions**:
   - `replay_logs()` in `are/simulation/replay.py` 
   - `find_log_position()` and `find_previous_step_position()` for navigation
   - Works by re-executing ActionLog entries sequentially

4. **Execution Trace Format**:
   - JSON files in `benchmark_*_results/` directories
   - Contains `world_logs` array with agent logs
   - Contains `apps` array with final app states
   - Contains metadata and configuration

#### ❌ What's Missing for Historical Replay

1. **Insufficient State Capture**: Current traces only contain:
   - Final app states (not intermediate states)
   - ActionLog entries (not complete agent memory)
   - Basic metadata (not environment/tool state)

2. **No State Restoration**: No mechanism to:
   - Restore agent conversation memory (`self.logs`)
   - Restore environment state (time, event queue, etc.)
   - Restore tool/action executor contexts
   - Position replay at arbitrary points

3. **UI Limitations**: Current editing UI assumes:
   - Live session with in-memory state
   - Direct access to running environment
   - Real-time agent instances

## Implementation Plan

### Phase 1: Enhanced State Capture System

#### 1.1 Design State Snapshot Schema
**File**: `are/simulation/state_snapshots.py`

Create comprehensive data structures to capture complete execution state:

```python
@dataclass
class AgentSnapshot:
    agent_id: str
    logs: List[Dict[str, Any]]  # Serialized BaseAgentLog objects  
    conversation_history: List[Dict[str, str]]  # LLM message history
    current_iteration: int
    total_iterations: int
    action_executor_state: Dict[str, Any]
    tool_contexts: Dict[str, Any]
    timestamp: float

@dataclass
class EnvironmentSnapshot:
    current_time: float
    tick_count: int
    time_manager_state: Dict[str, Any]
    event_queue_state: Dict[str, Any]
    pending_events: List[Dict[str, Any]]
    environment_state: str  # EnvironmentState enum
    timestamp: float

@dataclass
class ExecutionSnapshot:
    log_index: int  # Position in world_logs
    agent_snapshot: AgentSnapshot
    environment_snapshot: EnvironmentSnapshot
    app_snapshots: Dict[str, Dict[str, Any]]
    tool_snapshots: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: float
```

#### 1.2 Implement State Capture Logic
**File**: `are/simulation/state_capture.py`

Create utilities to extract complete state from running simulation:

```python
class StateCapture:
    @staticmethod
    def capture_agent_state(agent) -> AgentSnapshot:
        """Extract complete agent internal state including conversation memory."""
        
    @staticmethod
    def capture_environment_state(env: Environment) -> EnvironmentSnapshot:
        """Extract complete environment state including time and event queues."""
        
    @staticmethod
    def capture_complete_snapshot(agent, env: Environment, log_index: int) -> ExecutionSnapshot:
        """Capture complete execution state at specific log position."""
```

Key implementation details:
- Serialize `agent.logs` list (conversation memory)
- Capture `agent.build_history_from_logs()` output (LLM context)
- Save `TimeManager` state and current time position
- Capture `EventQueue.future_events` and `EventLog.past_events`
- Serialize all app states using existing `get_state()` methods

#### 1.3 Integrate Capture into Execution Pipeline
**Files**: 
- `are/simulation/environment.py` (Environment.run() method)
- `are/simulation/main.py` (execution orchestration)

Add snapshot capture at strategic points:
- Before each agent action
- After each significant state change
- At user-defined checkpoint intervals

```python
# In Environment.run() or similar
if self.capture_snapshots:
    snapshot = StateCapture.capture_complete_snapshot(
        agent=self.agent, 
        env=self, 
        log_index=len(self.world_logs)
    )
    self.execution_trace.add_snapshot(snapshot)
```

#### 1.4 Enhanced Trace Format
**File**: Update existing trace export in `are/simulation/data_handler/exporter.py`

Extend JSON execution traces to include:
```json
{
    "world_logs": [...],  // Existing
    "apps": [...],        // Existing  
    "snapshots": [        // NEW
        {
            "log_index": 42,
            "agent_snapshot": {...},
            "environment_snapshot": {...},
            "app_snapshots": {...},
            "timestamp": 1729875600.0
        }
    ],
    "metadata": {...}     // Existing + snapshot metadata
}
```

### Phase 2: State Restoration System

#### 2.1 Create State Restoration Engine
**File**: `are/simulation/state_restoration.py`

```python
class StateRestoration:
    @staticmethod
    def restore_agent_state(agent, snapshot: AgentSnapshot) -> bool:
        """Restore agent to previous state including conversation memory."""
        
    @staticmethod
    def restore_environment_state(env: Environment, snapshot: EnvironmentSnapshot) -> bool:
        """Restore environment state including time and event queues."""
        
    @staticmethod
    def restore_complete_state(agent, env: Environment, snapshot: ExecutionSnapshot) -> bool:
        """Restore complete execution state from snapshot."""
```

Key restoration tasks:
- Rebuild `agent.logs` from serialized data
- Restore LLM conversation context
- Reset `TimeManager` to correct time position
- Restore `EventQueue` and `EventLog` states
- Restore all app states using `set_state()` methods
- Reset iteration counters and execution context

#### 2.2 Historical Replay Engine
**File**: `are/simulation/historical_replay.py`

```python
class HistoricalReplayer:
    def __init__(self, trace_file_path: str):
        """Load historical trace from disk."""
        
    def restore_to_log_index(self, log_index: int) -> bool:
        """Restore execution state to specific log position."""
        
    def replay_from_modified_log(self, log_index: int, modified_content: str) -> bool:
        """Modify log at position and replay from there."""
        
    def get_available_snapshots(self) -> List[int]:
        """Return log indices where snapshots are available."""
```

Algorithm for restoration:
1. Find nearest snapshot before target log_index
2. Restore state from snapshot
3. Replay logs sequentially from snapshot to target position
4. Apply any modifications to target log
5. Continue replay from modified state

#### 2.3 Integration with Existing Replay
**File**: `are/simulation/replay.py`

Extend existing replay functions to work with restored state:

```python
def replay_logs_from_state(
    world_logs: List[BaseAgentLog], 
    env: Environment,
    starting_state: Optional[ExecutionSnapshot] = None
) -> None:
    """Enhanced replay that can start from arbitrary state."""
    
    if starting_state:
        StateRestoration.restore_complete_state(agent, env, starting_state)
    
    # Continue with existing replay logic
    replay_logs(world_logs, env)
```

### Phase 3: Backend API Extensions

#### 3.1 New GraphQL Mutations
**File**: `are/simulation/gui/server/graphql/mutation.py`

Add historical replay mutations:

```python
@strawberry.mutation
def load_historical_trace(
    self, info, session_id: str, trace_file_path: str
) -> ScenarioForGraphQL:
    """Load historical execution trace for replay."""
    
@strawberry.mutation  
def historical_amend_and_replay(
    self, info, session_id: str, log_index: int, content: str
) -> ScenarioForGraphQL:
    """Modify historical log and replay from that point."""
    
@strawberry.mutation
def get_historical_snapshots(
    self, info, session_id: str
) -> List[HistoricalSnapshotInfo]:
    """Get available restoration points in loaded trace."""
```

#### 3.2 Session State Management
**File**: `are/simulation/gui/server/are_simulation_gui.py`

Extend session management to handle historical traces:

```python
class SimulationSession:
    def __init__(self):
        self.historical_replayer: Optional[HistoricalReplayer] = None
        self.is_historical_session: bool = False
        # ... existing fields
        
    def load_historical_trace(self, trace_path: str) -> bool:
        """Load and prepare historical trace for replay."""
        
    def historical_replay_modified_output(self, log_index: int, content: str) -> bool:
        """Handle historical log modification and replay."""
```

#### 3.3 File Management System
**File**: `are/simulation/trace_manager.py`

Create system to manage trace files:

```python
class TraceManager:
    @staticmethod
    def list_available_traces() -> List[TraceInfo]:
        """List all available historical traces."""
        
    @staticmethod
    def validate_trace_format(trace_path: str) -> bool:
        """Verify trace has required snapshot data."""
        
    @staticmethod
    def get_trace_metadata(trace_path: str) -> Dict[str, Any]:
        """Extract trace metadata for UI display."""
```

### Phase 4: Frontend UI Extensions

#### 4.1 Historical Trace Loader
**File**: `are/simulation/gui/client/src/components/historical/HistoricalTraceLoader.tsx`

Create UI component to load historical traces:

```tsx
interface HistoricalTraceLoaderProps {
    onTraceLoaded: (traceInfo: TraceInfo) => void;
}

export function HistoricalTraceLoader(props: HistoricalTraceLoaderProps) {
    // File browser/selector for trace files
    // Display trace metadata (date, scenario, model, etc.)
    // Load button to initialize historical session
}
```

#### 4.2 Enhanced Agent Logs View
**File**: `are/simulation/gui/client/src/components/agent-logs/AgentLogDetailView.tsx`

Extend existing edit functionality for historical traces:

```tsx
// Add props to distinguish live vs historical sessions
interface AgentLogDetailViewProps {
    log: AgentLog;
    isHistoricalSession?: boolean;
    onHistoricalEdit?: (logIndex: number, content: string) => void;
}

// Modify edit button behavior based on session type
const handleSaveContent = async () => {
    if (isHistoricalSession) {
        // Use historical replay mutation
        await historicalAmendAndReplay(logIndex, editedContent);
    } else {
        // Use existing live session logic
        await commitAmendAndReplay(sessionId, log?.id, editedContent, notify);
    }
};
```

#### 4.3 Session Type Indicator
**File**: `are/simulation/gui/client/src/components/session/SessionTypeIndicator.tsx`

Add UI indicator for session type:

```tsx
export function SessionTypeIndicator() {
    const { sessionType, traceInfo } = useSession();
    
    return (
        <Box>
            {sessionType === 'historical' ? (
                <Chip 
                    icon={<HistoryIcon />} 
                    label={`Historical: ${traceInfo.filename}`}
                    color="secondary"
                />
            ) : (
                <Chip 
                    icon={<PlayArrowIcon />} 
                    label="Live Session" 
                    color="primary"
                />
            )}
        </Box>
    );
}
```

#### 4.4 Snapshot Navigation
**File**: `are/simulation/gui/client/src/components/historical/SnapshotNavigation.tsx`

Create UI for navigating between restoration points:

```tsx
export function SnapshotNavigation() {
    const { availableSnapshots, currentLogIndex } = useHistoricalSession();
    
    return (
        <Box>
            <Typography variant="h6">Restoration Points</Typography>
            <Timeline>
                {availableSnapshots.map(snapshot => (
                    <TimelineItem key={snapshot.logIndex}>
                        <TimelineContent>
                            Log #{snapshot.logIndex} - {snapshot.timestamp}
                            <Button onClick={() => restoreToSnapshot(snapshot.logIndex)}>
                                Restore
                            </Button>
                        </TimelineContent>
                    </TimelineItem>
                ))}
            </Timeline>
        </Box>
    );
}
```

### Phase 5: Integration and Testing

#### 5.1 End-to-End Workflow Implementation
**Files**: Multiple integration points

Connect all components for complete workflow:

1. **Load Historical Trace**:
   - File selection → TraceManager validation → Backend loading → UI update

2. **Navigate to Log Position**:
   - UI log selection → Backend state restoration → Frontend refresh

3. **Modify and Replay**:
   - Edit log content → Historical replay mutation → State restoration → Replay execution

4. **View Results**:
   - Updated world logs → UI refresh → New execution state display

#### 5.2 Backwards Compatibility
Ensure changes don't break existing live session functionality:

- Keep existing mutations and APIs
- Add session type detection
- Maintain current UI behavior for live sessions
- Graceful degradation when snapshot data unavailable

#### 5.3 Performance Optimization
Address potential performance issues:

- **Lazy Snapshot Loading**: Only capture snapshots when needed
- **Compression**: Compress snapshot data in trace files
- **Streaming**: Stream large trace files rather than loading entirely
- **Caching**: Cache restored states for quick navigation

## Implementation Checklist

### Backend Core (Essential)
- [ ] `StateCapture` class with complete state extraction
- [ ] `ExecutionSnapshot` data structures and serialization
- [ ] `StateRestoration` class with state rebuilding
- [ ] `HistoricalReplayer` engine for trace loading and replay
- [ ] Enhanced trace format with snapshot data
- [ ] Integration points in execution pipeline

### Backend API (Essential)
- [ ] `load_historical_trace` GraphQL mutation
- [ ] `historical_amend_and_replay` GraphQL mutation  
- [ ] Session management for historical traces
- [ ] File management and validation utilities

### Frontend Core (Essential)
- [ ] Historical trace loader component
- [ ] Enhanced `AgentLogDetailView` with historical support
- [ ] Session type detection and UI adaptation
- [ ] Basic snapshot navigation

### Advanced Features (Optional)
- [ ] Performance optimizations (compression, caching)
- [ ] Advanced snapshot navigation UI
- [ ] Batch replay operations
- [ ] Trace comparison and diff tools
- [ ] Export functionality for modified traces

## Success Criteria

1. **Load Historical Trace**: Users can load execution traces from disk into the GUI
2. **Navigate to Any Log**: Users can click on any log in historical trace and view details
3. **Edit Historical Logs**: Users can modify log content using the same UI as live sessions
4. **Replay from Modification**: Modified logs trigger complete state restoration and replay
5. **View Results**: Users can see the new execution path after modification
6. **Performance**: Historical replay performs reasonably (< 30 seconds for typical traces)
7. **Compatibility**: Live session functionality remains unchanged

## Risk Mitigation

1. **State Complexity**: Start with core state (agent logs, app states) before adding advanced features
2. **Performance**: Implement incremental loading and optimization from the start
3. **Data Size**: Consider trace file size growth and implement compression early
4. **Backwards Compatibility**: Maintain existing APIs and add historical functionality as extensions
5. **Testing**: Create comprehensive test suite with known trace files and expected outcomes

This implementation will enable powerful "time travel debugging" capabilities for ARE simulations, allowing researchers and developers to experiment with different decision points in historical executions.