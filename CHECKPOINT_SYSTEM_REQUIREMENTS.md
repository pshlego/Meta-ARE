# Checkpoint System Requirements for ARE Agent Benchmarking

## Overview

This document outlines the comprehensive requirements for implementing a checkpointing system for the ARE (Meta Agents Research Environments) agent benchmarking framework. The checkpointing system must capture the complete system state at key execution points to enable replay functionality from saved checkpoints.

**Testing Context**: The checkpoint implementation will be tested using the single scenario execution:
```bash
are-run -s scenario_find_image_file -a default --output_dir ./test_results
```

The current execution logs in `./test_results` serve as the baseline for understanding the scenario execution flow and validating checkpoint functionality.

## Design Principles

1. **Minimal Invasive Design**: All checkpoint/replay logic should be contained in a separate folder structure and use wrapper classes that inherit from key components and override specific functions to avoid contaminating existing code.

2. **Full System State Capture**: The system must capture ALL required information for complete system state reconstruction. Missing any component will break the replay functionality.

## System Architecture Analysis

Based on the analysis of the ARE codebase, the agent system execution follows this flow:

```
CLI (benchmark.py) → ScenarioExecutor.run_dataset() → MultiScenarioRunner.run_scenarios() → ScenarioRunner.run() → Environment.start() → Agent execution loop
```

## Required Checkpoint Information

### 1. Scenario-Level Information

#### 1.1 Scenario Configuration (`scenarios/scenario.py`)
- **scenario_id**: Unique identifier for the scenario
- **seed**: Random seed for reproducibility 
- **start_time**: Initial timestamp for the scenario
- **duration**: Maximum duration for scenario execution
- **run_number**: Which run this is (for multiple runs)
- **config**: Dataset configuration name (e.g., "search", "execution")
- **has_a2a_augmentation**: Whether Agent2Agent augmentation is enabled
- **status**: Current scenario status (Draft, Valid, etc.)
- **working_dir**: Working directory for file operations
- **time_increment_in_seconds**: Time increment between ticks
- **queue_based_loop**: Whether using queue-based or time-based event loop
- **tool_augmentation_config**: Configuration for tool behavior modification
- **env_events_config**: Configuration for environment event augmentation
- **gui_config**: GUI-specific configuration if applicable
- **augmentation_data**: Additional augmentation state data

#### 1.2 Scenario Events (`scenarios/scenario.py`)
- **events**: Complete list of AbstractEvent objects scheduled for the scenario
  - Event types: ENV, AGENT, CONDITION, VALIDATION, USER, STOP
  - Each event contains: event_time, event_id, action, dependencies, successors
- **apps**: List of all App instances involved in the scenario
- **tags**: CapabilityTag tuples describing scenario capabilities
- **hints**: List of Hint objects providing guidance
- **additional_system_prompt**: Extra system prompt information

### 2. Environment State (`environment.py`)

#### 2.1 Environment Configuration (`EnvironmentConfig`)
- **start_time**: Environment start time
- **duration**: Total environment duration
- **time_increment_in_seconds**: Time step increment
- **oracle_mode**: Whether running in oracle mode
- **dump_dir**: Directory for state dumps
- **exit_when_no_events**: Exit condition flag
- **queue_based_loop**: Event loop type
- **wait_for_user_input_timeout**: User input timeout
- **verbose**: Verbosity setting

#### 2.2 Environment Runtime State
- **current_time**: Current simulation time
- **state**: Environment state (SETUP, RUNNING, PAUSED, STOPPED, FAILED)
- **time_manager**: TimeManager instance with current time state
- **environment_type**: Type of environment (CLI, GUI, UNKNOWN)
- **notification_system**: Notification system state and configuration

#### 2.3 Event Management State
- **event_log**: Complete EventLog with all CompletedEvent entries
  - Each CompletedEvent contains: action, metadata, tool_name, execution results
- **event_queue**: EventQueue with pending events (PriorityQueue)
- **events_to_schedule**: Events waiting to be scheduled
- **event_loop_active**: Whether the event loop is currently running

#### 2.4 App Registration State
- **apps**: Dictionary mapping app names to App instances
- **app_protocols**: Dictionary of Protocol implementations
- **app_connections**: Connections between apps via protocols

### 3. Agent State (`agents/are_simulation_agent.py`)

#### 3.1 Agent Configuration
- **agent_name**: Name/type of the agent being used
- **agent_config**: Complete RunnableARESimulationAgentConfig
  - LLM engine configuration (model, provider, endpoint)
  - System prompt and behavioral parameters
  - Maximum iterations and turn limits
  - Simulated generation time configuration

#### 3.2 Agent Runtime State
- **agent_log**: Complete BaseAgentLog history with all interactions
  - SystemPromptLog, TaskLog, LLMInputLog, LLMOutputLog entries
  - Tool calls, observations, errors, thoughts, plans
  - Token usage statistics and timing information
- **current_turn**: Current turn number in the interaction
- **max_turns**: Maximum allowed turns
- **agent_id**: Unique agent identifier

#### 3.3 LLM Engine State
- **conversation_history**: Complete conversation context
- **token_usage**: Cumulative token usage statistics
  - prompt_tokens, completion_tokens, total_tokens, reasoning_tokens
  - duration_per_call for timing analysis
- **generation_cache**: Any cached generation results (if caching enabled)

### 4. Application State (`apps/app.py`)

For each App instance in the scenario:

#### 4.1 App Configuration
- **name**: App instance name
- **class_name**: App class type
- **seed**: App-specific random seed
- **failure_probability**: Probability of tool failures (if augmented)

#### 4.2 App Runtime State
- **app_state**: Complete internal state via get_state() method
- **is_state_modified**: Whether the app state has been modified
- **time_manager**: TimeManager instance reference
- **rng**: Random number generator state
- **tool_registries**: Registered tools by type (APP, USER, ENV, DATA)
- **add_event_callbacks**: Event callback registrations

#### 4.3 App-Specific State
Each app type has specific state that must be captured:
- **File system state**: For file-based apps (current directory, file contents)
- **Email state**: For email apps (mailbox contents, sent items)
- **Database state**: For database apps (table contents, connection state)
- **Web state**: For web apps (session state, cookies, DOM state)
- **System state**: For system apps (process state, environment variables)

### 5. Runner Configuration State

#### 5.1 MultiScenarioRunnerConfig
- **model**: Model name being used
- **model_provider**: Model provider (azure, openai, etc.)
- **endpoint**: Custom endpoint URL if applicable
- **agent**: Agent type being used
- **oracle**: Oracle mode flag
- **export**: Export flag for trace generation
- **output_dir**: Output directory for results
- **trace_dump_format**: Format for trace dumps (hf, lite, both)
- **max_concurrent_scenarios**: Concurrency limit
- **executor_type**: Executor type (thread, process)
- **enable_caching**: Caching enabled flag
- **judge_only**: Offline validation mode flag
- **timeout_seconds**: Scenario timeout limit

#### 5.2 A2A (Agent2Agent) Configuration
- **a2a_app_prop**: Proportion of apps to run as agents
- **a2a_app_agent**: Agent type for app agents
- **a2a_model**: Model for app agents
- **a2a_model_provider**: Provider for app agent models
- **a2a_endpoint**: Endpoint for app agent models

#### 5.3 Validation Configuration
- **judge_engine_config**: Judge model configuration
  - judge_model, judge_provider, judge_endpoint
- **simulated_generation_time_mode**: Time simulation mode
- **log_level**: Logging level configuration

### 6. Execution Context State

#### 6.1 Process/Thread State
- **execution_mode**: Whether running in thread or process mode
- **worker_pool_state**: State of concurrent execution pool
- **progress_tracking**: Current progress through scenario list
- **scenarios_iterator_state**: Position in scenario iteration

#### 6.2 Random State Management
- **global_seed**: Global random seed for reproducibility
- **scenario_specific_seeds**: Per-scenario seed derivatives
- **app_specific_seeds**: Per-app seed derivatives
- **rng_states**: Current state of all random number generators

#### 6.3 Timing and Performance State
- **execution_start_time**: When execution began
- **scenario_start_times**: Start time for each scenario
- **performance_metrics**: Timing and resource usage metrics
- **timeout_tracking**: Timeout states for long-running operations

### 7. Validation and Scoring State

#### 7.1 Validation Configuration
- **validation_events**: List of ValidationEvent objects
- **validation_conditions**: Condition checks for scenario completion
- **scoring_functions**: Functions used for result scoring
- **judge_model_config**: Configuration for validation models

#### 7.2 Results State
- **scenario_results**: Current ScenarioValidationResult objects
- **validation_scores**: Current validation scores
- **completion_status**: Whether scenarios have completed successfully
- **exception_state**: Any exceptions encountered during execution

## Checkpoint Trigger Strategy: Event-Based Approach

Given that the ARE system is fundamentally **event-driven**, and long idle periods (agent thinking, waiting) don't change system state, **pure event-based checkpointing** is the optimal strategy. Checkpoints should be created only at **critical event boundaries** where meaningful state transitions occur.

### Core Checkpointing Strategy

**Primary Approach**: **Event-Boundary Checkpoints**
- Checkpoints triggered before/after critical events that change system state
- Aligns with ARE's event-driven architecture (`are/simulation/types.py`)
- Eliminates wasteful checkpoints during idle periods (agent reasoning, timeouts)
- Ensures every checkpoint represents a meaningful system state transition

**Implementation Pattern**:
```python
# In CheckpointableEnvironment wrapper
def _execute_event(self, event: Event) -> CompletedEvent:
    # Pre-event checkpoint for critical events
    if self._is_critical_event(event):
        self.checkpoint_manager.create_checkpoint("before_" + event.event_type.value, event.event_time)
    
    # Execute original event
    completed_event = super()._execute_event(event)
    
    # Post-event checkpoint if state significantly changed
    if self._state_changed_significantly(completed_event):
        self.checkpoint_manager.create_checkpoint("after_" + event.event_type.value, event.event_time)
    
    return completed_event
```

### Critical Event Types for Checkpointing

#### **1. Mandatory Checkpoint Events (Always Capture)**

```python
class CriticalCheckpointEvents:
    """Events that always trigger checkpoint creation"""
    
    # System Lifecycle Events
    SCENARIO_INITIALIZATION = "scenario_init"      # Before first event execution
    SCENARIO_COMPLETION = "scenario_complete"      # After final validation
    
    # Agent State Transition Events  
    AGENT_TASK_RECEIVED = "agent_task_received"    # ENV→Agent message delivery
    AGENT_TURN_START = "agent_turn_start"         # Agent begins reasoning cycle
    AGENT_TOOL_CALL_START = "tool_call_start"     # Before AGENT event execution
    AGENT_TOOL_CALL_COMPLETE = "tool_call_complete" # After AGENT event completion
    AGENT_TURN_COMPLETE = "agent_turn_complete"   # Agent completes reasoning cycle
    
    # Environment State Events
    ENV_EVENT_START = "env_event_start"           # Before ENV event execution
    ENV_EVENT_COMPLETE = "env_event_complete"     # After ENV event completion
    
    # Validation Milestone Events
    VALIDATION_START = "validation_start"         # Before VALIDATION event
    VALIDATION_COMPLETE = "validation_complete"   # After VALIDATION result
    
    # Error and Recovery Events
    EXCEPTION_OCCURRED = "exception_occurred"     # On any event exception
    TIMEOUT_TRIGGERED = "timeout_triggered"      # On scenario timeout
```

#### **2. Event Classification for Checkpointing**

```python
class EventCheckpointClassifier:
    """Determines which events require checkpoints"""
    
    def is_critical_event(self, event: AbstractEvent) -> bool:
        """Check if event always requires checkpoint"""
        critical_types = {
            EventType.ENV,         # Environment actions (task delivery, notifications)
            EventType.AGENT,       # Agent actions (tool calls, app interactions)  
            EventType.VALIDATION,  # Scenario success/failure checks
            EventType.STOP        # Scenario termination
        }
        return event.event_type in critical_types
    
    def requires_pre_checkpoint(self, event: AbstractEvent) -> bool:
        """Events that need checkpoint before execution"""
        return (
            event.event_type in [EventType.ENV, EventType.AGENT, EventType.VALIDATION] or
            self._is_first_event_of_type(event) or
            self._has_complex_dependencies(event)
        )
    
    def requires_post_checkpoint(self, completed_event: CompletedEvent) -> bool:
        """Events that need checkpoint after execution"""
        return (
            completed_event.metadata.success and (
                completed_event.event_type == EventType.AGENT or  # Tool calls change app state
                completed_event.event_type == EventType.VALIDATION or  # Validation affects scenario
                completed_event.metadata.exception is not None  # Errors need capture
            )
        )
```

### Event-Based Checkpoint Data Structure

#### **Checkpoint Metadata (Event-Driven)**
```python
@dataclass
class EventCheckpointMetadata:
    checkpoint_id: str              # Unique checkpoint identifier
    scenario_id: str               # Associated scenario
    simulation_time: float         # Simulation time when event occurred
    wall_clock_time: float         # Real-world timestamp
    checkpoint_type: str           # Event boundary type (before_agent/after_env/etc)
    trigger_event_id: str          # ID of event that triggered checkpoint
    trigger_event_type: str        # Type of triggering event (ENV/AGENT/VALIDATION)
    
    # Event context
    event_sequence_number: int     # Sequential event number in scenario
    events_completed_count: int    # Total events processed so far
    events_pending_count: int      # Events remaining in queue
    
    # System state snapshot  
    environment_state: str         # RUNNING/PAUSED/etc
    agent_turn: int               # Current agent turn number
    app_states_modified: list[str] # Apps with modified state since last checkpoint
    
    # Checkpoint lineage
    previous_checkpoint_id: str    # Previous checkpoint in event sequence
    checkpoint_sequence: int       # Sequential checkpoint number
```

#### **Event-Synchronized State Capture**
```python
class EventSynchronizedCheckpoint:
    """Captures system state at specific event boundaries"""
    
    def __init__(self, trigger_event: AbstractEvent, checkpoint_type: str):
        self.trigger_event = trigger_event
        self.checkpoint_type = checkpoint_type  # "before_event" or "after_event"
        self.simulation_time = trigger_event.event_time
        self.captured_at = time.time()
    
    def capture_environment_at_event(self, env: Environment) -> dict:
        """Capture environment state at event boundary"""
        return {
            'current_time': env.current_time,
            'state': env.state.value,
            'time_manager_state': env.time_manager.get_state(),
            
            # Event system state at this event
            'event_log_up_to_event': self._get_event_log_snapshot(env),
            'event_queue_state': env.event_queue.serialize_state(),
            'current_event': self.trigger_event.to_dict() if self.checkpoint_type == "before_event" else None,
            
            # App registry state
            'registered_apps': list(env.apps.keys()),
            'app_protocols_state': env._get_app_protocols_state(),
            'notification_system_state': env.notification_system.get_state(),
        }
    
    def capture_agent_at_event(self, agent: RunnableARESimulationAgent) -> dict:
        """Capture agent state at event boundary"""
        return {
            'agent_config': agent.agent_config.get_model_dump(),
            'current_turn': agent.current_turn,
            
            # Agent log filtered to event boundary
            'agent_log': self._get_agent_log_up_to_event(agent),
            'conversation_context': agent._get_conversation_context(),
            'token_usage_cumulative': agent._get_cumulative_token_usage(),
            
            # Agent processing state
            'current_task': agent.get_current_task(),
            'pending_actions': agent.get_pending_actions(),
        }
    
    def _get_event_log_snapshot(self, env: Environment) -> list:
        """Get event log entries up to (but not including) current event"""
        if self.checkpoint_type == "before_event":
            # Exclude current event (not yet executed)
            return [e.to_dict() for e in env.event_log.get_events_before_time(self.simulation_time)]
        else:
            # Include current event (just executed)  
            return [e.to_dict() for e in env.event_log.get_events_up_to_time(self.simulation_time)]
```

#### **Event-Based Storage Structure**
```
checkpoints/
├── {scenario_id}/
│   ├── event_timeline.json            # Chronological event and checkpoint sequence
│   ├── checkpoint_manifest.json       # Index of all checkpoints with metadata
│   ├── event_{sequence}_before/       # Checkpoint before event execution
│   │   ├── metadata.json              # Event context and checkpoint info
│   │   ├── trigger_event.json         # Complete triggering event data
│   │   ├── environment_snapshot.json  # Environment state at event boundary
│   │   ├── agent_snapshot.json        # Agent state at event boundary
│   │   ├── apps_snapshot/             # App states at event boundary
│   │   │   ├── {app_name}_state.json
│   │   │   └── ...
│   │   └── execution_context.json     # Runner config, RNG states
│   ├── event_{sequence}_after/        # Checkpoint after event completion  
│   │   ├── metadata.json              # Event results and state changes
│   │   ├── completed_event.json       # CompletedEvent with results
│   │   ├── environment_snapshot.json  # Environment state after execution
│   │   ├── agent_snapshot.json        # Agent state after execution
│   │   ├── apps_snapshot/             # App states after execution
│   │   └── state_diff.json            # Changes from before→after (optional)
│   └── recovery_checkpoints/          # Error and exception checkpoints
│       ├── exception_{event_id}/
│       └── timeout_{timestamp}/
```

#### **Example Event Timeline Structure**
```json
{
  "scenario_id": "scenario_find_image_file",
  "total_events": 4,
  "total_checkpoints": 8,
  "timeline": [
    {
      "sequence": 0,
      "simulation_time": 0.0,
      "checkpoint_type": "scenario_initialization",
      "checkpoint_id": "init_0",
      "description": "Initial scenario setup before any events"
    },
    {
      "sequence": 1, 
      "simulation_time": 0.0,
      "event_type": "ENV",
      "event_id": "env_task_message_001",
      "checkpoints": {
        "before": "event_1_before",
        "after": "event_1_after" 
      },
      "description": "Task message delivery to agent"
    },
    {
      "sequence": 2,
      "simulation_time": 15.3,
      "event_type": "AGENT", 
      "event_id": "agent_reasoning_002",
      "checkpoints": {
        "before": "event_2_before"
      },
      "description": "Agent begins reasoning cycle"
    },
    {
      "sequence": 3,
      "simulation_time": 45.8,
      "event_type": "AGENT",
      "event_id": "agent_file_search_003", 
      "checkpoints": {
        "before": "event_3_before",
        "after": "event_3_after"
      },
      "description": "Agent tool call: file system search for image file"
    },
    {
      "sequence": 4,
      "simulation_time": 300.0,
      "event_type": "VALIDATION",
      "event_id": "validation_check_004",
      "checkpoints": {
        "before": "event_4_before", 
        "after": "event_4_after"
      },
      "description": "Validation: check if image file was found"
    },
    {
      "sequence": 5,
      "simulation_time": 300.1,
      "checkpoint_type": "scenario_completion",
      "checkpoint_id": "complete_5", 
      "description": "Final scenario completion state"
    }
  ]
}
```

#### **4. CLI Integration for Event-Based Checkpoints**

```bash
# Enable event-based checkpointing for single scenario
are-run -s scenario_find_image_file -a default \
  --output_dir ./test_results \
  --enable-checkpoints \
  --checkpoint-on-events "ENV,AGENT,VALIDATION" \
  --checkpoint-before-events \
  --checkpoint-after-events \
  --checkpoint-dir ./checkpoints

# Resume from specific event checkpoint
are-run resume \
  --checkpoint-id "scenario_find_image_file/event_3_after" \
  --continue-execution

# Resume from before specific event
are-run resume \
  --scenario-id "scenario_find_image_file" \
  --resume-before-event 3 \
  --continue-execution

# List event timeline with checkpoints  
are-run list-checkpoints \
  --scenario-id "scenario_find_image_file" \
  --show-event-timeline \
  --format table

# Debug specific event
are-run debug-event \
  --checkpoint-id "scenario_find_image_file/event_3_before" \
  --inspect-state \
  --compare-with "event_3_after"
```

#### **5. Event Timeline Navigation**

```python
# are/simulation/checkpoint/event_timeline.py
class EventTimeline:
    """Manages chronological organization of events and checkpoints"""
    
    def __init__(self, scenario_id: str):
        self.scenario_id = scenario_id
        self.timeline_entries = []  # Sorted by simulation_time
        
    def add_event(self, event_id: str, event_type: str, simulation_time: float, 
                  description: str = None):
        """Add event to timeline"""
        entry = {
            'type': 'event',
            'event_id': event_id,
            'event_type': event_type,
            'simulation_time': simulation_time,
            'description': description,
            'checkpoints': {}
        }
        self.timeline_entries.append(entry)
        self._sort_timeline()
    
    def add_checkpoint(self, checkpoint_id: str, simulation_time: float, 
                      checkpoint_type: str, associated_event_id: str = None):
        """Add checkpoint to timeline"""
        if associated_event_id:
            # Link to existing event
            event_entry = self._find_event_entry(associated_event_id)
            if event_entry:
                if checkpoint_type.startswith('before_'):
                    event_entry['checkpoints']['before'] = checkpoint_id
                elif checkpoint_type.startswith('after_'):
                    event_entry['checkpoints']['after'] = checkpoint_id
        else:
            # Standalone checkpoint (scenario init/completion)
            entry = {
                'type': 'checkpoint',
                'checkpoint_id': checkpoint_id,
                'checkpoint_type': checkpoint_type,
                'simulation_time': simulation_time,
            }
            self.timeline_entries.append(entry)
            self._sort_timeline()
    
    def get_checkpoint_before_event(self, event_sequence: int) -> str:
        """Get checkpoint ID captured before specific event"""
        event_entry = self.timeline_entries[event_sequence]
        return event_entry.get('checkpoints', {}).get('before')
    
    def get_checkpoint_after_event(self, event_sequence: int) -> str:  
        """Get checkpoint ID captured after specific event"""
        event_entry = self.timeline_entries[event_sequence]
        return event_entry.get('checkpoints', {}).get('after')
    
    def get_timeline_summary(self) -> str:
        """Generate human-readable event and checkpoint timeline"""
        timeline = []
        for entry in self.timeline_entries:
            if entry['type'] == 'event':
                line = f"Time {entry['simulation_time']:6.1f}s: [EVENT] {entry['event_type']}: {entry.get('description', entry['event_id'])}"
                timeline.append(line)
                
                # Add checkpoint info
                checkpoints = entry.get('checkpoints', {})
                if 'before' in checkpoints:
                    timeline.append(f"                    [CHECKPOINT] before → {checkpoints['before']}")
                if 'after' in checkpoints:
                    timeline.append(f"                    [CHECKPOINT] after → {checkpoints['after']}")
            else:
                line = f"Time {entry['simulation_time']:6.1f}s: [CHECKPOINT] {entry['checkpoint_type']} → {entry['checkpoint_id']}"
                timeline.append(line)
                
        return "\n".join(timeline)
```

### Benefits of Event-Based Approach

#### **1. Perfect Alignment with ARE Architecture**
- **Event-Driven Design**: Checkpoints naturally align with system's event boundaries
- **State Transition Focus**: Captures state only when meaningful changes occur
- **No Wasted Storage**: Eliminates checkpoints during idle periods (agent thinking)

#### **2. Optimal Storage Efficiency**
- **Minimal Checkpoints**: Only at critical event boundaries
- **Contextual Relevance**: Every checkpoint represents a meaningful system state
- **Clear Recovery Points**: Resume execution at logical event boundaries

#### **3. Enhanced Debugging and Development**
- **Event-Level Granularity**: Debug specific event execution
- **Before/After Comparison**: Compare state changes caused by events
- **Logical Breakpoints**: Set resume points at natural system boundaries

#### **4. Deterministic Replay**
- **Event Ordering Preserved**: Exact event sequence maintained in checkpoints
- **State Consistency**: Environment and Agent state synchronized at event boundaries
- **Random State Control**: RNG states captured at deterministic points

### Example Usage Scenarios

```python
# Resume just before agent tool call
checkpoint_data = replay_manager.load_checkpoint_before_event("scenario_find_image_file", 3)
restored_runner = replay_manager.resume_from_event_checkpoint(checkpoint_data)

# Compare system state before and after validation
before_state = replay_manager.load_checkpoint_before_event("scenario_find_image_file", 5)
after_state = replay_manager.load_checkpoint_after_event("scenario_find_image_file", 5)
state_diff = compare_checkpoint_states(before_state, after_state)

# Debug scenario by examining each event boundary
timeline = event_timeline.get_timeline_summary("scenario_find_image_file")
for event_seq in range(timeline.event_count):
    checkpoint = replay_manager.load_checkpoint_before_event("scenario_find_image_file", event_seq)
    analyze_system_state_at_checkpoint(checkpoint)
```

This event-based approach provides the most **logical and efficient checkpointing strategy** for the ARE system, creating checkpoints only when the system state meaningfully changes through event execution, while providing complete replay functionality and excellent debugging capabilities.

## Implementation Strategy

### Minimal Invasive Design Architecture

#### **1. Event-Based Wrapper Classes**

```python
# are/simulation/checkpoint/checkpointable_environment.py
class CheckpointableEnvironment(Environment):
    """Wrapper that adds event-based checkpointing to Environment without modification"""
    
    def __init__(self, base_environment: Environment, checkpoint_config: CheckpointConfig):
        self._base_env = base_environment
        self.checkpoint_manager = EventCheckpointManager(checkpoint_config)
        self.event_classifier = EventCheckpointClassifier()
        self.checkpoint_sequence = 0
    
    def _execute_event(self, event: Event) -> CompletedEvent:
        """Override event execution to add checkpoint triggers"""
        
        # Pre-event checkpoint for critical events
        if self.event_classifier.requires_pre_checkpoint(event):
            checkpoint_id = self._create_event_checkpoint("before", event)
            self.checkpoint_sequence += 1
        
        # Execute original event processing
        completed_event = self._base_env._execute_event(event)
        
        # Post-event checkpoint if state changed significantly
        if self.event_classifier.requires_post_checkpoint(completed_event):
            checkpoint_id = self._create_event_checkpoint("after", event, completed_event)
            self.checkpoint_sequence += 1
            
        return completed_event
    
    def _create_event_checkpoint(self, checkpoint_type: str, event: Event, 
                               completed_event: CompletedEvent = None) -> str:
        """Create checkpoint at event boundary"""
        checkpoint_type_name = f"{checkpoint_type}_{event.event_type.value.lower()}"
        
        return self.checkpoint_manager.create_event_checkpoint(
            checkpoint_type=checkpoint_type_name,
            trigger_event=event,
            completed_event=completed_event,
            sequence_number=self.checkpoint_sequence
        )

# are/simulation/checkpoint/checkpointable_scenario_runner.py  
class CheckpointableScenarioRunner(ScenarioRunner):
    """Wrapper that adds event-based checkpointing to ScenarioRunner"""
    
    def run(self, config: ScenarioRunnerConfig) -> ScenarioValidationResult:
        # Wrap environment with checkpointing capability
        base_env = Environment(config.env_config)
        checkpointable_env = CheckpointableEnvironment(base_env, config.checkpoint_config)
        
        # Create initial scenario checkpoint before any events
        checkpointable_env.checkpoint_manager.create_scenario_checkpoint(
            "scenario_initialization", 0.0
        )
        
        # Continue with modified config
        modified_config = config.copy()
        modified_config.environment = checkpointable_env
        
        try:
            result = super().run(modified_config)
            
            # Create final completion checkpoint
            checkpointable_env.checkpoint_manager.create_scenario_checkpoint(
                "scenario_completion", checkpointable_env.current_time
            )
            
            return result
            
        except Exception as e:
            # Create error checkpoint for debugging
            checkpointable_env.checkpoint_manager.create_error_checkpoint(
                "scenario_exception", checkpointable_env.current_time, e
            )
            raise
```

#### **2. Event-Based Checkpoint Manager**

```python
# are/simulation/checkpoint/event_checkpoint_manager.py
class EventCheckpointManager:
    """Manages creation, storage, and retrieval of event-based checkpoints"""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.event_timeline = EventTimeline()
        self.storage_backend = CheckpointStorage(config.checkpoint_dir)
        
    def create_event_checkpoint(self, checkpoint_type: str, trigger_event: Event,
                              completed_event: CompletedEvent = None, 
                              sequence_number: int = None) -> str:
        """Create checkpoint at event boundary"""
        
        # Generate checkpoint ID based on event
        checkpoint_id = f"{trigger_event.event_type.value.lower()}_{sequence_number}_{checkpoint_type}"
        
        # Capture synchronized state at event boundary
        event_checkpoint = EventSynchronizedCheckpoint(trigger_event, checkpoint_type)
        checkpoint_data = {
            'metadata': self._build_checkpoint_metadata(checkpoint_id, trigger_event, checkpoint_type),
            'trigger_event': trigger_event.to_dict(),
            'completed_event': completed_event.to_dict() if completed_event else None,
            'environment': event_checkpoint.capture_environment_at_event(self.environment),
            'agent': event_checkpoint.capture_agent_at_event(self.agent),
            'apps': {name: app.get_state() for name, app in self.environment.apps.items()},
            'execution_context': self._capture_execution_context(),
        }
        
        # Save checkpoint
        self.storage_backend.save_checkpoint(checkpoint_id, checkpoint_data)
        
        # Update timeline
        self.event_timeline.add_checkpoint(
            checkpoint_id, trigger_event.event_time, checkpoint_type, trigger_event.event_type
        )
        
        return checkpoint_id
    
    def create_scenario_checkpoint(self, checkpoint_type: str, simulation_time: float) -> str:
        """Create scenario lifecycle checkpoint (init/completion)"""
        checkpoint_id = f"{checkpoint_type}_{int(simulation_time * 1000)}"
        
        checkpoint_data = {
            'metadata': self._build_scenario_metadata(checkpoint_id, checkpoint_type, simulation_time),
            'environment': self._capture_full_environment_state(),
            'agent': self._capture_full_agent_state(),
            'apps': {name: app.get_state() for name, app in self.environment.apps.items()},
            'scenario': self._capture_scenario_state(),
        }
        
        self.storage_backend.save_checkpoint(checkpoint_id, checkpoint_data)
        return checkpoint_id

# are/simulation/checkpoint/event_replay_manager.py
class EventReplayManager:
    """Handles loading event checkpoints and resuming execution"""
    
    def load_checkpoint_before_event(self, scenario_id: str, event_sequence: int) -> dict:
        """Load checkpoint captured before specific event"""
        checkpoint_id = f"event_{event_sequence}_before"
        return self.storage_backend.load_checkpoint(scenario_id, checkpoint_id)
    
    def load_checkpoint_after_event(self, scenario_id: str, event_sequence: int) -> dict:
        """Load checkpoint captured after specific event completion"""
        checkpoint_id = f"event_{event_sequence}_after" 
        return self.storage_backend.load_checkpoint(scenario_id, checkpoint_id)
    
    def resume_from_event_checkpoint(self, checkpoint_data: dict) -> CheckpointableScenarioRunner:
        """Resume execution from event boundary checkpoint"""
        
        # Restore environment at event boundary
        env = self._restore_environment_from_event_checkpoint(checkpoint_data['environment'])
        
        # Restore agent state at event boundary
        agent = self._restore_agent_from_event_checkpoint(checkpoint_data['agent'])
        
        # Restore app states 
        for app_name, app_state in checkpoint_data['apps'].items():
            env.get_app(app_name).load_state(app_state)
        
        # Restore event system state
        self._restore_event_system_state(env, checkpoint_data)
        
        # Create runner ready to continue from this event boundary
        return CheckpointableScenarioRunner(env, agent)
    
    def get_event_timeline(self, scenario_id: str) -> EventTimeline:
        """Get chronological timeline of events and checkpoints"""
        return EventTimeline.load_from_scenario(scenario_id)
```

#### **3. Example Event Checkpoint Timeline**
Based on the single scenario example:

```
Time: 0.0s   - [CHECKPOINT] scenario_initialization (critical)
Time: 0.0s   - [EVENT] ENV: send_message_to_agent  
Time: 0.1s   - [CHECKPOINT] agent_task_received (critical)
Time: 15.3s  - [EVENT] AGENT: agent starts reasoning
Time: 15.3s  - [CHECKPOINT] agent_turn_start (critical)  
Time: 45.8s  - [EVENT] AGENT: tool call file_system.find_file
Time: 45.8s  - [CHECKPOINT] tool_call_start (critical)
Time: 46.1s  - [CHECKPOINT] tool_call_complete (critical)
Time: 60.2s  - [EVENT] AGENT: agent completes turn
Time: 60.2s  - [CHECKPOINT] agent_turn_complete (critical)
Time: 300.0s - [EVENT] VALIDATION: check image file found
Time: 300.0s - [CHECKPOINT] validation_start (critical)
Time: 300.1s - [CHECKPOINT] validation_complete (critical)
Time: 300.1s - [CHECKPOINT] scenario_completion (critical)
```

**Checkpoint File Structure for this Timeline:**
```
checkpoints/scenario_find_image_file/
├── event_timeline.json
├── checkpoint_manifest.json
├── scenario_initialization/
├── event_1_before/        # Before ENV: send_message_to_agent
├── event_1_after/         # After ENV: agent_task_received  
├── event_2_before/        # Before AGENT: reasoning starts
├── event_3_before/        # Before AGENT: file search tool call
├── event_3_after/         # After AGENT: file search completed
├── event_4_before/        # Before AGENT: turn completion
├── event_4_after/         # After AGENT: turn completed
├── event_5_before/        # Before VALIDATION: check file found
├── event_5_after/         # After VALIDATION: check completed
└── scenario_completion/    # Final scenario state
```
```