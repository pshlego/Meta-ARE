# ARE (Meta Agents Research Environments) Codebase Architecture Guide

## Overview

The ARE (Meta Agents Research Environments) framework is a sophisticated system for testing and benchmarking AI agents in realistic environments. This document provides a comprehensive guide to understanding the codebase architecture, component relationships, and execution flow.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           ARE Framework                          │
├─────────────────────────────────────────────────────────────────┤
│  CLI Layer (Entry Points)                                      │
│  ├── benchmark.py (Benchmarking scenarios)                     │
│  └── main.py (Individual scenario execution)                   │
├─────────────────────────────────────────────────────────────────┤
│  Execution Layer                                               │
│  ├── MultiScenarioRunner (Orchestrates multiple scenarios)     │
│  ├── ScenarioRunner (Executes single scenarios)                │
│  └── ScenarioExecutor (Dataset processing)                     │
├─────────────────────────────────────────────────────────────────┤
│  Core Simulation Layer                                         │
│  ├── Environment (Event loop & app management)                 │
│  ├── Scenarios (Test definitions)                              │
│  ├── Agents (AI agent implementations)                         │
│  └── Apps (Simulated applications)                             │
├─────────────────────────────────────────────────────────────────┤
│  Support Layer                                                 │
│  ├── Types & Events (Core data structures)                     │
│  ├── Validation (Result scoring)                               │
│  ├── Data Handling (Export/import)                             │
│  └── Utilities (Helpers & tools)                               │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure Explained

```
are/
├── simulation/                          # Main simulation framework
│   ├── cli/                            # Command-line interface components
│   ├── agents/                         # AI agent implementations
│   ├── apps/                          # Simulated applications
│   ├── benchmark/                     # Benchmarking system
│   ├── core/                          # Core simulation engine
│   ├── data/                          # Data structures and management
│   ├── data_handler/                  # Import/export functionality
│   ├── environment.py                 # Core environment class
│   ├── scenarios/                     # Test scenario definitions
│   ├── tests/                         # Test suite
│   ├── types.py                       # Core type definitions
│   ├── utils/                         # Utility functions
│   └── validation/                    # Result validation system
```

## Core Components Deep Dive

### 1. Entry Points (CLI Layer)

#### `main.py` - Individual Scenario Runner
**File**: `are/simulation/main.py`
- **Purpose**: Run single scenarios or small sets of scenarios
- **Usage**: `python -m are.simulation.main --scenario-id <id>`
- **Key Features**:
  - Scenario registry lookup
  - JSON scenario file loading
  - HuggingFace dataset integration
  - Export functionality

#### `benchmark.py` - Benchmarking System
**File**: `are/simulation/benchmark.py` (entry point) and `are/simulation/benchmark/cli.py` (main implementation)
- **Purpose**: Run large-scale benchmarks across datasets
- **Usage**: `are-benchmark run --dataset <path> --config <config>`
- **Key Features**:
  - Dataset processing (local/HuggingFace)
  - Multi-run execution for variance reduction
  - Statistical reporting
  - Result validation and scoring

### 2. Execution Layer

#### `MultiScenarioRunner`
**File**: `are/simulation/multi_scenario_runner.py`
```python
# Orchestrates execution of multiple scenarios
class MultiScenarioRunner:
    def run_with_events(config, scenarios_iterator, progress_description)
    # - Manages concurrent execution (threading/multiprocessing)
    # - Handles progress tracking
    # - Aggregates results from multiple scenario runs
    # - Exports traces and statistics
```

#### `ScenarioRunner` 
**File**: `are/simulation/scenario_runner.py`
```python
# Executes a single scenario instance
class ScenarioRunner:
    def run(config: ScenarioRunnerConfig)
    # - Sets up Environment with apps
    # - Configures and launches Agent
    # - Monitors scenario execution
    # - Handles timeouts and errors
    # - Exports scenario results
```

#### `ScenarioExecutor` (in benchmark/)
**File**: `are/simulation/benchmark/scenario_executor.py`
```python
# Processes datasets and coordinates benchmark runs
def run_dataset(model, dataset_path, config, split, ...):
    # - Loads scenarios from datasets
    # - Preprocesses scenarios (judge config, validation)
    # - Multiplies scenarios for multiple runs
    # - Configures MultiScenarioRunner
    # - Generates final reports
```

### 3. Core Simulation Layer

#### `Environment` - The Simulation Engine
**File**: `are/simulation/environment.py`

The Environment is the heart of the simulation system:

```python
class Environment:
    # Core State
    current_time: float              # Simulation time
    state: EnvironmentState         # SETUP/RUNNING/PAUSED/STOPPED
    time_manager: TimeManager       # Time progression control
    
    # Event System
    event_log: EventLog            # History of completed events
    event_queue: EventQueue        # Pending events (priority queue)
    
    # App Management
    apps: dict[str, App]           # Registered applications
    notification_system: NotificationSystem  # Inter-app communication
    
    # Core Methods
    def start()                    # Begin event loop
    def register_apps(apps)        # Register app instances
    def schedule_event(event)      # Add event to queue
    def tick()                     # Process one time step
```

**Event Loop Behavior**:
- **Time-based**: Fixed time increments, process ready events
- **Queue-based**: Jump to next event time, process immediately
- **Oracle mode**: Replay predetermined events without agent

#### `Scenario` - Test Definitions
**File**: `are/simulation/scenarios/scenario.py`

Scenarios define what the agent should accomplish:

```python
class Scenario:
    # Identity & Config
    scenario_id: str               # Unique identifier
    seed: int                     # Reproducibility seed
    
    # Simulation Setup
    apps: list[App]               # Applications to register
    events: list[AbstractEvent]   # Scheduled events
    start_time: float             # When to begin
    duration: float               # How long to run
    
    # Test Definition
    tags: tuple[CapabilityTag]    # What abilities are tested
    hints: list[Hint]            # Guidance for the agent
    
    # Behavioral Config
    tool_augmentation_config      # Modify tool behavior
    env_events_config            # Add random events
```

**Scenario Types**:
- **Programmatic**: Defined in Python code
- **JSON-imported**: Loaded from JSON files
- **Benchmark**: From datasets like GAIA2

#### `Agent` - AI Agent System
**File**: `are/simulation/agents/are_simulation_agent.py`

Agents are the AI systems being tested:

```python
class RunnableARESimulationAgent:
    # Configuration
    agent_config: AgentConfig      # Behavioral parameters
    llm_engine_config: LLMEngineConfig  # Model settings
    
    # Runtime State  
    agent_log: list[BaseAgentLog]  # Conversation history
    current_turn: int             # Interaction progress
    
    # Core Loop
    def run(env: Environment)
    # 1. Receive task from environment
    # 2. Generate thoughts and plans
    # 3. Execute actions via tools
    # 4. Receive observations
    # 5. Continue until task complete or timeout
```

**Agent Types**:
- **DefaultAgent** (`are/simulation/agents/default_agent/`): ReAct-style reasoning agent
- **AppAgent** (`are/simulation/agents/app_agent/`): Agents that control specific applications
- **CustomAgents**: User-defined agent implementations

#### `Apps` - Simulated Applications
**File**: `are/simulation/apps/app.py`

Apps provide tools and services for agents:

```python
class App:
    # Identity
    name: str                     # App instance name
    
    # State Management
    def get_state() -> dict       # Export current state
    def load_state(state: dict)   # Restore state
    def reset()                   # Return to initial state
    
    # Tool Registration
    @app_tool                     # Decorator for agent-callable methods (are/simulation/tool_utils.py)
    def some_function(self, args) # Tools available to agent
```

**Common App Types**:
- **EmailApp** (`are/simulation/apps/email_app.py`): Email management and communication
- **FileSystemApp** (`are/simulation/apps/file_system_app.py`): File and directory operations
- **WebBrowserApp** (`are/simulation/apps/web_browser_app.py`): Web browsing and interaction
- **DatabaseApp** (`are/simulation/apps/database_app.py`): Data storage and retrieval
- **CalendarApp** (`are/simulation/apps/calendar_app.py`): Scheduling and time management
- **SystemApp** (`are/simulation/apps/system.py`): System-level operations

### 4. Event System

#### Event Types
**File**: `are/simulation/types.py`
```python
class EventType(Enum):
    AGENT = "AGENT"        # Actions initiated by agent
    ENV = "ENV"           # Environment/scenario events  
    CONDITION = "CONDITION"  # Conditional triggers
    VALIDATION = "VALIDATION"  # Result validation
    USER = "USER"         # User interactions
    STOP = "STOP"         # Termination events
```

#### Event Lifecycle
1. **Event Creation**: Events defined in scenarios or generated during execution
2. **Scheduling**: Events added to priority queue with execution time
3. **Processing**: Environment processes events when time is reached
4. **Completion**: Events moved to event log with results and metadata
5. **Dependencies**: Successor events may be scheduled based on results

#### Event Examples
**Files**: Event classes in `are/simulation/types.py`, Action class in `are/simulation/types.py`
```python
# Agent taking an action
Event(
    event_type=EventType.AGENT,
    action=Action(function=email_app.send_email, args={...}),
    event_time=current_time
)

# Environment notification
Event(
    event_type=EventType.ENV,
    action=Action(function=send_message_to_agent, args={"message": "You have mail"}),
    event_time=current_time + 300
)

# Validation check
ValidationEvent(
    condition=lambda env: check_email_sent(env),
    timeout=3600
)
```

## Data Flow Walkthrough

### 1. Benchmark Execution Flow
```
User Command:
are-benchmark run --dataset /path/to/gaia2 --config search

↓

`are/simulation/benchmark/cli.py:main()`
├── Validates parameters
├── Loads dataset scenarios  
└── Calls run_dataset()

↓

`are/simulation/benchmark/scenario_executor.py:run_dataset()`
├── Creates scenarios iterator from dataset
├── Preprocesses scenarios (judge config, etc.)
├── Multiplies scenarios for num_runs
├── Creates MultiScenarioRunnerConfig
└── Calls MultiScenarioRunner.run_with_events()

↓

MultiScenarioRunner.run_with_events()
├── Sets up concurrent execution (threads/processes)
├── For each scenario:
│   ├── Creates ScenarioRunner
│   └── Calls ScenarioRunner.run()
├── Aggregates results
└── Exports traces and statistics

↓

ScenarioRunner.run()
├── Creates Environment with EnvironmentConfig
├── Registers scenario apps to environment
├── Creates and configures Agent
├── Starts environment event loop
├── Agent interacts with environment via tools
├── Validates results when scenario completes
└── Returns ScenarioValidationResult
```

### 2. Single Scenario Execution Flow

#### Flow Control Overview

The execution is controlled by the **Environment's event loop** (`are/simulation/environment.py`), which orchestrates all interactions between the Agent, Apps, and Events. Here's how the control flows:

```
ScenarioRunner.run() (`are/simulation/scenario_runner.py`)
│
├── 1. Setup Phase
│   ├── Create Environment with EnvironmentConfig
│   ├── Register Apps to Environment
│   ├── Create and Configure Agent
│   └── Schedule initial Events from Scenario
│
├── 2. Agent Launch Phase  
│   ├── Start Environment.event_loop() in background thread
│   └── Launch Agent.run(environment) in main thread
│
└── 3. Execution Phase (Concurrent)
    ├── Environment Event Loop (Background Thread)
    └── Agent Execution Loop (Main Thread)
```

#### Detailed Event Loop Mechanics

**File**: `are/simulation/environment.py:Environment.join()` and `Environment.tick()`

```python
# The main event loop control
def join(self):
    """Main event loop - runs until stop condition"""
    while not self.should_stop():
        if self.queue_based_loop:
            self._queue_based_tick()  # Jump to next event time
        else:
            self._time_based_tick()   # Fixed time increments
            time.sleep(self.time_increment_in_seconds)

def _time_based_tick(self):
    """Process events ready at current time"""
    # 1. Get all events ready for current time
    ready_events = self.event_queue.get_ready_events(self.current_time)
    
    # 2. Process each event
    for event in ready_events:
        completed_event = self._execute_event(event)
        self.event_log.add(completed_event)
        
        # 3. Schedule successor events if any
        for successor in event.successors:
            self.schedule_event(successor)
    
    # 4. Advance simulation time
    self.current_time += self.time_increment_in_seconds
```

#### Entity Interactions Explained

##### **1. Environment (The Orchestrator)**
**File**: `are/simulation/environment.py`

The Environment is the central controller that:
- **Manages Time**: Controls simulation time progression
- **Processes Events**: Executes events when they're ready
- **Coordinates Communication**: Routes messages between Agent and Apps
- **Maintains State**: Tracks all system state changes

```python
class Environment:
    def schedule_event(self, event: Event):
        """Add event to priority queue for future execution"""
        self.event_queue.put(event)
    
    def _execute_event(self, event: Event) -> CompletedEvent:
        """Execute an event and return the result"""
        try:
            # Execute the event's action
            result = event.action.function(**event.action.args)
            return CompletedEvent(event, result, success=True)
        except Exception as e:
            return CompletedEvent(event, None, success=False, exception=e)
```

##### **2. Agent (The Decision Maker)**
**File**: `are/simulation/agents/are_simulation_agent.py`

The Agent runs in its own execution context and:
- **Receives Tasks**: Gets notifications from Environment
- **Plans Actions**: Decides what tools to use
- **Calls Tools**: Invokes App methods (creates AGENT events)
- **Processes Results**: Interprets tool outputs

```python
class RunnableARESimulationAgent:
    def run(self, env: Environment):
        """Main agent execution loop"""
        while not self.should_stop():
            # 1. Check for new messages/notifications
            messages = env.get_messages_for_agent()
            
            # 2. Process messages and plan next action
            action_plan = self.plan_next_action(messages)
            
            # 3. Execute planned action (calls app tool)
            if action_plan:
                result = self.execute_action(action_plan)
                
                # 4. Log the interaction
                self.agent_log.append(ActionLog(action_plan, result))
```

##### **3. Apps (The Service Providers)**
**File**: `are/simulation/apps/app.py` and specific app implementations

Apps provide tools and maintain state:
- **Tool Registration**: Methods decorated with @app_tool become available
- **State Management**: Apps track their internal state
- **Service Provision**: Execute specific functionality for agents

```python
class App:
    @app_tool(description="Send an email message")
    def send_email(self, to: str, subject: str, body: str) -> str:
        """Tool method callable by agent"""
        # This method execution is wrapped in an Event
        email = Email(to=to, subject=subject, body=body)
        self.sent_emails.append(email)
        return f"Email sent to {to}"
```

#### Complete Execution Example: Email Sending Scenario

Let's trace through a concrete example where an agent needs to send an email:

**Scenario Setup** (`scenario.json` or programmatic):
```python
class EmailScenario(Scenario):
    def initialize(self):
        self.apps = [EmailApp(name="email")]
        self.events = [
            # Initial task for agent
            Event(
                event_type=EventType.ENV,
                event_time=0,
                action=Action(
                    function=send_message_to_agent,
                    args={"message": "Send email to john@example.com about meeting"}
                )
            ),
            # Validation check
            ValidationEvent(
                event_time=300,  # Check after 5 minutes
                condition=lambda env: len(env.get_app("email").sent_emails) > 0
            )
        ]
```

**Step-by-Step Execution Flow with Event Enumeration**:

```
Time: 0.0 seconds - System Initialization
├── Environment.start() initializes
│   ├── EventQueue created (empty priority queue)
│   ├── EventLog created (empty history)
│   └── TimeManager initialized
├── EmailApp registered to environment
│   └── App tools registered for agent access
├── Agent.run(env) starts in parallel thread
└── [EVENT #1] Initial Event scheduled
    ├── Type: ENV (EventType.ENV)
    ├── Action: send_message_to_agent("Send email to john@example.com about meeting")
    ├── event_time: 0.0
    ├── event_id: "env_task_message_001"
    └── Status: SCHEDULED → Added to EventQueue

Time: 0.0 seconds - [EVENT #1] ENV Event Processing
├── Environment.tick() processes EventQueue
├── [EVENT #1] ENV Event executed:
│   ├── Type: ENV (EventType.ENV)
│   ├── Action: send_message_to_agent() function called
│   ├── Message delivered to Agent notification queue
│   └── Status: EXECUTING → COMPLETED
├── [COMPLETED EVENT #1] logged to EventLog:
│   ├── Type: ENV (EventType.ENV)
│   ├── Result: "Message delivered successfully"
│   ├── metadata.success: True
│   ├── metadata.execution_time: 0.001s
│   └── metadata.exception: None
└── Agent receives task notification in message queue

Time: 0.1 seconds - Agent Processing & Tool Call
├── Agent.plan_next_action() processes notification
├── Agent reasoning: "I need to send an email using the email tool"
├── Agent initiates tool call: env.get_app("email").send_email()
├── Tool call intercepted by Environment
└── [EVENT #2] Auto-generated from agent tool call:
    ├── Type: AGENT (EventType.AGENT) 
    ├── Action: email_app.send_email(to="john@example.com", subject="Meeting", body="...")
    ├── event_time: 0.1
    ├── event_id: "agent_email_action_002"
    └── Status: SCHEDULED → Added to EventQueue

Time: 0.1 seconds - [EVENT #2] AGENT Event Processing
├── Environment.tick() processes EventQueue
├── [EVENT #2] AGENT Event executed:
│   ├── Type: AGENT (EventType.AGENT)
│   ├── Action: EmailApp.send_email() method executed
│   ├── EmailApp internal operations:
│   │   ├── Email object created
│   │   ├── Added to self.sent_emails list
│   │   └── App state modified (is_state_modified = True)
│   └── Status: EXECUTING → COMPLETED
├── [COMPLETED EVENT #2] logged to EventLog:
│   ├── Type: AGENT (EventType.AGENT)
│   ├── Result: "Email sent to john@example.com"
│   ├── metadata.success: True
│   ├── metadata.tool_name: "EmailApp__send_email"
│   ├── metadata.execution_time: 0.05s
│   └── metadata.exception: None
└── Result returned to Agent

Time: 0.2 seconds - Agent State Update
├── Agent receives successful tool execution result
├── Agent.agent_log updated with:
│   ├── ActionLog(action="send_email", result="success")
│   ├── ObservationLog("Email sent successfully")
│   └── current_turn incremented
├── Agent state updated: task_completed = True
├── Agent continues monitoring for new notifications
└── No new events generated (agent waits)

Time: 300.0 seconds - [EVENT #3] Validation Event
├── [EVENT #3] Pre-scheduled validation triggered:
│   ├── Type: VALIDATION (EventType.VALIDATION)
│   ├── Action: ConditionCheckAction(lambda env: len(env.get_app("email").sent_emails) > 0)
│   ├── event_time: 300.0
│   ├── event_id: "validation_email_sent_003"
│   └── Status: SCHEDULED → EXECUTING
├── Environment.tick() processes ValidationEvent:
│   ├── Validation condition evaluated
│   ├── email_app.sent_emails = [Email(...)] (length = 1)
│   ├── Condition result: True (email was sent)
│   └── Status: EXECUTING → COMPLETED
├── [COMPLETED EVENT #3] logged to EventLog:
│   ├── Type: VALIDATION (EventType.VALIDATION)
│   ├── Result: True (validation passed)
│   ├── metadata.success: True
│   ├── metadata.validation_result: "PASS"
│   └── metadata.exception: None
├── Scenario marked as successful
├── [EVENT #4] Auto-generated scenario completion:
│   ├── Type: STOP (EventType.STOP)
│   ├── Action: StopEnvironment()
│   ├── event_time: 300.0
│   ├── event_id: "scenario_stop_004"
│   └── Triggers environment shutdown
└── Environment.stop() called, event loop terminates

Final Event Summary:
├── Total Events: 4 (1 ENV + 1 AGENT + 1 VALIDATION + 1 STOP)
├── All Events Completed Successfully
├── EventLog contains 4 CompletedEvent entries
├── Agent completed 1 tool call successfully
├── EmailApp final state: 1 email sent
└── Scenario result: SUCCESS (score = 1.0)
```

**Event Types Used in This Flow**:

1. **ENV Event** (`EventType.ENV`)
   - **Purpose**: Environment-initiated actions (task assignment, notifications)
   - **Example**: `send_message_to_agent()` - delivers initial task to agent
   - **Trigger**: Pre-defined in scenario or environment logic

2. **AGENT Event** (`EventType.AGENT`)
   - **Purpose**: Agent-initiated actions (tool calls, app interactions)
   - **Example**: `EmailApp.send_email()` - agent uses email tool
   - **Trigger**: Auto-generated when agent calls any app tool

3. **VALIDATION Event** (`EventType.VALIDATION`)
   - **Purpose**: Check if scenario objectives are met
   - **Example**: Check if email was sent successfully
   - **Trigger**: Pre-scheduled in scenario definition

4. **STOP Event** (`EventType.STOP`)
   - **Purpose**: Terminate simulation cleanly
   - **Example**: Environment shutdown after scenario completion
   - **Trigger**: Auto-generated when validation passes or timeout reached

#### Key Interaction Patterns

##### **1. Agent → App Communication** 
```python
# Agent calls app tool (automatic event creation)
result = env.get_app("email").send_email(
    to="john@example.com",
    subject="Meeting",  
    body="Let's meet tomorrow"
)

# Behind the scenes:
# 1. Tool call intercepted by environment
# 2. AGENT Event created and scheduled
# 3. Event processed: EmailApp.send_email() executed
# 4. Result returned to agent
```

##### **2. Environment → Agent Communication**
```python
# Environment sends notification to agent
env.send_message_to_agent("You have new email")

# Behind the scenes:
# 1. ENV Event created with message
# 2. Event processed when time reached
# 3. Message added to agent's notification queue
# 4. Agent picks up message in next iteration
```

##### **3. Event Dependencies and Successors**
```python
# Events can trigger other events
email_event = Event(
    action=Action(function=email_app.send_email, args={...}),
    successors=[
        # Send notification 5 minutes later
        Event(
            event_time=current_time + 300,
            action=Action(function=send_notification, args={...})
        )
    ]
)
```

#### Concurrency and Thread Safety

**File**: `are/simulation/scenario_runner.py:ScenarioRunner.run()`

```python
def run(self, config: ScenarioRunnerConfig):
    # 1. Setup environment and apps
    env = Environment(config.env_config)
    env.register_apps(scenario.apps)
    
    # 2. Start environment event loop in background
    env_thread = threading.Thread(target=env.start)
    env_thread.start()
    
    # 3. Run agent in main thread
    agent = create_agent(config.agent_config)
    try:
        agent.run(env)  # Agent execution
    finally:
        env.stop()      # Clean shutdown
        env_thread.join()
```

The **Environment** acts as the central coordinator, ensuring thread-safe communication between the Agent and Apps through its event system. This design enables:
- **Deterministic Replay**: All interactions captured as events
- **Concurrent Execution**: Agent and Environment run independently
- **State Tracking**: Complete system state maintained
- **Flexible Scenarios**: Complex multi-step interactions supported

## Configuration System

### Hierarchical Configuration
```
Command Line Args
    ↓
MultiScenarioRunnerConfig (`are/simulation/scenarios/config.py`)
    ├── Model settings (model, provider, endpoint)
    ├── Agent settings (agent_name, max_turns)  
    ├── Execution settings (timeout, concurrency)
    ├── Output settings (export, trace_dump_format)
    └── Augmentation settings (a2a, noise, tool_failures)
    
    ↓ (per scenario)
    
ScenarioRunnerConfig (`are/simulation/scenarios/config.py`)
    ├── Inherits from MultiScenarioRunnerConfig
    ├── Scenario-specific settings
    └── Environment configuration
    
    ↓
    
EnvironmentConfig (`are/simulation/environment.py`)
    ├── Time management (start_time, duration, increment)
    ├── Event loop behavior (queue_based, oracle_mode)
    └── Runtime settings (timeout, verbose)
```

## Validation and Scoring System

### Validation Events
Scenarios include validation events that check if the agent accomplished the task:

**File**: `are/simulation/types.py` (ValidationEvent class)
```python
ValidationEvent(
    event_type=EventType.VALIDATION,
    condition=lambda env: check_task_completion(env),
    success_message="Task completed successfully",
    failure_message="Task was not completed",
    timeout=3600  # Max time to wait for completion
)
```

### Judge System
For complex validations, ARE uses LLM-based judges:

```python
# Judge evaluates agent's work
judge_result = judge_model.evaluate(
    task_description=scenario.task,
    agent_actions=agent.get_actions(),
    final_state=environment.get_state(),
    success_criteria=scenario.success_criteria
)
```

### Scoring
- **Binary**: 0.0 (failure) or 1.0 (success)
- **Partial**: Scores between 0.0 and 1.0 based on completion
- **Custom**: Scenario-specific scoring functions

## Key Design Patterns

### 1. Event-Driven Architecture
- All interactions happen through events
- Decouples components for flexibility
- Enables replay and debugging

### 2. App-Tool Pattern
- Apps expose functionality via decorated methods
- Tools are automatically registered and discoverable
- Provides consistent interface for agents

### 3. Configuration Inheritance
- Settings cascade from global to scenario-specific
- Allows fine-grained control while maintaining defaults
- Supports experimentation and customization

### 4. State Management
- Apps maintain internal state
- Environment tracks all state changes
- Enables checkpointing and replay

### 5. Async/Concurrent Execution
- Supports multiple scenarios running in parallel
- Thread or process-based execution
- Progress tracking and result aggregation

## Extension Points

### Adding New Apps
**Base Class**: `are/simulation/apps/app.py`
```python
class MyCustomApp(App):
    @app_tool(description="Does something useful")
    def my_tool(self, param: str) -> str:
        # Implementation
        return result
    
    def get_state(self) -> dict:
        return {"my_data": self.data}
    
    def load_state(self, state: dict):
        self.data = state["my_data"]
```

### Adding New Agents
**Base Class**: `are/simulation/agents/are_simulation_agent.py`
```python
class MyCustomAgent(RunnableARESimulationAgent):
    def __init__(self, config: MyAgentConfig):
        # Initialize agent
    
    def run(self, env: Environment) -> None:
        # Agent execution loop
```

### Adding New Scenarios
**Base Class**: `are/simulation/scenarios/scenario.py`
```python
class MyScenario(Scenario):
    scenario_id = "my_test_scenario"
    
    def initialize(self):
        # Set up apps and events
        self.apps = [EmailApp(), FileSystemApp()]
        self.events = [
            # Define what should happen
        ]
```

## Common Debugging Approaches

### 1. Trace Analysis
**File**: `are/simulation/data_handler/exporter.py` (JsonScenarioExporter)
- Export full traces with `--export` flag
- Examine event sequences in JSON files
- Look for unexpected event ordering or missing events

### 2. Agent Logs
**File**: `are/simulation/agents/agent_log.py` (BaseAgentLog and subclasses)
- Agent logs contain full conversation history
- Check for reasoning errors or tool misuse
- Verify agent is receiving correct observations

### 3. App State Inspection
- Use `app.get_state()` to examine app internal state
- Check if state changes match expectations
- Verify tool calls had intended effects

### 4. Event Timeline
- Environment event log shows complete execution history
- Useful for understanding timing issues
- Can identify performance bottlenecks

This architecture enables ARE to provide a flexible, powerful framework for testing AI agents in realistic environments while maintaining reproducibility and debuggability.