"""
Agent state capture for checkpoint functionality.

This module provides a simple function to capture agent state for checkpointing.
The most critical state is the agent logs, which contain the full chat history.

Restoration is handled separately in replay.py (for future replay functionality).
"""

import logging
from typing import Dict, Any, List
import time

from are.simulation.agents.are_simulation_agent import RunnableARESimulationAgent

logger = logging.getLogger(__name__)


def capture_agent_state(agent: RunnableARESimulationAgent) -> Dict[str, Any]:
    """
    Capture complete agent state for checkpointing.
    
    The agent state primarily consists of agent logs, which contain the full
    chat history needed to restore the agent's conversation context.
    
    Args:
        agent: The agent instance to capture state from
        
    Returns:
        Dictionary containing agent state needed for replay
    """
    try:
        agent_state = {
            "timestamp": time.time(),
            "agent_type": type(agent).__name__,
            "agent_class": f"{type(agent).__module__}.{type(agent).__name__}",
        }
        
        # Get the actual base agent that holds the logs
        # ARESimulationAgent wraps BaseAgent as self.react_agent
        base_agent = _get_base_agent(agent)
        
        if base_agent is None:
            agent_state["error"] = "Could not find base agent with logs"
            agent_state["logs"] = []
            agent_state["logs_count"] = 0
            return agent_state
        
        # Capture agent logs (THIS IS THE CHAT HISTORY - most critical for replay)
        logs = getattr(base_agent, 'logs', [])
        agent_state["logs"] = _capture_agent_logs(logs)
        agent_state["logs_count"] = len(logs)
        
        # Capture execution counters (needed to continue from correct iteration)
        agent_state["iterations"] = getattr(base_agent, 'iterations', 0)
        agent_state["planning_counter"] = getattr(base_agent, 'planning_counter', 0)
        
        # Capture current task
        agent_state["task"] = getattr(base_agent, 'task', "")
        
        # Capture custom state (any agent-specific state)
        custom_state = getattr(base_agent, 'custom_state', {})
        agent_state["custom_state"] = custom_state.copy() if custom_state else {}
        
        # Capture tool name mappings (for data augmentation reproducibility)
        tool_map = getattr(base_agent, 'original_tool_name_to_variant', {})
        agent_state["original_tool_name_to_variant"] = tool_map.copy() if tool_map else {}
        
        # Capture shuffle configuration (for reproducibility)
        agent_state["shuffle_seed"] = getattr(base_agent, 'shuffle_seed', None)
        agent_state["shuffle_tools"] = getattr(base_agent, 'shuffle_tools', False)
        
        # Capture agent ID for log correlation
        agent_state["agent_id"] = getattr(base_agent, 'agent_id', None)
        
        logger.debug(f"Captured agent state: {agent_state['logs_count']} logs, iteration {agent_state['iterations']}")
        
        return agent_state
        
    except Exception as e:
        logger.error(f"Failed to capture agent state: {e}", exc_info=True)
        return {
            "error": str(e),
            "timestamp": time.time(),
            "agent_type": type(agent).__name__ if agent else "unknown",
            "logs": [],
            "logs_count": 0
        }


def _get_base_agent(agent: Any) -> Any:
    """
    Get the underlying BaseAgent that holds the actual agent state (logs, etc).
    
    ARESimulationAgent wraps BaseAgent as self.react_agent.
    Other agent types may have different structures.
    """
    # Check if this IS a BaseAgent (has logs directly)
    if hasattr(agent, 'logs') and hasattr(agent, 'iterations'):
        return agent
    
    # ARESimulationAgent wraps BaseAgent as react_agent
    if hasattr(agent, 'react_agent'):
        return agent.react_agent
    
    # Try other common wrapper patterns
    if hasattr(agent, 'base_agent'):
        return agent.base_agent
    
    if hasattr(agent, 'agent'):
        return agent.agent
    
    return None


def _capture_agent_logs(logs: List) -> List[Dict[str, Any]]:
    """
    Serialize agent logs to dictionaries.
    
    These logs ARE the chat history - they contain:
    - SystemPromptLog: System prompts
    - TaskLog: User messages/tasks
    - LLMOutputThoughtActionLog: Agent responses
    - ObservationLog: Tool outputs
    - ErrorLog: Any errors encountered
    - etc.
    
    Args:
        logs: List of BaseAgentLog instances
        
    Returns:
        List of serialized log dictionaries
    """
    serialized_logs = []
    for log in logs:
        try:
            if hasattr(log, 'to_dict'):
                serialized_logs.append(log.to_dict())
            else:
                # Fallback: serialize manually
                logger.warning(f"Log type {type(log).__name__} doesn't have to_dict(), using manual serialization")
                serialized_logs.append({
                    "log_type": type(log).__name__,
                    "timestamp": getattr(log, 'timestamp', time.time()),
                    "agent_id": getattr(log, 'agent_id', None),
                    "error": "Manual serialization - may be incomplete"
                })
        except Exception as e:
            logger.warning(f"Failed to serialize log: {e}")
            serialized_logs.append({
                "error": str(e),
                "log_type": type(log).__name__
            })
    
    return serialized_logs
