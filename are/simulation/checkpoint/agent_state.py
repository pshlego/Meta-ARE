"""
Agent state capture and restoration for checkpoint/replay functionality.

This module provides functionality to capture and restore complete agent state,
including agent logs, execution history, and internal state for full replay capability.
"""

import logging
from typing import Dict, Any, Optional, List
import time

from are.simulation.agents.agent_log import BaseAgentLog
from are.simulation.agents.are_simulation_agent import RunnableARESimulationAgent

logger = logging.getLogger(__name__)


class AgentStateCapture:
    """
    Captures complete agent state for checkpointing.
    
    This class handles serialization of agent logs, execution state,
    and configuration to enable full agent state restoration.
    """
    
    @staticmethod
    def capture_agent_state(agent: RunnableARESimulationAgent) -> Dict[str, Any]:
        """
        Capture complete agent state.
        
        Args:
            agent: The agent instance to capture state from
            
        Returns:
            Dictionary containing complete agent state
        """
        try:
            agent_state = {
                "timestamp": time.time(),
                "agent_type": type(agent).__name__,
                "agent_class": f"{type(agent).__module__}.{type(agent).__name__}",
                "agent_id": getattr(agent, 'agent_id', None),
            }
            
            # Capture agent logs (execution history)
            if hasattr(agent, 'logs'):
                logs = getattr(agent, 'logs', [])
                agent_state["logs"] = AgentStateCapture._capture_agent_logs(logs)
                agent_state["logs_count"] = len(logs)
            else:
                agent_state["logs"] = []
                agent_state["logs_count"] = 0
            
            # Capture execution counters
            if hasattr(agent, 'iterations'):
                agent_state["iterations"] = getattr(agent, 'iterations', 0)
            if hasattr(agent, 'planning_counter'):
                agent_state["planning_counter"] = getattr(agent, 'planning_counter', 0)
            
            # Capture current task and attachments
            if hasattr(agent, 'task'):
                agent_state["task"] = getattr(agent, 'task', None)
            if hasattr(agent, 'attachments'):
                attachments = getattr(agent, 'attachments', [])
                agent_state["attachments"] = AgentStateCapture._serialize_attachments(attachments)
            
            # Capture custom state
            if hasattr(agent, 'custom_state'):
                custom_state = getattr(agent, 'custom_state', {})
                agent_state["custom_state"] = custom_state.copy() if custom_state else {}
            
            # Capture tool mappings (for data augmentation)
            if hasattr(agent, 'original_tool_name_to_variant'):
                tool_map = getattr(agent, 'original_tool_name_to_variant', {})
                agent_state["original_tool_name_to_variant"] = tool_map.copy() if tool_map else {}
            
            # Capture action executor state
            if hasattr(agent, 'action_executor'):
                action_executor = getattr(agent, 'action_executor', None)
                if action_executor:
                    agent_state["action_executor_state"] = AgentStateCapture._capture_action_executor_state(
                        action_executor
                    )
            
            # Capture initialization state
            if hasattr(agent, 'initialized'):
                agent_state["initialized"] = getattr(agent, 'initialized', False)
            if hasattr(agent, 'system_prompt'):
                agent_state["system_prompt"] = getattr(agent, 'system_prompt', None)
            if hasattr(agent, 'system_prompts'):
                agent_state["system_prompts"] = getattr(agent, 'system_prompts', None)
            
            # Capture shuffle configuration
            if hasattr(agent, 'shuffle_seed'):
                agent_state["shuffle_seed"] = getattr(agent, 'shuffle_seed', None)
            if hasattr(agent, 'shuffle_tools'):
                agent_state["shuffle_tools"] = getattr(agent, 'shuffle_tools', False)
            if hasattr(agent, 'shuffle_authorized_imports'):
                agent_state["shuffle_authorized_imports"] = getattr(agent, 'shuffle_authorized_imports', False)
            
            # Capture tool information
            if hasattr(agent, 'tools'):
                tools = getattr(agent, 'tools', {})
                agent_state["tools_info"] = {
                    "tool_count": len(tools) if tools else 0,
                    "tool_names": list(tools.keys()) if tools else []
                }
            
            logger.debug(f"Captured agent state: {agent_state['logs_count']} logs, iteration {agent_state.get('iterations', 0)}")
            
            return agent_state
            
        except Exception as e:
            logger.error(f"Failed to capture agent state: {e}", exc_info=True)
            return {
                "error": str(e),
                "timestamp": time.time(),
                "agent_type": type(agent).__name__ if agent else "unknown"
            }
    
    @staticmethod
    def _capture_agent_logs(logs: List[BaseAgentLog]) -> List[Dict[str, Any]]:
        """
        Serialize agent logs to dictionaries.
        
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
    
    @staticmethod
    def _serialize_attachments(attachments: Any) -> List[Dict[str, Any]]:
        """Serialize attachments."""
        try:
            if not attachments:
                return []
            
            serialized = []
            for attachment in attachments:
                if hasattr(attachment, 'to_dict'):
                    serialized.append(attachment.to_dict())
                else:
                    # Try basic serialization
                    serialized.append({
                        "type": type(attachment).__name__,
                        "str_repr": str(attachment)
                    })
            return serialized
        except Exception as e:
            logger.warning(f"Failed to serialize attachments: {e}")
            return []
    
    @staticmethod
    def _capture_action_executor_state(action_executor: Any) -> Dict[str, Any]:
        """Capture action executor state."""
        try:
            executor_state: Dict[str, Any] = {
                "executor_type": type(action_executor).__name__,
            }
            
            # Capture internal state
            if hasattr(action_executor, 'state'):
                # Don't serialize PIL images or other non-serializable objects
                state = getattr(action_executor, 'state', {})
                state_keys = list(state.keys()) if state else []
                executor_state["state_keys"] = state_keys
                executor_state["state_count"] = len(state_keys)
            
            # Capture tool information
            if hasattr(action_executor, 'tools'):
                tools = getattr(action_executor, 'tools', None)
                executor_state["tools_count"] = len(tools) if tools else 0
            
            return executor_state
        except Exception as e:
            logger.warning(f"Failed to capture action executor state: {e}")
            return {"error": str(e)}


class AgentStateRestorer:
    """
    Restores agent state from checkpoint data.
    """
    
    @staticmethod
    def restore_agent_logs(logs_data: List[Dict[str, Any]]) -> List[BaseAgentLog]:
        """
        Restore agent logs from serialized data.
        
        Args:
            logs_data: List of serialized log dictionaries
            
        Returns:
            List of BaseAgentLog instances
        """
        restored_logs = []
        
        for log_data in logs_data:
            try:
                # Use BaseAgentLog.from_dict() to reconstruct logs
                if "error" in log_data:
                    logger.warning(f"Skipping log with error: {log_data.get('error')}")
                    continue
                
                log = BaseAgentLog.from_dict(log_data)
                restored_logs.append(log)
            except Exception as e:
                logger.warning(f"Failed to restore log: {e}")
                continue
        
        logger.info(f"Restored {len(restored_logs)} agent logs from checkpoint")
        return restored_logs
    
    @staticmethod
    def get_initial_agent_logs_for_replay(agent_checkpoint_data: Dict[str, Any]) -> List[BaseAgentLog]:
        """
        Extract agent logs for passing to agent initialization.
        
        This is used to restore agent state by passing initial_agent_logs
        to the agent's run_scenario() method.
        
        Args:
            agent_checkpoint_data: Agent state from checkpoint
            
        Returns:
            List of BaseAgentLog instances for initialization
        """
        logs_data = agent_checkpoint_data.get("logs", [])
        return AgentStateRestorer.restore_agent_logs(logs_data)
    
    @staticmethod
    def validate_agent_state(agent_checkpoint_data: Dict[str, Any]) -> bool:
        """
        Validate that agent checkpoint data is complete and valid.
        
        Args:
            agent_checkpoint_data: Agent state from checkpoint
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ["agent_type", "logs"]
        
        for key in required_keys:
            if key not in agent_checkpoint_data:
                logger.error(f"Agent checkpoint missing required key: {key}")
                return False
        
        if "error" in agent_checkpoint_data:
            logger.error(f"Agent checkpoint contains error: {agent_checkpoint_data['error']}")
            return False
        
        return True


# Convenience functions
def capture_agent_state(agent: RunnableARESimulationAgent) -> Dict[str, Any]:
    """Convenience function to capture agent state."""
    return AgentStateCapture.capture_agent_state(agent)


def restore_agent_logs(checkpoint_data: Dict[str, Any]) -> List[BaseAgentLog]:
    """Convenience function to restore agent logs from checkpoint."""
    agent_data = checkpoint_data.get("agent", {})
    return AgentStateRestorer.get_initial_agent_logs_for_replay(agent_data)
