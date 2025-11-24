"""
Event classification system for checkpoint triggers.

This module determines which events require checkpoints and when,
implementing the event-based checkpointing strategy while maintaining
minimal invasive design.
"""

from typing import Set, Optional
import logging

# Import ARE event types (we'll need to adjust this import based on actual ARE structure)
# For now, we'll define the event types we expect to work with
class EventType:
    """Event types that can trigger checkpoints."""
    ENV = "ENV"
    AGENT = "AGENT"  
    VALIDATION = "VALIDATION"
    STOP = "STOP"
    CONDITION = "CONDITION"
    USER = "USER"


class EventCheckpointClassifier:
    """
    Determines which events require checkpoints based on the event-based strategy.
    
    This classifier implements the core logic for identifying critical event
    boundaries where system state changes meaningfully and checkpoints should
    be created.
    """
    
    def __init__(self, checkpoint_events: Optional[Set[str]] = None, verbose: bool = False):
        """
        Initialize the event classifier.
        
        Args:
            checkpoint_events: Set of event types to checkpoint (defaults to critical events)
            verbose: Enable detailed logging for classification decisions
        """
        # Default to critical events if none specified
        if checkpoint_events is None:
            checkpoint_events = {EventType.ENV, EventType.AGENT, EventType.VALIDATION, EventType.STOP}
        
        self.checkpoint_events = checkpoint_events
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Track event statistics for optimization
        self.events_seen = 0
        self.checkpoints_triggered = 0
    
    def is_critical_event(self, event_type: str) -> bool:
        """
        Check if an event type is considered critical for checkpointing.
        
        Critical events are those that cause meaningful system state changes
        and represent important boundaries in the execution flow.
        
        Args:
            event_type: Type of event to check
            
        Returns:
            True if this event type should trigger checkpoints
        """
        is_critical = event_type in self.checkpoint_events
        
        if self.verbose:
            self.logger.debug(f"Event type '{event_type}' is {'critical' if is_critical else 'non-critical'}")
        
        return is_critical
    
    def should_checkpoint_before_event(self, event_type: str, event_id: Optional[str] = None,
                                     is_first_event: bool = False) -> bool:
        """
        Determine if a checkpoint should be created before an event executes.
        
        Before-event checkpoints capture the system state just prior to
        execution, enabling replay from that exact point.
        
        Args:
            event_type: Type of the event
            event_id: Optional event identifier for specific event logic
            is_first_event: Whether this is the first event of its type
            
        Returns:
            True if a before-event checkpoint should be created
        """
        self.events_seen += 1
        
        # Always checkpoint before critical events
        if not self.is_critical_event(event_type):
            return False
        
        # Always checkpoint before the first event of each critical type
        if is_first_event:
            if self.verbose:
                self.logger.info(f"Before-checkpoint triggered: first {event_type} event")
            self.checkpoints_triggered += 1
            return True
        
        # Checkpoint before all critical events for precise replay points
        checkpoint_decision = True
        
        if self.verbose:
            self.logger.info(f"Before-checkpoint triggered: {event_type} event {event_id}")
        
        if checkpoint_decision:
            self.checkpoints_triggered += 1
        
        return checkpoint_decision
    
    def should_checkpoint_after_event(self, event_type: str, event_id: Optional[str] = None,
                                    execution_success: bool = True,
                                    state_changed: bool = True) -> bool:
        """
        Determine if a checkpoint should be created after an event completes.
        
        After-event checkpoints capture the system state following execution,
        useful for understanding the effects of the event.
        
        Args:
            event_type: Type of the event that completed
            event_id: Optional event identifier  
            execution_success: Whether the event executed successfully
            state_changed: Whether the event caused significant state changes
            
        Returns:
            True if an after-event checkpoint should be created
        """
        # Only checkpoint after critical events
        if not self.is_critical_event(event_type):
            return False
        
        # Always checkpoint after successful state-changing critical events
        if execution_success and state_changed:
            if self.verbose:
                self.logger.info(f"After-checkpoint triggered: {event_type} event {event_id} (successful, state changed)")
            self.checkpoints_triggered += 1
            return True
        
        # Checkpoint after failed critical events for debugging
        if not execution_success:
            if self.verbose:
                self.logger.info(f"After-checkpoint triggered: {event_type} event {event_id} (failed execution)")
            self.checkpoints_triggered += 1
            return True
        
        return False
    
    def get_checkpoint_type_name(self, trigger_type: str, event_type: str) -> str:
        """
        Generate a standardized checkpoint type name.
        
        Args:
            trigger_type: When the checkpoint was triggered (before/after)
            event_type: Type of event that triggered it
            
        Returns:
            Standardized checkpoint type name
        """
        return f"{trigger_type}_{event_type.lower()}"
    
    def analyze_event_for_checkpoint(self, event_type: str, event_id: Optional[str] = None,
                                   execution_phase: str = "before",
                                   execution_success: bool = True,
                                   state_changed: bool = True,
                                   is_first_event: bool = False) -> dict:
        """
        Comprehensive analysis of whether an event should trigger a checkpoint.
        
        This is the main entry point for checkpoint decision-making, providing
        detailed analysis and reasoning for the checkpoint decision.
        
        Args:
            event_type: Type of event to analyze
            event_id: Identifier for specific event
            execution_phase: Whether analyzing before or after execution
            execution_success: Whether execution was successful (after phase only)
            state_changed: Whether execution changed system state (after phase only)
            is_first_event: Whether this is the first event of this type
            
        Returns:
            Dictionary with checkpoint decision and reasoning
        """
        analysis = {
            "event_type": event_type,
            "event_id": event_id,
            "execution_phase": execution_phase,
            "should_checkpoint": False,
            "checkpoint_type": None,
            "reasoning": []
        }
        
        # Determine if checkpoint should be created
        if execution_phase == "before":
            analysis["should_checkpoint"] = self.should_checkpoint_before_event(
                event_type, event_id, is_first_event
            )
        elif execution_phase == "after":
            analysis["should_checkpoint"] = self.should_checkpoint_after_event(
                event_type, event_id, execution_success, state_changed
            )
        
        # Generate checkpoint type if needed
        if analysis["should_checkpoint"]:
            analysis["checkpoint_type"] = self.get_checkpoint_type_name(execution_phase, event_type)
        
        # Add reasoning
        if not self.is_critical_event(event_type):
            analysis["reasoning"].append(f"Event type '{event_type}' is not in critical event set")
        else:
            analysis["reasoning"].append(f"Event type '{event_type}' is critical")
            
            if execution_phase == "before":
                if is_first_event:
                    analysis["reasoning"].append("First event of this type requires checkpoint")
                else:
                    analysis["reasoning"].append("Critical event requires before-checkpoint")
            
            elif execution_phase == "after":
                if not execution_success:
                    analysis["reasoning"].append("Failed execution requires checkpoint for debugging")
                elif state_changed:
                    analysis["reasoning"].append("State-changing execution requires checkpoint")
                else:
                    analysis["reasoning"].append("No state change, checkpoint not needed")
        
        return analysis
    
    def get_statistics(self) -> dict:
        """Get classifier statistics for monitoring and optimization."""
        return {
            "events_seen": self.events_seen,
            "checkpoints_triggered": self.checkpoints_triggered,
            "checkpoint_rate": (self.checkpoints_triggered / max(1, self.events_seen)) * 100,
            "checkpoint_events": list(self.checkpoint_events)
        }
    
    def reset_statistics(self) -> None:
        """Reset classifier statistics."""
        self.events_seen = 0
        self.checkpoints_triggered = 0
    
    @classmethod
    def create_for_testing(cls, minimal_events: bool = True) -> 'EventCheckpointClassifier':
        """
        Create a classifier optimized for testing scenarios.
        
        Args:
            minimal_events: If True, only checkpoint AGENT events to reduce volume
            
        Returns:
            Classifier configured for testing
        """
        if minimal_events:
            checkpoint_events = {EventType.AGENT}  # Only agent events for minimal testing
        else:
            checkpoint_events = {EventType.ENV, EventType.AGENT, EventType.VALIDATION}
        
        return cls(checkpoint_events=checkpoint_events, verbose=True)
    
    @classmethod  
    def create_for_production(cls) -> 'EventCheckpointClassifier':
        """
        Create a classifier optimized for production scenarios.
        
        Returns:
            Classifier configured for production use
        """
        checkpoint_events = {EventType.ENV, EventType.AGENT, EventType.VALIDATION, EventType.STOP}
        return cls(checkpoint_events=checkpoint_events, verbose=False)