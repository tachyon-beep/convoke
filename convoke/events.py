"""
Event System for Convoke.

This module provides a simple event system to facilitate communication between
different modules in the Convoke system without creating direct dependencies.
It follows the publish-subscribe pattern where modules can publish events and
subscribe to events from other modules.
"""

import logging
from typing import Dict, List, Callable, Any, Optional


class EventBus:
    """
    A simple event bus that allows modules to publish and subscribe to events.

    The EventBus maintains a registry of event types and their subscribers.
    When an event is published, all subscribers for that event type are notified.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the event bus with an empty registry."""
        self._subscribers: Dict[str, List[Callable]] = {}
        self.logger = logger or logging.getLogger(__name__)

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribe to an event type with a callback function.

        Args:
            event_type: The type of event to subscribe to
            callback: The function to call when the event is published
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        self._subscribers[event_type].append(callback)
        self.logger.debug(f"Subscribed to event: {event_type}")

    def unsubscribe(self, event_type: str, callback: Callable) -> bool:
        """
        Unsubscribe from an event type.

        Args:
            event_type: The type of event to unsubscribe from
            callback: The callback function to remove

        Returns:
            True if the callback was successfully unsubscribed, False otherwise
        """
        if event_type not in self._subscribers:
            return False

        try:
            self._subscribers[event_type].remove(callback)
            self.logger.debug(f"Unsubscribed from event: {event_type}")
            return True
        except ValueError:
            return False

    def publish(self, event_type: str, **event_data) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event_type: The type of event to publish
            **event_data: Data to pass to the subscribers
        """
        if event_type not in self._subscribers:
            self.logger.debug(f"No subscribers for event: {event_type}")
            return

        self.logger.debug(f"Publishing event: {event_type} with data: {event_data}")
        for callback in self._subscribers[event_type]:
            try:
                callback(**event_data)
            except Exception as e:
                self.logger.error(f"Error in event subscriber for {event_type}: {e}")


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus(logger: Optional[logging.Logger] = None) -> EventBus:
    """
    Get the global event bus instance.

    This function creates a singleton instance of the EventBus if one doesn't exist.

    Args:
        logger: Optional logger for the event bus

    Returns:
        The global EventBus instance
    """
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus(logger)
    return _event_bus


# Constants for common event types
class EventTypes:
    """Constants for common event types used throughout the system."""

    # Workflow events
    WORKFLOW_STARTED = "workflow:started"
    WORKFLOW_COMPLETED = "workflow:completed"
    WORKFLOW_ERROR = "workflow:error"

    # Level events
    LEVEL_STARTED = "level:started"
    LEVEL_COMPLETED = "level:completed"
    LEVEL_ERROR = "level:error"

    # Task events
    TASK_CREATED = "task:created"
    TASK_STARTED = "task:started"
    TASK_COMPLETED = "task:completed"
    TASK_ERROR = "task:error"

    # Artifact events
    ARTIFACT_CREATED = "artifact:created"
    ARTIFACT_UPDATED = "artifact:updated"
    ARTIFACT_ACCESSED = "artifact:accessed"

    # Module events
    MODULE_CREATED = "module:created"
    MODULE_UPDATED = "module:updated"

    # Refinement events
    REFINEMENT_ITERATION_STARTED = "refinement:iteration:started"
    REFINEMENT_ITERATION_COMPLETED = "refinement:iteration:completed"

    # Agent decision events
    AGENT_DECISION = "agent:decision"
