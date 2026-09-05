"""Thread-safe in-memory Event Bus with wildcard routing and error isolation."""

from __future__ import annotations

import fnmatch
import logging
import threading
from collections import deque
from typing import Any, Callable

logger = logging.getLogger("astra.events.bus")


class EventBus:
    """
    In-memory Publish-Subscribe Event Bus for ASTRA-E subsystems.
    Decouples producers (cameras, AI models, procedure engine) from consumers
    (GUI, TTS, loggers, WebSocket streams).
    """

    def __init__(self, max_history: int = 500) -> None:
        self._subscribers: dict[str, list[Callable[[Any], None]]] = {}
        self._lock = threading.RLock()
        self._history: deque[tuple[str, Any]] = deque(maxlen=max_history)

    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        """
        Subscribe a handler callback to a topic or topic pattern (e.g. 'action.*', '*').
        """
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            if handler not in self._subscribers[topic]:
                self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        """Unsubscribe a handler from a topic."""
        with self._lock:
            if topic in self._subscribers:
                if handler in self._subscribers[topic]:
                    self._subscribers[topic].remove(handler)
                if not self._subscribers[topic]:
                    del self._subscribers[topic]

    def publish(self, topic: str, event: Any) -> int:
        """
        Publish an event to matching topic subscribers.
        Topic matching supports exact strings and glob patterns (e.g. 'action.*').
        Returns count of successfully invoked handlers.
        """
        with self._lock:
            self._history.append((topic, event))
            # Collect matching handlers under lock
            handlers_to_call: list[Callable[[Any], None]] = []
            for sub_topic, handlers in self._subscribers.items():
                if sub_topic == topic or fnmatch.fnmatch(topic, sub_topic):
                    handlers_to_call.extend(handlers)

        # Invoke handlers outside lock with exception isolation
        invoked_count = 0
        for handler in handlers_to_call:
            try:
                handler(event)
                invoked_count += 1
            except Exception as e:
                logger.error(
                    f"Error in event handler {handler} for topic '{topic}': {e}",
                    exc_info=True,
                )

        return invoked_count

    def get_history(self, limit: int | None = None) -> list[tuple[str, Any]]:
        """Retrieve recent published events."""
        with self._lock:
            events = list(self._history)
            if limit is not None:
                return events[-limit:]
            return events

    def clear(self) -> None:
        """Clear all subscribers and history."""
        with self._lock:
            self._subscribers.clear()
            self._history.clear()
