"""MultimodalNotifier: Routes astronaut alerts across GUI, audio/TTS, and event bus channels."""

from __future__ import annotations

import logging
from typing import Callable
from astra.contracts.assistance import AssistanceChannel, AssistanceEvent
from astra.contracts.system import EventTopic
from astra.events.bus import EventBus
from astra.assistance.tts import AudioAssistanceEngine

logger = logging.getLogger("astra.assistance.notifier")


class MultimodalNotifier:
    """
    Multimodal Notification Router (FR-016, FR-021).
    Dispatches assistance alerts to appropriate astronaut interface channels:
      - TTS: Spoken via AudioAssistanceEngine.
      - GUI: Pushed to registered UI callback listeners (e.g. WebSocket).
    """

    def __init__(
        self,
        event_bus: EventBus,
        tts_engine: AudioAssistanceEngine | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.tts = tts_engine
        self._gui_listeners: list[Callable[[AssistanceEvent], None]] = []
        self._history: list[AssistanceEvent] = []

        self.event_bus.subscribe(EventTopic.ASSISTANCE_ISSUED, self._on_assistance)

    def register_gui_listener(self, callback: Callable[[AssistanceEvent], None]) -> None:
        """Register UI subscriber for visual alerts."""
        self._gui_listeners.append(callback)

    def _on_assistance(self, event: AssistanceEvent) -> None:
        """Route assistance event across configured channels."""
        self._history.append(event)

        # 1. Spoken Audio Channel
        if AssistanceChannel.TTS in event.channels and self.tts:
            self.tts.process_assistance_event(event)

        # 2. Visual GUI Channel
        if AssistanceChannel.GUI in event.channels:
            for listener in self._gui_listeners:
                try:
                    listener(event)
                except Exception as e:
                    logger.error(f"Error dispatching assistance to GUI listener: {e}")

    @property
    def history(self) -> list[AssistanceEvent]:
        return list(self._history)
