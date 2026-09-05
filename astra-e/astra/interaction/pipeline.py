"""InteractionPipeline processing SceneObservations into InteractionEvents."""

from __future__ import annotations

import logging
from astra.contracts.interaction import InteractionEvent
from astra.contracts.perception import SceneObservation
from astra.events.bus import EventBus
from astra.interaction.analyzer import InteractionAnalyzer

logger = logging.getLogger("astra.interaction.pipeline")


class InteractionPipeline:
    """
    Consumes SceneObservations and outputs classified InteractionEvents.
    Publishes interaction events to the event bus.
    """

    def __init__(
        self,
        analyzer: InteractionAnalyzer | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.analyzer = analyzer or InteractionAnalyzer()
        self.event_bus = event_bus

    def process_observation(self, observation: SceneObservation) -> list[InteractionEvent]:
        """Process observation and emit detected interaction events."""
        events = self.analyzer.analyze(observation)

        if self.event_bus is not None:
            for event in events:
                self.event_bus.publish("interaction.event", event)

        return events

    def reset(self) -> None:
        """Reset internal analyzer state."""
        self.analyzer.reset()
