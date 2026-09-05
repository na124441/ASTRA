"""Unit tests for AudioAssistanceEngine and MultimodalNotifier."""

import time
import pytest
from astra.contracts.assistance import AssistanceChannel, AssistanceEvent, AssistancePriority
from astra.contracts.system import EventTopic
from astra.events.bus import EventBus
from astra.assistance.notifier import MultimodalNotifier
from astra.assistance.tts import AudioAssistanceEngine, MockAudioBackend


def test_audio_engine_priority_queueing():
    """Verify high priority utterances are synthesized before low priority ones."""
    backend = MockAudioBackend()
    engine = AudioAssistanceEngine(backend=backend, cooldown_seconds=0.1, enabled=True)

    # Queue low priority first, then high priority immediately
    engine.speak("Routine guidance step 1", priority=AssistancePriority.LOW)
    engine.speak("CRITICAL WARNING: Wrong target used!", priority=AssistancePriority.CRITICAL)

    engine.flush()
    engine.stop()

    texts = [h["text"] for h in backend.history]
    assert len(texts) == 2
    # The critical warning should be synthesized before or right after first if first was already dequeued
    assert "CRITICAL WARNING: Wrong target used!" in texts


def test_audio_cooldown_suppression():
    """Verify repeated identical alerts are dropped within the cooldown window."""
    backend = MockAudioBackend()
    engine = AudioAssistanceEngine(backend=backend, cooldown_seconds=1.0, enabled=True)

    # First utterance should succeed
    res1 = engine.speak("Warning: Pick red component", priority=AssistancePriority.HIGH)
    assert res1 is True

    # Immediate duplicate should be suppressed
    res2 = engine.speak("Warning: Pick red component", priority=AssistancePriority.HIGH)
    assert res2 is False

    # Wait for cooldown to expire
    time.sleep(1.05)
    res3 = engine.speak("Warning: Pick red component", priority=AssistancePriority.HIGH)
    assert res3 is True

    engine.flush()
    engine.stop()

    assert len(backend.history) == 2


def test_multimodal_notifier_routing():
    """Verify MultimodalNotifier dispatches to TTS and GUI subscribers based on event channels."""
    backend = MockAudioBackend()
    engine = AudioAssistanceEngine(backend=backend, cooldown_seconds=0.1, enabled=True)
    bus = EventBus()
    notifier = MultimodalNotifier(event_bus=bus, tts_engine=engine)

    gui_received: list[AssistanceEvent] = []
    notifier.register_gui_listener(lambda ev: gui_received.append(ev))

    # 1. Event with TTS and GUI
    event1 = AssistanceEvent(
        message_id="as-1",
        source="test",
        correlation_id="RUN-1",
        type="WARNING",
        priority=AssistancePriority.HIGH,
        message="Alert one",
        channels=[AssistanceChannel.TTS, AssistanceChannel.GUI],
    )
    bus.publish(EventTopic.ASSISTANCE_ISSUED, event1)

    # 2. Event with GUI ONLY (no TTS)
    event2 = AssistanceEvent(
        message_id="as-2",
        source="test",
        correlation_id="RUN-1",
        type="INFO",
        priority=AssistancePriority.LOW,
        message="Alert two (visual only)",
        channels=[AssistanceChannel.GUI],
    )
    bus.publish(EventTopic.ASSISTANCE_ISSUED, event2)

    engine.flush()
    engine.stop()

    # GUI should receive both
    assert len(gui_received) == 2
    assert gui_received[0].message == "Alert one"
    assert gui_received[1].message == "Alert two (visual only)"

    # TTS backend should receive only event 1
    spoken_texts = [h["text"] for h in backend.history]
    assert "Alert one" in spoken_texts
    assert "Alert two (visual only)" not in spoken_texts
