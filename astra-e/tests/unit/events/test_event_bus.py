"""Unit tests for thread-safe in-memory EventBus."""

from astra.events.bus import EventBus


def test_event_bus_publish_subscribe():
    """Verify subscription and publication on exact topic matches."""
    bus = EventBus()
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe("action.confirmed", handler)
    bus.publish("action.confirmed", {"action": "PICK"})

    assert len(received) == 1
    assert received[0]["action"] == "PICK"


def test_event_bus_wildcard_matching():
    """Verify wildcard pattern matching on topic hierarchy."""
    bus = EventBus()
    action_events = []
    all_events = []

    bus.subscribe("action.*", lambda e: action_events.append(e))
    bus.subscribe("*", lambda e: all_events.append(e))

    bus.publish("action.observed", {"id": 1})
    bus.publish("action.confirmed", {"id": 2})
    bus.publish("procedure.transitioned", {"id": 3})

    assert len(action_events) == 2
    assert len(all_events) == 3


def test_event_bus_exception_isolation():
    """Verify that an error in one handler does not halt execution of other handlers."""
    bus = EventBus()
    successful = []

    def broken_handler(event):
        raise RuntimeError("Simulated crash in handler")

    def healthy_handler(event):
        successful.append(event)

    bus.subscribe("test.topic", broken_handler)
    bus.subscribe("test.topic", healthy_handler)

    invoked_count = bus.publish("test.topic", {"status": "ok"})
    # Only healthy_handler completed successfully
    assert invoked_count == 1
    assert len(successful) == 1


def test_event_bus_unsubscribe():
    """Verify unsubscription stops delivery."""
    bus = EventBus()
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe("topic", handler)
    bus.publish("topic", 1)
    assert len(received) == 1

    bus.unsubscribe("topic", handler)
    bus.publish("topic", 2)
    assert len(received) == 1


def test_event_bus_history():
    """Verify event history logging."""
    bus = EventBus(max_history=5)
    for i in range(10):
        bus.publish("tick", i)

    history = bus.get_history()
    assert len(history) == 5
    assert history[-1] == ("tick", 9)
