"""Unit tests for ActionConfirmationEngine."""

from astra.activity.confirmation import ActionConfirmationEngine
from astra.contracts.activity import ActionObservation, TemporalWindow


def create_obs(action: str, obj: str | None = None, tgt: str | None = None, conf: float = 0.85) -> ActionObservation:
    return ActionObservation(
        source="test",
        correlation_id="RUN-1",
        action=action,
        object_id=obj,
        target_id=tgt,
        confidence=conf,
        temporal_window=TemporalWindow(start=0.0, end=1.0),
    )


def test_confirmation_persistence():
    """Verify engine promotes observation to ConfirmedAction only after sustained support."""
    engine = ActionConfirmationEngine(min_support_frames=4, confirmation_threshold=0.70)

    # 3 frames of PICK RED -> not yet confirmed
    for _ in range(3):
        res = engine.process_observation(create_obs("PICK", "RED_COMPONENT", None, conf=0.85))
        assert res is None

    # 4th frame -> confirmed!
    res4 = engine.process_observation(create_obs("PICK", "RED_COMPONENT", None, conf=0.85))
    assert res4 is not None
    assert res4.action == "PICK"
    assert res4.object_id == "RED_COMPONENT"
    assert res4.confirmation is not None
    assert res4.confirmation.stable_frames >= 4


def test_confirmation_flicker_suppression():
    """Verify single-frame noise spike does not trigger a false confirmation."""
    engine = ActionConfirmationEngine(min_support_frames=4)

    # 2 frames of PICK
    engine.process_observation(create_obs("PICK", "RED_COMPONENT", None, conf=0.85))
    engine.process_observation(create_obs("PICK", "RED_COMPONENT", None, conf=0.85))

    # 1 anomalous spike frame of CLOSE_CONTAINER
    spike = engine.process_observation(create_obs("CLOSE_CONTAINER", "CONTAINER", None, conf=0.99))
    assert spike is None

    # Resume PICK -> no false close
    res = engine.process_observation(create_obs("PICK", "RED_COMPONENT", None, conf=0.85))
    assert res is None  # still accumulating


def test_confirmation_abstention_on_low_confidence():
    """Verify engine explicitly abstains when prediction confidence is below threshold."""
    engine = ActionConfirmationEngine(abstain_threshold=0.55)
    res = engine.process_observation(create_obs("PICK", "RED_COMPONENT", None, conf=0.45))
    assert res is None


def test_physical_transition_plausibility():
    """Verify physically impossible transition is rejected."""
    engine = ActionConfirmationEngine(min_support_frames=2)

    # Establish last confirmed action as APPROACH
    engine.process_observation(create_obs("APPROACH", "RED_COMPONENT", None, conf=0.9))
    c1 = engine.process_observation(create_obs("APPROACH", "RED_COMPONENT", None, conf=0.9))
    assert c1 is not None and c1.action == "APPROACH"

    # Attempt immediate RELEASE without grasping or moving -> physically impossible!
    engine.process_observation(create_obs("RELEASE", "RED_COMPONENT", "TARGET_A", conf=0.9))
    c2 = engine.process_observation(create_obs("RELEASE", "RED_COMPONENT", "TARGET_A", conf=0.9))
    assert c2 is None
