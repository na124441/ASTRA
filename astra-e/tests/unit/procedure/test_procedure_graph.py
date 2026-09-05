"""Unit tests for ProcedureGraph and ProcedureValidator."""

import pytest
from astra.contracts.procedure import ProcedureDefinition, ProcedureStep
from astra.procedure.graph import ProcedureGraph
from astra.procedure.validator import ProcedureValidationError, ProcedureValidator


@pytest.fixture
def linear_procedure():
    return ProcedureDefinition(
        id="PROC-LINEAR",
        experiment_id="EXP-01",
        name="Linear Test",
        steps=[
            ProcedureStep(id="S1", action="OPEN_CONTAINER", allowed_next=["S2"]),
            ProcedureStep(id="S2", action="PICK", object="RED", allowed_next=["S3"]),
            ProcedureStep(id="S3", action="PLACE", object="RED", target="A", allowed_next=["S4"]),
            ProcedureStep(id="S4", action="CLOSE_CONTAINER", allowed_next=[]),
        ],
        initial_step_id="S1",
        terminal_step_ids=["S4"],
    )


def test_graph_initial_and_terminal(linear_procedure):
    """Verify initial step and terminal identification."""
    graph = ProcedureGraph(linear_procedure)
    assert graph.initial_step is not None
    assert graph.initial_step.id == "S1"
    assert graph.is_terminal_step("S4")
    assert not graph.is_terminal_step("S2")


def test_graph_next_steps(linear_procedure):
    """Verify outgoing transition queries."""
    graph = ProcedureGraph(linear_procedure)
    next_from_start = graph.get_allowed_next_steps(None)
    assert len(next_from_start) == 1
    assert next_from_start[0].id == "S1"

    next_from_s1 = graph.get_allowed_next_steps("S1")
    assert len(next_from_s1) == 1
    assert next_from_s1[0].id == "S2"


def test_graph_skipped_step_detection(linear_procedure):
    """Verify intermediate skipped steps are accurately identified."""
    graph = ProcedureGraph(linear_procedure)
    # Skipping from S1 straight to S4 skips S2 and S3
    skipped = graph.get_skipped_steps("S1", "S4")
    skipped_ids = [s.id for s in skipped]
    assert skipped_ids == ["S2", "S3"]


def test_graph_branching_support():
    """Verify graph supports parallel branching options."""
    branching_proc = ProcedureDefinition(
        id="PROC-BRANCH",
        experiment_id="EXP-02",
        steps=[
            ProcedureStep(id="S1", action="START", allowed_next=["OPT_A", "OPT_B"]),
            ProcedureStep(id="OPT_A", action="PICK", object="TOOL_A", allowed_next=["S3"]),
            ProcedureStep(id="OPT_B", action="PICK", object="TOOL_B", allowed_next=["S3"]),
            ProcedureStep(id="S3", action="MEASURE", allowed_next=[]),
        ],
        terminal_step_ids=["S3"],
    )
    graph = ProcedureGraph(branching_proc)
    next_from_s1 = graph.get_allowed_next_steps("S1")
    next_ids = {s.id for s in next_from_s1}
    assert next_ids == {"OPT_A", "OPT_B"}


def test_validator_rejects_missing_target():
    """Verify ProcedureValidator catches invalid step transitions."""
    bad_proc = ProcedureDefinition(
        id="PROC-BAD",
        experiment_id="EXP-BAD",
        steps=[
            ProcedureStep(id="S1", action="PICK", allowed_next=["NONEXISTENT_STEP"]),
        ],
    )
    with pytest.raises(ProcedureValidationError):
        ProcedureValidator.validate(bad_proc)
