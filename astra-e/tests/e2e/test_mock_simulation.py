"""End-to-end integration test for the mock runtime simulation."""

from apps.runtime.main import run_simulation


def test_simulation_e2e_run():
    """Verify mock simulation executes to completion with 4 valid transitions and 1 resolved violation."""
    results = run_simulation(verbose=False)
    assert results["valid_count"] == 4
    assert results["violation_count"] == 1
    assert results["completed"] == 1
