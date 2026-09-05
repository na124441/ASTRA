"""End-to-End system test for ASTRA-E EdgeRuntimeOrchestrator."""

import time
from astra.runtime.orchestrator import EdgeRuntimeOrchestrator
from astra.video.camera import MockCamera


def test_edge_runtime_orchestrator_e2e_pipeline(tmp_path):
    """
    Validates complete end-to-end execution of the EdgeRuntimeOrchestrator:
    Camera -> FrameBuffer -> Perception -> Kinematics -> Causal ML -> Confirmation
    -> Procedure Engine -> Assistance -> SQLite Ledger
    """
    db_file = tmp_path / "edge_e2e.db"
    camera = MockCamera(width=640, height=480, fps=30.0)

    orchestrator = EdgeRuntimeOrchestrator(
        camera=camera,
        db_path=db_file,
        tts_enabled=True,
        mock_tts=True,
    )

    try:
        # 1. Start experiment
        run_id = orchestrator.start_experiment(experiment_id="EXP001")
        assert run_id.startswith("RUN-")
        assert orchestrator.status == "RUNNING"
        assert orchestrator.procedure is not None

        # 2. Step 30 frames through the live pipeline
        telemetry_history = []
        for i in range(30):
            snap = orchestrator.step_frame()
            telemetry_history.append(snap)
            time.sleep(0.01)

        # 3. Verify video ingest & buffer
        assert len(orchestrator.frame_buffer) > 0
        assert orchestrator._latest_frame is not None
        assert orchestrator._latest_frame.shape == (480, 640, 3)

        # 4. Verify perception and tracking
        assert orchestrator._latest_scene is not None
        assert len(orchestrator._latest_scene.objects) > 0
        assert len(orchestrator._latest_scene.humans) > 0

        # 5. Verify MJPEG rendering
        jpeg_bytes = orchestrator.get_annotated_jpeg()
        assert len(jpeg_bytes) > 1000
        assert jpeg_bytes.startswith(b"\xff\xd8")  # Valid JPEG SOI header

        # 6. Verify telemetry structure
        latest = orchestrator.get_telemetry()
        assert latest["system"] == "ASTRA-E"
        assert latest["status"] == "RUNNING"
        assert latest["run_id"] == run_id
        assert len(latest["detections"]) > 0

        # 7. Pause and Resume
        orchestrator.pause_experiment()
        assert orchestrator.status == "PAUSED"
        orchestrator.resume_experiment()
        assert orchestrator.status == "RUNNING"

        # 8. Reset
        orchestrator.reset_experiment()
        assert orchestrator.status == "IDLE"

        # 9. Verify SQLite Audit Ledger
        report = orchestrator.ledger.export_audit_report(run_id)
        assert report.run_id == run_id
        assert report.status in ("RUNNING", "ABORTED", "COMPLETED")

    finally:
        orchestrator.shutdown()
