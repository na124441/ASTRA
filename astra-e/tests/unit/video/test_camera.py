"""Unit tests for Camera implementations and FrameBuffer."""

import numpy as np
import pytest
from astra.video.buffer import FrameBuffer
from astra.video.camera import MockCamera
from astra.video.capture import CapturePipeline


def test_mock_camera_stream():
    """Verify MockCamera frame generation, dimensions, and timestamps."""
    cam = MockCamera(width=320, height=240, fps=30.0, total_frames=10, loop=False)
    cam.start()


    frames_read = 0
    while True:
        ok, frame, ts = cam.read()
        if not ok:
            break
        assert frame is not None
        assert frame.shape == (240, 320, 3)
        assert ts > 0
        frames_read += 1

    assert frames_read == 10
    cam.stop()


def test_frame_buffer_ring_and_lookup():
    """Verify FrameBuffer stores, retrieves, and evicts frames in circular order."""
    buf = FrameBuffer(capacity=5)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Push 8 frames into buffer of capacity 5
    vfs = []
    for i in range(8):
        frame_copy = frame.copy()
        frame_copy[0, 0, 0] = i  # distinguish frame
        vf = buf.push(camera_id="CAM-01", frame=frame_copy, event_time=float(i))
        vfs.append(vf)

    assert len(buf) == 5

    # Frames 1, 2, 3 should have been evicted; frames 4, 5, 6, 7, 8 should exist
    assert buf.get_frame_by_id(1) is None
    assert buf.get_frame_by_id(2) is None
    assert buf.get_frame_by_id(3) is None

    retrieved = buf.get_frame(vfs[-1].frame_reference)
    assert retrieved is not None
    assert retrieved[0, 0, 0] == 7

    # Test recent window retrieval
    recent = buf.get_recent_frames(count=3)
    assert len(recent) == 3
    assert recent[-1][0].frame_id == 8


def test_capture_pipeline_callback():
    """Verify CapturePipeline single frame read and callback triggering."""
    cam = MockCamera(width=160, height=120, total_frames=5)
    cam.start()
    buf = FrameBuffer(capacity=10)
    pipeline = CapturePipeline(camera=cam, buffer=buf, correlation_id="TEST-RUN")

    invoked = []
    pipeline.add_callback(lambda vf, arr: invoked.append((vf.frame_id, arr.shape)))

    res = pipeline.read_single_frame()
    assert res is not None
    assert len(invoked) == 1
    assert invoked[0] == (1, (120, 160, 3))

    pipeline.stop()
