"""ASTRA-E Video Acquisition and Buffer package."""

from astra.video.buffer import FrameBuffer
from astra.video.camera import Camera, FileCamera, MockCamera, OpenCVCamera
from astra.video.capture import CapturePipeline
from astra.video.recorder import VideoRecorder

__all__ = [
    "Camera",
    "FileCamera",
    "MockCamera",
    "OpenCVCamera",
    "FrameBuffer",
    "CapturePipeline",
    "VideoRecorder",
]
