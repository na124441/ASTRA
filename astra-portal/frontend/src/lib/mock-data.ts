/**
 * ============================================================================
 * OWNER: Frontend Developer 2 (Zero-Blocker Mock Repository)
 * PURPOSE: Offline mock data so the frontend can build 100% of the UI
 *          without waiting for backend deployment.
 *
 * HOW TO EDIT:
 * 1. Add realistic model versions, sample runs, or download links.
 * 2. When backend is ready, frontend components can seamlessly toggle
 *    between MOCK_MODELS and live fetch('/api/v1/models').
 * ============================================================================
 */

import { ModelArtifact, ClientBinary, DatasetSample, SystemStatus } from './types';

export const MOCK_MODELS: ModelArtifact[] = [
  {
    id: "astra-exp001-int8",
    title: "ASTRA EXP001 Fast Action Recognizer",
    description: "Quantized 8-bit model optimized for edge devices, Raspberry Pi 4, and low-power space station hardware.",
    format: "ONNX (INT8)",
    precision: "INT8",
    size_mb: 142.5,
    sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    target_device: "Edge / Jetson Nano / Intel Core i5",
    download_url: "https://huggingface.co/na124441/astra-e-raw/resolve/main/models/exp001-int8.onnx",
    recommended: true,
    version: "v1.0.0",
    created_at: "2026-09-05"
  },
  {
    id: "astra-exp001-fp16",
    title: "ASTRA EXP001 High-Precision Spatial Model",
    description: "Full precision model for multi-camera 3D triangulation and complex zero-g procedure verification.",
    format: "PyTorch (.pt)",
    precision: "FP16",
    size_mb: 512.0,
    sha256: "9b7d8a9f24c3e8e19b62a5b1d44c82b0e77d91e1d743a129683b542e0f84693a",
    target_device: "NVIDIA RTX 3060+ / Cloud GPU",
    download_url: "https://huggingface.co/na124441/astra-e-raw/resolve/main/models/exp001-fp16.pt",
    recommended: false,
    version: "v1.0.0",
    created_at: "2026-09-05"
  },
  {
    id: "astra-exp001-tensorrt",
    title: "ASTRA EXP001 TensorRT Accelerated Engine",
    description: "Sub-50ms ultra-low-latency engine compiled specifically for NVIDIA embedded platforms.",
    format: "TensorRT",
    precision: "FP16",
    size_mb: 288.4,
    sha256: "a1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2",
    target_device: "NVIDIA Jetson Orin / AGX",
    download_url: "https://huggingface.co/na124441/astra-e-raw/resolve/main/models/exp001-tensorrt.engine",
    recommended: false,
    version: "v1.1.0",
    created_at: "2026-09-06"
  }
];

export const MOCK_BINARIES: ClientBinary[] = [
  {
    id: "astra-cli-win",
    platform: "Windows",
    filename: "astra-engine-v1.0.0-windows-x64.zip",
    version: "1.0.0",
    size_mb: 48.2,
    download_url: "https://github.com/na124441/ASTRA/releases/download/v1.0.0/astra-windows-x64.zip",
    sha256: "4c5b6a7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b"
  },
  {
    id: "astra-cli-linux",
    platform: "Linux",
    filename: "astra-engine-v1.0.0-linux-x64.tar.gz",
    version: "1.0.0",
    size_mb: 44.8,
    download_url: "https://github.com/na124441/ASTRA/releases/download/v1.0.0/astra-linux-x64.tar.gz",
    sha256: "7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b"
  }
];

export const MOCK_SAMPLES: DatasetSample[] = [
  {
    id: "sample-01",
    experiment_id: "EXP001",
    run_id: "RUN-0012",
    scenario: "NOMINAL",
    camera: "CAM-01 (Top-Down)",
    duration_sec: 42,
    size_mb: 18.5,
    download_url: "https://huggingface.co/na124441/astra-e-raw/resolve/main/samples/exp001_run0012_cam01.mp4"
  },
  {
    id: "sample-02",
    experiment_id: "EXP001",
    run_id: "RUN-0034",
    scenario: "FAULT",
    camera: "CAM-02 (Over-the-Shoulder)",
    duration_sec: 38,
    size_mb: 16.2,
    download_url: "https://huggingface.co/na124441/astra-e-raw/resolve/main/samples/exp001_run0034_cam02.mp4"
  }
];

export const MOCK_SYSTEM_STATUS: SystemStatus = {
  service: "ASTRA-E Inference Cluster",
  status: "operational",
  active_models: 3,
  dataset_runs_synced: 128,
  last_updated: "2026-09-06T10:00:00Z"
};
