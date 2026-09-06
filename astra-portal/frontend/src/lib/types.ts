/**
 * ============================================================================
 * OWNER: Backend Developer 2 (Shared Contract with Frontend)
 * PURPOSE: Central TypeScript definitions for all API responses and models.
 *
 * HOW TO EDIT / EXTEND:
 * 1. If you add fields in FastAPI (backend/api/models.py or inference.py),
 *    mirror them here to maintain end-to-end type safety.
 * 2. Frontend developers use these types for components and state.
 * ============================================================================
 */

export interface ModelArtifact {
  id: string;
  title: string;
  description: string;
  format: 'ONNX (INT8)' | 'PyTorch (.pt)' | 'TensorRT' | 'TorchScript';
  precision: 'INT8' | 'FP16' | 'FP32';
  size_mb: number;
  sha256: string;
  target_device: string;
  download_url: string;
  recommended: boolean;
  version: string;
  created_at: string;
}

export interface ClientBinary {
  id: string;
  platform: 'Windows' | 'Linux' | 'macOS' | 'PWA';
  filename: string;
  version: string;
  size_mb: number;
  download_url: string;
  sha256: string;
}

export interface DatasetSample {
  id: string;
  experiment_id: string;
  run_id: string;
  scenario: 'NOMINAL' | 'FAULT';
  camera: string;
  duration_sec: number;
  size_mb: number;
  download_url: string;
}

export interface InferenceResult {
  step_id: number;
  action_name: string;
  status: 'NOMINAL' | 'FAULT';
  confidence: number;
  inference_ms: number;
  anomaly_detected: boolean;
  timestamp: string;
}

export interface SystemStatus {
  service: string;
  status: 'operational' | 'degraded' | 'offline';
  active_models: number;
  dataset_runs_synced: number;
  last_updated: string;
}
