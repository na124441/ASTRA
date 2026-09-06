/**
 * ============================================================================
 * OWNER: Frontend Developer 1 & Backend Developer 2 Integration
 * PURPOSE: Robust client-side API layer for ASTRA-E portal.
 *          Handles backend health discovery, system telemetry, and action inference.
 * ============================================================================
 */

import { InferenceResult, SystemStatus } from './types';

const RAW_API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

// Normalize base URL to avoid double slashes
export const API_BASE_URL = RAW_API_BASE_URL.replace(/\/+$/, '');

const DEFAULT_TIMEOUT_MS = 10000;

export type ApiErrorType = 'TIMEOUT' | 'NETWORK' | 'HTTP_ERROR' | 'UNKNOWN';

export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public errorType: ApiErrorType = 'UNKNOWN'
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export interface HealthCheckResult {
  online: boolean;
  status: 'operational' | 'degraded' | 'offline';
  message: string;
  service?: string;
  latencyMs?: number;
}

/**
 * Universal timeout-wrapped fetch with AbortController signal
 */
async function fetchWithTimeout(
  endpoint: string,
  options: RequestInit = {},
  timeoutMs = DEFAULT_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort();
  }, timeoutMs);

  const url = endpoint.startsWith('http') ? endpoint : `${API_BASE_URL}${endpoint}`;

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } catch (error: unknown) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new ApiError(
        'Request timed out. The ASTRA-E backend service took too long to respond.',
        408,
        'TIMEOUT'
      );
    }

    throw new ApiError(
      'Unable to connect to the ASTRA-E backend gateway. Ensure the service is running.',
      0,
      'NETWORK'
    );
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Check health of the FastAPI backend root (/health)
 */
export async function checkBackendHealth(): Promise<HealthCheckResult> {
  const startTime = Date.now();
  try {
    const response = await fetchWithTimeout('/health', { method: 'GET' }, 3500);
    const latencyMs = Date.now() - startTime;

    if (!response.ok) {
      return {
        online: false,
        status: 'degraded',
        message: `Backend responded with HTTP ${response.status}`,
        latencyMs,
      };
    }

    const data = await response.json();
    return {
      online: true,
      status: 'operational',
      message: 'Inference gateway is online and responding',
      service: data.service || 'ASTRA-E Portal API',
      latencyMs,
    };
  } catch {
    return {
      online: false,
      status: 'offline',
      message: 'Backend is unreachable (localhost:8000). Demo mode is active.',
      latencyMs: Date.now() - startTime,
    };
  }
}

/**
 * Fetch live system telemetry from /api/v1/system/status
 */
export async function getSystemTelemetry(): Promise<SystemStatus | null> {
  try {
    const response = await fetchWithTimeout(
      '/api/v1/system/status',
      { method: 'GET' },
      4000
    );

    if (!response.ok) return null;
    return (await response.json()) as SystemStatus;
  } catch {
    return null;
  }
}

/**
 * Run real-time action classification on a demo clip via POST /api/v1/demo/inference
 */
export async function runInference(clipId: string): Promise<InferenceResult> {
  const response = await fetchWithTimeout(
    '/api/v1/demo/inference',
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ clip_id: clipId }),
    },
    12000
  );

  if (!response.ok) {
    let errorDetails = `Inference failed with status ${response.status}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorDetails = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // Keep default error text
    }

    throw new ApiError(errorDetails, response.status, 'HTTP_ERROR');
  }

  return (await response.json()) as InferenceResult;
}