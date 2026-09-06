/**
 * ============================================================================
 * OWNER: Backend Developer 2 (Full-Stack Glue)
 * PURPOSE: Next.js Server Actions & Route Handlers.
 *          Proxies client requests to the FastAPI backend without exposing
 *          backend URLs or internal tokens to the browser.
 *
 * HOW TO EDIT:
 * 1. Update `FASTAPI_BASE_URL` to point to production backend on Vercel/Railway.
 * 2. Add server actions for model inquiries, telemetry polling, or bug reports.
 * ============================================================================
 */

'use server';

import { InferenceResult, ModelArtifact, SystemStatus } from '@/lib/types';
import { MOCK_MODELS, MOCK_SYSTEM_STATUS } from '@/lib/mock-data';

const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL || 'http://localhost:8000';

/**
 * Fetch available models from FastAPI or fallback to mock data.
 */
export async function getModelsAction(): Promise<ModelArtifact[]> {
  try {
    const res = await fetch(`${FASTAPI_BASE_URL}/api/v1/models`, {
      next: { revalidate: 60 },
    });
    if (!res.ok) throw new Error('Backend unreachable');
    const data = await res.json();
    return data.models;
  } catch {
    // Fallback to static mock data if backend is offline
    return MOCK_MODELS;
  }
}

/**
 * Execute web action classification demo.
 */
export async function runInferenceAction(clipId: string): Promise<InferenceResult> {
  try {
    const res = await fetch(`${FASTAPI_BASE_URL}/api/v1/demo/inference`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clip_id: clipId }),
      cache: 'no-store',
    });
    if (!res.ok) throw new Error('Inference error');
    return await res.json();
  } catch {
    // Fallback simulation
    const isFault = clipId === 'sample-02';
    return {
      step_id: isFault ? 4 : 3,
      action_name: isFault
        ? 'FAULT DETECTED: Gasket Seal Misaligned'
        : 'Transfer Reagent to Well A1 via Calibrated Pipette',
      status: isFault ? 'FAULT' : 'NOMINAL',
      confidence: isFault ? 0.941 : 0.962,
      inference_ms: 118,
      anomaly_detected: isFault,
      timestamp: new Date().toISOString(),
    };
  }
}

/**
 * Query live system status.
 */
export async function getSystemStatusAction(): Promise<SystemStatus> {
  try {
    const res = await fetch(`${FASTAPI_BASE_URL}/api/v1/system/status`, {
      next: { revalidate: 15 },
    });
    if (!res.ok) throw new Error('Status unreachable');
    return await res.json();
  } catch {
    return MOCK_SYSTEM_STATUS;
  }
}
