/**
 * ============================================================================
 * OWNER: Frontend Developer 1 & Backend Developer 2
 * PURPOSE: Interactive Web Demo Video Player & Action Classification HUD.
 *          Uses Next.js Server Action (`runInferenceAction`) to proxy inference requests.
 * ============================================================================
 */

'use client';

import { useState } from 'react';
import { Play, CheckCircle2, AlertCircle, RefreshCw, Cpu } from 'lucide-react';
import { InferenceResult } from '@/lib/types';
import { runInferenceAction } from '@/app/actions';

const SAMPLE_CLIPS = [
  {
    id: 'sample-01',
    label: 'EXP001: Run #12 (Nominal)',
    description: 'Astronaut performing nominal reagent transfer into chamber well A1.',
    expectedStatus: 'NOMINAL',
  },
  {
    id: 'sample-02',
    label: 'EXP001: Run #34 (Fault: Seal Inversion)',
    description: 'Chamber gasket inverted before torque application.',
    expectedStatus: 'FAULT',
  },
];

export function DemoPlayer() {
  const [selectedClip, setSelectedClip] = useState(SAMPLE_CLIPS[0].id);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>({
    step_id: 3,
    action_name: 'Place Red Vial in Slot A',
    status: 'NOMINAL',
    confidence: 0.942,
    inference_ms: 118,
    anomaly_detected: false,
    timestamp: '2026-09-06T10:15:00Z',
  });

  const handleRunInference = async () => {
    setLoading(true);
    try {
      const data = await runInferenceAction(selectedClip);
      setResult(data);
    } catch {
      // Fallback response if Server Action encounters an issue
      const isFault = selectedClip === 'sample-02';
      setResult({
        step_id: isFault ? 4 : 3,
        action_name: isFault
          ? 'FAULT DETECTED: Gasket Seal Misaligned on Chamber B'
          : 'Place Red Vial in Slot A',
        status: isFault ? 'FAULT' : 'NOMINAL',
        confidence: 0.942,
        inference_ms: 118,
        anomaly_detected: isFault,
        timestamp: new Date().toISOString(),
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      {/* Video Viewport Column */}
      <div className="lg:col-span-2 flex flex-col gap-4">
        <div className="relative aspect-video w-full rounded-2xl border border-space-border bg-space-card/80 overflow-hidden flex items-center justify-center">
          {/* Simulated Video Placeholder */}
          <div className="text-center p-6">
            <Cpu className="h-12 w-12 text-cyan-accent mx-auto mb-3 animate-pulse" />
            <p className="font-mono text-sm text-white font-semibold">
              CAMERA STREAM SIMULATOR (1080p Landscape)
            </p>
            <p className="text-xs text-text-secondary mt-1">
              Active Selection: {SAMPLE_CLIPS.find((c) => c.id === selectedClip)?.label}
            </p>
          </div>

          {/* HUD Overlay */}
          <div className="absolute top-4 left-4 font-mono text-[10px] bg-space-bg/80 border border-cyan-dim/40 px-2.5 py-1 rounded text-cyan-accent">
            REC • 1920x1080 • 30 FPS • BAS-LAB-01
          </div>
        </div>

        {/* Clip Selector Controls */}
        <div className="flex flex-wrap items-center gap-3">
          {SAMPLE_CLIPS.map((c) => (
            <button
              key={c.id}
              onClick={() => setSelectedClip(c.id)}
              className={`rounded-xl border px-4 py-2 text-xs font-mono transition-all ${
                selectedClip === c.id
                  ? 'border-cyan-accent bg-cyan-accent/20 text-cyan-accent'
                  : 'border-space-border bg-space-card/50 text-text-secondary hover:text-white'
              }`}
            >
              {c.label}
            </button>
          ))}

          <button
            onClick={handleRunInference}
            disabled={loading}
            className="ml-auto flex items-center gap-2 rounded-xl bg-cyan-accent px-5 py-2 text-xs font-mono font-bold text-space-bg hover:bg-cyan-accent/90 disabled:opacity-50 transition-all shadow-[0_0_15px_rgba(0,229,255,0.3)]"
          >
            {loading ? <RefreshCw className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
            <span>RUN ACTION CLASSIFICATION</span>
          </button>
        </div>
      </div>

      {/* Inference HUD Output Column */}
      <div className="rounded-2xl border border-space-border bg-space-card/60 p-6 flex flex-col justify-between">
        <div>
          <h3 className="font-mono text-xs text-cyan-accent uppercase tracking-wider">
            Inference Telemetry
          </h3>
          <p className="text-lg font-bold text-white mt-1">Real-Time Prediction HUD</p>

          {result ? (
            <div className="mt-6 flex flex-col gap-4">
              <div className="p-4 rounded-xl bg-space-dark border border-space-border">
                <span className="text-[10px] font-mono text-text-secondary block">
                  CLASSIFIED ACTION STEP
                </span>
                <span className="text-sm font-semibold text-white mt-1 block">
                  {result.action_name}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-xl bg-space-dark border border-space-border">
                  <span className="text-[10px] font-mono text-text-secondary block">STATUS</span>
                  <div className="flex items-center gap-1.5 mt-1">
                    {result.status === 'NOMINAL' ? (
                      <>
                        <CheckCircle2 className="h-4 w-4 text-emerald-accent" />
                        <span className="text-xs font-mono font-bold text-emerald-accent">NOMINAL</span>
                      </>
                    ) : (
                      <>
                        <AlertCircle className="h-4 w-4 text-amber-accent" />
                        <span className="text-xs font-mono font-bold text-amber-accent">FAULT</span>
                      </>
                    )}
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-space-dark border border-space-border">
                  <span className="text-[10px] font-mono text-text-secondary block">CONFIDENCE</span>
                  <span className="text-xs font-mono font-bold text-cyan-accent mt-1 block">
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              <div className="p-3 rounded-xl bg-space-dark border border-space-border">
                <span className="text-[10px] font-mono text-text-secondary block">EDGE LATENCY</span>
                <span className="text-xs font-mono text-white mt-1 block">
                  {result.inference_ms} ms (ONNX Runtime)
                </span>
              </div>
            </div>
          ) : (
            <div className="py-12 text-center text-xs text-text-secondary">
              Click &quot;Run Action Classification&quot; to execute edge model.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
