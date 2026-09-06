/**
 * ============================================================================
 * OWNER: Frontend Developer 1
 * PURPOSE: Web Demo Sandbox Page.
 *          Streamlined with medium typography and clear, concise instructions.
 * ============================================================================
 */

import { DemoPlayer } from '@/components/demo-player';
import { Sparkles, FlaskConical, Layers, ShieldCheck } from 'lucide-react';

export default function DemoPage() {
  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8 py-12">
      {/* Header */}
      <div className="max-w-2xl mb-8">
        <div className="inline-flex items-center gap-1.5 rounded-full border border-cyan-dim/40 bg-space-card/80 px-3 py-1 text-xs font-mono text-cyan-accent mb-3">
          <Sparkles className="h-3 w-3 text-cyan-accent" />
          <span>EXP001 TESTBED</span>
        </div>

        <h1 className="text-2xl sm:text-4xl font-extrabold tracking-tight text-white">
          Live Model Sandbox
        </h1>

        <p className="mt-2 text-xs sm:text-sm text-text-secondary leading-relaxed">
          Test real-time microgravity action recognition on standard benchmark clips or preview local videos.
        </p>
      </div>

      {/* Protocol Summary Badges */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-6">
        <div className="rounded-lg border border-space-border bg-space-card/30 p-3 flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-cyan-dim/20 text-cyan-accent border border-cyan-dim/30 flex-shrink-0">
            <FlaskConical className="h-4 w-4" />
          </div>
          <div>
            <span className="text-[10px] font-mono text-cyan-accent uppercase block">Protocol</span>
            <span className="text-xs font-bold text-white">EXP001: Liquid Reagent</span>
          </div>
        </div>

        <div className="rounded-lg border border-space-border bg-space-card/30 p-3 flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-emerald-accent/20 text-emerald-accent border border-emerald-accent/30 flex-shrink-0">
            <Layers className="h-4 w-4" />
          </div>
          <div>
            <span className="text-[10px] font-mono text-emerald-accent uppercase block">Model Target</span>
            <span className="text-xs font-bold text-white">ONNX INT8 Edge Quantized</span>
          </div>
        </div>

        <div className="rounded-lg border border-space-border bg-space-card/30 p-3 flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-amber-accent/20 text-amber-accent border border-amber-accent/30 flex-shrink-0">
            <ShieldCheck className="h-4 w-4" />
          </div>
          <div>
            <span className="text-[10px] font-mono text-amber-accent uppercase block">Safety Check</span>
            <span className="text-xs font-bold text-white">Step Inversion & Anomaly Guard</span>
          </div>
        </div>
      </div>

      {/* Sandbox Player Component */}
      <DemoPlayer />
    </div>
  );
}
