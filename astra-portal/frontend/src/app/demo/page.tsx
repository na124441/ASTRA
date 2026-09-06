/**
 * ============================================================================
 * OWNER: Frontend Developer 1
 * PURPOSE: Web Demo Sandbox Page.
 * ============================================================================
 */

import { DemoPlayer } from '@/components/demo-player';

export default function DemoPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-16">
      <div className="max-w-2xl mb-12">
        <p className="font-mono text-xs text-cyan-accent uppercase tracking-widest">
          Interactive Evaluation Sandbox
        </p>
        <h1 className="mt-2 text-3xl font-extrabold text-white sm:text-4xl">
          Live Model Inference Testbed
        </h1>
        <p className="mt-3 text-sm text-text-secondary">
          Evaluate ASTRA-E action recognition models directly in your browser using standardized EXP001 microgravity clips.
        </p>
      </div>

      <DemoPlayer />
    </div>
  );
}
