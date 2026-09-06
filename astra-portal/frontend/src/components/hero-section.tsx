/**
 * ============================================================================
 * OWNER: Frontend Developer 1
 * PURPOSE: Hero Banner Component with dark space HUD visuals, typing effect,
 *          and key call-to-action buttons.
 *
 * HOW TO EDIT:
 * 1. Customize heading or tagline text.
 * 2. Tweak Framer Motion animation timings (duration, delay).
 * 3. Add telemetry stats (FPS, Accuracy, Latency badges).
 * ============================================================================
 */

'use client';

import Link from 'next/link';
import { Rocket, Download, Terminal, ChevronRight, Activity } from 'lucide-react';

export function HeroSection() {
  return (
    <section className="relative overflow-hidden py-24 sm:py-32">
      {/* Subtle Background Glow */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[350px] bg-cyan-accent/10 rounded-full blur-[120px] pointer-events-none" />

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 relative z-10 text-center">
        {/* Mission Badge */}
        <div className="inline-flex items-center gap-2 rounded-full border border-cyan-dim/40 bg-space-card/80 px-3.5 py-1.5 text-xs font-mono text-cyan-accent mb-8 shadow-[0_0_15px_rgba(0,229,255,0.15)]">
          <Activity className="h-3.5 w-3.5 animate-pulse text-emerald-accent" />
          <span>BHARTIYA ANTARIKSH STATION • SIH 26174</span>
        </div>

        {/* Main Title */}
        <h1 className="text-4xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight text-white max-w-4xl mx-auto leading-tight">
          Autonomous Space Task Recognition & Assistance
        </h1>

        {/* Subtitle / HUD Description */}
        <p className="mt-6 text-base sm:text-lg text-text-secondary max-w-2xl mx-auto leading-relaxed">
          Real-time spatial-temporal AI for microgravity science experiments. Guiding astronauts through complex protocols, verifying procedure steps, and catching zero-g anomalies with sub-150ms edge inference.
        </p>

        {/* Action Buttons */}
        <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
          <Link
            href="/downloads"
            className="flex items-center gap-2 rounded-xl bg-cyan-accent px-6 py-3.5 text-sm font-semibold text-space-bg hover:bg-cyan-accent/90 transition-all shadow-[0_0_20px_rgba(0,229,255,0.4)]"
          >
            <Download className="h-4 w-4" />
            <span>Download Models & Binaries</span>
          </Link>

          <Link
            href="/docs"
            className="flex items-center gap-2 rounded-xl border border-space-border bg-space-card/70 px-6 py-3.5 text-sm font-semibold text-white hover:bg-space-card hover:border-cyan-dim transition-all"
          >
            <Terminal className="h-4 w-4 text-cyan-accent" />
            <span>Quickstart Guide</span>
            <ChevronRight className="h-4 w-4 text-text-secondary" />
          </Link>

          <Link
            href="/demo"
            className="flex items-center gap-2 rounded-xl border border-emerald-accent/40 bg-emerald-accent/10 px-6 py-3.5 text-sm font-semibold text-emerald-accent hover:bg-emerald-accent/20 transition-all"
          >
            <Rocket className="h-4 w-4" />
            <span>Try Live Sandbox</span>
          </Link>
        </div>

        {/* Spec HUD Bar */}
        <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto border border-space-border/60 bg-space-dark/40 rounded-2xl p-4 backdrop-blur-sm">
          <div className="p-2">
            <p className="font-mono text-2xl font-bold text-cyan-accent">&lt;150ms</p>
            <p className="text-xs text-text-secondary mt-1">Edge Latency</p>
          </div>
          <div className="p-2">
            <p className="font-mono text-2xl font-bold text-emerald-accent">96.4%</p>
            <p className="text-xs text-text-secondary mt-1">Step Accuracy</p>
          </div>
          <div className="p-2">
            <p className="font-mono text-2xl font-bold text-white">INT8 / FP16</p>
            <p className="text-xs text-text-secondary mt-1">Quantized Models</p>
          </div>
          <div className="p-2">
            <p className="font-mono text-2xl font-bold text-amber-accent">Zero-G</p>
            <p className="text-xs text-text-secondary mt-1">Anomaly Guard</p>
          </div>
        </div>
      </div>
    </section>
  );
}
