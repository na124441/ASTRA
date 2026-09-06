/**
 * ============================================================================
 * OWNER: Frontend Developer 1
 * PURPOSE: Refined Hero Section with medium typography, concise messaging,
 *          and prominent ASTRA branding.
 * ============================================================================
 */

'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import {
  PlayCircle,
  Download,
  Terminal,
  ShieldCheck,
  Activity,
  CheckCircle2,
  Gauge,
  Cpu,
} from 'lucide-react';

export function HeroSection() {
  return (
    <section className="relative overflow-hidden pt-16 pb-20 sm:pt-24 sm:pb-28">
      {/* Soft background glow */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[300px] bg-cyan-accent/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute top-1/4 right-1/4 w-[300px] h-[200px] bg-emerald-accent/5 rounded-full blur-[100px] pointer-events-none" />

      {/* Subtle grid pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#141B2D_1px,transparent_1px),linear-gradient(to_bottom,#141B2D_1px,transparent_1px)] bg-[size:48px_48px] opacity-20 pointer-events-none" />

      <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8 relative z-10 text-center">
        {/* Mission Badge */}
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="inline-flex items-center gap-2 rounded-full border border-cyan-dim/30 bg-space-card/80 px-3.5 py-1 text-xs font-mono text-cyan-accent mb-6 shadow-[0_0_15px_rgba(0,229,255,0.1)] backdrop-blur-sm"
        >
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-accent opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-accent" />
          </span>
          <span>BHARATIYA ANTARIKSH STATION</span>
          <span className="text-text-secondary">•</span>
          <span className="text-emerald-accent">SIH 26174</span>
        </motion.div>

        {/* Highlighted ASTRA Title */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <div className="flex items-center justify-center gap-2 mb-2">
            <h1 className="text-5xl sm:text-6xl font-black tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-cyan-accent via-white to-emerald-accent drop-shadow-[0_0_20px_rgba(0,229,255,0.3)]">
              ASTRA<span className="text-cyan-accent">-E</span>
            </h1>
          </div>

          {/* Sub-headline: Autonomous Action Recognition */}
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white mt-1 max-w-3xl mx-auto">
            Autonomous Action Recognition for Bharatiya Antariksh Station
          </h2>
        </motion.div>

        {/* Short, simple, easy to understand description */}
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mt-4 text-sm sm:text-base text-text-secondary max-w-2xl mx-auto leading-relaxed"
        >
          Real-time AI monitoring for space science experiments. Verifies procedure compliance, detects zero-g anomalies, and guides astronauts with sub-150ms edge inference.
        </motion.p>

        {/* Call to Actions */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="mt-8 flex flex-wrap items-center justify-center gap-3.5"
        >
          <Link
            href="/demo"
            className="flex items-center gap-2 rounded-xl bg-cyan-accent px-5 py-2.5 text-xs sm:text-sm font-bold font-mono text-space-bg hover:bg-cyan-accent/90 transition-all shadow-[0_0_20px_rgba(0,229,255,0.3)] active:scale-[0.98]"
          >
            <PlayCircle className="h-4 w-4 fill-current" />
            <span>Try Live Sandbox</span>
          </Link>

          <Link
            href="/downloads"
            className="flex items-center gap-2 rounded-xl border border-space-border bg-space-card/70 px-5 py-2.5 text-xs sm:text-sm font-semibold text-white hover:bg-space-card hover:border-cyan-dim transition-all active:scale-[0.98]"
          >
            <Download className="h-4 w-4 text-cyan-accent" />
            <span>Download Models</span>
          </Link>

          <a
            href="https://astra-na124441s-projects.vercel.app/collector"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 rounded-xl border border-emerald-accent/40 bg-emerald-accent/10 px-4 py-2.5 text-xs sm:text-sm font-semibold text-emerald-accent hover:bg-emerald-accent/20 transition-all active:scale-[0.98]"
          >
            <ShieldCheck className="h-4 w-4" />
            <span>Collector PWA ↗</span>
          </a>

          <Link
            href="/docs"
            className="flex items-center gap-2 rounded-xl border border-space-border/60 bg-space-dark/40 px-4 py-2.5 text-xs sm:text-sm font-medium text-text-secondary hover:text-white transition-all"
          >
            <Terminal className="h-4 w-4 text-text-secondary" />
            <span>Run Guide</span>
          </Link>
        </motion.div>

        {/* Clean, medium-sized Telemetry Strip */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="mt-12 grid grid-cols-2 sm:grid-cols-4 gap-3 max-w-3xl mx-auto border border-space-border/60 bg-space-card/30 rounded-xl p-3 backdrop-blur-sm shadow-lg"
        >
          <div className="p-2 text-center border-r border-space-border/40 last:border-r-0">
            <div className="flex items-center justify-center gap-1.5 text-text-secondary text-[10px] font-mono">
              <Gauge className="h-3 w-3 text-cyan-accent" />
              <span>LATENCY</span>
            </div>
            <p className="font-mono text-lg sm:text-xl font-bold text-cyan-accent mt-0.5">&lt;150ms</p>
            <p className="text-[10px] text-text-secondary">Edge runtime</p>
          </div>

          <div className="p-2 text-center border-r border-space-border/40 last:border-r-0">
            <div className="flex items-center justify-center gap-1.5 text-text-secondary text-[10px] font-mono">
              <CheckCircle2 className="h-3 w-3 text-emerald-accent" />
              <span>ACCURACY</span>
            </div>
            <p className="font-mono text-lg sm:text-xl font-bold text-emerald-accent mt-0.5">96.4%</p>
            <p className="text-[10px] text-text-secondary">EXP001 test</p>
          </div>

          <div className="p-2 text-center border-r border-space-border/40 last:border-r-0">
            <div className="flex items-center justify-center gap-1.5 text-text-secondary text-[10px] font-mono">
              <Cpu className="h-3 w-3 text-white" />
              <span>PRECISION</span>
            </div>
            <p className="font-mono text-lg sm:text-xl font-bold text-white mt-0.5">INT8 / FP16</p>
            <p className="text-[10px] text-text-secondary">Quantized</p>
          </div>

          <div className="p-2 text-center">
            <div className="flex items-center justify-center gap-1.5 text-text-secondary text-[10px] font-mono">
              <Activity className="h-3 w-3 text-amber-accent" />
              <span>GUARD</span>
            </div>
            <p className="font-mono text-lg sm:text-xl font-bold text-amber-accent mt-0.5">Active</p>
            <p className="text-[10px] text-text-secondary">Zero-G alerts</p>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
