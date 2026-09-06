/**
 * ============================================================================
 * OWNER: Frontend Developer 1
 * PURPOSE: Interactive 4-Step Architecture Pipeline Visualizer.
 *          Clean, medium typography with simple, concise descriptions.
 * ============================================================================
 */

'use client';

import { useState } from 'react';
import { Camera, Cpu, AlertTriangle, CloudUpload, ArrowRight, ArrowDown, Activity, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface PipelineStage {
  step: string;
  id: string;
  title: string;
  subtitle: string;
  summary: string;
  icon: typeof Camera;
  accentColor: string;
  borderColor: string;
  glowColor: string;
  badge: string;
  latency: string;
  input: string;
  output: string;
  points: string[];
}

const PIPELINE_STAGES: PipelineStage[] = [
  {
    step: '01',
    id: 'camera',
    title: 'Multi-Camera Input',
    subtitle: 'Synchronized Video Feeds',
    summary: 'Captures top-down and shoulder angles at 1080p 30 FPS to eliminate blind spots in zero gravity.',
    icon: Camera,
    accentColor: 'text-cyan-accent',
    borderColor: 'border-cyan-accent/50',
    glowColor: 'rgba(0, 229, 255, 0.2)',
    badge: 'INPUT',
    latency: '<15ms',
    input: '1080p 30 FPS multi-angle camera feeds',
    output: 'Calibrated microgravity frame tensors',
    points: [
      'Over-the-shoulder and top-down workbench perspectives',
      'Optical anti-glare and microgravity light compensation',
    ],
  },
  {
    step: '02',
    id: 'engine',
    title: 'Spatial-Temporal AI',
    subtitle: 'Edge Action Classification',
    summary: 'Quantized INT8 model classifies astronaut actions, hand gestures, and lab tools in real time.',
    icon: Cpu,
    accentColor: 'text-emerald-accent',
    borderColor: 'border-emerald-accent/50',
    glowColor: 'rgba(0, 230, 118, 0.2)',
    badge: 'INFERENCE',
    latency: '80–120ms',
    input: 'Multi-frame image sequence',
    output: 'Action label + prediction confidence score',
    points: [
      'Compact INT8 weights (142 MB) optimized for station SBCs',
      'Detects pipette use, vial handling, and chamber actions',
    ],
  },
  {
    step: '03',
    id: 'guard',
    title: 'Procedure Guard',
    subtitle: 'Nominal & Anomaly Verification',
    summary: 'Compares detected actions against official protocols to instantly catch skipped or inverted steps.',
    icon: AlertTriangle,
    accentColor: 'text-amber-accent',
    borderColor: 'border-amber-accent/50',
    glowColor: 'rgba(255, 214, 0, 0.2)',
    badge: 'MONITOR',
    latency: '<5ms',
    input: 'Action sequence + experiment protocol',
    output: 'NOMINAL confirmation or FAULT alert',
    points: [
      'Immediate alert if steps are skipped or performed out of order',
      'Prevents sample loss and faulty experiment runs',
    ],
  },
  {
    step: '04',
    id: 'sync',
    title: 'Telemetry Sync',
    subtitle: 'Signed Ground Station Upload',
    summary: 'Buffers data locally when offline, then uploads signed session chunks to ground station repositories.',
    icon: CloudUpload,
    accentColor: 'text-cyan-accent',
    borderColor: 'border-cyan-dim/50',
    glowColor: 'rgba(0, 229, 255, 0.2)',
    badge: 'SYNC',
    latency: 'Async',
    input: 'Recorded session + inference logs',
    output: 'SHA-256 verified telemetry archive',
    points: [
      'Offline-first buffer with resumable chunked transfer',
      'Cryptographic checksum verification per chunk',
    ],
  },
];

export function ArchitectureDiagram() {
  const [activeStageIndex, setActiveStageIndex] = useState(1);
  const activeStage = PIPELINE_STAGES[activeStageIndex];

  return (
    <section className="py-20 border-t border-space-border/60 bg-space-dark/20 relative">
      <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <div className="text-center max-w-2xl mx-auto mb-12">
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-dim/30 bg-space-card/80 px-3 py-1 text-xs font-mono text-cyan-accent mb-3">
            <Activity className="h-3.5 w-3.5 text-cyan-accent animate-pulse" />
            <span>HOW IT WORKS</span>
          </div>

          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
            4-Step Recognition Pipeline
          </h2>

          <p className="mt-2.5 text-sm text-text-secondary leading-relaxed">
            From raw camera stream to astronaut assistance in under 150ms. Click any step below to inspect details.
          </p>
        </div>

        {/* 4 Pipeline Stage Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 relative">
          {PIPELINE_STAGES.map((s, idx) => {
            const Icon = s.icon;
            const isSelected = activeStageIndex === idx;

            return (
              <div key={s.id} className="relative flex flex-col">
                <button
                  onClick={() => setActiveStageIndex(idx)}
                  className={`group relative text-left rounded-xl border p-4 transition-all duration-200 h-full flex flex-col justify-between ${
                    isSelected
                      ? `${s.borderColor} bg-space-card shadow-md`
                      : 'border-space-border bg-space-card/40 hover:bg-space-card/70'
                  }`}
                  style={{
                    boxShadow: isSelected ? `0 0 20px ${s.glowColor}` : 'none',
                  }}
                >
                  <div>
                    {/* Top Row */}
                    <div className="flex items-center justify-between mb-2.5">
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono text-xs font-bold text-text-secondary">
                          STEP {s.step}
                        </span>
                        <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-space-dark border border-space-border text-cyan-accent">
                          {s.badge}
                        </span>
                      </div>

                      <div
                        className={`flex h-7 w-7 items-center justify-center rounded-lg border ${
                          isSelected
                            ? 'bg-space-dark border-cyan-dim/50 ' + s.accentColor
                            : 'bg-space-dark/60 border-space-border text-text-secondary'
                        }`}
                      >
                        <Icon className="h-3.5 w-3.5" />
                      </div>
                    </div>

                    <h3 className="text-sm sm:text-base font-bold text-white group-hover:text-cyan-accent transition-colors">
                      {s.title}
                    </h3>
                    <p className="text-xs font-mono text-cyan-accent mt-0.5">{s.subtitle}</p>

                    <p className="mt-2 text-xs text-text-secondary leading-relaxed">
                      {s.summary}
                    </p>
                  </div>

                  <div className="mt-3 pt-2.5 border-t border-space-border/60 flex items-center justify-between text-[11px] font-mono">
                    <span className="text-text-secondary">Latency</span>
                    <span className="text-white font-semibold">{s.latency}</span>
                  </div>
                </button>

                {/* Flow Connector Arrow */}
                {idx < PIPELINE_STAGES.length - 1 && (
                  <>
                    <div className="hidden lg:flex absolute -right-3 top-1/2 -translate-y-1/2 z-20 pointer-events-none items-center justify-center">
                      <div className="flex h-6 w-6 items-center justify-center rounded-full bg-space-dark border border-space-border text-cyan-accent shadow">
                        <ArrowRight className="h-3 w-3" />
                      </div>
                    </div>
                    <div className="flex lg:hidden justify-center py-1.5 text-cyan-accent/50">
                      <ArrowDown className="h-3.5 w-3.5" />
                    </div>
                  </>
                )}
              </div>
            );
          })}
        </div>

        {/* Selected Stage Detail Inspector */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeStage.id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            className="mt-8 rounded-xl border border-space-border bg-space-card/50 p-5 sm:p-6 backdrop-blur-sm"
          >
            <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-space-border/60">
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-space-dark border border-cyan-dim/30 text-cyan-accent">
                  {(() => {
                    const Icon = activeStage.icon;
                    return <Icon className="h-4 w-4" />;
                  })()}
                </div>
                <div>
                  <span className="text-[10px] font-mono text-cyan-accent font-bold uppercase">
                    Step {activeStage.step} Overview
                  </span>
                  <h3 className="text-base font-bold text-white">
                    {activeStage.title} — {activeStage.subtitle}
                  </h3>
                </div>
              </div>

              <div className="flex items-center gap-2 text-xs font-mono text-text-secondary bg-space-dark px-3 py-1 rounded-md border border-space-border">
                <Sparkles className="h-3 w-3 text-cyan-accent" />
                <span>Target: {activeStage.latency}</span>
              </div>
            </div>

            {/* Quick Specs Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4 text-xs">
              <div className="rounded-lg bg-space-dark/70 border border-space-border p-3">
                <span className="text-[10px] font-mono text-text-secondary block">INPUT</span>
                <p className="text-white font-medium mt-1">{activeStage.input}</p>
              </div>

              <div className="rounded-lg bg-space-dark/70 border border-space-border p-3">
                <span className="text-[10px] font-mono text-text-secondary block">OUTPUT</span>
                <p className="text-white font-medium mt-1">{activeStage.output}</p>
              </div>

              <div className="rounded-lg bg-space-dark/70 border border-space-border p-3">
                <span className="text-[10px] font-mono text-text-secondary block">KEY CAPABILITIES</span>
                <ul className="mt-1 space-y-1 text-text-secondary">
                  {activeStage.points.map((pt, i) => (
                    <li key={i} className="flex items-start gap-1.5">
                      <span className="text-cyan-accent">•</span>
                      <span>{pt}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </section>
  );
}
