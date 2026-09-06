/**
 * ============================================================================
 * OWNER: Frontend Developer 1
 * PURPOSE: Main Landing Page combining Hero, Architecture Visualizer,
 *          PWA Collector Launchpad, and System Capabilities.
 *          Streamlined with medium typography and simple, clear content.
 * ============================================================================
 */

import { HeroSection } from '@/components/hero-section';
import { ArchitectureDiagram } from '@/components/architecture-diagram';
import Link from 'next/link';
import {
  Shield,
  Smartphone,
  Zap,
  CheckCircle2,
  HardDriveDownload,
} from 'lucide-react';

export default function HomePage() {
  return (
    <div className="flex flex-col">
      {/* 1. Hero Section */}
      <HeroSection />

      {/* 2. 4-Step Architecture Pipeline */}
      <ArchitectureDiagram />

      {/* 3. PWA Collector Launchpad Section */}
      <section className="py-20 border-t border-space-border/60 bg-gradient-to-b from-space-dark/30 to-space-bg relative overflow-hidden">
        <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
            {/* Left Info Column */}
            <div className="lg:col-span-7">
              <div className="inline-flex items-center gap-2 rounded-full border border-emerald-accent/40 bg-emerald-accent/10 px-3 py-1 text-xs font-mono text-emerald-accent mb-3">
                <Smartphone className="h-3.5 w-3.5" />
                <span>COLLECTOR PWA</span>
              </div>

              <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight leading-snug">
                Zero-Install Recording Terminal
              </h2>

              <p className="mt-3 text-sm text-text-secondary leading-relaxed max-w-xl">
                Turn any phone or tablet onboard the station into a calibrated multi-camera recorder. Works offline, buffers video locally, and syncs automatically when the link is restored.
              </p>

              {/* Simple Feature Badges */}
              <div className="mt-5 grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-xs font-mono text-text-secondary">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-accent flex-shrink-0" />
                  <span>Landscape 1080p camera recording</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-accent flex-shrink-0" />
                  <span>Offline-first local storage buffer</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-accent flex-shrink-0" />
                  <span>SHA-256 integrity verification</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-accent flex-shrink-0" />
                  <span>Direct ground station sync</span>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="mt-6 flex flex-wrap items-center gap-3">
                <a
                  href="https://astra-na124441s-projects.vercel.app/collector"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 rounded-xl bg-emerald-accent px-5 py-2.5 text-xs sm:text-sm font-mono font-bold text-space-bg hover:bg-emerald-accent/90 transition-all shadow-[0_0_15px_rgba(0,230,118,0.3)] active:scale-[0.98]"
                >
                  <Smartphone className="h-4 w-4" />
                  <span>Launch Collector PWA ↗</span>
                </a>

                <Link
                  href="/downloads"
                  className="flex items-center gap-2 rounded-xl border border-space-border bg-space-card/60 px-4 py-2.5 text-xs sm:text-sm font-mono text-text-secondary hover:text-white transition-all"
                >
                  <HardDriveDownload className="h-4 w-4 text-cyan-accent" />
                  <span>Sample Datasets</span>
                </Link>
              </div>
            </div>

            {/* Right Card: Simplified PWA Terminal Preview */}
            <div className="lg:col-span-5">
              <div className="rounded-xl border border-emerald-accent/30 bg-space-dark/80 p-4 shadow-lg">
                <div className="flex items-center justify-between pb-2.5 border-b border-space-border/60">
                  <div className="flex items-center gap-2 font-mono text-xs text-white font-bold">
                    <span className="h-2 w-2 rounded-full bg-emerald-accent animate-pulse" />
                    <span>ASTRA Collector v1.0</span>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-space-card text-emerald-accent">
                    Online PWA
                  </span>
                </div>

                <div className="mt-3 space-y-2.5">
                  <div className="rounded-lg bg-space-card/50 p-2.5 border border-space-border text-xs font-mono">
                    <span className="text-[10px] text-text-secondary block">SESSION MODE</span>
                    <span className="text-white font-bold">EXP001 Multi-Angle Capture</span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
                    <div className="rounded bg-space-card/30 p-2 border border-space-border">
                      <span className="text-text-secondary block">BUFFER</span>
                      <span className="text-emerald-accent font-bold">IndexedDB OK</span>
                    </div>
                    <div className="rounded bg-space-card/30 p-2 border border-space-border">
                      <span className="text-text-secondary block">UPLOAD</span>
                      <span className="text-cyan-accent font-bold">Chunked Sync</span>
                    </div>
                  </div>

                  <a
                    href="https://astra-na124441s-projects.vercel.app/collector"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block text-center py-2 rounded-lg bg-space-card hover:bg-space-card/80 border border-emerald-accent/30 text-xs font-mono font-bold text-emerald-accent transition-colors"
                  >
                    Open Standalone PWA ↗
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Core Features Grid */}
      <section className="py-20 border-t border-space-border/60">
        <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-xl mx-auto mb-12">
            <p className="font-mono text-xs text-cyan-accent uppercase tracking-widest">
              Core Capabilities
            </p>
            <h2 className="mt-1.5 text-2xl sm:text-3xl font-bold tracking-tight text-white">
              Built for Space Station Conditions
            </h2>
            <p className="mt-2 text-xs sm:text-sm text-text-secondary">
              Engineered to handle microgravity lighting, low-power edge compute, and intermittent ground connectivity.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="rounded-xl border border-space-border bg-space-card/30 p-6 hover:bg-space-card/50 transition-all">
              <div className="h-9 w-9 rounded-lg bg-cyan-dim/20 border border-cyan-accent/50 flex items-center justify-center text-cyan-accent mb-4">
                <Zap className="h-4 w-4" />
              </div>
              <h3 className="text-base font-bold text-white">Edge Accelerated</h3>
              <p className="mt-1.5 text-xs text-text-secondary leading-relaxed">
                Lightweight INT8 quantized models run locally on station hardware (Jetson, Raspberry Pi) without requiring internet connectivity.
              </p>
            </div>

            <div className="rounded-xl border border-space-border bg-space-card/30 p-6 hover:bg-space-card/50 transition-all">
              <div className="h-9 w-9 rounded-lg bg-emerald-accent/20 border border-emerald-accent/50 flex items-center justify-center text-emerald-accent mb-4">
                <Smartphone className="h-4 w-4" />
              </div>
              <h3 className="text-base font-bold text-white">Zero-Install PWA</h3>
              <p className="mt-1.5 text-xs text-text-secondary leading-relaxed">
                Browser-based Progressive Web App turns standard mobile devices into synchronized lab experiment recorders with local storage.
              </p>
            </div>

            <div className="rounded-xl border border-space-border bg-space-card/30 p-6 hover:bg-space-card/50 transition-all">
              <div className="h-9 w-9 rounded-lg bg-amber-accent/20 border border-amber-accent/50 flex items-center justify-center text-amber-accent mb-4">
                <Shield className="h-4 w-4" />
              </div>
              <h3 className="text-base font-bold text-white">Cryptographic Checks</h3>
              <p className="mt-1.5 text-xs text-text-secondary leading-relaxed">
                SHA-256 checksums and immutable audit trails ensure experiment data is genuine and free from transmission corruption.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
