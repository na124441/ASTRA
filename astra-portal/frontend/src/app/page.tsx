/**
 * ============================================================================
 * OWNER: Frontend Developer 1
 * PURPOSE: Main Landing Page combining Hero, Architecture, and Features.
 * ============================================================================
 */

import { HeroSection } from '@/components/hero-section';
import { ArchitectureDiagram } from '@/components/architecture-diagram';
import Link from 'next/link';
import { Shield, Smartphone, Zap } from 'lucide-react';

export default function HomePage() {
  return (
    <div className="flex flex-col">
      {/* Hero */}
      <HeroSection />

      {/* Architecture Visualizer */}
      <ArchitectureDiagram />

      {/* Capabilities Feature Grid */}
      <section className="py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="rounded-2xl border border-space-border bg-space-card/30 p-8">
              <div className="h-10 w-10 rounded-xl bg-cyan-dim/20 border border-cyan-accent flex items-center justify-center text-cyan-accent mb-4">
                <Zap className="h-5 w-5" />
              </div>
              <h3 className="text-lg font-bold text-white">Edge Accelerated</h3>
              <p className="mt-2 text-xs text-text-secondary leading-relaxed">
                Lightweight INT8 quantized models designed to run locally on low-power station SBCs with no internet dependency.
              </p>
            </div>

            <div className="rounded-2xl border border-space-border bg-space-card/30 p-8">
              <div className="h-10 w-10 rounded-xl bg-emerald-accent/20 border border-emerald-accent flex items-center justify-center text-emerald-accent mb-4">
                <Smartphone className="h-5 w-5" />
              </div>
              <h3 className="text-lg font-bold text-white">Zero-Install Collector</h3>
              <p className="mt-2 text-xs text-text-secondary leading-relaxed">
                Progressive Web App that lets any phone or tablet become a calibrated multi-angle experiment recorder.
              </p>
            </div>

            <div className="rounded-2xl border border-space-border bg-space-card/30 p-8">
              <div className="h-10 w-10 rounded-xl bg-amber-accent/20 border border-amber-accent flex items-center justify-center text-amber-accent mb-4">
                <Shield className="h-5 w-5" />
              </div>
              <h3 className="text-lg font-bold text-white">Cryptographic Verification</h3>
              <p className="mt-2 text-xs text-text-secondary leading-relaxed">
                SHA-256 signed uploads and immutable audit logs ensure scientific integrity across all experiment telemetry.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
