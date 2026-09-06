/**
 * ============================================================================
 * OWNER: Frontend Developer 1
 * PURPOSE: Interactive Architecture Pipeline visualizing the 4 core stages
 *          of ASTRA-E action recognition.
 * ============================================================================
 */

import { Camera, Cpu, AlertTriangle, CloudUpload } from 'lucide-react';

const PIPELINE_STAGES = [
  {
    step: '01',
    title: 'Multi-Camera Perception',
    subtitle: 'Synchronized 1080p Landscape',
    desc: 'High-speed edge capture across top-down, side-profile, and chest-mount angles for zero-g occlusion resistance.',
    icon: Camera,
    color: 'text-cyan-accent',
    border: 'border-cyan-dim/40',
  },
  {
    step: '02',
    title: 'Spatial-Temporal Engine',
    subtitle: 'ONNX / TensorRT Edge Inference',
    desc: 'Deep action recognition model classifying experimental sub-tasks (pipetting, chamber seals, centrifuge timing).',
    icon: Cpu,
    color: 'text-emerald-accent',
    border: 'border-emerald-accent/40',
  },
  {
    step: '03',
    title: 'Mistake Guard & Anomaly HUD',
    subtitle: 'Nominal vs. Fault Detection',
    desc: 'Real-time astronaut alerts when a procedural step is skipped, inverted, or executed with wrong lab implements.',
    icon: AlertTriangle,
    color: 'text-amber-accent',
    border: 'border-amber-accent/40',
  },
  {
    step: '04',
    title: 'Audited Telemetry Sync',
    subtitle: 'Hugging Face Private Repository',
    desc: 'Immutable SHA-256 validated chunked uploads to ground station dataset repository for post-mission flight reviews.',
    icon: CloudUpload,
    color: 'text-cyan-accent',
    border: 'border-cyan-dim/40',
  },
];

export function ArchitectureDiagram() {
  return (
    <section className="py-20 border-t border-space-border/60 bg-space-dark/30">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-2xl mx-auto mb-16">
          <p className="font-mono text-xs text-cyan-accent uppercase tracking-widest">
            Pipeline Architecture
          </p>
          <h2 className="mt-2 text-3xl font-bold tracking-tight text-white sm:text-4xl">
            From Camera Pixels to Astronaut Guidance
          </h2>
          <p className="mt-4 text-sm text-text-secondary">
            How ASTRA-E continuously inspects and validates experimental runs inside the station lab module.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {PIPELINE_STAGES.map((s) => {
            const Icon = s.icon;
            return (
              <div
                key={s.step}
                className={`relative rounded-2xl border ${s.border} bg-space-card/50 p-6 flex flex-col justify-between hover:bg-space-card transition-all`}
              >
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <span className="font-mono text-xs font-bold text-text-secondary">
                      STAGE {s.step}
                    </span>
                    <Icon className={`h-5 w-5 ${s.color}`} />
                  </div>
                  <h3 className="text-lg font-bold text-white">{s.title}</h3>
                  <p className="text-xs font-mono text-cyan-accent mt-0.5">{s.subtitle}</p>
                  <p className="mt-3 text-xs text-text-secondary leading-relaxed">
                    {s.desc}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
