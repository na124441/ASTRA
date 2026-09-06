/**
 * ============================================================================
 * OWNER: Frontend Developer 2
 * PURPOSE: Software & Model Download Hub Page.
 * ============================================================================
 */

import { MOCK_MODELS, MOCK_BINARIES, MOCK_SAMPLES } from '@/lib/mock-data';
import { ModelCard } from '@/components/model-card';
import { ChecksumVerifier } from '@/components/checksum-verifier';
import { Download, Film, HardDrive } from 'lucide-react';

export default function DownloadsPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-16">
      {/* Header */}
      <div className="max-w-2xl mb-12">
        <p className="font-mono text-xs text-cyan-accent uppercase tracking-widest">
          Artifact Distribution Hub
        </p>
        <h1 className="mt-2 text-3xl font-extrabold text-white sm:text-4xl">
          Downloads: Models, Binaries & Sample Datasets
        </h1>
        <p className="mt-3 text-sm text-text-secondary">
          Download edge-quantized ONNX models, compiled desktop executables, and experimental sample video sequences.
        </p>
      </div>

      {/* Section 1: Models */}
      <div className="mb-16">
        <div className="flex items-center gap-2 mb-6">
          <HardDrive className="h-5 w-5 text-cyan-accent" />
          <h2 className="text-xl font-bold text-white">Trained Action Recognition Models</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {MOCK_MODELS.map((model) => (
            <ModelCard key={model.id} model={model} />
          ))}
        </div>

        {/* Checksum Verifier */}
        <ChecksumVerifier />
      </div>

      {/* Section 2: Executable Binaries */}
      <div className="mb-16 pt-12 border-t border-space-border/60">
        <div className="flex items-center gap-2 mb-6">
          <Download className="h-5 w-5 text-emerald-accent" />
          <h2 className="text-xl font-bold text-white">Desktop & CLI Engine Binaries</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {MOCK_BINARIES.map((bin) => (
            <div key={bin.id} className="rounded-2xl border border-space-border bg-space-card/40 p-6 flex items-center justify-between">
              <div>
                <span className="text-[10px] font-mono text-emerald-accent uppercase">
                  {bin.platform} (x64) • v{bin.version}
                </span>
                <h3 className="text-base font-bold text-white mt-1">{bin.filename}</h3>
                <p className="text-xs font-mono text-text-secondary mt-1">Size: {bin.size_mb} MB</p>
              </div>

              <a
                href={bin.download_url}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-xl bg-space-dark border border-space-border px-4 py-2 text-xs font-mono font-bold text-white hover:border-emerald-accent transition-colors"
              >
                DOWNLOAD ↗
              </a>
            </div>
          ))}
        </div>
      </div>

      {/* Section 3: EXP001 Sample Videos */}
      <div className="pt-12 border-t border-space-border/60">
        <div className="flex items-center gap-2 mb-6">
          <Film className="h-5 w-5 text-amber-accent" />
          <h2 className="text-xl font-bold text-white">EXP001 Benchmark Video Sequences</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {MOCK_SAMPLES.map((s) => (
            <div key={s.id} className="rounded-2xl border border-space-border bg-space-card/40 p-6 flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-cyan-accent uppercase">
                    {s.experiment_id} • {s.run_id}
                  </span>
                  <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded ${
                    s.scenario === 'NOMINAL' ? 'bg-emerald-accent/20 text-emerald-accent' : 'bg-amber-accent/20 text-amber-accent'
                  }`}>
                    {s.scenario}
                  </span>
                </div>
                <h3 className="text-sm font-bold text-white mt-1">{s.camera}</h3>
                <p className="text-xs text-text-secondary mt-1">Duration: {s.duration_sec}s • {s.size_mb} MB</p>
              </div>

              <a
                href={s.download_url}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-xl bg-space-dark border border-space-border px-4 py-2 text-xs font-mono font-bold text-white hover:border-cyan-accent transition-colors"
              >
                DOWNLOAD MP4 ↗
              </a>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
