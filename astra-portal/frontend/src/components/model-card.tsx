/**
 * ============================================================================
 * OWNER: Frontend Developer 2
 * PURPOSE: Model Download Card Component with SHA-256 Copy and direct download.
 *
 * HOW TO EDIT:
 * 1. Customize styling of tags, badges, and download button.
 * 2. Add a dialog modal for advanced metadata inspection.
 * ============================================================================
 */

'use client';

import { useState } from 'react';
import { Download, Copy, Check, HardDrive, Cpu, Sparkles } from 'lucide-react';
import { ModelArtifact } from '@/lib/types';

interface ModelCardProps {
  model: ModelArtifact;
}

export function ModelCard({ model }: ModelCardProps) {
  const [copied, setCopied] = useState(false);

  const copySha256 = () => {
    navigator.clipboard.writeText(model.sha256);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`relative rounded-2xl border p-6 flex flex-col justify-between transition-all ${
      model.recommended
        ? 'border-cyan-accent bg-space-card/90 shadow-[0_0_25px_rgba(0,229,255,0.15)]'
        : 'border-space-border bg-space-card/40 hover:bg-space-card/70'
    }`}>
      {model.recommended && (
        <div className="absolute -top-3 right-6 flex items-center gap-1 rounded-full bg-cyan-accent px-3 py-0.5 text-[10px] font-mono font-bold uppercase tracking-wider text-space-bg">
          <Sparkles className="h-3 w-3" />
          <span>Recommended for Edge</span>
        </div>
      )}

      <div>
        <div className="flex items-start justify-between gap-4 mb-3">
          <div>
            <span className="font-mono text-[10px] text-cyan-accent uppercase tracking-wider">
              {model.format} • {model.precision}
            </span>
            <h3 className="text-lg font-bold text-white mt-1">{model.title}</h3>
          </div>
          <span className="rounded-lg bg-space-dark px-2.5 py-1 text-xs font-mono font-bold text-white border border-space-border">
            {model.size_mb} MB
          </span>
        </div>

        <p className="text-xs text-text-secondary leading-relaxed mb-4">
          {model.description}
        </p>

        <div className="flex items-center gap-2 text-[11px] font-mono text-text-secondary mb-4">
          <Cpu className="h-3.5 w-3.5 text-cyan-accent" />
          <span>Target: {model.target_device}</span>
        </div>
      </div>

      <div className="pt-4 border-t border-space-border/60">
        {/* SHA-256 Copy Row */}
        <div className="flex items-center justify-between gap-2 mb-4 bg-space-dark/80 p-2 rounded-xl border border-space-border">
          <div className="flex items-center gap-1.5 overflow-hidden">
            <HardDrive className="h-3 w-3 text-text-secondary flex-shrink-0" />
            <span className="font-mono text-[10px] text-text-secondary truncate">
              {model.sha256}
            </span>
          </div>
          <button
            onClick={copySha256}
            className="flex items-center gap-1 px-2 py-1 rounded bg-space-card hover:bg-cyan-accent/20 text-cyan-accent text-[10px] font-mono flex-shrink-0 transition-colors"
          >
            {copied ? <Check className="h-3 w-3 text-emerald-accent" /> : <Copy className="h-3 w-3" />}
            <span>{copied ? 'Copied!' : 'SHA'}</span>
          </button>
        </div>

        {/* Download Button */}
        <a
          href={model.download_url}
          target="_blank"
          rel="noopener noreferrer"
          className="w-full flex items-center justify-center gap-2 rounded-xl bg-cyan-accent/10 border border-cyan-accent py-2.5 text-xs font-mono font-bold text-cyan-accent hover:bg-cyan-accent hover:text-space-bg transition-all shadow-[0_0_12px_rgba(0,229,255,0.2)]"
        >
          <Download className="h-4 w-4" />
          <span>DOWNLOAD {model.format}</span>
        </a>
      </div>
    </div>
  );
}
