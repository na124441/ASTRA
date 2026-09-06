/**
 * ============================================================================
 * OWNER: Frontend Developer 2
 * PURPOSE: Interactive SHA-256 Checksum Verifier using browser Web Crypto API.
 * ============================================================================
 */

'use client';

import { useState } from 'react';
import { ShieldCheck, Upload, CheckCircle2, XCircle, RefreshCw } from 'lucide-react';

export function ChecksumVerifier() {
  const [fileName, setFileName] = useState('');
  const [computedSha256, setComputedSha256] = useState('');
  const [verifying, setVerifying] = useState(false);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setVerifying(true);

    try {
      const buffer = await file.arrayBuffer();
      const digestBuffer = await crypto.subtle.digest('SHA-256', buffer);
      const hashArray = Array.from(new Uint8Array(digestBuffer));
      const hashHex = hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
      setComputedSha256(hashHex);
    } catch {
      setComputedSha256('Calculation failed');
    }
    setVerifying(false);
  };

  return (
    <div className="rounded-2xl border border-space-border bg-space-card/40 p-6 mt-8">
      <div className="flex items-center gap-2 mb-2">
        <ShieldCheck className="h-5 w-5 text-cyan-accent" />
        <h3 className="text-base font-bold text-white">Local Model Integrity Verifier</h3>
      </div>
      <p className="text-xs text-text-secondary leading-relaxed">
        Verify that your downloaded model weights match official Bhartiya Antariksh Station releases using client-side Web Crypto (no file upload required).
      </p>

      <div className="mt-4 flex flex-col sm:flex-row items-center gap-4">
        <label className="flex items-center gap-2 rounded-xl border border-space-border bg-space-dark px-4 py-2.5 text-xs font-mono text-cyan-accent hover:border-cyan-dim cursor-pointer transition-colors">
          <Upload className="h-4 w-4" />
          <span>Select Local .onnx / .pt File</span>
          <input
            type="file"
            onChange={handleFileChange}
            className="hidden"
            accept=".onnx,.pt,.engine"
          />
        </label>

        {fileName && (
          <span className="text-xs font-mono text-text-secondary truncate max-w-xs">
            File: {fileName}
          </span>
        )}
      </div>

      {verifying && (
        <div className="mt-4 flex items-center gap-2 text-xs font-mono text-cyan-accent">
          <RefreshCw className="h-3.5 w-3.5 animate-spin" />
          <span>Computing SHA-256 hash in browser...</span>
        </div>
      )}

      {computedSha256 && !verifying && (
        <div className="mt-4 p-3 rounded-xl bg-space-dark border border-space-border flex flex-col gap-1 font-mono text-xs">
          <span className="text-[10px] text-text-secondary">CALCULATED SHA-256:</span>
          <span className="text-cyan-accent break-all">{computedSha256}</span>
        </div>
      )}
    </div>
  );
}
