/**
 * ============================================================================
 * OWNER: Frontend Developer 2
 * PURPOSE: Interactive Code Block with OS switcher (PowerShell / Bash)
 *          and 1-click clipboard copy button.
 * ============================================================================
 */

'use client';

import { useState } from 'react';
import { Copy, Check, Terminal } from 'lucide-react';

interface CodeBlockProps {
  title?: string;
  bashCode: string;
  powershellCode: string;
}

export function CodeBlock({ title, bashCode, powershellCode }: CodeBlockProps) {
  const [activeOS, setActiveOS] = useState<'powershell' | 'bash'>('powershell');
  const [copied, setCopied] = useState(false);

  const activeCode = activeOS === 'powershell' ? powershellCode : bashCode;

  const copyCode = () => {
    navigator.clipboard.writeText(activeCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="rounded-2xl border border-space-border bg-space-dark overflow-hidden my-6">
      {/* Header Bar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-space-card/60 border-b border-space-border text-xs">
        <div className="flex items-center gap-2 text-text-secondary font-mono">
          <Terminal className="h-3.5 w-3.5 text-cyan-accent" />
          <span>{title || 'Terminal Execution'}</span>
        </div>

        <div className="flex items-center gap-2">
          {/* OS Switcher Tabs */}
          <div className="flex rounded-lg bg-space-dark p-0.5 border border-space-border text-[11px] font-mono">
            <button
              onClick={() => setActiveOS('powershell')}
              className={`px-2.5 py-0.5 rounded-md transition-all ${
                activeOS === 'powershell'
                  ? 'bg-cyan-accent/20 text-cyan-accent font-bold'
                  : 'text-text-secondary hover:text-white'
              }`}
            >
              PowerShell
            </button>
            <button
              onClick={() => setActiveOS('bash')}
              className={`px-2.5 py-0.5 rounded-md transition-all ${
                activeOS === 'bash'
                  ? 'bg-cyan-accent/20 text-cyan-accent font-bold'
                  : 'text-text-secondary hover:text-white'
              }`}
            >
              Bash (Linux)
            </button>
          </div>

          {/* Copy Button */}
          <button
            onClick={copyCode}
            className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-space-card hover:bg-space-card/80 text-text-secondary hover:text-white font-mono text-[11px] transition-colors"
          >
            {copied ? <Check className="h-3 w-3 text-emerald-accent" /> : <Copy className="h-3 w-3" />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>
        </div>
      </div>

      {/* Code Area */}
      <pre className="p-4 font-mono text-xs text-text-primary overflow-x-auto leading-relaxed">
        <code>{activeCode}</code>
      </pre>
    </div>
  );
}
