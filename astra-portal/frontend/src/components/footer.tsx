/**
 * ============================================================================
 * OWNER: Frontend Developer 1
 * PURPOSE: Global Space Footer with SIH 26174 credits and repository links.
 * ============================================================================
 */

import Link from 'next/link';

export function Footer() {
  return (
    <footer className="border-t border-space-border bg-space-dark/60 py-10 mt-20">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-6 text-xs text-text-secondary">
        <div>
          <p className="font-mono text-white font-semibold">
            ASTRA-E: Autonomous Space Task Recognition & Assistance for Experiments
          </p>
          <p className="mt-1">
            Smart India Hackathon 2026 • Problem Statement SIH 26174 • Bhartiya Antariksh Station
          </p>
        </div>

        <div className="flex items-center gap-6">
          <Link href="/downloads" className="hover:text-cyan-accent transition-colors">
            Models
          </Link>
          <Link href="/docs" className="hover:text-cyan-accent transition-colors">
            Run Guide
          </Link>
          <Link href="/demo" className="hover:text-cyan-accent transition-colors">
            Live Sandbox
          </Link>
          <a
            href="https://github.com/na124441/ASTRA"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-cyan-accent transition-colors font-mono"
          >
            GitHub ↗
          </a>
        </div>
      </div>
    </footer>
  );
}
