/**
 * ============================================================================
 * OWNER: Frontend Developer 1
 * PURPOSE: Global Navigation Bar with Space/ISRO styling and mobile support.
 *
 * HOW TO EDIT:
 * 1. Add/modify nav links in the `NAV_LINKS` array.
 * 2. Update status badge to pull from backend `/api/v1/system/status`.
 * ============================================================================
 */

'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Rocket, Download, BookOpen, PlayCircle, ShieldCheck } from 'lucide-react';

const NAV_LINKS = [
  { href: '/', label: 'Overview', icon: Rocket },
  { href: '/downloads', label: 'Downloads', icon: Download },
  { href: '/docs', label: 'Run Guide', icon: BookOpen },
  { href: '/demo', label: 'Live Sandbox', icon: PlayCircle },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 w-full border-b border-space-border bg-space-bg/90 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Brand */}
        <Link href="/" className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-cyan-dim/20 border border-cyan-accent text-cyan-accent">
            <Rocket className="h-5 w-5" />
          </div>
          <div className="flex flex-col">
            <span className="font-mono text-sm font-bold tracking-wider text-white">
              ASTRA<span className="text-cyan-accent">-E</span>
            </span>
            <span className="text-[10px] tracking-widest text-text-secondary uppercase">
              BAS • SIH 26174
            </span>
          </div>
        </Link>

        {/* Desktop Links */}
        <nav className="hidden md:flex items-center gap-1">
          {NAV_LINKS.map(({ href, label, icon: Icon }) => {
            const isActive = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={`flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-space-card text-cyan-accent border border-cyan-dim/40'
                    : 'text-text-secondary hover:bg-space-card/50 hover:text-white'
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Launch Collector CTA */}
        <div className="flex items-center gap-3">
          <a
            href="https://astra-na124441s-projects.vercel.app/collector"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 rounded-lg border border-cyan-accent bg-cyan-accent/10 px-3.5 py-1.5 text-xs font-mono font-semibold text-cyan-accent hover:bg-cyan-accent/20 transition-all shadow-[0_0_12px_rgba(0,229,255,0.2)]"
          >
            <ShieldCheck className="h-3.5 w-3.5" />
            <span>COLLECTOR PWA ↗</span>
          </a>
        </div>
      </div>
    </header>
  );
}
