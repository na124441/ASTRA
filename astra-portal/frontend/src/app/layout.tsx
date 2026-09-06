/**
 * ============================================================================
 * OWNER: Frontend Developer 1
 * PURPOSE: Root App Layout wrapping all pages with Navbar, Footer, and theme.
 *
 * HOW TO EDIT:
 * 1. Add global providers (Theme, Toast, Lenis smooth scroll).
 * 2. Update page SEO metadata (title, description).
 * ============================================================================
 */

import type { Metadata } from 'next';
import './globals.css';
import { Navbar } from '@/components/navbar';
import { Footer } from '@/components/footer';

export const metadata: Metadata = {
  title: 'ASTRA-E — Autonomous Space Task Recognition for Bhartiya Antariksh Station',
  description:
    'Real-time edge action recognition, multi-angle spatial validation, and interactive astronaut assistance for space laboratory experiments (SIH 26174).',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-space-bg text-text-primary flex flex-col antialiased selection:bg-cyan-accent selection:text-space-bg">
        <Navbar />
        <main className="flex-1">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
