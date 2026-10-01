import type { Metadata, Viewport } from 'next';
import { GeistSans } from 'geist/font/sans';
import { GeistMono } from 'geist/font/mono';
import { SITE_NAME, TAGLINE } from '@/lib/config/site';
import './globals.css';

/**
 * Type: Geist and Geist Mono, self-hosted through next/font (the `geist` package wraps next/font/local), so there is
 * no layout shift and no network at build time.
 *
 * Changed from upstream (ably-labs/jev-pong, Apache-2.0): no analytics, no social card, no vendor metadata.
 */

export const metadata: Metadata = {
  title: SITE_NAME,
  description: TAGLINE,
  applicationName: SITE_NAME,
};

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#F5F6F8' },
    { media: '(prefers-color-scheme: dark)', color: '#0A0B0F' },
  ],
  colorScheme: 'light dark',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${GeistSans.variable} ${GeistMono.variable} h-full antialiased`}
    >
      <body className="bg-bg text-fg flex min-h-full flex-col">{children}</body>
    </html>
  );
}
