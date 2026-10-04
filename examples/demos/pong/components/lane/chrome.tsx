'use client';

/**
 * The furniture the page shares: the header and the footer.
 *
 * Changed from upstream (ably-labs/jev-pong, Apache-2.0): the sponsor lockups, the live-game navigation and the
 * brand marks are removed. The footer says where the code comes from.
 */

import type { ReactNode } from 'react';
import { CREDITS, SITE_NAME, UPSTREAM_URL } from '@/lib/config/site';

export function SiteHeader({ right }: { right?: ReactNode }) {
  return (
    <header className="flex h-[52px] items-center justify-between gap-3 lg:h-[72px] lg:gap-6">
      <span className="text-[15px] font-bold tracking-[-0.02em] whitespace-nowrap lg:text-[16px]">
        {SITE_NAME}
      </span>
      {right}
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="border-hair text-fg-muted flex flex-col gap-1 border-t pt-3 text-[12px] leading-[1.5] sm:pt-4 sm:text-[13px]">
      <p>{CREDITS}</p>
      <p>
        Based on{' '}
        <a href={UPSTREAM_URL} className="text-fg hover:underline">
          ably-labs/jev-pong
        </a>{' '}
        (Apache-2.0), changed to talk to any System One server. See ATTRIBUTION.md.
      </p>
    </footer>
  );
}
