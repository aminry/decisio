'use client';

/**
 * The court's colours, read out of the CSS custom properties in
 * app/globals.css.
 *
 * A canvas cannot inherit `var(--court)`, so the one place the site turns a
 * token into a colour string is here. Nothing else in components/ names a
 * colour: the DOM uses the Tailwind mapping, the canvas uses this.
 */

import { useEffect, useState } from 'react';
import type { CourtTheme } from '@/lib/render/court';

/** Fallbacks only matter before the stylesheet has applied. */
const FALLBACK: Record<string, string> = {
  '--court': '#FFFFFF',
  '--hair': '#E3E5EA',
  '--court-line': '#D5D8DE',
  '--paddle': '#14161C',
  '--ball': '#5B606B',
  '--accent': '#FF5416',
  '--fg-muted': '#5B606B',
};

function token(style: CSSStyleDeclaration, name: string): string {
  return style.getPropertyValue(name).trim() || FALLBACK[name] || '#000000';
}

/**
 * The five court colours for one lane. The first lane's ball and trail are `--accent`;
 * every other lane's are `--ball`. Paddles are `--paddle` in every lane.
 */
export function readCourtTheme(model: string): CourtTheme {
  if (typeof window === 'undefined') {
    return {
      court: FALLBACK['--court'],
      hair: FALLBACK['--hair'],
      courtLine: FALLBACK['--court-line'],
      paddle: FALLBACK['--paddle'],
      ball: model === 'sys1' ? FALLBACK['--accent'] : FALLBACK['--ball'],
      fgMuted: FALLBACK['--fg-muted'],
    };
  }
  const style = getComputedStyle(document.documentElement);
  return {
    court: token(style, '--court'),
    hair: token(style, '--hair'),
    courtLine: token(style, '--court-line'),
    paddle: token(style, '--paddle'),
    ball: token(style, model === 'sys1' ? '--accent' : '--ball'),
    fgMuted: token(style, '--fg-muted'),
  };
}

/** True when the visitor has asked for less motion (motion note 08). */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    if (typeof matchMedia !== 'function') return;
    const media = matchMedia('(prefers-reduced-motion: reduce)');
    const read = () => setReduced(media.matches);
    read();
    media.addEventListener('change', read);
    return () => media.removeEventListener('change', read);
  }, []);

  return reduced;
}
