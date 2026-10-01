'use client';

/**
 * "replay 2.0 s" — where the recorded run has got to.
 *
 * The lanes report the time of the newest snapshot they have shown; this keeps
 * running smoothly between them, and starts again when the replay loops.
 */

import { useCallback, useEffect, useRef, type MutableRefObject } from 'react';

export interface ReplayClock {
  /** Give the clock a snapshot time, in milliseconds into the recording. */
  note: (tMs: number) => void;
  /** Back to 0.0 s. Called when the page restarts every lane together. */
  reset: () => void;
  ref: MutableRefObject<{ base: number; at: number }>;
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

export function useReplayClock(): ReplayClock {
  const ref = useRef({ base: 0, at: 0 });

  const note = useCallback((tMs: number) => {
    const current = ref.current;
    // A big step backwards is the replay looping, not a slow lane.
    if (tMs > current.base || tMs < current.base - 2000) {
      ref.current = { base: tMs, at: now() };
    }
  }, []);

  // The loop is the page's decision, not something to be inferred from the
  // stream: the counters restart on the same tick, so the clock does too and
  // the two always agree.
  const reset = useCallback(() => {
    ref.current = { base: 0, at: 0 };
  }, []);

  return { note, reset, ref };
}

export function ReplayTimer({ clock, label = 'replay' }: { clock: ReplayClock; label?: string }) {
  const valueRef = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const el = valueRef.current;
      if (!el) return;
      const { base, at } = clock.ref.current;
      const ms = at === 0 ? 0 : base + (now() - at);
      const text = `${(Math.max(0, ms) / 1000).toFixed(1)} s`;
      if (el.textContent !== text) el.textContent = text;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [clock]);

  return (
    <span className="mono flex items-baseline gap-1.5 whitespace-nowrap lg:gap-2">
      <span className="text-fg-muted text-[11px] lg:text-[12px]">{label}</span>
      <span ref={valueRef} className="text-[13px] font-medium lg:text-[14px]">
        0.0 s
      </span>
    </span>
  );
}
