'use client';

/**
 * The latency number — the biggest thing in the row at every width
 * (design/SPEC.md, must-not-get-wrong d).
 *
 * Motion note 04: it shows `max(last landed latency, time since the last
 * snapshot)`. While the count-up has overtaken the last reading the model is
 * still thinking, so the number drops to weight 400 in `--fg-inflight`. When a
 * decision lands it snaps to the real value, flares 1 -> 1.06 -> 1 over 180 ms
 * and flashes to pure white (or pure black on paper) for 120 ms.
 *
 * The digits are written straight to the DOM node from one rAF loop rather
 * than through React state: a lane updates sixty times a second and must not
 * re-render the page to do it.
 *
 * What it says is decided by `laneNumberAt` in lib/ui/lane-number.ts, which is
 * pure and tested — including the state a `live` lane has and a recording does
 * not: two missed heartbeats means the connection rather than a slow model, so
 * the count stops and the row says "waiting…" until a frame lands.
 */

import { useEffect, useRef, useState } from 'react';
import { laneNumberAt } from '@/lib/ui/lane-number';
import { countsLine, type LaneNumbers } from './use-lane';
import { usePrefersReducedMotion } from './theme';

export type NumberScale = 'lane' | 'play' | 'watch';

/**
 * Type scale from design/SPEC.md: 48 desktop, 34 phone and watch. The play
 * surface goes bigger than the spec's 48 on purpose — it is the one page where
 * the number is the claim rather than a column, and it has the room.
 */
const SCALE: Record<NumberScale, { number: string; unit: string; counts: string; peak: number }> = {
  lane: {
    number: 'text-[34px] lg:text-[48px]',
    unit: 'text-[13px] lg:text-[16px]',
    counts: 'text-[12px] lg:text-[13px]',
    peak: 1.06,
  },
  play: {
    number: 'text-[44px] lg:text-[64px]',
    unit: 'text-[15px] lg:text-[20px]',
    counts: 'text-[12px] lg:text-[13px]',
    peak: 1.12,
  },
  watch: {
    number: 'text-[34px] lg:text-[48px]',
    unit: 'text-[13px] lg:text-[16px]',
    counts: 'text-[12px] lg:text-[13px]',
    peak: 1.1,
  },
};

const PULSE_MS = 180;
const PULSE_OUT_MS = 60;

export interface LaneNumberProps {
  /** The first lane's number is `--accent`; every other lane's is `--fg`. */
  model: string;
  numbers: LaneNumbers;
  scale?: NumberScale;
  /** Show the "N decisions · M returns" line under the number. */
  counts?: boolean;
  /**
   * This lane is a live game, so its frames are heartbeated and silence means
   * something. Leave it off for a recording.
   */
  live?: boolean;
  /**
   * Show "<model> is thinking" above the number while a decision is in flight.
   * On /play it is the answer to "is anything happening?" during the two or
   * three hundred milliseconds when the ball is still.
   */
  thinkingLabel?: string;
  className?: string;
}

export function LaneNumber({
  model,
  numbers,
  scale = 'lane',
  counts = true,
  live = false,
  thinkingLabel,
  className,
}: LaneNumberProps) {
  const valueRef = useRef<HTMLSpanElement | null>(null);
  const unitRef = useRef<HTMLSpanElement | null>(null);
  const thinkingRef = useRef<HTMLSpanElement | null>(null);
  const stateRef = useRef(numbers);
  const stalledRef = useRef(false);
  const [stalled, setStalled] = useState(false);
  const reduced = usePrefersReducedMotion();
  const sizes = SCALE[scale];
  const landedColor = model === 'sys1' ? 'var(--accent)' : 'var(--fg)';

  // The rAF loop below reads the latest numbers without re-subscribing.
  useEffect(() => {
    stateRef.current = numbers;
  }, [numbers]);

  // The flare. One animation per landed decision, never per frame.
  useEffect(() => {
    const el = valueRef.current;
    if (!el || reduced || numbers.landedKey === 0) return;
    if (typeof el.animate !== 'function') return;
    el.animate(
      [
        { transform: 'scale(1)', offset: 0 },
        { transform: `scale(${sizes.peak})`, offset: PULSE_OUT_MS / PULSE_MS },
        { transform: 'scale(1)', offset: 1 },
      ],
      { duration: PULSE_MS, easing: 'ease-out' },
    );
  }, [numbers.landedKey, reduced, sizes.peak]);

  // The digits. `max(lastLatency, elapsed)`, written every frame.
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const el = valueRef.current;
      const unit = unitRef.current;
      if (!el) return;

      const nowMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
      const { value, inFlight, waiting, flashing } = laneNumberAt(
        { ...stateRef.current, live },
        nowMs,
      );
      // The caption is the one thing here that is React's: it changes once per
      // stall, not once per frame.
      if (waiting !== stalledRef.current) {
        stalledRef.current = waiting;
        setStalled(waiting);
      }

      const text = value === null ? '—' : String(Math.round(value));
      if (el.textContent !== text) el.textContent = text;

      // Nothing has landed yet, so there is no reading to colour. A 64px orange
      // dash reads as a number; a muted one reads as the placeholder it is.
      const color =
        value === null
          ? 'var(--fg-muted)'
          : inFlight
            ? 'var(--fg-inflight)'
            : flashing && !reduced
              ? 'var(--flash)'
              : landedColor;
      if (el.style.color !== color) el.style.color = color;
      const weight = inFlight ? '400' : '500';
      if (el.style.fontWeight !== weight) el.style.fontWeight = weight;
      // The unit follows the number in flight, and is muted otherwise.
      const unitColor = inFlight ? 'var(--fg-inflight)' : 'var(--fg-muted)';
      if (unit && unit.style.color !== unitColor) unit.style.color = unitColor;

      // "thinking": the same in-flight fact, said in words for the one
      // surface where the ball has stopped and nothing else explains why.
      const think = thinkingRef.current;
      if (think) {
        const shown = inFlight && !waiting ? '1' : '0';
        if (think.style.opacity !== shown) think.style.opacity = shown;
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [landedColor, live, reduced]);

  const line = countsLine(numbers.decisions, numbers.returns);

  return (
    <div className={`flex flex-col items-end gap-[3px] lg:gap-[5px] ${className ?? ''}`}>
      {thinkingLabel !== undefined && (
        <span
          ref={thinkingRef}
          aria-hidden
          className="fade-150 text-fg-muted flex items-center gap-[7px] text-[12px] font-medium lg:text-[13px]"
          style={{ opacity: 0 }}
        >
          <span className="thinking-dot" />
          {thinkingLabel}
        </span>
      )}
      <div className="mono flex items-baseline gap-[5px] leading-none lg:gap-1.5">
        <span
          ref={valueRef}
          className={`${sizes.number} tracking-[-0.02em]`}
          style={{ color: landedColor, fontWeight: 500, display: 'inline-block' }}
        >
          —
        </span>
        <span ref={unitRef} className={sizes.unit} style={{ color: 'var(--fg-muted)' }}>
          ms
        </span>
      </div>
      {stalled ? (
        <span className={`dip text-fg-muted ${sizes.counts}`}>waiting…</span>
      ) : (
        counts && (
          // Re-keyed on the text so the 120 ms dip replays whenever it changes.
          <span key={line} className={`dip text-fg-muted ${sizes.counts}`}>
            {line}
          </span>
        )
      )}
    </div>
  );
}

export default LaneNumber;
