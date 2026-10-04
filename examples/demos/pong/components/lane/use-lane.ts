'use client';

import { useEffect, useRef, useState } from 'react';
import type { LaneConfig, LaneRunner, LaneStatus, Snapshot } from '@/lib/game/types';
import { isReturn } from '@/lib/record/stats';

/** How many decisions we keep for the rolling average. */
export const HISTORY = 24;
export interface LaneSource {
  config: LaneConfig;
  runner: LaneRunner;
}

export interface LaneFeed {
  snapshot: Snapshot | null;
  /** Latencies of the last HISTORY decisions, oldest first. */
  latencies: number[];
}

const EMPTY: LaneFeed = { snapshot: null, latencies: [] };

interface OwnedFeed extends LaneFeed {
  owner: LaneRunner | null;
}

/** Append one latency, keeping at most HISTORY entries. */
export function pushLatency(values: number[], ms: number): number[] {
  return values.length < HISTORY ? [...values, ms] : [...values.slice(1), ms];
}

/**
 * Subscribe to one runner. Also keeps the latency history the row renders.
 *
 * State is only ever set from the subscription callback — the feed carries the
 * runner it belongs to, so a swapped runner starts from nothing without a
 * synchronous reset.
 */
export function useLaneFeed(runner: LaneRunner | null | undefined): LaneFeed {
  const [feed, setFeed] = useState<OwnedFeed>({ ...EMPTY, owner: null });

  useEffect(() => {
    if (!runner) return;
    return runner.on('snapshot', (s) => {
      setFeed((prev) => {
        const base: LaneFeed = prev.owner === runner ? prev : EMPTY;
        const latencies =
          s.latencyMs !== null && Number.isFinite(s.latencyMs)
            ? pushLatency(base.latencies, s.latencyMs)
            : base.latencies;
        return { owner: runner, snapshot: s, latencies };
      });
    });
  }, [runner]);

  if (!runner || feed.owner !== runner) return EMPTY;
  return feed;
}

/**
 * Build lane runners inside an effect and dispose them on unmount.
 *
 * A LaneRunner is single-use: `stop()` is terminal, so a runner cannot be
 * restarted. React StrictMode mounts, cleans up and mounts again in dev, which
 * would leave every lane dead. Creating them in the effect means a remount gets
 * fresh runners. Change `key` to deal a new set (e.g. "play again").
 */
export function useLaneSources(make: () => LaneSource[], key: string): LaneSource[] {
  const [sources, setSources] = useState<LaneSource[]>([]);
  const makeRef = useRef(make);

  useEffect(() => {
    makeRef.current = make;
  });

  useEffect(() => {
    let live = true;
    const built = makeRef.current();
    // Out of the effect body so this is a subscription-style update, not a
    // cascading render.
    queueMicrotask(() => {
      if (live) setSources(built);
    });
    return () => {
      live = false;
      setSources([]);
      for (const s of built) s.runner.stop();
    };
  }, [key]);

  return sources;
}

/* ------------------------------------------------- what the row displays */

const TERMINAL: ReadonlySet<LaneStatus> = new Set<LaneStatus>(['over', 'credits', 'error']);

export interface LaneNumbers {
  /** The latency of the most recent decision that landed. */
  lastLatencyMs: number | null;
  /** `performance.now()` when the last snapshot arrived. */
  landedAt: number;
  /**
   * Bumped once per landed decision. The pulse and the flash key off this, so
   * they fire exactly once however often the row re-renders.
   */
  landedKey: number;
  /** Decisions so far. The tick counter is monotonic for a whole game. */
  decisions: number;
  /** Ball reversals at a paddle, counted off the snapshots this client saw. */
  returns: number;
  /** The lane has stopped; nothing is in flight and the number holds. */
  stopped: boolean;
}

const NOTHING: LaneNumbers = {
  lastLatencyMs: null,
  landedAt: 0,
  landedKey: 0,
  decisions: 0,
  returns: 0,
  stopped: false,
};

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

/**
 * Everything the number column shows, derived from the snapshot stream.
 *
 * This is React's "adjust state when a prop changes" pattern rather than an
 * effect: the new snapshot is accounted for in the same render it arrives in,
 * so the number never shows one decision behind.
 */
export function useLaneNumbers(snapshot: Snapshot | null): LaneNumbers {
  const [seen, setSeen] = useState<Snapshot | null>(null);
  const [state, setState] = useState<LaneNumbers>(NOTHING);

  if (snapshot !== seen) {
    const restart = seen !== null && snapshot !== null && snapshot.tick < seen.tick;
    const base = restart || snapshot === null ? NOTHING : state;
    setSeen(snapshot);
    if (snapshot === null) {
      setState(NOTHING);
    } else {
      const landed = snapshot.latencyMs !== null && Number.isFinite(snapshot.latencyMs);
      const returned = !restart && seen !== null && isReturn(seen, snapshot);
      setState({
        lastLatencyMs: landed ? snapshot.latencyMs : base.lastLatencyMs,
        landedAt: now(),
        landedKey: base.landedKey + (landed ? 1 : 0),
        decisions: snapshot.tick,
        returns: base.returns + (returned ? 1 : 0),
        stopped: TERMINAL.has(snapshot.status),
      });
    }
  }

  return snapshot === null ? NOTHING : state;
}

/** "5 decisions · 0 returns", with the singular where it is due. */
export function countsLine(decisions: number, returns: number): string {
  const d = `${decisions} decision${decisions === 1 ? '' : 's'}`;
  const r = `${returns} return${returns === 1 ? '' : 's'}`;
  return `${d} · ${r}`;
}
