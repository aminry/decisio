/**
 * Whole-run numbers for one lane: how many decisions, how fast, how many
 * returns and misses. The recorder writes these to public/replay-stats.json and
 * the clip's end card counts them up. Pure functions over Snapshot arrays; no
 * rendering here, so the recording path does not depend on the presentation.
 */
import type { ModelId, Snapshot } from '../game/types';

/** Whole-lane numbers, for the end card and for public/replay-stats.json. */
export interface LaneStats {
  model: ModelId;
  label: string;
  decisions: number;
  decisionsPerSecond: number;
  avgMs: number;
  p50Ms: number;
  p95Ms: number;
  returns: number;
  misses: number;
  score: [number, number];
  durationMs: number;
}

/** The ball reversed direction between two snapshots: a paddle returned it. */
export function isReturn(prev: Snapshot, next: Snapshot): boolean {
  if (prev.ball.vx === 0 || next.ball.vx === 0) return false;
  return Math.sign(prev.ball.vx) !== Math.sign(next.ball.vx);
}

/** The score changed between two snapshots: somebody missed. */
export function isMiss(prev: Snapshot, next: Snapshot): boolean {
  return prev.score[0] !== next.score[0] || prev.score[1] !== next.score[1];
}

/**
 * Whole-run numbers for a lane. `scripts/record-replay.ts` writes these to
 * public/replay-stats.json; the end card reads the `statsIn` variant instead.
 */
export function laneStats(
  model: ModelId,
  label: string,
  snapshots: Snapshot[],
): LaneStats {
  const latencies = snapshots
    .map((s) => s.latencyMs)
    .filter((ms): ms is number => ms != null && Number.isFinite(ms));
  const sorted = [...latencies].sort((a, b) => a - b);
  const durationMs =
    snapshots.length > 1 ? snapshots[snapshots.length - 1].t - snapshots[0].t : 0;

  let returns = 0;
  let misses = 0;
  for (let i = 1; i < snapshots.length; i += 1) {
    if (isReturn(snapshots[i - 1], snapshots[i])) returns += 1;
    if (isMiss(snapshots[i - 1], snapshots[i])) misses += 1;
  }

  const last = snapshots[snapshots.length - 1];
  const avg = sorted.length
    ? Math.round(sorted.reduce((a, b) => a + b, 0) / sorted.length)
    : 0;
  const p50 = sorted.length ? Math.round(sorted[Math.floor((sorted.length - 1) / 2)]) : 0;
  const p95 = sorted.length
    ? Math.round(sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))])
    : 0;

  return {
    model,
    label,
    decisions: latencies.length,
    decisionsPerSecond:
      durationMs > 0 ? Math.round((latencies.length / durationMs) * 1000 * 100) / 100 : 0,
    avgMs: avg,
    p50Ms: p50,
    p95Ms: p95,
    returns,
    misses,
    score: last ? [last.score[0], last.score[1]] : [0, 0],
    durationMs,
  };
}
