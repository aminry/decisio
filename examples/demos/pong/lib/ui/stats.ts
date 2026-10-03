/**
 * The measured run, as the page prints it.
 *
 * Every number is computed from the recording itself (the lanes' snapshots), so re-recording changes the page with it.
 * Nothing here is typed in. Changed from upstream: the figures used to come from a stats file written beside the
 * recording, and the copy named a specific model and its rivals; both are now derived from whichever lanes the
 * recording holds.
 */

import { WINDOW_SECONDS } from '../config/site';
import type { Replay } from '../game/types';
import { laneStats } from '../record/stats';

export interface LaneStat {
  model: string;
  label: string;
  provider: string;
  decisionsPerSec: number;
  avgMs: number;
  p95Ms: number;
  returns: number;
  /** Decisions counted over the first WINDOW_SECONDS of the recording. */
  inWindow: number;
}

/** One row per lane, in the order the recording holds them. Lanes are never re-sorted. */
export function laneStatsFor(replay: Replay): LaneStat[] {
  return replay.lanes.map((lane) => {
    const stats = laneStats(lane.model, lane.label, lane.snapshots);
    const first = lane.snapshots[0]?.t ?? 0;
    const inWindow = lane.snapshots.filter(
      (s) => s.latencyMs != null && s.t - first <= WINDOW_SECONDS * 1000,
    ).length;
    return {
      model: lane.model as string,
      label: lane.label,
      provider: lane.provider ?? '',
      decisionsPerSec: stats.decisionsPerSecond,
      avgMs: stats.avgMs,
      p95Ms: stats.p95Ms,
      returns: stats.returns,
      inWindow,
    };
  });
}

/**
 * The claim, in two sentences, built from the numbers. The first lane is the one the page is about; the rest are
 * what it is compared with.
 */
export function headlineSentences(rows: LaneStat[]): { lead: string; leadMs: string; others: string } {
  const [first, ...rest] = rows;
  if (!first) return { lead: '', leadMs: '', others: '' };
  const others = rest.map((row) => row.avgMs);
  const range =
    others.length === 0
      ? ''
      : `The others take ${(Math.min(...others) / 1000).toFixed(1)} to ${(Math.max(...others) / 1000).toFixed(1)} seconds.`;
  return { lead: `${first.label} returns a decision in`, leadMs: `${Math.round(first.avgMs)} ms`, others: range };
}
