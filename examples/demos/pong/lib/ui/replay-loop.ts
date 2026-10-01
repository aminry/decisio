/**
 * When the recorded run starts again.
 *
 * Each lane is its own runner with its own timings, and the four recordings do
 * not end on the same millisecond — one lane ran 45.0s, another 44.7s. Left to
 * themselves the lanes would each loop when they personally finished and drift
 * apart within a minute, which is exactly what "same serve, same rules" must
 * not look like. So none of them loops: the page restarts all four together,
 * one period after they started.
 *
 * The period is the longest lane plus a short hold on its final frame.
 */

import type { Replay } from '../game/types';

/** Long enough for the last decision to land and be read. */
export const LOOP_HOLD_MS = 900;

/**
 * How long a full pass of the recording takes, or null if there is nothing to
 * play. `Snapshot.t` is milliseconds from the start of the lane, so the last
 * frame's `t` is the lane's whole length.
 */
export function replayPeriodMs(replay: Replay | null, holdMs: number = LOOP_HOLD_MS): number | null {
  if (replay === null) return null;
  let longest = 0;
  for (const lane of replay.lanes) {
    const last = lane.snapshots[lane.snapshots.length - 1];
    if (last !== undefined && last.t > longest) longest = last.t;
  }
  return longest <= 0 ? null : Math.round(longest + holdMs);
}
