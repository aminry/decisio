/**
 * lib/ui/lane-number.ts — what the latency number says at this instant.
 *
 * Motion note 04 in one function: the number is
 * `max(last landed latency, time since the last snapshot)`, so while a model is
 * thinking the reading counts up past its own last answer and the row tells the
 * truth about the wait rather than freezing on a flattering number.
 *
 * A LIVE lane has one more state that a recording cannot have. The worker
 * publishes a frame at least every HEARTBEAT_MS whatever the model is doing
 * (lib/worker/game-worker.ts), so silence longer than two of them is not a slow
 * decision — it is the connection. Counting on up to 5030 ms would be reporting
 * a model latency nobody measured, which is exactly what the number did the
 * first time this was played. It stops at the stall instead and the row says
 * "waiting…" until a frame lands.
 *
 * lib/render/timeline.ts has the same rule for a recording, where every frame
 * and its time is known in advance. This is the live one. Both are pure.
 */

import { HEARTBEAT_MS } from '../game/types';

/** Two heartbeats of silence on a live lane. */
export const STALL_MS = HEARTBEAT_MS * 2;

/** How long the colour flashes when a decision lands. */
export const FLASH_MS = 120;

export interface LaneNumberState {
  /** The latency of the most recent decision that landed. */
  lastLatencyMs: number | null;
  /** Clock reading when the last snapshot arrived. 0 before the first one. */
  landedAt: number;
  /** Bumped once per landed decision, so the flash fires exactly once. */
  landedKey: number;
  /** The lane has stopped: nothing is in flight and the number holds. */
  stopped: boolean;
  /** This lane is a live game, so its frames are heartbeated. */
  live: boolean;
}

export interface LaneNumberFrame {
  /** The digits, or null for the em dash before anything has happened. */
  value: number | null;
  /** The count-up has overtaken the last reading: weight 400, muted colour. */
  inFlight: boolean;
  /** Nothing has arrived for two heartbeats on a live lane. */
  waiting: boolean;
  /** Within the flash window after a decision landed. */
  flashing: boolean;
}

export function laneNumberAt(state: LaneNumberState, nowMs: number): LaneNumberFrame {
  const age = state.stopped || state.landedAt === 0 ? 0 : Math.max(0, nowMs - state.landedAt);
  const waiting = state.live && age > STALL_MS;
  const elapsed = waiting ? STALL_MS : age;
  const floor = state.lastLatencyMs ?? 0;
  // A stalled lane reads as in flight, because it is: a decision is out there
  // somewhere. What it must not do is keep counting.
  const inFlight = waiting || elapsed > floor;
  const value = state.lastLatencyMs === null && elapsed === 0 ? null : Math.max(floor, elapsed);
  const flashing = !inFlight && state.landedKey > 0 && elapsed < FLASH_MS;
  return { value, inFlight, waiting, flashing };
}
