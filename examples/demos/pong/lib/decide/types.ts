import type { Move } from '../game/types';

/** What one model call returns: the move, and how long the call took. */
export interface ModelDecision {
  move: Move;
  /** Wall-clock milliseconds around the model call only. */
  latencyMs: number;
}
