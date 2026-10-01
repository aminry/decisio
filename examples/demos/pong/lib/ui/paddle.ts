/**
 * lib/ui/paddle.ts — where YOUR paddle is, on the frame you are looking at.
 *
 * Everything else on the court is drawn from the wire: one game, in one place,
 * and the browser is not it. The player's own paddle is the exception, and it
 * has to be, because the round trip is real — key down, publish, worker paddle
 * tick, publish, render — and 150-200ms of it lands entirely between your hand
 * and the thing your hand is supposed to be moving. Drawing only the wire
 * position is what made the game feel like steering a boat.
 *
 * So the client keeps its own paddle and moves it the moment a key goes down,
 * at exactly the speed the worker moves it (HUMAN_PADDLE_SPEED, continuously),
 * and then RECONCILES it toward the wire. The reconciliation is the important
 * half: a prediction that is never corrected is a lie that grows, and a dropped
 * input would leave the drawn paddle somewhere the real one never went. Pulling
 * it back a quarter of the way each frame means the two can never be more than
 * a few units apart, whatever gets lost.
 *
 * BUT NOT WHILE THE KEY IS DOWN. Reconciling during a press pulls the paddle
 * back toward a position that is 150-200ms old, every single frame, and the
 * wire only updates in 8-unit jumps at 10Hz — so the drawn paddle shivers
 * against the hand holding it, and a press long enough to open a PADDLE_H gap
 * snaps it backwards outright. For the duration of a press the PLAYER is the
 * authority: the paddle moves at the worker's own speed and nothing drags it.
 * The moment the key comes up the wire takes over again and the prediction is
 * pulled back onto it — which is where a lost input would otherwise strand it,
 * so the guarantee is kept, just settled on release instead of during.
 *
 * AND NOT UNTIL THE WIRE HAS HEARD THE STOP. On release the wire is still
 * showing a paddle from a round trip ago — one that is still moving, 10 to 20
 * units short of where the hand let go. Reconciling against that pulls the
 * drawn paddle backwards, and then the frames catch up and push it forward
 * again: the "jump up and down" on every keypress. So each input carries a
 * sequence number, every frame says which one the worker had applied
 * (`Snapshot.inputSeq`), and the prediction holds still until a frame
 * acknowledges the player's latest input. By then the two are within a couple
 * of units, and the ease is invisible.
 *
 * Pure: give it the numbers, get the next position. No refs, no clock, no React.
 */

import { COURT_H, HUMAN_PADDLE_SPEED, PADDLE_H, type Move } from '../game/types';

/**
 * The range a paddle CENTRE may sit in. Same derivation as the engine's
 * PADDLE_MIN_Y / PADDLE_MAX_Y (lib/game/engine.ts) — from the contract, so the
 * drawn paddle cannot stop anywhere the real one would not.
 */
export const PADDLE_MIN_Y = PADDLE_H / 2;
export const PADDLE_MAX_Y = COURT_H - PADDLE_H / 2;

/** Fraction of the gap to the wire closed per 60fps frame, while idle. */
export const RECONCILE_PER_FRAME = 0.25;
/** Below this the prediction is simply the wire. Stops sub-unit jitter. */
export const RECONCILE_THRESHOLD = 1;
/**
 * Above this the prediction is not a prediction of anything: the worker has
 * recentred the paddles for a serve, or we have just joined. Cut to the wire —
 * but only when idle, because a cut under the player's own hand is the worst
 * kind of jump.
 */
export const RECONCILE_SNAP = PADDLE_H;
/** The frame the reconciliation rate is quoted at. 120Hz closes it as fast. */
export const FRAME_MS = 1000 / 60;

export function clampPaddle(y: number): number {
  return y < PADDLE_MIN_Y ? PADDLE_MIN_Y : y > PADDLE_MAX_Y ? PADDLE_MAX_Y : y;
}

export interface PredictPaddleOptions {
  /** The predicted position on the previous frame. */
  current: number;
  /** Where the worker says the paddle is. Authority, and always a little old. */
  wireY: number;
  /** The direction being asked for on this frame. */
  held: Move;
  /** Milliseconds since the previous frame. */
  dtMs: number;
  /**
   * Court y under a finger or mouse, while one is down. A pointer is a
   * POSITION rather than a direction, so the paddle goes where it is put and
   * the input stream tells the worker to follow it; nothing is reconciled while
   * it is held, or the paddle would lag the finger by the round trip again.
   */
  pointerY?: number | null;
  /** Court units per second. The worker's own figure by default. */
  speed?: number;
  /**
   * Whether `wireY` comes from a frame that has applied the player's latest
   * input. False means the wire is older than the hand, and nothing is
   * reconciled against it. Default true, for a wire with no inputs in flight.
   */
  wireAcked?: boolean;
}

/** Where to draw the human's paddle on this frame. */
export function predictPaddle(options: PredictPaddleOptions): number {
  const { current, wireY, held, dtMs, pointerY = null, wireAcked = true } = options;
  if (pointerY !== null) return clampPaddle(pointerY);

  const speed = options.speed ?? HUMAN_PADDLE_SPEED;
  const dir = held === 'up' ? -1 : held === 'down' ? 1 : 0;
  const dt = dtMs > 0 ? dtMs : 0;
  const moved = clampPaddle(current + dir * speed * (dt / 1000));

  // Held: the player owns the paddle. No reconcile, and above all no snap —
  // see the header. The clamp still applies, so it can never leave the court.
  if (dir !== 0) return moved;
  // Released, but the wire has not yet heard the release: it is still showing
  // the paddle where it was a round trip ago. Hold where the hand stopped.
  if (!wireAcked) return moved;

  const gap = wireY - moved;
  const distance = Math.abs(gap);
  if (distance <= RECONCILE_THRESHOLD) return clampPaddle(wireY);
  if (distance >= RECONCILE_SNAP) return clampPaddle(wireY);

  // Frame-rate independent: a quarter of the gap per 60fps frame, so a 120Hz
  // display converges at the same speed in milliseconds rather than twice as
  // fast, and a dropped frame does not leave the paddle behind.
  const k = 1 - Math.pow(1 - RECONCILE_PER_FRAME, dt / FRAME_MS);
  return clampPaddle(moved + gap * k);
}
