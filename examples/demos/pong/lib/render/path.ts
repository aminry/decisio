/**
 * Sub-tick ball path, shared by the React court and the headless clip renderer.
 *
 * The engine only tells us where the ball is at the END of each decision. To
 * show the ball *travelling* at the model's speed we have to reproduce what it
 * did in between, including a wall bounce or a paddle return that happened
 * mid-segment. This walks the same geometry as `advanceSegment` in
 * lib/game/engine.ts, but stops after an arbitrary fraction of the segment.
 *
 * It is deliberately decoupled from engine.ts internals: every constant below
 * is derived from the fixed contract in lib/game/types.ts. Small divergences
 * do not matter because the caller blends the simulated path onto the real
 * end-of-tick position.
 *
 * It turns at LEFT_PLANE / RIGHT_PLANE, the same planes the engine uses and the
 * same ones the paddles are drawn on, so a trail bends where the ball is seen
 * to hit the paddle rather than at the court edge behind it.
 */

import {
  BALL_R,
  COURT_H,
  LEFT_PLANE,
  PADDLE_H,
  RIGHT_PLANE,
  SEGMENT_X,
} from '../game/types';

export const BALL_MIN_Y = BALL_R;
export const BALL_MAX_Y = COURT_H - BALL_R;
export const PADDLE_REACH = PADDLE_H / 2 + BALL_R;
/** Mirrors MAX_ANGLE_DEG in lib/game/engine.ts. */
const MAX_ANGLE_DEG = 35;
const MAX_VY = SEGMENT_X * Math.tan((MAX_ANGLE_DEG * Math.PI) / 180);
const EPS = 1e-9;

export interface Ball {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Reflect y into [lo, hi] and report whether it turned around an odd number of times. */
function fold(y: number, lo: number, hi: number): { y: number; flipped: boolean } {
  const span = hi - lo;
  if (span <= 0) return { y: lo, flipped: false };
  let t = (y - lo) % (2 * span);
  if (t < 0) t += 2 * span;
  const flipped = t > span;
  return { y: lo + (flipped ? 2 * span - t : t), flipped };
}

/**
 * Where the ball is after travelling `dist` units of x along its segment.
 * `leftY` / `rightY` are the paddle positions for this tick — the engine moves
 * paddles before the ball, so these are the destination snapshot's values.
 *
 * On a miss the ball keeps going and leaves the court, which is what you want
 * to see: the point is conceded on screen, not teleported away.
 */
export function pathAt(start: Ball, leftY: number, rightY: number, dist: number): Ball {
  let { x, y, vx, vy } = start;
  if (!Number.isFinite(vx) || vx === 0) return { x, y, vx, vy };

  let remaining = Math.max(0, dist);
  for (let guard = 0; remaining > EPS && guard < 64; guard += 1) {
    const dir: -1 | 1 = vx > 0 ? 1 : -1;
    const plane = dir > 0 ? RIGHT_PLANE : LEFT_PLANE;
    const distToPlane = Math.max(0, (plane - x) * dir);
    const crosses = distToPlane <= remaining + EPS;
    const step = Math.min(distToPlane, remaining);

    const folded = fold(y + vy * (step / Math.abs(vx)), BALL_MIN_Y, BALL_MAX_Y);
    y = folded.y;
    if (folded.flipped) vy = -vy;
    x = crosses ? plane : x + dir * step;
    remaining -= step;
    if (!crosses) break;

    const paddleY = dir > 0 ? rightY : leftY;
    if (Math.abs(y - paddleY) <= PADDLE_REACH + EPS) {
      const offset = clamp((y - paddleY) / (PADDLE_H / 2), -1, 1);
      vx = -dir * SEGMENT_X;
      vy = offset * MAX_VY;
    } else {
      // Missed. Fly straight off the court for whatever is left of the segment.
      x += dir * remaining;
      y += vy * (remaining / Math.abs(vx));
      remaining = 0;
    }
  }
  return { x, y, vx, vy };
}
