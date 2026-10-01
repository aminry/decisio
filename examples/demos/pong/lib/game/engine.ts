/**
 * Pong lanes — deterministic, pure game engine.
 *
 * Every function here is pure: it takes an EngineState and returns a new one.
 * Nothing mutates, nothing reads a clock, nothing touches the DOM. Given the
 * same seed and the same sequence of moves you get the same game, in Node and
 * in the browser.
 *
 * THE MECHANIC — "ball steps per decision":
 *   One model decision = one tick = the ball advances exactly SEGMENT_X along x
 *   (eight of those cross from one paddle plane to the other). The model's
 *   paddle moves at most MODEL_PADDLE_STEP. A slow model is a slow ball.
 *   Nothing is skipped.
 *
 * WHERE THE BALL TURNS:
 *   At the paddle PLANES — LEFT_PLANE = 4 and RIGHT_PLANE = 156 — never at the
 *   court edges. A plane is the paddle's inner face, which is exactly where the
 *   renderer draws it (lib/render/court.ts derives both from PADDLE_INSET), so
 *   the ball turns ON the paddle instead of passing through it and bouncing off
 *   the wall behind. The ball centre never goes past a plane.
 *
 * CONVENTIONS (documented, because the integrator needs them):
 *   - `serveDir` is the x DIRECTION OF TRAVEL of the next serve: +1 means the
 *     ball is served toward the right paddle, -1 toward the left.
 *   - After a point, serveDir points TOWARD THE SIDE THAT JUST LOST — the ball
 *     keeps travelling the way it was going when the point was conceded, so the
 *     loser receives.
 *   - `tick` is MONOTONIC for the whole game. It starts at 0, is incremented by
 *     every applyTick, and is never reset by a serve. Replays and UI keys want a
 *     monotonic counter.
 *   - serve() re-centres BOTH paddles. This is what makes the scripted left
 *     paddle provably unbeatable (see the never-miss note on autoLeftY).
 *   - applyTick AUTO-SERVES when the status is 'idle' or 'point', so the engine
 *     can never be driven into an invalid state. serve() is also exported, and
 *     the runner calls it explicitly so it can emit a 'serving' snapshot.
 */

import {
  BALL_R,
  COURT_H,
  COURT_W,
  LEFT_PLANE,
  MODEL_PADDLE_STEP,
  PADDLE_H,
  POINTS_TO_WIN,
  RIGHT_PLANE,
  SEGMENT_X,
  type DecisionState,
  type EngineState,
  type Move,
} from './types';

/** x plane of the left paddle (the scripted wall / the human in play mode). */
export const LEFT_X = LEFT_PLANE;
/** x plane of the right paddle (the model). */
export const RIGHT_X = RIGHT_PLANE;
export const CENTRE_X = COURT_W / 2;
export const CENTRE_Y = COURT_H / 2;
/** Paddle centre is clamped to this range so the paddle stays inside the court. */
export const PADDLE_MIN_Y = PADDLE_H / 2;
export const PADDLE_MAX_Y = COURT_H - PADDLE_H / 2;
/** Ball centre is reflected inside this range (top/bottom walls). */
export const BALL_MIN_Y = BALL_R;
export const BALL_MAX_Y = COURT_H - BALL_R;
/** Serves and returns are held inside +/- this angle from horizontal. */
export const MAX_ANGLE_DEG = 35;
/** |vy| at the steepest angle, with |vx| pinned to SEGMENT_X. */
export const MAX_VY = SEGMENT_X * Math.tan((MAX_ANGLE_DEG * Math.PI) / 180);
/** A paddle covers its centre +/- this, ball radius included. */
export const PADDLE_REACH = PADDLE_H / 2 + BALL_R;
/** The scripted "wall" moves at twice the model's step. */
export const WALL_STEP = 2 * MODEL_PADDLE_STEP;

/** Tolerance for "the ball reached the plane", in court units. Float noise is ~1e-14. */
const EPS = 1e-9;

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Round to 1 decimal and normalise -0 to 0, so wire payloads are stable. */
function r1(n: number): number {
  const v = Math.round(n * 10) / 10;
  return v === 0 ? 0 : v;
}

/**
 * mulberry32. Returns the drawn value and the NEXT cursor, so callers keep the
 * cursor in EngineState.rng instead of hiding it in a closure.
 */
export function nextRandom(state: number): { value: number; state: number } {
  const a = (state + 0x6d2b79f5) | 0;
  let t = a;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return { value: ((t ^ (t >>> 14)) >>> 0) / 4294967296, state: a };
}

/**
 * Reflect a y coordinate into [lo, hi] as many times as needed (triangle fold).
 * `flipped` is true when an odd number of reflections happened, i.e. vy flips.
 *
 * Folding composes: fold(fold(u) + d) === fold(u + d) with the sign carried, so
 * stepping the ball in pieces gives exactly the same answer as predicting the
 * whole flight in one go. predictIntercept relies on that.
 */
function fold(y: number, lo: number, hi: number): { y: number; flipped: boolean } {
  const span = hi - lo;
  if (span <= 0) return { y: lo, flipped: false };
  const period = 2 * span;
  let m = (y - lo) % period;
  if (m < 0) m += period;
  if (m <= span) return { y: lo + m, flipped: false };
  return { y: lo + (period - m), flipped: true };
}

/** A fresh, unserved game. Paddles centred, ball parked, status 'idle'. */
export function createEngine(seed: number, serveDir: -1 | 1 = 1): EngineState {
  return {
    seed,
    rng: seed | 0,
    tick: 0,
    ball: { x: CENTRE_X, y: CENTRE_Y, vx: 0, vy: 0 },
    leftY: CENTRE_Y,
    rightY: CENTRE_Y,
    score: [0, 0],
    status: 'idle',
    serveDir,
  };
}

/**
 * Put the ball at the centre travelling in `serveDir` at an angle drawn from the
 * PRNG within +/- MAX_ANGLE_DEG. |vx| is pinned to SEGMENT_X, so the ball always
 * covers exactly one segment per tick and rallies never speed up (v0 clarity).
 * Both paddles return to the centre. A finished game ('over') is not re-served.
 */
export function serve(state: EngineState): EngineState {
  if (state.status === 'over') return state;
  const { value, state: rng } = nextRandom(state.rng);
  const angle = (value * 2 - 1) * ((MAX_ANGLE_DEG * Math.PI) / 180);
  return {
    ...state,
    rng,
    ball: {
      x: CENTRE_X,
      y: CENTRE_Y,
      vx: state.serveDir * SEGMENT_X,
      vy: SEGMENT_X * Math.tan(angle),
    },
    leftY: CENTRE_Y,
    rightY: CENTRE_Y,
    status: 'playing',
  };
}

/**
 * The y (ball centre) at which the ball will cross that paddle's x plane, after
 * however many wall bounces. Null when the ball is moving away from that side
 * (or is parked between points). Agrees exactly with what applyTick produces.
 */
export function predictIntercept(state: EngineState, side: 'left' | 'right'): number | null {
  const { x, y, vx, vy } = state.ball;
  if (!Number.isFinite(vx) || vx === 0) return null;
  const toward = side === 'right' ? vx > 0 : vx < 0;
  if (!toward) return null;
  const plane = side === 'right' ? RIGHT_X : LEFT_X;
  const t = (plane - x) / vx;
  if (t < 0) return null;
  return fold(y + vy * t, BALL_MIN_Y, BALL_MAX_Y).y;
}

/** Exactly what a model sees: numbers only, 1 decimal, about 125 bytes of JSON. */
export function toDecisionState(state: EngineState, side: 'left' | 'right'): DecisionState {
  const intercept = predictIntercept(state, side);
  const toward = side === 'right' ? state.ball.vx > 0 : state.ball.vx < 0;
  return {
    court: { w: COURT_W, h: COURT_H },
    ball: {
      x: r1(state.ball.x),
      y: r1(state.ball.y),
      vx: r1(state.ball.vx),
      vy: r1(state.ball.vy),
    },
    paddle: { y: r1(side === 'right' ? state.rightY : state.leftY), h: PADDLE_H },
    interceptY: intercept === null ? null : r1(intercept),
    dir: toward ? 'toward' : 'away',
  };
}

/** Which way to move a paddle to reach targetY. 'stay' inside half a step. */
export function moveToward(paddleY: number, targetY: number | null, step = MODEL_PADDLE_STEP): Move {
  if (targetY === null) return 'stay';
  const d = targetY - paddleY;
  if (Math.abs(d) <= step / 2) return 'stay';
  return d < 0 ? 'up' : 'down';
}

function moveDelta(move: Move, step: number): number {
  if (move === 'up') return -step;
  if (move === 'down') return step;
  return 0;
}

/**
 * Move ONE paddle by `units`, clamped to the court. Nothing else changes: not
 * the ball, not `tick`, not the score, not the status.
 *
 * This exists for a worker, where the two clocks are separate. A human
 * paddle runs on its own 10Hz tick so it feels live, while the ball still
 * advances once per model decision (`applyTick`). `applyTick` is the only thing
 * that moves the ball, so calling this in between can never corrupt a game — at
 * worst a paddle is somewhere else when the ball arrives, which is the point.
 *
 * `units` is court units for THIS call, and the caller decides what that is: the
 * worker passes HUMAN_PADDLE_SPEED x its tick length, so a human paddle travels
 * at a speed rather than in one jump per decision. A big step on a slow tick is
 * what made the human paddle teleport and overshoot.
 *
 * Returns the same object when the move is 'stay' or the paddle is already
 * against the clamp, so a caller can use identity to decide whether to publish.
 */
export function applyPaddleMove(
  state: EngineState,
  side: 'left' | 'right',
  move: Move,
  units: number,
): EngineState {
  const delta = moveDelta(move, units);
  if (delta === 0) return state;
  if (side === 'left') {
    const leftY = clamp(state.leftY + delta, PADDLE_MIN_Y, PADDLE_MAX_Y);
    return leftY === state.leftY ? state : { ...state, leftY };
  }
  const rightY = clamp(state.rightY + delta, PADDLE_MIN_Y, PADDLE_MAX_Y);
  return rightY === state.rightY ? state : { ...state, rightY };
}

/**
 * The scripted left paddle ("the wall"): chase the predicted intercept at up to
 * WALL_STEP per tick; drift back to the centre at MODEL_PADDLE_STEP when the
 * ball is heading away.
 *
 * THE REACH BUDGET (all of it, for both paddles). The clamped paddle range is
 * [10, 90] — 80 units wide — and a paddle moves at the START of a tick, so the
 * moves available before the ball arrives are:
 *
 *   - A return: the ball ends the tick exactly ON the far plane (every distance
 *     in the game is a whole number of segments), so it is SEGMENTS_PER_CROSSING
 *     = 8 ticks away, and a paddle gets 8 moves.
 *       wall  8 * WALL_STEP (24)         = 192 > 80.  Margin 112.
 *       model 8 * MODEL_PADDLE_STEP (12) =  96 > 80.  Margin 16.
 *   - A serve: the centre line is CROSSING_X / 2 = 76 units from either plane,
 *     i.e. 4 ticks, and both paddles are re-centred, so at most 40 to cover.
 *       wall  4 * 24 = 96 > 40.  model 4 * 12 = 48 > 40.
 *   - A human paddle is on a speed, not a step: HUMAN_PADDLE_SPEED (80/s) over
 *     the 8 * PLAY_MIN_STEP_MS = 2.8s a crossing takes is 224 units > 80.
 *   - A clamped centre still covers the ball: the intercept lies in [1.5, 98.5],
 *     so |intercept - clamp(intercept, 10, 90)| <= 8.5 < PADDLE_REACH (11.5).
 *
 * That is why the wall provably never misses, and why a model that answers on
 * every tick is not out-run by the ball. engine.test.ts checks the wall
 * empirically over 500+ rallies as well.
 */
function autoLeftY(s: EngineState): number {
  const target = predictIntercept(s, 'left');
  const goal = clamp(target === null ? CENTRE_Y : target, PADDLE_MIN_Y, PADDLE_MAX_Y);
  const step = target === null ? MODEL_PADDLE_STEP : WALL_STEP;
  const d = clamp(goal - s.leftY, -step, step);
  return clamp(s.leftY + d, PADDLE_MIN_Y, PADDLE_MAX_Y);
}

/**
 * Advance the ball exactly one segment along x, reflecting off the top and
 * bottom walls and resolving a paddle plane crossing if one happens on the way.
 * Paddles have already moved when this runs.
 */
function advanceSegment(s: EngineState): EngineState {
  let { x, y, vx, vy } = s.ball;
  let remaining = SEGMENT_X;
  let [leftScore, rightScore] = s.score;
  let serveDir = s.serveDir;
  let conceded = false;

  for (let guard = 0; remaining > EPS && guard < 64; guard += 1) {
    const dir: -1 | 1 = vx > 0 ? 1 : -1;
    const plane = dir > 0 ? RIGHT_X : LEFT_X;
    const distToPlane = Math.max(0, (plane - x) * dir);
    // Within EPS counts as reaching the plane this tick: float noise must never
    // push a crossing into the next tick and hand a paddle a free extra move.
    const crosses = distToPlane <= remaining + EPS;
    const step = Math.min(distToPlane, remaining);

    const folded = fold(y + vy * (step / Math.abs(vx)), BALL_MIN_Y, BALL_MAX_Y);
    y = folded.y;
    if (folded.flipped) vy = -vy;
    x = crosses ? plane : x + dir * step;
    remaining -= step;

    if (!crosses) break;

    const paddleY = dir > 0 ? s.rightY : s.leftY;
    if (Math.abs(y - paddleY) <= PADDLE_REACH + EPS) {
      // Hit. Reflect vx (speed unchanged) and take vy from where it struck the
      // paddle: dead centre returns flat, the edges return at MAX_ANGLE_DEG.
      const offset = clamp((y - paddleY) / (PADDLE_H / 2), -1, 1);
      vx = -dir * SEGMENT_X;
      vy = offset * MAX_VY;
    } else {
      // Miss. The other side scores; the next serve travels the same way the
      // ball was going, i.e. toward the side that just lost the point.
      if (dir > 0) leftScore += 1;
      else rightScore += 1;
      serveDir = dir;
      conceded = true;
      break;
    }
  }

  if (conceded) {
    const over = leftScore >= POINTS_TO_WIN || rightScore >= POINTS_TO_WIN;
    return {
      ...s,
      ball: { x: CENTRE_X, y: CENTRE_Y, vx: 0, vy: 0 },
      score: [leftScore, rightScore],
      status: over ? 'over' : 'point',
      serveDir,
    };
  }
  return { ...s, ball: { x, y, vx, vy } };
}

/**
 * One decision = one tick. Paddles move first (clamped), then the ball advances
 * exactly one segment. `leftMove` is 'auto' for the scripted wall, or a Move
 * applied as one MODEL_PADDLE_STEP — which is what the recorder's two-model
 * lanes want. A live human does NOT come through here: the worker moves that
 * paddle continuously with `applyPaddleMove` and passes 'stay'.
 * Auto-serves out of 'idle' and 'point'; 'over' is a no-op so a runaway caller
 * cannot corrupt a finished game.
 */
export function applyTick(state: EngineState, rightMove: Move, leftMove: Move | 'auto'): EngineState {
  if (state.status === 'over') return state;
  const served = state.status === 'playing' ? state : serve(state);

  const rightY = clamp(
    served.rightY + moveDelta(rightMove, MODEL_PADDLE_STEP),
    PADDLE_MIN_Y,
    PADDLE_MAX_Y,
  );
  const leftY =
    leftMove === 'auto'
      ? autoLeftY(served)
      : clamp(served.leftY + moveDelta(leftMove, MODEL_PADDLE_STEP), PADDLE_MIN_Y, PADDLE_MAX_Y);

  return advanceSegment({ ...served, leftY, rightY, tick: served.tick + 1 });
}
