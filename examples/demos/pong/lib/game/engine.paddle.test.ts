/**
 * applyPaddleMove — the paddle-only helper a worker uses for human
 * paddles. Kept in its own file so the engine's own test file is untouched.
 *
 * It takes the distance in COURT UNITS now, because a human paddle moves at a
 * speed (HUMAN_PADDLE_SPEED) applied on every worker paddle tick, not in one
 * jump per model decision.
 */

/** What the worker passes on a 100ms paddle tick: 80 units/s = 8 units. */
const TICK_UNITS = (HUMAN_PADDLE_SPEED * 100) / 1000;
import { describe, expect, it } from 'vitest';
import { HUMAN_PADDLE_SPEED, MODEL_PADDLE_STEP, type EngineState } from './types';
import {
  CENTRE_Y,
  PADDLE_MAX_Y,
  PADDLE_MIN_Y,
  applyPaddleMove,
  createEngine,
  serve,
} from './engine';

/** A live game: served, so the ball has velocity and the status is 'playing'. */
function playing(): EngineState {
  return serve(createEngine(7));
}

describe('applyPaddleMove', () => {
  it('moves the named paddle by the units it is given and leaves the other alone', () => {
    const s = playing();

    const up = applyPaddleMove(s, 'left', 'up', TICK_UNITS);
    expect(up.leftY).toBe(CENTRE_Y - TICK_UNITS);
    expect(up.rightY).toBe(s.rightY);

    const down = applyPaddleMove(s, 'right', 'down', TICK_UNITS);
    expect(down.rightY).toBe(CENTRE_Y + TICK_UNITS);
    expect(down.leftY).toBe(s.leftY);

    // The units are the caller's: a model lane still steps once per decision.
    expect(applyPaddleMove(s, 'right', 'up', MODEL_PADDLE_STEP).rightY).toBe(
      CENTRE_Y - MODEL_PADDLE_STEP,
    );
  });

  it('covers the whole paddle range in a second at the human speed', () => {
    // 10 paddle ticks = 1s = 80 units, which is exactly the clamped range.
    let s = { ...playing(), leftY: PADDLE_MIN_Y };
    for (let i = 0; i < 10; i += 1) s = applyPaddleMove(s, 'left', 'down', TICK_UNITS);
    expect(s.leftY).toBe(PADDLE_MAX_Y);
  });

  it('clamps to the court and stops moving at the edge', () => {
    const s = playing();

    let top = s;
    for (let i = 0; i < 20; i += 1) top = applyPaddleMove(top, 'left', 'up', TICK_UNITS);
    expect(top.leftY).toBe(PADDLE_MIN_Y);

    let bottom = s;
    for (let i = 0; i < 20; i += 1) bottom = applyPaddleMove(bottom, 'right', 'down', TICK_UNITS);
    expect(bottom.rightY).toBe(PADDLE_MAX_Y);
  });

  it('returns the same object for "stay" and for a move that is already clamped', () => {
    const s = playing();
    expect(applyPaddleMove(s, 'left', 'stay', TICK_UNITS)).toBe(s);
    expect(applyPaddleMove(s, 'left', 'up', 0)).toBe(s);

    const atTop: EngineState = { ...s, leftY: PADDLE_MIN_Y };
    expect(applyPaddleMove(atTop, 'left', 'up', TICK_UNITS)).toBe(atTop);
  });

  it('never touches the ball, the tick, the score or the status', () => {
    const s: EngineState = { ...playing(), tick: 12, score: [2, 3] };
    const moved = applyPaddleMove(s, 'left', 'down', TICK_UNITS);

    expect(moved.ball).toEqual(s.ball);
    expect(moved.tick).toBe(12);
    expect(moved.score).toEqual([2, 3]);
    expect(moved.status).toBe(s.status);
    expect(moved.seed).toBe(s.seed);
    expect(moved.rng).toBe(s.rng);
    expect(moved.serveDir).toBe(s.serveDir);
  });

  it('does not mutate the state it was given', () => {
    const s = playing();
    const before = JSON.stringify(s);
    applyPaddleMove(s, 'right', 'up', TICK_UNITS);
    expect(JSON.stringify(s)).toBe(before);
  });
});
