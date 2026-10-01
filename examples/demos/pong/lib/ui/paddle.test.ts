import { describe, expect, it } from 'vitest';
import { COURT_H, HUMAN_PADDLE_SPEED, PADDLE_H, type Move } from '../game/types';
import {
  PADDLE_MAX_Y,
  PADDLE_MIN_Y,
  RECONCILE_SNAP,
  clampPaddle,
  predictPaddle,
} from './paddle';

const FRAME = 1000 / 60;

/** Run `frames` frames of prediction against a wire that does not move. */
function run(
  start: number,
  wireY: number,
  held: Move,
  frames: number,
  dtMs = FRAME,
): number {
  let y = start;
  for (let i = 0; i < frames; i += 1) {
    y = predictPaddle({ current: y, wireY, held, dtMs });
  }
  return y;
}

describe('clampPaddle', () => {
  it('keeps the paddle inside the court, the same way the engine does', () => {
    expect(PADDLE_MIN_Y).toBe(PADDLE_H / 2);
    expect(PADDLE_MAX_Y).toBe(COURT_H - PADDLE_H / 2);
    expect(clampPaddle(-40)).toBe(PADDLE_MIN_Y);
    expect(clampPaddle(1000)).toBe(PADDLE_MAX_Y);
    expect(clampPaddle(50)).toBe(50);
  });
});

describe('predictPaddle', () => {
  it('moves at HUMAN_PADDLE_SPEED while a direction is held', () => {
    // The worker moved it too, so there is nothing to reconcile: what is left
    // is the prediction's own speed.
    const step = (HUMAN_PADDLE_SPEED * 100) / 1000;
    const up = predictPaddle({ current: 50, wireY: 50 - step, held: 'up', dtMs: 100 });
    expect(up).toBeCloseTo(50 - step, 6);

    const down = predictPaddle({ current: 50, wireY: 50 + step, held: 'down', dtMs: 100 });
    expect(down).toBeCloseTo(50 + step, 6);
  });

  it('holds still on stay', () => {
    expect(predictPaddle({ current: 40, wireY: 40, held: 'stay', dtMs: 16 })).toBe(40);
  });

  it('stops at the top and bottom of the court', () => {
    expect(run(PADDLE_MIN_Y, PADDLE_MIN_Y, 'up', 60)).toBe(PADDLE_MIN_Y);
    expect(run(PADDLE_MAX_Y, PADDLE_MAX_Y, 'down', 60)).toBe(PADDLE_MAX_Y);
  });

  it('puts the paddle where a finger is, and never outside the court', () => {
    expect(predictPaddle({ current: 50, wireY: 50, held: 'up', dtMs: 16, pointerY: 22 })).toBe(22);
    expect(predictPaddle({ current: 50, wireY: 50, held: 'stay', dtMs: 16, pointerY: 0 })).toBe(
      PADDLE_MIN_Y,
    );
    expect(predictPaddle({ current: 50, wireY: 50, held: 'stay', dtMs: 16, pointerY: COURT_H })).toBe(
      PADDLE_MAX_Y,
    );
  });

  it('closes a small disagreement with the wire within a few frames', () => {
    // An input that never arrived: the prediction moved, the real paddle did
    // not. Nothing is held now, so the drawn paddle must end up on the wire.
    const settled = run(42, 50, 'stay', 30);
    expect(settled).toBe(50);
  });

  it('cuts to the wire when the two are far apart', () => {
    // A serve recentres both paddles. There is nothing to interpolate between.
    expect(predictPaddle({ current: 90, wireY: 50, held: 'stay', dtMs: 16 })).toBe(50);
    expect(Math.abs(90 - 50)).toBeGreaterThanOrEqual(RECONCILE_SNAP);
  });

  it('never drags the paddle back toward the wire while a key is held', () => {
    // THE SHIVER. The wire is 150-200ms old and only moves in 8-unit jumps at
    // 10Hz, so a press spends every frame being pulled backwards. While the key
    // is down the player is the authority: pure HUMAN_PADDLE_SPEED, whatever
    // the wire says.
    const behind = run(50, 62, 'up', 6);
    const ahead = run(50, 38, 'up', 6);
    const alone = run(50, 50, 'up', 6);
    expect(behind).toBeCloseTo(alone, 9);
    expect(ahead).toBeCloseTo(alone, 9);
    expect(alone).toBeCloseTo(50 - (HUMAN_PADDLE_SPEED * 6 * FRAME) / 1000, 6);
  });

  it('holds where the hand stopped until the wire has heard the stop', () => {
    // Released at 30. The wire, a round trip behind, still shows 44 and is
    // still moving; without the ack it would drag the paddle back to 44 and
    // then forward again as the frames caught up.
    const held = predictPaddle({ current: 30, wireY: 44, held: 'stay', dtMs: 16, wireAcked: false });
    expect(held).toBe(30);
    // Once a frame carries the stop, the small remaining gap is closed as usual.
    const acked = predictPaddle({ current: 30, wireY: 31.5, held: 'stay', dtMs: 16, wireAcked: true });
    expect(acked).toBeGreaterThan(30);
    expect(acked).toBeLessThan(31.5);
  });

  it('never snaps under the player\'s hand', () => {
    // A gap of PADDLE_H or more cuts to the wire when idle (a serve recentre).
    // Doing that mid-press is the worst jump of all, so it does not happen.
    const gap = RECONCILE_SNAP + 10;
    const held = predictPaddle({ current: 50, wireY: 50 + gap, held: 'up', dtMs: FRAME });
    expect(held).toBeLessThan(50);
    expect(held).toBeCloseTo(50 - (HUMAN_PADDLE_SPEED * FRAME) / 1000, 6);
  });

  it('keeps moving while a key is held, and comes back to the wire on release', () => {
    // The worst case: every input is lost, so the wire never moves at all. The
    // paddle still answers the key for the whole press — and the moment the key
    // comes up it is pulled back onto the truth, so nothing is ever stranded.
    const pressed = run(50, 50, 'up', 120);
    expect(pressed).toBeLessThan(50);
    expect(pressed).toBe(PADDLE_MIN_Y); // 2s of 'up' reaches the top

    const released = run(pressed, 50, 'stay', 60);
    expect(released).toBe(50);
    expect(Math.abs(pressed - 50)).toBeGreaterThanOrEqual(PADDLE_H / 2);
  });

  it('still stops at the edge of the court while held', () => {
    expect(run(PADDLE_MIN_Y + 1, 50, 'up', 60)).toBe(PADDLE_MIN_Y);
    expect(run(PADDLE_MAX_Y - 1, 50, 'down', 60)).toBe(PADDLE_MAX_Y);
  });

  it('lets a finger override a held key, as a position rather than a direction', () => {
    expect(
      predictPaddle({ current: 50, wireY: 20, held: 'up', dtMs: FRAME, pointerY: 70 }),
    ).toBe(70);
  });

  it('reconciles at the same rate per millisecond whatever the frame rate', () => {
    const at60 = run(40, 50, 'stay', 12, FRAME);
    const at120 = run(40, 50, 'stay', 24, FRAME / 2);
    expect(at120).toBeCloseTo(at60, 3);
  });
});
