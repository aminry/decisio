import { describe, expect, it } from 'vitest';
import {
  BALL_R,
  COURT_H,
  COURT_W,
  LEFT_PLANE,
  MODEL_PADDLE_STEP,
  PADDLE_H,
  POINTS_TO_WIN,
  RIGHT_PLANE,
  SEGMENTS_PER_CROSSING,
  SEGMENT_X,
  type EngineState,
  type Move,
} from './types';
import {
  BALL_MAX_Y,
  BALL_MIN_Y,
  CENTRE_X,
  CENTRE_Y,
  MAX_ANGLE_DEG,
  PADDLE_MAX_Y,
  PADDLE_MIN_Y,
  PADDLE_REACH,
  WALL_STEP,
  applyTick,
  clamp,
  createEngine,
  moveToward,
  predictIntercept,
  serve,
  toDecisionState,
} from './engine';

/** A right paddle that always plays the textbook move. */
function perfectRight(s: EngineState): Move {
  return moveToward(s.rightY, predictIntercept(s, 'right'), MODEL_PADDLE_STEP);
}

describe('createEngine', () => {
  it('starts idle, centred and scoreless', () => {
    const s = createEngine(42);
    expect(s.status).toBe('idle');
    expect(s.leftY).toBe(CENTRE_Y);
    expect(s.rightY).toBe(CENTRE_Y);
    expect(s.score).toEqual([0, 0]);
    expect(s.tick).toBe(0);
    expect(s.ball).toEqual({ x: CENTRE_X, y: CENTRE_Y, vx: 0, vy: 0 });
    expect(s.serveDir).toBe(1);
  });

  it('honours an explicit serve direction', () => {
    expect(createEngine(42, -1).serveDir).toBe(-1);
  });

  it('is immutable: serve and applyTick return new objects', () => {
    const a = createEngine(7);
    const b = serve(a);
    const c = applyTick(b, 'up', 'auto');
    expect(b).not.toBe(a);
    expect(a.status).toBe('idle');
    expect(c).not.toBe(b);
    expect(b.tick).toBe(0);
  });
});

describe('serve', () => {
  it('puts the ball at the centre with |vx| = SEGMENT_X and an angle within +/-35 degrees', () => {
    for (let seed = 0; seed < 200; seed += 1) {
      const s = serve(createEngine(seed));
      expect(s.status).toBe('playing');
      expect(s.ball.x).toBe(CENTRE_X);
      expect(s.ball.y).toBe(CENTRE_Y);
      expect(Math.abs(s.ball.vx)).toBeCloseTo(SEGMENT_X, 10);
      const deg = (Math.atan2(s.ball.vy, Math.abs(s.ball.vx)) * 180) / Math.PI;
      expect(Math.abs(deg)).toBeLessThanOrEqual(MAX_ANGLE_DEG + 1e-9);
    }
  });

  it('serves in serveDir and re-centres both paddles', () => {
    const left = serve({ ...createEngine(3, -1), leftY: 20, rightY: 80 });
    expect(left.ball.vx).toBeLessThan(0);
    expect(left.leftY).toBe(CENTRE_Y);
    expect(left.rightY).toBe(CENTRE_Y);
    expect(serve(createEngine(3, 1)).ball.vx).toBeGreaterThan(0);
  });

  it('advances the PRNG cursor so successive serves differ', () => {
    const a = serve(createEngine(9));
    const b = serve({ ...a, status: 'point' });
    expect(b.rng).not.toBe(a.rng);
    expect(b.ball.vy).not.toBe(a.ball.vy);
  });

  it('refuses to re-serve a finished game', () => {
    const over = { ...serve(createEngine(1)), status: 'over' as const };
    expect(serve(over)).toBe(over);
  });
});

describe('predictIntercept', () => {
  it('returns null when the ball is moving away or parked', () => {
    const s = serve(createEngine(5, 1)); // heading right
    expect(predictIntercept(s, 'left')).toBeNull();
    expect(predictIntercept(s, 'right')).not.toBeNull();
    expect(predictIntercept(createEngine(5), 'right')).toBeNull();
  });

  it('agrees with the y the ball actually crosses at, over many trajectories', () => {
    let checked = 0;
    for (let seed = 0; seed < 60; seed += 1) {
      let s = serve(createEngine(seed));
      for (let i = 0; i < 400 && s.status !== 'over'; i += 1) {
        const side = s.ball.vx > 0 ? 'right' : 'left';
        const predicted = predictIntercept(s, side);
        if (predicted === null) break;
        const plane = side === 'right' ? RIGHT_PLANE : LEFT_PLANE;
        // Step until the ball reaches that plane, with both paddles frozen so
        // nothing but walls can change the trajectory.
        let probe = s;
        let crossedY: number | null = null;
        for (let k = 0; k < 12; k += 1) {
          const before = probe.ball.vx;
          probe = applyTick(probe, 'stay', 'stay');
          if (probe.status !== 'playing') {
            // Conceded: the ball reset, so read the crossing from the miss.
            crossedY = null;
            break;
          }
          if (Math.sign(probe.ball.vx) !== Math.sign(before)) {
            crossedY = probe.ball.y; // recorded after the bounce, y is continuous
            break;
          }
          if (Math.abs(probe.ball.x - plane) < 1e-9) {
            crossedY = probe.ball.y;
            break;
          }
        }
        if (crossedY !== null) {
          expect(crossedY).toBeCloseTo(predicted, 6);
          checked += 1;
        }
        // Continue the real game with paddles that keep the rally alive.
        s = applyTick(s, perfectRight(s), 'auto');
      }
    }
    expect(checked).toBeGreaterThan(200);
  });

  it('predicts a wall bounce, not a straight line', () => {
    const s: EngineState = {
      ...serve(createEngine(1)),
      ball: { x: CENTRE_X, y: 90, vx: SEGMENT_X, vy: 18 },
    };
    const predicted = predictIntercept(s, 'right');
    expect(predicted).not.toBeNull();
    // Straight-line y would be 90 + 18*3 = 144, far outside the court.
    expect(predicted as number).toBeGreaterThanOrEqual(BALL_MIN_Y - 1e-9);
    expect(predicted as number).toBeLessThanOrEqual(BALL_MAX_Y + 1e-9);
  });
});

describe('toDecisionState', () => {
  it('is numbers only, rounded to 1 decimal, with the right dir', () => {
    const s = serve(createEngine(11, 1));
    const d = toDecisionState(s, 'right');
    expect(d.court).toEqual({ w: COURT_W, h: COURT_H });
    expect(d.paddle).toEqual({ y: CENTRE_Y, h: PADDLE_H });
    expect(d.dir).toBe('toward');
    expect(toDecisionState(s, 'left').dir).toBe('away');
    expect(toDecisionState(s, 'left').interceptY).toBeNull();
    for (const n of [d.ball.x, d.ball.y, d.ball.vx, d.ball.vy, d.paddle.y, d.interceptY ?? 0]) {
      expect(Number.isFinite(n)).toBe(true);
      expect(Math.round(n * 10) / 10).toBe(n);
    }
    expect(JSON.parse(JSON.stringify(d))).toEqual(d);
  });

  it('stays small on the wire', () => {
    const bytes = JSON.stringify(toDecisionState(serve(createEngine(2)), 'right')).length;
    expect(bytes).toBeLessThan(260);
  });
});

describe('applyTick — ball motion', () => {
  it('advances exactly SEGMENT_X along x per tick', () => {
    let s = serve(createEngine(17));
    for (let i = 0; i < 5; i += 1) {
      const before = s.ball.x;
      s = applyTick(s, 'stay', 'auto');
      if (s.status !== 'playing') break;
      const travelled = Math.abs(s.ball.x - before);
      // Either a clean segment, or a segment folded around a paddle plane.
      const folded = Math.abs(2 * (RIGHT_PLANE - before) - SEGMENT_X);
      expect(Math.min(Math.abs(travelled - SEGMENT_X), Math.abs(travelled - folded))).toBeLessThan(1e-9);
    }
  });

  it('keeps the ball inside the court and |vx| constant', () => {
    let s = serve(createEngine(23));
    for (let i = 0; i < 2000 && s.status !== 'over'; i += 1) {
      s = applyTick(s, perfectRight(s), 'auto');
      if (s.status === 'playing') {
        expect(s.ball.y).toBeGreaterThanOrEqual(BALL_MIN_Y - 1e-9);
        expect(s.ball.y).toBeLessThanOrEqual(BALL_MAX_Y + 1e-9);
        expect(s.ball.x).toBeGreaterThanOrEqual(LEFT_PLANE - 1e-9);
        expect(s.ball.x).toBeLessThanOrEqual(RIGHT_PLANE + 1e-9);
        expect(Math.abs(s.ball.vx)).toBeCloseTo(SEGMENT_X, 10);
      }
    }
  });

  it('increments tick on every applyTick, across points, without resetting', () => {
    // Both paddles parked, so points are conceded and re-served repeatedly:
    // the tick counter must stay monotonic straight through them.
    let s = createEngine(31);
    let i = 0;
    while (s.status !== 'over' && i < 200) {
      i += 1;
      s = applyTick(s, 'stay', 'stay');
      expect(s.tick).toBe(i);
    }
    expect(s.score[0] + s.score[1]).toBe(POINTS_TO_WIN); // points really were conceded
    expect(s.tick).toBeGreaterThan(POINTS_TO_WIN); // and the counter never rewound
  });
});

describe('the paddle planes', () => {
  it('turns the ball at the paddle plane, not at the court edge', () => {
    // Head-on at the right paddle from the centre line: the ball must come back
    // from x = RIGHT_PLANE (156), never from the wall behind it (160).
    const flat: EngineState = {
      ...serve(createEngine(1)),
      ball: { x: CENTRE_X, y: CENTRE_Y, vx: SEGMENT_X, vy: 0 },
    };
    let s = flat;
    while (s.ball.vx > 0) s = applyTick(s, 'stay', 'stay');
    expect(s.ball.x).toBeCloseTo(RIGHT_PLANE, 9);

    while (s.ball.vx < 0) s = applyTick(s, 'stay', 'stay');
    expect(s.ball.x).toBeCloseTo(LEFT_PLANE, 9);
  });

  it('never lets the ball centre past either plane, over 500+ rallies', () => {
    // The ball is drawn as a disc at this position and the paddle is drawn ON
    // the plane, so a ball beyond the plane is a ball inside the paddle.
    let checked = 0;
    for (let seed = 1; seed <= 60; seed += 1) {
      let s = serve(createEngine(seed));
      for (let i = 0; i < 400; i += 1) {
        // Every combination of paddle behaviour: perfect, parked and wrong.
        const right: Move = i % 7 === 0 ? 'up' : perfectRight(s);
        s = applyTick(s, right, i % 11 === 0 ? 'down' : 'auto');
        expect(s.ball.x).toBeGreaterThanOrEqual(LEFT_PLANE - 1e-9);
        expect(s.ball.x).toBeLessThanOrEqual(RIGHT_PLANE + 1e-9);
        checked += 1;
      }
    }
    expect(checked).toBeGreaterThan(500);
  });

  it('crosses plane to plane in exactly SEGMENTS_PER_CROSSING decisions', () => {
    let s = serve(createEngine(8));
    while (s.ball.x !== RIGHT_PLANE && s.ball.x !== LEFT_PLANE) {
      s = applyTick(s, perfectRight(s), 'auto');
    }
    const startedAt = s.tick;
    const from = s.ball.x;
    do {
      s = applyTick(s, perfectRight(s), 'auto');
    } while (s.ball.x === from || (s.ball.x !== RIGHT_PLANE && s.ball.x !== LEFT_PLANE));
    expect(s.tick - startedAt).toBe(SEGMENTS_PER_CROSSING);
    expect(SEGMENT_X * SEGMENTS_PER_CROSSING).toBeCloseTo(RIGHT_PLANE - LEFT_PLANE, 9);
  });
});

describe('applyTick — paddles', () => {
  it('moves the model paddle by exactly MODEL_PADDLE_STEP', () => {
    const s = serve(createEngine(4));
    expect(applyTick(s, 'up', 'stay').rightY).toBe(s.rightY - MODEL_PADDLE_STEP);
    expect(applyTick(s, 'down', 'stay').rightY).toBe(s.rightY + MODEL_PADDLE_STEP);
  });

  it('clamps the right paddle to the court at both ends', () => {
    const base = serve(createEngine(4));
    expect(applyTick({ ...base, rightY: PADDLE_MIN_Y }, 'up', 'stay').rightY).toBe(PADDLE_MIN_Y);
    expect(applyTick({ ...base, rightY: PADDLE_MIN_Y + 5 }, 'up', 'stay').rightY).toBe(PADDLE_MIN_Y);
    expect(applyTick({ ...base, rightY: PADDLE_MAX_Y }, 'down', 'stay').rightY).toBe(PADDLE_MAX_Y);
    expect(applyTick({ ...base, rightY: PADDLE_MAX_Y - 5 }, 'down', 'stay').rightY).toBe(PADDLE_MAX_Y);
  });

  it('clamps a manual left paddle the same way', () => {
    const base = serve(createEngine(4));
    expect(applyTick({ ...base, leftY: PADDLE_MIN_Y }, 'stay', 'up').leftY).toBe(PADDLE_MIN_Y);
    expect(applyTick({ ...base, leftY: PADDLE_MIN_Y + 5 }, 'stay', 'up').leftY).toBe(PADDLE_MIN_Y);
    expect(applyTick({ ...base, leftY: PADDLE_MAX_Y - 5 }, 'stay', 'down').leftY).toBe(PADDLE_MAX_Y);
    expect(applyTick({ ...base, leftY: 50 }, 'stay', 'up').leftY).toBe(50 - MODEL_PADDLE_STEP);
  });

  it('keeps the scripted wall inside the court too', () => {
    let s = serve(createEngine(4));
    for (let i = 0; i < 300 && s.status !== 'over'; i += 1) {
      s = applyTick(s, perfectRight(s), 'auto');
      expect(s.leftY).toBeGreaterThanOrEqual(PADDLE_MIN_Y);
      expect(s.leftY).toBeLessThanOrEqual(PADDLE_MAX_Y);
      expect(s.rightY).toBeGreaterThanOrEqual(PADDLE_MIN_Y);
      expect(s.rightY).toBeLessThanOrEqual(PADDLE_MAX_Y);
    }
  });

  it("'stay' does not move a paddle", () => {
    const s = serve(createEngine(4));
    const next = applyTick(s, 'stay', 'stay');
    expect(next.rightY).toBe(s.rightY);
    expect(next.leftY).toBe(s.leftY);
  });
});

describe('the scripted wall never misses', () => {
  it('concedes zero points across 500+ rallies against a perfect right paddle', () => {
    let rallies = 0;
    for (let seed = 1; seed <= 60; seed += 1) {
      let s = serve(createEngine(seed));
      for (let i = 0; i < 3000 && s.status !== 'over'; i += 1) {
        const wasHeadingLeft = s.ball.vx < 0;
        s = applyTick(s, perfectRight(s), 'auto');
        // The right paddle scores only when the wall misses. It never may.
        expect(s.score[1]).toBe(0);
        if (wasHeadingLeft && s.status === 'playing' && s.ball.vx > 0) rallies += 1;
      }
      // And now that MODEL_PADDLE_STEP gives the model the reach to answer
      // every ball, NOBODY scores: a perfect rally runs forever. This used to
      // assert the wall won 5-0, which only happened because the model ran out
      // of paddle (6 x 10 = 60 units of reach against a range of 80).
      expect(s.score).toEqual([0, 0]);
      expect(s.status).toBe('playing');
    }
    expect(rallies).toBeGreaterThanOrEqual(500);
  });

  it('gives the model the reach to cover every intercept it is sent', () => {
    // The reach budget from the engine header, checked rather than asserted: a
    // model answering on every tick covers the ball when it arrives, from any
    // angle. 8 decisions x 12 units = 96 against a paddle range of 80.
    let returns = 0;
    let worst = 0;
    for (let seed = 1; seed <= 25; seed += 1) {
      let s = serve(createEngine(seed));
      for (let i = 0; i < 600; i += 1) {
        const target = predictIntercept(s, 'right');
        const next = applyTick(s, perfectRight(s), 'auto');
        if (target !== null && next.status === 'playing' && s.ball.vx > 0 && next.ball.vx < 0) {
          worst = Math.max(worst, Math.abs(target - next.rightY));
          returns += 1;
        }
        s = next;
      }
    }
    expect(returns).toBeGreaterThan(100);
    expect(worst).toBeLessThanOrEqual(PADDLE_REACH + 1e-9);
    expect(SEGMENTS_PER_CROSSING * MODEL_PADDLE_STEP).toBeGreaterThan(PADDLE_MAX_Y - PADDLE_MIN_Y);
    expect(SEGMENTS_PER_CROSSING * WALL_STEP).toBeGreaterThan(PADDLE_MAX_Y - PADDLE_MIN_Y);
  });

  it('covers the intercept it was chasing at the moment of the return', () => {
    for (let seed = 1; seed <= 25; seed += 1) {
      let s = serve(createEngine(seed));
      for (let i = 0; i < 600 && s.status !== 'over'; i += 1) {
        const target = predictIntercept(s, 'left');
        const next = applyTick(s, perfectRight(s), 'auto');
        if (target !== null && next.status === 'playing' && s.ball.vx < 0 && next.ball.vx > 0) {
          expect(Math.abs(target - next.leftY)).toBeLessThanOrEqual(PADDLE_REACH + 1e-9);
        }
        s = next;
      }
    }
  });
});

describe('scoring, serve direction and game over', () => {
  it('scores for the left when the right paddle never moves', () => {
    let s = serve(createEngine(13, 1));
    const parked = { ...s, rightY: PADDLE_MIN_Y, ball: { ...s.ball, y: BALL_MAX_Y - 1, vy: 0 } };
    s = parked;
    let ticks = 0;
    while (s.status === 'playing' && ticks < 20) {
      s = applyTick(s, 'stay', 'auto');
      ticks += 1;
    }
    expect(s.status).toBe('point');
    expect(s.score).toEqual([1, 0]);
    // Ball parked at the centre, ready to be served back toward the loser.
    expect(s.ball).toEqual({ x: CENTRE_X, y: CENTRE_Y, vx: 0, vy: 0 });
    expect(s.serveDir).toBe(1);
  });

  it('serves toward the side that just lost the point', () => {
    // Right loses -> serveDir +1 (back toward the right).
    let s = serve(createEngine(13, 1));
    s = { ...s, rightY: PADDLE_MIN_Y, ball: { ...s.ball, y: BALL_MAX_Y - 1, vy: 0 } };
    while (s.status === 'playing') s = applyTick(s, 'stay', 'auto');
    expect(s.serveDir).toBe(1);
    expect(serve(s).ball.vx).toBeGreaterThan(0);

    // Left loses -> serveDir -1 (back toward the left).
    let t = serve(createEngine(13, -1));
    t = { ...t, leftY: PADDLE_MIN_Y, ball: { ...t.ball, y: BALL_MAX_Y - 1, vy: 0 } };
    while (t.status === 'playing') t = applyTick(t, 'stay', 'stay');
    expect(t.score).toEqual([0, 1]);
    expect(t.serveDir).toBe(-1);
    expect(serve(t).ball.vx).toBeLessThan(0);
  });

  it('auto-serves out of idle and out of point', () => {
    const idle = createEngine(19);
    const first = applyTick(idle, 'stay', 'auto');
    expect(first.status).toBe('playing');
    expect(first.tick).toBe(1);

    let s = serve(createEngine(13, 1));
    s = { ...s, rightY: PADDLE_MIN_Y, ball: { ...s.ball, y: BALL_MAX_Y - 1, vy: 0 } };
    while (s.status === 'playing') s = applyTick(s, 'stay', 'auto');
    expect(s.status).toBe('point');
    const resumed = applyTick(s, 'stay', 'auto');
    expect(resumed.status).toBe('playing');
    expect(resumed.score).toEqual(s.score);
  });

  it("reaches 'over' at POINTS_TO_WIN and then ignores further ticks", () => {
    let s = createEngine(29, 1);
    for (let i = 0; i < 400 && s.status !== 'over'; i += 1) {
      // Right paddle parked at the top, so the left wins every point.
      s = applyTick({ ...s, rightY: PADDLE_MIN_Y }, 'up', 'auto');
    }
    expect(s.status).toBe('over');
    expect(s.score).toEqual([POINTS_TO_WIN, 0]);
    const after = applyTick(s, 'down', 'auto');
    expect(after).toBe(s);
  });
});

describe('determinism', () => {
  it('replays identically from the same seed and move list', () => {
    const moves: Move[] = ['up', 'stay', 'down', 'down', 'up', 'stay'];
    const run = () => {
      let s = serve(createEngine(2026));
      const trace: EngineState[] = [];
      for (let i = 0; i < 500 && s.status !== 'over'; i += 1) {
        s = applyTick(s, moves[i % moves.length], 'auto');
        trace.push(s);
      }
      return trace;
    };
    expect(JSON.stringify(run())).toBe(JSON.stringify(run()));
  });

  it('diverges between seeds', () => {
    const run = (seed: number) => {
      let s = serve(createEngine(seed));
      for (let i = 0; i < 40; i += 1) s = applyTick(s, 'stay', 'auto');
      return JSON.stringify(s);
    };
    expect(run(1)).not.toBe(run(2));
  });
});

describe('helpers', () => {
  it('clamp and moveToward behave', () => {
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(-5, 0, 10)).toBe(0);
    expect(clamp(50, 0, 10)).toBe(10);
    expect(moveToward(50, null)).toBe('stay');
    expect(moveToward(50, 50 + MODEL_PADDLE_STEP / 2 - 0.01)).toBe('stay');
    expect(moveToward(50, 20)).toBe('up');
    expect(moveToward(50, 80)).toBe('down');
  });

  it('paddle reach accounts for the ball radius', () => {
    expect(PADDLE_REACH).toBe(PADDLE_H / 2 + BALL_R);
  });
});
