/**
 * LiveSampler, driven entirely on fake times: every method takes the clock, so
 * a test can push a snapshot "at 1000ms" and read the view "at 1150ms" without
 * waiting for anything.
 */
import { describe, expect, it } from 'vitest';
import {
  COURT_H,
  COURT_W,
  RIGHT_PLANE,
  SEGMENT_X,
  type LaneStatus,
  type Snapshot,
} from '../game/types';
import {
  CATCH_UP,
  MAX_INTERVAL_MS,
  MIN_INTERVAL_MS,
  createLiveSampler,
  type LiveSampler,
  type LiveView,
} from './live';

let tick = 0;

/** A snapshot with everything defaulted to a live, playing lane. */
function snap(over: Partial<Snapshot> = {}): Snapshot {
  tick += 1;
  return {
    t: tick * 100,
    tick,
    seed: 1,
    ball: { x: COURT_W / 2, y: COURT_H / 2, vx: SEGMENT_X, vy: 2 },
    leftY: 50,
    rightY: 50,
    score: [0, 0],
    model: 'jev',
    latencyMs: 200,
    move: 'stay',
    status: 'playing' as LaneStatus,
    ...over,
  };
}

/**
 * A lane mid-rally: two decisions one segment and 300ms apart on the worker's
 * clock, arriving on time.
 */
function rally(humanSide: 'left' | 'right' | null = null) {
  tick = 0;
  const sampler = createLiveSampler({ humanSide });
  const first = snap({
    t: 1_000,
    ball: { x: 60, y: 40, vx: SEGMENT_X, vy: 2 },
    leftY: 40,
    rightY: 40,
  });
  const second = snap({
    t: 1_300,
    ball: { x: 60 + SEGMENT_X, y: 42, vx: SEGMENT_X, vy: 2 },
    leftY: 50,
    rightY: 60,
  });
  sampler.push(first, 1_000);
  sampler.push(second, 1_300);
  return { sampler, first, second };
}

function view(v: LiveView | null): LiveView {
  if (v === null) throw new Error('expected a view');
  return v;
}

describe('createLiveSampler', () => {
  it('has nothing to show before the first snapshot', () => {
    expect(createLiveSampler().viewAt(0)).toBeNull();
    expect(createLiveSampler().latest()).toBeNull();
  });

  it('holds on the first snapshot rather than inventing motion', () => {
    tick = 0;
    const sampler = createLiveSampler();
    const only = snap();
    sampler.push(only, 500);
    const v = view(sampler.viewAt(900));
    expect(v.ballX).toBe(only.ball.x);
    expect(v.ballY).toBe(only.ball.y);
    expect(sampler.latest()).toBe(only);
  });

  it("walks from the previous snapshot to the latest over the step's own duration", () => {
    const { sampler, first, second } = rally();

    // At the moment the latest arrived, the ball is still on the previous one.
    const start = view(sampler.viewAt(1_300));
    expect(start.ballX).toBeCloseTo(first.ball.x, 6);

    // Half way through the measured interval, half a segment along.
    const middle = view(sampler.viewAt(1_450));
    expect(middle.ballX).toBeGreaterThan(first.ball.x);
    expect(middle.ballX).toBeLessThan(second.ball.x);
    expect(middle.ballX).toBeCloseTo(first.ball.x + SEGMENT_X / 2, 1);

    // And it lands exactly on the worker's own position, never past it.
    const end = view(sampler.viewAt(1_600));
    expect(end.ballX).toBeCloseTo(second.ball.x, 6);
    expect(end.ballY).toBeCloseTo(second.ball.y, 6);
    expect(end.settled).toBe(true);
  });

  it('never draws ahead of the latest snapshot, however late the next one is', () => {
    const { sampler, second } = rally();
    for (const at of [1_600, 2_000, 10_000, 120_000]) {
      const v = view(sampler.viewAt(at));
      expect(v.ballX).toBeCloseTo(second.ball.x, 6);
      expect(v.ballY).toBeCloseTo(second.ball.y, 6);
      expect(v.settled).toBe(true);
    }
  });

  it('clamps the interval it draws over', () => {
    // Two frames that really move the ball: a paddle frame repeats the ball it
    // already published and is not a step at all (see the press tests below).
    const step = (at: number) => {
      const sampler = createLiveSampler();
      sampler.push(snap({ t: 0, ball: { x: 60, y: 50, vx: SEGMENT_X, vy: 0 } }), 0);
      sampler.push(snap({ t: at, ball: { x: 60 + SEGMENT_X, y: 50, vx: SEGMENT_X, vy: 0 } }), at);
      return sampler.intervalMs();
    };

    tick = 0;
    expect(step(10)).toBe(MIN_INTERVAL_MS);
    tick = 0;
    expect(step(9_000)).toBe(MAX_INTERVAL_MS);
  });

  it('crosses a segment in MAX_INTERVAL_MS even when the model took four seconds', () => {
    tick = 0;
    const sampler = createLiveSampler();
    sampler.push(snap({ t: 0, ball: { x: 60, y: 50, vx: SEGMENT_X, vy: 0 } }), 0);
    sampler.push(snap({ t: 4_000, ball: { x: 60 + SEGMENT_X, y: 50, vx: SEGMENT_X, vy: 0 } }), 4_000);
    // Arrived at 4000, drawn over 1500, then waits for the next decision.
    expect(view(sampler.viewAt(4_000 + MAX_INTERVAL_MS)).ballX).toBeCloseTo(60 + SEGMENT_X, 6);
    expect(view(sampler.viewAt(4_000 + MAX_INTERVAL_MS / 2)).ballX).toBeCloseTo(
      60 + SEGMENT_X / 2,
      1,
    );
  });

  it('takes the human paddle exactly from the latest snapshot and eases the model one', () => {
    const { sampler, first, second } = rally('left');

    const early = view(sampler.viewAt(1_310));
    // The human's side is the worker's word: no easing, no lag.
    expect(early.leftY).toBe(second.leftY);
    // The model's side is on its way from the previous position.
    expect(early.rightY).toBeGreaterThan(first.rightY);
    expect(early.rightY).toBeLessThan(second.rightY);

    expect(view(sampler.viewAt(1_600)).rightY).toBeCloseTo(second.rightY, 6);
  });

  it('eases both paddles when nobody is playing (a spectator)', () => {
    const { sampler, first, second } = rally(null);
    const early = view(sampler.viewAt(1_305));
    expect(early.leftY).toBeGreaterThan(first.leftY);
    expect(early.leftY).toBeLessThan(second.leftY);
  });

  it('holds the ball still on a frame that did not move it', () => {
    // Paddle frames and heartbeats carry the same ball. Walking a segment
    // across one of those would swing the ball out and drag it back.
    tick = 0;
    const sampler = createLiveSampler();
    const ball = { x: 100, y: 30, vx: -SEGMENT_X, vy: 3 };
    sampler.push(snap({ ball: { ...ball }, leftY: 40 }), 0);
    sampler.push(snap({ ball: { ...ball }, leftY: 48, tick: 7 }), 100);

    for (const at of [100, 130, 170, 200, 400]) {
      const v = view(sampler.viewAt(at));
      expect(v.ballX).toBeCloseTo(ball.x, 6);
      expect(v.ballY).toBeCloseTo(ball.y, 6);
    }
  });

  it('flies the ball off the court on a conceded point, then hides it', () => {
    tick = 0;
    const sampler = createLiveSampler();
    sampler.push(snap({ t: 0, ball: { x: RIGHT_PLANE - SEGMENT_X, y: 20, vx: SEGMENT_X, vy: 0 } }), 0);
    // The worker parks the ball at the centre and the score moves.
    sampler.push(
      snap({
        t: 400,
        ball: { x: COURT_W / 2, y: COURT_H / 2, vx: 0, vy: 0 },
        score: [1, 0],
        status: 'point',
      }),
      400,
    );

    const leaving = view(sampler.viewAt(600));
    expect(leaving.ballHidden).toBe(false);
    expect(leaving.ballX).toBeGreaterThan(RIGHT_PLANE - SEGMENT_X);
    expect(leaving.score).toEqual([1, 0]);

    const gone = view(sampler.viewAt(900));
    expect(gone.ballHidden).toBe(true);
  });

  it('cuts rather than sweeps when the lane restarts', () => {
    tick = 0;
    const sampler = createLiveSampler();
    sampler.push(snap({ t: 9_000, tick: 40, ball: { x: 120, y: 80, vx: -SEGMENT_X, vy: 0 } }), 0);
    // A new game on the same channel: the lane clock and the tick both rewind.
    const fresh = snap({ t: 100, tick: 1, ball: { x: 60, y: 20, vx: SEGMENT_X, vy: 0 } });
    sampler.push(fresh, 300);

    // No 60-unit sweep across the court: the new game starts where it starts.
    expect(view(sampler.viewAt(310)).ballX).toBeCloseTo(fresh.ball.x, 6);
  });

  it('ignores a snapshot that arrives out of order', () => {
    tick = 0;
    const sampler = createLiveSampler();
    // A late frame has a lower tick as well, so it must not read as a restart.
    const a = snap({ t: 1_000, tick: 10 });
    const b = snap({ t: 900, tick: 9 });
    sampler.push(a, 0);
    sampler.push(b, 50);
    expect(sampler.latest()).toBe(a);
  });

  it('draws a trail along the real path, nearest the ball first', () => {
    const { sampler } = rally();
    const spans = [2, 5, 9, 14];
    const v = view(sampler.viewAt(1_450, spans));

    expect(v.trail).toHaveLength(spans.length);
    let previous = 0;
    for (const point of v.trail) {
      const d = Math.hypot(point.x - v.ballX, point.y - v.ballY);
      expect(d).toBeGreaterThan(previous);
      previous = d;
    }
    // The nearest disc is about its span behind the ball, along the path.
    expect(Math.hypot(v.trail[0].x - v.ballX, v.trail[0].y - v.ballY)).toBeCloseTo(
      spans[0],
      0,
    );
  });

  // ---------------------------------------------------------------- the jerk
  //
  // A held key makes the worker publish a paddle frame every PADDLE_TICK_MS
  // (100ms) BETWEEN the ball steps, which are PLAY_MIN_STEP_MS (350ms) apart.
  // Those frames carry the same ball. If they touch the ball's clock at all
  // the step is drawn over 100ms instead of 350ms and then holds — which is
  // exactly what "the ball jerks the moment I press a key" looks like.
  //
  // So: reconstruct that stream and measure the ball. Constant speed, forward
  // only, no hold longer than one sampled frame. The sampler is LIVE, so the
  // frames are pushed and the view is read on one clock, in order, the way the
  // page does it — sampling after the fact would read a sampler that has
  // already seen the future.
  describe('with paddle frames arriving between ball steps', () => {
    const BALL_STEP_MS = 350;
    const PADDLE_STEP_MS = 100;
    const SAMPLE_MS = 16;
    const START_X = 40;
    const STEPS = 4;

    /** The frames a player holding 'up' during a rally actually receives. */
    function pressFrames(): Array<{ at: number; frame: Snapshot }> {
      tick = 0;
      // Straight across the court, away from both planes: no bounce, so the
      // ball's x speed is the only thing being measured.
      const ball = (i: number) => ({ x: START_X + SEGMENT_X * i, y: 50, vx: SEGMENT_X, vy: 0 });
      const out: Array<{ at: number; frame: Snapshot }> = [];
      let paddleY = 80;

      for (let step = 0; step <= STEPS; step += 1) {
        const at = step * BALL_STEP_MS;
        // The decision frame: the ball moves, and so does the human paddle.
        paddleY -= 8;
        out.push({ at, frame: snap({ t: at, ball: ball(step), leftY: paddleY, rightY: 50 }) });
        // The paddle ticks in between. Same ball, new leftY, no new latency.
        for (let k = 1; k * PADDLE_STEP_MS < BALL_STEP_MS; k += 1) {
          paddleY -= 8;
          const tickAt = at + k * PADDLE_STEP_MS;
          out.push({
            at: tickAt,
            frame: snap({ t: tickAt, ball: ball(step), leftY: paddleY, rightY: 50, latencyMs: null }),
          });
        }
      }
      return out;
    }

    /**
     * Replay the stream at 60fps: push whatever has arrived by `at`, then read
     * the view. Exactly the order court.tsx does it in.
     */
    function replay(): { sampler: LiveSampler; xs: number[]; intervals: number[] } {
      const feed = pressFrames();
      const sampler = createLiveSampler({ humanSide: 'left' });
      const xs: number[] = [];
      const intervals: number[] = [];
      let next = 0;

      for (let at = 0; at <= STEPS * BALL_STEP_MS; at += SAMPLE_MS) {
        while (next < feed.length && feed[next].at <= at) {
          sampler.push(feed[next].frame, feed[next].at);
          next += 1;
        }
        // The first step has nothing to walk from, so it is not measured.
        if (at < BALL_STEP_MS) continue;
        xs.push(view(sampler.viewAt(at)).ballX);
        intervals.push(sampler.intervalMs());
      }
      return { sampler, xs, intervals };
    }

    it('measures the ball step from the ball frames, not from the paddle frames', () => {
      const { intervals } = replay();
      // 100ms here would be the paddle tick being mistaken for a ball step.
      expect(new Set(intervals)).toEqual(new Set([BALL_STEP_MS]));
    });

    it('moves the ball at a constant speed right through the press', () => {
      const { xs } = replay();
      const deltas: number[] = [];
      for (let i = 1; i < xs.length; i += 1) deltas.push(xs[i] - xs[i - 1]);

      const sorted = [...deltas].sort((a, b) => a - b);
      const median = sorted[Math.floor(sorted.length / 2)];
      // One segment per ball step, sampled every SAMPLE_MS.
      expect(median).toBeCloseTo((SEGMENT_X * SAMPLE_MS) / BALL_STEP_MS, 3);

      // No 3x burst, and no hold: every sampled frame moves the ball forward by
      // the same amount, within a tenth.
      for (const d of deltas) {
        expect(d).toBeGreaterThan(median * 0.9);
        expect(d).toBeLessThan(median * 1.1);
      }
    });

    it('still draws the human paddle exactly where the worker put it', () => {
      const { sampler } = replay();
      const wire = sampler.latest() as Snapshot;
      expect(view(sampler.viewAt(STEPS * BALL_STEP_MS)).leftY).toBe(wire.leftY);
    });

    it('eases the model paddle over its OWN step, not the human paddle tick', () => {
      // The model's paddle moves once per decision. A press does not make it
      // move faster, and the 100ms tick of the other paddle must not time it.
      tick = 0;
      const sampler = createLiveSampler({ humanSide: 'left' });
      const ball = (i: number) => ({ x: 40 + SEGMENT_X * i, y: 50, vx: SEGMENT_X, vy: 0 });
      sampler.push(snap({ ball: ball(0), leftY: 80, rightY: 40 }), 0);
      // Three paddle ticks: the human moves, the model does not.
      sampler.push(snap({ ball: ball(0), leftY: 72, rightY: 40 }), 100);
      sampler.push(snap({ ball: ball(0), leftY: 64, rightY: 40 }), 200);
      sampler.push(snap({ ball: ball(0), leftY: 56, rightY: 40 }), 300);
      // The decision: now the model's paddle moves, 350ms after its last one.
      sampler.push(snap({ ball: ball(1), leftY: 48, rightY: 60 }), 350);

      // Eased over PADDLE_SETTLE of 350ms (~122ms), so it is still on its way
      // at +60ms and has arrived by +150ms. Timed off the 100ms paddle tick it
      // would have finished in ~35ms and been there already.
      const early = view(sampler.viewAt(410)).rightY;
      expect(early).toBeGreaterThan(40);
      expect(early).toBeLessThan(60);
      expect(view(sampler.viewAt(500)).rightY).toBeCloseTo(60, 6);
    });
  });

  it("times a step by the worker's clock, not by when its frames happened to arrive", () => {
    tick = 0;
    const sampler = createLiveSampler();
    sampler.push(snap({ t: 0, ball: { x: 40, y: 50, vx: SEGMENT_X, vy: 0 } }), 0);
    // A 350ms step whose frame the network delivered 50ms late.
    sampler.push(snap({ t: 350, ball: { x: 40 + SEGMENT_X, y: 50, vx: SEGMENT_X, vy: 0 } }), 400);
    expect(sampler.intervalMs()).toBe(350);
    // Walked from its arrival, at the step's own speed.
    expect(view(sampler.viewAt(400 + 175)).ballX).toBeCloseTo(40 + SEGMENT_X / 2, 6);
  });

  it('continues from where the ball is drawn when a frame lands early, never jumping', () => {
    tick = 0;
    const sampler = createLiveSampler();
    const straight = (x: number) => ({ x, y: 50, vx: SEGMENT_X, vy: 0 });
    sampler.push(snap({ t: 0, ball: straight(40) }), 0);
    // A slow decision (600ms), then a fast one (350ms): the third frame lands
    // while the ball is still 58% of the way through the 600ms walk.
    sampler.push(snap({ t: 600, ball: straight(40 + SEGMENT_X) }), 600);
    const before = view(sampler.viewAt(949)).ballX;
    sampler.push(snap({ t: 950, ball: straight(40 + 2 * SEGMENT_X) }), 950);
    const after = view(sampler.viewAt(951)).ballX;

    // No teleport to the end of the old step (which would be 8 units at once).
    expect(after - before).toBeGreaterThanOrEqual(0);
    expect(after - before).toBeLessThan(0.5);

    // From here the ball makes up the leftover at no more than CATCH_UP x the
    // new step's speed, and lands exactly on the newest frame.
    const walkMs = Math.max(350, (350 * (SEGMENT_X * (1 - 350 / 600) + SEGMENT_X)) / SEGMENT_X / CATCH_UP);
    expect(sampler.intervalMs()).toBeCloseTo(walkMs, 6);
    let previous = after;
    for (let at = 951; at <= 950 + walkMs; at += 16) {
      const x = view(sampler.viewAt(at)).ballX;
      expect(x).toBeGreaterThanOrEqual(previous);
      previous = x;
    }
    expect(view(sampler.viewAt(950 + walkMs)).ballX).toBeCloseTo(40 + 2 * SEGMENT_X, 6);
  });

  it('carries the lane numbers through untouched', () => {
    const { sampler, second } = rally();
    const v = view(sampler.viewAt(1_400));
    expect(v.latencyMs).toBe(second.latencyMs);
    expect(v.tick).toBe(second.tick);
    expect(v.status).toBe(second.status);
    expect(v.score).toEqual(second.score);
  });
});
