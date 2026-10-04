import { describe, expect, it } from 'vitest';
import {
  COURT_H,
  COURT_W,
  SEGMENT_X,
  type ModelId,
  type Replay,
  type Snapshot,
} from '../game/types';
import { pathAt, type Ball } from './path';
import { createLaneTimeline, createReplayTimeline, laneStats } from './timeline';
import { CLIP_TOKENS } from './tokens';

/**
 * A tiny synthetic lane, built by walking the same sub-tick path the engine
 * walks and rounding the result the way a recorded replay does. The rounding
 * matters: it is exactly the divergence the timeline has to blend out, so a
 * test on rounded snapshots proves the blend works.
 */
function buildLane(steps: Array<{ latencyMs: number; rightY: number; leftY: number }>): {
  model: ModelId;
  label: string;
  snapshots: Snapshot[];
} {
  const round = (b: Ball): Ball => ({
    x: Number(b.x.toFixed(2)),
    y: Number(b.y.toFixed(2)),
    vx: Number(b.vx.toFixed(3)),
    vy: Number(b.vy.toFixed(3)),
  });

  // vy is chosen so the ball is still within PADDLE_REACH of a paddle parked
  // near the centre when it arrives: 8 segments per crossing (4 from the centre
  // line) times 3 units of y is 12, and the fixtures park their paddles within
  // 2 units of the serve height. A steeper serve would miss on the first
  // crossing and there would be no rally to sample.
  let ball: Ball = { x: COURT_W / 2, y: COURT_H / 2, vx: SEGMENT_X, vy: 3 };
  let t = 0;
  let leftY = 50;
  let rightY = 50;
  const score: [number, number] = [0, 0];

  const snapshots: Snapshot[] = [
    {
      t: 0,
      tick: 0,
      seed: 1,
      ball: { ...ball },
      leftY,
      rightY,
      score: [...score] as [number, number],
      model: 'mock',
      latencyMs: null,
      move: null,
      status: 'serving',
    },
  ];

  steps.forEach((step, i) => {
    leftY = step.leftY;
    rightY = step.rightY;
    t += step.latencyMs;
    ball = round(pathAt(ball, leftY, rightY, SEGMENT_X));
    // Off the court means the point was conceded.
    if (ball.x < -1 || ball.x > COURT_W + 1) {
      if (ball.x > COURT_W) score[0] += 1;
      else score[1] += 1;
      ball = { x: COURT_W / 2, y: COURT_H / 2, vx: 0, vy: 0 };
    }
    snapshots.push({
      t,
      tick: i + 1,
      seed: 1,
      ball: { ...ball },
      leftY,
      rightY,
      score: [...score] as [number, number],
      model: 'mock',
      latencyMs: step.latencyMs,
      move: 'stay',
      status: 'playing',
    });
  });

  return { model: 'mock', label: 'Mock', snapshots };
}

/** Three crossings with both paddles tracking, then one deliberate miss. */
function rallyThenMiss() {
  const chase: Array<{ latencyMs: number; rightY: number; leftY: number }> = [];
  // Paddles parked in the middle: the ball is served at y = 50 with a small
  // vy, so it comes back within reach for the first few crossings.
  for (let i = 0; i < 18; i += 1) {
    chase.push({ latencyMs: 90 + (i % 5) * 20, rightY: 52, leftY: 48 });
  }
  return chase;
}

function replayOf(lanes: Array<ReturnType<typeof buildLane>>): Replay {
  return { version: 1, recordedAt: '2026-09-17T00:00:00.000Z', seed: 1, lanes };
}

describe('createLaneTimeline', () => {
  const lane = buildLane(rallyThenMiss());
  const timeline = createLaneTimeline(lane, CLIP_TOKENS);

  it('lands exactly on the recorded ball position at every snapshot time', () => {
    for (const snap of lane.snapshots) {
      if (snap.ball.vx === 0) continue; // parked: nothing in play to match
      const view = timeline.viewAt(snap.t);
      expect(view.ballX).toBeCloseTo(snap.ball.x, 6);
      expect(view.ballY).toBeCloseTo(snap.ball.y, 6);
    }
  });

  it('serves with a 300 ms ease-out and then waits for the decision', () => {
    // Motion note 01: the ball leaves the centre immediately, in every lane at
    // once, and is parked one step out long before the model has answered.
    const slow = createLaneTimeline(
      buildLane([
        { latencyMs: 2000, rightY: 50, leftY: 50 },
        { latencyMs: 300, rightY: 50, leftY: 50 },
      ]),
      CLIP_TOKENS,
    );
    const serveMs = CLIP_TOKENS.motion.serveMs;
    const start = slow.viewAt(0).ballX;
    const early = slow.viewAt(serveMs * 0.5).ballX;
    const done = slow.viewAt(serveMs).ballX;

    expect(start).toBeCloseTo(COURT_W / 2, 6);
    expect(early).toBeGreaterThan(start);
    expect(done).toBeGreaterThan(early);
    // Eased, not linear: half the time has covered more than half the step.
    expect(early - start).toBeGreaterThan((done - start) * 0.5);
    // And then it holds until the decision actually lands, at 2000 ms.
    expect(slow.viewAt(1500).ballX).toBeCloseTo(done, 6);
  });

  it('is continuous across snapshot boundaries', () => {
    for (let i = 1; i < lane.snapshots.length; i += 1) {
      const prev = lane.snapshots[i - 1];
      const snap = lane.snapshots[i];
      // A conceded point resets the ball to the centre on purpose; that is the
      // one place the position is allowed to jump.
      if (prev.ball.vx === 0 || snap.ball.vx === 0) continue;

      const before = timeline.viewAt(snap.t - 1);
      const after = timeline.viewAt(snap.t + 1);
      const jump = Math.hypot(after.ballX - before.ballX, after.ballY - before.ballY);
      // 2 ms of travel at the fastest step here is well under one court unit.
      expect(jump).toBeLessThan(1.5);
    }
  });

  it('never stalls: the ball keeps moving right through the interval', () => {
    const snap = lane.snapshots[3];
    const next = lane.snapshots[4];
    const span = next.t - snap.t;
    const quarter = timeline.viewAt(snap.t + span * 0.25);
    const half = timeline.viewAt(snap.t + span * 0.5);
    const threeQuarters = timeline.viewAt(snap.t + span * 0.75);

    const d1 = Math.abs(half.ballX - quarter.ballX);
    const d2 = Math.abs(threeQuarters.ballX - half.ballX);
    expect(d1).toBeGreaterThan(SEGMENT_X * 0.2);
    expect(d2).toBeGreaterThan(SEGMENT_X * 0.2);
  });

  it('eases the paddles over the settle fraction of the interval, then holds', () => {
    const lanes = buildLane([
      { latencyMs: 100, rightY: 60, leftY: 50 },
      { latencyMs: 100, rightY: 60, leftY: 50 },
    ]);
    const tl = createLaneTimeline(lanes, CLIP_TOKENS);
    const settleMs = 100 * CLIP_TOKENS.motion.paddleSettle;

    expect(tl.viewAt(0).rightY).toBeCloseTo(50, 6);
    expect(tl.viewAt(settleMs * 0.5).rightY).toBeGreaterThan(50);
    expect(tl.viewAt(settleMs * 0.5).rightY).toBeLessThan(60);
    expect(tl.viewAt(settleMs).rightY).toBeCloseTo(60, 6);
    // Settled: it does not keep drifting for the rest of the interval.
    expect(tl.viewAt(90).rightY).toBeCloseTo(60, 6);
  });

  it('counts a return whenever the ball reverses at a paddle', () => {
    let reversals = 0;
    for (let i = 1; i < lane.snapshots.length; i += 1) {
      const a = lane.snapshots[i - 1].ball.vx;
      const b = lane.snapshots[i].ball.vx;
      if (a !== 0 && b !== 0 && Math.sign(a) !== Math.sign(b)) reversals += 1;
    }
    expect(reversals).toBeGreaterThan(0);
    expect(timeline.viewAt(timeline.durationMs).returns).toBe(reversals);
  });

  it('counts a miss on every score change', () => {
    // Four steps to reach the right plane from the centre line, one more for
    // the ball to leave the court behind the paddle that walked away.
    const missed = buildLane([
      { latencyMs: 100, rightY: 50, leftY: 50 },
      { latencyMs: 100, rightY: 50, leftY: 50 },
      // Paddle walks away; the ball leaves the court and the point is conceded.
      { latencyMs: 100, rightY: 5, leftY: 50 },
      { latencyMs: 100, rightY: 5, leftY: 50 },
      { latencyMs: 100, rightY: 5, leftY: 50 },
      { latencyMs: 100, rightY: 5, leftY: 50 },
    ]);
    const tl = createLaneTimeline(missed, CLIP_TOKENS);
    const changes = missed.snapshots.filter(
      (s, i) =>
        i > 0 &&
        (s.score[0] !== missed.snapshots[i - 1].score[0] ||
          s.score[1] !== missed.snapshots[i - 1].score[1]),
    ).length;

    expect(changes).toBe(1);
    expect(tl.viewAt(tl.durationMs).misses).toBe(1);
  });

  it('reports the latency of the last decision that landed', () => {
    const first = lane.snapshots[1];
    expect(timeline.viewAt(0).latencyMs).toBeNull();
    expect(timeline.viewAt(first.t).latencyMs).toBe(first.latencyMs);
    expect(timeline.viewAt(first.t - 1).latencyMs).toBeNull();
  });

  it('clamps before the first snapshot and freezes after the last', () => {
    const last = lane.snapshots[lane.snapshots.length - 1];
    const past = timeline.viewAt(timeline.durationMs + 60_000);
    expect(past.tick).toBe(last.tick);
    expect(timeline.viewAt(-5_000).tick).toBe(0);
  });

  it('keeps the ball inside the court while it is in play', () => {
    for (let t = 0; t <= timeline.durationMs; t += 7) {
      const view = timeline.viewAt(t);
      if (view.ballHidden) continue;
      // The ball is allowed past the paddle plane on a miss, but never wildly.
      expect(view.ballX).toBeGreaterThan(-SEGMENT_X * 1.5);
      expect(view.ballX).toBeLessThan(COURT_W + SEGMENT_X * 1.5);
      expect(view.ballY).toBeGreaterThanOrEqual(0);
      expect(view.ballY).toBeLessThanOrEqual(COURT_H);
      expect(view.trail.length).toBe(0);
    }
  });
});

describe('numberAt', () => {
  // A quick decision, then one that takes far longer: exactly the case the
  // count-up exists for.
  const lane = buildLane([
    { latencyMs: 400, rightY: 50, leftY: 50 },
    { latencyMs: 2000, rightY: 50, leftY: 50 },
    { latencyMs: 400, rightY: 50, leftY: 50 },
  ]);
  const timeline = createLaneTimeline(lane, CLIP_TOKENS);
  const arrival = lane.snapshots[1].t;
  const m = CLIP_TOKENS.motion;

  it('counts up from zero in the in-flight style before the first decision', () => {
    // Motion note 04: max(last landed latency, elapsed since the snapshot).
    expect(timeline.numberAt(0).value).toBeNull();
    const counting = timeline.numberAt(250);
    expect(counting.value).toBeCloseTo(250, 6);
    expect(counting.inFlight).toBe(true);
  });

  it('snaps to the landed value, flares, then flashes out', () => {
    const landed = timeline.numberAt(arrival);
    expect(landed.value).toBe(400);
    expect(landed.inFlight).toBe(false);
    expect(landed.flash).toBeCloseTo(1, 6);

    // The flare rises for 60 ms and falls back over the remaining 120 ms.
    expect(timeline.numberAt(arrival).pulse).toBe(0);
    expect(timeline.numberAt(arrival + m.pulseOutMs).pulse).toBeCloseTo(1, 6);
    expect(timeline.numberAt(arrival + m.pulseMs).pulse).toBe(0);
    expect(timeline.numberAt(arrival + m.flashMs).flash).toBe(0);
  });

  it('overtakes the last reading once the next decision runs long', () => {
    const over = timeline.numberAt(arrival + 700);
    expect(over.inFlight).toBe(true);
    expect(over.value).toBeCloseTo(700, 6);
  });

  it('does not flare on the serve, which carries no latency', () => {
    expect(timeline.numberAt(0).pulse).toBe(0);
    expect(timeline.numberAt(0).flash).toBe(0);
  });
});

describe('trails', () => {
  const lane = buildLane(rallyThenMiss());
  const timeline = createLaneTimeline(lane, CLIP_TOKENS);

  it('samples the real path, one point per span, nearest the ball first', () => {
    const spans = [2, 5, 9, 14, 20];
    const t = lane.snapshots[6].t + 40;
    const view = timeline.viewAt(t, spans);

    expect(view.trail.length).toBe(spans.length);
    let previous = Math.hypot(view.trail[0].x - view.ballX, view.trail[0].y - view.ballY);
    for (let i = 1; i < view.trail.length; i += 1) {
      const d = Math.hypot(view.trail[i].x - view.ballX, view.trail[i].y - view.ballY);
      expect(d).toBeGreaterThan(previous);
      previous = d;
    }
    // Every disc stays on the court, which is what "samples the real path"
    // buys over drawing a straight line backwards.
    for (const point of view.trail) {
      expect(point.y).toBeGreaterThanOrEqual(0);
      expect(point.y).toBeLessThanOrEqual(COURT_H);
    }
  });

  it('never reaches back past the start of the rally', () => {
    const view = timeline.viewAt(20, [200, 400]);
    expect(view.trail.length).toBeLessThan(2);
  });
});

describe('stats', () => {
  const lane = buildLane(rallyThenMiss());

  it('laneStats summarises decisions, latency and outcomes', () => {
    const stats = laneStats('mock', 'Mock', lane.snapshots);
    expect(stats.decisions).toBe(lane.snapshots.length - 1);
    expect(stats.avgMs).toBeGreaterThan(0);
    expect(stats.p95Ms).toBeGreaterThanOrEqual(stats.avgMs);
    expect(stats.decisionsPerSecond).toBeGreaterThan(0);
    expect(stats.durationMs).toBe(lane.snapshots[lane.snapshots.length - 1].t);
  });

  it('statsIn only counts what happened inside the window', () => {
    const timeline = createLaneTimeline(lane, CLIP_TOKENS);
    const whole = timeline.statsAt(timeline.durationMs);
    const tail = timeline.statsIn(timeline.durationMs / 2, timeline.durationMs);
    expect(tail.decisions).toBeLessThan(whole.decisions);
    expect(tail.returns).toBeLessThanOrEqual(whole.returns);
  });
});

describe('createReplayTimeline', () => {
  it('builds one timeline per lane and reports the longest', () => {
    const short = buildLane([{ latencyMs: 100, rightY: 50, leftY: 50 }]);
    const long = buildLane([
      { latencyMs: 100, rightY: 50, leftY: 50 },
      { latencyMs: 900, rightY: 50, leftY: 50 },
    ]);
    const { lanes, durationMs } = createReplayTimeline(replayOf([short, long]), CLIP_TOKENS);
    expect(lanes).toHaveLength(2);
    expect(durationMs).toBe(1000);
  });

  it('survives an empty lane', () => {
    const empty = { model: 'mock' as ModelId, label: 'Mock', snapshots: [] };
    const { lanes } = createReplayTimeline(replayOf([empty]), CLIP_TOKENS);
    const view = lanes[0].viewAt(500);
    expect(view.ballHidden).toBe(true);
    expect(view.score).toEqual([0, 0]);
    expect(lanes[0].numberAt(500)).toEqual({ value: null, inFlight: false, pulse: 0, flash: 0 });
  });
});
