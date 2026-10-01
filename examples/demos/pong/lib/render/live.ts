/**
 * lib/render/live.ts — a LIVE lane, sampled at any instant.
 *
 * lib/render/timeline.ts does this for a recording, where every snapshot and
 * its time is known in advance. A live lane has neither: snapshots arrive when
 * the model answers and the network allows, and the next one may be 200ms or
 * four seconds away. This is the same idea for that case, and nothing more.
 *
 * THE RULE: never draw ahead of the latest snapshot.
 *   When a ball frame lands, the ball starts walking toward it from wherever it
 *   is drawn and gets there over the time that step took on the WORKER's clock
 *   (the gap between the two frames' `t`) — so the ball is always walking a
 *   path the worker has already confirmed, one step behind. If the next frame
 *   is late the ball simply stops on the latest one and waits, which is the
 *   truth: the model has not answered yet.
 *
 * Two things this is careful about, because each was once a visible jerk:
 *   - The step is timed by the worker's clock, not by when its frames happened
 *     to arrive. Arrival carries the network's jitter on top, and a 350ms step
 *     whose frame landed 40ms late was being drawn 40ms slow, then the next
 *     one 40ms fast.
 *   - A frame that lands EARLY — before the ball has finished walking the
 *     previous step, because this step was shorter than that one — does not
 *     teleport the ball to the end of the old step. The walk continues from
 *     where the ball is drawn and covers the rest of the old step plus the new
 *     one, squeezed by at most CATCH_UP so the ball does not fall further
 *     behind. A latency spike therefore reads as the ball slowing and then
 *     easing back up to speed, never as a jump.
 *
 * That is the opposite of what the site did before, which was to start a tween
 * on arrival sized by the PREVIOUS decision's latency. A step that ran short
 * froze, a step that ran long got cut off, and the ball moved in lurches that
 * had nothing to do with the model.
 *
 * NOT EVERY FRAME IS A BALL FRAME, and this is the whole reason the sampler
 * keeps more than one clock. The worker publishes on three schedules: a
 * decision frame when the ball moves, a paddle frame every PADDLE_TICK_MS
 * (100ms) for as long as a paddle is moving, and a heartbeat every
 * HEARTBEAT_MS. Watching a game you only ever see the first kind, roughly
 * PLAY_MIN_STEP_MS (350ms) apart. PRESS A KEY and paddle frames start landing
 * between them — and a sampler that timed the ball off "the last two frames"
 * would then draw the ball's 350ms step in 100ms and hold it for the other
 * 250. That is what "the ball jerks the moment I press a key" was.
 *
 * So there are two clocks, and a frame only winds the one it belongs to:
 *   the ball clock   advanced ONLY by frames that moved the ball (or started
 *                    and stopped it). A paddle or heartbeat frame carries the
 *                    ball it already had, so it must not touch this at all.
 *   a paddle clock   one per side, advanced only when THAT side's paddle
 *                    moved — so the model's paddle eases over the decisions it
 *                    moves on, and a human paddle being watched eases over its
 *                    own 10Hz tick, and neither one is timed off the other.
 *
 * Paddles differ by side. The human's paddle is taken EXACTLY from the latest
 * snapshot — it is the worker's authoritative position, and the client overlays
 * its own local paddle on top of it anyway. The model's paddle is eased, because
 * it moves in one step per decision and a hard jump reads as a glitch.
 *
 * Pure, and clock-free: every method takes the time. Tests pass fake ones.
 */

import { SEGMENT_X, type LaneStatus, type Snapshot } from '../game/types';
import type { CourtView } from './court';
import { pathAt, type Ball } from './path';

/** Shortest interval a step is stretched over. Below this, motion is a jump. */
export const MIN_INTERVAL_MS = 80;
/** Longest. A four-second decision still crosses its segment in 1.5s, then waits. */
export const MAX_INTERVAL_MS = 1500;
/** Fraction of the interval the model's paddle takes to settle. */
export const PADDLE_SETTLE = 0.35;
/**
 * How much faster than the step's own speed the ball may run while it has a
 * leftover to make up (see the header). 1.15: a leftover of half a segment is
 * gone within three steps, and a 15% change of speed is below what reads as a
 * lurch.
 */
export const CATCH_UP = 1.15;

/** What a live lane looks like at one instant. */
export interface LiveView extends CourtView {
  /** The latency of the most recent decision that has landed. */
  latencyMs: number | null;
  tick: number;
  status: LaneStatus;
  /** True once the ball has arrived at the latest snapshot and is waiting. */
  settled: boolean;
}

export interface LiveSamplerOptions {
  /**
   * The side a human is playing. That paddle is taken exactly from the latest
   * snapshot, never eased. Null (the default) eases both, which is what a
   * spectator wants.
   */
  humanSide?: 'left' | 'right' | null;
  minIntervalMs?: number;
  maxIntervalMs?: number;
  paddleSettle?: number;
}

export interface LiveSampler {
  /** A snapshot off the wire, and the time it arrived. Out-of-order is ignored. */
  push(snapshot: Snapshot, arrivedAtMs: number): void;
  /**
   * The lane at `nowMs`. `trailSpans` are distances behind the ball in COURT
   * units, nearest first — build them with `trailSpans()` from ./court.
   */
  viewAt(nowMs: number, trailSpans?: number[]): LiveView | null;
  /** The newest snapshot pushed, or null before the first one. */
  latest(): Snapshot | null;
  /**
   * The interval the current BALL step is being drawn over, after clamping —
   * the gap between the last two ball frames on the worker's clock. Paddle and
   * heartbeat frames do not appear in it, however many of them land mid-step.
   */
  intervalMs(): number;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function clamp01(p: number): number {
  return clamp(p, 0, 1);
}

function easeOutCubic(p: number): number {
  return 1 - Math.pow(1 - p, 3);
}

/** A parked ball means the point is over and nothing is in play. */
function parked(s: Snapshot): boolean {
  return s.ball.vx === 0;
}

/**
 * A new game behind the same sampler: both the tick counter and the lane clock
 * have gone back a long way. A frame that merely arrived late has a lower tick
 * too, which is why the clock has to rewind as well before this says yes —
 * otherwise a late frame would be taken for a new game and cut to.
 */
export const RESTART_REWIND_MS = 2_000;

function isRestart(prev: Snapshot, next: Snapshot): boolean {
  return next.tick < prev.tick && next.t + RESTART_REWIND_MS < prev.t;
}

export function createLiveSampler(options: LiveSamplerOptions = {}): LiveSampler {
  const {
    humanSide = null,
    minIntervalMs = MIN_INTERVAL_MS,
    maxIntervalMs = MAX_INTERVAL_MS,
    paddleSettle = PADDLE_SETTLE,
  } = options;

  /** The newest frame of ANY kind. Its numbers (score, status, tick) are law. */
  let last: Snapshot | null = null;

  /* -------------------------------------------------------- the ball clock */
  /**
   * Wound only by a frame that moved the ball, or started or stopped it.
   * `target` is the ball frame being walked TO, `at` is when it arrived, and
   * `interval` is the gap to the ball frame before it — never the gap to some
   * paddle tick that happened to land in between.
   */
  let target: Snapshot | null = null;
  let ballAt = 0;
  let ballInterval = minIntervalMs;
  /**
   * Court x-units the current walk covers: one segment, plus whatever was left
   * of the previous walk when this frame landed early.
   */
  let stepDist = SEGMENT_X;
  /** Simulation error at p = 1, blended out so the ball lands exactly. */
  let errX = 0;
  let errY = 0;
  /** Where the ball starts this step. Only meaningful when `walkable`. */
  let from: Ball | null = null;
  /**
   * Whether the ball actually advances over this step. Paddle frames, heartbeat
   * frames and the serve frame all arrive with the ball where it already was,
   * and walking a segment across one of those would swing the ball out and drag
   * it back. Only a frame that moved the ball is walked.
   */
  let walkable = false;
  /** The point was conceded on this step: the ball leaves the court and goes. */
  let conceded = false;
  /** The ball reappeared at the centre line: fade it in rather than pop it. */
  let serving = false;

  /* ----------------------------------------------------- the paddle clocks */
  /**
   * One per side, wound only when that side's paddle moved. A paddle that has
   * not moved is finished easing and simply sits at `to`.
   */
  interface Ease {
    from: number;
    to: number;
    at: number;
    interval: number;
  }

  /** A paddle that is where it is, with its clock started. Nothing in flight. */
  function startEase(y: number, at: number): Ease {
    return { from: y, to: y, at, interval: minIntervalMs };
  }

  function wind(ease: Ease, y: number, arrivedAtMs: number): Ease {
    if (y === ease.to) return ease;
    return {
      from: ease.to,
      to: y,
      at: arrivedAtMs,
      interval: clamp(arrivedAtMs - ease.at, minIntervalMs, maxIntervalMs),
    };
  }

  function easedAt(ease: Ease, nowMs: number): number {
    if (ease.from === ease.to) return ease.to;
    const p = ease.interval <= 0 ? 1 : (nowMs - ease.at) / ease.interval;
    const settle = paddleSettle <= 0 ? 1 : easeOutCubic(clamp01(p / paddleSettle));
    return ease.from + (ease.to - ease.from) * settle;
  }

  let leftEase: Ease = startEase(0, 0);
  let rightEase: Ease = startEase(0, 0);

  /** How far through its walk the ball is at `nowMs`, 0..1. */
  function progressAt(nowMs: number): number {
    return clamp01(ballInterval <= 0 ? 1 : (nowMs - ballAt) / ballInterval);
  }

  /** The ball as drawn at progress `p`, with the velocity it has there. */
  function drawnBall(p: number): Ball {
    const ball = target as Snapshot;
    const sim = pathAt(from as Ball, ball.leftY, ball.rightY, stepDist * p);
    return { x: sim.x + errX * p, y: sim.y + errY * p, vx: sim.vx, vy: sim.vy };
  }

  /** Everything the ball clock owns, cut to `snapshot` with nothing in flight. */
  function cutTo(snapshot: Snapshot, arrivedAtMs: number): void {
    target = snapshot;
    ballAt = arrivedAtMs;
    from = { ...snapshot.ball };
    stepDist = SEGMENT_X;
    walkable = false;
    conceded = false;
    serving = false;
    errX = 0;
    errY = 0;
  }

  function push(snapshot: Snapshot, arrivedAtMs: number): void {
    if (last === null) {
      last = snapshot;
      cutTo(snapshot, arrivedAtMs);
      leftEase = startEase(snapshot.leftY, arrivedAtMs);
      rightEase = startEase(snapshot.rightY, arrivedAtMs);
      return;
    }
    // An out-of-order or duplicate frame would rewind the ball on screen.
    if (snapshot.t < last.t && !isRestart(last, snapshot)) return;

    const restart = isRestart(last, snapshot);
    last = snapshot;

    // Paddles first: they move on every kind of frame, ball or not.
    if (restart) {
      leftEase = startEase(snapshot.leftY, arrivedAtMs);
      rightEase = startEase(snapshot.rightY, arrivedAtMs);
    } else {
      leftEase = wind(leftEase, snapshot.leftY, arrivedAtMs);
      rightEase = wind(rightEase, snapshot.rightY, arrivedAtMs);
    }

    if (restart) {
      cutTo(snapshot, arrivedAtMs);
      return;
    }

    // THE TEST FOR A BALL FRAME. A paddle tick and a heartbeat both repeat the
    // ball they already published, so neither of these is true for them and the
    // ball clock is left exactly as it was — still mid-step, still at the speed
    // the decisions set. A serve moves no ball either, but it does un-park one,
    // which the fade-in is timed off.
    const previous = target as Snapshot;
    const moved = previous.ball.x !== snapshot.ball.x || previous.ball.y !== snapshot.ball.y;
    const startedOrStopped = parked(previous) !== parked(snapshot);
    if (!moved && !startedOrStopped) return;

    // Landed early? Then the ball is still mid-walk, and the walk continues
    // from where it is drawn rather than jumping to the end of the old step.
    const p = progressAt(arrivedAtMs);
    const leftover = walkable && p < 1 ? stepDist * (1 - p) : 0;
    const start: Ball = leftover > 0 ? drawnBall(p) : { ...previous.ball };

    // The step's own duration, on the worker's clock. The leftover is walked
    // at that same speed, squeezed by at most CATCH_UP.
    const stepMs = clamp(snapshot.t - previous.t, minIntervalMs, maxIntervalMs);
    stepDist = SEGMENT_X + leftover;
    ballInterval = Math.max(stepMs, (stepMs * stepDist) / SEGMENT_X / CATCH_UP);
    target = snapshot;
    ballAt = arrivedAtMs;
    from = start;
    conceded = !parked(previous) && parked(snapshot);
    serving = parked(previous) && !parked(snapshot);
    walkable = !parked(previous) && moved;

    // Where our own path walk thinks the walk ends. The difference is blended
    // out across it so the ball lands EXACTLY on the worker's position. A
    // conceded step has no such position — the ball is parked at the centre
    // while the real one flew off the court — so it is walked unblended.
    errX = 0;
    errY = 0;
    if (walkable && !conceded) {
      const end = pathAt(from, snapshot.leftY, snapshot.rightY, stepDist);
      errX = snapshot.ball.x - end.x;
      errY = snapshot.ball.y - end.y;
    }
  }

  /** Ball position `back` court-x-units before where it is now. */
  function ballBack(p: number, back: number): { x: number; y: number; ok: boolean } {
    const ball = target as Snapshot;
    // A step that does not move the ball still has a ball: the latest position.
    const base: Ball = walkable ? (from as Ball) : ball.ball;
    const travelled = walkable ? stepDist * p : 0;

    if (back <= travelled) {
      const dist = travelled - back;
      const sim = pathAt(base, ball.leftY, ball.rightY, dist);
      const q = stepDist === 0 ? 0 : dist / stepDist;
      return { x: sim.x + errX * q, y: sim.y + errY * q, ok: true };
    }
    // Behind the start of this step. Only this ball's own velocity is known, so
    // walk backwards along it: exact through a wall bounce, and a rally that
    // would reach a paddle has ended anyway.
    if (base.vx === 0) return { x: 0, y: 0, ok: false };
    const behind = pathAt(
      { x: base.x, y: base.y, vx: -base.vx, vy: -base.vy },
      ball.leftY,
      ball.rightY,
      back - travelled,
    );
    return { x: behind.x, y: behind.y, ok: true };
  }

  function viewAt(nowMs: number, spans: number[] = []): LiveView | null {
    if (last === null || target === null || from === null) return null;

    const p = progressAt(nowMs);

    // A conceded point keeps the ball visible while it leaves the court, and
    // hides it once it has arrived at the parked position.
    const hidden = parked(target) && (!conceded || p >= 1);
    const here = hidden ? target.ball : ballBack(p, 0);
    const trail: Array<{ x: number; y: number }> = [];

    if (!hidden) {
      for (const back of spans) {
        const point = ballBack(p, back);
        if (!point.ok) break;
        trail.push({ x: point.x, y: point.y });
      }
    }

    // Paddles: the human's is the worker's word, exactly. The model's is eased
    // over the first slice of its own step, because the engine moves it in one
    // jump at the start of a tick.
    const leftY = humanSide === 'left' ? last.leftY : easedAt(leftEase, nowMs);
    const rightY = humanSide === 'right' ? last.rightY : easedAt(rightEase, nowMs);

    return {
      ballX: here.x,
      ballY: here.y,
      leftY,
      rightY,
      trail,
      score: [last.score[0], last.score[1]],
      ballAlpha: serving && !hidden ? clamp01(p / 0.25) : 1,
      ballHidden: hidden,
      latencyMs: last.latencyMs,
      tick: last.tick,
      status: last.status,
      settled: p >= 1,
    };
  }

  return {
    push,
    viewAt,
    latest: () => last,
    intervalMs: () => ballInterval,
  };
}
