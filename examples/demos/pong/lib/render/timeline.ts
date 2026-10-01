/**
 * lib/render/timeline.ts — a recorded lane, sampled at any instant.
 *
 * A replay stores one snapshot per model decision. That is the truth, but it
 * is not a video: between two snapshots there is nothing. This module fills
 * the gap so the clip can be sampled at 30 fps.
 *
 * The rule that makes the clip honest (motion note 01):
 *
 *   - At t = 0 the scripted serve tweens the ball ONE STEP out of the centre
 *     over 300 ms ease-out, in every lane at once. The ball then waits, because
 *     the model has not decided yet — which is the whole demonstration.
 *   - Every later step is a LINEAR tween whose duration is exactly that
 *     decision's recorded latency, arriving as the decision lands. Nothing is
 *     eased inside a step, so on-screen ball speed IS decision speed.
 *
 * Everything here is pure. Give it a lane and a time, get a view back.
 */

import {
  COURT_H,
  COURT_W,
  SEGMENT_X,
  type LaneStatus,
  type ModelId,
  type Replay,
  type Snapshot,
} from '../game/types';
import type { CourtView } from './court';
import { pathAt, type Ball } from './path';
import { isMiss, isReturn, laneStats, type LaneStats } from '../record/stats';

// Re-exported so the end card and the tests keep one import path.
export { laneStats, type LaneStats };
import { CLIP_TOKENS, type ClipTokens } from './tokens';

export type ReplayLane = Replay['lanes'][number];

/** What one lane looks like at one instant. */
export interface LaneView extends CourtView {
  /** The latency of the most recent decision that has landed. */
  latencyMs: number | null;
  tick: number;
  decisions: number;
  returns: number;
  misses: number;
  score: [number, number];
  status: LaneStatus;
}

/** The latency number, motion note 04. */
export interface LaneNumber {
  /** max(last landed latency, time since the last snapshot). Null before either. */
  value: number | null;
  /** True while the count-up has overtaken the last landed reading. */
  inFlight: boolean;
  /** 0..1 size pulse — 1 at the peak of the flare. */
  pulse: number;
  /** 0..1 blend toward pure white, for the 120 ms flash on landing. */
  flash: number;
}


export interface LaneTimeline {
  model: ModelId;
  label: string;
  /** Time of the last snapshot. Sampling past it freezes the lane. */
  durationMs: number;
  /**
   * The lane at `tMs`. `trailSpans` are distances behind the ball in COURT
   * units, nearest first — build them with `trailSpans()` from ./court.
   */
  viewAt(tMs: number, trailSpans?: number[]): LaneView;
  /** The latency number and its flare. */
  numberAt(tMs: number): LaneNumber;
  /** Counts for the window [fromMs, toMs]. What the end card wants. */
  statsIn(fromMs: number, toMs: number): LaneStats;
  /** Counts as of `tMs`. */
  statsAt(tMs: number): LaneStats;
}

function easeOutCubic(p: number): number {
  return 1 - Math.pow(1 - p, 3);
}

function clamp01(p: number): number {
  return p < 0 ? 0 : p > 1 ? 1 : p;
}

/** A parked ball means the point is over and nothing is in play. */
function parked(s: Snapshot): boolean {
  return s.ball.vx === 0;
}

/** A rewind: a looping replay, or a fresh runner behind the same lane. */
function isRestart(prev: Snapshot, next: Snapshot): boolean {
  return next.tick < prev.tick || (next.status === 'serving' && prev.status !== 'serving');
}

const TERMINAL: ReadonlySet<LaneStatus> = new Set<LaneStatus>(['over', 'credits', 'error']);

/** One interval between two snapshots, pre-solved. */
interface Step {
  startMs: number;
  endMs: number;
  from: Ball;
  to: Snapshot;
  /** Simulation error at p = 1, blended out linearly so we land exactly. */
  errX: number;
  errY: number;
  /** The ball sets off from the centre line: 300 ms ease-out, then it waits. */
  serve: boolean;
  /** No continuous path from the previous snapshot: hold and cut. */
  cut: boolean;
  /** The point was conceded during this step; the ball leaves the court. */
  conceded: boolean;
  /** Index of the first step of this rally. The trail never reaches past it. */
  rallyFrom: number;
}

export function createLaneTimeline(
  lane: ReplayLane,
  tokens: ClipTokens = CLIP_TOKENS,
): LaneTimeline {
  const frames = lane.snapshots;
  const n = frames.length;

  // --- running totals, one entry per snapshot ------------------------------
  const returnsAt: number[] = new Array(n).fill(0);
  const missesAt: number[] = new Array(n).fill(0);
  const decisionsAt: number[] = new Array(n).fill(0);
  const latencyAt: Array<number | null> = new Array(n).fill(null);
  for (let i = 0; i < n; i += 1) {
    const prev = i > 0 ? frames[i - 1] : null;
    const landed = frames[i].latencyMs != null;
    returnsAt[i] = (prev ? returnsAt[i - 1] : 0) + (prev && isReturn(prev, frames[i]) ? 1 : 0);
    missesAt[i] = (prev ? missesAt[i - 1] : 0) + (prev && isMiss(prev, frames[i]) ? 1 : 0);
    decisionsAt[i] = (prev ? decisionsAt[i - 1] : 0) + (landed ? 1 : 0);
    latencyAt[i] = frames[i].latencyMs ?? (i > 0 ? latencyAt[i - 1] : null);
  }

  // --- one pre-solved step per interval ------------------------------------
  const steps: Step[] = [];
  let rallyFrom = 0;
  for (let i = 0; i + 1 < n; i += 1) {
    const prev = frames[i];
    const next = frames[i + 1];
    const restart = isRestart(prev, next);
    // The opening serve and every serve after a point both start from rest at
    // the centre line, so both get the scripted 300 ms push-off.
    const serve = i === 0 || restart || parked(prev);
    const conceded = !parked(prev) && parked(next);
    const cut = parked(prev) && parked(next);

    if (serve || restart || cut) rallyFrom = steps.length;

    const from: Ball =
      parked(prev) || restart
        ? { x: COURT_W / 2, y: COURT_H / 2, vx: next.ball.vx, vy: next.ball.vy }
        : { ...prev.ball };

    // Where our own path simulation thinks the segment ends. Blend the
    // difference out across the interval so the ball lands EXACTLY on the
    // engine's position at the boundary — that is what makes viewAt continuous
    // across snapshots.
    let errX = 0;
    let errY = 0;
    if (!cut && !conceded && from.vx !== 0) {
      const end = pathAt(from, next.leftY, next.rightY, SEGMENT_X);
      errX = next.ball.x - end.x;
      errY = next.ball.y - end.y;
    }

    steps.push({
      startMs: prev.t,
      endMs: next.t,
      from,
      to: next,
      errX,
      errY,
      serve: serve && !cut,
      cut,
      conceded,
      rallyFrom,
    });
  }

  const durationMs = n > 0 ? frames[n - 1].t : 0;

  /** Index of the last snapshot that has landed at or before `tMs`. */
  function indexAt(tMs: number): number {
    if (n === 0) return -1;
    if (tMs <= frames[0].t) return 0;
    if (tMs >= frames[n - 1].t) return n - 1;
    let lo = 0;
    let hi = n - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (frames[mid].t <= tMs) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  /** Index of the step covering `tMs`, or -1 when there is nothing to walk. */
  function stepAt(tMs: number): number {
    if (steps.length === 0) return -1;
    return Math.min(indexAt(tMs), steps.length - 1);
  }

  /**
   * How far into its step the ball is, 0..1 of one SEGMENT_X.
   *
   * A serve is the only eased motion in the whole clip, and it finishes long
   * before the decision it belongs to lands — the ball then waits, which is
   * the point being made.
   */
  function progressOf(step: Step, tMs: number): number {
    if (step.cut) return 1;
    const span = step.endMs - step.startMs;
    if (step.serve) {
      const serveMs = Math.min(tokens.motion.serveMs, span > 0 ? span : tokens.motion.serveMs);
      return easeOutCubic(clamp01(serveMs <= 0 ? 1 : (tMs - step.startMs) / serveMs));
    }
    return span <= 0 ? 1 : clamp01((tMs - step.startMs) / span);
  }

  /**
   * Ball position `back` court-x-units before where it is now.
   *
   * `clamped` means the walk hit the start of the rally: there is no more path
   * behind it, so the trail stops there rather than piling discs on one point.
   */
  function ballBack(
    si: number,
    p: number,
    back: number,
  ): { x: number; y: number; clamped: boolean } {
    let i = si;
    let travelled = SEGMENT_X * p;
    let remaining = back;
    let clamped = false;
    while (remaining > travelled) {
      if (i <= steps[i].rallyFrom) {
        clamped = true;
        break;
      }
      remaining -= travelled;
      i -= 1;
      travelled = SEGMENT_X;
    }
    const dist = Math.max(0, travelled - remaining);
    const step = steps[i];
    const q = SEGMENT_X === 0 ? 0 : dist / SEGMENT_X;
    const sim = pathAt(step.from, step.to.leftY, step.to.rightY, dist);
    return { x: sim.x + step.errX * q, y: sim.y + step.errY * q, clamped };
  }

  function viewAt(tMs: number, spans: number[] = []): LaneView {
    if (n === 0) {
      return {
        ballX: COURT_W / 2,
        ballY: COURT_H / 2,
        leftY: COURT_H / 2,
        rightY: COURT_H / 2,
        trail: [],
        score: [0, 0],
        ballHidden: true,
        latencyMs: null,
        tick: 0,
        decisions: 0,
        returns: 0,
        misses: 0,
        status: 'idle',
      };
    }

    const i = indexAt(tMs);
    const landed = frames[i];
    const si = stepAt(tMs);
    const step = si >= 0 ? steps[si] : null;

    let leftY = landed.leftY;
    let rightY = landed.rightY;
    let ballX = landed.ball.x;
    let ballY = landed.ball.y;
    let hidden = parked(landed);
    let alpha = 1;
    let trail: Array<{ x: number; y: number }> = [];

    if (step) {
      const p = progressOf(step, tMs);
      // Paddles settle over the first slice of the interval and then hold:
      // the engine moves them at the START of a tick.
      const span = step.endMs - step.startMs;
      const linear = span <= 0 ? 1 : clamp01((tMs - step.startMs) / span);
      const prev = frames[si];
      const ease = easeOutCubic(clamp01(linear / tokens.motion.paddleSettle));
      leftY = prev.leftY + (step.to.leftY - prev.leftY) * ease;
      rightY = prev.rightY + (step.to.rightY - prev.rightY) * ease;

      if (step.cut) {
        ballX = step.to.ball.x;
        ballY = step.to.ball.y;
        hidden = parked(step.to);
      } else {
        const here = ballBack(si, p, 0);
        ballX = here.x;
        ballY = here.y;
        hidden = false;
        if (step.serve) alpha = clamp01(p / 0.25);
        for (const back of spans) {
          const point = ballBack(si, p, back);
          if (point.clamped) break;
          trail.push({ x: point.x, y: point.y });
        }
      }
    }

    if (hidden) trail = [];

    return {
      ballX,
      ballY,
      leftY,
      rightY,
      trail,
      score: [landed.score[0], landed.score[1]],
      ballAlpha: alpha,
      ballHidden: hidden,
      latencyMs: latencyAt[i],
      tick: landed.tick,
      decisions: decisionsAt[i],
      returns: returnsAt[i],
      misses: missesAt[i],
      status: landed.status,
    };
  }

  function numberAt(tMs: number): LaneNumber {
    if (n === 0) return { value: null, inFlight: false, pulse: 0, flash: 0 };

    const i = indexAt(tMs);
    const landed = frames[i];
    const last = latencyAt[i];
    const age = Math.max(0, tMs - landed.t);
    // A lane that has stopped has nothing in flight; its number holds.
    const stopped = TERMINAL.has(landed.status) || (i === n - 1 && tMs > durationMs);
    const elapsed = stopped ? 0 : age;
    const floor = last ?? 0;
    const value = last === null && elapsed === 0 ? null : Math.max(floor, elapsed);

    const m = tokens.motion;
    const flared = landed.latencyMs != null;
    const pulse =
      flared && age < m.pulseMs
        ? age < m.pulseOutMs
          ? age / m.pulseOutMs
          : 1 - (age - m.pulseOutMs) / (m.pulseMs - m.pulseOutMs)
        : 0;
    const flash = flared && age < m.flashMs ? 1 - age / m.flashMs : 0;

    return { value, inFlight: elapsed > floor, pulse: clamp01(pulse), flash: clamp01(flash) };
  }

  function statsIn(fromMs: number, toMs: number): LaneStats {
    if (n === 0) {
      return {
        model: lane.model,
        label: lane.label,
        decisions: 0,
        decisionsPerSecond: 0,
        avgMs: 0,
        p50Ms: 0,
        p95Ms: 0,
        returns: 0,
        misses: 0,
        score: [0, 0],
        durationMs: 0,
      };
    }
    // Snapshots that LANDED inside the window. The first one is the state the
    // window opened with, so counting starts from the one after it.
    const hi = indexAt(toMs);
    const lo = Math.min(hi, indexAt(fromMs));
    const stats = laneStats(lane.model, lane.label, frames.slice(lo, hi + 1));
    return { ...stats, durationMs: Math.max(0, frames[hi].t - frames[lo].t) };
  }

  function statsAt(tMs: number): LaneStats {
    return statsIn(Number.NEGATIVE_INFINITY, tMs);
  }

  return { model: lane.model, label: lane.label, durationMs, viewAt, numberAt, statsIn, statsAt };
}

/** Every lane of a replay, plus the length of the longest one. */
export function createReplayTimeline(
  replay: Replay,
  tokens: ClipTokens = CLIP_TOKENS,
): { lanes: LaneTimeline[]; durationMs: number } {
  const lanes = replay.lanes.map((lane) => createLaneTimeline(lane, tokens));
  const durationMs = lanes.reduce((max, l) => Math.max(max, l.durationMs), 0);
  return { lanes, durationMs };
}

