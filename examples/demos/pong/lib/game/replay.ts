/**
 * Pong lanes — recording and replaying lanes.
 *
 * A replay is just the snapshots a live lane emitted. Playing one back is a
 * LaneRunner too, so the UI renders a recorded run and a live run with exactly
 * the same component. Timing comes from Snapshot.t, so a 4-second rally and
 * a 40-second GPT rally replay at their real relative speeds.
 */

import type { LaneRunner, LaneRunnerEvents, Replay, Snapshot } from './types';

export interface ReplayRunnerOptions {
  /** Playback rate multiplier. 2 = twice as fast. Default 1. */
  speed?: number;
  /** Restart from the first snapshot when the recording ends. Default false. */
  loop?: boolean;
}

type Listeners = {
  [E in keyof LaneRunnerEvents]: Set<LaneRunnerEvents[E]>;
};

export function createReplayRunner(
  replay: Replay,
  laneIndex: number,
  opts: ReplayRunnerOptions = {},
): LaneRunner {
  const lane = replay.lanes[laneIndex];
  if (!lane) throw new RangeError(`replay has no lane at index ${laneIndex}`);
  const speed = opts.speed && opts.speed > 0 ? opts.speed : 1;
  const loop = opts.loop ?? false;
  const frames = lane.snapshots;

  const listeners: Listeners = { snapshot: new Set(), end: new Set() };
  let last: Snapshot | null = null;
  let started = false;
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let index = 0;
  /** Offset added to every emitted t so it stays monotonic across loops. */
  let baseT = 0;
  const span = frames.length > 0 ? frames[frames.length - 1].t - frames[0].t : 0;

  function emit(s: Snapshot): void {
    if (stopped) return;
    last = s;
    for (const cb of listeners.snapshot) cb(s);
  }

  function schedule(ms: number, fn: () => void): void {
    timer = setTimeout(() => {
      timer = null;
      if (!stopped) fn();
    }, Math.max(0, ms));
  }

  function step(): void {
    const frame = frames[index];
    const s: Snapshot = { ...frame, t: baseT + frame.t };
    emit(s);
    index += 1;

    if (index >= frames.length) {
      if (!loop) {
        for (const cb of listeners.end) cb(s);
        stopped = true;
        return;
      }
      baseT += span;
      index = 0;
      schedule(frames[0].t / speed, step);
      return;
    }
    schedule((frames[index].t - frame.t) / speed, step);
  }

  return {
    model: lane.model,

    start(): void {
      if (started || stopped) return;
      started = true;
      if (frames.length === 0) {
        stopped = true;
        return;
      }
      schedule(frames[0].t / speed, step);
    },

    stop(): void {
      if (stopped) return;
      stopped = true;
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
    },

    setLeftInput(): void {
      // A replay ignores input by design.
    },

    on<E extends keyof LaneRunnerEvents>(event: E, cb: LaneRunnerEvents[E]): () => void {
      listeners[event].add(cb);
      return () => {
        listeners[event].delete(cb);
      };
    },

    latest(): Snapshot | null {
      return last;
    },
  };
}

/** Collect everything a runner emits, so a live run can be saved as a replay. */
export function createRecorder(runner: LaneRunner): {
  snapshots(): Snapshot[];
  stop(): Snapshot[];
} {
  const collected: Snapshot[] = [];
  let off: (() => void) | null = runner.on('snapshot', (s) => {
    collected.push(s);
  });
  return {
    snapshots: () => collected.slice(),
    stop: () => {
      off?.();
      off = null;
      return collected.slice();
    },
  };
}

export function buildReplay(lanes: Replay['lanes'], seed: number): Replay {
  return {
    version: 1,
    recordedAt: new Date().toISOString(),
    seed,
    lanes: lanes.map((l) => ({ ...l, snapshots: l.snapshots.slice() })),
  };
}
