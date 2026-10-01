import { afterEach, describe, expect, it, vi } from 'vitest';
import { buildReplay, createRecorder, createReplayRunner } from './replay';
import { createLaneRunner } from './runner';
import { createMockDecider } from './mock-decider';
import type { LaneRunner, ModelId, Replay, Snapshot } from './types';

function snap(t: number, tick: number, over: Partial<Snapshot> = {}): Snapshot {
  return {
    t,
    tick,
    seed: 1,
    ball: { x: 10 + tick, y: 50, vx: 26.7, vy: 0 },
    leftY: 50,
    rightY: 50,
    score: [0, 0],
    model: 'mock',
    latencyMs: 100,
    move: 'stay',
    status: 'playing',
    ...over,
  };
}

function fixture(model: ModelId = 'mock'): Replay {
  return buildReplay(
    [{ model, label: 'Mock', snapshots: [snap(0, 0), snap(100, 1), snap(250, 2), snap(450, 3)] }],
    1,
  );
}

describe('buildReplay', () => {
  it('builds a v1 replay and copies the snapshot arrays', () => {
    const snapshots = [snap(0, 0)];
    const r = buildReplay([{ model: 'jev', label: 'Jev', snapshots }], 42);
    expect(r.version).toBe(1);
    expect(r.seed).toBe(42);
    expect(Number.isNaN(Date.parse(r.recordedAt))).toBe(false);
    expect(r.lanes).toHaveLength(1);
    expect(r.lanes[0]).toEqual({ model: 'jev', label: 'Jev', snapshots: [snap(0, 0)] });
    snapshots.push(snap(1, 1));
    expect(r.lanes[0].snapshots).toHaveLength(1); // defensive copy
  });

  it('round-trips through JSON', () => {
    const r = fixture();
    expect(JSON.parse(JSON.stringify(r))).toEqual(r);
  });
});

describe('createReplayRunner', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('rejects a lane index that is not there', () => {
    expect(() => createReplayRunner(fixture(), 3)).toThrow(RangeError);
  });

  it('exposes the recorded model', () => {
    expect(createReplayRunner(fixture('gemini'), 0).model).toBe('gemini');
  });

  it('emits snapshots honouring the recorded deltas', () => {
    vi.useFakeTimers();
    const runner = createReplayRunner(fixture(), 0);
    const seen: Snapshot[] = [];
    runner.on('snapshot', (s) => seen.push(s));
    runner.start();

    expect(seen).toHaveLength(0); // nothing before the clock moves
    vi.advanceTimersByTime(0);
    expect(seen.map((s) => s.tick)).toEqual([0]);
    vi.advanceTimersByTime(99);
    expect(seen).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(seen.map((s) => s.tick)).toEqual([0, 1]);
    vi.advanceTimersByTime(150);
    expect(seen.map((s) => s.tick)).toEqual([0, 1, 2]);
    vi.advanceTimersByTime(200);
    expect(seen.map((s) => s.tick)).toEqual([0, 1, 2, 3]);
    expect(seen.map((s) => s.t)).toEqual([0, 100, 250, 450]);
    expect(runner.latest()?.tick).toBe(3);
  });

  it('divides the deltas by speed', () => {
    vi.useFakeTimers();
    const runner = createReplayRunner(fixture(), 0, { speed: 4 });
    const seen: Snapshot[] = [];
    runner.on('snapshot', (s) => seen.push(s));
    runner.start();
    vi.advanceTimersByTime(0);
    expect(seen).toHaveLength(1);
    vi.advanceTimersByTime(25); // 100 / 4
    expect(seen).toHaveLength(2);
    vi.advanceTimersByTime(38); // 250 / 4 = 62.5 elapsed
    expect(seen).toHaveLength(3);
    vi.advanceTimersByTime(50); // 450 / 4 = 112.5 elapsed
    expect(seen).toHaveLength(4);
    expect(seen.map((s) => s.t)).toEqual([0, 100, 250, 450]); // recorded t is untouched
  });

  it('emits end once the recording runs out, and stops', () => {
    vi.useFakeTimers();
    const runner = createReplayRunner(fixture(), 0);
    const seen: Snapshot[] = [];
    const ended: Snapshot[] = [];
    runner.on('snapshot', (s) => seen.push(s));
    runner.on('end', (s) => ended.push(s));
    runner.start();
    vi.advanceTimersByTime(1000);

    expect(seen).toHaveLength(4);
    expect(ended).toHaveLength(1);
    expect(ended[0].tick).toBe(3);
    vi.advanceTimersByTime(5000);
    expect(seen).toHaveLength(4);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('loops with t re-based so the clock never rewinds', () => {
    vi.useFakeTimers();
    const runner = createReplayRunner(fixture(), 0, { loop: true });
    const seen: Snapshot[] = [];
    const ended: Snapshot[] = [];
    runner.on('snapshot', (s) => seen.push(s));
    runner.on('end', (s) => ended.push(s));
    runner.start();
    vi.advanceTimersByTime(1000);

    expect(seen.length).toBeGreaterThan(4);
    expect(ended).toHaveLength(0); // a loop never ends
    expect(seen.slice(0, 4).map((s) => s.tick)).toEqual([0, 1, 2, 3]);
    expect(seen[4].tick).toBe(0); // back to the start
    for (let i = 1; i < seen.length; i += 1) {
      expect(seen[i].t).toBeGreaterThanOrEqual(seen[i - 1].t);
    }
    expect(seen[4].t).toBe(450); // re-based by the span of one pass
    runner.stop();
  });

  it('stop() clears the timers and silences the runner', () => {
    vi.useFakeTimers();
    const runner = createReplayRunner(fixture(), 0, { loop: true });
    const seen: Snapshot[] = [];
    runner.on('snapshot', (s) => seen.push(s));
    runner.start();
    vi.advanceTimersByTime(120);
    const count = seen.length;
    expect(count).toBeGreaterThan(0);
    runner.stop();
    expect(vi.getTimerCount()).toBe(0);
    vi.advanceTimersByTime(5000);
    expect(seen).toHaveLength(count);
  });

  it('handles an empty lane without hanging', () => {
    vi.useFakeTimers();
    const empty = buildReplay([{ model: 'mock', label: 'Empty', snapshots: [] }], 1);
    const runner = createReplayRunner(empty, 0);
    const seen: Snapshot[] = [];
    runner.on('snapshot', (s) => seen.push(s));
    runner.start();
    vi.advanceTimersByTime(1000);
    expect(seen).toHaveLength(0);
    expect(runner.latest()).toBeNull();
  });

  it('ignores left input', () => {
    const runner: LaneRunner = createReplayRunner(fixture(), 0);
    expect(() => runner.setLeftInput('up')).not.toThrow();
    runner.stop();
  });

  it('on() returns a working unsubscribe', () => {
    vi.useFakeTimers();
    const runner = createReplayRunner(fixture(), 0);
    const seen: Snapshot[] = [];
    const off = runner.on('snapshot', (s) => seen.push(s));
    runner.start();
    vi.advanceTimersByTime(0);
    off();
    vi.advanceTimersByTime(1000);
    expect(seen).toHaveLength(1);
  });
});

describe('createRecorder', () => {
  it('collects what a runner emits and stops collecting on stop()', async () => {
    const runner = createLaneRunner({
      model: 'mock',
      decider: createMockDecider({ latencyMs: 0, accuracy: 0, seed: 3 }),
      seed: 11,
      leftInput: 'auto',
    });
    const recorder = createRecorder(runner);
    const done = new Promise<void>((resolve) => runner.on('end', () => resolve()));
    runner.start();
    await done;

    const snapshots = recorder.stop();
    expect(snapshots.length).toBeGreaterThan(5);
    expect(snapshots[0].status).toBe('serving');
    expect(snapshots[snapshots.length - 1].status).toBe('over');
    expect(snapshots.map((s) => s.tick)).toEqual(snapshots.map((s) => s.tick).slice().sort((a, b) => a - b));
    expect(recorder.snapshots()).toHaveLength(snapshots.length);
  });

  it('records a live lane and replays it frame for frame', async () => {
    const runner = createLaneRunner({
      model: 'mock',
      decider: createMockDecider({ latencyMs: 1, accuracy: 0, seed: 3 }),
      seed: 11,
      leftInput: 'auto',
    });
    const recorder = createRecorder(runner);
    const done = new Promise<void>((resolve) => runner.on('end', () => resolve()));
    runner.start();
    await done;
    const recorded = recorder.stop();

    const replay = buildReplay([{ model: 'mock', label: 'Mock', snapshots: recorded }], 11);
    const player = createReplayRunner(replay, 0, { speed: 1000 });
    const played: Snapshot[] = [];
    player.on('snapshot', (s) => played.push(s));
    const finished = new Promise<Snapshot>((resolve) => player.on('end', resolve));
    player.start();
    const final = await finished;

    expect(played).toEqual(recorded);
    expect(final).toEqual(recorded[recorded.length - 1]);
  });
});
