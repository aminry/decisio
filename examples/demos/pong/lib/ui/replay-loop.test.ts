import { describe, expect, it } from 'vitest';
import type { Replay, Snapshot } from '../game/types';
import { LOOP_HOLD_MS, replayPeriodMs } from './replay-loop';

function frame(t: number): Snapshot {
  return {
    t,
    tick: 0,
    seed: 1,
    ball: { x: 80, y: 50, vx: 1, vy: 0 },
    leftY: 50,
    rightY: 50,
    score: [0, 0],
    model: 'jev',
    latencyMs: null,
    move: null,
    status: 'playing',
  };
}

function replay(ends: number[]): Replay {
  return {
    version: 1,
    recordedAt: '2026-09-17T20:09:00.815Z',
    seed: 1,
    lanes: ends.map((end) => ({
      model: 'jev' as const,
      label: 'Jev',
      snapshots: [frame(0), frame(end)],
    })),
  };
}

describe('replayPeriodMs', () => {
  it('is the longest lane plus the hold, so all four restart together', () => {
    expect(replayPeriodMs(replay([43836, 45002, 44222]))).toBe(45002 + LOOP_HOLD_MS);
  });

  it('has nothing to say about nothing', () => {
    expect(replayPeriodMs(null)).toBeNull();
    expect(replayPeriodMs(replay([]))).toBeNull();
    expect(replayPeriodMs({ ...replay([0]), lanes: [{ model: 'jev', label: 'Jev', snapshots: [] }] })).toBeNull();
  });

  it('takes the hold as given', () => {
    expect(replayPeriodMs(replay([1000]), 0)).toBe(1000);
  });
});
