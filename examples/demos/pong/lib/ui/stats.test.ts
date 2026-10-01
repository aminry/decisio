import { describe, expect, it } from 'vitest';
import type { Replay, Snapshot } from '../game/types';
import { headlineSentences, laneStatsFor } from './stats';

function snap(t: number, latencyMs: number | null, vx = 1): Snapshot {
  return {
    t,
    ball: { x: 0, y: 0, vx, vy: 0 },
    paddle: { y: 0 },
    score: [0, 0],
    status: 'playing',
    latencyMs,
  } as unknown as Snapshot;
}

function replay(): Replay {
  return {
    version: 1,
    recordedAt: '2026-10-01T00:00:00.000Z',
    seed: 1,
    lanes: [
      { model: 'sys1', label: 'Decisio', provider: 'System One server', snapshots: [snap(0, null), snap(100, 100), snap(200, 120), snap(13_000, 140)] },
      { model: 'cmp', label: 'Other', provider: 'chat', snapshots: [snap(0, null), snap(3000, 3000), snap(6000, 3200)] },
    ],
  };
}

describe('laneStatsFor', () => {
  it('has one row per lane, in the recording order', () => {
    expect(laneStatsFor(replay()).map((row) => row.model)).toEqual(['sys1', 'cmp']);
  });

  it('carries the provider the recorder wrote, never a literal', () => {
    expect(laneStatsFor(replay())[1]?.provider).toBe('chat');
  });

  it('counts decisions inside the first window only', () => {
    const rows = laneStatsFor(replay());
    expect(rows[0]?.inWindow).toBe(2);
    expect(rows[1]?.inWindow).toBe(2);
  });
});

describe('headlineSentences', () => {
  it('quotes the measurement of the first lane and the range of the rest', () => {
    const line = headlineSentences(laneStatsFor(replay()));
    expect(line.lead).toBe('Decisio returns a decision in');
    expect(line.leadMs).toBe('120 ms');
    expect(line.others).toBe('The others take 3.1 to 3.1 seconds.');
  });

  it('has no em dash and no hype', () => {
    const all = Object.values(headlineSentences(laneStatsFor(replay()))).join(' ');
    expect(all).not.toMatch(/\u2014/);
  });

  it('is empty with no lanes', () => {
    expect(headlineSentences([])).toEqual({ lead: '', leadMs: '', others: '' });
  });
});
