import { describe, expect, it } from 'vitest';
import { HEARTBEAT_MS } from '../game/types';
import { FLASH_MS, STALL_MS, laneNumberAt, type LaneNumberState } from './lane-number';

function state(over: Partial<LaneNumberState> = {}): LaneNumberState {
  return {
    lastLatencyMs: 300,
    landedAt: 1_000,
    landedKey: 4,
    stopped: false,
    live: true,
    ...over,
  };
}

describe('laneNumberAt', () => {
  it('is an em dash until something has happened', () => {
    expect(laneNumberAt(state({ lastLatencyMs: null, landedAt: 0, landedKey: 0 }), 5_000).value).toBe(
      null,
    );
  });

  it('holds the landed reading until the wait overtakes it', () => {
    const frame = laneNumberAt(state(), 1_200);
    expect(frame.value).toBe(300);
    expect(frame.inFlight).toBe(false);
  });

  it('counts up past the last reading while a decision is out', () => {
    const frame = laneNumberAt(state(), 1_650);
    expect(frame.value).toBe(650);
    expect(frame.inFlight).toBe(true);
    expect(frame.waiting).toBe(false);
  });

  it('flashes for the first FLASH_MS after a decision lands', () => {
    expect(laneNumberAt(state(), 1_000 + FLASH_MS - 1).flashing).toBe(true);
    expect(laneNumberAt(state(), 1_000 + FLASH_MS + 1).flashing).toBe(false);
  });

  it('stops counting after two missed heartbeats on a live lane', () => {
    expect(STALL_MS).toBe(HEARTBEAT_MS * 2);

    const justBefore = laneNumberAt(state(), 1_000 + STALL_MS - 10);
    expect(justBefore.waiting).toBe(false);
    expect(justBefore.value).toBe(STALL_MS - 10);

    const stalled = laneNumberAt(state(), 1_000 + 30_000);
    expect(stalled.waiting).toBe(true);
    expect(stalled.value).toBe(STALL_MS);
    expect(stalled.inFlight).toBe(true);
  });

  it('keeps the reading it had when a slow model stalls, rather than shrinking it', () => {
    // GPT answering in 6s is a real number; the stall must not erase it.
    const stalled = laneNumberAt(state({ lastLatencyMs: 6_000 }), 1_000 + 30_000);
    expect(stalled.waiting).toBe(true);
    expect(stalled.value).toBe(6_000);
  });

  it('never stalls a recording, however long the decision took', () => {
    const recorded = laneNumberAt(state({ live: false }), 1_000 + 9_000);
    expect(recorded.waiting).toBe(false);
    expect(recorded.value).toBe(9_000);
    expect(recorded.inFlight).toBe(true);
  });

  it('holds still once the lane has stopped', () => {
    const over = laneNumberAt(state({ stopped: true }), 1_000 + 60_000);
    expect(over.value).toBe(300);
    expect(over.inFlight).toBe(false);
    expect(over.waiting).toBe(false);
  });
});
