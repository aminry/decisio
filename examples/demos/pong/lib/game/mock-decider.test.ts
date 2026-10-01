import { describe, expect, it, vi } from 'vitest';
import { createMockDecider } from './mock-decider';
import { createEngine, serve, toDecisionState } from './engine';
import { MODEL_PADDLE_STEP, type DecisionState } from './types';

function state(paddleY: number, interceptY: number | null): DecisionState {
  const base = toDecisionState(serve(createEngine(1)), 'right');
  return { ...base, paddle: { ...base.paddle, y: paddleY }, interceptY };
}

describe('createMockDecider', () => {
  it('plays the correct move toward the intercept', async () => {
    const d = createMockDecider({ latencyMs: 0 });
    expect((await d.decide(state(50, 20))).ok && (await d.decide(state(50, 20)))).toBeTruthy();
    await expect(d.decide(state(50, 20)).then((r) => (r.ok ? r.move : r.error))).resolves.toBe('up');
    await expect(d.decide(state(50, 80)).then((r) => (r.ok ? r.move : r.error))).resolves.toBe('down');
    await expect(
      d.decide(state(50, 50 + MODEL_PADDLE_STEP / 4)).then((r) => (r.ok ? r.move : r.error)),
    ).resolves.toBe('stay');
    await expect(d.decide(state(50, null)).then((r) => (r.ok ? r.move : r.error))).resolves.toBe('stay');
  });

  it('reports the model and a latency', async () => {
    const res = await createMockDecider({ latencyMs: 7 }).decide(state(50, 20));
    expect(res.ok).toBe(true);
    if (res.ok) {
      expect(res.model).toBe('mock');
      expect(res.latencyMs).toBe(7);
    }
  });

  it('waits roughly the configured latency', async () => {
    const t0 = Date.now();
    await createMockDecider({ latencyMs: 30 }).decide(state(50, 20));
    expect(Date.now() - t0).toBeGreaterThanOrEqual(25);
  });

  it('draws a latency inside the given range, deterministically per seed', async () => {
    const run = async () => {
      const d = createMockDecider({ latencyMs: [5, 9], seed: 77 });
      const out: number[] = [];
      for (let i = 0; i < 5; i += 1) {
        const r = await d.decide(state(50, 20));
        if (r.ok) out.push(r.latencyMs);
      }
      return out;
    };
    const a = await run();
    expect(a).toEqual(await run());
    for (const ms of a) {
      expect(ms).toBeGreaterThanOrEqual(5);
      expect(ms).toBeLessThanOrEqual(9);
    }
    expect(new Set(a).size).toBeGreaterThan(1); // it really is jittering
  });

  it('is wrong about (1 - accuracy) of the time, deterministically', async () => {
    const run = async () => {
      const d = createMockDecider({ latencyMs: 0, accuracy: 0.5, seed: 4 });
      const moves: string[] = [];
      for (let i = 0; i < 200; i += 1) {
        const r = await d.decide(state(50, 20)); // correct answer is always 'up'
        if (r.ok) moves.push(r.move);
      }
      return moves;
    };
    const moves = await run();
    expect(moves).toEqual(await run());
    const wrong = moves.filter((m) => m !== 'up').length;
    expect(wrong).toBeGreaterThan(60);
    expect(wrong).toBeLessThan(140);
  });

  it('never errs at accuracy 1', async () => {
    const d = createMockDecider({ latencyMs: 0, accuracy: 1, seed: 4 });
    for (let i = 0; i < 50; i += 1) {
      const r = await d.decide(state(50, 20));
      expect(r.ok && r.move).toBe('up');
    }
  });

  it('runs out of credits after failAfter decisions', async () => {
    const d = createMockDecider({ latencyMs: 0, failAfter: 3 });
    for (let i = 0; i < 3; i += 1) expect((await d.decide(state(50, 20))).ok).toBe(true);
    const res = await d.decide(state(50, 20));
    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.error).toBe('out_of_credits');
      expect(res.message).toMatch(/credit/i);
    }
  });

  it('honours an AbortSignal, before and during the wait', async () => {
    const d = createMockDecider({ latencyMs: 500 });
    const pre = new AbortController();
    pre.abort();
    await expect(d.decide(state(50, 20), pre.signal)).rejects.toThrow(/abort/i);

    const mid = new AbortController();
    const pending = d.decide(state(50, 20), mid.signal);
    mid.abort();
    await expect(pending).rejects.toThrow(/abort/i);
  });

  it('clears its timer when aborted', async () => {
    vi.useFakeTimers();
    try {
      const d = createMockDecider({ latencyMs: 10_000 });
      const ctrl = new AbortController();
      const pending = d.decide(state(50, 20), ctrl.signal).catch((e: Error) => e.name);
      ctrl.abort();
      await expect(pending).resolves.toBe('AbortError');
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });
});
