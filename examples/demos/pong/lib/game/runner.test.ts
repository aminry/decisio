import { describe, expect, it } from 'vitest';
import { createLaneRunner } from './runner';
import { createMockDecider } from './mock-decider';
import { CENTRE_Y } from './engine';
import { MODEL_PADDLE_STEP, type DecideResponse, type Decider, type LaneRunner, type Snapshot } from './types';

/** Resolve once the runner has emitted `n` snapshots. */
function collect(runner: LaneRunner, n: number, timeoutMs = 4000): Promise<Snapshot[]> {
  return new Promise((resolve, reject) => {
    const out: Snapshot[] = [];
    const timer = setTimeout(() => {
      off();
      reject(new Error(`only ${out.length}/${n} snapshots before timeout`));
    }, timeoutMs);
    const off = runner.on('snapshot', (s) => {
      out.push(s);
      if (out.length >= n) {
        clearTimeout(timer);
        off();
        resolve(out);
      }
    });
    runner.start();
  });
}

/** Resolve on the runner's 'end' event. */
function untilEnd(runner: LaneRunner, timeoutMs = 8000): Promise<Snapshot> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('no end event before timeout')), timeoutMs);
    runner.on('end', (s) => {
      clearTimeout(timer);
      resolve(s);
    });
    runner.start();
  });
}

const strip = (s: Snapshot) => ({ ...s, t: 0, latencyMs: 0 });

describe('createLaneRunner', () => {
  it('emits a serving snapshot synchronously on start', () => {
    const runner = createLaneRunner({
      model: 'mock',
      decider: createMockDecider({ latencyMs: 1000 }),
      seed: 5,
      leftInput: 'auto',
    });
    const seen: Snapshot[] = [];
    runner.on('snapshot', (s) => seen.push(s));
    expect(runner.latest()).toBeNull();
    runner.start();

    expect(seen).toHaveLength(1);
    const first = seen[0];
    expect(first.status).toBe('serving');
    expect(first.latencyMs).toBeNull();
    expect(first.move).toBeNull();
    expect(first.tick).toBe(0);
    expect(first.seed).toBe(5);
    expect(first.model).toBe('mock');
    expect(first.score).toEqual([0, 0]);
    expect(first.leftY).toBe(CENTRE_Y);
    expect(first.ball.vx).not.toBe(0); // already serving, so the UI can draw it
    expect(runner.latest()).toEqual(first);
    runner.stop();
  });

  it('advances one tick per decision and measures latency', async () => {
    const runner = createLaneRunner({
      model: 'mock',
      decider: createMockDecider({ latencyMs: 20 }),
      seed: 5,
      leftInput: 'auto',
    });
    const snaps = await collect(runner, 4);
    runner.stop();

    expect(snaps[0].status).toBe('serving');
    expect(snaps.slice(1).map((s) => s.tick)).toEqual([1, 2, 3]);
    for (const s of snaps.slice(1)) {
      expect(s.status).toBe('playing');
      expect(s.move).not.toBeNull();
      expect(s.latencyMs).not.toBeNull();
      expect(s.latencyMs as number).toBeGreaterThanOrEqual(15);
      expect(s.latencyMs as number).toBeLessThan(400);
    }
    // Lane-local clock advances with the decisions.
    expect(snaps[3].t).toBeGreaterThan(snaps[1].t);
  });

  it('a slower model produces a slower ball (the whole point of the demo)', async () => {
    const fast = createLaneRunner({
      model: 'mock',
      decider: createMockDecider({ latencyMs: 2 }),
      seed: 5,
      leftInput: 'auto',
    });
    const slow = createLaneRunner({
      model: 'mock',
      decider: createMockDecider({ latencyMs: 40 }),
      seed: 5,
      leftInput: 'auto',
    });
    const [f, s] = await Promise.all([collect(fast, 4), collect(slow, 4)]);
    fast.stop();
    slow.stop();
    // Same tick, same ball position — but the slow lane took much longer to get there.
    expect(f[3].ball).toEqual(s[3].ball);
    expect(s[3].t).toBeGreaterThan(f[3].t * 3);
  });

  it('is deterministic for a given seed and decider', async () => {
    const run = async () => {
      const runner = createLaneRunner({
        model: 'mock',
        decider: createMockDecider({ latencyMs: 0, accuracy: 0.7, seed: 99 }),
        seed: 2026,
        leftInput: 'auto',
      });
      const snaps = await collect(runner, 40);
      runner.stop();
      return snaps.map(strip);
    };
    expect(await run()).toEqual(await run());
  });

  it('serves again after a point and keeps playing', async () => {
    const runner = createLaneRunner({
      model: 'mock',
      decider: createMockDecider({ latencyMs: 0, accuracy: 0, seed: 3 }), // always wrong, concedes
      seed: 11,
      leftInput: 'auto',
    });
    const seen: Snapshot[] = [];
    runner.on('snapshot', (x) => seen.push(x));
    await untilEnd(runner);
    const snaps = seen;
    const pointAt = snaps.findIndex((s) => s.status === 'point');
    expect(pointAt).toBeGreaterThan(0);
    expect(snaps[pointAt].score[0]).toBe(1);
    // The rally resumes: the next snapshot is a live tick, not another point.
    expect(snaps[pointAt + 1].status).toBe('playing');
    expect(snaps[pointAt + 1].ball.vx).not.toBe(0);
    expect(snaps[pointAt + 1].tick).toBe(snaps[pointAt].tick + 1);
  });

  it("ends the lane at 'over' and stops emitting", async () => {
    const runner = createLaneRunner({
      model: 'mock',
      decider: createMockDecider({ latencyMs: 0, accuracy: 0, seed: 3 }),
      seed: 11,
      leftInput: 'auto',
    });
    const seen: Snapshot[] = [];
    runner.on('snapshot', (s) => seen.push(s));
    const final = await untilEnd(runner);

    expect(final.status).toBe('over');
    expect(final.score[0]).toBe(5);
    expect(seen[seen.length - 1]).toEqual(final);
    const count = seen.length;
    await new Promise((r) => setTimeout(r, 30));
    expect(seen).toHaveLength(count);
  });

  it("maps out_of_credits to status 'credits' and ends", async () => {
    const runner = createLaneRunner({
      model: 'jev',
      decider: createMockDecider({ latencyMs: 0, failAfter: 3 }),
      seed: 8,
      leftInput: 'auto',
    });
    const seen: Snapshot[] = [];
    runner.on('snapshot', (s) => seen.push(s));
    const final = await untilEnd(runner);

    expect(final.status).toBe('credits');
    expect(final.message).toMatch(/credit/i);
    expect(final.model).toBe('jev');
    expect(seen.filter((s) => s.status === 'playing')).toHaveLength(3);
    expect(seen[seen.length - 1]).toEqual(final);
  });

  it("maps any other failure to status 'error' with the message", async () => {
    const decider: Decider = {
      async decide(): Promise<DecideResponse> {
        return { ok: false, error: 'rate_limited', message: 'slow down' };
      },
    };
    const runner = createLaneRunner({ model: 'gpt', decider, seed: 8, leftInput: 'auto' });
    const final = await untilEnd(runner);
    expect(final.status).toBe('error');
    expect(final.message).toBe('slow down');
  });

  it("turns a thrown decider into status 'error'", async () => {
    const decider: Decider = {
      async decide(): Promise<DecideResponse> {
        throw new Error('network exploded');
      },
    };
    const runner = createLaneRunner({ model: 'gpt', decider, seed: 8, leftInput: 'auto' });
    const final = await untilEnd(runner);
    expect(final.status).toBe('error');
    expect(final.message).toBe('network exploded');
  });

  it('stop() aborts the in-flight decision and emits nothing further', async () => {
    let seenSignal: AbortSignal | undefined;
    const decider: Decider = {
      decide(_state, signal) {
        seenSignal = signal;
        return new Promise((resolve, reject) => {
          signal?.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
          setTimeout(() => resolve({ ok: true, move: 'stay', latencyMs: 0, model: 'mock' }), 5000);
        });
      },
    };
    const runner = createLaneRunner({ model: 'mock', decider, seed: 1, leftInput: 'auto' });
    const seen: Snapshot[] = [];
    const ended: Snapshot[] = [];
    runner.on('snapshot', (s) => seen.push(s));
    runner.on('end', (s) => ended.push(s));
    runner.start();

    await new Promise((r) => setTimeout(r, 10));
    expect(seenSignal?.aborted).toBe(false);
    runner.stop();
    expect(seenSignal?.aborted).toBe(true);

    await new Promise((r) => setTimeout(r, 40));
    expect(seen).toHaveLength(1); // only the initial serving snapshot
    expect(ended).toHaveLength(0);
    runner.stop(); // idempotent
  });

  it('start() is idempotent and start() after stop() does nothing', async () => {
    const runner = createLaneRunner({
      model: 'mock',
      decider: createMockDecider({ latencyMs: 5 }),
      seed: 1,
      leftInput: 'auto',
    });
    const seen: Snapshot[] = [];
    runner.on('snapshot', (s) => seen.push(s));
    runner.start();
    runner.start();
    expect(seen).toHaveLength(1);
    runner.stop();
    runner.start();
    await new Promise((r) => setTimeout(r, 30));
    expect(seen).toHaveLength(1);
  });

  it('honours startDelayMs before the first decision', async () => {
    const runner = createLaneRunner({
      model: 'mock',
      decider: createMockDecider({ latencyMs: 0 }),
      seed: 1,
      leftInput: 'auto',
      startDelayMs: 120,
    });
    const seen: Snapshot[] = [];
    runner.on('snapshot', (s) => seen.push(s));
    runner.start();
    await new Promise((r) => setTimeout(r, 40));
    expect(seen).toHaveLength(1);
    await new Promise((r) => setTimeout(r, 150));
    expect(seen.length).toBeGreaterThan(1);
    runner.stop();
  });

  it('consumes a manual left move once per tick, then defaults to stay', async () => {
    const runner = createLaneRunner({
      model: 'mock',
      decider: createMockDecider({ latencyMs: 1 }),
      seed: 6,
      leftInput: 'manual',
    });
    const pending = collect(runner, 3);
    runner.setLeftInput('up');
    const snaps = await pending;
    runner.stop();
    expect(snaps[1].leftY).toBe(CENTRE_Y - MODEL_PADDLE_STEP); // consumed
    expect(snaps[2].leftY).toBe(CENTRE_Y - MODEL_PADDLE_STEP); // not repeated
  });

  it('ignores left input in auto mode', async () => {
    const runner = createLaneRunner({
      model: 'mock',
      decider: createMockDecider({ latencyMs: 1 }),
      seed: 6,
      leftInput: 'auto',
    });
    runner.setLeftInput('down');
    const snaps = await collect(runner, 3);
    runner.stop();
    // The wall tracks the ball; it does not take the queued 'down'.
    expect(snaps[2].leftY).not.toBe(CENTRE_Y + MODEL_PADDLE_STEP * 2);
  });

  it('on() returns a working unsubscribe', async () => {
    const runner = createLaneRunner({
      model: 'mock',
      decider: createMockDecider({ latencyMs: 1 }),
      seed: 6,
      leftInput: 'auto',
    });
    const seen: Snapshot[] = [];
    const off = runner.on('snapshot', (s) => seen.push(s));
    runner.start();
    expect(seen).toHaveLength(1);
    off();
    await new Promise((r) => setTimeout(r, 30));
    expect(seen).toHaveLength(1);
    expect(runner.latest()?.tick).toBeGreaterThan(0); // still running, just unsubscribed
    runner.stop();
  });
  describe('transient decider failures', () => {
    function flakyDecider(failures: number, error: 'model_error' | 'rate_limited' = 'model_error'): Decider {
      let calls = 0;
      return {
        async decide(): Promise<DecideResponse> {
          calls += 1;
          if (calls <= failures) return { ok: false, error, message: `flake ${calls}` };
          return { ok: true, move: 'stay', latencyMs: 0, model: 'jev' };
        },
      };
    }

    it('holds the paddle and keeps the ball moving through two failures, then recovers', async () => {
      const runner = createLaneRunner({ model: 'jev', decider: flakyDecider(2), seed: 8, leftInput: 'auto' });
      const seen: Snapshot[] = [];
      runner.on('snapshot', (s) => seen.push(s));
      runner.start();
      await new Promise<void>((resolve) => {
        const check = () => {
          if (seen.filter((s) => s.tick > 0).length >= 6) resolve();
          else setTimeout(check, 0);
        };
        check();
      });
      runner.stop();

      const ticks = seen.filter((s) => s.tick > 0);
      expect(ticks.every((s) => s.status !== 'error' && s.status !== 'credits')).toBe(true);
      // The two held ticks are marked, the ball still advanced one tick each.
      expect(ticks[0].message).toBe('held: model_error');
      expect(ticks[1].message).toBe('held: model_error');
      expect(ticks[2].message).toBeUndefined();
      for (let i = 1; i < ticks.length; i += 1) expect(ticks[i].tick).toBe(ticks[i - 1].tick + 1);
    });

    it('ends with status error on the third consecutive failure, carrying the last message', async () => {
      const runner = createLaneRunner({ model: 'gpt', decider: flakyDecider(99, 'rate_limited'), seed: 8, leftInput: 'auto' });
      const seen: Snapshot[] = [];
      runner.on('snapshot', (s) => seen.push(s));
      const final = await untilEnd(runner);

      expect(final.status).toBe('error');
      expect(final.message).toBe('flake 3');
      expect(seen.filter((s) => s.status === 'playing' && s.tick > 0)).toHaveLength(2);
    });

    it('resets the failure count after a success', async () => {
      let calls = 0;
      const decider: Decider = {
        async decide(): Promise<DecideResponse> {
          calls += 1;
          // fail, fail, ok, fail, fail, ok ... never three in a row
          if (calls % 3 !== 0) return { ok: false, error: 'model_error', message: 'flake' };
          return { ok: true, move: 'stay', latencyMs: 0, model: 'jev' };
        },
      };
      const runner = createLaneRunner({ model: 'jev', decider, seed: 8, leftInput: 'auto' });
      const seen: Snapshot[] = [];
      runner.on('snapshot', (s) => seen.push(s));
      runner.start();
      await new Promise<void>((resolve) => {
        const check = () => (seen.filter((s) => s.tick > 0).length >= 9 ? resolve() : setTimeout(check, 0));
        check();
      });
      runner.stop();
      expect(seen.some((s) => s.status === 'error')).toBe(false);
    });

    it('honours a custom maxConsecutiveFailures', async () => {
      const runner = createLaneRunner({
        model: 'jev',
        decider: flakyDecider(99),
        seed: 8,
        leftInput: 'auto',
        maxConsecutiveFailures: 1,
      });
      const final = await untilEnd(runner);
      expect(final.status).toBe('error');
      expect(final.message).toBe('flake 1');
    });
  });
});
