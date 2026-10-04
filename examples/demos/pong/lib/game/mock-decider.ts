/**
 * Pong lanes — mock decider. A fake model with a dial for latency, accuracy and
 * running out of credits. Used by the UI in dev and by the engine tests, so a
 * lane can be driven with no network and no API key.
 */

import { MODEL_PADDLE_STEP } from './types';
import { moveToward, nextRandom } from './engine';
import type { DecideResponse, Decider, DecisionState, Move } from './types';

const MOVES: Move[] = ['up', 'down', 'stay'];

export interface MockDeciderOptions {
  /** Fixed latency, or [min, max] drawn from the seeded PRNG. */
  latencyMs: number | [number, number];
  /** 0..1. Probability the decider plays the correct move. Default 1. */
  accuracy?: number;
  /** PRNG seed for latency jitter and wrong moves. Default 1. */
  seed?: number;
  /** After this many successful decisions, fail with 'out_of_credits'. */
  failAfter?: number;
}

function abortError(): Error {
  const err = new Error('The operation was aborted.');
  err.name = 'AbortError';
  return err;
}

function delay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError());
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(abortError());
    };
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

export function createMockDecider(opts: MockDeciderOptions): Decider {
  const { latencyMs, accuracy = 1, seed = 1, failAfter } = opts;
  let rng = seed | 0;
  let calls = 0;

  const draw = (): number => {
    const next = nextRandom(rng);
    rng = next.state;
    return next.value;
  };

  return {
    async decide(state: DecisionState, signal?: AbortSignal): Promise<DecideResponse> {
      const wait = Array.isArray(latencyMs)
        ? latencyMs[0] + draw() * (latencyMs[1] - latencyMs[0])
        : latencyMs;

      await delay(wait, signal);

      if (failAfter !== undefined && calls >= failAfter) {
        return { ok: false, error: 'out_of_credits', message: 'Mock decider ran out of credits.' };
      }
      calls += 1;

      const correct = moveToward(state.paddle.y, state.interceptY, MODEL_PADDLE_STEP);
      let move = correct;
      if (accuracy < 1 && draw() >= accuracy) {
        const wrong = MOVES.filter((m) => m !== correct);
        move = wrong[Math.min(wrong.length - 1, Math.floor(draw() * wrong.length))];
      }

      return { ok: true, move, latencyMs: wait, model: 'mock' };
    },
  };
}
