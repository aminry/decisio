/**
 * Pong lanes — lane runner. One lane, one clock, one model.
 *
 * The loop is deliberately serial: ask the decider, wait however long it takes,
 * apply exactly one tick, emit. That is the whole trick of the demo — the ball
 * moves at the speed of the model.
 *
 * Works in the browser and in Node: no DOM, timing via performance.now() with a
 * Date.now() fallback.
 */

import { applyTick, serve, createEngine, toDecisionState } from './engine';
import type {
  Decider,
  EngineState,
  LaneRunner,
  LaneRunnerEvents,
  ModelId,
  Move,
  Snapshot,
} from './types';

/** Monotonic-ish clock that works in both runtimes. */
export function now(): number {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

export interface LaneRunnerOptions {
  model: ModelId;
  decider: Decider;
  seed: number;
  /** 'auto' = the scripted wall. 'manual' = whatever setLeftInput last stored. */
  leftInput: 'auto' | 'manual';
  /** Stagger lanes so they do not all fire their first request at once. */
  startDelayMs?: number;
  /** Ball direction of the first serve. Defaults to +1 (toward the model). */
  serveDir?: -1 | 1;
  /**
   * Transient decider failures (model_error, rate_limited, a thrown fetch) do
   * not end the lane: the tick is played as 'stay' and the ball still advances.
   * This many failures in a row end it with status 'error'. out_of_credits
   * always ends at once. Mirrors the game worker.
   */
  maxConsecutiveFailures?: number;
  /**
   * Called once per tick with the complete engine state after it and, when the model was asked, the request as sent,
   * the whole answer and the latency: the run's trajectory (added for this repository). Also called on every serve.
   */
  onTick?: (tick: TickRecord) => void;
}

/** One trajectory tick (examples/demos/tools/trajectory.py): the state after the tick, and the decision behind it. */
export interface TickRecord {
  kind: 'serve' | 'tick';
  tick: number;
  /** Unix seconds when the tick was applied. */
  t_wall: number;
  state: EngineState;
  decision:
    | null
    | { request: unknown; answer: unknown; chosen: Move; latency_ms: number; call_ms: number }
    | { error: string; message: string; chosen: Move; latency_ms: number };
}

export const MAX_CONSECUTIVE_FAILURES = 3;

type Listeners = {
  [E in keyof LaneRunnerEvents]: Set<LaneRunnerEvents[E]>;
};

export function createLaneRunner(opts: LaneRunnerOptions): LaneRunner {
  const {
    model,
    decider,
    seed,
    leftInput,
    startDelayMs = 0,
    serveDir = 1,
    maxConsecutiveFailures = MAX_CONSECUTIVE_FAILURES,
    onTick,
  } = opts;
  const record = (kind: TickRecord['kind'], decision: TickRecord['decision']) =>
    onTick?.({ kind, tick: state.tick, t_wall: Date.now() / 1000, state: structuredClone(state), decision });

  const listeners: Listeners = { snapshot: new Set(), end: new Set() };
  let state: EngineState = createEngine(seed, serveDir);
  let last: Snapshot | null = null;
  let started = false;
  let stopped = false;
  let startedAt = 0;
  let pendingLeft: Move = 'stay';
  let delayTimer: ReturnType<typeof setTimeout> | null = null;
  const abort = new AbortController();

  function snapshot(over: Partial<Snapshot> = {}): Snapshot {
    return {
      t: Math.max(0, now() - startedAt),
      tick: state.tick,
      seed: state.seed,
      ball: { ...state.ball },
      leftY: state.leftY,
      rightY: state.rightY,
      score: [state.score[0], state.score[1]],
      model,
      latencyMs: null,
      move: null,
      status: state.status,
      ...over,
    };
  }

  function emit(s: Snapshot): void {
    if (stopped) return;
    last = s;
    for (const cb of listeners.snapshot) cb(s);
  }

  function end(final: Snapshot): void {
    for (const cb of listeners.end) cb(final);
    stopped = true;
  }

  function consumeManualMove(): Move {
    const m = pendingLeft;
    pendingLeft = 'stay';
    return m;
  }

  async function loop(): Promise<void> {
    let failures = 0;
    while (!stopped) {
      const decision = toDecisionState(state, 'right');
      const t0 = now();
      let res;
      try {
        res = await decider.decide(decision, abort.signal);
      } catch (err) {
        if (stopped) return;
        res = {
          ok: false as const,
          error: 'model_error' as const,
          message: err instanceof Error ? err.message : String(err),
        };
      }
      if (stopped) return;
      const latencyMs = now() - t0;

      let move: Move;
      let message: string | undefined;
      if (res.ok) {
        failures = 0;
        move = res.move;
      } else {
        failures += 1;
        const fatal = res.error === 'out_of_credits' || failures >= maxConsecutiveFailures;
        if (fatal) {
          const failed = snapshot({
            latencyMs,
            status: res.error === 'out_of_credits' ? 'credits' : 'error',
            message: res.message,
          });
          emit(failed);
          end(failed);
          return;
        }
        // Hold the paddle for this tick; the ball still moves one segment.
        move = 'stay';
        message = `held: ${res.error}`;
      }

      state = applyTick(state, move, leftInput === 'auto' ? 'auto' : consumeManualMove());
      const s = snapshot(message === undefined ? { latencyMs, move } : { latencyMs, move, message });
      emit(s);
      if (onTick) {
        const response = (res.ok ? res.trace?.response : undefined) as { answers?: unknown; model?: unknown } | undefined;
        record(
          'tick',
          res.ok
            ? {
                request: res.trace?.request ?? null,
                answer: response?.answers ?? null,
                chosen: move,
                latency_ms: latencyMs,
                call_ms: res.latencyMs,
              }
            : { error: res.error, message: res.message, chosen: move, latency_ms: latencyMs },
        );
      }

      if (state.status === 'over') {
        end(s);
        return;
      }
      // A conceded point re-serves straight away, so the next decision sees a
      // live ball rather than a parked one.
      if (state.status === 'point') {
        state = serve(state);
        record('serve', null);
      }
    }
  }

  return {
    model,

    start(): void {
      if (started || stopped) return;
      started = true;
      startedAt = now();
      state = serve(state);
      // Status 'serving' (not 'playing') so the UI can render the court before
      // the first decision has come back.
      emit(snapshot({ status: 'serving' }));
      record('serve', null);
      if (startDelayMs > 0) {
        delayTimer = setTimeout(() => {
          delayTimer = null;
          if (!stopped) void loop();
        }, startDelayMs);
      } else {
        void loop();
      }
    },

    stop(): void {
      if (stopped) return;
      stopped = true;
      if (delayTimer !== null) {
        clearTimeout(delayTimer);
        delayTimer = null;
      }
      abort.abort();
    },

    setLeftInput(move: Move): void {
      pendingLeft = move;
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
