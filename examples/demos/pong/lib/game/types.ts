/**
 * Pong lanes — shared contract.
 *
 * Every module (engine, decide API, UI) builds against this file.
 * Change it only by agreement; everything else may depend on it.
 *
 * Core mechanic ("ball steps per decision"):
 *   Each lane has its own clock. One model decision = one tick. On every tick the
 *   ball advances exactly one SEGMENT along its velocity (bouncing off top/bottom),
 *   and the model's paddle moves by at most MODEL_PADDLE_STEP in the chosen
 *   direction. A slow model therefore means a slow ball. Nothing is skipped.
 *
 * Court: 160 x 100 units. Origin top-left. y grows downward. "up" DEcreases y.
 * The paddles do NOT stand on the court edge: each one is PADDLE_INSET in from
 * it, and the ball turns at that paddle's inner face (LEFT_PLANE / RIGHT_PLANE),
 * which is where the paddle is drawn. Reflecting at the edge instead is what put
 * the ball through the paddle on screen.
 * Left paddle is a scripted, zero-latency perfect returner ("the wall"); in play
 * mode it is the human. Right paddle is the model.
 */

export const COURT_W = 160;
export const COURT_H = 100;
export const PADDLE_H = 20;
export const BALL_R = 1.5;
/** How far a paddle's inner face stands in from the court edge. */
export const PADDLE_INSET = 4;
/** The x plane the ball turns at on each side: the paddle's inner face. */
export const LEFT_PLANE = PADDLE_INSET;
export const RIGHT_PLANE = COURT_W - PADDLE_INSET;
/** Plane to plane. This — not COURT_W — is what the ball actually crosses. */
export const CROSSING_X = RIGHT_PLANE - LEFT_PLANE;
/** Decisions the ball takes to cross from one paddle to the other. */
export const SEGMENTS_PER_CROSSING = 8;
/** Ball travels this far along x per decision. */
export const SEGMENT_X = CROSSING_X / SEGMENTS_PER_CROSSING;
/**
 * Max model paddle movement per DECISION. 8 decisions per crossing x 12 = 96
 * units of reach against a paddle range of 80, so a model that answers on every
 * tick can always get there (see the reach budget in lib/game/engine.ts).
 */
export const MODEL_PADDLE_STEP = 12;
/**
 * A human paddle moves in COURT UNITS PER SECOND, continuously: the worker
 * integrates the held direction over real elapsed time, on input arrival and on
 * its own paddle tick, so a 300ms press moves the paddle 24 units whether the
 * tick fell in the middle of it or not. 80 units/s covers the whole 80-unit
 * paddle range in one second, while the ball needs at least
 * 8 x PLAY_MIN_STEP_MS = 2.1s to reach them.
 */
export const HUMAN_PADDLE_SPEED = 80;
export const POINTS_TO_WIN = 5;

/* --------------------------------------------------------- live game pacing */
/**
 * These four are the worker's, but they live here because a browser has to know
 * them too: what "get ready" means, and how long silence may last before a lane
 * is stalled rather than still. The worker is the only thing that owns a clock.
 */

/**
 * In a mode a human plays, a ball step never completes faster than this, even
 * when the model answers sooner. A model at ~200ms would otherwise cross the court
 * in 1.6s, which is not a game, it is a reflex test. The published `latencyMs`
 * is still the real model latency — this paces the ball, it does not fake a
 * number. Demo mode is unpaced: there is nobody to be fair to.
 *
 * 260ms: a crossing takes 2.1s, and the human paddle covers the court in 1s.
 * (350 was the first guess; it played slow.)
 */
export const PLAY_MIN_STEP_MS = 260;
/** Wait this long after the players are present before the first serve. */
export const SERVE_DELAY_MS = 2500;
/** And this long after a point before the next serve. */
export const POINT_SERVE_DELAY_MS = 1200;
/** A player's latest input expires this long after it arrived. */
export const INPUT_TTL_MS = 400;
/** A frame goes out at least this often, so silence always means "stalled". */
export const HEARTBEAT_MS = 1000;

export type Move = 'up' | 'down' | 'stay';

/** A lane id from the lane configuration (lib/config/lanes.ts), or one of the local deciders below. */
export type ModelId = string;

/** Deciders that never reach a model: the scripted wall, a person, and the test mock. */
export const LOCAL_MODEL_IDS = ['human', 'wall', 'mock'] as const;

/**
 * How a lane asks its model. `systemone`: a typed choice question over POST <baseUrl>/v1/systemone, no text generated.
 * `chat`: the same words as a system prompt to an OpenAI-compatible chat server, answered as structured output.
 */
export type DecideKind = 'systemone' | 'chat' | 'local';

export interface LaneConfig {
  id: ModelId;
  /** Display name, e.g. "Decisio", "Qwen3 0.6B (chat)". */
  label: string;
  /** Who serves it, as the lane's tag reads. Null for local deciders. */
  provider: string | null;
  kind: DecideKind;
  /** Accent colour for the lane (CSS colour). */
  color?: string;
  /** Server root: `<baseUrl>/v1/systemone` for `systemone`, `<baseUrl>/chat/completions` for `chat`. */
  baseUrl?: string;
  /** Sent with the request when the server wants a model name (always, for `chat`). */
  model?: string;
  /** Optional key for a server that wants one; a local server needs none. */
  apiKey?: string;
  /** Extra JSON fields for a `chat` request body, e.g. {"chat_template_kwargs": {"enable_thinking": false}}. */
  extraBody?: Record<string, unknown>;
}

/** Numbers only, about 125 bytes of JSON. This is exactly what the model sees. */
export interface DecisionState {
  court: { w: number; h: number };
  ball: { x: number; y: number; vx: number; vy: number };
  paddle: { y: number; h: number };
  /**
   * Predicted y (centre) where the ball will cross the paddle's x plane,
   * after wall bounces. Null when the ball is moving away from the paddle.
   */
  interceptY: number | null;
  /** 'toward' when the ball is heading at this paddle. */
  dir: 'toward' | 'away';
}

export type DecideMode = 'demo' | 'play';

export interface DecideRequest {
  model: ModelId;
  state: DecisionState;
  mode: DecideMode;
}

export type DecideErrorCode =
  | 'out_of_credits'
  | 'rate_limited'
  | 'model_error'
  | 'model_not_allowed'
  | 'bad_request';

export type DecideResponse =
  | { ok: true; move: Move; latencyMs: number; model: ModelId }
  | { ok: false; error: DecideErrorCode; message: string };

/** Anything that turns a DecisionState into a Move. Server route, mock, human input. */
export interface Decider {
  decide(state: DecisionState, signal?: AbortSignal): Promise<DecideResponse>;
}

export type LaneStatus =
  | 'idle'
  | 'serving'
  | 'playing'
  | 'point'
  | 'over'
  | 'credits'
  | 'error';

/**
 * Full lane state after a tick. Small enough to publish at every tick
 * (a fast model ~10/s) and to store thousands of in a replay file.
 */
export interface Snapshot {
  /** ms since the lane started (lane-local clock). */
  t: number;
  /** Decision counter, starts at 0 on serve. */
  tick: number;
  seed: number;
  ball: { x: number; y: number; vx: number; vy: number };
  leftY: number;
  rightY: number;
  /** [left, right] */
  score: [number, number];
  model: ModelId;
  /** Latency of the decision that produced this tick. Null on serve/reset. */
  latencyMs: number | null;
  move: Move | null;
  status: LaneStatus;
  /**
   * Human-readable note. On 'error'/'credits' the failure text; on 'over' the
   * end reason (worker); on a 'playing' tick that a transient failure held,
   * 'held: <error>' (runner). Never contains secrets.
   */
  message?: string;
  /**
   * Per human side, the `seq` of the newest input the worker had applied when
   * this frame was taken. A browser predicts its own paddle and only pulls it
   * back onto the wire once the wire has heard its latest input — before that
   * the wire position is simply older than the player's hand, and reconciling
   * against it drags the paddle backwards (lib/ui/paddle.ts).
   */
  inputSeq?: Partial<Record<'left' | 'right', number>>;
}

/** Pure engine state (no timing). Engine functions are deterministic given seed. */
export interface EngineState {
  seed: number;
  rng: number; // PRNG cursor (mulberry32 state)
  tick: number;
  ball: { x: number; y: number; vx: number; vy: number };
  leftY: number;
  rightY: number;
  score: [number, number];
  status: LaneStatus;
  /** Who is serving next: -1 left, +1 right. */
  serveDir: -1 | 1;
}

export interface LaneRunnerEvents {
  snapshot: (s: Snapshot) => void;
  end: (final: Snapshot) => void;
}

/**
 * Drives one lane: ask the decider, apply the move, advance the ball, emit.
 * Implemented in lib/game/runner.ts (used by the recorder and the replay hero).
 * Live games are driven by lib/worker/game-worker.ts, which uses the engine directly.
 */
export interface LaneRunner {
  readonly model: ModelId;
  start(): void;
  stop(): void;
  /** Left paddle input for play mode; ignored in demo mode. */
  setLeftInput(move: Move): void;
  on<E extends keyof LaneRunnerEvents>(event: E, cb: LaneRunnerEvents[E]): () => void;
  latest(): Snapshot | null;
}

/** Recorded run, replayed on "/" as the hero. Timing is preserved via Snapshot.t. */
export interface Replay {
  version: 1;
  recordedAt: string;
  seed: number;
  lanes: Array<{
    model: ModelId;
    label: string;
    /** Who served the lane, its accent and how it was asked: written by the recorder, read by the page. */
    provider?: string | null;
    color?: string;
    kind?: DecideKind;
    snapshots: Snapshot[];
  }>;
}

/**
 * The lanes a recording runs when nothing configures others: the System One server on this machine. More lanes (other
 * servers, a chat model) come from the lane configuration, `PONG_LANES` or `PONG_LANES_FILE` (lib/config/lanes.ts).
 * Lanes keep the order they are given and are never re-sorted.
 */
export const LANES: LaneConfig[] = [
  { id: 'sys1', label: 'Decisio', provider: 'System One server', kind: 'systemone', color: '#FF5416', baseUrl: 'http://127.0.0.1:8100' },
];
