/**
 * The one entry point the rest of the app uses to turn a DecisionState into a Move.
 *
 * It reads the lane's route from the lane configuration (lib/config/lanes.ts): `kind: 'systemone'` goes to a System One
 * server (./systemone.ts), `kind: 'chat'` to a chat model's structured output (./chat.ts). It puts a hard timeout
 * around the call and reports the latency that call measured. Nothing else is added to that number.
 *
 * Everything that can go wrong comes back as a typed DecideResponse: callers never see an exception, a stack, or a
 * provider message that might carry credentials.
 */
import { APICallError } from 'ai';
import { loadLanes } from '../config/lanes';
import { LOCAL_MODEL_IDS, type LaneConfig } from '../game/types';
import type { DecideErrorCode, DecideRequest, DecideResponse } from '../game/types';
import { decideWithChat } from './chat';
import { decideWithSystemOne, SystemOneHttpError } from './systemone';

export { expectedMove, stateForModel, DECISION_INSTRUCTIONS, MOVE_CRITERIA } from './prompt';
export { decideWithSystemOne } from './systemone';
export { decideWithChat, parseMove } from './chat';
export type { ModelDecision } from './types';

export const DEFAULT_TIMEOUT_MS = 20_000;

export interface DecideOptions {
  signal?: AbortSignal;
  /** Hard ceiling on one model call. Default 20000ms. */
  timeoutMs?: number;
  /** The lanes to look the model up in. Default: the configured lanes. */
  lanes?: LaneConfig[];
}

type DecideFailure = Extract<DecideResponse, { ok: false }>;

function fail(error: DecideErrorCode, message: string): DecideFailure {
  return { ok: false, error, message };
}

export async function decide(req: DecideRequest, opts: DecideOptions = {}): Promise<DecideResponse> {
  const { model, state } = req;
  const lanes = opts.lanes ?? loadLanes();
  const lane = lanes.find((l) => l.id === model);
  if (!lane) {
    const local = (LOCAL_MODEL_IDS as readonly string[]).includes(model);
    return fail('bad_request', local ? `"${model}" is decided locally and has no model call.` : `Unknown lane "${safeText(String(model))}".`);
  }

  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const controller = new AbortController();
  let timedOut = false;

  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const forwardAbort = () => controller.abort();
  if (opts.signal?.aborted) controller.abort();
  else opts.signal?.addEventListener('abort', forwardAbort, { once: true });

  try {
    const decision =
      lane.kind === 'systemone'
        ? await decideWithSystemOne(lane, state, { signal: controller.signal })
        : await decideWithChat(lane, state, { signal: controller.signal });

    return { ok: true, move: decision.move, latencyMs: Math.round(decision.latencyMs), model, trace: decision.trace };
  } catch (err) {
    if (timedOut) return fail('model_error', `Timed out after ${timeoutMs}ms.`);
    if (opts.signal?.aborted) return fail('model_error', 'Cancelled.');
    return mapDecideError(err);
  } finally {
    clearTimeout(timer);
    opts.signal?.removeEventListener('abort', forwardAbort);
  }
}

/**
 * Turns anything thrown by the SDK or the fetch into a typed failure. 429 is a rate limit; anything else is a model
 * error with a one-line reason.
 */
export function mapDecideError(err: unknown): DecideFailure {
  const status = statusCodeOf(err);

  if (status === 429) {
    return fail('rate_limited', 'The server is rate limiting this lane.');
  }
  return fail('model_error', safeText(messageOf(err)));
}

function statusCodeOf(err: unknown): number | undefined {
  if (err instanceof SystemOneHttpError) return err.statusCode;
  if (APICallError.isInstance(err)) return err.statusCode;
  const status = (err as { statusCode?: unknown } | null)?.statusCode;
  return typeof status === 'number' ? status : undefined;
}

function messageOf(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  return typeof err === 'string' && err ? err : 'The model call failed.';
}

/** One line, no stack, no key-shaped strings, and short enough for a lane label. */
function safeText(text: string): string {
  const oneLine = text.replace(/\s+/g, ' ').trim();
  const redacted = oneLine.replace(/\b(?:sk|vck|eyJ)[A-Za-z0-9._-]{12,}/g, '[redacted]');
  if (!redacted) return 'The model call failed.';
  return redacted.length > 200 ? `${redacted.slice(0, 197)}...` : redacted;
}
