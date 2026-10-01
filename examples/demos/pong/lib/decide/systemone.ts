/**
 * A System One lane: one `POST <baseUrl>/v1/systemone` per decision (replaces upstream's hosted-API call).
 *
 * The model is handed the state as data and answers the typed question from ./prompt.ts, so there is no prose to write
 * and nothing to parse back. The chat lane in ./chat.ts is given the identical state and the identical words as
 * structured output instead. The clock starts immediately before the request and stops once the answer is parsed, so
 * `latencyMs` is the model call (the HTTP round trip included) and nothing else. No retries: a retry inside the call
 * would be charged to this decision's latency. One call, one tick.
 */
import type { DecisionState, LaneConfig } from '../game/types';
import { MOVE_QUESTIONS, isMove, stateForModel } from './prompt';
import type { ModelDecision } from './types';

export interface SystemOneOptions {
  signal?: AbortSignal;
  /** Test seam: a fetch implementation instead of the global one. */
  fetchImpl?: typeof fetch;
}

/** A non-200 answer from the server, with its status so the caller can tell a rate limit from a fault. */
export class SystemOneHttpError extends Error {
  constructor(
    readonly statusCode: number,
    message: string,
  ) {
    super(message);
    this.name = 'SystemOneHttpError';
  }
}

export async function decideWithSystemOne(
  lane: LaneConfig,
  state: DecisionState,
  opts: SystemOneOptions = {},
): Promise<ModelDecision> {
  if (!lane.baseUrl) throw new Error(`Lane "${lane.id}" has no baseUrl.`);
  const doFetch = opts.fetchImpl ?? fetch;
  const body: Record<string, unknown> = { state: stateForModel(state), questions: MOVE_QUESTIONS };
  if (lane.model) body.model = lane.model;
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (lane.apiKey) headers.authorization = `Bearer ${lane.apiKey}`;

  const startedAt = performance.now();
  const response = await doFetch(`${lane.baseUrl}/v1/systemone`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal: opts.signal,
  });
  if (!response.ok) {
    throw new SystemOneHttpError(response.status, `System One server answered ${response.status}.`);
  }
  const answer = (await response.json()) as { answers?: { move?: { choice?: unknown } } };
  const latencyMs = performance.now() - startedAt;

  const choice = answer.answers?.move?.choice;
  if (!isMove(choice)) throw new Error(`The server returned an unknown choice: ${String(choice)}`);
  return { move: choice, latencyMs };
}
