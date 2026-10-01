/**
 * The chat lane: the same decision, put to a language model on an OpenAI-compatible server (a local one).
 *
 * It gets the identical DecisionState and the identical instruction text as the System One lane (./prompt.ts). What
 * differs is how the answer comes back: a language model cannot answer a typed question, so it returns structured
 * output, a one-field object checked against `moveSchema`. The call is kept as small as the comparison allows: one tiny
 * object, temperature 0, few output tokens, and thinking switched off through the lane's `extraBody` where the model has
 * a switch (a model that thinks before it answers is latency the demo would measure by accident). As in ./systemone.ts
 * the clock runs around the model call alone and there are no retries.
 *
 * Changed from upstream: the hosted gateway and its per-provider reasoning switches are replaced by a lane's own
 * `baseUrl`, `model` and `extraBody`.
 */
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { generateObject, type JSONValue, type LanguageModel } from 'ai';
import { z } from 'zod';
import type { DecisionState, LaneConfig, Move } from '../game/types';
import { LLM_SYSTEM_PROMPT, MOVES, stateForModel } from './prompt';
import type { ModelDecision } from './types';

/** Enough for the object plus the small amount of slack servers need. */
export const MAX_OUTPUT_TOKENS = 256;

export const moveSchema = z.object({ move: z.enum(MOVES) });

export interface ChatOptions {
  signal?: AbortSignal;
  /** Test seam: a language model instance instead of one built from the lane. */
  modelOverride?: LanguageModel;
}

/** Validates whatever came back and returns the move, or throws a short reason. */
export function parseMove(value: unknown): Move {
  const parsed = moveSchema.safeParse(value);
  if (parsed.success) return parsed.data.move;
  throw new Error(`Model returned an unusable decision: ${describe(value)}`);
}

function describe(value: unknown): string {
  let text: string;
  try {
    text = JSON.stringify(value) ?? String(value);
  } catch {
    text = String(value);
  }
  return text.length > 80 ? `${text.slice(0, 77)}...` : text;
}

export function chatModelFor(lane: LaneConfig): LanguageModel {
  if (!lane.baseUrl || !lane.model) throw new Error(`Lane "${lane.id}" needs a baseUrl and a model.`);
  const provider = createOpenAICompatible({
    name: lane.id,
    baseURL: lane.baseUrl,
    apiKey: lane.apiKey,
    supportsStructuredOutputs: true,
  });
  return provider(lane.model);
}

export async function decideWithChat(
  lane: LaneConfig,
  state: DecisionState,
  opts: ChatOptions = {},
): Promise<ModelDecision> {
  const model = opts.modelOverride ?? chatModelFor(lane);

  const startedAt = performance.now();
  const result = await generateObject({
    model,
    schema: moveSchema,
    schemaName: 'PaddleMove',
    schemaDescription: 'The move for the right paddle on this tick.',
    system: LLM_SYSTEM_PROMPT,
    prompt: JSON.stringify(stateForModel(state)),
    temperature: 0,
    maxOutputTokens: MAX_OUTPUT_TOKENS,
    maxRetries: 0,
    abortSignal: opts.signal,
    providerOptions: lane.extraBody ? { [lane.id]: lane.extraBody as Record<string, JSONValue> } : undefined,
  });
  const latencyMs = performance.now() - startedAt;

  return { move: parseMove(result.object), latencyMs };
}
