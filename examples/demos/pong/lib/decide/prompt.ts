/**
 * What every model is asked.
 *
 * A decision is one question — which way should the right paddle move? — with
 * three answers: up, down, stay. Every lane is given exactly two things: the
 * ~130-byte numeric DecisionState from lib/game/types.ts, and the instruction
 * text in this file. A System One lane takes them as a typed choice question (MOVE_QUESTIONS,
 * used by ./systemone.ts); the chat lane takes the same words as a system prompt and
 * answer with structured output (LLM_SYSTEM_PROMPT, used by ./llm.ts). Both are
 * built from DECISION_INSTRUCTIONS and MOVE_CRITERIA below and from nothing
 * else, because two lanes answering slightly different questions would make the
 * latency comparison meaningless.
 *
 * Court semantics (see lib/game/types.ts): y = 0 is the top edge and y grows
 * downward, so "up" DEcreases y. paddle.y is the paddle CENTRE. interceptY is
 * where the ball will cross the paddle's x plane, or null when the ball is
 * moving away.
 */
import type { JSONValue } from 'ai';
import type { DecisionState, Move } from '../game/types';

/** Every legal answer, in a fixed order. */
export const MOVES = ['up', 'down', 'stay'] as const;

/** The paddle counts as covering interceptY inside this many units. */
export const STAY_TOLERANCE = 5;

export const DECISION_INSTRUCTIONS = [
  'You control the right paddle in a game of Pong.',
  'The state is a JSON object of numbers: the court size, the ball position and velocity,',
  'your paddle (paddle.y is its CENTRE, paddle.h its height), interceptY, and dir.',
  'y = 0 is the top edge and y grows downward, so "up" DECREASES y and "down" INCREASES y.',
  'interceptY is the y value where the ball will cross your paddle plane after any wall bounces;',
  'it is null when the ball is moving away from you.',
  `Move so that paddle.y covers interceptY. Treat the paddle as already covering it when paddle.y is within ${STAY_TOLERANCE} units of interceptY.`,
  'Answer with one move only.',
].join(' ');

/** Option descriptions. Identical wording for the System One and chat lanes. */
export const MOVE_CRITERIA = {
  up: `interceptY is more than ${STAY_TOLERANCE} units above paddle.y (interceptY < paddle.y - ${STAY_TOLERANCE}). Moving up decreases paddle.y.`,
  down: `interceptY is more than ${STAY_TOLERANCE} units below paddle.y (interceptY > paddle.y + ${STAY_TOLERANCE}). Moving down increases paddle.y.`,
  stay: `interceptY is null, or paddle.y is already within ${STAY_TOLERANCE} units of interceptY.`,
} as const;

/** The whole question set handed to `evaluate()`: one choice, id "move". */
export const MOVE_QUESTIONS = {
  move: {
    type: 'choice',
    instructions: DECISION_INSTRUCTIONS,
    criteria: MOVE_CRITERIA,
  },
} as const;

/** The same instructions and criteria, flattened into a system prompt. */
export const LLM_SYSTEM_PROMPT = [
  DECISION_INSTRUCTIONS,
  '',
  'Options:',
  ...MOVES.map((move) => `- ${move}: ${MOVE_CRITERIA[move]}`),
].join('\n');

/**
 * Prompt variants for testing fixes against the oracle (added for this repository; the default is unchanged):
 *   tolerance6  the stay band is 6 units, half the paddle's 12-unit step (the game's own rule: inside it a move
 *               overshoots), instead of 5
 *   offset      the state also carries `offset`, interceptY minus paddle.y (null while the ball moves away): the
 *               distance the question is about, an observation computed from the state, not a move
 * Selected with PONG_PROMPT_VARIANT (a "+"-joined list), read by the System One lane.
 */
export type PromptVariant = { tolerance: number; offset: boolean; name: string };

export function promptVariant(name: string | undefined = undefined): PromptVariant {
  const parts = new Set((name ?? 'default').split('+').filter((p) => p && p !== 'default'));
  for (const p of parts) if (p !== 'tolerance6' && p !== 'offset') throw new Error(`unknown prompt variant: ${p}`);
  return { tolerance: parts.has('tolerance6') ? 6 : STAY_TOLERANCE, offset: parts.has('offset'), name: name ?? 'default' };
}

/** The move question for a variant; the default variant gives exactly MOVE_QUESTIONS. */
export function moveQuestionsFor(v: PromptVariant) {
  if (v.tolerance === STAY_TOLERANCE && !v.offset) return MOVE_QUESTIONS;
  const t = v.tolerance;
  const instructions = [
    DECISION_INSTRUCTIONS.replace(`within ${STAY_TOLERANCE} units`, `within ${t} units`),
    ...(v.offset ? ['offset is interceptY minus paddle.y (null when the ball is moving away).'] : []),
  ].join(' ');
  const criteria = Object.fromEntries(
    MOVES.map((m) => [m, MOVE_CRITERIA[m].replaceAll(`${STAY_TOLERANCE} units`, `${t} units`).replaceAll(`- ${STAY_TOLERANCE})`, `- ${t})`).replaceAll(`+ ${STAY_TOLERANCE})`, `+ ${t})`)]),
  );
  return { move: { type: 'choice', instructions, criteria } } as const;
}

/** The state for a variant: stateForModel, plus `offset` when the variant asks for it. */
export function stateForVariant(state: DecisionState, v: PromptVariant): Record<string, JSONValue> {
  const base = stateForModel(state);
  if (!v.offset) return base;
  const offset = state.interceptY === null ? null : Math.round((state.interceptY - state.paddle.y) * 10) / 10;
  return { ...base, offset };
}

/**
 * The exact payload handed to a model: the DecisionState, numbers only, no prose.
 * Rebuilt field by field so nothing a caller bolted onto the object leaks out.
 */
export function stateForModel(state: DecisionState): Record<string, JSONValue> {
  return {
    court: { w: state.court.w, h: state.court.h },
    ball: { x: state.ball.x, y: state.ball.y, vx: state.ball.vx, vy: state.ball.vy },
    paddle: { y: state.paddle.y, h: state.paddle.h },
    interceptY: state.interceptY,
    dir: state.dir,
  };
}

export function isMove(value: unknown): value is Move {
  return typeof value === 'string' && (MOVES as readonly string[]).includes(value);
}

/**
 * The move the instructions ask for, computed locally. Tests use it to check that
 * a model's answer maps to the same rule; the game never calls it to play.
 */
export function expectedMove(state: DecisionState): Move {
  const target = state.interceptY;
  if (target === null) return 'stay';
  const delta = target - state.paddle.y;
  if (Math.abs(delta) <= STAY_TOLERANCE) return 'stay';
  return delta < 0 ? 'up' : 'down';
}
