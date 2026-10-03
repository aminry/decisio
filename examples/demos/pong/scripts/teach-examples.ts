/**
 * Oracle-labelled examples of the move question, for registering a task (added for this repository).
 *
 * Plays seeded training games with the engine (seeds the caller keeps disjoint from the evaluated ones). The right paddle
 * follows the oracle (engine.moveToward toward engine.predictIntercept) with probability 1 - EXPLORE and a uniformly random
 * move otherwise, so off-target positions are visited too. Every decision while the ball comes toward the paddle becomes a
 * candidate example: the question exactly as the System One lane asks it (lib/decide/prompt.ts) and the oracle's move as
 * the answer. Ball-away states are skipped: the oracle's "stay" there is a convention, not a decision it scores. Then PER
 * examples per move are drawn with a seeded shuffle, one game at most contributing a quarter of a move's examples.
 *
 *   pnpm exec tsx scripts/teach-examples.ts --seeds 30000-30039 --per 10 > examples.json
 */
import { applyTick, createEngine, moveToward, predictIntercept, toDecisionState } from '../lib/game/engine';
import type { EngineState, Move } from '../lib/game/types';
import { MOVE_QUESTIONS, stateForModel } from '../lib/decide/prompt';

const EXPLORE = 0.35;

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1]! : fallback;
}

function mulberry(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const [lo, hi] = arg('seeds', '30000-30039').split('-').map(Number) as [number, number];
const per = Number(arg('per', '10'));
const pool: Record<Move, Array<{ seed: number; tick: number; state: unknown }>> = { up: [], down: [], stay: [] };
for (let seed = lo; seed <= hi; seed += 1) {
  const rand = mulberry(seed * 7919);
  let s: EngineState = createEngine(seed, seed % 2 === 0 ? 1 : -1);
  for (let i = 0; i < 400 && s.status !== 'over'; i += 1) {
    const target = predictIntercept(s, 'right');
    const oracle = moveToward(s.rightY, target);
    if (target !== null && s.status === 'playing') {
      pool[oracle].push({ seed, tick: s.tick, state: stateForModel(toDecisionState(s, 'right')) });
    }
    const moves: Move[] = ['up', 'down', 'stay'];
    const play = rand() < EXPLORE ? moves[Math.floor(rand() * 3)]! : oracle;
    s = applyTick(s, play, 'auto');
  }
}
const pick = mulberry(lo * 31 + hi);
const out: Array<{ request: unknown; answer: Move; seed: number; tick: number }> = [];
for (const move of ['up', 'down', 'stay'] as Move[]) {
  const c = pool[move];
  for (let i = c.length - 1; i > 0; i -= 1) {
    const j = Math.floor(pick() * (i + 1));
    [c[i], c[j]] = [c[j]!, c[i]!];
  }
  const perGame = new Map<number, number>();
  for (const e of c) {
    if (out.filter((o) => o.answer === move).length >= per) break;
    const n = perGame.get(e.seed) ?? 0;
    if (n >= Math.max(1, Math.floor(per / 4))) continue;
    perGame.set(e.seed, n + 1);
    out.push({ request: { state: e.state, questions: MOVE_QUESTIONS }, answer: move, seed: e.seed, tick: e.tick });
  }
}
process.stderr.write(`pool: up ${pool.up.length}, down ${pool.down.length}, stay ${pool.stay.length}; drawn ${out.length}\n`);
process.stdout.write(`${JSON.stringify({ question: 'move', seeds: [lo, hi], per, explore: EXPLORE, examples: out }, null, 1)}\n`);
