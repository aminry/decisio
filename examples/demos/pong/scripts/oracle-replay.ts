/**
 * Score recorded Pong lanes against the oracle (added for this repository).
 *
 * The oracle is a perfect-information paddle policy: from the true ball (x, y, vx, vy) it computes where the ball crosses
 * the paddle's plane after the wall bounces (engine.predictIntercept, which agrees exactly with applyTick) and moves the
 * paddle toward that point at the full step (engine.moveToward: stay inside half a step). Three things are computed for
 * every decision of a recorded lane, from the snapshot before it:
 *   oracle     the move above
 *   question   the move the question's own wording asks for (prompt.expectedMove: stay within 5 units)
 *   outcome    whether the ball is returned if the model's move is played and the oracle's moves follow, and the same for the
 *              oracle's move (the engine itself plays it out), so a disagreement is harmful only if it turns a return into a miss
 * Each reconstructed tick is checked against the next recorded snapshot, so the reconstruction is verified on every row.
 *
 *   pnpm exec tsx scripts/oracle-replay.ts [--variant tolerance6] <replay.json> [more.json ...] > rows.jsonl
 *
 * --variant names the prompt variant the lane was asked with (lib/decide/prompt.ts), so `question` is the move that
 * variant's wording asks for.
 */
import { readFileSync } from 'node:fs';
import { applyTick, moveToward, predictIntercept, toDecisionState } from '../lib/game/engine';
import type { EngineState, Move, Replay, Snapshot } from '../lib/game/types';
import { promptVariant } from '../lib/decide/prompt';

function stateOf(s: Snapshot): EngineState {
  return {
    seed: s.seed, rng: 0, tick: s.tick, ball: { ...s.ball }, leftY: s.leftY, rightY: s.rightY,
    score: [s.score[0], s.score[1]], status: 'playing', serveDir: 1,
  };
}

/** Play `first`, then the oracle's moves, until the ball is returned (its x velocity flips) or the point is lost. */
function outcome(start: EngineState, first: Move): 'return' | 'miss' | 'unresolved' {
  let s = applyTick(start, first, 'auto');
  const dir = Math.sign(start.ball.vx);
  for (let i = 0; i < 40; i += 1) {
    if (s.score[0] !== start.score[0]) return 'miss';
    if (s.status !== 'playing') return 'miss';
    if (Math.sign(s.ball.vx) !== dir) return 'return';
    s = applyTick(s, moveToward(s.rightY, predictIntercept(s, 'right')), 'auto');
  }
  return 'unresolved';
}

const args = process.argv.slice(2);
const vi = args.indexOf('--variant');
const variant = promptVariant(vi >= 0 ? args[vi + 1] : undefined);
const files = vi >= 0 ? args.filter((_, i) => i !== vi && i !== vi + 1) : args;

/** The move the variant's wording asks for: stay within its tolerance of interceptY. */
function wordingMove(ds: { interceptY: number | null; paddle: { y: number } }): Move {
  if (ds.interceptY === null) return 'stay';
  const d = ds.interceptY - ds.paddle.y;
  if (Math.abs(d) <= variant.tolerance) return 'stay';
  return d < 0 ? 'up' : 'down';
}

for (const file of files) {
  const replay = JSON.parse(readFileSync(file, 'utf8')) as Replay;
  for (const lane of replay.lanes) {
    const snaps = lane.snapshots;
    for (let i = 1; i < snaps.length; i += 1) {
      const prev = snaps[i - 1];
      const cur = snaps[i];
      if (!prev || !cur || cur.move === null || prev.status !== 'playing' || prev.ball.vx === 0) continue;
      const start = stateOf(prev);
      const ds = toDecisionState(start, 'right');
      const rebuilt = applyTick(start, cur.move, 'auto');
      const verified = Math.abs(rebuilt.rightY - cur.rightY) < 1e-6 && (cur.status !== 'playing' || Math.abs(rebuilt.ball.y - cur.ball.y) < 1e-6);
      const oracle = moveToward(start.rightY, predictIntercept(start, 'right'));
      const question = wordingMove(ds);
      const d = ds.interceptY === null ? null : ds.interceptY - ds.paddle.y;
      process.stdout.write(`${JSON.stringify({
        file, variant: variant.name, lane: lane.model, label: lane.label, seed: replay.seed, tick: cur.tick, move: cur.move, oracle, question, state: ds,
        delta: d, outcome_model: outcome(start, cur.move), outcome_oracle: outcome(start, oracle), verified, latency_ms: cur.latencyMs,
      })}\n`);
    }
  }
}
