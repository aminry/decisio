/**
 * Records a real run of every configured lane and writes public/replay.json (what the page replays) plus
 * public/replay-stats.json (the numbers).
 *
 *   PONG_LANES_FILE=lanes.json pnpm record
 *
 * Options via env: RECORD_MS (default 45000), SEED (default 20260917), LANES=sys1,cmpa (a subset), OUT_DIR (default
 * public). Run it on the machine that runs the servers, so the latencies are the servers' and not a network's.
 * The lanes are described in lib/config/lanes.ts (lanes.example.json).
 */
import { mkdir, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import type { ModelId } from '../lib/game/types';
import { recordLanes } from '../lib/record/record-lanes';

const DURATION_MS = Number(process.env.RECORD_MS ?? 45_000);
const SEED = Number(process.env.SEED ?? 20260917);
const ONLY: ModelId[] | undefined = process.env.LANES?.trim()
  ? process.env.LANES.split(',').map((s) => s.trim())
  : undefined;

async function main() {
  const { replay, stats } = await recordLanes({
    seconds: DURATION_MS / 1000,
    seed: SEED,
    lanes: ONLY,
    onProgress: ({ elapsedMs, lanes }) => {
      const line = lanes
        .map(
          (l) =>
            `${l.model} t${l.tick} ${l.score.join('-')} ${l.latencyMs ?? '-'}ms ${l.status}`,
        )
        .join(' | ');
      console.log(`${Math.round(elapsedMs / 1000)}s  ${line}`);
    },
  });

  const replayJson = JSON.stringify(replay);
  const outDir = process.env.OUT_DIR ?? new URL('../public/', import.meta.url).pathname;
  await mkdir(outDir, { recursive: true });
  const replayOut = join(resolve(outDir), 'replay.json');
  await writeFile(replayOut, replayJson);

  const statsOut = join(resolve(outDir), 'replay-stats.json');
  await writeFile(statsOut, `${JSON.stringify(stats, null, 2)}\n`);

  console.log(`\nWrote ${replayOut} (${(replayJson.length / 1024).toFixed(0)} KB)`);
  console.log(`Wrote ${statsOut}`);
  for (const lane of stats.lanes) {
    console.log(
      `${lane.label.padEnd(18)} ticks=${String(lane.ticks).padStart(4)} ` +
        `${lane.decisionsPerSec}/s avg=${lane.avgMs}ms p95=${lane.p95Ms}ms ` +
        `returns=${lane.returns} misses=${lane.misses} ` +
        `score=${lane.finalScore.join('-')} status=${lane.status}`,
    );
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
