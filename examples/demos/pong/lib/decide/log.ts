/**
 * Decision logging (added for this repository).
 *
 * With PONG_DECISION_LOG=<path> set, every model decision appends one JSON line: the request the server received and the
 * full answer, with the latency. The measurement script (examples/demos/tools/measure_pong.py) sets it, compresses the file
 * and lists it in the run's manifest. Unset, nothing is written.
 */
import { appendFileSync } from 'node:fs';

export function logDecision(row: Record<string, unknown>): void {
  const path = process.env.PONG_DECISION_LOG;
  if (!path) return;
  appendFileSync(path, `${JSON.stringify({ demo: 'pong', ts: Date.now() / 1000, seed: process.env.SEED ?? null, ...row })}\n`);
}
