/**
 * Which lanes a recording runs and where each one's model is served (added for this repository).
 *
 * Upstream had four lanes wired to a hosted gateway. Here a lane is any server: a System One server
 * (`kind: "systemone"`, `POST <baseUrl>/v1/systemone`, no key) or an OpenAI-compatible chat server
 * (`kind: "chat"`, `POST <baseUrl>/chat/completions`), listed in `PONG_LANES` (JSON) or in the file named by
 * `PONG_LANES_FILE`. With neither, the one default lane runs: a System One server on 127.0.0.1:8100.
 */
import { readFileSync } from 'node:fs';
import { z } from 'zod';
import { LANES, type LaneConfig } from '../game/types';

const laneSchema = z.object({
  id: z.string().regex(/^[a-z0-9][a-z0-9_-]*$/, 'lane ids are lower-case words'),
  label: z.string().min(1),
  provider: z.string().nullable().default(null),
  kind: z.enum(['systemone', 'chat']),
  color: z.string().default('#C8CCD4'),
  baseUrl: z.string().url(),
  model: z.string().optional(),
  apiKey: z.string().optional(),
  extraBody: z.record(z.string(), z.unknown()).optional(),
});

const lanesSchema = z.array(laneSchema).min(1).max(8);

export function parseLanes(text: string): LaneConfig[] {
  const lanes = lanesSchema.parse(JSON.parse(text)) as LaneConfig[];
  const ids = new Set(lanes.map((lane) => lane.id));
  if (ids.size !== lanes.length) throw new Error('lane ids must be unique');
  for (const lane of lanes) {
    if (lane.kind === 'chat' && !lane.model) throw new Error(`chat lane "${lane.id}" needs a model name`);
  }
  return lanes.map((lane) => ({ ...lane, baseUrl: lane.baseUrl?.replace(/\/+$/, '') }));
}

export function loadLanes(env: Record<string, string | undefined> = process.env): LaneConfig[] {
  if (env.PONG_LANES) return parseLanes(env.PONG_LANES);
  if (env.PONG_LANES_FILE) return parseLanes(readFileSync(env.PONG_LANES_FILE, 'utf8'));
  return LANES;
}

/** What goes in a lane's tag: its provider, or its kind when none is given. */
export function laneTag(lane: LaneConfig): string {
  return lane.provider ?? (lane.kind === 'chat' ? 'chat model' : 'System One server');
}
