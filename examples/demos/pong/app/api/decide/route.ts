/**
 * POST /api/decide — one model decision, over HTTP.
 *
 * WHAT   Takes a DecisionState and a lane, asks that model for one move, and
 *        answers with the move and the latency of the call. The body is
 *        validated here so `lib/decide` only ever sees a well-formed request.
 *        Status codes carry the failure as well as the JSON body, so a caller
 *        can react to 402 (budget gone) or 429 (rate limited) without reading
 *        the payload.
 * WHO    Nothing in the replay page. This endpoint is here so one decision can be exercised on its own, with curl or
 *        from a test, against a lane's server.
 * NEEDS  The lane configuration (lib/config/lanes.ts): `model` is a lane id.
 * REFUSES a body that is not `{ model, state, mode }` with a finite-number state (400) and an unknown lane (400).
 */
import { z } from 'zod';
import { decide } from '../../../lib/decide';
import type { DecideErrorCode, DecideResponse, DecideMode } from '../../../lib/game/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const MODES = ['demo', 'play'] as const satisfies readonly DecideMode[];

const finite = z.number().refine(Number.isFinite, 'must be a finite number');

const stateSchema = z.object({
  court: z.object({ w: finite, h: finite }),
  ball: z.object({ x: finite, y: finite, vx: finite, vy: finite }),
  paddle: z.object({ y: finite, h: finite }),
  interceptY: finite.nullable(),
  dir: z.enum(['toward', 'away']),
});

const bodySchema = z.object({
  model: z.string().min(1).max(64),
  state: stateSchema,
  mode: z.enum(MODES),
});

const STATUS: Record<DecideErrorCode, number> = {
  bad_request: 400,
  model_not_allowed: 403,
  out_of_credits: 402,
  rate_limited: 429,
  model_error: 502,
};

function send(body: DecideResponse): Response {
  return Response.json(body, {
    status: body.ok ? 200 : STATUS[body.error],
    headers: { 'Cache-Control': 'no-store' },
  });
}

export async function POST(request: Request): Promise<Response> {
  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return send({ ok: false, error: 'bad_request', message: 'Body must be JSON.' });
  }

  const parsed = bodySchema.safeParse(payload);
  if (!parsed.success) {
    const where = parsed.error.issues
      .slice(0, 3)
      .map((issue) => `${issue.path.join('.') || 'body'}: ${issue.message}`)
      .join('; ');
    return send({ ok: false, error: 'bad_request', message: `Invalid request. ${where}` });
  }

  return send(await decide(parsed.data, { signal: request.signal }));
}
