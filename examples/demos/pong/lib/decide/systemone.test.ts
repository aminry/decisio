import { describe, expect, it, vi } from 'vitest';
import type { DecisionState, LaneConfig } from '../game/types';
import { decide } from './index';
import { decideWithSystemOne, SystemOneHttpError } from './systemone';

const STATE: DecisionState = {
  court: { w: 800, h: 500 },
  ball: { x: 400, y: 200, vx: 5, vy: 1 },
  paddle: { y: 250, h: 80 },
  interceptY: 180,
  dir: 'toward',
};

const LANE: LaneConfig = { id: 'sys1', label: 'S', provider: null, kind: 'systemone', baseUrl: 'http://127.0.0.1:8100' };

function reply(body: unknown, status = 200): typeof fetch {
  return vi.fn(async () => new Response(JSON.stringify(body), { status })) as unknown as typeof fetch;
}

describe('decideWithSystemOne', () => {
  it('posts the state and the move question to <baseUrl>/v1/systemone with no key', async () => {
    const fetchImpl = reply({ answers: { move: { choice: 'up' } } });
    const out = await decideWithSystemOne(LANE, STATE, { fetchImpl });
    expect(out.move).toBe('up');
    expect(out.latencyMs).toBeGreaterThanOrEqual(0);
    const [url, init] = (fetchImpl as unknown as ReturnType<typeof vi.fn>).mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://127.0.0.1:8100/v1/systemone');
    const headers = init.headers as Record<string, string>;
    expect(headers.authorization).toBeUndefined();
    const body = JSON.parse(String(init.body));
    expect(Object.keys(body.questions)).toEqual(['move']);
    expect(body.model).toBeUndefined();
  });

  it('sends the model name only when the lane has one', async () => {
    const fetchImpl = reply({ answers: { move: { choice: 'stay' } } });
    await decideWithSystemOne({ ...LANE, model: 'm1' }, STATE, { fetchImpl });
    const init = (fetchImpl as unknown as ReturnType<typeof vi.fn>).mock.calls[0]?.[1] as RequestInit;
    expect(JSON.parse(String(init.body)).model).toBe('m1');
  });

  it('carries the status of a failed call', async () => {
    const err = await decideWithSystemOne(LANE, STATE, { fetchImpl: reply({}, 429) }).catch((e) => e);
    expect(err).toBeInstanceOf(SystemOneHttpError);
    expect(err.statusCode).toBe(429);
  });

  it('refuses an unknown choice', async () => {
    await expect(decideWithSystemOne(LANE, STATE, { fetchImpl: reply({ answers: { move: { choice: 'left' } } }) })).rejects.toThrow(
      /unknown choice/,
    );
  });
});

describe('decide', () => {
  it('maps a rate limit to rate_limited and a bad lane to bad_request', async () => {
    const original = globalThis.fetch;
    globalThis.fetch = reply({}, 429);
    try {
      expect(await decide({ model: 'sys1', state: STATE, mode: 'demo' }, { lanes: [LANE] })).toMatchObject({ ok: false, error: 'rate_limited' });
      expect(await decide({ model: 'nope', state: STATE, mode: 'demo' }, { lanes: [LANE] })).toMatchObject({ ok: false, error: 'bad_request' });
    } finally {
      globalThis.fetch = original;
    }
  });

  it('answers with the move and the latency on success', async () => {
    const original = globalThis.fetch;
    globalThis.fetch = reply({ answers: { move: { choice: 'down' } } });
    try {
      const out = await decide({ model: 'sys1', state: STATE, mode: 'demo' }, { lanes: [LANE] });
      expect(out).toMatchObject({ ok: true, move: 'down', model: 'sys1' });
    } finally {
      globalThis.fetch = original;
    }
  });
});
