import { describe, expect, it } from 'vitest';
import { LANES } from '../game/types';
import { loadLanes, parseLanes } from './lanes';

const ONE = { id: 'a', label: 'A', kind: 'systemone', baseUrl: 'http://127.0.0.1:8100/' };

describe('parseLanes', () => {
  it('fills the defaults and trims the trailing slash', () => {
    const [lane] = parseLanes(JSON.stringify([ONE]));
    expect(lane?.baseUrl).toBe('http://127.0.0.1:8100');
    expect(lane?.provider).toBeNull();
    expect(lane?.color).toBeTruthy();
  });

  it('needs a model for a chat lane', () => {
    expect(() => parseLanes(JSON.stringify([{ ...ONE, kind: 'chat' }]))).toThrow(/needs a model/);
    expect(parseLanes(JSON.stringify([{ ...ONE, kind: 'chat', model: 'm' }]))).toHaveLength(1);
  });

  it('refuses duplicate ids, an empty list and more than eight lanes', () => {
    expect(() => parseLanes(JSON.stringify([ONE, ONE]))).toThrow(/unique/);
    expect(() => parseLanes('[]')).toThrow();
    const nine = Array.from({ length: 9 }, (_, i) => ({ ...ONE, id: `l${i}` }));
    expect(() => parseLanes(JSON.stringify(nine))).toThrow();
  });
});

describe('loadLanes', () => {
  it('falls back to the default lane', () => {
    expect(loadLanes({})).toBe(LANES);
  });

  it('reads PONG_LANES', () => {
    expect(loadLanes({ PONG_LANES: JSON.stringify([ONE]) }).map((l) => l.id)).toEqual(['a']);
  });
});
