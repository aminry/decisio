import { describe, expect, it } from 'vitest';
import type { DecisionState } from '../game/types';
import { MOVE_QUESTIONS, moveQuestionsFor, promptVariant, stateForModel, stateForVariant } from './prompt';

const STATE: DecisionState = {
  court: { w: 160, h: 100 },
  ball: { x: 99, y: 37.8, vx: 19, vy: -12.2 },
  paddle: { y: 50, h: 20 },
  interceptY: 1.9,
  dir: 'toward',
};

describe('prompt variants', () => {
  it('the default variant is exactly today\'s question and state', () => {
    const v = promptVariant(undefined);
    expect(moveQuestionsFor(v)).toBe(MOVE_QUESTIONS);
    expect(stateForVariant(STATE, v)).toEqual(stateForModel(STATE));
  });

  it('tolerance6 moves the stay band to 6 units everywhere it is stated', () => {
    const q = moveQuestionsFor(promptVariant('tolerance6')).move;
    const text = JSON.stringify(q);
    expect(text).not.toMatch(/\b5 units|- 5\)|\+ 5\)/);
    expect(q.instructions).toContain('within 6 units');
    expect(q.criteria.up).toContain('(interceptY < paddle.y - 6)');
    expect(q.criteria.stay).toContain('within 6 units');
  });

  it('offset adds the signed distance to the state and describes it, and is null while the ball moves away', () => {
    const v = promptVariant('tolerance6+offset');
    expect(stateForVariant(STATE, v).offset).toBeCloseTo(-48.1, 6);
    expect(stateForVariant({ ...STATE, interceptY: null, dir: 'away' }, v).offset).toBeNull();
    expect(moveQuestionsFor(v).move.instructions).toContain('offset is interceptY minus paddle.y');
  });

  it('refuses an unknown variant', () => {
    expect(() => promptVariant('verdict')).toThrow(/unknown prompt variant/);
  });
});
