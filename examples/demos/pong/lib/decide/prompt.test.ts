import { describe, expect, it } from 'vitest';
import type { DecisionState } from '../game/types';
import { MOVE_QUESTIONS, WORDS_QUESTIONS, moveQuestionsFor, promptVariant, stateForModel, stateForVariant } from './prompt';

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

  it('words states the arrival relative to the paddle in a sentence, no coordinates, and defines the moves', () => {
    const v = promptVariant('words');
    expect(v.tolerance).toBe(6);
    expect(stateForVariant(STATE, v)).toEqual({ ball: 'The ball will arrive 48.1 units above the centre of your paddle.' });
    expect(stateForVariant({ ...STATE, interceptY: 61.3 }, v).ball).toBe('The ball will arrive 11.3 units below the centre of your paddle.');
    expect(stateForVariant({ ...STATE, interceptY: 50 }, v).ball).toBe('The ball will arrive level with the centre of your paddle.');
    expect(stateForVariant({ ...STATE, interceptY: null, dir: 'away' }, v).ball).toBe('The ball is moving away from your paddle.');
    const q = moveQuestionsFor(v);
    expect(q).toBe(WORDS_QUESTIONS);
    expect(q.move.criteria.up).toBe('Move the paddle 12 units toward the top of the screen.');
    // the options say what a move does, never when to choose it
    expect(JSON.stringify(q.move.criteria)).not.toMatch(/interceptY|when|if /);
    expect(() => promptVariant('words+offset')).toThrow(/stands alone/);
  });

  it('refuses an unknown variant', () => {
    expect(() => promptVariant('verdict')).toThrow(/unknown prompt variant/);
  });
});
