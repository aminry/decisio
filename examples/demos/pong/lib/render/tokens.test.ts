import { describe, expect, it } from 'vitest';
import { LANES } from '../game/types';
import { contrastRatio, mixHex, parseHex, relativeLuminance, withAlpha } from './color';
import { COURT_W, LEFT_PLANE } from '../game/types';
import { COURT_SIZES, type CourtSurface } from './court';
import {
  CLIP_TOKENS,
  ballColorFor,
  endCardTotalMs,
  numberColorFor,
  type ClipLayout,
} from './tokens';

const LAYOUTS: ClipLayout[] = ['wide', 'square'];
const SURFACES = Object.keys(COURT_SIZES) as CourtSurface[];

describe('CLIP_TOKENS', () => {
  it('is the dark set from design/tokens.css', () => {
    const c = CLIP_TOKENS.colors;
    expect(c.bg).toBe('#0A0B0F');
    expect(c.surface).toBe('#12141A');
    expect(c.court).toBe('#12141A');
    expect(c.fg).toBe('#F2F3F5');
    expect(c.fgMuted).toBe('#8B909B');
    expect(c.fgInflight).toBe('#6B7080');
    expect(c.paddle).toBe('#E7E9EE');
    expect(c.ball).toBe('#C8CCD4');
    expect(c.accent).toBe('#FF5416');
    expect(c.barMuted).toBe('#2A2D36');
  });

  it('gives the accent to the first lane and to nobody else', () => {
    for (const lane of LANES) {
      const jev = lane.id === 'sys1';
      expect(ballColorFor(CLIP_TOKENS, lane.id), lane.id).toBe(
        jev ? CLIP_TOKENS.colors.accent : CLIP_TOKENS.colors.ball,
      );
      expect(numberColorFor(CLIP_TOKENS, lane.id), lane.id).toBe(
        jev ? CLIP_TOKENS.colors.accent : CLIP_TOKENS.colors.fg,
      );
    }
  });

  it('keeps every colour legible on the clip ground', () => {
    const bg = CLIP_TOKENS.colors.bg;
    // Body text and the muted tags carry real copy, so they clear 4.5:1.
    expect(contrastRatio(CLIP_TOKENS.colors.fg, bg)).toBeGreaterThan(4.5);
    expect(contrastRatio(CLIP_TOKENS.colors.fgMuted, bg)).toBeGreaterThan(4.5);
    // The in-flight number is never smaller than 34px, so 3:1 is the bar.
    expect(contrastRatio(CLIP_TOKENS.colors.fgInflight, bg)).toBeGreaterThan(3);
    expect(contrastRatio(CLIP_TOKENS.colors.accent, bg)).toBeGreaterThan(3);
  });

  it('gives every text role a family and a weight', () => {
    const f = CLIP_TOKENS.fonts;
    for (const role of [f.title, f.name, f.headline, f.body, f.tag, f.number, f.numberInFlight, f.mono]) {
      expect(role.family.length).toBeGreaterThan(0);
      expect(role.weight).toBeGreaterThan(0);
    }
    // Every stack ends in a generic family so a bare machine still renders.
    expect(f.stacks.mono).toMatch(/monospace$/);
    expect(f.stacks.sans).toMatch(/sans-serif$/);
    // The in-flight number is the same size at weight 400 (motion note 04).
    expect(f.numberInFlight.weight).toBe(400);
    expect(f.number.weight).toBe(500);
  });

  it('declares Geist font sources with at least one candidate path each', () => {
    expect(CLIP_TOKENS.fonts.sources.length).toBeGreaterThan(0);
    for (const source of CLIP_TOKENS.fonts.sources) {
      expect(source.paths.length).toBeGreaterThan(0);
      expect(source.family).toMatch(/^Geist/);
      expect(['normal', 'italic']).toContain(source.style);
    }
    const families = new Set(CLIP_TOKENS.fonts.sources.map((s) => s.family));
    expect([...families].sort()).toEqual(['Geist', 'Geist Mono']);
  });

  it('sizes both layouts to the brief, with room for four lanes', () => {
    expect(CLIP_TOKENS.sizes.wide.width).toBe(1200);
    expect(CLIP_TOKENS.sizes.wide.height).toBe(675);
    expect(CLIP_TOKENS.sizes.square.width).toBe(1080);
    expect(CLIP_TOKENS.sizes.square.height).toBe(1080);

    for (const layout of LAYOUTS) {
      const s = CLIP_TOKENS.sizes[layout];
      const content = s.width - s.padX * 2;
      if (s.laneMode === 'row') {
        // The court column must be the widest thing in the row.
        const courtW = content - s.nameColW - s.numberColW - s.colGap * 2;
        expect(courtW, layout).toBe(s.courtW);
      } else {
        expect(s.courtW, layout).toBe(content);
      }
      // Four lanes have to fit between the header and the credit strip.
      const lanesH =
        s.height - s.padTop - s.padBottom - s.headerH - s.stripH - s.blockGap * 2;
      expect(lanesH / 4, layout).toBeGreaterThan(s.courtH);
      // The number is the largest thing in the row (must-not-get-wrong d).
      expect(s.numberSize, layout).toBeGreaterThan(s.nameSize);
    }
  });

  it('carries the motion the renderers reproduce', () => {
    const m = CLIP_TOKENS.motion;
    expect(m.serveMs).toBe(300);
    expect(m.pulseMs).toBe(180);
    expect(m.pulseOutMs).toBe(60);
    expect(m.flashMs).toBe(120);
    expect(m.paddleSettle).toBeGreaterThan(0);
    expect(m.paddleSettle).toBeLessThanOrEqual(1);
    // 07: freeze 400 + cross-fade 300 + hold 2500 = 3.2s, so 12s play = 15.2s.
    expect(endCardTotalMs()).toBe(3200);
  });

  it('carries the trail rule from motion note 03', () => {
    const t = CLIP_TOKENS.trail;
    expect(t.referenceMs).toBe(430);
    expect(t.minFraction).toBeCloseTo(24 / 260, 9);
    expect(t.growth).toBeCloseTo(1.12, 9);
    expect(t.headAlpha).toBeGreaterThan(t.tailAlpha);
    expect(t.tailScale).toBeCloseTo(0.35, 9);
  });

  it('carries the approved copy', () => {
    const copy = CLIP_TOKENS.copy;
    expect(copy.endHeadline).toBe('Decisions in {n} seconds');
    expect(copy.tagline).toBe('One step per model decision');
  });

});

describe('COURT_SIZES', () => {
  // Paddle HEIGHT and paddle x are no longer in this table: both are derived
  // from the contract by drawCourt (PADDLE_H, LEFT_PLANE / RIGHT_PLANE) so the
  // drawn bar is the hitbox and sits on the plane the ball turns at. The rows
  // below therefore assert everything the table still owns, thickness included.
  it('matches the court size table in design/SPEC.md', () => {
    expect(COURT_SIZES['home-desktop']).toMatchObject({ w: 704, h: 80, ballR: 4.5, paddleW: 4, trailPx: 145, discs: 5 });
    expect(COURT_SIZES['home-phone']).toMatchObject({ w: 358, h: 64, ballR: 4, paddleW: 3, trailPx: 110, discs: 5 });
    expect(COURT_SIZES.play).toMatchObject({ w: 704, h: 300, ballR: 7, paddleW: 6, radius: 8 });
    expect(COURT_SIZES.watch).toMatchObject({ w: 358, h: 200, ballR: 5, paddleW: 4 });
    expect(COURT_SIZES['clip-wide']).toMatchObject({ w: 640, h: 100, ballR: 7, paddleW: 5, trailPx: 260, discs: 7 });
    expect(COURT_SIZES['clip-square']).toMatchObject({ w: 992, h: 130, ballR: 8, paddleW: 6, trailPx: 320, discs: 7 });
  });

  it('keeps every surface self-consistent', () => {
    for (const surface of SURFACES) {
      const s = COURT_SIZES[surface];
      const scale = s.w / COURT_W;
      expect(s.w, surface).toBeGreaterThan(s.h);
      // The paddle is drawn from its plane outward, so it must fit in the inset.
      expect(s.paddleW, surface).toBeLessThanOrEqual(LEFT_PLANE * scale);
      expect(s.lineInset * 2, surface).toBeLessThan(s.h);
      expect(s.discs, surface).toBeGreaterThan(0);
      expect(s.ballR * 2, surface).toBeLessThan(s.h);
    }
    // Only the two playing surfaces put the score inside the court.
    expect(COURT_SIZES.play.score).not.toBeNull();
    expect(COURT_SIZES.watch.score).not.toBeNull();
    expect(COURT_SIZES['home-desktop'].score).toBeNull();
    expect(COURT_SIZES['clip-wide'].score).toBeNull();
  });
});

describe('colour helpers', () => {
  it('parses both hex forms', () => {
    expect(parseHex('#fff')).toEqual([255, 255, 255]);
    expect(parseHex('#FF5416')).toEqual([255, 84, 22]);
    expect(parseHex('nonsense')).toEqual([0, 0, 0]);
  });

  it('mixes and clamps', () => {
    expect(mixHex('#000000', '#ffffff', 0)).toBe('#000000');
    expect(mixHex('#000000', '#ffffff', 1)).toBe('#ffffff');
    expect(mixHex('#000000', '#ffffff', 0.5)).toBe('#808080');
    expect(mixHex('#000000', '#ffffff', 5)).toBe('#ffffff');
  });

  it('adds alpha as rgba', () => {
    expect(withAlpha('#FF5416', 0.5)).toBe('rgba(255, 84, 22, 0.5)');
  });

  it('contrastRatio spans black on white', () => {
    expect(contrastRatio('#000000', '#ffffff')).toBeCloseTo(21, 1);
    expect(contrastRatio('#123456', '#123456')).toBeCloseTo(1, 6);
    expect(relativeLuminance('#ffffff')).toBeCloseTo(1, 6);
    expect(relativeLuminance('#000000')).toBeCloseTo(0, 6);
  });
});
