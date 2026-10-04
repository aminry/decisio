/**
 * lib/render/tokens.ts — the ONE place the clip's look is defined.
 *
 * DESIGNER: this file is the whole visual contract for the headless renderer.
 * Every colour, font, size, radius, trail and timing the clip uses is read
 * from `CLIP_TOKENS`, so swapping this object restyles the clip, the stills
 * and the square cut without touching drawing code.
 *
 * The values are the DARK set from design/tokens.css, which is the only set
 * the clip uses: the clip ground is #0A0B0F and the court is #12141A whatever
 * the viewer's system is doing. The site reads the same tokens as CSS custom
 * properties out of app/globals.css.
 *
 * Layout note: `sizes` carries one block per output ratio. `wide` is the
 * 1200x675 social frame with a three-column lane (name | court | number);
 * `square` is 1080x1080 and stacks the name row over a full-width court, the
 * same way the site does on a phone.
 */

import type { ClipFontRole } from './text';

export type { ClipFontRole };

// ---------------------------------------------------------------------------
// Fonts
// ---------------------------------------------------------------------------

/**
 * A font file the headless renderer registers before drawing.
 *
 * `paths` are repo-relative candidates tried in order; the first that exists
 * is registered under `family`. The `geist` npm package ships static TTFs for
 * every weight, which @napi-rs/canvas registers directly, so nothing has to be
 * vendored — `public/fonts/` is listed second as an escape hatch for a machine
 * without node_modules.
 */
export interface ClipFontSource {
  family: string;
  weight: number;
  style: 'normal' | 'italic';
  paths: string[];
}

export interface ClipFonts {
  sources: ClipFontSource[];
  /** Family stacks. Registered families first, system fallbacks last. */
  stacks: { sans: string; mono: string };
  /** Geist 700 -0.02em — the wordmark. */
  title: ClipFontRole;
  /** Geist 600 — lane names and end-card names. */
  name: ClipFontRole;
  /** Geist 700 -0.03em — the end-card headline. */
  headline: ClipFontRole;
  /** Geist 400 — taglines, subtitles, small lines. */
  body: ClipFontRole;
  /** Geist 400 +0.04em caps — provider tags. */
  tag: ClipFontRole;
  /** Geist Mono 500 -0.02em tabular — the latency number and the end-card count. */
  number: ClipFontRole;
  /** Geist Mono 400 -0.02em — the number while a decision is in flight. */
  numberInFlight: ClipFontRole;
  /** Geist Mono 500 — the elapsed timer and the URL. */
  mono: ClipFontRole;
}

// ---------------------------------------------------------------------------
// Colour, geometry, motion
// ---------------------------------------------------------------------------

/** The dark half of design/tokens.css. Names match the CSS custom properties. */
export interface ClipColors {
  bg: string;
  surface: string;
  hair: string;
  fg: string;
  fgMuted: string;
  fgInflight: string;
  court: string;
  courtLine: string;
  paddle: string;
  ball: string;
  accent: string;
  barMuted: string;
}

/**
 * Trail geometry, shared by the clip and the site.
 *
 * `trail_px = clamp(maxPx * referenceMs / latency, maxPx * minFraction, maxPx)`
 * where `maxPx` comes from the court size table in ./court.
 */
export interface ClipTrail {
  /** Latency at which the trail is drawn at its full length. */
  referenceMs: number;
  /** Shortest trail, as a fraction of the full length. 24/260 on the clip. */
  minFraction: number;
  /** Each disc sits this much further behind the last one. */
  growth: number;
  /** Alpha of the disc nearest the ball, and of the one furthest away. */
  headAlpha: number;
  tailAlpha: number;
  /** Radius of the furthest disc, as a fraction of the ball radius. */
  tailScale: number;
}

/** Every timing in the clip, in milliseconds. Motion notes 01, 04, 07. */
export interface ClipMotion {
  /** 01 — the scripted serve tweens one step out of centre, ease-out. */
  serveMs: number;
  /** 04 — the number snaps and pulses when a decision lands. */
  pulseMs: number;
  pulseOutMs: number;
  pulseScale: number;
  /** 04 — and flashes to pure white for this long. */
  flashMs: number;
  /**
   * Paddles move at the START of a tick in the engine, so on the clip they
   * finish this far into the interval between two snapshots and then hold.
   */
  paddleSettle: number;
  /** 07 — freeze, cross-fade, bars, hold. */
  freezeMs: number;
  crossFadeMs: number;
  barGrowMs: number;
  barStaggerMs: number;
  holdMs: number;
}

export interface ClipLayoutSizes {
  width: number;
  height: number;
  padTop: number;
  padX: number;
  padBottom: number;

  // --- header -------------------------------------------------------------
  headerH: number;
  /** Gap between header, lane stack and bottom strip. */
  blockGap: number;
  titleSize: number;
  taglineSize: number;
  elapsedLabelSize: number;
  elapsedSize: number;

  // --- lanes --------------------------------------------------------------
  /** 'row' = name | court | number. 'stack' = name row over a full-width court. */
  laneMode: 'row' | 'stack';
  laneGap: number;
  nameColW: number;
  numberColW: number;
  colGap: number;
  /** Gap between the name row and the court when `laneMode` is 'stack'. */
  stackGap: number;
  courtW: number;
  courtH: number;
  nameSize: number;
  providerSize: number;
  numberSize: number;
  unitSize: number;
  countsSize: number;

  // --- bottom strip -------------------------------------------------------
  stripH: number;
  poweredSize: number;
  creditsSize: number;

  // --- end card -----------------------------------------------------------
  endPadTop: number;
  endPadX: number;
  endPadBottom: number;
  endTitleSize: number;
  endHeadlineSize: number;
  endSubtitleSize: number;
  endNameColW: number;
  endCountColW: number;
  endColGap: number;
  endRowGap: number;
  /**
   * Most air the end card puts between its blocks. Without it the square frame,
   * which is much taller than its content, would spread them to the corners.
   */
  endMaxGap: number;
  endNameSize: number;
  endBarH: number;
  endBarRadius: number;
  endReturnsSize: number;
  endCountSize: number;
  endUrlLabelSize: number;
  endUrlSize: number;
}

export interface ClipCopy {
  title: string;
  tagline: string;
  elapsedLabel: string;
  msLabel: string;
  /** `{n}` is replaced with the number of seconds of play in the clip. */
  endHeadline: string;
  endSubtitle: string;
}

export interface ClipTokens {
  colors: ClipColors;
  fonts: ClipFonts;
  sizes: {
    wide: ClipLayoutSizes;
    square: ClipLayoutSizes;
    social: ClipLayoutSizes;
    duel: ClipLayoutSizes;
  };
  trail: ClipTrail;
  motion: ClipMotion;
  copy: ClipCopy;
}

/**
 * wide   1200 x 675, the OG card and a desktop post.
 * square 1080 x 1080, four lanes stacked as on a phone.
 * social 1080 x 1350 (4:5), the same four lanes with room to breathe: taller
 *        courts, a bigger ball and a number you can read in a feed at arm's
 *        length. This is the one for X and LinkedIn on a phone.
 * duel   1080 x 1350, two or three lanes only (two or three lanes), each court
 *        a third of the frame tall and the number the size of a headline. The
 *        meme cut: one glance says which ball is moving.
 */
export type ClipLayout = 'wide' | 'square' | 'social' | 'duel';

// ---------------------------------------------------------------------------
// The defaults
// ---------------------------------------------------------------------------

const SANS = '"Geist", system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif';
const MONO = '"Geist Mono", ui-monospace, Menlo, "DejaVu Sans Mono", monospace';

const GEIST_SANS = 'node_modules/geist/dist/fonts/geist-sans';
const GEIST_MONO = 'node_modules/geist/dist/fonts/geist-mono';

function sans(weight: number, file: string): ClipFontSource {
  return {
    family: 'Geist',
    weight,
    style: 'normal',
    paths: [GEIST_SANS + '/' + file, 'public/fonts/' + file],
  };
}

function mono(weight: number, file: string): ClipFontSource {
  return {
    family: 'Geist Mono',
    weight,
    style: 'normal',
    paths: [GEIST_MONO + '/' + file, 'public/fonts/' + file],
  };
}

export const CLIP_TOKENS: ClipTokens = {
  colors: {
    bg: '#0A0B0F',
    surface: '#12141A',
    hair: 'rgba(255, 255, 255, 0.08)',
    fg: '#F2F3F5',
    fgMuted: '#8B909B',
    fgInflight: '#6B7080',
    court: '#12141A',
    courtLine: 'rgba(255, 255, 255, 0.16)',
    paddle: '#E7E9EE',
    ball: '#C8CCD4',
    accent: '#FF5416',
    barMuted: '#2A2D36',
  },

  fonts: {
    sources: [
      sans(400, 'Geist-Regular.ttf'),
      sans(500, 'Geist-Medium.ttf'),
      sans(600, 'Geist-SemiBold.ttf'),
      sans(700, 'Geist-Bold.ttf'),
      mono(400, 'GeistMono-Regular.ttf'),
      mono(500, 'GeistMono-Medium.ttf'),
    ],
    stacks: { sans: SANS, mono: MONO },
    title: { family: SANS, weight: 700, tracking: -0.02 },
    name: { family: SANS, weight: 600 },
    headline: { family: SANS, weight: 700, tracking: -0.03 },
    body: { family: SANS, weight: 400 },
    tag: { family: SANS, weight: 400, tracking: 0.04, uppercase: true },
    number: { family: MONO, weight: 500, tracking: -0.02 },
    numberInFlight: { family: MONO, weight: 400, tracking: -0.02 },
    mono: { family: MONO, weight: 500 },
  },

  sizes: {
    // 1200 x 675, padding 28 / 48 / 24, lanes 196 | 1fr | 220 with a 24 gap.
    wide: {
      width: 1200,
      height: 675,
      padTop: 28,
      padX: 48,
      padBottom: 24,

      headerH: 28,
      blockGap: 12,
      titleSize: 22,
      taglineSize: 14,
      elapsedLabelSize: 13,
      elapsedSize: 18,

      laneMode: 'row',
      laneGap: 10,
      nameColW: 196,
      numberColW: 220,
      colGap: 24,
      stackGap: 8,
      courtW: 640,
      courtH: 100,
      nameSize: 20,
      providerSize: 12,
      numberSize: 56,
      unitSize: 18,
      countsSize: 13,

      stripH: 54,
      poweredSize: 13,
      creditsSize: 12,

      endPadTop: 52,
      endPadX: 64,
      endPadBottom: 32,
      endTitleSize: 18,
      endHeadlineSize: 60,
      endSubtitleSize: 17,
      endNameColW: 220,
      endCountColW: 140,
      endColGap: 24,
      endRowGap: 14,
      endMaxGap: 56,
      endNameSize: 20,
      endBarH: 24,
      endBarRadius: 4,
      endReturnsSize: 12,
      endCountSize: 48,
      endUrlLabelSize: 15,
      endUrlSize: 22,
    },

    // 1080 x 1080, padding 44 / 44 / 36. Lanes stack, like the phone.
    square: {
      width: 1080,
      height: 1080,
      padTop: 44,
      padX: 44,
      padBottom: 36,

      headerH: 28,
      blockGap: 14,
      titleSize: 24,
      taglineSize: 15,
      elapsedLabelSize: 13,
      elapsedSize: 20,

      laneMode: 'stack',
      laneGap: 16,
      nameColW: 0,
      numberColW: 0,
      colGap: 0,
      stackGap: 8,
      courtW: 992,
      courtH: 130,
      nameSize: 22,
      providerSize: 12,
      numberSize: 52,
      unitSize: 18,
      countsSize: 13,

      stripH: 54,
      poweredSize: 13,
      creditsSize: 12,

      endPadTop: 60,
      endPadX: 64,
      endPadBottom: 40,
      endTitleSize: 20,
      endHeadlineSize: 64,
      endSubtitleSize: 19,
      endNameColW: 240,
      endCountColW: 150,
      endColGap: 28,
      endRowGap: 22,
      endMaxGap: 64,
      endNameSize: 24,
      endBarH: 28,
      endBarRadius: 4,
      endReturnsSize: 13,
      endCountSize: 52,
      endUrlLabelSize: 17,
      endUrlSize: 24,
    },

    // 1080 x 1350, padding 48 / 48 / 40. Stacked, with everything a size up.
    social: {
      width: 1080,
      height: 1350,
      padTop: 48,
      padX: 48,
      padBottom: 40,

      headerH: 32,
      blockGap: 18,
      titleSize: 28,
      taglineSize: 17,
      elapsedLabelSize: 14,
      elapsedSize: 22,

      laneMode: 'stack',
      laneGap: 22,
      nameColW: 0,
      numberColW: 0,
      colGap: 0,
      stackGap: 10,
      courtW: 984,
      courtH: 150,
      nameSize: 26,
      providerSize: 13,
      numberSize: 84,
      unitSize: 22,
      countsSize: 15,

      stripH: 54,
      poweredSize: 14,
      creditsSize: 12,

      endPadTop: 90,
      endPadX: 72,
      endPadBottom: 48,
      endTitleSize: 22,
      endHeadlineSize: 78,
      endSubtitleSize: 22,
      endNameColW: 280,
      endCountColW: 170,
      endColGap: 32,
      endRowGap: 32,
      endMaxGap: 96,
      endNameSize: 28,
      endBarH: 34,
      endBarRadius: 5,
      endReturnsSize: 14,
      endCountSize: 64,
      endUrlLabelSize: 19,
      endUrlSize: 28,
    },

    // 1080 x 1350 for two or three lanes. Courts fill what is left of the
    // height (layoutFrame shrinks them to fit when there are three).
    duel: {
      width: 1080,
      height: 1350,
      padTop: 56,
      padX: 56,
      padBottom: 44,

      headerH: 36,
      blockGap: 28,
      titleSize: 30,
      taglineSize: 20,
      elapsedLabelSize: 15,
      elapsedSize: 26,

      laneMode: 'stack',
      laneGap: 44,
      nameColW: 0,
      numberColW: 0,
      colGap: 0,
      stackGap: 14,
      courtW: 968,
      courtH: 300,
      nameSize: 40,
      providerSize: 16,
      numberSize: 120,
      unitSize: 30,
      countsSize: 20,

      stripH: 56,
      poweredSize: 14,
      creditsSize: 12,

      endPadTop: 100,
      endPadX: 72,
      endPadBottom: 48,
      endTitleSize: 22,
      endHeadlineSize: 78,
      endSubtitleSize: 22,
      endNameColW: 300,
      endCountColW: 200,
      endColGap: 32,
      endRowGap: 40,
      endMaxGap: 120,
      endNameSize: 32,
      endBarH: 40,
      endBarRadius: 6,
      endReturnsSize: 15,
      endCountSize: 84,
      endUrlLabelSize: 24,
      endUrlSize: 38,
    },
  },

  trail: {
    referenceMs: 430,
    minFraction: 24 / 260,
    growth: 1.12,
    headAlpha: 0.5,
    tailAlpha: 0.07,
    tailScale: 0.35,
  },

  motion: {
    serveMs: 300,
    pulseMs: 180,
    pulseOutMs: 60,
    pulseScale: 1.06,
    flashMs: 120,
    paddleSettle: 0.38,
    freezeMs: 400,
    crossFadeMs: 300,
    barGrowMs: 600,
    barStaggerMs: 80,
    holdMs: 2500,
  },

  copy: {
    title: 'Pong lanes',
    tagline: 'One step per model decision',
    elapsedLabel: 'elapsed',
    msLabel: 'ms',
    endHeadline: 'Decisions in {n} seconds',
    endSubtitle: 'Same serve, same rules. The ball moves one step per model decision.',
  },
};

/**
 * The ball and trail colour for a lane. Only the first lane carries the accent; every
 * other lane uses the neutral ball colour (lane accents rule, SPEC).
 */
export function ballColorFor(tokens: ClipTokens, model: string): string {
  return model === 'sys1' ? tokens.colors.accent : tokens.colors.ball;
}

/** The latency number's colour: the first lane's is the accent, the others' is text. */
export function numberColorFor(tokens: ClipTokens, model: string): string {
  return model === 'sys1' ? tokens.colors.accent : tokens.colors.fg;
}

/** How long the end card adds to a clip: freeze + cross-fade + hold. */
export function endCardTotalMs(tokens: ClipTokens = CLIP_TOKENS): number {
  const m = tokens.motion;
  return m.freezeMs + m.crossFadeMs + m.holdMs;
}
