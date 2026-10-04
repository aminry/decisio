/**
 * lib/render/court.ts — one court, drawn once, used twice.
 *
 * This is the whole visual of a lane's court: bed, centre line, the score when
 * a surface shows one, paddles, trail and ball. It is pure — it reads a
 * `CourtView` (a position, in court units) and a `CourtStyle` (a look, in
 * device pixels) and paints into any `ClipCtx`. It owns no state, no timing
 * and no canvas.
 *
 * The React court (components/lane/court.tsx) calls it from its rAF loop after
 * it has tweened a snapshot into a view. The clip renderer calls it from
 * lib/render/frame.ts after lib/render/timeline.ts has sampled the replay at a
 * frame time. Same pixels, by construction.
 *
 * Court units come from the contract in lib/game/types.ts: 160 x 100, origin
 * top-left, y growing downward.
 *
 * DESIGNER: `COURT_SIZES` below is the court size table from design/SPEC.md,
 * one row per surface, in the reference pixels the artboards were drawn at.
 * `courtStyle()` scales the whole row by the width the court is actually given,
 * so a court is never a different shape, only a different size.
 *
 * Two things are NOT in that table, on purpose: how tall a paddle is and how
 * far in it stands. Both come from the contract (PADDLE_H, and LEFT_PLANE /
 * RIGHT_PLANE from PADDLE_INSET), because a paddle drawn taller than its hitbox
 * or offset from the plane the ball turns at is a lie — and it was: the ball
 * used to pass visibly through both paddles. Only thickness is styling.
 */

import { COURT_H, COURT_W, LEFT_PLANE, PADDLE_H, RIGHT_PLANE } from '../game/types';
import type { Box, ClipCtx } from './ctx';
import { drawText, type ClipFontRole } from './text';
import { CLIP_TOKENS, type ClipTokens, type ClipTrail } from './tokens';

// ---------------------------------------------------------------------------
// The size table
// ---------------------------------------------------------------------------

export type CourtSurface =
  | 'home-desktop'
  | 'home-phone'
  | 'play'
  | 'play-phone'
  | 'watch'
  | 'clip-wide'
  | 'clip-square'
  | 'clip-social'
  | 'clip-duel';

/** One row of design/SPEC.md "Court sizes", in reference pixels. */
export interface CourtSize {
  /** Reference size. Everything else is scaled by (actual width / this). */
  w: number;
  h: number;
  ballR: number;
  /**
   * Paddle THICKNESS only. Its height and its x both come from the contract
   * (PADDLE_H, LEFT_PLANE / RIGHT_PLANE), so the bar you see is the hitbox.
   */
  paddleW: number;
  /** Corner radius of the court. */
  radius: number;
  /** Centre line: dash pattern and how far it is held off the top and bottom. */
  dash: [number, number];
  lineInset: number;
  /** Trail length at `trail.referenceMs`, and how many discs draw it. */
  trailPx: number;
  discs: number;
  /** The score inside the court, at top centre. Null on surfaces without one. */
  score: { size: number; gap: number; baseline: number } | null;
}

export const COURT_SIZES: Record<CourtSurface, CourtSize> = {
  'home-desktop': {
    w: 704,
    h: 80,
    ballR: 4.5,
    paddleW: 4,
    radius: 6,
    dash: [3, 7],
    lineInset: 10,
    trailPx: 145,
    discs: 5,
    score: null,
  },
  'home-phone': {
    w: 358,
    h: 64,
    ballR: 4,
    paddleW: 3,
    radius: 6,
    dash: [3, 6],
    lineInset: 8,
    trailPx: 110,
    discs: 5,
    score: null,
  },
  play: {
    w: 704,
    h: 300,
    ballR: 7,
    paddleW: 6,
    radius: 8,
    dash: [3, 7],
    lineInset: 14,
    trailPx: 100,
    discs: 5,
    score: { size: 16, gap: 20, baseline: 34 },
  },
  /** Play on a phone: nearly square so a thumb has room to move the paddle. */
  'play-phone': {
    w: 358,
    h: 250,
    ballR: 6,
    paddleW: 5,
    radius: 8,
    dash: [3, 6],
    lineInset: 12,
    trailPx: 70,
    discs: 5,
    score: { size: 14, gap: 16, baseline: 28 },
  },
  watch: {
    w: 358,
    h: 200,
    ballR: 5,
    paddleW: 4,
    radius: 6,
    dash: [3, 6],
    lineInset: 12,
    trailPx: 55,
    discs: 5,
    score: { size: 14, gap: 14, baseline: 28 },
  },
  'clip-wide': {
    w: 640,
    h: 100,
    ballR: 7,
    paddleW: 5,
    radius: 6,
    dash: [3, 7],
    lineInset: 10,
    trailPx: 260,
    discs: 7,
    score: null,
  },
  'clip-square': {
    w: 992,
    h: 130,
    ballR: 8,
    paddleW: 6,
    radius: 6,
    dash: [3, 7],
    lineInset: 12,
    trailPx: 320,
    discs: 7,
    score: null,
  },
  'clip-social': {
    w: 984,
    h: 150,
    ballR: 13,
    paddleW: 9,
    radius: 8,
    dash: [4, 8],
    lineInset: 14,
    trailPx: 380,
    discs: 8,
    score: null,
  },
  'clip-duel': {
    w: 968,
    h: 320,
    ballR: 18,
    paddleW: 12,
    radius: 10,
    dash: [5, 10],
    lineInset: 16,
    trailPx: 440,
    discs: 9,
    score: null,
  },
};

/**
 * Which "home" court a given width gets. The home lane is one component that
 * stacks below the desktop grid, so the surface follows the measured width
 * rather than a media query — no hydration guessing, and it is right during a
 * resize as well.
 */
export function homeSurfaceFor(widthPx: number): CourtSurface {
  return widthPx >= 520 ? 'home-desktop' : 'home-phone';
}

/** Play switches to the phone court below the same breakpoint as home. */
export function playSurfaceFor(widthPx: number): CourtSurface {
  return widthPx >= 520 ? 'play' : 'play-phone';
}

// ---------------------------------------------------------------------------
// The look
// ---------------------------------------------------------------------------

/** The five court colours, from design/tokens.css. */
export interface CourtTheme {
  court: string;
  hair: string;
  courtLine: string;
  paddle: string;
  /** The ball and its trail. --accent in the first lane, --ball everywhere else. */
  ball: string;
  /** The score inside the court. */
  fgMuted: string;
}

/** The court's look in device pixels, ready to draw. */
export interface CourtStyle {
  bed: string;
  frame: string;
  centreLine: string;
  dash: [number, number];
  lineInset: number;
  radius: number;
  paddle: string;
  /** Thickness. The bar's height and x are derived by `drawCourt` itself. */
  paddleW: number;
  ball: string;
  ballR: number;
  trail: { spans: number[]; headAlpha: number; tailAlpha: number; tailScale: number };
  score: { size: number; gap: number; baseline: number; color: string; font: ClipFontRole } | null;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * How long the trail is, in the same pixels as `maxPx`.
 *
 * Motion note 03: `clamp(maxPx * 430 / latency, maxPx * 24/260, maxPx)`. A fast
 * lane streaks; a slow one barely smears, which is the whole read.
 */
export function trailLength(maxPx: number, latencyMs: number | null, trail: ClipTrail): number {
  const scaled =
    latencyMs !== null && latencyMs > 0 ? (maxPx * trail.referenceMs) / latencyMs : maxPx;
  return clamp(scaled, maxPx * trail.minFraction, maxPx);
}

/**
 * Cumulative distances behind the ball for each trail disc, nearest first.
 *
 * Spacing grows by `growth` per disc and the last disc lands exactly at
 * `length`, so the trail is always the length motion note 03 asks for however
 * many discs a surface draws.
 */
export function trailSpans(length: number, discs: number, growth: number): number[] {
  if (discs <= 0 || length <= 0) return [];
  let total = 0;
  for (let i = 0; i < discs; i += 1) total += Math.pow(growth, i);
  const first = length / total;
  const out: number[] = [];
  let acc = 0;
  let gap = first;
  for (let i = 0; i < discs; i += 1) {
    acc += gap;
    out.push(acc);
    gap *= growth;
  }
  return out;
}

/**
 * Build the court look for one surface at the width it is actually drawn at.
 *
 * `latencyMs` is the decision the ball is currently travelling on: it sets the
 * trail length, so the trail is a reading of speed rather than decoration.
 */
export function courtStyle(
  surface: CourtSurface,
  widthPx: number,
  theme: CourtTheme,
  latencyMs: number | null,
  tokens: ClipTokens = CLIP_TOKENS,
): CourtStyle {
  const size = COURT_SIZES[surface];
  const k = size.w > 0 ? widthPx / size.w : 1;
  const trail = tokens.trail;
  const spanPx = trailLength(size.trailPx * k, latencyMs, trail);

  return {
    bed: theme.court,
    frame: theme.hair,
    centreLine: theme.courtLine,
    dash: [size.dash[0] * k, size.dash[1] * k],
    lineInset: size.lineInset * k,
    radius: size.radius * k,
    paddle: theme.paddle,
    paddleW: size.paddleW * k,
    ball: theme.ball,
    ballR: size.ballR * k,
    trail: {
      // In COURT units, because the view's trail is in court units.
      spans: trailSpans(spanPx, size.discs, trail.growth).map((px) => (px / widthPx) * COURT_W),
      headAlpha: trail.headAlpha,
      tailAlpha: trail.tailAlpha,
      tailScale: trail.tailScale,
    },
    score: size.score
      ? {
          size: size.score.size * k,
          gap: size.score.gap * k,
          baseline: size.score.baseline * k,
          color: theme.fgMuted,
          font: tokens.fonts.mono,
        }
      : null,
  };
}

// ---------------------------------------------------------------------------
// The drawing
// ---------------------------------------------------------------------------

/** Everything the court needs to know about where things are, in court units. */
export interface CourtView {
  ballX: number;
  ballY: number;
  leftY: number;
  rightY: number;
  /**
   * Trail disc positions, NEAREST THE BALL FIRST. One per `style.trail.spans`
   * entry; a shorter list simply draws a shorter trail (the start of a rally).
   */
  trail: Array<{ x: number; y: number }>;
  score: [number, number];
  /** 0..1 ball opacity, for the fade-in on a serve. Defaults to 1. */
  ballAlpha?: number;
  /** True after a conceded point, when the ball is parked off-play. */
  ballHidden?: boolean;
}

function roundRectPath(
  ctx: ClipCtx,
  x: number,
  y: number,
  w: number,
  h: number,
  radius: number,
): void {
  const r = Math.max(0, Math.min(radius, w / 2, h / 2));
  ctx.beginPath();
  if (r > 0 && typeof ctx.roundRect === 'function') {
    ctx.roundRect(x, y, w, h, r);
    return;
  }
  ctx.moveTo(x, y);
  ctx.lineTo(x + w, y);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x, y + h);
  ctx.closePath();
}

/**
 * Paint one court into `box`.
 *
 * Nothing is clipped: `box` is assumed to be the court's own rectangle and
 * every coordinate is derived from it, so two courts never overlap.
 */
export function drawCourt(
  ctx: ClipCtx,
  view: CourtView,
  box: Box,
  style: CourtStyle,
): void {
  const { x: ox, y: oy, w, h } = box;
  if (w <= 1 || h <= 1) return;

  const sx = w / COURT_W;
  const sy = h / COURT_H;

  // --- bed + hairline frame ------------------------------------------------
  ctx.save();
  roundRectPath(ctx, ox + 0.5, oy + 0.5, w - 1, h - 1, style.radius);
  ctx.fillStyle = style.bed;
  ctx.fill();
  ctx.strokeStyle = style.frame;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();

  // --- centre line ---------------------------------------------------------
  ctx.save();
  ctx.strokeStyle = style.centreLine;
  ctx.lineWidth = 1;
  ctx.setLineDash([style.dash[0], style.dash[1]]);
  ctx.beginPath();
  const midX = ox + Math.round(w / 2) + 0.5;
  ctx.moveTo(midX, oy + style.lineInset);
  ctx.lineTo(midX, oy + h - style.lineInset);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();

  // --- score, either side of the centre line -------------------------------
  if (style.score) {
    const s = style.score;
    ctx.save();
    ctx.fillStyle = s.color;
    drawText(ctx, s.font, s.size, String(view.score[0]), midX - s.gap, oy + s.baseline, 'right');
    drawText(ctx, s.font, s.size, String(view.score[1]), midX + s.gap, oy + s.baseline, 'left');
    ctx.restore();
  }

  // --- paddles -------------------------------------------------------------
  // THE BAR IS THE HITBOX. Its height is PADDLE_H in court units, and its inner
  // face sits exactly on the plane the engine turns the ball at (LEFT_PLANE /
  // RIGHT_PLANE), so what you see is what the ball hits. A paddle grows OUTWARD
  // from its plane toward the court edge; only its thickness is a styling
  // choice.
  ctx.save();
  ctx.fillStyle = style.paddle;
  const paddleH = PADDLE_H * sy;
  const pr = style.paddleW / 2;
  roundRectPath(
    ctx,
    ox + LEFT_PLANE * sx - style.paddleW,
    oy + view.leftY * sy - paddleH / 2,
    style.paddleW,
    paddleH,
    pr,
  );
  ctx.fill();
  roundRectPath(
    ctx,
    ox + RIGHT_PLANE * sx,
    oy + view.rightY * sy - paddleH / 2,
    style.paddleW,
    paddleH,
    pr,
  );
  ctx.fill();
  ctx.restore();

  // --- trail: nearest the ball is the biggest and brightest -----------------
  const discs = view.trail.length;
  if (discs > 0) {
    const t = style.trail;
    const total = style.trail.spans.length || discs;
    ctx.save();
    ctx.fillStyle = style.ball;
    for (let i = 0; i < discs; i += 1) {
      // u runs 0 at the disc next to the ball to 1 at the far end.
      const u = total <= 1 ? 0 : i / (total - 1);
      ctx.globalAlpha = t.headAlpha + (t.tailAlpha - t.headAlpha) * u;
      // Radius falls from the ball's own r to tailScale * r across the trail.
      const fall = (i + 1) / (total + 1);
      ctx.beginPath();
      ctx.arc(
        ox + view.trail[i].x * sx,
        oy + view.trail[i].y * sy,
        style.ballR * (1 - (1 - t.tailScale) * fall),
        0,
        Math.PI * 2,
      );
      ctx.fill();
    }
    ctx.restore();
  }

  // --- ball ----------------------------------------------------------------
  if (!view.ballHidden) {
    ctx.save();
    ctx.globalAlpha = view.ballAlpha ?? 1;
    ctx.fillStyle = style.ball;
    ctx.beginPath();
    ctx.arc(ox + view.ballX * sx, oy + view.ballY * sy, style.ballR, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}
