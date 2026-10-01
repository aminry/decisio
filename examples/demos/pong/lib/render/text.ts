/**
 * Text helpers shared by the court, the frame renderer and the site.
 *
 * Canvas has no letter-spacing in the API subset both backends support, and
 * the design leans on it (provider tags are uppercase at +0.04em, every number
 * is at -0.02em). So tracked text is drawn one glyph at a time here, and
 * measured the same way, which keeps alignment honest.
 */

import type { ClipCtx } from './ctx';

/** A resolved text style: a CSS font-family list, a weight and optional tracking. */
export interface ClipFontRole {
  family: string;
  weight: number;
  /** Extra tracking as a fraction of the font size, e.g. -0.02 or 0.04. */
  tracking?: number;
  /** ALL CAPS, like the provider tags. */
  uppercase?: boolean;
}

/** `"500 16px <stack>"` — the CSS shorthand both backends parse. */
export function fontString(role: ClipFontRole, sizePx: number): string {
  return `${role.weight} ${sizePx}px ${role.family}`;
}

/** Applies the role's `uppercase` flag. Tracking is applied when drawing. */
export function applyCase(role: ClipFontRole, text: string): string {
  return role.uppercase ? text.toUpperCase() : text;
}

/**
 * Width of `text` in `role` at `sizePx`, including tracking. The trailing
 * letter's tracking is not counted, so a right-aligned string sits flush.
 */
export function measureRole(
  ctx: ClipCtx,
  role: ClipFontRole,
  sizePx: number,
  text: string,
): number {
  const s = applyCase(role, text);
  ctx.font = fontString(role, sizePx);
  const base = ctx.measureText(s).width;
  const track = (role.tracking ?? 0) * sizePx;
  return base + track * Math.max(0, s.length - 1);
}

export type TextAlign = 'left' | 'right' | 'center';

/**
 * Draw `text` at (x, y) in `role`. `align` is applied here rather than through
 * ctx.textAlign, because tracked text has to be laid out glyph by glyph and
 * the two mechanisms do not agree on the advance of the last character.
 *
 * `y` is the text baseline. Returns the width that was drawn.
 */
export function drawText(
  ctx: ClipCtx,
  role: ClipFontRole,
  sizePx: number,
  text: string,
  x: number,
  y: number,
  align: TextAlign = 'left',
): number {
  const s = applyCase(role, text);
  const width = measureRole(ctx, role, sizePx, s);
  const startX = align === 'left' ? x : align === 'right' ? x - width : x - width / 2;

  ctx.font = fontString(role, sizePx);
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';

  const track = (role.tracking ?? 0) * sizePx;
  if (track === 0) {
    ctx.fillText(s, startX, y);
    return width;
  }

  let cursor = startX;
  for (const ch of s) {
    ctx.fillText(ch, cursor, y);
    cursor += ctx.measureText(ch).width + track;
  }
  return width;
}

/**
 * Shrink `sizePx` until the text fits `maxWidth`. Returns the size to use.
 * Lane labels are author-supplied, so a long one must not run into the court.
 */
export function fitSize(
  ctx: ClipCtx,
  role: ClipFontRole,
  sizePx: number,
  text: string,
  maxWidth: number,
  minPx = 8,
): number {
  let size = sizePx;
  while (size > minPx && measureRole(ctx, role, size, text) > maxWidth) {
    size -= 1;
  }
  return size;
}

/** Shorten with an ellipsis so it fits `maxWidth` at a fixed size. */
export function ellipsize(
  ctx: ClipCtx,
  role: ClipFontRole,
  sizePx: number,
  text: string,
  maxWidth: number,
): string {
  if (measureRole(ctx, role, sizePx, text) <= maxWidth) return text;
  let out = text;
  while (out.length > 1 && measureRole(ctx, role, sizePx, `${out}…`) > maxWidth) {
    out = out.slice(0, -1);
  }
  return `${out}…`;
}
