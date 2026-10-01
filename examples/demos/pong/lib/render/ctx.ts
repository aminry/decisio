/**
 * The 2D drawing surface the renderers are allowed to use.
 *
 * The same drawing code runs in two places: the browser's
 * `CanvasRenderingContext2D` (components/lane/court.tsx) and
 * `@napi-rs/canvas` (scripts/render-clip.ts). This interface is the
 * intersection of the two, so a drawing function that type-checks against it
 * is guaranteed to run on both. Both real contexts satisfy it structurally —
 * no casts anywhere.
 *
 * Optional members are the flourishes. They exist on both backends today, but
 * they are declared optional so the drawing code has to check before calling,
 * and so a minimal stub context (a test double, a future SVG backend) still
 * satisfies the interface.
 *
 * Paint properties are `unknown` on purpose: the DOM types them as
 * `string | CanvasGradient | CanvasPattern` and @napi-rs/canvas uses its own
 * gradient and pattern classes. `unknown` is the only type both are assignable
 * to. Everything this codebase assigns to them is a colour string or a
 * gradient made by `createRadialGradient`.
 */

export interface ClipGradient {
  addColorStop(offset: number, color: string): void;
}

export interface ClipTextMetrics {
  width: number;
}

export interface ClipCtx {
  // --- state ---------------------------------------------------------------
  save(): void;
  restore(): void;

  // --- paint ---------------------------------------------------------------
  fillStyle: unknown;
  strokeStyle: unknown;
  globalAlpha: number;
  lineWidth: number;
  shadowColor: string;
  shadowBlur: number;

  // --- text ----------------------------------------------------------------
  font: string;
  textAlign: string;
  textBaseline: string;
  fillText(text: string, x: number, y: number): void;
  measureText(text: string): ClipTextMetrics;

  // --- shapes --------------------------------------------------------------
  fillRect(x: number, y: number, w: number, h: number): void;
  strokeRect(x: number, y: number, w: number, h: number): void;
  beginPath(): void;
  closePath(): void;
  moveTo(x: number, y: number): void;
  lineTo(x: number, y: number): void;
  arc(x: number, y: number, r: number, start: number, end: number): void;
  fill(): void;
  stroke(): void;
  setLineDash(segments: number[]): void;

  // --- optional flourishes -------------------------------------------------
  roundRect?(x: number, y: number, w: number, h: number, radii?: number): void;
  createRadialGradient?(
    x0: number,
    y0: number,
    r0: number,
    x1: number,
    y1: number,
    r1: number,
  ): ClipGradient;
}

/** A rectangle in device pixels. Every draw function is given one to fill. */
export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}
