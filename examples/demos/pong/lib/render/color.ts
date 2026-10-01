/**
 * Tiny colour maths for the renderers.
 *
 * The clip is drawn with the 2D API subset that the browser canvas and
 * @napi-rs/canvas both support, which has no colour-mix primitive. So any
 * blend (the latency pulse tinting the number toward the lane accent, a
 * translucent scrim on the end card) is computed here and handed to the
 * context as a finished colour string.
 */

/** #rgb / #rrggbb / #rrggbbaa -> [r, g, b]. Anything else comes back as black. */
export function parseHex(hex: string): [number, number, number] {
  const h = hex.trim().replace('#', '');
  // Without this guard an 8-character word ("nonsense") parses to NaN and
  // poisons every colour derived from it.
  if (!/^(?:[0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})$/i.test(h)) return [0, 0, 0];
  if (h.length === 3) {
    const r = Number.parseInt(h[0] + h[0], 16);
    const g = Number.parseInt(h[1] + h[1], 16);
    const b = Number.parseInt(h[2] + h[2], 16);
    return [r, g, b];
  }
  return [
    Number.parseInt(h.slice(0, 2), 16),
    Number.parseInt(h.slice(2, 4), 16),
    Number.parseInt(h.slice(4, 6), 16),
  ];
}

/** Linear blend from `a` to `b`. t = 0 is all `a`, t = 1 is all `b`. */
export function mixHex(a: string, b: string, t: number): string {
  const k = t < 0 ? 0 : t > 1 ? 1 : t;
  const [ar, ag, ab] = parseHex(a);
  const [br, bg, bb] = parseHex(b);
  const to2 = (n: number) => Math.round(n).toString(16).padStart(2, '0');
  return `#${to2(ar + (br - ar) * k)}${to2(ag + (bg - ag) * k)}${to2(ab + (bb - ab) * k)}`;
}

/** Same colour, given an alpha. Returns an `rgba()` string both backends parse. */
export function withAlpha(hex: string, alpha: number): string {
  const [r, g, b] = parseHex(hex);
  const a = alpha < 0 ? 0 : alpha > 1 ? 1 : alpha;
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

/* ------------------------------------------------------- contrast (WCAG 2) */

function channel(v: number): number {
  const s = v / 255;
  return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
}

/** WCAG relative luminance, 0 (black) to 1 (white). */
export function relativeLuminance(hex: string): number {
  const [r, g, b] = parseHex(hex);
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

/** WCAG contrast ratio, 1 (identical) to 21 (black on white). */
export function contrastRatio(a: string, b: string): number {
  const la = relativeLuminance(a);
  const lb = relativeLuminance(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

