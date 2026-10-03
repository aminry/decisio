/**
 * The sub-tick ball path now lives in lib/render so the headless clip renderer
 * (scripts/render-clip.ts) and this React court draw the same motion.
 *
 * Kept as a re-export so existing imports of './path' keep working.
 */

export { pathAt, BALL_MIN_Y, BALL_MAX_Y, PADDLE_REACH, type Ball } from '@/lib/render/path';
