'use client';

/**
 * components/lane/court.tsx — the demo itself.
 *
 * Where the ball is at this instant is NOT this file's decision. It comes from
 * a sampler in lib/render, which is the same code the headless clip renderer
 * uses, so the page and the video agree by construction:
 *
 *   live      lib/render/live.ts. Snapshots arrive when the model answers and
 *             the network allows. The ball walks from the previous snapshot to
 *             the latest one over the interval the latest one actually took —
 *             so it is always one confirmed interval behind, never ahead, and
 *             when a decision is late it simply stops and waits. That IS the
 *             demonstration (motion note 02).
 *   recorded  lib/render/timeline.ts. Every frame and its time is known up
 *             front, so a lane is a pure function of one clock: linear inside a
 *             step, the scripted push-off on a serve (motion note 01). All four
 *             lanes read the SAME clock, from `<Lanes>` through
 *             `RecordedLaneContext`, which is what keeps them in phase and
 *             restarts them together.
 *
 * What the old tween did instead — start on arrival, sized by the PREVIOUS
 * decision's latency — is why the ball moved in lurches that had nothing to do
 * with the model.
 *
 * The one thing drawn from neither is the player's own paddle: it is predicted
 * locally (lib/ui/paddle.ts) so it answers the key instantly, and reconciled
 * toward the wire whenever no key is down, so it can never be stranded by a
 * lost input. Nothing reconciles it DURING a press — that was the shiver.
 *
 * WHEN a snapshot arrived is as much a part of the sampler's input as the
 * snapshot itself: it times the ball's step off it. So it is taken as early as
 * React allows (a layout effect, before paint) rather than somewhere down the
 * paint, and a caller that can read the true wire arrival time — the instant
 * the message arrived — passes it as `arrivedAtMs` instead.
 *
 * The court's PIXELS come from `drawCourt` in lib/render/court.ts. This file
 * owns the loop, the sizing and the pointer; it does not own the look.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
} from 'react';
import { COURT_H, COURT_W, type Move, type Snapshot } from '@/lib/game/types';
import {
  COURT_SIZES,
  courtStyle,
  drawCourt,
  homeSurfaceFor, playSurfaceFor,
  type CourtSurface,
  type CourtView,
} from '@/lib/render/court';
import { createLiveSampler, type LiveSampler } from '@/lib/render/live';
import type { LaneTimeline } from '@/lib/render/timeline';
import { CLIP_TOKENS } from '@/lib/render/tokens';
import { predictPaddle } from '@/lib/ui/paddle';
import { readCourtTheme } from './theme';

/** A frame longer than this is a tab coming back, not a slow machine. */
const MAX_FRAME_MS = 100;

/** Which court size table row a surface prop resolves to at this width. */
export type CourtSurfaceProp = CourtSurface | 'home';

function resolveSurface(surface: CourtSurfaceProp, widthPx: number): CourtSurface {
  if (surface === 'home') return homeSurfaceFor(widthPx);
  if (surface === 'play') return playSurfaceFor(widthPx);
  return surface;
}

/**
 * A recorded lane and the clock every lane on the page is sampled at.
 *
 * `<Lanes>` provides it — it is the only thing that knows both the recording
 * and when the current pass started — and a recorded `<Court>` reads it. The
 * clock is shared deliberately: four lanes sampled at four slightly different
 * instants are not "same serve, same rules", whatever the copy says.
 */
export interface RecordedLaneSource {
  timeline: LaneTimeline;
  /** Milliseconds into the recording, right now. */
  clockMs: () => number;
}

export const RecordedLaneContext = createContext<RecordedLaneSource | null>(null);

/** Shown before there is anything to draw: an empty, still court. */
const EMPTY_VIEW: CourtView = {
  ballX: COURT_W / 2,
  ballY: COURT_H / 2,
  leftY: COURT_H / 2,
  rightY: COURT_H / 2,
  trail: [],
  score: [0, 0],
  ballHidden: true,
};

export interface CourtProps {
  snapshot: Snapshot | null;
  /** Picks the ball colour: the first lane's is `--accent`, the others' are `--ball`. */
  model: string;
  surface: CourtSurfaceProp;
  /**
   * This lane is playing a recording, so the ball comes from the shared
   * timeline rather than from snapshot arrivals. Snapshots still arrive, and
   * still drive the counters and the latency number.
   */
  recorded?: boolean;
  /**
   * The side a human is playing on this court. That paddle is predicted here
   * rather than drawn from the wire — see lib/ui/paddle.ts for why.
   */
  humanSide?: 'left' | 'right';
  /** The direction being asked for on this frame, from `usePaddleInput`. */
  held?: RefObject<Move>;
  /**
   * The `seq` of the newest input this player has sent, from `usePlayerInput`.
   * The predicted paddle is reconciled against a frame only once that frame's
   * `inputSeq` has caught up with it — see lib/ui/paddle.ts.
   */
  sentSeq?: RefObject<number>;
  /**
   * When `snapshot` came off the wire, on the `performance.now()` clock — the
   * instant the message arrived. The sampler measures the ball's
   * step from it, so the closer it is to the truth the steadier the ball.
   * Omitted, it is read in a layout effect, which is the earliest this
   * component can see the snapshot for itself.
   */
  arrivedAtMs?: number;
  /** Draw the score inside the court. Off where a scoreboard already shows it. */
  showScore?: boolean;
  /** Court-space y (0..COURT_H) while a pointer is down, null on release. */
  onPointerY?: (y: number | null) => void;
  className?: string;
  ariaLabel?: string;
}

/**
 * `useLayoutEffect`, except on the server, where React would warn about it and
 * it would do nothing anyway. The ingest needs the layout timing: it is reading
 * a clock, and the number it reads is what the ball's motion is measured over.
 */
const useIngestEffect = typeof window === 'undefined' ? useEffect : useLayoutEffect;

/** The latest snapshot, drawn as it is: what reduced motion asks for (08). */
function stillView(snapshot: Snapshot | null): CourtView {
  if (snapshot === null) return EMPTY_VIEW;
  return {
    ballX: snapshot.ball.x,
    ballY: snapshot.ball.y,
    leftY: snapshot.leftY,
    rightY: snapshot.rightY,
    trail: [],
    score: snapshot.score,
    ballHidden: snapshot.ball.vx === 0,
  };
}

export function Court({
  snapshot,
  model,
  surface,
  recorded = false,
  humanSide,
  held,
  sentSeq,
  arrivedAtMs,
  showScore = true,
  onPointerY,
  className,
  ariaLabel,
}: CourtProps) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // A recorded lane is driven by the page's clock, not by arrivals.
  const fromContext = useContext(RecordedLaneContext);
  const recordedSource = recorded ? fromContext : null;

  // --- render state, all in refs: the rAF loop must not re-subscribe ---------
  const sizeRef = useRef({ w: 0, h: 0, dpr: 1 });
  const snapRef = useRef<Snapshot | null>(null);
  /** The last latency that actually landed. Heartbeat frames carry none, and
   *  a trail that jumped to full length on one would be reading nothing. */
  const latencyRef = useRef<number | null>(null);
  const themeRef = useRef(readCourtTheme(model));
  const surfaceRef = useRef<CourtSurfaceProp>(surface);
  const showScoreRef = useRef(showScore);
  const recordedRef = useRef<RecordedLaneSource | null>(recordedSource);
  const reducedRef = useRef(false);
  /** Court y under the pointer while it is down. A position, not a direction. */
  const pointerRef = useRef<number | null>(null);
  /** The locally predicted human paddle. Null until there is a wire to start from. */
  const predictedRef = useRef<number | null>(null);
  const humanRef = useRef<'left' | 'right' | undefined>(humanSide);
  const heldRef = useRef<RefObject<Move> | undefined>(held);
  const sentSeqRef = useRef<RefObject<number> | undefined>(sentSeq);

  // One sampler for the life of this court: it keeps the last two snapshots and
  // the times they arrived, so rebuilding it would lose the step in progress.
  // `humanSide` belongs to the court, not to a render — /play mounts a new one
  // per game (keyed by game id) — so reading it once here is the whole story.
  const samplerRef = useRef<LiveSampler | null>(null);
  if (samplerRef.current === null) {
    samplerRef.current = createLiveSampler({ humanSide: humanSide ?? null });
  }

  useEffect(() => {
    themeRef.current = readCourtTheme(model);
    surfaceRef.current = surface;
    showScoreRef.current = showScore;
    recordedRef.current = recordedSource;
    humanRef.current = humanSide;
    heldRef.current = held;
    sentSeqRef.current = sentSeq;
  }, [model, surface, showScore, recordedSource, humanSide, held, sentSeq]);

  // --- ingest a new snapshot -------------------------------------------------
  // A LAYOUT effect, and the clock is read on its first line: this is the
  // earliest point after the snapshot reaches React, and it is before paint,
  // where a passive effect would be a whole frame behind it. The sampler times
  // the ball's step off this number.
  useIngestEffect(() => {
    if (!snapshot) return;
    const arrived =
      arrivedAtMs ?? (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const prev = snapRef.current;
    // Paddle and heartbeat frames repeat the tick, never the lane clock, so
    // this drops re-renders and nothing else. The sampler needs every frame:
    // the paddles move on the ones the ball does not.
    if (prev && prev.tick === snapshot.tick && prev.t === snapshot.t) return;
    snapRef.current = snapshot;
    if (snapshot.latencyMs !== null && Number.isFinite(snapshot.latencyMs)) {
      latencyRef.current = snapshot.latencyMs;
    }
    samplerRef.current?.push(snapshot, arrived);
  }, [snapshot, arrivedAtMs]);

  // --- sizing ----------------------------------------------------------------
  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;

    const motion =
      typeof matchMedia === 'function' ? matchMedia('(prefers-reduced-motion: reduce)') : null;
    const dark =
      typeof matchMedia === 'function' ? matchMedia('(prefers-color-scheme: dark)') : null;
    reducedRef.current = !!motion?.matches;

    const onMotion = () => {
      reducedRef.current = !!motion?.matches;
    };
    const onScheme = () => {
      themeRef.current = readCourtTheme(model);
    };
    motion?.addEventListener('change', onMotion);
    dark?.addEventListener('change', onScheme);

    const measure = () => {
      const rect = wrap.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(rect.width));
      const h = Math.max(1, Math.round(rect.height));
      if (sizeRef.current.w === w && sizeRef.current.h === h && sizeRef.current.dpr === dpr) {
        return;
      }
      sizeRef.current = { w, h, dpr };
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    };

    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(wrap);
    return () => {
      ro.disconnect();
      motion?.removeEventListener('change', onMotion);
      dark?.removeEventListener('change', onScheme);
    };
  }, [model]);

  // --- the loop --------------------------------------------------------------
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let raf = 0;
    let lastFrameAt = 0;

    const draw = (nowMs: number) => {
      raf = requestAnimationFrame(draw);
      const { w, h, dpr } = sizeRef.current;
      if (w <= 1 || h <= 1) return;

      const dtMs = lastFrameAt === 0 ? 0 : Math.min(nowMs - lastFrameAt, MAX_FRAME_MS);
      lastFrameAt = nowMs;

      const reduced = reducedRef.current;
      const recording = recordedRef.current;
      const tMs = recording ? recording.clockMs() : 0;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const box = { x: 0, y: 0, w, h };
      // The trail is a reading of speed, so its length is set by the decision
      // the ball is travelling on — which has to be known before the view is
      // sampled, because the spans are what the sample fills in.
      const latencyMs = recording ? recording.timeline.numberAt(tMs).value : latencyRef.current;
      const style = courtStyle(
        resolveSurface(surfaceRef.current, w),
        w,
        themeRef.current,
        latencyMs,
        CLIP_TOKENS,
      );
      if (!showScoreRef.current) style.score = null;
      const spans = reduced ? [] : style.trail.spans;

      // 08: with reduced motion a live lane simply shows the latest snapshot.
      // A recording still moves, because the movement IS the recording.
      const sampled: CourtView | null = recording
        ? recording.timeline.viewAt(tMs, spans)
        : reduced
          ? stillView(snapRef.current)
          : (samplerRef.current?.viewAt(nowMs, spans) ?? null);

      if (sampled === null) {
        drawCourt(ctx, EMPTY_VIEW, box, style);
        return;
      }

      // --- the player's own paddle ------------------------------------------
      // The sampler hands back the wire position exactly for the human's side.
      // That is the truth, and it is also 150-200ms old, so what is DRAWN is
      // the local prediction: instant on the key, pulled back to the wire every
      // frame (lib/ui/paddle.ts).
      const human = humanRef.current;
      let view = sampled;
      if (human !== undefined) {
        const wireY = human === 'left' ? sampled.leftY : sampled.rightY;
        // Has the latest frame applied everything this player has sent? Before
        // the first input of a game there is nothing to wait for.
        const sent = sentSeqRef.current?.current ?? 0;
        const heard = snapRef.current?.inputSeq?.[human] ?? 0;
        const predicted = predictPaddle({
          current: predictedRef.current ?? wireY,
          wireY,
          held: heldRef.current?.current ?? 'stay',
          dtMs,
          pointerY: pointerRef.current,
          wireAcked: sent === 0 || heard >= sent,
        });
        predictedRef.current = predicted;
        view =
          human === 'left'
            ? { ...sampled, leftY: predicted }
            : { ...sampled, rightY: predicted };
      }

      drawCourt(ctx, view, box, style);
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  // --- pointer (play mode) ---------------------------------------------------
  const toCourtY = useCallback((clientY: number): number => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect || rect.height === 0) return COURT_H / 2;
    const p = (clientY - rect.top) / rect.height;
    return Math.max(0, Math.min(COURT_H, p * COURT_H));
  }, []);

  const interactive = !!onPointerY;
  const onDown = useCallback(
    (e: ReactPointerEvent<HTMLCanvasElement>) => {
      if (!onPointerY) return;
      e.currentTarget.setPointerCapture(e.pointerId);
      const y = toCourtY(e.clientY);
      pointerRef.current = y;
      onPointerY(y);
    },
    [onPointerY, toCourtY],
  );
  const onMove = useCallback(
    (e: ReactPointerEvent<HTMLCanvasElement>) => {
      if (!onPointerY || !e.currentTarget.hasPointerCapture(e.pointerId)) return;
      const y = toCourtY(e.clientY);
      pointerRef.current = y;
      onPointerY(y);
    },
    [onPointerY, toCourtY],
  );
  const onUp = useCallback(
    (e: ReactPointerEvent<HTMLCanvasElement>) => {
      if (!onPointerY) return;
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }
      pointerRef.current = null;
      onPointerY(null);
    },
    [onPointerY],
  );

  // The pointer is released with the court; a finger left "down" in a ref would
  // pin the predicted paddle where it last was.
  useEffect(() => {
    if (interactive) return;
    pointerRef.current = null;
  }, [interactive]);

  // The court keeps the shape its surface was drawn at and simply scales. The
  // home court is the exception: it is the desktop court in the desktop grid
  // and the phone court in the phone stack, switched by a container query so
  // the shape always agrees with the width the canvas measures.
  const home = surface === 'home';
  const size = COURT_SIZES[home ? 'home-desktop' : surface];

  return (
    <div className={`court-container ${className ?? ''}`} style={{ maxWidth: size.w }}>
      <div
        ref={wrapRef}
        className={home ? 'court-home w-full' : 'w-full'}
        style={home ? undefined : { aspectRatio: `${size.w} / ${size.h}` }}
      >
        <canvas
          ref={canvasRef}
          className="block h-full w-full"
          style={interactive ? { touchAction: 'none', cursor: 'ns-resize' } : undefined}
          role="img"
          aria-label={ariaLabel ?? 'Pong court'}
          onPointerDown={interactive ? onDown : undefined}
          onPointerMove={interactive ? onMove : undefined}
          onPointerUp={interactive ? onUp : undefined}
          onPointerCancel={interactive ? onUp : undefined}
        />
      </div>
    </div>
  );
}

export default Court;
