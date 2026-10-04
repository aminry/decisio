'use client';

/**
 * The stack. Starts every runner on mount, stops them on unmount, and feeds
 * each row its own snapshot stream.
 *
 * It also owns the ONE clock the recorded lanes are drawn from. A replayed lane
 * is not animated by the arrival of its snapshots — it is a pure function of
 * time (lib/render/timeline.ts), and every lane on the page is sampled at the
 * same instant from here. That is what "same serve, same rules" means on
 * screen: the four courts cannot drift apart, because there is nothing for them
 * to drift with. The clock starts when this set of runners starts, and the page
 * deals a fresh set once per pass (lib/ui/replay-loop.ts), so a loop restarts
 * the clock and the four lanes together.
 */

import { useCallback, useEffect, useMemo, useRef } from 'react';
import type { LaneConfig, Replay } from '@/lib/game/types';
import { createLaneTimeline } from '@/lib/render/timeline';
import { Lane } from './lane';
import { useLaneFeed, type LaneSource } from './use-lane';
import { RecordedLaneContext, type CourtSurfaceProp, type RecordedLaneSource } from './court';

export interface LanesProps {
  lanes: LaneSource[];
  surface?: CourtSurfaceProp;
  /** A recorded run draws its ball from the timeline below, not from arrivals. */
  recorded?: boolean;
  /** The recording being played, which is what the timelines are built from. */
  replay?: Replay | null;
  /** Reports the replay clock, so the page can show "replay 2.0 s". */
  onTime?: (tMs: number) => void;
  /**
   * Rows to draw while there is nothing to play — before the recording has
   * loaded, and in the frame between one pass and the next. Without them the
   * page would be the right height only some of the time.
   */
  placeholder?: LaneConfig[];
  className?: string;
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

export function Lanes({
  lanes,
  surface = 'home',
  recorded = false,
  replay = null,
  onTime,
  placeholder,
  className,
}: LanesProps) {
  const startedAt = useRef(0);

  useEffect(() => {
    startedAt.current = now();
    for (const lane of lanes) lane.runner.start();
    return () => {
      for (const lane of lanes) lane.runner.stop();
    };
  }, [lanes]);

  // Read by every recorded court on every animation frame, so it allocates
  // nothing and never changes identity.
  const clockMs = useCallback(() => (startedAt.current === 0 ? 0 : now() - startedAt.current), []);

  const empty = lanes.length === 0 && placeholder !== undefined;

  return (
    <div className={`flex flex-col gap-3.5 lg:gap-2.5 ${className ?? ''}`}>
      {empty
        ? placeholder.map((config) => (
            <Lane key={config.id} config={config} snapshot={null} surface={surface} />
          ))
        : lanes.map((lane, i) => (
            <LaneRow
              key={`${lane.config.id}-${i}`}
              source={lane}
              surface={surface}
              recorded={recorded}
              replay={replay}
              clockMs={clockMs}
              onTime={onTime}
            />
          ))}
    </div>
  );
}

function LaneRow({
  source,
  surface,
  recorded,
  replay,
  clockMs,
  onTime,
}: {
  source: LaneSource;
  surface: CourtSurfaceProp;
  recorded: boolean;
  replay: Replay | null;
  clockMs: () => number;
  onTime?: (tMs: number) => void;
}) {
  const { snapshot } = useLaneFeed(source.runner);
  const t = snapshot?.t;

  useEffect(() => {
    if (t !== undefined) onTime?.(t);
  }, [t, onTime]);

  // The recorded lane this row is playing. `createLaneTimeline` pre-solves
  // every step in the recording, so it is built once per recording rather than
  // once per pass — the clock above is the only thing a loop restarts.
  const lane = replay?.lanes.find((l) => l.model === source.config.id) ?? null;
  const recordedSource = useMemo<RecordedLaneSource | null>(
    () => (recorded && lane !== null ? { timeline: createLaneTimeline(lane), clockMs } : null),
    [recorded, lane, clockMs],
  );

  const row = (
    <Lane config={source.config} snapshot={snapshot} surface={surface} recorded={recorded} />
  );

  if (recordedSource === null) return row;
  return <RecordedLaneContext.Provider value={recordedSource}>{row}</RecordedLaneContext.Provider>;
}

export default Lanes;
