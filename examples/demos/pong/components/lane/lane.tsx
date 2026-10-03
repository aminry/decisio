'use client';

/**
 * One lane row.
 *
 *   desktop   176px name | court | 200px number
 *   phone     name row over a full-width court
 *
 * The number is the biggest thing in the row at every width, the name drops to
 * a second line before the number ever shrinks, and the only colour in the row
 * is the first lane's: ball, trail and number (design/SPEC.md, lane accents).
 */

import type { ReactNode } from 'react';
import type { LaneConfig, Snapshot } from '@/lib/game/types';
import { Court, type CourtSurfaceProp } from './court';
import { LaneNumber, type NumberScale } from './number';
import { useLaneNumbers } from './use-lane';

export interface LaneProps {
  config: LaneConfig;
  snapshot: Snapshot | null;
  surface?: CourtSurfaceProp;
  scale?: NumberScale;
  /** A recorded run tweens a step over its full latency; a live one caps it. */
  recorded?: boolean;
  /** Fades in beside the name. Never pushes layout (motion note 05). */
  badge?: ReactNode;
  onPointerY?: (y: number | null) => void;
  className?: string;
}

export function Lane({
  config,
  snapshot,
  surface = 'home',
  scale = 'lane',
  recorded = false,
  badge,
  onPointerY,
  className,
}: LaneProps) {
  const numbers = useLaneNumbers(snapshot);
  const score = snapshot?.score ?? [0, 0];

  return (
    <article className={`lane-row ${className ?? ''}`}>
      <div className="lane-name flex min-w-0 flex-col gap-[3px] lg:gap-1">
        <div className="flex min-w-0 items-center gap-2">
          <h3 className="truncate text-[16px] leading-[1.1] font-semibold lg:text-[18px]">
            {config.label}
          </h3>
          {badge}
        </div>
        <p className="tag text-[11px] lg:text-[11.5px]">
          {config.provider}
        </p>
      </div>

      <div className="lane-court min-w-0">
        <Court
          snapshot={snapshot}
          model={config.id}
          surface={surface}
          recorded={recorded}
          onPointerY={onPointerY}
          ariaLabel={`${config.label} court, score ${score[0]} to ${score[1]}`}
        />
      </div>

      <LaneNumber
        live={!recorded}
        model={config.id}
        numbers={numbers}
        scale={scale}
        className="lane-number"
      />
    </article>
  );
}

export default Lane;
