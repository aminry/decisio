'use client';

/**
 * The front page. The claim, then the lanes, then the numbers behind both.
 *
 * It replays public/replay.json, a recording of a real run, through the same renderer a live game would use. The
 * lanes are whatever the recording holds (written by `pnpm record`); the headline and the results strip are computed
 * from the recording, so no number on the page was typed by hand.
 *
 * The lanes are independent runners, and the recordings do not end on the same millisecond, so nothing here loops on
 * its own: the page restarts all of them together one period after they started (lib/ui/replay-loop.ts) and resets
 * the clock with them.
 *
 * Changed from upstream (ably-labs/jev-pong, Apache-2.0): the fixed four-lane list, the live-play, arena and
 * explainer pages, and the sponsor copy are gone.
 *
 *   ?clean=1  strips everything except the lanes (screen recording)
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { TAGLINE, WINDOW_SECONDS } from '@/lib/config/site';
import { createReplayRunner } from '@/lib/game/replay';
import type { LaneConfig } from '@/lib/game/types';
import { replayPeriodMs } from '@/lib/ui/replay-loop';
import { headlineSentences, laneStatsFor, type LaneStat } from '@/lib/ui/stats';
import { SiteFooter, SiteHeader } from './chrome';
import { Lanes } from './lanes';
import { ReplayTimer, useReplayClock } from './replay-timer';
import { useLaneSources, type LaneSource } from './use-lane';
import { useReplay } from './use-replay';

const REPLAY_URL = '/replay.json';

export interface HomeProps {
  clean: boolean;
}

export function Home({ clean }: HomeProps) {
  const { replay, state } = useReplay(REPLAY_URL);
  const rows = useMemo(() => (replay ? laneStatsFor(replay) : []), [replay]);
  // The lanes the recording holds, in its order. They are also what the page draws while it waits.
  const configs = useMemo<LaneConfig[]>(
    () =>
      (replay?.lanes ?? []).map((l) => ({
        id: l.model as string,
        label: l.label,
        provider: l.provider ?? '',
        kind: l.kind ?? 'systemone',
        color: l.color,
      })),
    [replay],
  );
  const clock = useReplayClock();

  // One clock for all four lanes. Each runner plays its own recording once; the
  // page deals a fresh set every `period`, which is the only thing that makes
  // the four restart on the same frame.
  const period = useMemo(() => replayPeriodMs(replay), [replay]);
  const [pass, setPass] = useState(0);

  useEffect(() => {
    if (period === null) return;
    const timer = setInterval(() => setPass((n) => n + 1), period);
    return () => clearInterval(timer);
  }, [period]);

  const { reset } = clock;
  useEffect(() => {
    reset();
  }, [pass, reset]);

  const make = useCallback((): LaneSource[] => {
    if (!replay) return [];
    const sources: LaneSource[] = [];
    // Lane order is the recording's and is never re-sorted.
    configs.forEach((config, index) => {
      sources.push({ config, runner: createReplayRunner(replay, index, { loop: false }) });
    });
    return sources;
  }, [replay, configs]);

  const lanes = useLaneSources(make, `${replay?.recordedAt ?? 'pending'}#${pass}`);
  const stack = (
    <Lanes lanes={lanes} surface="home" recorded replay={replay} placeholder={configs} onTime={clock.note} />
  );

  if (state === 'error') {
    return (
      <main className="mx-auto w-full max-w-[1280px] px-4 py-10 lg:px-20">
        <p className="text-fg-muted text-[14px]">
          No recording yet. Start the System One server and run <code className="mono">pnpm record</code>; it writes
          public/replay.json.
        </p>
      </main>
    );
  }

  // Recording mode: the lanes and nothing else.
  if (clean) {
    return (
      <main className="mx-auto flex w-full max-w-[1280px] flex-1 flex-col justify-center px-4 py-6 lg:px-20">
        {stack}
      </main>
    );
  }

  return (
    <main className="mx-auto w-full max-w-[1280px] px-4 pb-6 lg:px-20">
      <SiteHeader />

      <Headline rows={rows} />

      <div className="flex items-baseline justify-between gap-4 pt-4 lg:pt-6">
        <p className="text-fg-muted text-[11.5px] lg:text-[13px]">
          Recorded run{replay ? ` · ${new Date(replay.recordedAt).toISOString().slice(0, 10)}` : ''} · seed{' '}
          {replay?.seed ?? ''}
        </p>
        <ReplayTimer clock={clock} />
      </div>

      <section aria-label="Model lanes" className="pt-2.5 lg:pt-3">
        {stack}
      </section>

      <Results rows={rows} />

      <div className="pt-9 lg:pt-14">
        <SiteFooter />
      </div>
    </main>
  );
}

/* ---------------------------------------------------------------- the claim */

function Headline({ rows }: { rows: LaneStat[] }) {
  const h = headlineSentences(rows);
  if (!h.lead) return null;
  const [value, unit] = h.leadMs.split(' ');
  return (
    <div className="flex flex-col gap-2 pt-5 lg:gap-2.5 lg:pt-9">
      <h1 className="max-w-[980px] text-[21px] leading-[1.22] font-bold tracking-[-0.025em] text-pretty lg:text-[33px] lg:leading-[1.14]">
        {h.lead}{' '}
        <span className="text-accent-ink whitespace-nowrap">
          <span className="mono tracking-[-0.04em]">{value}</span>
          <span className="pl-[0.22em]">{unit}</span>
        </span>
        . {h.others}
      </h1>
      <p className="text-fg-muted text-[14px] leading-[1.4] lg:text-[16px]">{TAGLINE}</p>
    </div>
  );
}

/* -------------------------------------------------------- the results strip */

const COLUMNS: Array<{ key: keyof Pick<LaneStat, 'decisionsPerSec' | 'avgMs' | 'p95Ms' | 'inWindow'>; label: string }> = [
  { key: 'decisionsPerSec', label: 'decisions/s' },
  { key: 'avgMs', label: 'avg ms' },
  { key: 'p95Ms', label: 'p95 ms' },
  { key: 'inWindow', label: `in first ${WINDOW_SECONDS} s` },
];

function cellText(row: LaneStat, key: (typeof COLUMNS)[number]['key']): string {
  const value = row[key];
  return key === 'decisionsPerSec' ? value.toFixed(2) : String(Math.round(value));
}

/** The same lanes as numbers: what the courts above just showed, in a form you can quote. */
function Results({ rows }: { rows: LaneStat[] }) {
  if (rows.length === 0) return null;
  return (
    <section aria-label="Measured results" className="pt-6 lg:pt-8">
      <div className="results-row border-hair border-b pb-1.5">
        <span className="results-name tag hidden lg:block">model</span>
        {COLUMNS.map((column) => (
          <span key={column.key} className="tag text-right text-[10px] lg:text-[12px]">
            {column.label}
          </span>
        ))}
      </div>

      <div className="flex flex-col">
        {rows.map((row) => {
          const ink = row.model === 'sys1' ? 'text-accent-ink' : 'text-fg';
          return (
            <div key={row.model} className="results-row border-hair border-b py-2.5 lg:py-3">
              <div className="results-name flex min-w-0 flex-col gap-[2px]">
                <span className="truncate text-[14px] leading-[1.15] font-semibold lg:text-[15px]">
                  {row.label}
                </span>
                <span className="tag text-[10.5px] lg:text-[11px]">{row.provider}</span>
              </div>
              {COLUMNS.map((column) => (
                <span
                  key={column.key}
                  className={`mono ${ink} text-right text-[17px] leading-none font-medium tracking-[-0.02em] lg:text-[20px]`}
                >
                  {cellText(row, column.key)}
                </span>
              ))}
            </div>
          );
        })}
      </div>
    </section>
  );
}

export default Home;
