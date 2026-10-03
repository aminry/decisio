'use client';

import { useEffect, useState } from 'react';
import type { Replay } from '@/lib/game/types';

function isReplay(value: unknown): value is Replay {
  if (typeof value !== 'object' || value === null) return false;
  const r = value as Partial<Replay>;
  return (
    r.version === 1 &&
    Array.isArray(r.lanes) &&
    r.lanes.length > 0 &&
    r.lanes.every((l) => Array.isArray(l?.snapshots))
  );
}

export type ReplayState = 'loading' | 'ready' | 'error';

export interface ReplayResult {
  replay: Replay | null;
  state: ReplayState;
}

interface Loaded {
  url: string;
  replay: Replay | null;
  state: ReplayState;
}

/** Load a recorded run. `url` of null means "do not load" (live mode). */
export function useReplay(url: string | null): ReplayResult {
  const [loaded, setLoaded] = useState<Loaded | null>(null);

  useEffect(() => {
    if (url === null) return;
    let live = true;
    // 'default' revalidates against the server (    // must-revalidate), so a re-recorded replay reaches returning browsers.
    fetch(url, { cache: 'default' })
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((body: unknown) => {
        if (!live) return;
        if (!isReplay(body)) throw new Error('not a replay');
        setLoaded({ url, replay: body, state: 'ready' });
      })
      .catch(() => {
        if (live) setLoaded({ url, replay: null, state: 'error' });
      });
    return () => {
      live = false;
    };
  }, [url]);

  if (url === null) return { replay: null, state: 'ready' };
  if (loaded === null || loaded.url !== url) return { replay: null, state: 'loading' };
  return { replay: loaded.replay, state: loaded.state };
}
