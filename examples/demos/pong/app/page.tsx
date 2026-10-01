import { Home } from '@/components/lane/home';

/**
 * `/` is the demo. It replays a recorded run from public/replay.json, so it costs nothing and looks the same on
 * every machine.
 *
 *   ?clean=1  hide everything except the lanes (for screen recording)
 */
export default async function Page({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const value = params.clean;
  const first = Array.isArray(value) ? value[0] : value;

  return <Home clean={first === '1' || first === 'true' || first === ''} />;
}
