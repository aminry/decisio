// Plays one scenario headlessly with the same world, traffic, autopilot, and step function as the
// app. Two clocks:
//   lockstep  the sim waits for each decision, so results measure decision quality alone and are
//             reproducible for the Rules brain
//   realtime  the sim advances with the wall clock while requests are in flight, as in the app, so
//             model latency counts

import { World } from "../sim/world.js";
import { NpcFleet } from "../sim/npc.js";
import { stepWorld } from "../sim/step.js";
import { Autopilot } from "../brain/brain.js";
import { Route } from "../map/route.js";
import { DriveMetrics } from "./metrics.js";

const DT = 1 / 60;

// Yield to the event loop without the 1 s timer clamp browsers apply to hidden tabs.
const channel = new MessageChannel();
const waiting = [];
channel.port1.onmessage = () => { const r = waiting.shift(); if (r) r(); };
export const yieldNow = () => new Promise((r) => { waiting.push(r); channel.port2.postMessage(0); });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// A world set up for a scenario, ready to step. The app uses this to replay a scenario in 3D.
export function setupScenario(map, sc, { brain = "rules", npcs = 40, weather = "dry", style = null, vehicleSpec = undefined, onDecision = null, onEvent = null } = {}) {
  const world = new World(map, { seed: sc.traffic_seed, weather, vehicleSpec });
  const lane = map.lane(sc.start.edge, sc.start.lane);
  world.placeOnLane(lane, sc.start.s);
  const fleet = new NpcFleet(world, { count: npcs, seed: sc.traffic_seed });
  const events = [];
  const autopilot = new Autopilot(world, {
    onDecision: (d) => onDecision && onDecision(d),
    onEvent: (e) => { events.push({ t: Math.round(world.t * 10) / 10, ...e }); if (onEvent) onEvent(e); },
  });
  if (style) autopilot.style = style;
  autopilot.setBrain(brain);
  world.route = new Route(sc.route, map, world.ego.spec);
  world.destination = sc.goal;
  autopilot.setEnabled(true);
  return { world, fleet, autopilot, events };
}

export async function runScenario(map, sc, { brain = "rules", npcs = 40, mode = "lockstep", weather = "dry", style = null, vehicleSpec = undefined, shouldStop = () => false, timeoutMs = null, collectExamples = false, wording = "default" } = {}) {
  // the state's wording variant for this drive (added for this repository; brain/state.js reads it)
  globalThis.__fsdWording = wording;
  const metrics = new DriveMetrics(new Route(sc.route, map).length);
  let worldRef = null;
  const examples = [];
  const onDecision = (d) => {
    // the motion question exactly as it would be asked, labelled with the deciding brain's motion (rules: the oracle)
    if (collectExamples && d.questions && d.questions.motion) examples.push({ state: d.state, question: d.questions.motion, label: d.motion, t: worldRef ? Math.round(worldRef.t * 100) / 100 : null });
    if (d.meta && d.meta.source === "jev" && d.meta.latency_ms) { metrics.latencies.push(d.meta.latency_ms); if (d.meta.server_ms) metrics.serverLatencies.push(d.meta.server_ms); }
    // every model decision (and every timeout fallback) with its oracle, for the error analysis (added for this repository)
    if (d.meta && d.meta.oracle) {
      const asked = (q) => (d.answers && d.answers[q] && !d.answers[q].local ? { choice: d.answers[q].choice, probabilities: d.answers[q].probabilities } : null);
      metrics.decisions.push({ t: worldRef ? Math.round(worldRef.t * 100) / 100 : null, source: d.meta.source, fallback: d.meta.fallback || null,
        error: d.meta.error || null, motion: d.motion, chosen: d.chosenId, motion_answer: asked("motion"), vector_answer: asked("vector"),
        oracle: d.meta.oracle, latency_ms: d.meta.latency_ms ?? null, server_ms: d.meta.server_ms ?? null, flags: d.flags });
    }
  };
  const ctx = setupScenario(map, sc, { brain, npcs, weather, style, vehicleSpec, onDecision });
  worldRef = ctx.world;
  if (timeoutMs) ctx.autopilot.timeoutMs = timeoutMs;
  const { world, autopilot, events } = ctx;
  const limit = Math.max(120, sc.tags.length_m / 2.5);
  const wallStart = performance.now();
  let arrived = false, steps = 0;
  while (world.t < limit && !shouldStop()) {
    const road = stepWorld(ctx, DT, world.t * 1000);
    for (const e of world.events) if (e.type === "collision") events.push(e);
    metrics.record(world, autopilot.snap, road, DT);
    steps++;
    if (!autopilot.enabled) { arrived = events.some((e) => e.type === "arrived"); break; }
    if (mode === "lockstep") {
      if (autopilot.inFlight && autopilot.firing) await autopilot.firing;
      while (autopilot.rerouting) await sleep(5);
      if (steps % 240 === 0) await yieldNow();
    } else {
      // keep sim time from running ahead of the wall clock
      const ahead = world.t * 1000 - (performance.now() - wallStart);
      if (ahead > 4) await sleep(ahead);
      else if (steps % 60 === 0) await yieldNow();
    }
  }
  const result = metrics.summary(world, autopilot, arrived);
  return { id: sc.id, tags: sc.tags, ...result, events: events.slice(0, 50), wall_ms: Math.round(performance.now() - wallStart), ...(collectExamples ? { examples } : {}) };
}
