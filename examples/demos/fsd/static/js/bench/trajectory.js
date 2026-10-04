// Trajectory recording for the benchmark (added for this repository). One recorder per drive keeps the
// complete world state at a fixed rate of simulated time, every applied model decision with the request
// and the answer, and every violation at the step it happened; examples/demos/tools/trajectory.py writes
// it and render_fsd.py draws a clip from it. The recorder only reads the world, so a recorded drive is
// the same drive as an unrecorded one.
//
// Positions are metres (x east, y north), headings radians counter-clockwise from +x. A vehicle's x, y is
// the centre of its box (the simulation keeps the rear axle; the ego's is in `rear_axle`). `t_wall` is
// the page's clock in Unix seconds (performance.timeOrigin + performance.now()), `t_sim` the world's.

import { pointAt, headingAt } from "../map/mapdata.js";
import { GUTTER, CURB, streetOf } from "../map/streets.js";
import { sceneryFor } from "../map/scenery.js";

export const TICK_STEPS = 6;      // a tick every 6 physics steps of 1/60 s: 10 Hz of simulated time
const PARKED_RADIUS_M = 100;      // parked cars never move by themselves; each tick lists those this close to the ego
const OFF_ROAD_FAIL_S = 1;        // a drive fails once it has been off the road this long (metrics.js)
const VIOLATIONS = new Set(["collision", "red_light", "stop_sign", "failed_to_yield"]);
const FAILURE_EVENT = { collision: "collision", "red light": "red_light", "stop sign": "stop_sign", "failed to yield": "failed_to_yield" };

const r2 = (x) => Math.round(x * 100) / 100;
const r3 = (x) => Math.round(x * 1000) / 1000;
const pt = (p) => [r2(p[0]), r2(p[1])];
export const wallNow = () => Math.round((performance.timeOrigin + performance.now()) * 10) / 1e4;
const wallOf = (ms) => (ms === undefined || ms === null ? null : Math.round((performance.timeOrigin + ms) * 10) / 1e4);

// The static geometry a top-down renderer needs, once per map: streets with their asphalt and sidewalk
// offsets, lanes, junction discs, stop and yield lines, signal heads with their group, buildings and the
// solid scenery a car can hit.
const geometryCache = new WeakMap();
export function mapGeometry(map) {
  if (geometryCache.has(map)) return geometryCache.get(map);
  const edges = [], stopLines = [], signals = [];
  const half = new Map();
  for (const e of map.edges.values()) {
    const [blvd, walk] = streetOf(e);
    edges.push({ id: e.id, from: e.from, to: e.to, pts: e.pts.map(pt), name: e.name || "", cls: e.cls, oneway: !!e.oneway,
      lanes: e.lanes, lane_width: e.lane_width, lane_offsets: e.lane_offsets, asphalt: e.asphalt,
      surface: [r2(e.asphalt[0] - GUTTER), r2(e.asphalt[1] + GUTTER)], curb: CURB, boulevard: blvd, sidewalk: walk,
      ring: !!e.ring, control: e.control ? { type: e.control.type, id: e.control.id, group: e.control.group ?? null, s_line: e.control.s_line } : null });
    const h = Math.max(Math.abs(e.asphalt[0]), Math.abs(e.asphalt[1])) + GUTTER;
    for (const n of [e.from, e.to]) half.set(n, Math.max(half.get(n) || 0, h));
    const c = e.control;
    if (!c) continue;
    // the painted line as render/roads.js places it: across this direction's lanes at s_line
    const left = e.oneway ? e.asphalt[0] + 0.15 : 0.12;
    const right = e.asphalt[1] + GUTTER - 0.1 - (c.type === "yield" && e.parking ? e.parking[1] : 0);
    const p = pointAt(e.pts, e.cum, c.s_line), hd = headingAt(e.pts, e.cum, c.s_line);
    const nx = Math.sin(hd), ny = -Math.cos(hd);   // to the right of travel
    const line = [pt([p[0] + nx * left, p[1] + ny * left]), pt([p[0] + nx * right, p[1] + ny * right])];
    stopLines.push({ edge: e.id, type: c.type, control: c.id, group: c.group ?? null, heading: r3(hd), line });
    if (c.type === "signal") signals.push({ intersection: c.id, group: c.group, edge: e.id, heading: r3(hd), line,
      at: pt([p[0] + nx * (right + 0.8) + Math.cos(hd) * 0.8, p[1] + ny * (right + 0.8) + Math.sin(hd) * 0.8]) });
  }
  const degree = new Map();
  for (const e of map.edges.values()) for (const n of [e.from, e.to]) degree.set(n, (degree.get(n) || 0) + 1);
  const junctions = [];
  for (const [id, n] of map.nodes) if ((degree.get(id) || 0) >= 2 && half.has(id)) junctions.push({ node: id, x: r2(n.x), y: r2(n.y), r: r2(half.get(id)) });
  const scenery = sceneryFor(map);
  const geometry = {
    name: map.pack.name, extent: map.extent, gutter: GUTTER,
    edges,
    lanes: map.laneList.map((l) => ({ edge: l.edge, idx: l.idx, pts: l.pts.map(pt) })),
    junctions,
    intersections: [...map.intersections.values()].map((i) => ({ id: i.id, x: i.x, y: i.y, plan: i.plan,
      approaches: (i.approaches || []).map((a) => ({ edge: a.edge, group: a.group, s_line: a.s_line })) })),
    stop_lines: stopLines,
    signals,
    roundabouts: [...(map.roundabouts || new Map()).values()].map((r) => ({ id: r.id, x: r.x, y: r.y, outer_r: r.outer_r, island_r: r.island_r })),
    buildings: (map.pack.buildings || []).map((b) => b.pts.map(pt)),
    obstacles: {
      circles: scenery.circles.map((c) => ({ kind: c.kind, x: r2(c.x), y: r2(c.y), r: r2(c.radius) })),
      polygons: scenery.polygons.map((p) => ({ kind: p.kind, pts: p.pts.map(pt) })),
    },
  };
  geometryCache.set(map, geometry);
  return geometry;
}

const vehicle = (n) => {
  const [cx, cy] = n.center;
  return { id: n.id, kind: n.kind || n.spec.kind || "car", x: r2(cx), y: r2(cy), psi: r3(n.psi), v: r2(n.v),
    length: n.spec.length, width: n.spec.width };
};
const parkedCar = (n) => {
  const [cx, cy] = n.center;
  return { id: n.id, x: r2(cx), y: r2(cy), psi: r3(n.psi), length: n.spec.length, width: n.spec.width };
};

export class TrajectoryRecorder {
  // `run`: what main.js knows about the drive (suite seed, scenario index); the rest comes from the drive itself.
  constructor(map, sc, ctx, run) {
    this.map = map; this.ctx = ctx;
    this.ticks = []; this.events = []; this.all = [];
    this.decisionPending = null; this.localPending = null;
    this.route = ctx.world.route;
    this.onRoad = true;
    this.offRoadFailed = null;
    const w = ctx.world, ap = ctx.autopilot;
    this.head = {
      run: { ...run, scenario_id: sc.id, traffic_seed: sc.traffic_seed, tags: sc.tags, map: map.pack.name,
        timeout_ms: ap.timeoutMs, tick_hz: 60 / TICK_STEPS, physics_hz: 60, parked_radius_m: PARKED_RADIUS_M,
        ego: { length: w.ego.spec.length, width: w.ego.spec.width, rear_overhang: w.ego.spec.rearOverhang } },
      map: { ...mapGeometry(map), route: sc.route.polyline.map(pt), destination: pt(sc.goal),
        start: { x: r2(sc.start.x), y: r2(sc.start.y), psi: r3(sc.start.psi) } },
    };
    this.tick(0, w.roadInfo());
  }

  // Every decision the autopilot applies (Autopilot.onDecision). A model answer or a timeout fallback goes
  // into the next tick's `decision`; a decision taken without asking (no eligible candidate, nothing to ask)
  // goes into `local_decision`.
  decision(d) {
    const meta = d.meta || {};
    const applied = wallNow();
    const asked = d.asked || {};
    const common = {
      chosen: { motion: d.motion, maneuver: d.chosenId }, source: meta.source, flags: d.flags || [],
      oracle: meta.oracle || null, asked_t_sim: asked.t_sim ?? null, asked_t_wall: wallOf(asked.wall_ms), applied_t_wall: applied,
      local_answers: Object.fromEntries(Object.entries(d.answers || {}).filter(([, a]) => a && a.local)),
    };
    const waited = asked.wall_ms !== undefined ? Math.round((performance.now() - asked.wall_ms) * 10) / 10 : null;
    if (meta.source === "jev" && d.trace && d.trace.response) {
      this.decisionPending = { request: d.trace.request, answer: d.trace.response.answers || {}, latency_ms: meta.latency_ms ?? null,
        round_trip_ms: meta.round_trip_ms ?? null, server_ms: meta.server_ms ?? null, client_wait_ms: waited,
        rules_fallback: false, fallback: meta.fallback || null, model: meta.model || null, ...common };
    } else if (meta.source === "jev") {
      // the call returned no answer (the demo's timeout can fire while the body is read, and api() then hands back
      // an empty response): the motion defaults to drive and the rules driver picks the manoeuvre
      this.decisionPending = { request: { state: d.state, questions: d.questions }, error: "the response carried no answer",
        latency_ms: waited, rules_fallback: false, fallback: meta.fallback || null, ...common };
    } else if (meta.source === "rules_fallback") {
      this.decisionPending = { request: { state: d.state, questions: d.questions }, error: meta.error || "failed",
        latency_ms: waited, rules_fallback: true, fallback: "rules", ...common };
    } else {
      this.localPending = { source: meta.source || "local", motion: d.motion, maneuver: d.chosenId, flags: d.flags || [] };
    }
  }

  // The autopilot's own events (safety brake, deadlock, fallback, reroute, arrived).
  event(e) { this.pushEvent({ ...e, violation: false }); }

  pushEvent(e) {
    const ev = { ...e, t_sim: r3(this.ctx.world.t), t_wall: wallNow() };
    this.events.push(ev);
    this.all.push(ev);
  }

  // After every physics step: violations as they happen, and a tick on the 10 Hz grid or whenever
  // something happened that the next tick must carry.
  step(steps, road) {
    const w = this.ctx.world;
    for (const e of w.events) if (VIOLATIONS.has(e.type)) this.pushEvent({ ...e, violation: true });
    const onRoad = !!road.on_road;
    if (!onRoad && this.onRoad) this.pushEvent({ type: "off_road", violation: true, distance_to_road_m: r2(road.distance_to_road) });
    this.onRoad = onRoad;
    if (!this.offRoadFailed && w.violations.off_road_s >= OFF_ROAD_FAIL_S) this.offRoadFailed = { t_sim: r3(w.t), t_wall: wallNow() };
    const routeChanged = w.route && w.route !== this.route;
    if (steps % TICK_STEPS === 0 || this.events.length || this.decisionPending || this.localPending || routeChanged || !this.ctx.autopilot.enabled) this.tick(steps, road);
  }

  tick(steps, road) {
    const w = this.ctx.world, ap = this.ctx.autopilot, ego = w.ego;
    const [cx, cy] = ego.center;
    const signals = {};
    for (const id of w.map.intersections.keys()) { const p = w.phase(id); signals[id] = { A: p.A, B: p.B }; }
    const v = w.violations;
    const t = {
      tick: steps, t_wall: wallNow(), t_sim: r3(w.t),
      ego: { x: r2(cx), y: r2(cy), psi: r3(ego.psi), v: r2(ego.v), a: r2(ego.a), steer: r3(ego.delta), yaw_rate: r3(ego.r),
        length: ego.spec.length, width: ego.spec.width, rear_axle: [r2(ego.x), r2(ego.y)], signal: ego.signal || null,
        on_road: !!road.on_road, road: road.name || "", limit: r2(road.limit || 0),
        maneuver: ap.executing && ap.executing.candidate ? ap.executing.candidate.id : null, in_flight: !!ap.inFlight, enabled: ap.enabled },
      npcs: w.npcs.map(vehicle),
      pedestrians: w.crowd.list.map((p) => ({ id: p.id, x: r2(p.x), y: r2(p.y), psi: r3(p.psi), v: r2(p.v), crossing: !!p.crossing })),
      parked: w.parked.near(ego.x, ego.y, PARKED_RADIUS_M).map(parkedCar),
      doors: w.parked.activeDoors.map((d) => ({ id: d.id, owner: d.owner.id, x: r2(d.x), y: r2(d.y), psi: r3(d.psi), length: d.spec.length, width: d.spec.width, angle: r3(d.angle) })),
      signals,
      violations: { collisions: v.collisions, collisions_at_fault: v.collisions_at_fault, red_lights_run: v.red_lights_run,
        stop_signs_run: v.stop_signs_run, failed_to_yield: v.failed_to_yield || 0, off_road_s: r2(v.off_road_s),
        safety_brakes: v.safety_brakes, fallbacks: v.fallbacks, deadlock_overrides: v.deadlock_overrides },
      decision: this.decisionPending,
    };
    if (this.events.length) t.events = this.events;
    if (this.localPending) t.local_decision = this.localPending;
    if (w.route && w.route !== this.route) { t.route = w.route.pts.map(pt); this.route = w.route; }
    this.events = []; this.decisionPending = null; this.localPending = null;
    this.ticks.push(t);
  }

  // The drive's last tick and its outcome; `result` is DriveMetrics.summary's.
  finish(result, steps) {
    if (this.ticks[this.ticks.length - 1].tick !== steps || this.events.length || this.decisionPending) this.tick(steps, this.ctx.world._road || this.ctx.world.roadInfo());
    const last = this.ticks[this.ticks.length - 1];
    const failedAt = {};
    for (const f of result.failures) {
      const first = FAILURE_EVENT[f] ? this.all.find((e) => e.violation && e.type === FAILURE_EVENT[f]) : null;
      if (first) failedAt[f] = { t_sim: first.t_sim, t_wall: first.t_wall };
      else if (f === "off road" && this.offRoadFailed) failedAt[f] = this.offRoadFailed;
      else failedAt[f] = { t_sim: last.t_sim, t_wall: last.t_wall };
    }
    const { latencies_ms, server_ms, decision_log, ...metrics } = result;
    return {
      header: this.head, ticks: this.ticks,
      end: { pass: result.pass, failures: result.failures, arrived: result.arrived, failed_at: failedAt,
        violations: this.all.filter((e) => e.violation), steps, ticks: this.ticks.length, metrics,
        model_decisions: latencies_ms.length },
    };
  }
}
