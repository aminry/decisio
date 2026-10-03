# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Classify every driving decision that disagrees with the rules driver (oracle_decisions.jsonl.gz, measure_fsd.py).

The oracle is the demo's rules driver (static/js/brain/rules.js) on the same snapshot and eligible candidates.

motion (drive or stop) is scored against the rules motion on the snapshot current when the answer was APPLIED (in
realtime mode the car moves on while the request is in flight). Classes, in order:
  latency    the model's motion equals the rules motion on the snapshot it was asked about but not on the one it was
             applied to, or the answer never came (a timeout replaced it with the rules driver's)
  near-tie   p(model's motion) - p(the oracle's) < margin
  judgement  anything else

vector (the manoeuvre among the eligible candidates) is asked when the car drives. Agreement is reported exactly and
within a rules-cost tolerance (a candidate whose rules cost is within `tol` of the best's is acceptable); a disagreement
beyond the tolerance is a near-tie or a judgement error by the same margin rule. The rules driver's cost rewards
progress, so a manoeuvre miss is also split by kind: a speed choice among lane-keeping candidates (a style choice the oracle does
not settle), or a stop candidate (stop at the line, for a pedestrian, at the destination) passed over.

The motion question's own wording also defines when "stop" is right ("Hold completely still right now. Correct only when
the car is already at the line (`intersection.distance` is "at") with a red or unseen signal, a stop not yet completed,
... when a pedestrian is crossing right in front (`pedestrian.distance` is "at") ... or when the car has reached the
destination"), and the state labels a line "at" within 6 m while the rules driver holds still only within 1.5 m of it.
With `--server-log` (the run's decisions.jsonl.gz, joined in call order on the motion and manoeuvre choices) each
motion is also scored against that wording: a motion the wording asks for is acceptable, and a disagreement with the
rules driver that the wording explains is counted as "wording vs rules" (the prompt-or-state class), not as an error.

The prompt-or-state class needs a written detector, defined from the first error list of a run on the served model and
before any fix is tested (correctness/PLAN.md); until then the column says so.

    python fsd_error_analysis.py runs/<run>/fsd/oracle_decisions.jsonl.gz [--margin 0.10] [--tol 1.0] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from decision_log import read


def wording_motion(state: dict | None) -> str | None:
    """The motion the question's wording asks for, from the state the model was given (None: no state)."""
    if not state:
        return None
    i = state.get("intersection") or {}
    at_line = i.get("distance") == "at" and not i.get("entered")
    if at_line and (
        i.get("signal") in ("red", "yellow", "unknown", None)
        and i.get("control") == "signal"
        or (i.get("control") == "stop" and not i.get("stop_completed"))
        or (i.get("control") in ("yield", "roundabout") and i.get("ring_traffic"))
    ):
        return "stop"
    p = state.get("pedestrian") or {}
    if p.get("distance") == "at" and p.get("in_path"):
        return "stop"
    nav = state.get("nav") or {}
    if nav.get("arrived"):
        return "stop"
    return "drive"


def motion_class(d: dict, margin: float, one_hot: bool = False) -> str | None:
    o = d["oracle"]
    if d["source"] == "rules_fallback":
        return "latency" if (d.get("error") or "").startswith("timeout") else "no answer (error)"
    ans = d.get("motion_answer")
    if not ans:
        return None  # the motion was decided locally, not asked
    m = ans["choice"]
    if m == o["motion_at_arrival"]:
        return None
    if m == o["motion"]:
        return "latency"
    if d.get("_state") is not None and m == wording_motion(d["_state"]):
        return "wording vs rules"
    if one_hot or "one-hot" in str(ans.get("probabilities_source", "")):
        return "near-tie or judgement (no distribution)"
    p = ans["probabilities"]
    return "near-tie" if p[m] - p.get(o["motion"], 0.0) < margin else "judgement"


def vector_class(d: dict, margin: float, tol: float, one_hot: bool = False) -> tuple[str | None, bool]:
    """(class or None, exact agreement); None when the vector was not asked or the model stopped."""
    ans, o = d.get("vector_answer"), d["oracle"]
    if not ans or d["source"] == "rules_fallback" or d.get("motion") == "stop":
        return None, False
    m = ans["choice"]
    if m == o["vector"]:
        return "agree", True
    costs = o.get("costs") or {}
    if m in costs and o["vector"] in costs and costs[m] <= costs[o["vector"]] + tol:
        return "agree within tolerance", False
    if one_hot or "one-hot" in str(ans.get("probabilities_source", "")):
        return "near-tie or judgement (no distribution)", False
    p = ans["probabilities"]
    return ("near-tie" if p[m] - p.get(o["vector"], 0.0) < margin else "judgement"), False


def miss_kind(d: dict, tol: float) -> str | None:
    """For a manoeuvre beyond the cost tolerance: a stop candidate passed over, or a speed choice among lane-keeping."""
    cls, _ = vector_class(d, 0.10, tol)
    if cls in (None, "agree", "agree within tolerance"):
        return None
    o, m = d["oracle"]["vector"], d["vector_answer"]["choice"]
    if o.startswith("stop_") and not m.startswith("stop_"):
        return "a stop candidate passed over"
    if o.startswith("keep_lane_") and m.startswith("keep_lane_"):
        return "a speed choice among lane-keeping candidates"
    return "other"


def analyse(rows: list[dict], margin: float, tol: float, one_hot: bool = False) -> dict:
    by_run = defaultdict(list)
    for r in rows:
        by_run[(r.get("run"), r.get("seed"))].append(r)
    m_cls, v_cls, kinds = Counter(), Counter(), Counter()
    m_asked = v_asked = 0
    per_run = []
    for key, rs in sorted(by_run.items(), key=lambda kv: str(kv[0])):
        mc, vc = Counter(), Counter()
        for d in rs:
            c = motion_class(d, margin, one_hot)
            if d.get("motion_answer") or d["source"] == "rules_fallback":
                m_asked += 1
                mc[c or "agree"] += 1
            vclass, _ = vector_class(d, margin, tol, one_hot)
            if vclass:
                v_asked += 1
                vc[vclass] += 1
        m_cls.update(mc)
        v_cls.update(vc)
        per_run.append({"run": key, "decisions": len(rs), "motion": dict(mc), "vector": dict(vc)})
        for d in rs:
            k = miss_kind(d, tol)
            if k:
                kinds[k] += 1
    return {
        "margin": margin,
        "tol": tol,
        "decisions": len(rows),
        "motion_asked": m_asked,
        "motion": dict(m_cls),
        "vector_asked": v_asked,
        "vector": dict(v_cls),
        "per_run": per_run,
        "manoeuvre_miss_kinds": dict(kinds),
    }


def table(res: dict) -> str:
    m, v = res["motion"], res["vector"]
    nd = "near-tie or judgement (no distribution)"
    lines = [
        f"{res['decisions']} model decisions; margin {res['margin']}, cost tolerance {res['tol']}",
        "",
        "| question | asked | agree | agree within tolerance | latency | near-tie | judgement "
        "| near-tie or judgement (no distribution) | no answer | prompt or state |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| motion | {res['motion_asked']} | {m.get('agree', 0)} | | {m.get('latency', 0)} | {m.get('near-tie', 0)} "
        f"| {m.get('judgement', 0)} | {m.get(nd, 0)} | {m.get('no answer (error)', 0)} "
        f"| {m.get('wording vs rules', 0)} (wording vs rules) |",
        f"| manoeuvre | {res['vector_asked']} | {v.get('agree', 0)} | {v.get('agree within tolerance', 0)} | "
        f"| {v.get('near-tie', 0)} | {v.get('judgement', 0)} | {v.get(nd, 0)} | | detector not yet defined |",
    ]
    k = res.get("manoeuvre_miss_kinds") or {}
    lines += [
        "",
        "Manoeuvre misses beyond the tolerance, by kind: " + ", ".join(f"{n} {name}" for name, n in k.items()),
    ]
    return "\n".join(lines)


def join_states(rows: list[dict], srv: list[dict]) -> None:
    """Give each answered oracle row the request state from the server's log. Both logs are in call order; the server
    also logs requests the demo abandoned at its timeout (their answers arrive late), so the walk skips server rows
    until the motion and manoeuvre choices match, and gives timeout rows (no answer) no state."""
    j, missed = 0, 0
    for o in rows:
        om = (o.get("motion_answer") or {}).get("choice")
        ov = (o.get("vector_answer") or {}).get("choice")
        if om is None and ov is None:
            continue
        k = j
        while k < len(srv):
            ans = (srv[k].get("response") or {}).get("answers") or {}
            if (ans.get("motion") or {}).get("choice") == om and (ans.get("vector") or {}).get("choice") == ov:
                break
            k += 1
        if k == len(srv):
            missed += 1
            continue
        o["_state"] = (srv[k].get("request") or {}).get("state")
        j = k + 1
    if missed:
        print(f"warning: {missed} answered decisions had no matching server row (no wording check for them)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="+")
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--server-log", nargs="*", default=[], help="the run's decisions.jsonl.gz, for the wording check")
    ap.add_argument("--one-hot", action="store_true", help="the player's answers carry no distribution (a chat model)")
    a = ap.parse_args()
    rows = [r for p in a.log for r in read(p)]
    if a.server_log:
        join_states(rows, [r for p in a.server_log for r in read(p)])
    res = analyse(rows, a.margin, a.tol, a.one_hot)
    print(table(res))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
