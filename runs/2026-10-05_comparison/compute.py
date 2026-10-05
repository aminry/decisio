"""Apply RULE.md to the board snapshots and this repository's run records.

Writes results.json (every value, interval and mark) and tables.md (the tables docs/comparison.md quotes).
Run from the repository root: python runs/2026-10-05_comparison/compute.py
"""

import gzip
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent

BOARD = json.load(gzip.open(HERE / "snapshots/decision-index_index-v0.2.1.json.gz"))
JEVBENCH = json.load(gzip.open(HERE / "snapshots/jevbench_v1.4.2.2-results.json.gz"))

# the card-hour price of the RTX PRO 6000 these runs used: about $26 for 17.6 card-hours on vast.ai, 2026-10-04 to 05
CARD_USD_PER_HOUR = 26 / 17.6
# requests the kit sends for one pass over edition 0.2.1 (manifest.json of runs/2026-10-05_decision-index)
SUITE_REQUESTS = 150_759
FORECAST = 48

OURS = [
    {
        "key": "qwen3.6-35b-a3b",
        "label": "decisio · Qwen3.6-35B-A3B",
        "run": "2026-10-05_decision-index/decisio-0.4.0-qwen3.6-35b-a3b",
        "jevbench": "2026-10-03_qiv-default/jevbench_qiv",
    },
    {
        "key": "gemma-4-12b",
        "label": "decisio · Gemma 4 12B",
        "run": "2026-10-05_decision-index/decisio-0.4.0-gemma-4-12b",
        "jevbench": "2026-10-04_gemma-base/jevbench_gemma",
    },
    {
        "key": "gemma-4-31b",
        "label": "decisio · Gemma 4 31B",
        "run": "2026-10-05_decision-index/decisio-0.6.0-gemma-4-31b",
        "jevbench": "2026-10-04_gemma-4-31b/jevbench_g31",
    },
]


def chance(bid):
    return next(c["chance"] for c in BOARD["suite"]["chance_levels"] if c["id"] == bid)


def gold(bid):
    return BOARD["benchmarks"][str(bid)].get("gold") or 1.0


def n_items(bid, scored):
    jev_cases = BOARD["jev"]["results"][str(bid)].get("cases")
    return min(x for x in (BOARD["benchmarks"][str(bid)]["cases"], scored, jev_cases) if x)


def half_width(a, b, n):
    m = (a + b) / 2
    return 1.96 * math.sqrt(2 * m * (1 - m) / n)


def mark(d, h):
    return "ahead" if d > h else "behind" if d < -h else "level"


def skill_var(bid, a, b, n):
    """Variance of the difference of two benchmark skills under RULE.md."""
    m = (a + b) / 2
    scale = 0.25 if bid == FORECAST else 1 - chance(bid)
    return 2 * m * (1 - m) / (n * scale**2)


def board_value(entry, bid):
    """The compared value of one board entry (Jev or a model): raw, or ForecastBench's Brier loss."""
    if bid == FORECAST:
        return entry["results"][str(bid)]["score"]
    return entry["benchmarks"][str(bid)]["raw"]


def our_value(scores, bid):
    if bid == FORECAST:
        return scores["benchmarks"][str(bid)]["score"]
    return scores["index_benchmarks"][str(bid)]["raw"]


def open_entries(count=3):
    """The open-weights entries with the highest index: a public weights repository or served checkpoint."""
    rows = [m for m in BOARD["models"] if m["meta"].get("weights_repo") or m["meta"].get("served_checkpoint")]
    rows.sort(key=lambda m: -m["scores"]["balanced_skill"])
    return rows[:count]


def jevbench_row(key):
    return next(s for s in JEVBENCH["systems"] if s["key"] == key)


def main():
    areas = BOARD["suite"]["areas"]
    area_weights = BOARD["suite"]["area_weights"]
    jev = BOARD["jev"]
    jev_areas = {c["id"]: c for c in jev["categories"]}
    others = open_entries()
    jev_jb = jevbench_row("jev-1.13.0")

    results = {
        "rule": "RULE.md",
        "sources": {
            "decision_index_board": {
                "url": "https://huggingface.co/spaces/multimodalart/jev-decision-index",
                "file": "data/index-v0.2.1.json",
                "space_commit": "cdbd1cab3b6eb1811ebd82d2686ad563a4d0fc0d",
                "generated_utc": BOARD["generated_utc"],
                "read_utc": "2026-10-05",
            },
            "jevbench_board": {
                "url": "https://github.com/fstandhartinger/jevbench",
                "file": "results/v1.4.2.2/jevbench-v1.4.2.2-results.json",
                "commit": "bb05a335bc809e61b20c0f745d25499a82b326fc",
                "revision": JEVBENCH["revision"],
                "generated_utc": JEVBENCH["generated_utc"],
                "read_utc": "2026-10-05",
            },
        },
        "card_usd_per_hour": round(CARD_USD_PER_HOUR, 4),
        "open_entries": [
            {
                "engine": m["engine"],
                "name": m["long_name"],
                "kind": m["meta"]["kind"],
                "base_model": m["meta"]["base_model"],
            }
            for m in others
        ],
        "bases": {},
    }

    for base in OURS:
        scores = json.loads((RUNS / base["run"] / "scores.json").read_text())
        tiers = {
            t: json.loads((RUNS / base["jevbench"] / t / "summary.json").read_text())["n_correct"]
            for t in ("easy", "original", "hard")
        }
        rows, area_vars = {}, {}
        for area, ids in areas.items():
            var_sum, weight_sum = 0.0, 0.0
            for bid in ids:
                ours, theirs = our_value(scores, bid), board_value(jev, bid)
                n = n_items(bid, scores["benchmarks"][str(bid)]["scored_requests"])
                h = half_width(ours, theirs, n)
                lower = BOARD["benchmarks"][str(bid)]["lower"]
                d = (theirs - ours) if lower else (ours - theirs)
                rows[bid] = {"ours": ours, "jev": theirs, "n": n, "h": h, "d": d, "lower": lower, "mark": mark(d, h)}
                w = gold(bid)
                var_sum += w * w * skill_var(bid, ours, theirs, n)
                weight_sum += w
            area_vars[area] = var_sum / weight_sum**2
        area_rows = {}
        our_areas = {a["id"]: a for a in scores["areas"]}
        for area in areas:
            ours, theirs = 100 * our_areas[area]["skill"], 100 * jev_areas[area]["skill"]
            h = 1.96 * math.sqrt(area_vars[area]) * 100
            area_rows[area] = {"ours": ours, "jev": theirs, "h": h, "d": ours - theirs, "mark": mark(ours - theirs, h)}
        index_var = sum(area_weights[a] ** 2 * area_vars[a] for a in areas)
        h = 1.96 * math.sqrt(index_var) * 100
        ours, theirs = scores["decision_index"], jev["scores"]["balanced_skill"]
        index_row = {"ours": ours, "jev": theirs, "h": h, "d": ours - theirs, "mark": mark(ours - theirs, h)}
        correct = sum(tiers.values())
        jev_correct = jev_jb["public_accuracy"] * 231
        h = half_width(correct / 231, jev_jb["public_accuracy"], 231)
        d = correct / 231 - jev_jb["public_accuracy"]
        jevbench = {
            "ours": correct,
            "jev": jev_correct,
            "h": h,
            "d": d,
            "mark": mark(d, h),
            "hard_ours": tiers["hard"],
            "hard_jev_220": jev_jb["tiers"]["hard"],
            "hard_mark": "not compared",
        }
        counts = {k: sum(1 for r in rows.values() if r["mark"] == k) for k in ("ahead", "level", "behind")}
        latency = scores["latency_ms"]
        results["bases"][base["key"]] = {
            "label": base["label"],
            "run": base["run"],
            "benchmarks": {str(k): v for k, v in rows.items()},
            "areas": area_rows,
            "index": index_row,
            "jevbench_231": jevbench,
            "counts": counts,
            "latency_ms": latency,
            "suite_pass_usd_est": latency["mean"] / 1000 * SUITE_REQUESTS / 3600 * CARD_USD_PER_HOUR,
        }

    results["jev"] = {
        "index": jev["scores"]["balanced_skill"],
        "areas": {a: 100 * jev_areas[a]["skill"] for a in areas},
        "latency_ms": jev["latency"],
        "suite_pass_usd": sum(jev["results"][str(b)].get("usd") or 0 for ids in areas.values() for b in ids),
        "jevbench_231": jev_jb["public_accuracy"] * 231,
        "jevbench_hard_220": jev_jb["tiers"]["hard"],
    }
    results["others"] = {}
    for m in others:
        oa = {c["id"]: c for c in m["categories"]}
        results["others"][m["engine"]] = {
            "name": m["long_name"],
            "index": m["scores"]["balanced_skill"],
            "areas": {a: 100 * oa[a]["skill"] for a in areas},
            "benchmarks": {str(b): board_value(m, b) for ids in areas.values() for b in ids},
            "latency_ms": m["latency"],
            "suite_pass_usd_est": m["latency"]["mean"] / 1000 * SUITE_REQUESTS / 3600 * CARD_USD_PER_HOUR,
        }
    results["hosted_chat_models_jevbench_231"] = [
        {"key": s["key"], "display": s["display"], "correct": round(s["public_accuracy"] * 231)}
        for s in JEVBENCH["systems"]
        if s.get("class") == "llm-baseline" and s.get("public_accuracy") is not None
    ]

    (HERE / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    (HERE / "tables.md").write_text(tables(results, areas))


SYMBOL = {"ahead": "▲", "level": "=", "behind": "▼"}


def tables(r, areas):
    keys = [b["key"] for b in OURS]
    others = list(r["others"])
    out = ["<!-- generated by compute.py; do not edit -->", ""]
    head = "| Benchmark | n | " + " | ".join(r["bases"][k]["label"] for k in keys) + " | Jev 1.13 | "
    head += " | ".join(r["others"][o]["name"] for o in others) + " |"
    for area, ids in areas.items():
        out += [f"### {area}", "", head, "| --- | ---: |" + " ---: |" * (len(keys) + 1 + len(others))]
        for bid in ids:
            b = BOARD["benchmarks"][str(bid)]
            first = r["bases"][keys[0]]["benchmarks"][str(bid)]
            name = b["dataset"] + (", Brier (lower is better)" if bid == FORECAST else f", {b['metric']}")
            ours = [r["bases"][k]["benchmarks"][str(bid)] for k in keys]
            cells = [f"{c['ours']:.3f} {SYMBOL[c['mark']]}" for c in ours]
            cells.append(f"{first['jev']:.3f}")
            cells += [f"{r['others'][o]['benchmarks'][str(bid)]:.3f}" for o in others]
            out.append(f"| {name} | {first['n']:,} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["### intervals (half-width h, per base)", ""]
    out += ["| Benchmark | " + " | ".join(keys) + " |", "| --- |" + " ---: |" * 3]
    for ids in areas.values():
        for bid in ids:
            cells = [f"{r['bases'][k]['benchmarks'][str(bid)]['h']:.3f}" for k in keys]
            out.append(f"| {BOARD['benchmarks'][str(bid)]['dataset']} | " + " | ".join(cells) + " |")
    out += ["", "### summary", ""]
    for k in keys:
        b = r["bases"][k]
        jb = b["jevbench_231"]
        area_text = ", ".join(
            f"{a} {v['ours']:.2f} vs {v['jev']:.2f} h {v['h']:.2f} {v['mark']}" for a, v in b["areas"].items()
        )
        out.append(
            f"- {b['label']}: index {b['index']['ours']:.2f} vs {b['index']['jev']:.2f} (h {b['index']['h']:.2f}, "
            f"{b['index']['mark']}); benchmarks ahead {b['counts']['ahead']}, level {b['counts']['level']}, "
            f"behind {b['counts']['behind']}; areas "
            + area_text
            + f"; JevBench {jb['ours']} vs {jb['jev']:.1f} (h {jb['h']:.3f}, {jb['mark']}); "
            f"suite pass ${b['suite_pass_usd_est']:.2f}"
        )
    out.append(f"- Jev suite pass ${r['jev']['suite_pass_usd']:.2f}")
    for o, v in r["others"].items():
        out.append(f"- {v['name']}: index {v['index']:.2f}, suite pass est ${v['suite_pass_usd_est']:.2f}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
