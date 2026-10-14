"""What certification gives on recorded answers: decisio's own certification procedure on the 31B suite capture of
`runs/2026-10-04_gemma-4-31b/capture_g31.json.gz`, task by task, with repeated random splits.

For each suite task and certification-set size n: draw n of the task's items 1,000 times (a seeded permutation each time), certify
on them at alpha and delta with `decisio.readout.certify.threshold`, and read the true risk of the threshold as the error among
all the task's items at or above it (the task's items, taken as the population). Reported: the median share of the task's
items the threshold certifies (0 when nothing is certified), the share of draws in which nothing was certified, and the share of
draws in which the certificate was broken (true risk above alpha). The same for the empirical threshold with no bound (the most
lenient one whose error on the n items is at most alpha), which is what a certificate without the bound would give.

The probabilities are the capture's plain readouts at the served temperatures of the same run (`temps_g31.json`: choice 4.672,
other 5.252, applied as `decisio.serve.temperature.apply_temperature`), the form a task without its own correction serves.

    uv run python runs/2026-10-14_certified-answers/certify_public_records.py --out runs/2026-10-14_certified-answers/results
"""

import argparse
import gzip
import json
import math
import zlib
from pathlib import Path

import numpy as np

from decisio.readout import certify
from decisio.serve.temperature import apply_temperature

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / "runs/2026-10-04_gemma-4-31b/capture_g31.json.gz"
TEMPS = ROOT / "runs/2026-10-04_gemma-4-31b/temps_g31.json"
SPLITS, SEED = 1000, 20261014
GRID = [(0.05, 0.1), (0.1, 0.1)]
SIZES = {"mmlu_pro_text_x350": (75, 175)}  # every other task has 150 items: half of them
DEFAULT_SIZES = (75,)


def load():
    t = json.load(TEMPS.open())["temperatures"]
    rows = [r for r in json.load(gzip.open(CAPTURE, "rt"))["rows"] if r["set"] == "suite"]
    tasks = {}
    for r in rows:
        kind = r["answer"]["type"]
        p = apply_temperature(np.asarray(r["p_plain"], float), t["choice"] if kind == "choice" else t["global"])
        tasks.setdefault(r["task"], []).append((float(p.max()), int(np.argmax(p)) == int(r["gold"])))
    return {k: (np.array([c for c, _ in v]), np.array([o for _, o in v])) for k, v in tasks.items()}


def cell(conf, ok, n, alpha, delta, name):
    N = len(conf)
    order = np.argsort(-conf)
    c_sorted, errs = conf[order], np.cumsum(~ok[order])

    def risk(thr):
        if not math.isfinite(thr):
            return None, 0.0
        k = int(np.searchsorted(-c_sorted, -thr, side="right"))
        return (float(errs[k - 1] / k) if k else None), k / N

    rng = np.random.default_rng([SEED, zlib.crc32(name.encode()), n, int(alpha * 1000), int(delta * 1000)])
    out = {"bounded": {"share": [], "broken": []}, "no_bound": {"share": [], "broken": []}}
    for _ in range(SPLITS):
        cal = rng.permutation(N)[:n]
        for key, thr in (
            ("bounded", certify.threshold(conf[cal], ok[cal], alpha, delta)),
            ("no_bound", certify.empirical_threshold(conf[cal], ok[cal], alpha)),
        ):
            r, share = risk(thr)
            out[key]["share"].append(share)
            out[key]["broken"].append(r is not None and r > alpha + 1e-12)
    return {
        k: {
            "median_certified_share": float(np.median(v["share"])),
            "nothing_certified": float(np.mean([s == 0 for s in v["share"]])),
            "certificate_broken": float(np.mean(v["broken"])),
        }
        for k, v in out.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for name, (conf, ok) in load().items():
        for n in SIZES.get(name, DEFAULT_SIZES):
            for alpha, delta in GRID:
                rows.append(
                    {"task": name, "items": len(conf), "accuracy": float(ok.mean()), "certification_examples": n,
                     "alpha": alpha, "delta": delta, "splits": SPLITS, **cell(conf, ok, n, alpha, delta, name)}
                )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "certification_on_the_31b_suite.json").write_text(json.dumps(rows, indent=1))
    lines = [
        "| Task (31B suite capture) | Items | Accuracy | Certification examples | alpha | Median certified share | Nothing certified | Broken, bounded | Broken, no bound |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['task']} | {r['items']} | {r['accuracy']:.3f} | {r['certification_examples']} | {r['alpha']} | "
            f"{r['bounded']['median_certified_share']:.2f} | {r['bounded']['nothing_certified']:.2f} | "
            f"{r['bounded']['certificate_broken']:.3f} | {r['no_bound']['certificate_broken']:.3f} |"
        )
    (out / "certification_on_the_31b_suite.md").write_text("\n".join(lines) + "\n")
    rows2 = [{"alpha": al, "delta": de, "minimum_examples": certify.minimum_examples(al, de)} for al in (0.02, 0.05, 0.1, 0.2) for de in (0.05, 0.1)]
    (out / "minimum_examples.json").write_text(json.dumps(rows2, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
