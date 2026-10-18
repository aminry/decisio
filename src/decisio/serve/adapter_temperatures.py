# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""An adapter's own temperatures, per question type, carried with the adapter.

A LoRA adapter changes the scale of the readout, so the base's temperatures (fitted on the base's plain readouts) do
not fit it: served at the base's, a tuned 31B hedged on 70% of a benchmark's yes/no items (its top answer was right
on 70 of 74). An adapter therefore carries its own temperatures, `choice`, `noul` (yes/no) and `score`, in a file
`decisio_adapter.json` beside its weights, or in a file given with `--adapter-temperatures NAME=FILE` for an adapter
whose folder is not ours to change. A request that names the adapter is served at them; a type the file does not list
keeps the base's temperature.

    {"schema": "decisio.adapter/1",
     "temperatures": {"choice": 2.144, "noul": 3.068, "score": 2.6},
     "fit": {"objective": {...}, "read_at": {...}, "n": {...}, "data": "..."}}

`fit` writes such a file from an adapter's readouts on labelled items that are not a benchmark's:

    python -m decisio.serve.adapter_temperatures fit --reads reads.jsonl --out decisio_adapter.json \\
        --data "what the items are"

One row per item, `{"type": "choice" | "noul" | "score", "probs": [...], "gold": <index into probs>}`. The best
probabilities to read are the raw ones (`p_plain` in a `--debug-readout` response, temperature 1). Probabilities as a
server returned them are accepted with the temperature they were served at, in the row (`"read_at"`) or per type in
`--read-at`; they are turned back into the raw readout before the fit, so a fit does not depend on the temperature of
the server that read the items. Take care that the temperature named is the one the question's type was served at:
a score item sent as a choice question is served at the choice temperature.
Every type is fitted by minimum log loss by default: the temperature at which the probabilities are calibrated on those
items. A score question can instead be fitted for the error of its expected position (`--score-objective
expected_position_error`; the quantity JevBench scores, and what a caller reads from a score distribution): the
flattest temperature whose score CC is within `--tolerance` points (default 1) of the best. That error is flat over a
wide range of temperatures, so a tighter tolerance picks noise. The temperature is sharper than the calibrated one, so
the distribution is no longer calibrated, and the file says which objective was used (`fit.objective`).

Standard library and numpy only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

SIDECAR = "decisio_adapter.json"
SCHEMA = "decisio.adapter/1"
TYPES = ("choice", "noul", "score")
LOG_LOSS = "log_loss"
EXPECTED_POSITION = "expected_position_error"
GRID = np.exp(
    np.linspace(math.log(0.05), math.log(8.0), 1500)
)  # kappa, the factor on log p of a raw readout: T = 1 / kappa


class AdapterFileError(ValueError):
    """An adapter file that cannot be used, with the reason and the file."""


def read(path) -> dict:
    """The validated content of an adapter file: {"temperatures": {type: T}, "fit": dict | None, "file": str}."""
    p = Path(path)
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise AdapterFileError(f"{p}: cannot read the adapter file ({e})") from e
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise AdapterFileError(
            f"{p}: not a {SCHEMA} file (schema {doc.get('schema') if isinstance(doc, dict) else None!r})"
        )
    temps = doc.get("temperatures")
    if not isinstance(temps, dict) or not temps:
        raise AdapterFileError(f"{p}: `temperatures` must be a non-empty object of {', '.join(TYPES)}")
    out = {}
    for k, v in temps.items():
        if k not in TYPES:
            raise AdapterFileError(f"{p}: unknown question type {k!r} in `temperatures` (expected {', '.join(TYPES)})")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise AdapterFileError(f"{p}: the {k} temperature must be a positive number, got {v!r}")
        out[k] = float(v)
    return {"temperatures": out, "fit": doc.get("fit"), "file": str(p)}


def load(adapters: dict, overrides: dict | None = None) -> dict:
    """{adapter name: {"temperatures": ..., "fit": ..., "file": ..., "source": "sidecar" | "flag"}} for the adapters
    that have one. `adapters` is {name: folder}; `overrides` {name: file} (`--adapter-temperatures`), which wins over
    the folder's own file. Raises AdapterFileError for a file that cannot be used or an override for an adapter that
    is not loaded."""
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(adapters))
    if unknown:
        raise AdapterFileError(
            f"--adapter-temperatures names {unknown}, which --adapter did not load (loaded: {sorted(adapters)})"
        )
    out = {}
    for name, folder in adapters.items():
        if name in overrides:
            out[name] = {**read(overrides[name]), "source": "flag"}
        elif (Path(folder) / SIDECAR).is_file():
            out[name] = {**read(Path(folder) / SIDECAR), "source": "sidecar"}
    return out


# ---- fitting ---------------------------------------------------------------------------------------------------------


def rescale(p, kappa: float):
    """p proportional to p**kappa, renormalised: the readout served at T_read / kappa when it was served at T_read."""
    q = np.clip(np.asarray(p, float), 1e-12, None) ** kappa
    return q / q.sum()


def log_loss(P, gold, kappa: float) -> float:
    tot = 0.0
    for p, g in zip(P, gold):
        z = np.log(np.clip(np.asarray(p, float), 1e-12, None)) * kappa
        z = z - z.max()
        tot -= z[g] - math.log(float(np.exp(z).sum()))
    return tot / len(P)


def fit_log_loss(P, gold) -> float:
    """The kappa on the grid with the lowest mean log loss."""
    sizes = {len(p) for p in P}
    if len(sizes) == 1:  # one number of options: all grid points at once
        L = np.log(np.clip(np.asarray(P, float), 1e-12, None))
        g = np.asarray(gold)
        z = L[None, :, :] * GRID[:, None, None]
        z = z - z.max(axis=2, keepdims=True)
        lse = np.log(np.exp(z).sum(axis=2))
        ll = (
            lse - np.take_along_axis(z, np.broadcast_to(g[None, :, None], (len(GRID), len(g), 1)), axis=2)[:, :, 0]
        ).mean(axis=1)
        return float(GRID[int(np.argmin(ll))])
    best, bk = float("inf"), 1.0
    for k in GRID:
        ll = log_loss(P, gold, float(k))
        if ll < best:
            best, bk = ll, float(k)
    return bk


def position_cc(P, gold, kappa: float = 1.0) -> float:
    """100 * (1 - mean nMAE / mean nMAE_chance) of the expected position of each (rescaled) distribution: JevBench's
    score reading, nMAE = |expected position - gold| / (K - 1), chance the mean over the K levels."""
    err, chance = [], []
    for p, g in zip(P, gold):
        q = rescale(p, kappa)
        k = len(q)
        err.append(abs(float(np.dot(np.arange(k), q)) - g) / (k - 1))
        chance.append(sum(abs(lv - g) for lv in range(k)) / k / (k - 1))
    return 100.0 * (1.0 - float(np.mean(err)) / float(np.mean(chance)))


def fit_expected_position(P, gold, tolerance: float = 1.0) -> tuple[float, float, float]:
    """(kappa, its CC, the best CC on the grid): the smallest kappa, so the flattest distribution, whose score CC is
    within `tolerance` points of the best on the grid."""
    cc = [position_cc(P, gold, float(k)) for k in GRID]
    best = max(cc)
    for k, c in zip(GRID, cc):
        if c >= best - tolerance:
            return float(k), c, best
    raise AssertionError("unreachable: the best point is within tolerance of itself")


def fit(
    rows, read_at: dict | None = None, tolerance: float = 1.0, data: str = "", score_objective: str = LOG_LOSS
) -> dict:
    """The adapter file for labelled rows [{"type", "probs", "gold", "read_at"?}]; types with no rows are left out.
    A row's probabilities are turned back to the raw readout with its own `read_at`, else its type's in `read_at`, else
    1.0 (already raw); the temperatures are for the raw readout, as the server applies them."""
    if score_objective not in (LOG_LOSS, EXPECTED_POSITION):
        raise ValueError(f"score objective must be {LOG_LOSS} or {EXPECTED_POSITION}")
    read_at = read_at or {}
    by = {t: ([], []) for t in TYPES}
    for r in rows:
        if r["type"] not in by:
            raise ValueError(f"unknown question type {r['type']!r}")
        t_read = float(r.get("read_at") or read_at.get(r["type"]) or 1.0)
        p = np.asarray(r["probs"], float)
        by[r["type"]][0].append(p if t_read == 1.0 else rescale(p, t_read))
        by[r["type"]][1].append(int(r["gold"]))
    temps, kappa, n, objective, detail = {}, {}, {}, {}, {}
    for t, (P, y) in by.items():
        if not P:
            continue
        n[t] = len(P)
        if t == "score" and score_objective == EXPECTED_POSITION:
            k, cc, best = fit_expected_position(P, y, tolerance)
            objective[t] = EXPECTED_POSITION
            detail[t] = {
                "cc_at_fit": round(cc, 2),
                "cc_best_on_grid": round(best, 2),
                "cc_unscaled": round(position_cc(P, y, 1.0), 2),
                "tolerance": tolerance,
                "log_loss_at_fit": round(log_loss(P, y, k), 4),
                "log_loss_best": round(log_loss(P, y, fit_log_loss(P, y)), 4),
            }
        else:
            k = fit_log_loss(P, y)
            objective[t] = LOG_LOSS
            if t == "score":
                detail[t] = {
                    "cc_at_fit": round(position_cc(P, y, k), 2),
                    "cc_unscaled": round(position_cc(P, y, 1.0), 2),
                }
        kappa[t] = round(k, 4)
        temps[t] = round(1.0 / k, 3)
    return {
        "schema": SCHEMA,
        "temperatures": temps,
        "fit": {"objective": objective, "kappa_on_raw_readout": kappa, "n": n, "detail": detail, "data": data},
    }


def _rows(paths):
    for p in paths:
        op = gzip.open if str(p).endswith(".gz") else open
        with op(p, "rt") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m decisio.serve.adapter_temperatures", description=__doc__.split("\n\n")[0]
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="write an adapter file from labelled readouts")
    f.add_argument(
        "--reads", action="append", required=True, help="a JSONL (or .gz) of {type, probs, gold}; repeatable"
    )
    f.add_argument(
        "--read-at",
        default="",
        help="the temperatures rows without their own `read_at` were served at, per type (default: raw readouts), "
        "e.g. choice=4.672,noul=5.252",
    )
    f.add_argument("--out", required=True)
    f.add_argument(
        "--tolerance", type=float, default=1.0, help="score: points of CC the flattest accepted temperature may give up"
    )
    f.add_argument(
        "--score-objective",
        choices=(LOG_LOSS, EXPECTED_POSITION),
        default=LOG_LOSS,
        help="how the score temperature is fitted (default: log loss, as the other types)",
    )
    f.add_argument("--data", default="", help="what the items are (kept in the file)")
    sp = sub.add_parser("fit-split", help="Lab 1's split fit from a model's stored reads on the H and dev pools")
    sp.add_argument("--h", required=True, help="the model's reads of the H pool (JSONL or .gz: id, probs, p_plain)")
    sp.add_argument("--h-keys", required=True, help="{id: {gold, ...}} of the H pool")
    sp.add_argument(
        "--dev", required=True, help="the model's reads of the dev pool (typed: choice, yes/no, true score)"
    )
    sp.add_argument("--dev-keys", required=True, help="{id: {gold, type}} of the dev pool")
    sp.add_argument("--variant", choices=("half0", "half1", "full", "dev", "devhalf0", "devhalf1"), default="half0")
    sp.add_argument("--use", choices=("probs", "p_plain"), default="probs")
    sp.add_argument(
        "--served", default="", help="temperatures the reads were served at, e.g. choice=4.672,noul=5.252,score=5.252"
    )
    sp.add_argument("--dev-served", default="", help="the same for the dev pool where it differs, e.g. score=4.672")
    sp.add_argument("--out", required=True)
    sp.add_argument("--data", default="")
    c = sub.add_parser("check", help="validate an adapter file and print it")
    c.add_argument("file")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        print(json.dumps(read(a.file), indent=1))
        return 0
    if a.cmd == "fit-split":
        from decisio.serve import split_fit

        def temps(text):
            return {k: float(v) for k, v in (kv.split("=", 1) for kv in text.split(",") if kv)}

        doc = split_fit.fit_split(
            split_fit.read_reads(a.h),
            json.loads(Path(a.h_keys).read_text()),
            split_fit.read_reads(a.dev),
            json.loads(Path(a.dev_keys).read_text()),
            variant=a.variant,
            use=a.use,
            served=temps(a.served),
            dev_served=temps(a.dev_served),
            data=a.data,
        )
        Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
        print(
            json.dumps(
                {k: doc["fit"][k] for k in ("variant", "kappa", "kappa_against_decisio_defaults", "n")}
                | {"temperatures": doc["temperatures"]},
                indent=1,
            )
        )
        return 0
    read_at = {k: float(v) for k, v in (kv.split("=", 1) for kv in a.read_at.split(",") if kv)}
    doc = fit(list(_rows(a.reads)), read_at, a.tolerance, a.data, a.score_objective)
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
