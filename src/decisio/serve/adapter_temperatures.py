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
     "choice_bands": {"2-10": 1.9, "11-26": 2.6, "27-255": 3.1},
     "fit": {"objective": {...}, "read_at": {...}, "n": {...}, "data": "..."}}

`choice_bands` is optional: a choice temperature per number of options, `"lo-hi"` bands that do not overlap, within 2 to
255. A choice question is served at the temperature of the band its option count falls in; an option count in no band is
served at `temperatures.choice`, then at the base's. An adapter's over-confidence on choice grows with the number of
options, and the one choice temperature of a flat file cannot follow it. Yes/no and score questions are not banded.

`fit` writes such a file from an adapter's readouts on labelled items that are not a benchmark's:

    python -m decisio.serve.adapter_temperatures fit --reads reads.jsonl --out decisio_adapter.json \\
        --data "what the items are"

One row per item, `{"type": "choice" | "noul" | "score", "probs": [...], "gold": <index into probs>}`. The best
probabilities to read are the raw ones (`p_plain` in a `--debug-readout` response, temperature 1). Probabilities as a
server returned them are accepted with the temperature they were served at, in the row (`"read_at"`) or per type in
`--read-at`; they are turned back into the raw readout before the fit, so a fit does not depend on the temperature of
the server that read the items. Take care that the temperature named is the one the question's type was served at:
a score item sent as a choice question is served at the choice temperature.
Choice is also fitted per band of option count (`--choice-bands`, default `2-10,11-26,27-255`; `none` for the one
temperature only), each by minimum log loss on the choice rows whose option count is in the band. A band with fewer than
`--min-band-items` rows (default 20) is not fitted and is served at the temperature fitted on all choice rows, which is
always written beside the bands; and when only one band has rows the file has no bands, as they would repeat that one.
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
DEFAULT_BANDS = ((2, 10), (11, 26), (27, 255))  # the choice option counts a temperature is fitted for
MAX_OPTIONS = 255  # a choice question has at most this many options (systemone refuses more with a 422)
MIN_BAND_ITEMS = 20
GRID = np.exp(
    np.linspace(math.log(0.05), math.log(8.0), 1500)
)  # kappa, the factor on log p of a raw readout: T = 1 / kappa


class AdapterFileError(ValueError):
    """An adapter file that cannot be used, with the reason and the file."""


def band_name(lo: int, hi: int) -> str:
    return f"{lo}-{hi}"


def parse_bands(spec) -> tuple:
    """((lo, hi), ...) from "2-10,11-26" (or a list of "lo-hi" strings), sorted; "none" or "" is no bands. Raises
    ValueError for a band that is malformed, outside 2 to 255, or overlaps another."""
    if isinstance(spec, str):
        spec = [b for b in spec.replace(" ", "").split(",") if b]
    if spec in ([], ["none"]):
        return ()
    out = []
    for b in spec:
        lo, sep, hi = str(b).partition("-")
        if not (sep and lo.isdigit() and hi.isdigit()):
            raise ValueError(f"band {b!r} must be written lo-hi, for example 11-26")
        lo, hi = int(lo), int(hi)
        if not 2 <= lo <= hi <= MAX_OPTIONS:
            raise ValueError(f"band {b!r} must satisfy 2 <= lo <= hi <= {MAX_OPTIONS}")
        out.append((lo, hi))
    out.sort()
    for a, b in zip(out, out[1:]):
        if b[0] <= a[1]:
            raise ValueError(f"bands {band_name(*a)} and {band_name(*b)} overlap")
    return tuple(out)


def band_table(choice_bands: dict) -> list:
    """[(lo, hi, temperature)] sorted by option count, from {"lo-hi": temperature}."""
    return sorted((*(int(x) for x in b.split("-")), float(t)) for b, t in choice_bands.items())


def band_of(bands, k: int):
    """The (lo, hi) band an option count falls in, or None."""
    for lo, hi in bands:
        if lo <= k <= hi:
            return (lo, hi)
    return None


def fit_choice_bands(P, gold, bands, min_items: int, fit_fn) -> tuple[dict, dict]:
    """({(lo, hi): (kappa, n)}, {(lo, hi): n}) for choice items P (one probability vector each) with gold indices: each
    band's kappa by `fit_fn(P, gold)` on the items whose option count is in the band, and the bands left unfitted with
    the number of items they had (under `min_items`; empty bands are not listed). When the items fall in only one band
    nothing is returned: a band fitted on all the items is the one temperature already."""
    groups: dict = {}
    for p, g in zip(P, gold):
        b = band_of(bands, len(p))
        if b is not None:
            groups.setdefault(b, ([], []))
            groups[b][0].append(p)
            groups[b][1].append(g)
    if len(groups) < 2:
        return {}, {}
    fitted = {b: (fit_fn(ps, ys), len(ps)) for b, (ps, ys) in groups.items() if len(ps) >= min_items}
    skipped = {b: len(ps) for b, (ps, _) in groups.items() if len(ps) < min_items}
    return fitted, skipped


def serving_temperatures(entry: dict) -> dict:
    """What the server holds for an adapter and shows in /health and the fingerprint: {type: T} and, when the file has
    them, "choice_bands": {"lo-hi": T}."""
    return {**entry["temperatures"], **({"choice_bands": dict(entry["choice_bands"])} if entry["choice_bands"] else {})}


def read(path) -> dict:
    """The validated content of an adapter file: {"temperatures": {type: T}, "choice_bands": {"lo-hi": T}, "layout":
    the prompt layout the adapter was trained and measured on or None, "fit": dict | None, "file": str}."""
    p = Path(path)
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise AdapterFileError(f"{p}: cannot read the adapter file ({e})") from e
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise AdapterFileError(
            f"{p}: not a {SCHEMA} file (schema {doc.get('schema') if isinstance(doc, dict) else None!r})"
        )
    out, bands = validate_temperatures(doc.get("temperatures"), doc.get("choice_bands"), str(p))
    from decisio.serve.layout import check_layout

    try:
        layout = check_layout(doc.get("layout"), str(p))
    except ValueError as e:
        raise AdapterFileError(str(e)) from e
    return {"temperatures": out, "choice_bands": bands, "layout": layout, "fit": doc.get("fit"), "file": str(p)}


def validate_temperatures(temps, raw_bands, where: str) -> tuple[dict, dict]:
    """({type: T}, {"lo-hi": T}) from the `temperatures` and `choice_bands` values of an adapter file or of a tuned
    repository's defaults (decisio.hub), checked: known types, positive finite numbers, bands within 2 to 255 that do
    not overlap, and at least one of the two. Raises AdapterFileError naming `where`."""
    if raw_bands is not None and (not isinstance(raw_bands, dict) or not raw_bands):
        raise AdapterFileError(f"{where}: `choice_bands` must be a non-empty object of lo-hi: temperature")
    if not isinstance(temps, dict) or (not temps and not raw_bands):
        raise AdapterFileError(f"{where}: `temperatures` must be a non-empty object of {', '.join(TYPES)}")

    def temperature(what, v):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise AdapterFileError(f"{where}: the {what} temperature must be a positive number, got {v!r}")
        return float(v)

    out = {}
    for k, v in temps.items():
        if k not in TYPES:
            raise AdapterFileError(
                f"{where}: unknown question type {k!r} in `temperatures` (expected {', '.join(TYPES)})"
            )
        out[k] = temperature(k, v)
    bands = {}
    if raw_bands:
        try:
            parsed = parse_bands(list(raw_bands))
        except ValueError as e:
            raise AdapterFileError(f"{where}: `choice_bands`: {e}") from e
        for (lo, hi), key in zip(parsed, sorted(raw_bands, key=lambda b: int(b.partition("-")[0]))):
            bands[band_name(lo, hi)] = temperature(f"choice band {key}", raw_bands[key])
    return out, bands


def load(adapters: dict, overrides: dict | None = None) -> dict:
    """{adapter name: {"temperatures": ..., "choice_bands": ..., "fit": ..., "file": ..., "source": "sidecar" | "flag"}}
    for the adapters
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
    rows,
    read_at: dict | None = None,
    tolerance: float = 1.0,
    data: str = "",
    score_objective: str = LOG_LOSS,
    bands=DEFAULT_BANDS,
    min_band_items: int = MIN_BAND_ITEMS,
) -> dict:
    """The adapter file for labelled rows [{"type", "probs", "gold", "read_at"?}]; types with no rows are left out.
    A row's probabilities are turned back to the raw readout with its own `read_at`, else its type's in `read_at`, else
    1.0 (already raw); the temperatures are for the raw readout, as the server applies them. Choice rows are also fitted
    per band of option count (`bands`, none for the one temperature only); see the module's account."""
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
    bands = parse_bands(bands) if isinstance(bands, str) else tuple(bands)
    temps, kappa, n, objective, detail, banded = {}, {}, {}, {}, {}, {}
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
        if t == "choice" and bands:
            fitted, skipped = fit_choice_bands(P, y, bands, min_band_items, fit_log_loss)
            if fitted:
                banded = {band_name(*b): round(1.0 / kb, 3) for b, (kb, _) in sorted(fitted.items())}
                detail[t] = {
                    "bands": {
                        band_name(*b): {"n": nb, "kappa": round(kb, 4)} for b, (kb, nb) in sorted(fitted.items())
                    },
                    "not_fitted_under_min_items": {band_name(*b): m for b, m in sorted(skipped.items())},
                    "min_band_items": min_band_items,
                }
    return {
        "schema": SCHEMA,
        "temperatures": temps,
        **({"choice_bands": banded} if banded else {}),
        "fit": {"objective": objective, "kappa_on_raw_readout": kappa, "n": n, "detail": detail, "data": data},
    }


def _rows(paths):
    for p in paths:
        op = gzip.open if str(p).endswith(".gz") else open
        with op(p, "rt") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def layout_arg(parser):
    parser.add_argument(
        "--layout",
        choices=("plain", "header"),
        default=None,
        help="the prompt layout the reads were taken under, and the model was trained on: recorded in the file, and a "
        "server that loads this adapter serves that layout (none recorded: plain; docs/running.md)",
    )


def bands_args(parser):
    parser.add_argument(
        "--choice-bands",
        default=",".join(band_name(*b) for b in DEFAULT_BANDS),
        help="the option-count bands choice is also fitted for, lo-hi,lo-hi (default %(default)s); none for the one "
        "choice temperature only",
    )
    parser.add_argument(
        "--min-band-items",
        type=int,
        default=MIN_BAND_ITEMS,
        help="a band with fewer choice items is not fitted and keeps the all-choice temperature (default %(default)s)",
    )


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
    bands_args(f)
    layout_arg(f)
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
    bands_args(sp)
    layout_arg(sp)
    c = sub.add_parser("check", help="validate an adapter file and print it")
    c.add_argument("file")
    a = ap.parse_args(argv)
    if a.cmd != "check":
        try:
            a.bands = parse_bands(a.choice_bands)
        except ValueError as e:
            ap.error(f"--choice-bands: {e}")
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
            bands=a.bands,
            min_band_items=a.min_band_items,
        )
        if a.layout:
            doc["layout"] = a.layout
        Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
        print(
            json.dumps(
                {k: doc["fit"][k] for k in ("variant", "kappa", "kappa_against_decisio_defaults", "n")}
                | {"temperatures": doc["temperatures"]}
                | ({"choice_bands": doc["choice_bands"]} if "choice_bands" in doc else {}),
                indent=1,
            )
        )
        return 0
    read_at = {k: float(v) for k, v in (kv.split("=", 1) for kv in a.read_at.split(",") if kv)}
    doc = fit(
        list(_rows(a.reads)),
        read_at,
        a.tolerance,
        a.data,
        a.score_objective,
        a.bands,
        a.min_band_items,
    )
    if a.layout:
        doc["layout"] = a.layout
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
