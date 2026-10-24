# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Per-type temperatures for an adapted model, fitted on a split of its stored reads, written as the file #207 loads.

A model tuned to be calibrated near temperature 1 is served by decisio at the base's temperatures (choice 4.67, yes/no
and score 5.25): the base's were fitted on the base's plain readouts, and on a tuned model they flatten every yes/no
answer into the abstention band. The fix is its own temperature per question type, fitted on data that is not a
benchmark's. This is the split-fit method, as a tool:

  choice     the kappa that minimises log loss on the model's reads of the H pool (776 program-keyed choice items); and
             one per band of option count (2-10, 11-26, 27-255) on the pool's choice items with that many options, so
             that a choice temperature follows the over-confidence that grows with the number of options
  yes/no     the same on its reads of the dev pool's yes/no items
  score      the same on its reads of the dev pool's score items (true score questions)

where a read is `probs`, the probabilities as the server returned them, served at the base's temperatures, and a kappa
is the exponent of the rescale p -> p**kappa (renormalised); the temperature to serve is then `T_served / kappa`. Fitted
on the plain readout (`p_plain`, which `--debug-readout` returns at temperature 1) the same answer is `T = 1 / kappa`;
`--use p_plain` does that and `--use probs` (the default) fits on the served probabilities.

The variants are: `half0` fits on the items whose id has an even sha256 prefix (their V1, the headline) and
reports the other half as held-out, `half1` the swap (V2), `full` all of the pool (V3), `dev` takes the choice fit from
the dev pool too (V4), and `devhalf0` and `devhalf1` do the same on halves of the dev pool with the other half held
out (the choice kappa of a model whose H-fitted choice temperature does not transfer to public-source questions).
It never reads a benchmark: the pools are the program-keyed H pool and the public-source dev pool, and the keys files
carry their gold answers.

    python -m decisio.serve.adapter_temperatures fit-split --h H_reads.jsonl.gz --h-keys h_keys.json \\
        --dev D_reads.jsonl.gz --dev-keys d500_keys.json --out decisio_adapter.json --variant half0

Keys files: {item id: {"gold": index into the item's probs, "type": "choice" | "noul" | "score"}} (the H keys need no
type). Standard library and numpy only.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math

import numpy as np

from decisio.serve.adapter_temperatures import (
    DEFAULT_BANDS,
    MIN_BAND_ITEMS,
    SCHEMA,
    TYPES,
    band_name,
    band_of,
    fit_choice_bands,
    parse_bands,
)

GRID = np.exp(np.linspace(math.log(0.2), math.log(8.0), 781))  # the kappa grid of the fits
SERVED = {"choice": 4.672, "noul": 5.252, "score": 5.252}  # decisio 0.14.0 to 0.16.0's defaults for the Gemma bases
VARIANTS = ("half0", "half1", "full", "dev", "devhalf0", "devhalf1")
FLOOR = 1e-12  # a probability under this counts as this, as in the fits' log loss


def half(item_id: str) -> int:
    return int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16) % 2


def read_reads(path) -> dict:
    """{id: {"probs": [...], "p_plain": [...] | None}} from a JSONL (or .gz) of the box reader's rows."""
    op = gzip.open if str(path).endswith(".gz") else open
    out = {}
    with op(path, "rt") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = {"probs": r["probs"], "p_plain": r.get("p_plain")}
    return out


def rescale(p, kappa: float):
    q = np.clip(np.asarray(p, float), FLOOR, None) ** kappa
    return q / q.sum()


def mean_log_loss(P, gold, kappa: float) -> float:
    return float(-np.mean([math.log(max(rescale(p, kappa)[g], FLOOR)) for p, g in zip(P, gold)]))


def fit_kappa(P, gold, grid=GRID) -> float:
    """The kappa on the grid with the lowest mean log loss (the smallest one on a tie)."""
    by_k: dict[int, tuple[list, list]] = {}
    for p, g in zip(P, gold):
        by_k.setdefault(len(p), ([], []))[0].append(np.asarray(p, float))
        by_k[len(p)][1].append(int(g))
    total = np.zeros(len(grid))
    for ps, gs in by_k.values():
        L = np.log(np.clip(np.stack(ps), FLOOR, None))  # items x options
        z = L[None, :, :] * grid[:, None, None]
        z = z - z.max(axis=2, keepdims=True)
        logq = np.take_along_axis(z, np.broadcast_to(np.asarray(gs)[None, :, None], (len(grid), len(gs), 1)), axis=2)[
            :, :, 0
        ]
        logq = logq - np.log(np.exp(z).sum(axis=2))
        total += -np.maximum(logq, math.log(FLOOR)).sum(axis=1)
    return float(grid[int(np.argmin(total))])


def ece_tie(conf, ok, orders: int = 200) -> float:
    """The tie-robust ECE: ten equal-count bins, ties broken by a seeded shuffle, averaged over `orders`."""
    conf, ok = np.round(np.asarray(conf, float), 9), np.asarray(ok, float)
    rng = np.random.default_rng(0)
    n = len(conf)
    vals = []
    for _ in range(orders):
        order = np.lexsort((rng.random(n), conf))
        vals.append(sum(len(c) / n * abs(ok[c].mean() - conf[c].mean()) for c in np.array_split(order, 10)))
    return float(np.mean(vals))


def ece_width10(conf, ok) -> float:
    """Top-label ECE with ten equal-width bins (the ship gates' definition, `ece10`): the bins are [0, 0.1),
    ..., [0.9, 1.0], each weighted by its share of the items."""
    conf, ok = np.clip(np.asarray(conf, float), 0, 1), np.asarray(ok, float)
    idx = np.minimum((conf * 10).astype(int), 9)
    n = len(conf)
    c = np.bincount(idx, minlength=10)
    hits = np.bincount(idx, weights=ok, minlength=10)
    mass = np.bincount(idx, weights=conf, minlength=10)
    m = c > 0
    return float(np.sum(c[m] / n * np.abs(hits[m] / c[m] - mass[m] / c[m])))


def swap_stats(dev_reads, dev_keys, ids, use: str) -> dict:
    """One group's swap-validated figures (see `swap_validated`) for the items `ids`."""
    kap = {}
    for h in (0, 1):
        fit = [i for i in ids if half(i) == h]
        kap[h] = fit_kappa([dev_reads[i][use] for i in fit], [dev_keys[i]["gold"] for i in fit])
    q = {i: rescale(dev_reads[i][use], kap[1 - half(i)]) for i in ids}
    p0 = {i: np.asarray(dev_reads[i][use], float) for i in ids}
    gold = {i: dev_keys[i]["gold"] for i in ids}

    def ece(P, fn):
        return fn([P[i].max() for i in ids], [int(P[i].argmax() == gold[i]) for i in ids])

    return {
        "n": len(ids),
        "kappa_half0": round(kap[0], 3),
        "kappa_half1": round(kap[1], 3),
        "kappa_ratio": round(max(kap.values()) / min(kap.values()), 3),
        "ece10_unscaled": round(ece(p0, ece_width10), 4),
        "ece10_swap_validated": round(ece(q, ece_width10), 4),
        "ece_equal_count_swap_validated": round(ece(q, ece_tie), 4),
    }


def swap_validated(
    dev_reads, dev_keys, *, use: str = "probs", bands=DEFAULT_BANDS, min_band_items: int = MIN_BAND_ITEMS
) -> dict:
    """The ship gates' ECE per type on the dev pool: the kappa of each half (by log loss) is applied to the OTHER half's
    items, and the ECE is taken over all of the pool's items of that type, pooled (ten equal-width bins). Also the
    ECE of the same items unscaled, the same pooled swap with the older equal-count tie-robust ECE, and the two
    halves' kappas with their ratio (the gates call a fit unstable when the ratio is over 1.25).

    Where the pool's choice items fall in more than one band of option count, `choice` also holds `bands`: the same
    figures for the items of each band, each fitted and scored within the band (a band needs `min_band_items` / 2
    items in each half, else it is listed with its count and not scored)."""
    out = {}
    for t in TYPES:
        ids = [i for i in sorted(dev_keys) if i in dev_reads and dev_keys[i]["type"] == t]
        if not ids:
            continue
        out[t] = swap_stats(dev_reads, dev_keys, ids, use)
        if t == "choice" and bands:
            groups: dict = {}
            for i in ids:
                b = band_of(bands, len(dev_reads[i][use]))
                if b is not None:
                    groups.setdefault(b, []).append(i)
            if len(groups) > 1:
                per = {}
                for b, g in sorted(groups.items()):
                    halves = [sum(half(i) == h for i in g) for h in (0, 1)]
                    if min(halves) < max(1, min_band_items // 2):
                        per[band_name(*b)] = {
                            "n": len(g),
                            "not_scored": f"under {max(1, min_band_items // 2)} items in a half",
                        }
                    else:
                        per[band_name(*b)] = swap_stats(dev_reads, dev_keys, g, use)
                out[t]["bands"] = per
    return out


def heldout(P, gold, kappa: float) -> dict:
    Q = [rescale(p, kappa) for p in P]
    P = [np.asarray(p, float) for p in P]
    return {
        "n": len(P),
        "log_loss_kappa1": round(mean_log_loss(P, gold, 1.0), 4),
        "log_loss_fit": round(mean_log_loss(P, gold, kappa), 4),
        "ece_kappa1": round(ece_tie([p.max() for p in P], [int(p.argmax() == g) for p, g in zip(P, gold)]), 4),
        "ece_fit": round(ece_tie([q.max() for q in Q], [int(q.argmax() == g) for q, g in zip(Q, gold)]), 4),
        "ece_definition": "equal-count bins, ties shuffled (200 orders), this half only",
    }


def fit_split(
    h_reads,
    h_keys,
    dev_reads,
    dev_keys,
    *,
    variant: str = "half0",
    use: str = "probs",
    served: dict | None = None,
    dev_served: dict | None = None,
    data: str = "",
    bands=DEFAULT_BANDS,
    min_band_items: int = MIN_BAND_ITEMS,
) -> dict:
    """The adapter file (schema `decisio.adapter/1`) fitted by the split method.

    Choice is also fitted per band of option count (`bands`; none for the one temperature only) on the choice pool's
    fit items in the band, with the held-out figures of each band beside it. When the pool's items fall in one band only
    (the H pool's 3 to 6 options) the file has no bands. A band with fewer than `min_band_items` fit items is not fitted
    and is served at the temperature of all the pool's choice items.

    `served`: the temperatures the reads' `probs` were served at (default the Gemma bases'). `dev_served`: where the dev
    pool's reads were served at other temperatures (S18's dev score items were sent as choice questions and served at
    the choice temperature: `{"score": 4.672}`). With `use="p_plain"` the reads are the plain readouts and the served
    temperatures are 1."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if use not in ("probs", "p_plain"):
        raise ValueError("use must be probs or p_plain")
    served = {**SERVED, **(served or {})}
    if use == "p_plain":
        served = {t: 1.0 for t in TYPES}
    dev_served = {**served, **(dev_served or {})}
    field = use

    def vec(rec, name):
        v = rec[field]
        if v is None:
            raise ValueError(f"item {name}: the read has no {field} (start the server with --debug-readout)")
        return v

    def ids_of(keys, reads, t=None):
        return [i for i in sorted(keys) if i in reads and (t is None or keys[i]["type"] == t)]

    pools = {
        "choice": ("dev", ids_of(dev_keys, dev_reads, "choice"))
        if variant in ("dev", "devhalf0", "devhalf1")
        else ("H", ids_of(h_keys, h_reads)),
        "noul": ("dev", ids_of(dev_keys, dev_reads, "noul")),
        "score": ("dev", ids_of(dev_keys, dev_reads, "score")),
    }
    bands = parse_bands(bands) if isinstance(bands, str) else tuple(bands)
    temps, kappa, checks, vs_default, banded, band_notes = {}, {}, {}, {}, {}, {}
    for t in TYPES:
        src, ids = pools[t]
        reads, keys = (h_reads, h_keys) if src == "H" else (dev_reads, dev_keys)
        h = 0 if variant in ("half0", "devhalf0") else 1
        fit_ids = [i for i in ids if variant in ("full", "dev") or half(i) == h]
        out_ids = [i for i in ids if variant in ("half0", "half1", "devhalf0", "devhalf1") and half(i) != h]
        if not fit_ids:
            raise ValueError(f"no {t} items to fit on in the {src} pool")
        P = [vec(reads[i], i) for i in fit_ids]
        y = [keys[i]["gold"] for i in fit_ids]
        k = fit_kappa(P, y)
        base = (dev_served if src == "dev" else served)[t]
        kappa[t], temps[t], vs_default[t] = round(k, 4), round(base / k, 3), round(SERVED[t] * k / base, 3)
        checks[t] = {"pool": src, "n_fit": len(fit_ids), "kappa_fit": round(k, 3), "served_at": base}
        if out_ids:
            checks[t]["heldout"] = heldout([vec(reads[i], i) for i in out_ids], [keys[i]["gold"] for i in out_ids], k)
        if t == "choice" and bands:
            fitted, skipped = fit_choice_bands(P, y, bands, min_band_items, fit_kappa)
            if fitted:
                checks[t]["bands"] = {}
                for b, (kb, nb) in sorted(fitted.items()):
                    banded[band_name(*b)] = round(base / kb, 3)
                    entry = {"n_fit": nb, "kappa_fit": round(kb, 3), "served_at": base}
                    held = [i for i in out_ids if band_of(bands, len(vec(reads[i], i))) == b]
                    if held:
                        entry["heldout"] = heldout(
                            [vec(reads[i], i) for i in held], [keys[i]["gold"] for i in held], kb
                        )
                    checks[t]["bands"][band_name(*b)] = entry
                band_notes = {
                    "not_fitted_under_min_items": {band_name(*b): m for b, m in sorted(skipped.items())},
                    "min_band_items": min_band_items,
                }
    return {
        "schema": SCHEMA,
        "temperatures": temps,
        **({"choice_bands": banded} if banded else {}),
        "fit": {
            "method": "split-fit: per-type kappa by minimum log loss on a split of the model's own reads",
            "variant": variant,
            "reads": use,
            "objective": {t: "log_loss" for t in TYPES},
            "kappa": kappa,
            "kappa_against_decisio_defaults": vs_default,
            "served_at": {t: (dev_served if pools[t][0] == "dev" else served)[t] for t in TYPES},
            "n": {t: checks[t]["n_fit"] for t in TYPES},
            "swap_validated_dev": swap_validated(
                dev_reads, dev_keys, use=field, bands=bands, min_band_items=min_band_items
            ),
            **({"choice_bands": band_notes} if band_notes else {}),
            "checks": checks,
            "data": data,
        },
    }
