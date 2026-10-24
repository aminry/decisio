# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The split fit (`decisio.serve.split_fit`, `python -m decisio.serve.adapter_temperatures fit-split`): the
per-type temperature fit from a model's stored reads, written as the file #207 loads. Synthetic reads; the
reproduction of the original fits from the real reads is in `runs/2026-10-18_adapter-fit/`.

F1  the kappa is the argmin of the mean log loss over the fits' grid, equal to a plain loop (ragged option counts, a
    probability under the floor included)
F2  the variants: half0 and half1 fit on opposite halves of the ids (even sha256 prefix is half 0) and report the
    other as held-out, full uses all, dev takes the choice fit from the dev pool; the choice pool is H, the others
    the dev pool
F3  the temperature is the served one over kappa; where the dev pool was served at other temperatures (S18's score
    items at the choice temperature) it is that one; fitted on p_plain it is 1 over kappa and agrees with the probs path
F4  the file loads as an adapter file and the CLI writes it; a read without p_plain, an unknown variant and an empty
    pool are refused with the reason

  uv run pytest -q tests/unit/test_split_fit.py
"""

import gzip
import json
import math

import numpy as np
import pytest

from decisio.serve import adapter_temperatures as at
from decisio.serve import split_fit as sf
from decisio.serve.temperature import apply_temperature


def reference_kappa(P, gold):
    """The plain loop: rescale, floor, mean log loss, smallest grid point on a tie."""
    best = min(
        (
            -np.mean([math.log(max(sf.rescale(p, k)[g], sf.FLOOR)) for p, g in zip(P, gold)]),
            k,
        )
        for k in sf.GRID
    )
    return float(best[1])


def synth(n, k, t_true, t_served, seed, tiny=False):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        z = rng.normal(0, 4, k)
        raw = np.exp(z) / np.exp(z).sum()
        gold = int(rng.choice(k, p=apply_temperature(raw, t_true)))
        if tiny:
            raw = np.where(raw < 1e-6, 1e-15, raw)
            raw = raw / raw.sum()
        out.append({"raw": raw, "gold": gold, "served": apply_temperature(raw, t_served)})
    return out


def reads_and_keys(items, prefix, types=None):
    reads, keys = {}, {}
    for n, it in enumerate(items):
        i = f"{prefix}-{n}"
        reads[i] = {"probs": it["served"].tolist(), "p_plain": it["raw"].tolist()}
        keys[i] = {"gold": it["gold"], **({"type": types} if types else {})}
    return reads, keys


def pools(t_true=2.0, seed=1):
    """An H pool of choice items and a dev pool of choice, yes/no and score items, read at decisio's defaults."""
    h = synth(300, 4, t_true, 4.672, seed)
    d = synth(60, 5, t_true, 4.672, seed + 1) + synth(80, 2, 3.0, 5.252, seed + 2) + synth(70, 5, 4.0, 5.252, seed + 3)
    types = ["choice"] * 60 + ["noul"] * 80 + ["score"] * 70
    H, hk = reads_and_keys(h, "h")
    D, dk = {}, {}
    for n, (it, ty) in enumerate(zip(d, types)):
        r, k = reads_and_keys([it], f"d{n}", ty)
        D.update(r), dk.update(k)
    return H, hk, D, dk


def test_f1_the_kappa_is_the_argmin_over_the_grid_and_equals_a_plain_loop():
    items = synth(120, 4, 2.0, 4.672, 5) + synth(40, 7, 2.0, 4.672, 6, tiny=True)  # ragged, with tiny probabilities
    P = [it["served"] for it in items]
    y = [it["gold"] for it in items]
    assert sf.fit_kappa(P, y) == reference_kappa(P, y)
    clean = synth(200, 5, 2.0, 4.672, 7)
    k = sf.fit_kappa([c["served"] for c in clean], [c["gold"] for c in clean])
    assert 4.672 / k == pytest.approx(2.0, rel=0.15)  # the calibrated temperature of the synthetic data


def test_f2_the_variants_split_by_the_id_hash():
    H, hk, D, dk = pools()
    assert {sf.half(i) for i in H} == {0, 1} and 0.4 < np.mean([sf.half(i) for i in H]) < 0.6
    v0 = sf.fit_split(H, hk, D, dk, variant="half0")
    v1 = sf.fit_split(H, hk, D, dk, variant="half1")
    full = sf.fit_split(H, hk, D, dk, variant="full")
    dev = sf.fit_split(H, hk, D, dk, variant="dev")
    n0 = sum(sf.half(i) == 0 for i in H)
    assert v0["fit"]["checks"]["choice"]["n_fit"] == n0 and v0["fit"]["checks"]["choice"]["heldout"]["n"] == len(H) - n0
    assert v1["fit"]["checks"]["choice"]["n_fit"] == len(H) - n0
    assert full["fit"]["checks"]["choice"]["n_fit"] == len(H) and "heldout" not in full["fit"]["checks"]["choice"]
    assert dev["fit"]["checks"]["choice"]["pool"] == "dev" and dev["fit"]["checks"]["choice"]["n_fit"] == 60
    assert full["fit"]["checks"]["noul"]["pool"] == "dev" and full["fit"]["checks"]["score"]["n_fit"] == 70
    dh = sf.fit_split(H, hk, D, dk, variant="devhalf0")
    nd0 = sum(sf.half(i) == 0 for i in D if dk[i]["type"] == "choice")
    assert dh["fit"]["checks"]["choice"]["pool"] == "dev" and dh["fit"]["checks"]["choice"]["n_fit"] == nd0
    assert dh["fit"]["checks"]["choice"]["heldout"]["n"] == 60 - nd0
    assert dh["fit"]["kappa"]["noul"] == v0["fit"]["kappa"]["noul"]  # yes/no and score are the half0 fits
    h = v0["fit"]["checks"]["choice"]["heldout"]
    assert (
        h["log_loss_fit"] < h["log_loss_kappa1"] and h["ece_fit"] < h["ece_kappa1"]
    )  # the fit helps on the other half
    # the half0 fit really uses only the even half
    ids = [i for i in sorted(hk) if sf.half(i) == 0]
    k = sf.fit_kappa([H[i]["probs"] for i in ids], [hk[i]["gold"] for i in ids])
    assert v0["fit"]["kappa"]["choice"] == round(k, 4)


def test_f3_temperatures_are_the_served_ones_over_kappa_and_follow_the_reads():
    H, hk, D, dk = pools()
    f = sf.fit_split(H, hk, D, dk, variant="full")
    for t, base in (("choice", 4.672), ("noul", 5.252), ("score", 5.252)):
        assert f["temperatures"][t] == pytest.approx(base / f["fit"]["kappa"][t], abs=2e-3)
    # the dev score items served at the choice temperature (S18): the dev pool's own served temperature is used
    ds = reads_and_keys(synth(70, 5, 4.0, 4.672, 9), "s", "score")
    D2, dk2 = (
        {**{k: v for k, v in D.items() if dk[k]["type"] != "score"}, **ds[0]},
        {**{k: v for k, v in dk.items() if v["type"] != "score"}, **ds[1]},
    )
    g = sf.fit_split(H, hk, D2, dk2, variant="full", dev_served={"score": 4.672})
    k = g["fit"]["kappa"]["score"]
    assert g["temperatures"]["score"] == pytest.approx(4.672 / k, abs=2e-3)
    assert g["fit"]["kappa_against_decisio_defaults"]["score"] == pytest.approx(score_against_default(k), abs=2e-3)
    # on the plain readout the temperature is 1 / kappa, and the two paths agree to a grid step
    p = sf.fit_split(H, hk, D, dk, variant="full", use="p_plain")
    for t in ("choice", "noul", "score"):
        assert p["fit"]["served_at"][t] == 1.0
        assert p["temperatures"][t] == pytest.approx(f["temperatures"][t], rel=0.03), t


def score_against_default(k):  # kappa against the 5.252 default for reads served at 4.672
    return k * 5.252 / 4.672


def test_f4_the_file_loads_and_the_cli_writes_it_and_bad_input_is_refused(tmp_path, capsys):
    H, hk, D, dk = pools()
    f = sf.fit_split(H, hk, D, dk, variant="half0", data="synthetic")
    path = tmp_path / "decisio_adapter.json"
    path.write_text(json.dumps(f))
    assert at.read(path)["temperatures"] == f["temperatures"] and f["fit"]["data"] == "synthetic"

    # the CLI, from JSONL.gz reads and keys files
    def dump(reads, name):
        with gzip.open(tmp_path / name, "wt") as fh:
            for i, r in reads.items():
                fh.write(json.dumps({"id": i, **r}) + "\n")

    dump(H, "h.jsonl.gz"), dump(D, "d.jsonl.gz")
    (tmp_path / "hk.json").write_text(json.dumps(hk)), (tmp_path / "dk.json").write_text(json.dumps(dk))
    out = tmp_path / "out.json"
    assert (
        at.main(
            [
                "fit-split",
                "--h",
                str(tmp_path / "h.jsonl.gz"),
                "--h-keys",
                str(tmp_path / "hk.json"),
                "--dev",
                str(tmp_path / "d.jsonl.gz"),
                "--dev-keys",
                str(tmp_path / "dk.json"),
                "--variant",
                "half0",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text())["temperatures"] == f["temperatures"]
    with pytest.raises(ValueError, match="variant"):
        sf.fit_split(H, hk, D, dk, variant="third")
    bare = {i: {"probs": r["probs"], "p_plain": None} for i, r in H.items()}
    with pytest.raises(ValueError, match="--debug-readout"):
        sf.fit_split(bare, hk, D, dk, use="p_plain")
    with pytest.raises(ValueError, match="no noul items"):
        sf.fit_split(H, hk, {i: r for i, r in D.items() if dk[i]["type"] != "noul"}, dk)


def test_f5_the_ship_gates_ece_is_equal_width_and_pooled_out_of_half():
    """The ship gates (the ship gates' note) take the ECE with ten equal-width bins, pooled over the dev pool's items
    of a
    type, each scored with the kappa fitted on the other half. Held-out figures per half use equal-count bins."""
    assert sf.ece_width10([0.95, 0.95, 0.55, 0.55], [1, 1, 1, 0]) == pytest.approx(0.5 * 0.05 + 0.5 * 0.05)
    assert sf.ece_width10([1.0] * 4, [1, 1, 1, 1]) == 0.0 and sf.ece_width10([0.0, 0.05], [0, 0]) == pytest.approx(
        0.025
    )
    H, hk, D, dk = pools()
    sv = sf.swap_validated(D, dk)
    assert set(sv) == {"choice", "noul", "score"} and sv["choice"]["n"] == 60
    ids = [i for i in sorted(dk) if dk[i]["type"] == "choice"]
    kap = {
        h: sf.fit_kappa(
            [D[i]["probs"] for i in ids if sf.half(i) == h], [dk[i]["gold"] for i in ids if sf.half(i) == h]
        )
        for h in (0, 1)
    }
    Q = [sf.rescale(D[i]["probs"], kap[1 - sf.half(i)]) for i in ids]
    want = sf.ece_width10([q.max() for q in Q], [int(q.argmax() == dk[i]["gold"]) for q, i in zip(Q, ids)])
    assert sv["choice"]["ece10_swap_validated"] == round(want, 4)
    assert sv["choice"]["kappa_ratio"] == round(max(kap.values()) / min(kap.values()), 3)
    f = sf.fit_split(H, hk, D, dk)
    assert f["fit"]["swap_validated_dev"] == sv  # reported with every fit, whatever the variant
    # the swap judges a calibrating fit: scaled ECE under the unscaled one on data served at the wrong temperature
    assert sv["choice"]["ece10_swap_validated"] < sv["choice"]["ece10_unscaled"]
