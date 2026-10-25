# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""`hub_fp8 merge --gptq CALIBRATION`: the merge rounds with error compensation, on a tiny Gemma 4 text checkpoint (the
calibration rendering is replaced by random token rows: the tiny model has no tokenizer):

I1  the compensated merge keeps the format: the same scales as the plain merge, FP8 codes on the grid, the loader's
    requantization the identity, written and loaded fingerprints identical; some codes differ from round to nearest
    and the layer-output error the report gives is lower
I2  it composes with --keep-bf16: kept groups are the stochastic bf16 of the fp32 merge, the rest compensated
I3  the report: MERGE_FP8.json carries the method, settings, the calibration file's digest and the errors; the tuned
    record (hub.merge_facts) carries the method, settings and digest, not the counts
I4  refusals: a null merge, a calibration that renders to nothing
I5  the command line takes the flags

  uv run pytest -q tests/unit/test_hub_fp8_gptq.py
"""

import gzip
import json

import pytest
import torch
from test_fp8_gptq_pass import checkpoint, sequences
from test_hub_fp8 import FP8, read_all

from decisio import fp8_gptq_pass, hub, hub_fp8


@pytest.fixture
def setup(tmp_path, monkeypatch):
    d, cfg, model = checkpoint(tmp_path)
    calib = tmp_path / "calib.jsonl.gz"
    with gzip.open(calib, "wt") as f:
        f.write(json.dumps({"state": "x", "questions": {}}) + "\n")
    seqs = sequences(40, 24, 5)
    monkeypatch.setattr(fp8_gptq_pass, "render_calibration", lambda *a, **k: seqs)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: object())
    # an adapter of the tiny layers' projections
    from test_hub_fp8_merge import adapter

    src = hub_fp8.Source(d)
    names = {n: tuple(src.get(n).shape) for n in src.map if hub_fp8.PROJ.match(n)}
    ad = adapter(tmp_path, names, layout="multimodal", seed=11)
    return d, calib, ad, src


def test_i1_the_compensated_merge_keeps_the_format(setup, tmp_path):
    d, calib, ad, src = setup
    plain = tmp_path / "plain"
    hub_fp8.merge(d, ad, plain)
    out = tmp_path / "gptq"
    meta = hub_fp8.merge(d, ad, out, gptq=hub_fp8.Gptq(calib))
    a, b = read_all(plain), read_all(out)
    assert set(a) == set(b)
    moved = 0
    for n in a:
        if a[n].dtype == FP8:
            assert b[n].dtype == FP8 and a[n].shape == b[n].shape
            moved += int((a[n].view(torch.uint8) != b[n].view(torch.uint8)).sum())
        else:  # scales and every other tensor: the plain merge's, byte for byte
            assert torch.equal(a[n], b[n]), n
    assert moved > 0  # some codes are a step from where round to nearest puts them
    assert meta["loader_changed_elements"] == 0 and meta["written_vs_loaded_differ"] == []
    g = meta["gptq"]
    assert g["output_error_compensated"] < g["output_error_round_to_nearest"]
    assert g["requests"] == 40 and g["tokens"] == 40 * 24


def test_i2_it_composes_with_keep_bf16(setup, tmp_path):
    d, calib, ad, src = setup
    out = tmp_path / "keep"
    keep = hub_fp8.parse_keep(["layers:1"])
    meta = hub_fp8.merge(d, ad, out, gptq=hub_fp8.Gptq(calib), keep=keep, seed=3)
    b = read_all(out)
    plain = tmp_path / "plain"
    hub_fp8.merge(d, ad, plain, keep=keep, seed=3)
    a = read_all(plain)
    for n in b:
        if ".layers.1." in n and n.endswith("_proj.weight"):
            assert b[n].dtype == torch.bfloat16 and torch.equal(a[n], b[n]), (
                n
            )  # kept: the stochastic bf16, as without gptq
    assert any(
        b[n].dtype == FP8 and not torch.equal(a[n].view(torch.uint8), b[n].view(torch.uint8))
        for n in b
        if n in a and a[n].dtype == FP8
    )
    assert meta["kept_bf16"]["groups"] and meta["gptq"]["method"].startswith("error-compensating")


def test_i3_the_report_and_the_tuned_record(setup, tmp_path):
    d, calib, ad, src = setup
    out = tmp_path / "out"
    hub_fp8.merge(d, ad, out, gptq=hub_fp8.Gptq(calib, damp=0.02, blocksize=64))
    report = json.loads((out / "MERGE_FP8.json").read_text())
    assert report["gptq"]["damp"] == 0.02 and report["gptq"]["blocksize"] == 64
    assert len(report["gptq"]["calibration_sha256"]) == 64
    facts = hub.merge_facts(report)
    assert set(facts["gptq"]) == {"method", "damp", "blocksize", "max_tokens", "calibration_sha256"}
    assert "requests" not in json.dumps(facts) and "tokens" not in facts["gptq"]
    assert "gptq" not in hub.merge_facts({k: v for k, v in report.items() if k != "gptq"})


def test_i4_refusals(setup, tmp_path, monkeypatch):
    d, calib, ad, src = setup
    with pytest.raises(ValueError, match="does not combine with --null"):
        hub_fp8.merge(d, ad, tmp_path / "a", null=True, gptq=hub_fp8.Gptq(calib))
    monkeypatch.setattr(fp8_gptq_pass, "render_calibration", lambda *a, **k: [])
    with pytest.raises(ValueError, match="no calibration request renders"):
        hub_fp8.merge(d, ad, tmp_path / "b", gptq=hub_fp8.Gptq(calib))


def test_i5_the_command_line(setup, tmp_path, capsys):
    d, calib, ad, src = setup
    rc = hub_fp8.main(
        [
            "merge",
            "--source",
            str(d),
            "--adapter",
            str(ad),
            "--out",
            str(tmp_path / "o"),
            "--gptq",
            str(calib),
            "--gptq-damp",
            "0.05",
            "--gptq-blocksize",
            "32",
            "--gptq-limit",
            "8",
        ]
    )
    assert rc == 0 and "identical" in capsys.readouterr().out
    assert json.loads((tmp_path / "o" / "MERGE_FP8.json").read_text())["gptq"]["blocksize"] == 32
