# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The calculator's reader from a tuned repository's `defaults.calculator_reader` (docs/calculator.md, "From the
model's repository"). No network, no GPU, no torch: stub taggers and missing paths stand in for the model.

R1  the file's check: the reader, the tagger's form (a Hub id at a full commit, or a directory path), all four
    thresholds as numbers from 0 to 1, and defaults.calculator set
R2  at start the repository's reader applies where no reader flag is given and the block is shown; any reader flag
    keeps it out; with no record and no flag the rule reader serves as before
R3  thresholds from the repository are a dict: loaded, named in the start line and in the task fingerprint
R4  a tagger that cannot be loaded never stops the start: the rule reader serves, the start line and /health say why,
    and the fingerprints are the rule reader's; the cascade loads the same on every backend
R5  `tuned-config` writes the record, checked as above
R6  an older decisio, which does not know the key, still loads the repository and serves with the rule reader

  uv run pytest -q tests/unit/test_calculator_reader_config.py
"""

import json
import logging
import types

import pytest
from test_calculator import FREIGHT, FREIGHT_BLOCK
from test_calculator_cascade_serving import make
from test_calculator_serving import ask
from test_hub_tuned import ADAPTER_SHA, TEMPS, build
from test_over_limit import plain_rows  # noqa: F401  (the fixture applies to this module)

from decisio import hub
from decisio.calculator.cascade import StubTagger
from decisio.serve import calculator_serving as calcs
from decisio.serve import vllm_engine as ve

COMMIT = "0123456789abcdef0123456789abcdef01234567"
THRESHOLDS = {"role": 0.915, "link": 0.54, "flag": 0.82, "band": 0.02}
CASCADE = {"reader": "cascade", "tagger": f"owner/tagger@{COMMIT}", "thresholds": dict(THRESHOLDS)}


def config_with(reader=CASCADE, calculator="calc-v1"):
    c = build()
    if calculator is not None:
        c["defaults"]["calculator"] = calculator
    if reader is not None:
        c["defaults"]["calculator_reader"] = reader
    return c


def repo_of(config):
    return types.SimpleNamespace(defaults=config["defaults"], name="r", revision="abc", defaults_sha256="d")


def start_args(**flags):
    """The server's flags as parsed (`build_parser`), with `flags` given on the command line."""
    argv = []
    for name, value in flags.items():
        argv += [f"--{name.replace('_', '-')}", value]
    return ve.build_parser().parse_args(argv)


# ---- R1 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reader",
    [
        CASCADE,
        {"reader": "rules"},
        {**CASCADE, "tagger": "./taggers/v1"},  # a local directory, named as a path
        {**CASCADE, "tagger": "/srv/taggers/v1"},
        {**CASCADE, "thresholds": {"role": 0.5, "link": 1, "flag": 0.9, "band": 0}},
    ],
)
def test_r1_a_well_formed_reader_is_accepted(reader):
    hub.check_sections(config_with(reader))


@pytest.mark.parametrize(
    ("reader", "message"),
    [
        ("cascade", "defaults.calculator_reader must be an object"),
        ({"reader": "model"}, 'reader must be "rules" or "cascade"'),
        ({"tagger": CASCADE["tagger"]}, 'reader must be "rules" or "cascade"'),
        ({"reader": "rules", "tagger": CASCADE["tagger"]}, "tagger belong to the cascade reader"),
        ({**CASCADE, "tagger": None}, "tagger must name the cascade reader's tagger"),
        ({**CASCADE, "tagger": ""}, "tagger must name the cascade reader's tagger"),
        ({**CASCADE, "tagger": "owner/tagger@main"}, "full 40-character commit hash"),
        ({**CASCADE, "tagger": "owner/tagger"}, "full 40-character commit hash"),
        ({**CASCADE, "tagger": f"owner/tagger@{COMMIT[:12]}"}, "full 40-character commit hash"),
        ({**CASCADE, "tagger": f"tagger@{COMMIT}"}, "full 40-character commit hash"),
        ({k: v for k, v in CASCADE.items() if k != "thresholds"}, "thresholds must be an object"),
        ({**CASCADE, "thresholds": [0.9, 0.9, 0.9, 0.0]}, "thresholds must be an object"),
        ({**CASCADE, "thresholds": {"role": 0.9, "link": 0.9, "flag": 0.9}}, "thresholds: missing band"),
        ({**CASCADE, "thresholds": {**THRESHOLDS, "role": "0.9"}}, "thresholds.role must be a number from 0 to 1"),
        ({**CASCADE, "thresholds": {**THRESHOLDS, "flag": True}}, "thresholds.flag must be a number from 0 to 1"),
        ({**CASCADE, "thresholds": {**THRESHOLDS, "link": 1.5}}, "thresholds.link must be a number from 0 to 1"),
        ({**CASCADE, "thresholds": {**THRESHOLDS, "band": -0.1}}, "thresholds.band must be a number from 0 to 1"),
        ({**CASCADE, "thresholds": {**THRESHOLDS, "role": float("nan")}}, "thresholds.role must be a number"),
        ({**CASCADE, "thresholds": {**THRESHOLDS, "role": 0.3}}, "at least 0.5"),  # the reader's own rule
        ({**CASCADE, "thresholds": {**THRESHOLDS, "role": 0.99, "band": 0.02}}, "leave room for the band"),
    ],
)
def test_r1_a_malformed_reader_is_refused(reader, message):
    with pytest.raises(ValueError, match=message):
        hub.check_sections(config_with(reader))


def test_r1_the_reader_needs_the_block_format():
    with pytest.raises(ValueError, match="defaults.calculator_reader needs defaults.calculator"):
        hub.check_sections(config_with(calculator=None))
    assert "calculator_reader" in hub.DEFAULTS_KEYS


def test_r1_unknown_keys_inside_are_ignored_with_a_warning(caplog):
    reader = {**CASCADE, "dtype": "bfloat16", "thresholds": {**THRESHOLDS, "rows": 238}}
    with caplog.at_level(logging.WARNING, logger="decisio.hub"):
        hub.check_sections(config_with(reader))
    assert "'dtype'" in caplog.text and "'thresholds.rows'" in caplog.text


# ---- R2 -------------------------------------------------------------------------------------------------------------


def test_r2_the_repository_reader_applies_without_flags():
    args = start_args()
    ve.apply_repository_defaults(args, repo_of(config_with()))
    assert (args.calculator_version, args.calculator_reader, args.calculator_tagger) == (
        "calc-v1",
        "cascade",
        CASCADE["tagger"],
    )
    assert args.calculator_thresholds == THRESHOLDS and args.calculator_reader_source == "repository"
    assert args.repository_defaults["calculator_reader"] == CASCADE  # the REPOSITORY DEFAULTS line shows it


@pytest.mark.parametrize(
    ("flags", "reader"),
    [
        ({"calculator_reader": "rules"}, "rules"),
        ({"calculator_reader": "cascade", "calculator_tagger": "./mine"}, "cascade"),
        ({"calculator_thresholds": "mine.json"}, "rules"),  # a partial set never pairs with the repository's tagger
    ],
)
def test_r2_a_reader_flag_beats_the_repository(flags, reader):
    args = start_args(**flags)
    ve.apply_repository_defaults(args, repo_of(config_with()))
    assert args.calculator_reader == reader and args.calculator_reader_source == "flags"
    assert args.calculator_tagger == flags.get("calculator_tagger")
    assert args.calculator_thresholds == flags.get("calculator_thresholds")
    assert "calculator_reader" not in args.repository_defaults
    assert args.repository_defaults["calculator_reader_set_by"].startswith("--calculator-")


@pytest.mark.parametrize("flags", [{"calculator": "off"}, {"mode": "packed"}])
def test_r2_no_block_no_reader(flags):
    args = start_args(**flags)
    ve.apply_repository_defaults(args, repo_of(config_with()))
    assert args.calculator_version is None and args.calculator_reader == "rules"
    assert args.calculator_reader_source is None and "calculator_reader" not in args.repository_defaults


def test_r2_without_a_record_the_rule_reader_serves_as_before():
    for config in (config_with(reader=None), config_with(reader=None, calculator=None)):
        args = start_args()
        ve.apply_repository_defaults(args, repo_of(config))
        assert args.calculator_reader == "rules" and args.calculator_tagger is None
        assert args.calculator_thresholds is None and args.calculator_reader_source is None
        assert not any(k.startswith("calculator_reader") for k in args.repository_defaults)
        assert ve.prepare_calculator_reader(args) is None and args.calculator_reader_id is None
    args = types.SimpleNamespace(mode="separate", calculator="auto")  # a caller's namespace without the reader flags
    ve.apply_repository_defaults(args, repo_of(config_with(reader=None)))
    assert (args.calculator_reader, args.calculator_tagger, args.calculator_thresholds) == ("rules", None, None)


# ---- R3 -------------------------------------------------------------------------------------------------------------


def test_r3_thresholds_from_a_dict(tmp_path):
    t = calcs.load_thresholds(THRESHOLDS)
    assert (t.role, t.link, t.flag, t.band) == (0.915, 0.54, 0.82, 0.02)
    with pytest.raises(ValueError, match="defaults.calculator_reader.thresholds: missing link"):
        calcs.load_thresholds({"role": 0.9, "flag": 0.9})
    assert calcs.check_flags("cascade", str(tmp_path), THRESHOLDS, "auto") is None
    r = calcs.load_reader("cascade", str(tmp_path), THRESHOLDS, "cpu", "vllm", loader=lambda *a: StubTagger({}))
    assert r.threshold_facts() == THRESHOLDS
    assert r.start_line().endswith("thresholds from the repository's defaults.calculator_reader")


def test_r3_the_task_fingerprint_names_dict_thresholds(tmp_path):
    ident = calcs.reader_identity("cascade", CASCADE["tagger"], THRESHOLDS)
    assert ident.startswith(f"cascade@{COMMIT};thresholds=sha256:")
    assert calcs.reader_identity("cascade", CASCADE["tagger"], dict(reversed(THRESHOLDS.items()))) == ident
    assert calcs.reader_identity("cascade", CASCADE["tagger"], {**THRESHOLDS, "band": 0.03}) != ident
    thr = tmp_path / "thr.json"  # a file keeps its own identity, its bytes, as before
    thr.write_text(json.dumps(THRESHOLDS))
    assert calcs.reader_identity("cascade", CASCADE["tagger"], str(thr)) != ident


# ---- R4 -------------------------------------------------------------------------------------------------------------


def cascade_args(tagger, backend="vllm"):
    args = start_args(backend=backend)
    ve.apply_repository_defaults(args, repo_of(config_with({**CASCADE, "tagger": tagger})))
    return args


def test_r4_a_missing_tagger_falls_back_to_the_rule_reader(tmp_path, capsys):
    args = cascade_args(str(tmp_path / "no-such-tagger"))
    fallback = ve.prepare_calculator_reader(args)
    out = capsys.readouterr().out
    assert out.startswith("CALCULATOR READER rules (fallback): the cascade reader's tagger")
    assert "not a directory" in out and "reads states with the rule reader" in out
    assert args.calculator_reader == "rules" and args.calculator_reader_id is None
    assert ve.load_calculator_reader(args, fallback) is fallback and capsys.readouterr().out == ""
    client, so, eng = make(fallback)
    r = ask(client)
    assert eng.seen[-1] == f"{FREIGHT}\n{FREIGHT_BLOCK}"
    assert r.headers["x-decisio-calculator"] == "shown; lines=9; version=calc-v1"  # the rule reader's, no path
    health = client.get("/health").json()["systemone"]["calculator"]
    assert health["reader"] == "rules" and "paths" not in health
    assert health["fallback"]["wanted"] == "cascade" and "not a directory" in health["fallback"]["reason"]
    assert so.stack_facts()["serving"] == make()[1].stack_facts()["serving"]  # the rule reader's fingerprint


def test_r4_files_that_do_not_match_their_sums_fall_back(tmp_path, monkeypatch):
    monkeypatch.setattr(calcs, "_available", lambda d: d == "cpu")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "SHA256SUMS").write_text("0" * 64 + "  model.safetensors\n")
    args = cascade_args(str(tmp_path))
    fallback = ve.prepare_calculator_reader(args)
    assert "does not match SHA256SUMS" in fallback.facts()["fallback"]["reason"]


def test_r4_a_missing_torch_falls_back(tmp_path, monkeypatch):
    def no_torch(device):
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(calcs, "_available", no_torch)
    args = cascade_args(str(tmp_path))
    reason = ve.prepare_calculator_reader(args).facts()["fallback"]["reason"]
    assert reason.startswith("ImportError: No module named 'torch'") and "needs torch and transformers" in reason


def test_r4_a_refused_download_falls_back(monkeypatch):
    def refused(name, revision=None):
        raise OSError("403 Forbidden: the repository is gated")

    monkeypatch.setattr(calcs, "_available", lambda d: d == "cpu")
    monkeypatch.setattr("decisio.calculator.cascade.tagger.resolve", refused)
    args = cascade_args(CASCADE["tagger"])
    assert "403 Forbidden" in ve.prepare_calculator_reader(args).facts()["fallback"]["reason"]


def test_r4_a_load_that_fails_after_the_engine_falls_back_and_retargets_the_tasks(tmp_path, monkeypatch, capsys):
    from decisio.readout.letters import DEFAULT_FORMAT
    from decisio.serve.tasks import TaskStore

    monkeypatch.setattr(calcs, "prepare_tagger", lambda tagger, device, backend: tmp_path)
    args = cascade_args(CASCADE["tagger"])
    args.served_name, args.model, args.model_identity = "s", "m", None
    args.pad_to, args.pad_where, args.pad_policy = None, "front", "always"
    args.hide_index_keys, args.desnake_labels = True, True
    args.family = types.SimpleNamespace(key="gemma-4-12b")
    assert ve.prepare_calculator_reader(args) is None and args.calculator_reader_id.startswith("cascade@")
    store = TaskStore(fingerprint=ve.task_store_fingerprint(args, DEFAULT_FORMAT))
    cascade_fp = store.fingerprint

    def boom(path, revision, device):
        raise RuntimeError("CUDA out of memory")

    reader = ve.load_calculator_reader(args, None, store, DEFAULT_FORMAT, loader=boom)
    assert not reader.cascade and "CUDA out of memory" in reader.facts()["fallback"]["reason"]
    assert capsys.readouterr().out.startswith("CALCULATOR READER rules (fallback)")
    assert args.calculator_reader_id is None and store.fingerprint != cascade_fp
    assert store.fingerprint == ve.task_store_fingerprint(args, DEFAULT_FORMAT)


@pytest.mark.parametrize("backend", ["vllm", "mlx", "hf"])
def test_r4_the_cascade_loads_on_every_backend(tmp_path, monkeypatch, backend, capsys):
    monkeypatch.setattr(calcs, "_available", lambda d: d == "cpu")
    monkeypatch.setattr(calcs, "prepare_tagger", lambda tagger, device, b: tmp_path)
    seen = []

    def loader(path, revision, device):
        seen.append((path, revision, device))
        return StubTagger({})

    args = cascade_args(CASCADE["tagger"], backend=backend)
    assert ve.prepare_calculator_reader(args) is None
    reader = ve.load_calculator_reader(args, None, loader=loader)
    assert reader.cascade and seen == [(str(tmp_path), None, "cpu")]  # the files made ready, on the CPU here
    assert reader.tagger_facts == {"name": "owner/tagger", "revision": COMMIT, "device": "cpu"}
    assert reader.threshold_facts() == THRESHOLDS
    assert capsys.readouterr().out.startswith(f"CALCULATOR READER cascade: tagger owner/tagger@{COMMIT} on cpu")


def test_r4_a_prepared_local_tagger(tmp_path, monkeypatch):
    """`prepare_tagger` on a real directory: the sums hold and the tagger's format is the one this decisio reads."""
    pytest.importorskip("transformers")
    from decisio.calculator.cascade.model import FORMAT, LABELS

    monkeypatch.setattr(calcs, "_available", lambda d: d == "cpu")
    (tmp_path / "cascade_tagger.json").write_text(json.dumps({"format": FORMAT, "labels": list(LABELS)}))
    assert calcs.prepare_tagger(str(tmp_path), "auto", "mlx") == tmp_path
    (tmp_path / "cascade_tagger.json").write_text(json.dumps({"format": FORMAT + 1, "labels": list(LABELS)}))
    with pytest.raises(ValueError, match="unknown format"):
        calcs.prepare_tagger(str(tmp_path), "auto", "mlx")


# ---- R5 -------------------------------------------------------------------------------------------------------------


def test_r5_tuned_config_writes_the_reader(tmp_path, capsys):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "model-00001-of-00004.safetensors").write_bytes(b"w")
    temps = tmp_path / "decisio_adapter.json"
    temps.write_text(json.dumps(TEMPS))
    report = tmp_path / "MERGE_FP8.json"
    report.write_text(json.dumps({"adapter_sha256": ADAPTER_SHA, "adapter_scale": 2.0, "adapted_tensors": 840}))
    thr = tmp_path / "thresholds.json"
    thr.write_text(json.dumps({**THRESHOLDS, "rows_with_a_case": 238}))
    out = tmp_path / "out.json"
    argv = [
        "tuned-config", "--base", "gemma-4-31b", "--temperatures", str(temps), "--adapter-dir", str(adapter),
        "--adapter-revision", "abc", "--merge-report", str(report), "--weights-dir", str(merged), "--out", str(out),
        "--no-noul-commit", "--calculator", "calc-v1",
    ]  # fmt: skip
    reader = ["--calculator-reader", "cascade", "--calculator-tagger", CASCADE["tagger"]]
    assert hub.main([*argv, *reader, "--calculator-thresholds", str(thr)]) == 0
    written = json.loads(out.read_text())["defaults"]["calculator_reader"]
    assert written == CASCADE  # only the four values of the thresholds file
    out.unlink()
    for extra, message in (
        ([*reader], "thresholds must be an object"),
        ([*reader[:3], "owner/tagger@main", "--calculator-thresholds", str(thr)], "full 40-character commit"),
        (["--calculator-tagger", CASCADE["tagger"]], "need --calculator-reader cascade"),
    ):
        capsys.readouterr()
        assert hub.main([*argv, *extra]) == 1
        assert message in capsys.readouterr().out and not out.exists()
    no_format = [a for a in argv if a not in ("--calculator", "calc-v1")]
    assert hub.main([*no_format, *reader, "--calculator-thresholds", str(thr)]) == 1
    assert "needs defaults.calculator" in capsys.readouterr().out


# ---- R6 -------------------------------------------------------------------------------------------------------------


def test_r6_an_older_decisio_ignores_the_key(monkeypatch, caplog):
    monkeypatch.setattr(hub, "DEFAULTS_KEYS", hub.DEFAULTS_KEYS - {"calculator_reader"})  # as 0.50.1 to this release
    config = config_with()
    with caplog.at_level(logging.WARNING, logger="decisio.hub"):
        hub.check_sections(config)  # loads: the key is ignored, with a warning
    assert "'calculator_reader'" in caplog.text and "ignored" in caplog.text
    args = types.SimpleNamespace(mode="separate", calculator="auto")  # its server knew no reader from a repository
    ve.apply_repository_defaults(args, repo_of(config))
    assert args.repository_defaults["ignored_keys"] == ["calculator_reader"]
    assert args.calculator_version == "calc-v1"
