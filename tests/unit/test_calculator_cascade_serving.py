# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The cascade reader in the request path (decisio.serve.calculator_serving.Reader; docs/calculator.md, "The cascade
reader (preview)"). A stub tagger stands in for the model: no download, no torch.

C1  a cascade server shows a block the rule reader cannot read, says which path accepted it, and times the reading
C2  the rule reader (the default) is unchanged: the same header, no timing stage, the same /health and fingerprint
C3  without thresholds only the rule reader's agreement and the stated figures accept; a request's own amounts bypass
    the reading; registration reads use the same reader
C4  /health counts per path, and never any state text
C5  the serving-stack fingerprint carries the reader only with the cascade
C6  the start: flag checks, the tagger's form, the thresholds file, the device, the note where no block is shown

  uv run pytest -q tests/unit/test_calculator_cascade_serving.py
"""

import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from test_calculator import FREIGHT, FREIGHT_BLOCK
from test_calculator_cascade import EXPECTED, STATE, TH, tags_for
from test_calculator_input import FREIGHT_INPUT
from test_calculator_serving import Engine, ask
from test_over_limit import plain_rows  # noqa: F401  (the fixture applies to this module)

from decisio.calculator import from_input
from decisio.calculator.cascade import StubTagger
from decisio.serve import calculator_serving as calcs
from decisio.serve.systemone import SystemOne
from decisio.serve.tasks import TaskStore
from decisio.serve.vllm_engine import build_parser, make_app

BLOCK = from_input(EXPECTED).text
FACTS = {"name": "owner/tagger", "revision": "a" * 40, "device": "cpu"}


def cascade(thresholds=TH, answers=None, facts=FACTS):
    tagger = StubTagger(answers if answers is not None else {STATE: tags_for()})
    return calcs.Reader("cascade", tagger, facts, thresholds, "thresholds.json" if thresholds else None)


def make(reader=None, calculator="calc-v1"):
    eng = Engine(limit=10_000_000, layout="plain")
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"), calculator=calculator, calculator_reader=reader)
    return TestClient(make_app(eng, so)), so, eng


def stages(r) -> dict:
    return dict(kv.split("=") for kv in r.headers.get("x-decisio-stages", "").split(";") if kv)


# ---- C1 -------------------------------------------------------------------------------------------------------------


def test_c1_the_cascade_shows_a_block_the_rule_reader_cannot_read():
    assert isinstance(make()[1].read_state(None, STATE, examples=True), str)  # the rule reader: no block
    client, so, eng = make(cascade())
    r = ask(client, state=STATE)
    assert r.status_code == 200, r.text
    assert eng.seen[-1] == f"{STATE}\n{BLOCK}"
    assert r.headers["x-decisio-calculator"] == f"shown; lines={BLOCK.count(chr(10))}; version=calc-v1; path=threshold"
    assert float(stages(r)["calculator"]) >= 0


def test_c1_a_state_both_read_is_the_rule_reader_bytes():
    client, _, eng = make(cascade(answers={}))  # the tagger knows nothing: the rule reader's block stands
    r = ask(client)
    assert eng.seen[-1] == f"{FREIGHT}\n{FREIGHT_BLOCK}"
    assert r.headers["x-decisio-calculator"] == "shown; lines=9; version=calc-v1; path=rules_only"


def test_c1_a_tagger_failure_is_no_block_never_a_failed_request():
    def boom(state, context):
        raise RuntimeError("no model")

    client, _, eng = make(cascade(answers=boom))
    r = ask(client, state=STATE)
    assert r.status_code == 200 and eng.seen[-1] == STATE
    assert r.headers["x-decisio-calculator"] == "none; reason=cascade-tagger-error"


# ---- C2 -------------------------------------------------------------------------------------------------------------


def test_c2_the_rule_reader_is_unchanged():
    client, so, eng = make()
    assert not so.calc_reader.cascade and so.calc_reader.kwargs() == {}
    r = ask(client)
    assert eng.seen[-1] == f"{FREIGHT}\n{FREIGHT_BLOCK}"
    assert r.headers["x-decisio-calculator"] == "shown; lines=9; version=calc-v1"
    assert "calculator" not in stages(r)
    health = client.get("/health").json()["systemone"]["calculator"]
    assert health == {"version": "calc-v1", "requests": 1, "shown": 1, "from_input": 0, "none": {}}
    r = ask(client, state=STATE)
    assert eng.seen[-1] == STATE and "path=" not in r.headers["x-decisio-calculator"]


# ---- C3 -------------------------------------------------------------------------------------------------------------


def test_c3_without_thresholds_the_threshold_rule_is_off():
    client, so, eng = make(cascade(thresholds=None))
    r = ask(client, state=STATE)
    assert eng.seen[-1] == STATE
    assert r.headers["x-decisio-calculator"] == "none; reason=cascade-no-thresholds"
    assert so.calc_reader.facts()["thresholds"] == "off"
    assert "calculator" in stages(r)


def test_c3_the_request_amounts_bypass_the_reading():
    client, _, eng = make(cascade())
    r = ask(client, state=STATE, calculator_input=FREIGHT_INPUT)
    assert eng.seen[-1] == f"{STATE}\n{from_input(FREIGHT_INPUT).text}"
    assert "path=" not in r.headers["x-decisio-calculator"] and "calculator" not in stages(r)


def test_c3_registration_reads_use_the_same_reader():
    _, so, _ = make(cascade())
    assert str(so.read_state(None, STATE, examples=True)) == f"{STATE}\n{BLOCK}"
    assert so.read_state(None, STATE) == STATE  # outside a request and not an example: no block, as before


def test_c3_a_server_that_shows_no_block_reads_nothing():
    client, so, eng = make(cascade(), calculator=None)
    assert not so.calc_reader.cascade
    r = ask(client, state=STATE)
    assert eng.seen[-1] == STATE and "x-decisio-calculator" not in r.headers
    assert "calculator" not in client.get("/health").json()["systemone"]


# ---- C4 -------------------------------------------------------------------------------------------------------------


def test_c4_health_counts_per_path():
    client, _, _ = make(cascade())
    ask(client, state=STATE)
    ask(client)  # FREIGHT: the stub has no tags for it, so the rule reader's block stands
    ask(client, state="The parcel arrived late and the customer asks for a call back.")
    ask(client, calculator=False)
    health = client.get("/health").json()["systemone"]["calculator"]
    assert health["reader"] == "cascade"
    assert health["tagger"] == FACTS
    assert health["thresholds"] == {"role": 0.9, "link": 0.9, "flag": 0.9, "band": 0.02}
    assert health["paths"] == {"abstain": 1, "rules_only": 1, "threshold": 1}
    assert (health["requests"], health["shown"]) == (3, 2)
    assert health["none"] == {"cascade-unknown-text": 1, "off-by-request": 1}
    assert STATE not in json.dumps(health)


# ---- C5 -------------------------------------------------------------------------------------------------------------


def test_c5_the_fingerprint_carries_the_reader_only_with_the_cascade():
    rules = make()[1].stack_facts()["serving"]
    assert "calculator_reader" not in rules
    assert make(calcs.Reader())[1].stack_facts()["serving"] == rules
    on = make(cascade())[1].stack_facts()["serving"]
    th = cascade().threshold_facts()
    assert on["calculator_reader"] == {"reader": "cascade", "tagger": "a" * 40, "thresholds": th}
    assert {k: v for k, v in on.items() if k != "calculator_reader"} == rules
    other = make(cascade(facts={**FACTS, "revision": "b" * 40}))[1].stack_facts()["serving"]
    assert other["calculator_reader"] != on["calculator_reader"]
    off = make(cascade(thresholds=None))[1].stack_facts()["serving"]
    assert off["calculator_reader"]["thresholds"] == "off"
    assert "calculator_reader" not in make(cascade(), calculator=None)[1].stack_facts()["serving"]


# ---- C6 -------------------------------------------------------------------------------------------------------------


def test_c6_the_flags_and_their_defaults():
    from decisio.serve.vllm_engine import apply_repository_defaults

    args = build_parser().parse_args([])
    # no reader flag is None, so a repository's reader can be told from a flag; with no repository it is the rule reader
    assert (args.calculator_reader, args.calculator_tagger, args.calculator_thresholds, args.calculator_device) == (
        None,
        None,
        None,
        "auto",
    )
    apply_repository_defaults(args, None)
    assert (args.calculator_reader, args.calculator_reader_source, args.calculator_version) == ("rules", None, None)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--calculator-reader", "model"])


def test_c6_start_checks(tmp_path):
    assert calcs.check_flags("rules", None, None, "auto") is None
    assert "needs --calculator-tagger" in calcs.check_flags("cascade", None, None, "auto")
    why = calcs.check_flags("rules", str(tmp_path), None, "cpu")
    assert "--calculator-tagger, --calculator-device: for --calculator-reader cascade" in why
    assert calcs.check_flags("cascade", str(tmp_path), None, "auto") is None
    assert "full 40-character commit" in calcs.check_flags("cascade", "owner/tagger@main", None, "auto")
    assert "not a directory" in calcs.check_flags("cascade", str(tmp_path / "missing"), None, "auto")
    assert calcs.parse_tagger("owner/tagger@" + "c" * 40) == ("owner/tagger", "c" * 40)
    assert calcs.parse_tagger(str(tmp_path)) == (str(tmp_path), None)
    assert calcs.not_used_note("cascade", None).startswith("CALCULATOR READER cascade: not used")
    assert calcs.not_used_note("cascade", "calc-v1") is None and calcs.not_used_note("rules", None) is None


def test_c6_the_thresholds_file(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"role": 0.915, "link": 0.54, "flag": 0.82, "band": 0.02, "rows_with_a_case": 238}))
    t = calcs.load_thresholds(str(good))
    assert (t.role, t.link, t.flag, t.band) == (0.915, 0.54, 0.82, 0.02)
    assert calcs.load_thresholds(None) is None
    for name, content, message in (
        ("missing.json", {"role": 0.9, "flag": 0.9}, "missing link"),
        ("range.json", {"role": 0.3, "link": 0.9, "flag": 0.9}, "at least 0.5"),
        ("list.json", [0.9], "missing role, link, flag"),
    ):
        (tmp_path / name).write_text(json.dumps(content))
        with pytest.raises(ValueError, match=message):
            calcs.load_thresholds(str(tmp_path / name))
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(ValueError, match="not a readable JSON file"):
        calcs.load_thresholds(str(tmp_path / "bad.json"))
    assert "missing link" in calcs.check_flags("cascade", str(tmp_path), str(tmp_path / "missing.json"), "auto")


def test_c6_the_device(monkeypatch):
    monkeypatch.setattr(calcs, "_available", lambda d: d == "cpu")
    assert calcs.tagger_device("auto", "vllm") == "cpu"
    assert calcs.tagger_device("cpu", "vllm") == "cpu"
    with pytest.raises(ValueError, match="no cuda device"):
        calcs.tagger_device("cuda", "vllm")
    monkeypatch.setattr(calcs, "_available", lambda d: True)
    assert calcs.tagger_device("auto", "vllm") == "cuda"
    assert calcs.tagger_device("auto", "mlx") == "cpu" and calcs.tagger_device("mps", "mlx") == "mps"


def test_c6_the_reader_loads_once_with_its_identity(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    calls = []

    def loader(path, revision, device):
        calls.append((path, revision, device))
        return StubTagger({})

    r = calcs.load_reader("cascade", str(tmp_path), None, "cpu", "vllm", loader=loader)
    assert calls == [(str(tmp_path), None, "cpu")]
    assert r.cascade and r.tagger_facts["revision"].startswith("sha256:") and r.threshold_facts() == "off"
    rev = r.tagger_facts["revision"]
    assert r.start_line() == f"CALCULATOR READER cascade: tagger {tmp_path}@{rev} on cpu, thresholds off"
    (tmp_path / "model.safetensors").write_bytes(b"other weights")
    assert calcs.local_revision(tmp_path) != rev  # other weights, another identity
    (tmp_path / "SHA256SUMS").write_text("abc  model.safetensors\n")  # with sums: their hash (checked at load)
    assert calcs.local_revision(tmp_path) == "sha256:" + hashlib.sha256(b"abc  model.safetensors\n").hexdigest()
    hub = calcs.load_reader("cascade", "owner/tagger@" + "d" * 40, None, "cpu", "vllm", loader=loader)
    assert hub.tagger_facts == {"name": "owner/tagger", "revision": "d" * 40, "device": "cpu"}
    assert calcs.load_reader("rules", None, None, "auto", "vllm").kwargs() == {}
    with pytest.raises(ValueError, match="needs a tagger"):
        calcs.Reader("cascade")


@pytest.mark.slow  # imports torch and transformers (about 10 s); the model itself is tiny
def test_c6_a_tiny_saved_tagger_through_the_server(tmp_path):
    from test_calculator_cascade import _tiny_tagger

    _tiny_tagger(tmp_path)
    r = calcs.load_reader("cascade", str(tmp_path), None, "cpu", "vllm")
    sums = (tmp_path / "SHA256SUMS").read_bytes()
    assert r.tagger_facts == {
        "name": str(tmp_path),
        "revision": "sha256:" + hashlib.sha256(sums).hexdigest(),
        "device": "cpu",
    }
    client, so, _ = make(r)
    first, second = ask(client, state=STATE), ask(client, state=STATE)
    assert first.status_code == 200 and "calculator" in stages(first)
    assert (
        first.headers["x-decisio-calculator"] == second.headers["x-decisio-calculator"]
    )  # an untrained model, read the same
    assert sum(client.get("/health").json()["systemone"]["calculator"]["paths"].values()) == 2


def test_the_task_fingerprint_names_the_cascade_reader(tmp_path):
    import types

    from decisio.serve import calculator_serving as calcs
    from decisio.serve.vllm_engine import task_store_fingerprint

    tagger = tmp_path / "tagger"
    tagger.mkdir()
    (tagger / "SHA256SUMS").write_text("0" * 64 + "  config.json\n")
    thr = tmp_path / "thr.json"
    thr.write_text('{"role": 0.8, "link": 0.6, "flag": 0.8, "band": 0.02}')
    assert calcs.reader_identity("rules", str(tagger), str(thr)) is None
    ident = calcs.reader_identity("cascade", str(tagger), str(thr))
    assert ident.startswith("cascade@sha256:") and ";thresholds=sha256:" in ident
    assert calcs.reader_identity("cascade", "owner/name@" + "a" * 40, None) == "cascade@" + "a" * 40 + ";thresholds=off"

    def fp(reader_id):
        args = types.SimpleNamespace(
            served_name="s", model="m", model_identity=None, pad_to=None, pad_where="front", hide_index_keys=True,
            desnake_labels=True, pad_policy="always", family=types.SimpleNamespace(key="gemma-4-12b"),
            calculator_version="calc-v1", calculator_reader_id=reader_id,
        )  # fmt: skip
        from decisio.readout.letters import DEFAULT_FORMAT

        return task_store_fingerprint(args, DEFAULT_FORMAT)

    assert fp(None) != fp(ident) and '"calculator": "calc-v1"' in fp(None)


def test_the_tagger_reads_one_request_at_a_time():
    import threading
    import time

    from decisio.serve import calculator_serving as calcs

    inside, peak = [0], [0]

    class Slow:
        def predict(self, state, context=None):
            inside[0] += 1
            peak[0] = max(peak[0], inside[0])
            time.sleep(0.01)
            inside[0] -= 1

    reader = calcs.Reader("cascade", tagger=Slow())
    threads = [threading.Thread(target=reader.tagger.predict, args=("s",)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1


def test_a_tagger_precision_other_than_float32_is_named(tmp_path):
    from decisio.serve import calculator_serving as calcs

    seen = {}

    def loader(name, revision, device, dtype="float32"):
        seen["dtype"] = dtype
        return calcs._LockedTagger(type("T", (), {"predict": lambda self, s, c=None: None})())

    plain = calcs.load_reader("cascade", "owner/tagger@" + "e" * 40, None, "cpu", "vllm", loader=loader)
    assert seen["dtype"] == "float32" and "dtype" not in plain.fingerprint()
    half = calcs.load_reader("cascade", "owner/tagger@" + "e" * 40, None, "cpu", "vllm", loader=loader, dtype="float16")
    assert seen["dtype"] == "float16" and half.fingerprint()["dtype"] == "float16"
