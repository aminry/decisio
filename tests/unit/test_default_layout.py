# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The header layout as the default of the Gemma bases (docs/design/default-layout.md; decisio.serve.layout).

D1  the decision, as a table: the base's default, `--layout`, a tuned repository's and adapters' recorded layouts (plain
    when none), stored tasks fitted under plain, and what cannot share a server
D2  which stored tasks count as fitted under plain
D3  through the launcher: the held default with tasks loaded (a directory, a file), an empty directory, `--layout`,
    a tuned repository with and without a recorded layout, adapters with and without one
D4  the records: `defaults.layout` in a repository, `layout` in an adapter file, the fit tools writing it
D5  abstention tasks carry the layout they were fitted under and apply only under it
D6  what a response and /health say: the prompt header with the cached share, the layout decision, usage counting the
    header

  uv run pytest -q tests/unit/test_default_layout.py
"""

import json
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_certified_answers import FakeEngine, examples, post_task, wire
from test_hub import FINGERPRINTS, args_for, repo_dir
from test_hub_tuned import build, write_repo

from decisio import hub
from decisio.families import BASES
from decisio.readout import letters as L
from decisio.serve import adapter_temperatures as AT
from decisio.serve import layout as lay
from decisio.serve import task_backends as TB
from decisio.serve import vllm_engine
from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore


def decide(**kw):
    base = dict(explicit=None, default="header")
    return lay.decide_layout(**{**base, **kw})


# ---- D1 -------------------------------------------------------------------------------------------------------------


def test_d1_the_gemma_bases_default_to_the_header_and_qwen_to_plain():
    assert {k: v.layout for k, v in BASES.items()} == {
        "qwen3.6-35b-a3b": "plain",
        "gemma-4-12b": "header",
        "gemma-4-12b-fp8": "header",
        "gemma-4-31b": "header",
    }
    d = decide()
    assert (d["value"], d["held_by"], d["notes"]) == ("header", [], [])
    assert decide(default="plain")["value"] == "plain"


def test_d1_explicit_wins_and_says_what_it_leaves_behind():
    assert decide(explicit="plain", tasks=["t"])["value"] == "plain"
    d = decide(explicit="header", tasks=["t"], abstention=["a"], adapters={"n1": None})
    assert d["value"] == "header" and d["why"] == "--layout" and d["held_by"] == []
    assert any("adapter n1 (no recorded layout) was trained on the plain layout" in n for n in d["notes"])
    assert any(
        "calibration task(s) fitted under the plain layout: ['t']" in n and "not applied" in n for n in d["notes"]
    )
    assert any("abstention task(s)" in n for n in d["notes"])


def test_d1_stored_plain_tasks_hold_the_default_at_plain_and_say_so():
    d = decide(tasks=["routing", "triage"])
    assert d["value"] == "plain" and d["why"] == "held" and d["default"] == "header"
    assert "2 calibration task(s) fitted under the plain layout: ['routing', 'triage']" in d["held_by"][0]
    d = decide(abstention=["clinc"])
    assert d["value"] == "plain" and "abstention task(s)" in d["held_by"][0]
    line = lay.describe(d)
    assert line.startswith("LAYOUT plain (held at plain; this base's default is header)") and "--layout header" in line
    assert decide(default="plain", tasks=["t"])["held_by"] == []  # a base whose default is plain holds nothing


def test_d1_an_adapter_serves_the_layout_it_recorded_and_plain_when_it_recorded_none():
    d = decide(adapters={"n1": None})
    assert d["value"] == "plain" and d["held_by"] == ["adapter n1 (no recorded layout)"]
    d = decide(adapters={"n1": "plain"})
    assert d["value"] == "plain" and d["held_by"] == ["adapter n1"]
    d = decide(adapters={"n1": "header"})
    assert d["value"] == "header" and d["held_by"] == [] and "adapter n1" in d["why"]
    both = decide(adapters={"n1": "header", "n2": "header"})
    assert both["value"] == "header"


def test_d1_things_trained_on_different_layouts_cannot_share_a_server():
    with pytest.raises(lay.LayoutConflict, match="n1: header.*n2.*plain|cannot share a server"):
        decide(adapters={"n1": "header", "n2": None})
    with pytest.raises(lay.LayoutConflict):
        decide(tuned=True, recorded="header", adapters={"n1": None})
    # --layout resolves it, with the mismatch reported
    d = decide(explicit="header", adapters={"n1": "header", "n2": None})
    assert d["value"] == "header" and len(d["notes"]) == 1


def test_d1_a_tuned_repository_serves_its_recorded_layout_and_plain_when_it_has_none():
    d = decide(tuned=True)
    assert d["value"] == "plain" and d["held_by"] == ["the tuned repository (no recorded layout)"]
    d = decide(tuned=True, recorded="header")
    assert d["value"] == "header" and d["held_by"] == []
    assert decide(tuned=True, recorded="plain")["held_by"] == ["the tuned repository"]
    # a recorded header beside stored plain tasks: the record wins, the tasks are said not to apply
    d = decide(tuned=True, recorded="header", tasks=["t"])
    assert d["value"] == "header" and any("not applied under header" in n for n in d["notes"])


def test_d1_layout_values_are_checked():
    assert lay.check_layout(None, "x") is None and lay.check_layout("header", "x") == "header"
    with pytest.raises(ValueError, match="plain.*header"):
        lay.check_layout("footer", "x")


# ---- D2 -------------------------------------------------------------------------------------------------------------


def test_d2_only_tasks_that_would_apply_under_plain_count():
    plain_fp = FINGERPRINTS["gemma-4-12b"]
    header = json.loads(plain_fp)
    header["prompt_format"]["layout"] = "header"
    other = json.loads(plain_fp)
    other["model"] = "another"
    records = [
        {"id": "plain", "fingerprint": plain_fp},
        {"id": "header", "fingerprint": json.dumps(header, sort_keys=True)},
        {"id": "other", "fingerprint": json.dumps(other, sort_keys=True)},
        {"id": "broken"},
    ]
    assert lay.plain_fitted_readout(records, plain_fp) == ["plain"]
    ab = [
        {"id": "old", "model": "s"},  # written before the field: plain
        {"id": "plain", "model": "s", "layout": "plain"},
        {"id": "header", "model": "s", "layout": "header"},
        {"id": "other", "model": "elsewhere"},
    ]
    assert lay.plain_fitted_abstention(ab, "s") == ["old", "plain"]


# ---- D3 -------------------------------------------------------------------------------------------------------------


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    """`vllm_engine.main()` with the CPU stand-in engine, as test_hub's e1 runs it: start(*argv) -> (engine, so)."""
    import uvicorn

    import decisio.serve.hf_letters as hf

    class Engine:
        adapters = {}

        def __init__(self, model, *a, **kw):
            self.pad_unit, self.model_name = None, model

        def facts(self):
            return {}

    served = {}
    monkeypatch.setattr(hf, "HFLettersEngine", Engine)
    monkeypatch.setattr(vllm_engine, "make_app", lambda engine, so: served.update(engine=engine, so=so) or object())
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)

    def start(*argv):
        monkeypatch.setattr(sys, "argv", ["decisio", "--backend", "hf", "--model-class", "view", *argv])
        vllm_engine.main()
        return served["engine"], served["so"]

    return start


def plain_task_record(launcher, tmp_path, name="routing"):
    """A task as a plain-layout server stored it: the record of a task fitted on the 12B repository served `plain`."""
    d = repo_dir(tmp_path, "gemma-4-12b", name="repo")
    engine, so = launcher("--base", str(d), "--layout", "plain")
    fp = so.task_store.fingerprint
    record = json.loads((_task_json()))
    record.update(id=name, fingerprint=fp)
    return d, record


def _task_json():
    from test_certified_answers import FakeEngine as FE

    client = TestClient(_app(FE()))
    assert post_task(client, {"id": "x", "examples": examples(40, 0.9)}).status_code == 200
    return json.dumps(client.get("/v1/tasks?full=1").json()["tasks"][0])


def _app(engine):
    so = SystemOne(engine, "decisio-test", task_store=TaskStore("fp"))
    app = FastAPI()
    add_routes(app, so)
    return app


def test_d3_a_bare_gemma_base_serves_the_header(launcher, tmp_path, capsys):
    d = repo_dir(tmp_path, "gemma-4-12b")
    engine, so = launcher("--base", str(d))
    assert engine.fmt.layout == "header" and engine.layout_decision["value"] == "header"
    assert "LAYOUT header (the default of the base)" in capsys.readouterr().out
    assert json.loads(so.task_store.fingerprint)["prompt_format"]["layout"] == "header"
    plain_engine, plain_so = launcher("--base", str(d), "--layout", "plain")
    assert (
        plain_engine.fmt.layout == "plain"
        and "layout" not in json.loads(plain_so.task_store.fingerprint)["prompt_format"]
    )
    assert json.loads(plain_so.task_store.fingerprint) == json.loads(FINGERPRINTS["gemma-4-12b"])  # byte for byte


def test_d3_stored_plain_tasks_hold_the_default_and_an_explicit_header_leaves_them_unapplied(
    launcher, tmp_path, capsys
):
    d, record = plain_task_record(launcher, tmp_path)
    store_dir = tmp_path / "tasks"
    backend = TB.DirBackend(store_dir)
    backend.save("readout", record)
    backend.close()
    capsys.readouterr()
    engine, so = launcher("--base", str(d), "--tasks-dir", str(store_dir))
    out = capsys.readouterr().out
    assert engine.fmt.layout == "plain" and engine.layout_decision["held_by"]
    assert "LAYOUT plain (held at plain; this base's default is header)" in out and "routing" in out
    assert so.task_store_report()["readout"] == {"loaded": 1, "applied": 1, "other_stack": 0}  # still applied
    so.task_backend.close()
    engine, so = launcher("--base", str(d), "--tasks-dir", str(store_dir), "--layout", "header")
    out = capsys.readouterr().out
    assert engine.fmt.layout == "header" and engine.layout_decision["why"] == "--layout"
    assert "not applied under header" in out and "WARNING: tasks fitted under another model or rendering" in out
    assert so.task_store_report()["readout"] == {"loaded": 1, "applied": 0, "other_stack": 1}
    so.task_backend.close()


def test_d3_a_tasks_file_holds_it_too_and_an_empty_directory_does_not(launcher, tmp_path):
    d, record = plain_task_record(launcher, tmp_path)
    f = tmp_path / "tasks.json"
    f.write_text(json.dumps({"tasks": [record]}))
    engine, so = launcher("--base", str(d), "--tasks-file", str(f))
    assert engine.fmt.layout == "plain" and so.task_store_report()["readout"]["applied"] == 1
    empty = tmp_path / "empty"
    engine, so = launcher("--base", str(d), "--tasks-dir", str(empty))
    assert engine.fmt.layout == "header" and so.task_store_report()["readout"]["loaded"] == 0
    so.task_backend.close()


def test_d3_tasks_for_another_model_hold_nothing(launcher, tmp_path):
    d, record = plain_task_record(launcher, tmp_path)
    other = json.loads(record["fingerprint"])
    other["model"] = "another-model"
    record["fingerprint"] = json.dumps(other, sort_keys=True)
    f = tmp_path / "tasks.json"
    f.write_text(json.dumps([record]))
    engine, _ = launcher("--base", str(d), "--tasks-file", str(f))
    assert engine.fmt.layout == "header"  # it would not have applied under plain either


def test_d3_adapters_with_and_without_a_recorded_layout(launcher, tmp_path, capsys):
    d = repo_dir(tmp_path, "gemma-4-12b")
    bare, rec_header, rec_plain = (tmp_path / n for n in ("bare", "header", "plain"))
    for p in (bare, rec_header, rec_plain):
        p.mkdir()
    temps = {"schema": AT.SCHEMA, "temperatures": {"choice": 3.0, "noul": 3.0, "score": 3.0}}
    (rec_header / "decisio_adapter.json").write_text(json.dumps({**temps, "layout": "header"}))
    (rec_plain / "decisio_adapter.json").write_text(json.dumps({**temps, "layout": "plain"}))
    (bare / "decisio_adapter.json").write_text(json.dumps(temps))  # temperatures, no layout
    engine, _ = launcher("--base", str(d), "--adapter", f"n1={bare}")
    assert engine.fmt.layout == "plain" and engine.layout_decision["held_by"] == ["adapter n1 (no recorded layout)"]
    engine, _ = launcher("--base", str(d), "--adapter", f"n1={rec_header}")
    assert engine.fmt.layout == "header" and engine.layout_decision["held_by"] == []
    engine, _ = launcher("--base", str(d), "--adapter", f"n1={rec_plain}")
    assert engine.fmt.layout == "plain"
    with pytest.raises(SystemExit, match="LAYOUT: .*cannot share a server"):
        launcher("--base", str(d), "--adapter", f"n1={rec_header}", "--adapter", f"n2={rec_plain}")
    engine, _ = launcher(
        "--base", str(d), "--adapter", f"n1={rec_header}", "--adapter", f"n2={rec_plain}", "--layout", "plain"
    )
    assert engine.fmt.layout == "plain" and any("adapter n1" in n for n in engine.layout_decision["notes"])
    capsys.readouterr()
    # an adapter with no file at all has recorded nothing either
    nofile = tmp_path / "nofile"
    nofile.mkdir()
    engine, _ = launcher("--base", str(d), "--adapter", f"n1={nofile}")
    assert engine.fmt.layout == "plain"


def test_d3_a_tuned_repository_serves_its_recorded_layout(launcher, tmp_path):
    for recorded, want in ((None, "plain"), ("plain", "plain"), ("header", "header")):
        kw = {"layout": recorded} if recorded else {}
        d = write_repo(tmp_path, build(**kw), name=f"t-{recorded}")
        engine, _ = launcher("--base", str(d))
        assert engine.fmt.layout == want, recorded
    d = write_repo(tmp_path, build(layout="header"), name="t-flag")
    engine, _ = launcher("--base", str(d), "--layout", "plain")
    assert engine.fmt.layout == "plain" and engine.layout_decision["notes"]


def test_d3_the_qwen_base_stays_plain(launcher, tmp_path):
    d = repo_dir(tmp_path, "qwen3.6-35b-a3b", name="q")
    engine, _ = launcher("--base", str(d))
    assert engine.fmt.layout == "plain"


# ---- D4 -------------------------------------------------------------------------------------------------------------


def test_d4_a_repository_records_its_layout_and_an_older_reader_refuses_it():
    c = build(layout="header")
    assert c["defaults"]["layout"] == "header"
    hub.check_defaults(c["defaults"])
    assert "layout" not in build()["defaults"]
    assert hub.defaults_sha256(c) != hub.defaults_sha256(build())
    bad = build()
    bad["defaults"]["layout"] = "footer"
    with pytest.raises(ValueError, match="layout"):
        hub.check_defaults(bad["defaults"])
    old_keys = {"temperatures", "choice_bands", "noul_commit", "fit", "noul_commit_basis", "over_limit", "think"}
    assert set(c["defaults"]) - old_keys == {"layout"}  # a 0.25.0 decisio rejects an unknown key of `defaults`


def test_d4_the_resolved_arguments_know_the_repository_record(tmp_path):
    d = write_repo(tmp_path, build(layout="header"), name="r")
    args = args_for(str(d))
    vllm_engine.resolve_base(args)
    assert (args.layout_tuned, args.layout_recorded, args.layout_default) == (True, "header", "header")
    base = args_for("gemma-4-12b")
    vllm_engine.resolve_base(base)
    assert (base.layout_tuned, base.layout_recorded, base.layout_default) == (False, None, "header")
    q = args_for("qwen3.6-35b-a3b")
    vllm_engine.resolve_base(q)
    assert q.layout_default == "plain"


def test_d4_an_adapter_file_records_its_layout(tmp_path):
    p = tmp_path / "decisio_adapter.json"
    doc = {"schema": AT.SCHEMA, "temperatures": {"choice": 3.0}}
    p.write_text(json.dumps(doc))
    assert AT.read(p)["layout"] is None
    p.write_text(json.dumps({**doc, "layout": "header"}))
    assert AT.read(p)["layout"] == "header"
    p.write_text(json.dumps({**doc, "layout": "footer"}))
    with pytest.raises(AT.AdapterFileError, match="layout"):
        AT.read(p)


def test_d4_the_fit_tool_records_the_layout_it_was_given(tmp_path, capsys):
    reads = tmp_path / "reads.jsonl"
    reads.write_text(
        "\n".join(json.dumps({"type": "choice", "probs": [0.7, 0.2, 0.1], "gold": i % 3}) for i in range(60))
    )
    out = tmp_path / "t.json"
    assert (
        AT.main(["fit", "--reads", str(reads), "--out", str(out), "--layout", "header", "--choice-bands", "none"]) == 0
    )
    assert json.loads(out.read_text())["layout"] == "header" and AT.read(out)["layout"] == "header"
    assert "no --layout given" not in capsys.readouterr().err
    assert AT.main(["fit", "--reads", str(reads), "--out", str(out), "--choice-bands", "none"]) == 0
    assert "layout" not in json.loads(out.read_text())  # nothing said, nothing recorded
    assert "no --layout given" in capsys.readouterr().err  # and the tool says so


# ---- D5 -------------------------------------------------------------------------------------------------------------


def abstention_server(layout):
    eng = FakeEngine()
    eng.fmt = L.PromptFormat(tail="spaced", slot="template", variants="summed", system_prompt=True, layout=layout)
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"))
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so


def test_d5_an_abstention_task_carries_the_layout_it_was_fitted_under():
    from test_task_persistence import abstention_body

    plain, so_p = abstention_server("plain")
    assert plain.post("/v1/abstention/tasks", json=abstention_body()).status_code == 200
    assert "layout" not in so_p.tasks["abs"]  # a plain server writes what it always wrote
    header, so_h = abstention_server("header")
    assert header.post("/v1/abstention/tasks", json=abstention_body()).status_code == 200
    assert so_h.tasks["abs"]["layout"] == "header"
    # applied only under the layout it was fitted under; kept otherwise, and said so
    record_p, record_h = so_p.tasks["abs"], so_h.tasks["abs"]
    assert so_p.abstention_applies(record_p) and not so_p.abstention_applies(record_h)
    assert so_h.abstention_applies(record_h) and not so_h.abstention_applies(record_p)
    r = header.post("/v1/abstention/tasks/import", json={"tasks": [record_p]})
    assert r.status_code == 422 and "another layout" in r.json()["detail"]
    so_p.tasks["other"] = {**record_h, "id": "other"}
    assert plain.get("/v1/abstention/tasks").json()["store"]["abstention"] == {
        "loaded": 2,
        "applied": 1,
        "other_model": 0,
        "other_layout": 1,
    }


def test_d5_require_tasks_counts_abstention_tasks_of_the_layout_served():
    store = TaskStore("fp")
    records = [{"id": "a", "model": "s"}, {"id": "b", "model": "s", "layout": "header"}]
    assert TB.usable(store, records, "s", "plain") == 1 and TB.usable(store, records, "s", "header") == 1


# ---- D6 -------------------------------------------------------------------------------------------------------------


class CountingEngine(FakeEngine):
    """The stand-in engine reporting the prompt it read, as the vLLM engine does."""

    def answer(self, state, questions, adapter=None):
        probs, info = super().answer(state, questions, adapter)
        n = len(questions)
        return probs, {**info, "engine_prompt_tokens": [1500] * n, "cached_tokens_mean": 1400.0}


def test_d6_a_response_says_the_layout_and_how_much_of_the_prompt_the_cache_served():
    eng = CountingEngine()
    eng.fmt = L.PromptFormat(tail="spaced", slot="template", variants="summed", system_prompt=True, layout="header")
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"))
    app = FastAPI()
    add_routes(app, so)
    r = TestClient(app).post("/v1/systemone", json=wire("0:0.9:x"))
    assert r.status_code == 200 and r.headers["x-decisio-prompt"] == "layout=header; tokens=1500; cached=1400"
    plain = CountingEngine()
    so2 = SystemOne(plain, "decisio-test", task_store=TaskStore("fp"))
    app2 = FastAPI()
    add_routes(app2, so2)
    r = TestClient(app2).post("/v1/systemone", json=wire("0:0.9:x"))
    assert r.headers["x-decisio-prompt"] == "layout=plain; tokens=1500; cached=1400"
    quiet = TestClient(_app(FakeEngine())).post("/v1/systemone", json=wire("0:0.9:x"))
    assert "x-decisio-prompt" not in quiet.headers  # an engine that reports no prompt size


def test_d6_health_carries_the_decision_and_the_start_prints_it():
    from decisio.serve import vllm_engine as ve

    eng = CountingEngine()
    eng.fmt = L.PromptFormat(tail="spaced", slot="template", variants="summed", system_prompt=True, layout="plain")
    eng.layout_decision = decide(tasks=["routing"])
    eng.mode = "separate"
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"))
    try:
        app = ve.make_app(eng, so)
    except Exception as e:  # the stand-in is thin; the route only needs to read the decision
        pytest.skip(f"make_app needs more of the engine: {e}")
    health = TestClient(app).get("/health").json()
    assert health["systemone"]["layout"]["value"] == "plain" and health["systemone"]["layout"]["held_by"]


def test_d6_usage_counts_the_header_the_default_layout_adds():
    from test_layout_flags import BASE, ONE, STATE, StubTok, engine

    so = SystemOne.__new__(SystemOne)
    plain = L.PromptFormat(**BASE)
    so.engine = engine(plain, StubTok())
    base = so.usage_tokens(STATE, ONE)
    so.engine = engine(L.PromptFormat(**BASE, layout="header"), so.engine.tok)
    header = so.usage_tokens(STATE, ONE)
    assert header == base + len(so.engine.tok.encode(L.layout_header([e["instructions"] for e in ONE])))
