# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Think-mode temperatures: an answer that thought is served at its own temperature per question type.

Q1  the answer after thinking is the mix at the think temperature; a sure question and a type without one keep the
    single-pass temperature, and which questions think is decided at the single-pass temperature
Q2  an adapter's think temperature, then its own single-pass temperature, then the server's think temperature, then the
    single-pass temperature
Q3  the fingerprint holds them only when the server thinks and has any; a certificate made under others is
    `think_mode_changed`
Q4  the adapter file: validated, read, shown to the server, and `fit-think` recovers a known temperature from labelled
    mixes and leaves the single-pass fit as it was
Q5  a tuned repository: `defaults.think_temperatures` validated, written by `tuned_config`, reaches the arguments,
    a flag beats it, the fit's data stays out, an older reader refuses the key; the flags are in the parser

  uv run pytest -q tests/unit/test_think_temperatures.py
"""

import json

import numpy as np
import pytest
from test_hub import args_for
from test_hub_tuned import build, write_repo
from test_think import ask, expected, first_read, make, probs_of

from decisio import hub
from decisio.serve import adapter_temperatures as AT
from decisio.serve import stack, vllm_engine
from decisio.serve import think as T
from decisio.serve.vllm_engine import resolve_base

LP2 = np.log(np.array([0.05, 0.9, 0.05]))  # the stand-in engine's second read for "…:x:1"


# ---- Q1 -------------------------------------------------------------------------------------------------------------


def test_q1_the_answer_after_thinking_is_the_mix_at_the_think_temperature():
    client, so, eng = make(think=True, temperatures={"choice": 2.0}, think_temperatures={"choice": 5.0})
    r = ask(client, "0:0.5:x:1")  # unsure at the single-pass 2.0
    assert r.status_code == 200, r.text
    assert np.allclose(probs_of(r.json()["answers"]["q"], 3), expected(first_read(0, 0.5, 3), LP2, 5.0), atol=1e-9)
    sure = ask(client, "0:0.99:x:1")  # a sure question is read as it always was, at the single-pass temperature
    assert np.allclose(
        probs_of(sure.json()["answers"]["q"], 3), expected(first_read(0, 0.99, 3), np.log(first_read(0, 0.99, 3)), 2.0)
    )


def test_q1_a_type_without_a_think_temperature_keeps_the_single_pass_one():
    client, so, eng = make(think=True, temperatures={"choice": 2.0}, think_temperatures={"score": 7.0})
    r = ask(client, "0:0.5:x:1")
    assert np.allclose(probs_of(r.json()["answers"]["q"], 3), expected(first_read(0, 0.5, 3), LP2, 2.0), atol=1e-9)
    s = ask(client, "0:0.5:x:1", qtype="score")  # score has one: 7.0 on the mix; the unsure test read it at the default
    assert s.status_code == 200 and len(eng.calls) == 2
    got = np.array([s.json()["answers"]["q"]["probabilities"][str(i)] for i in range(3)])
    assert np.allclose(got, expected(first_read(0, 0.5, 3), LP2, 7.0), atol=1e-9)


def test_q1_which_questions_think_is_decided_at_the_single_pass_temperature():
    # plain 0.9 is sure at T=1, and a think temperature of 4 would make it unsure; it must not think
    client, so, eng = make(think=True, think_temperatures={"choice": 4.0})
    r = ask(client, "0:0.9:x:1")
    assert eng.calls == [] and r.json()["thinking"]["per_question"]["q"] == {"thought": False, "why": T.SURE}
    # and a question unsure at the single-pass temperature thinks whatever the think temperature is
    client, so, eng = make(think=True, temperatures={"choice": 3.0}, think_temperatures={"choice": 1.0})
    ask(client, "0:0.9:x:1")
    assert len(eng.calls) == 1


def test_q1_without_think_temperatures_the_answers_are_what_they_were():
    a = ask(*make(think=True, temperatures={"choice": 2.0})[:1], "0:0.5:x:1").json()["answers"]["q"]
    b = ask(*make(think=True, temperatures={"choice": 2.0}, think_temperatures={})[:1], "0:0.5:x:1").json()["answers"][
        "q"
    ]
    assert a == b


# ---- Q2 -------------------------------------------------------------------------------------------------------------


def test_q2_adapter_then_its_own_single_pass_then_the_servers_think_then_single_pass():
    so = make(
        temperatures={"choice": 2.0, "score": 2.5},
        think_temperatures={"choice": 5.0, "score": 6.0},
        adapter_temperatures={
            "a": {"choice": 3.0, "score": 3.5, "think": {"score": 8.0}},
            "b": {"noul": 4.0},
        },
    )[1]
    assert so.think_temperature_of("score", "a") == 8.0  # the adapter's think temperature
    assert so.think_temperature_of("choice", "a") == 3.0  # its own single-pass one; the base's think does not fit it
    assert so.think_temperature_of("choice", "b") == 5.0  # an adapter with none for the type: the server's think
    assert so.think_temperature_of("score", None) == 6.0 and so.think_temperature_of("score", "zzz") == 6.0
    assert so.think_temperature_of("noul", None) == so.temperature_of("noul", None)  # nothing set: the single-pass
    assert so.think_temperature_of("noul", "b") == 4.0


def test_q2_the_adapter_name_in_the_request_reaches_the_think_temperature():
    client, so, eng = make(
        think=True, adapter_temperatures={"n1": {"choice": 1.0, "think": {"choice": 6.0}}}, think_temperatures={}
    )
    eng.adapters = {"n1": "x"}
    r = ask(client, "0:0.5:x:1", adapter="n1")
    assert r.status_code == 200, r.text
    assert np.allclose(probs_of(r.json()["answers"]["q"], 3), expected(first_read(0, 0.5, 3), LP2, 6.0), atol=1e-9)


# ---- Q3 -------------------------------------------------------------------------------------------------------------


def test_q3_the_fingerprint_holds_think_temperatures_only_for_a_server_that_thinks_and_has_them():
    plain = make(think=True)[1].stack_facts()
    assert "temperatures" not in plain["serving"]["think"]
    withT = make(think=True, think_temperatures={"score": 4.75})[1].stack_facts()
    assert withT["serving"]["think"]["temperatures"] == {"score": 4.75}
    assert stack.differences(plain, withT) == ["serving.think.temperatures"]
    other = make(think=True, think_temperatures={"score": 5.0})[1].stack_facts()
    assert stack.differences(withT, other) == ["serving.think.temperatures.score"]
    adapter = make(think=True, adapter_temperatures={"n1": {"choice": 1.0, "think": {"score": 4.0}}})[1].stack_facts()
    assert adapter["serving"]["think"]["temperatures"]["adapters"] == {"n1": {"score": 4.0}}
    off = make(think=False, think_temperatures={"score": 4.75})[1].stack_facts()  # thinking off: they change nothing
    assert "think" not in off["serving"]


def test_q3_a_certificate_made_under_other_think_temperatures_is_think_mode_changed():
    from test_certified_answers import examples, post_task, wire  # noqa: F401  (the helpers K8 uses)
    from test_think import CRIT  # noqa: F401

    # the reason mapping: every difference under serving.think is the think reason, nothing else is
    a = make(think=True, think_temperatures={"score": 4.0})[1]
    b = make(think=True, think_temperatures={"score": 5.0})[1]
    diff = stack.differences(a.stack_facts(), b.stack_facts())
    assert diff and all(d == "serving.think" or d.startswith("serving.think.") for d in diff)


# ---- Q4 -------------------------------------------------------------------------------------------------------------

BASE_DOC = {
    "schema": AT.SCHEMA,
    "temperatures": {"choice": 1.6359, "noul": 3.7277, "score": 2.5936},
    "choice_bands": {"2-10": 1.5, "11-26": 2.0},
    "fit": {"objective": {"score": "log_loss"}, "kappa_on_raw_readout": {"score": 0.385}, "n": {"score": 400}},
}


def test_q4_think_temperatures_are_validated():
    assert AT.validate_think_temperatures(None, "f") == {}
    assert AT.validate_think_temperatures({"score": 4, "choice": 1.9}, "f") == {"score": 4.0, "choice": 1.9}
    for bad in (
        {},
        [],
        {"nope": 1.0},
        {"score": 0},
        {"score": -1.0},
        {"score": True},
        {"score": "3"},
        {"score": float("inf")},
    ):
        with pytest.raises(AT.AdapterFileError):
            AT.validate_think_temperatures(bad, "f")


def test_q4_a_file_with_them_is_read_and_shown_to_the_server(tmp_path):
    doc = {**BASE_DOC, "think_temperatures": {"score": 4.75}}
    p = tmp_path / "decisio_adapter.json"
    p.write_text(json.dumps(doc))
    got = AT.read(p)
    assert got["think_temperatures"] == {"score": 4.75} and got["temperatures"]["score"] == 2.5936
    assert AT.serving_temperatures(got)["think"] == {"score": 4.75}
    p.write_text(json.dumps(BASE_DOC))
    plain = AT.read(p)
    assert plain["think_temperatures"] == {} and "think" not in AT.serving_temperatures(plain)
    p.write_text(json.dumps({**BASE_DOC, "think_temperatures": {"score": -2}}))
    with pytest.raises(AT.AdapterFileError, match="think temperature"):
        AT.read(p)


def synthetic_rows(true_t, n=600, k=4, seed=0):
    """Labelled mixes whose gold is drawn from softmax(log p / true_t): a raw mix that is over-confident by true_t."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        p = np.exp(rng.normal(0, 2.0, k))
        p /= p.sum()
        z = np.log(p) / true_t
        q = np.exp(z - z.max())
        q /= q.sum()
        rows.append({"type": "score", "probs": p.tolist(), "gold": int(rng.choice(k, p=q))})
    return rows


def test_q4_fit_think_recovers_a_known_temperature_and_leaves_the_single_pass_fit_alone():
    out = AT.add_think(BASE_DOC, synthetic_rows(4.0))
    assert out["think_temperatures"].keys() == {"score"}
    assert out["think_temperatures"]["score"] == pytest.approx(4.0, rel=0.2)
    assert out["temperatures"] == BASE_DOC["temperatures"] and out["choice_bands"] == BASE_DOC["choice_bands"]
    assert out["fit"]["n"] == BASE_DOC["fit"]["n"]
    assert out["fit"]["think"]["n"] == {"score": 600} and out["fit"]["think"]["objective"] == {"score": "log_loss"}
    assert BASE_DOC["fit"].get("think") is None  # the input document is not changed
    with pytest.raises(ValueError, match="no think rows"):
        AT.add_think(BASE_DOC, [])


def test_q4_fit_think_takes_rows_served_at_a_temperature_back_to_the_raw_mix():
    raw = synthetic_rows(3.0, seed=1)
    served = [{**r, "probs": AT.rescale(r["probs"], 1 / 1.5).tolist(), "read_at": 1.5} for r in raw]
    a, b = AT.add_think(BASE_DOC, raw), AT.add_think(BASE_DOC, served)
    assert a["think_temperatures"]["score"] == pytest.approx(b["think_temperatures"]["score"], rel=0.02)


def test_q4_the_fit_think_command_writes_the_file(tmp_path):
    base = tmp_path / "decisio_adapter.json"
    base.write_text(json.dumps(BASE_DOC))
    reads = tmp_path / "think.jsonl"
    reads.write_text("\n".join(json.dumps(r) for r in synthetic_rows(4.0)))
    out = tmp_path / "out.json"
    assert AT.main(["fit-think", "--into", str(base), "--reads", str(reads), "--out", str(out)]) == 0
    got = AT.read(out)  # still a valid adapter file
    assert (
        got["think_temperatures"]["score"] == pytest.approx(4.0, rel=0.2)
        and got["choice_bands"] == BASE_DOC["choice_bands"]
    )
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(SystemExit):
        AT.main(["fit-think", "--into", str(base), "--reads", str(empty), "--out", str(tmp_path / "x.json")])


def test_q4_the_mix_a_fit_reads_is_the_one_the_server_serves():
    p1, lp2 = np.array([0.6, 0.3, 0.1]), np.log(np.array([0.2, 0.2, 0.6]))
    z = (np.log(p1) + lp2) / 2  # the geometric mean of the two reads, normalised: what a fit takes as `probs`
    want = np.exp(z - z.max())
    assert np.allclose(T.mix(p1, lp2), want / want.sum(), atol=1e-15)


# ---- Q5 -------------------------------------------------------------------------------------------------------------


def test_q5_tuned_config_writes_them_and_the_digest_follows():
    c = build(think={"factor": 0.79}, think_temperatures={"score": 4.75})
    assert c["defaults"]["think_temperatures"] == {"score": 4.75}
    assert hub.defaults_sha256(c) != hub.defaults_sha256(build(think={"factor": 0.79}))
    assert "think_temperatures" not in build()["defaults"]
    hub.check_defaults(c["defaults"])  # validated


def test_q5_a_bad_think_temperatures_default_is_refused():
    for bad in ({}, {"nope": 2.0}, {"score": 0}, {"score": "x"}):
        config = build()
        config["defaults"]["think_temperatures"] = bad
        with pytest.raises(ValueError):
            hub.check_defaults(config["defaults"])


def test_q5_the_think_fit_in_a_repository_keeps_the_method_and_leaves_the_data_out():
    fit = {
        "method": "m",
        "think": {
            "objective": {"score": "log_loss"},
            "kappa_on_raw_readout": {"score": 0.21},
            "n": {"score": 103},
            "detail": {"x": 1},
            "data": "d",
        },
    }
    c = build(fit=fit, think_temperatures={"score": 4.75})
    assert c["defaults"]["fit"]["think"] == {
        "objective": {"score": "log_loss"},
        "kappa_on_raw_readout": {"score": 0.21},
    }
    assert "103" not in json.dumps(c["defaults"]["fit"])


def test_q5_the_repository_default_reaches_the_arguments_and_a_flag_beats_it(tmp_path):
    d = write_repo(tmp_path, build(think={"factor": 0.79}, think_temperatures={"score": 4.75, "choice": 2.1}), name="t")
    args = args_for(str(d))
    resolve_base(args)
    assert (args.think_temperature_score, args.think_temperature_choice, args.think_temperature_noul) == (
        4.75,
        2.1,
        None,
    )
    assert args.repository_defaults["think_temperatures"] == {"score": 4.75, "choice": 2.1}
    args = args_for(str(d), think_temperature_score=6.0)
    resolve_base(args)
    assert args.think_temperature_score == 6.0 and args.repository_defaults["think_temperatures"] == {"choice": 2.1}
    plain = write_repo(tmp_path, build(), name="p")
    args = args_for(str(plain))
    resolve_base(args)
    assert args.think_temperature_score is None and "think_temperatures" not in args.repository_defaults
    base = args_for("gemma-4-31b")
    resolve_base(base)
    assert base.think_temperature_choice is None


def test_q5_the_flags_are_in_the_parser():
    a = vllm_engine.build_parser().parse_args(["--think-temperature-score", "4.75", "--think-temperature-noul", "3"])
    assert (a.think_temperature_score, a.think_temperature_noul, a.think_temperature_choice) == (4.75, 3.0, None)


def test_q5_an_older_reader_refuses_the_key():
    old_keys = {"temperatures", "choice_bands", "noul_commit", "fit", "noul_commit_basis", "over_limit", "think"}
    config = build(think={"factor": 0.79}, think_temperatures={"score": 4.75})
    assert set(config["defaults"]) - old_keys == {"think_temperatures"}  # a 0.23.0 decisio rejects unknown keys
