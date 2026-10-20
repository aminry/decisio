# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Think when unsure (`decisio.serve.think`): an unsure question is read again after the served model has written its
own reasoning, and the two reads are mixed in log space.

K1  the pieces: thresholds times the factor, the confidence of a yes/no, the mix, the settings checks, the stop tokens
K2  off by default: no engine call for thinking, no `thinking` field, the answers as they were
K3  on: only the unsure questions think, in one engine call; the answer is the mix at the type's temperature; the
    commit rule and a yes/no asked as letters keep their orientation
K4  control: the request field beats the server's setting either way, a bad value is a 422, an engine that cannot write
    refuses a request that asks, a lower factor thinks on fewer questions
K5  what stays at one read, and says why: a question of a registered task, more than 16 options, two-order requests
K6  the report: `thinking` in the body, usage with the reasoning tokens, the headers, `/health`
K7  deterministic: the same request answers the same whatever came before it
K8  certificates: the fingerprint holds how the server thinks only when it does; a certified task is
    `think_mode_changed` on a server that thinks or stops thinking, and certified again when the mode is back
K9  a tuned repository: `defaults.think` validated, turns thinking on with its factor, flags beat it, an older reader
    refuses the key; the flags are in the parser
K10 the second read's options are the ones the first read has, after the suffix (the real tokenizer)

  uv run pytest -q tests/unit/test_think.py
"""

import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_certified_answers import FakeEngine as CertEngine
from test_certified_answers import examples, post_task, wire
from test_hub import args_for
from test_hub_tuned import build, write_repo

from decisio import hub
from decisio.serve import think as T
from decisio.serve import vllm_engine
from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore
from decisio.serve.temperature import apply_temperature
from decisio.serve.vllm_engine import resolve_base

CRIT = {"option_0": "billing", "option_1": "shipping", "option_2": "access"}


def crit(k):
    return {f"option_{i}": f"o{i}" for i in range(k)}


class ThinkEngine(CertEngine):
    """The certified-answers stand-in plus `think`: the state's message is `top:conf:x:second`; the first read puts
    `conf` on option `top`, the second read puts 0.9 on option `second`."""

    adapters: dict = {}

    def __init__(self, can=True):
        super().__init__()
        self.calls, self.can = [], can

    def can_think(self):
        return self.can

    def think(self, state, questions, adapter=None, budget=512, subset=None):
        subset = list(range(len(questions))) if subset is None else subset
        questions = [questions[i] for i in subset]
        self.calls.append((state, [q.get("kind") for q in questions], adapter, budget, subset))
        second = int(str(state["message"]).split(":")[3])
        out = []
        for q in questions:
            k = len(q["options"]) if q.get("options") else 2
            p = np.full(k, 0.1 / (k - 1))
            p[second % k] = 0.9
            out.append(np.log(p))
        return out, [40 + i for i in range(len(questions))], {"generate_ms": 5.0, "read_ms": 1.0}


def make(think=False, factor=1.0, budget=512, engine=None, **kw):
    eng = engine or ThinkEngine()
    so = SystemOne(
        eng, "decisio-test", task_store=TaskStore("fp"), think=think, think_factor=factor, think_budget=budget, **kw
    )
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so, eng


def ask(client, message, k=3, qtype="choice", **extra):
    body = {"state": {"message": message}, "model": "m", "questions": {"q": question(k, qtype)}, **extra}
    return client.post("/v1/systemone", json=body)


def question(k=3, qtype="choice"):
    if qtype == "noul":
        return {"type": "noul", "instructions": "Is it a complaint?"}
    if qtype == "score":
        return {"type": "score", "instructions": "How urgent?", "criteria": [f"level {i}" for i in range(k)]}
    return {"type": "choice", "instructions": "Which queue?", "criteria": crit(k)}


def probs_of(answer, k):
    return np.array([answer["probabilities"][f"option_{i}"] for i in range(k)])


def expected(p1, lp2, temperature):
    mixed = T.mix(p1, lp2)
    return apply_temperature(mixed, temperature) if temperature != 1.0 else mixed


def first_read(top, conf, k):
    p = np.full(k, (1 - conf) / (k - 1))
    p[top] = conf
    return p


# ---- K1 -------------------------------------------------------------------------------------------------------------


def test_k1_thresholds_times_the_factor_and_the_confidence_of_a_yes_no():
    assert T.TAU == {"choice": 0.75, "noul": 0.92, "score": 0.70}
    p = np.array([0.7, 0.2, 0.1])
    assert T.is_unsure(p, "choice", 1.0) and not T.is_unsure(p, "choice", 0.9)  # 0.7 < 0.75, not < 0.675
    assert T.confidence(np.array([0.05, 0.95]), "noul") == 0.95  # max(P(yes), P(no))
    assert T.is_unsure(np.array([0.9, 0.1]), "noul", 1.0) and not T.is_unsure(np.array([0.9, 0.1]), "noul", 0.79)
    assert T.is_unsure(np.array([0.5, 0.5]), "score", 1.0) and not T.is_unsure(np.array([0.5, 0.5]), "score", 0.7)


def test_k1_the_mix_is_the_mean_of_the_log_probabilities_normalised():
    p1, p2 = np.array([0.6, 0.3, 0.1]), np.array([0.2, 0.2, 0.6])
    m = T.mix(p1, np.log(p2))
    want = np.sqrt(p1 * p2)
    assert np.allclose(m, want / want.sum()) and abs(m.sum() - 1) < 1e-12
    assert np.allclose(T.mix(p1, np.log(p1)), p1)  # the same read twice changes nothing


def test_k1_settings_are_checked():
    assert T.check_settings({"factor": 0.79}, "d") == {"factor": 0.79}
    assert T.check_settings({"factor": 1, "budget": 256}, "d") == {"factor": 1.0, "budget": 256}
    for bad in ({}, {"factor": 0}, {"factor": 2}, {"factor": True}, {"factor": "1"}, {"factor": 1, "budget": 8}):
        with pytest.raises(ValueError):
            T.check_settings(bad, "d")
    with pytest.raises(ValueError, match="unknown keys"):
        T.check_settings({"factor": 1, "tau": 0.5}, "d")
    with pytest.raises(ValueError):
        T.check_budget(True, "b")


def test_k1_the_stop_tokens_are_the_end_of_sequence_and_of_the_turn():
    class Tok:
        eos_token_id = 1

        def get_vocab(self):
            return {"<turn|>": 105, "x": 3}

        def convert_tokens_to_ids(self, name):
            return 105

    assert T.stop_ids(Tok()) == [1, 105]


# ---- K2 -------------------------------------------------------------------------------------------------------------


def test_k2_off_by_default_nothing_thinks_and_nothing_is_added():
    client, so, eng = make()
    r = ask(client, "0:0.5:x:1")  # an unsure question
    body = r.json()
    assert r.status_code == 200 and "thinking" not in body and "thinking_tokens" not in body["usage"]
    assert eng.calls == [] and "x-decisio-think-ms" not in r.headers
    assert body["usage"]["output_tokens"] == 1
    assert np.allclose(probs_of(body["answers"]["q"], 3), first_read(0, 0.5, 3))


# ---- K3 -------------------------------------------------------------------------------------------------------------


def test_k3_an_unsure_question_is_mixed_and_a_sure_one_is_not():
    client, so, eng = make(think=True, temperatures={"choice": 2.0})
    r = ask(client, "0:0.5:x:1")  # 0.5 / 2.0 temperature => unsure
    assert r.status_code == 200, r.text
    got = probs_of(r.json()["answers"]["q"], 3)
    lp2 = np.log(np.array([0.05, 0.9, 0.05]))
    assert np.allclose(got, expected(first_read(0, 0.5, 3), lp2, 2.0), atol=1e-9)
    assert len(eng.calls) == 1 and eng.calls[0][3] == 512
    sure = ask(client, "0:0.99:x:1")  # served at T=2 the top is still above 0.75
    assert np.allclose(probs_of(sure.json()["answers"]["q"], 3), apply_temperature(first_read(0, 0.99, 3), 2.0))
    assert len(eng.calls) == 1  # no second call for the sure one
    assert sure.json()["thinking"]["per_question"]["q"] == {"thought": False, "why": T.SURE}


def test_k3_the_unsure_test_reads_the_served_confidence_not_the_plain_one():
    client, so, eng = make(think=True, temperatures={"choice": 3.0})
    ask(client, "0:0.9:x:1")  # plain 0.9 is sure; served at T=3 it is below 0.75
    assert len(eng.calls) == 1
    client, so, eng = make(think=True)
    ask(client, "0:0.9:x:1")
    assert eng.calls == []


def test_k3_all_the_unsure_questions_of_a_request_share_one_engine_call():
    client, so, eng = make(think=True)
    body = wire("0:0.5:x:1")
    body["questions"]["q2"] = question(4)
    body["questions"]["q3"] = {"type": "noul", "instructions": "Is it a complaint?"}
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    assert len(eng.calls) == 1 and len(eng.calls[0][1]) == 3  # one call, three questions
    assert r.json()["thinking"]["questions"] == 3 and r.json()["thinking"]["of"] == 3


def test_k3_a_yes_no_asked_as_letters_is_mixed_in_the_orientation_it_is_served_in():
    from pydantic import TypeAdapter

    from decisio.serve.systemone import Question

    client, so, eng = make(think=True, noul_rendering="letters")
    asked = so.engine_question(TypeAdapter(Question).validate_python(question(2, "noul")))
    assert asked[2] is True, "asked as letters, a yes/no is read false first and reversed"
    r = ask(client, "0:0.6:x:1", qtype="noul")  # the engine's read [0.6, 0.4] is (yes, no) = [0.4, 0.6]
    assert r.status_code == 200, r.text
    # the second read, in the engine's order (false, true), is [0.1, 0.9]: (yes, no) = [0.9, 0.1]
    want = T.mix(np.array([0.4, 0.6]), np.log(np.array([0.9, 0.1])))
    assert abs(r.json()["answers"]["q"]["noul"] - want[0]) < 1e-9


def test_k3_the_commit_rule_is_applied_after_the_mix():
    client, so, eng = make(think=True, noul_commit=True)
    r = ask(client, "0:0.6:x:1", qtype="noul")
    py = r.json()["answers"]["q"]["noul"]
    assert py in (0.2, 0.8) or not 0.2 < py < 0.8  # committed, whatever the mix was


# ---- K4 -------------------------------------------------------------------------------------------------------------


def test_k4_the_request_field_beats_the_servers_setting_either_way():
    client, so, eng = make(think=False)
    r = ask(client, "0:0.5:x:1", think=True)
    assert r.status_code == 200 and r.json()["thinking"]["questions"] == 1 and len(eng.calls) == 1
    client, so, eng = make(think=True)
    r = ask(client, "0:0.5:x:1", think=False)
    assert r.status_code == 200 and "thinking" not in r.json() and eng.calls == []
    r = ask(client, "0:0.5:x:1", think=None)  # null: the server's own
    assert "thinking" in r.json() and len(eng.calls) == 1


def test_k4_a_bad_value_is_a_422_and_an_engine_that_cannot_write_refuses_the_request():
    client, *_ = make()
    for bad in ("yes", 1, ["true"]):
        r = ask(client, "0:0.5:x:1", think=bad)
        assert r.status_code == 422 and "think" in r.json()["detail"], bad
    client, so, eng = make(engine=ThinkEngine(can=False))
    r = ask(client, "0:0.5:x:1", think=True)
    assert r.status_code == 422 and "cannot write reasoning" in r.json()["detail"]
    assert ask(client, "0:0.5:x:1").status_code == 200  # the server's own setting is off


def test_k4_a_lower_factor_thinks_on_fewer_questions():
    base = [("0:0.5:x:1", 3), ("0:0.7:x:1", 3), ("0:0.9:x:1", 3)]
    thought = []
    for factor in (1.0, 0.9, 0.6):
        client, so, eng = make(think=True, factor=factor)
        n = sum(1 for m, k in base if ask(client, m, k).json()["thinking"]["questions"])
        thought.append(n)
    assert thought == [2, 1, 0]  # 0.5 and 0.7 are under 0.75; only 0.5 is under 0.675; none under 0.45


def test_k4_the_budget_reaches_the_engine():
    client, so, eng = make(think=True, budget=128)
    ask(client, "0:0.5:x:1")
    assert eng.calls[0][3] == 128


# ---- K5 -------------------------------------------------------------------------------------------------------------


def test_k5_more_than_sixteen_options_stay_at_one_read_and_say_so():
    client, so, eng = make(think=True)
    r = ask(client, "0:0.1:x:1", k=17)
    assert r.json()["thinking"]["per_question"]["q"] == {"thought": False, "why": T.TOO_MANY_OPTIONS}
    assert eng.calls == []
    r = ask(client, "0:0.1:x:1", k=16)
    assert r.json()["thinking"]["per_question"]["q"]["thought"] is True


def test_k5_a_question_a_registered_task_answers_stays_at_one_read():
    client, so, eng = make(think=True)
    r = post_task(client, {"id": "routing", "examples": examples(80, 0.9)})
    assert r.status_code == 200, r.text
    out = client.post("/v1/systemone", json={**wire("0:0.5:x:1"), "think": True})
    assert out.status_code == 200, out.text
    assert out.json()["thinking"]["per_question"]["q1"] == {"thought": False, "why": T.REGISTERED_TASK}
    assert eng.calls == []


def test_k5_two_order_requests_are_not_combined():
    client, so, eng = make(think=True)
    r = ask(client, "0:0.5:x:1", orders=2)
    assert r.status_code == 200 and eng.calls == []
    assert r.json()["thinking"]["per_question"]["q"] == {"thought": False, "why": T.NOT_COMBINED}


# ---- K6 -------------------------------------------------------------------------------------------------------------


def test_k6_the_response_the_usage_and_the_headers_report_what_thinking_cost():
    client, so, eng = make(think=True)
    body = wire("0:0.5:x:1")
    body["questions"]["q2"] = question(3)
    r = client.post("/v1/systemone", json=body)
    out = r.json()
    th = out["thinking"]
    assert th["questions"] == 2 and th["tokens"] == 40 + 41 and th["factor"] == 1.0 and th["budget"] == 512
    assert th["per_question"] == {"q1": {"thought": True, "tokens": 40}, "q2": {"thought": True, "tokens": 41}}
    assert th["ms"] > 0
    assert out["usage"]["thinking_tokens"] == 81 and out["usage"]["output_tokens"] == 2 + 81
    assert r.headers["x-decisio-think-tokens"] == "81" and float(r.headers["x-decisio-think-ms"]) > 0
    assert "think=" in r.headers["x-decisio-stages"]
    assert so.thought == {"requests": 1, "questions": 2, "tokens": 81, "ms": so.thought["ms"]}


def test_k6_health_says_the_setting_and_the_counts():
    client, so, eng = make(think=True, factor=0.79, budget=256)
    ask(client, "0:0.5:x:1")
    eng.repository_defaults, eng.repository = None, None
    from types import SimpleNamespace

    eng.fmt = SimpleNamespace(facts=lambda: {})
    try:
        body = TestClient(vllm_engine.make_app(eng, so)).get("/health").json()
    except Exception as e:  # make_app wants more of a real engine than the stand-in has
        pytest.skip(f"make_app needs a fuller engine: {e}")
    h = body["systemone"]["think"]
    assert h["default"] is True and h["factor"] == 0.79 and h["budget"] == 256 and h["requests"] == 1
    assert h["questions"] == 1 and h["tokens"] == 40


# ---- K7 -------------------------------------------------------------------------------------------------------------


def test_k7_the_same_request_answers_the_same_whatever_came_before_it():
    client, so, eng = make(think=True)
    first = ask(client, "0:0.5:x:1").json()["answers"]["q"]
    for i in range(60):  # a history of thinking requests: nothing is capped, nothing is remembered
        ask(client, f"0:0.5:x:{i % 3}")
    again = ask(client, "0:0.5:x:1")
    assert again.json()["answers"]["q"] == first and again.json()["thinking"]["questions"] == 1
    fresh_client, *_ = make(think=True)
    assert ask(fresh_client, "0:0.5:x:1").json()["answers"]["q"] == first


# ---- K8 -------------------------------------------------------------------------------------------------------------


def test_k8_the_fingerprint_holds_how_the_server_thinks_only_when_it_does():
    off = make()[1].stack_facts()
    assert "think" not in off["serving"]
    on = make(think=True)[1].stack_facts()
    assert set(on["serving"]["think"]) == {"factor", "budget", "procedure_sha256"}
    other = make(think=True, factor=0.79)[1].stack_facts()
    assert other["serving"]["think"] != on["serving"]["think"]
    assert {k: v for k, v in on["serving"].items() if k != "think"} == off["serving"]


def test_k8_a_certificate_is_think_mode_changed_on_a_server_that_thinks_and_certified_again_when_it_stops():
    client, so, eng = make()
    r = post_task(client, {"id": "routing", "examples": examples(400, 0.96), "certify": {}})
    assert r.status_code == 200 and r.json()["certificate"]["certified"], r.text
    msg = "0:0.99:x:1"
    assert client.post("/v1/systemone", json=wire(msg)).json()["answers"]["q1"]["certified"] is True

    def flip(on):
        so.think = on
        so.__dict__.pop("_stack", None)
        so.__dict__.pop("_stack_sha", None)

    flip(True)
    a = client.post("/v1/systemone", json=wire(msg)).json()["answers"]["q1"]
    assert a["certified"] is False and a["certified_reason"] == "think_mode_changed"
    status = client.get("/v1/tasks").json()["tasks"][0]["status"]
    assert status["state"] == "suspended" and "serving.think" in status["changed"]
    flip(False)
    assert client.post("/v1/systemone", json=wire(msg)).json()["answers"]["q1"]["certified"] is True
    # a different change of the stack is still its own reason
    so.noul_commit = True
    so.__dict__.pop("_stack", None)
    so.__dict__.pop("_stack_sha", None)
    b = client.post("/v1/systemone", json=wire(msg)).json()["answers"]["q1"]
    assert b["certified"] is False and b["certified_reason"] == "serving_stack_changed"


# ---- K9 -------------------------------------------------------------------------------------------------------------


def test_k9_a_repositorys_think_default_is_validated_and_reaches_the_arguments(tmp_path):
    d = write_repo(tmp_path, build(think={"factor": 0.79, "budget": 256}), name="t")
    args = args_for(str(d))
    resolve_base(args)
    assert args.think is True and args.think_factor == 0.79 and args.think_budget == 256
    assert args.repository_defaults["think"] == {"factor": 0.79, "budget": 256}
    # a flag beats the repository: --no-think, --think-factor, --think-budget
    args = args_for(str(d), think=False)
    resolve_base(args)
    assert args.think is False and args.repository_defaults["think_set_by"] == "--no-think"
    args = args_for(str(d), think_factor=0.9, think_budget=64)
    resolve_base(args)
    assert (args.think, args.think_factor, args.think_budget) == (True, 0.9, 64)
    # absent: off, factor 1, budget 512
    plain = write_repo(tmp_path, build(), name="p")
    args = args_for(str(plain))
    resolve_base(args)
    assert (args.think, args.think_factor, args.think_budget) == (False, 1.0, 512)
    base = args_for("gemma-4-31b")
    resolve_base(base)
    assert base.think is False
    # --think on a base: on at the defaults
    on = args_for("gemma-4-31b", think=True)
    resolve_base(on)
    assert (on.think, on.think_factor) == (True, 1.0)


def test_k9_a_bad_think_default_is_refused():
    config = build()
    for bad in ({"factor": 0}, {"factor": 0.8, "budget": 3}, {"factor": 0.8, "tau": 1}, {}, 0.8):
        config["defaults"]["think"] = bad
        with pytest.raises(ValueError):
            hub.check_defaults(config["defaults"])


def test_k9_tuned_config_writes_the_think_default(tmp_path):
    config = build(think={"factor": 0.79})
    assert config["defaults"]["think"] == {"factor": 0.79}
    assert hub.defaults_sha256(config) != hub.defaults_sha256(build())  # the digest, hence the identity, follows it


def test_k9_the_flags_are_in_the_parser():
    a = vllm_engine.build_parser().parse_args(["--think", "--think-factor", "0.79", "--think-budget", "256"])
    assert (a.think, a.think_factor, a.think_budget) == (True, 0.79, 256)
    b = vllm_engine.build_parser().parse_args(["--no-think"])
    assert b.think is False and vllm_engine.build_parser().parse_args([]).think is None


def test_k9_an_older_reader_refuses_the_key():
    old_keys = {"temperatures", "choice_bands", "noul_commit", "fit", "noul_commit_basis", "over_limit"}
    config = build(think={"factor": 0.79})
    assert set(config["defaults"]) - old_keys == {"think"}  # a 0.20.0 decisio rejects unknown keys of `defaults`


# ---- K10 ------------------------------------------------------------------------------------------------------------


def gemma_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("google/gemma-4-31B-it", local_files_only=True)
    except Exception as e:  # not in the local cache
        pytest.skip(f"the Gemma 4 tokenizer is not cached: {e}")


def test_k10_the_options_after_the_suffix_are_the_tokens_the_first_read_uses():
    from decisio.families import GEMMA4
    from decisio.readout.letters import PromptFormat, label_groups

    tok = gemma_tokenizer()
    fmt = PromptFormat(
        tail=GEMMA4.prompt_tail,
        slot=GEMMA4.answer_slot,
        variants=GEMMA4.label_variants,
        system_prompt=GEMMA4.system_prompt,
    )
    base = tok(T.SUFFIX, add_special_tokens=False)["input_ids"]
    for k in (2, 3, 8, 16):
        cands = [" " + chr(65 + i) for i in range(k)]
        groups = label_groups(tok, cands, fmt)
        after = []
        for i in range(k):
            full = tok(T.SUFFIX + " " + chr(65 + i), add_special_tokens=False)["input_ids"]
            assert full[: len(base)] == base and len(full) == len(base) + 1
            after.append(full[-1])
        assert [g[0] for g in groups] == after, "the first read's label tokens are the ones that follow the suffix"
    json.dumps(T.TAU)


# ---- K11 ------------------------------------------------------------------------------------------------------------


class FakeVLLM:
    """Stands in for `vllm` so that `LettersEngine.think` runs without a GPU: it records what was sent and answers a
    generation (more than one token asked) with fixed tokens and a read (one token, logprobs) with fixed letters."""

    def __init__(self):
        self.sent = []

    class SamplingParams:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class TokensPrompt:
        def __init__(self, prompt_token_ids, **kw):
            self.prompt_token_ids = prompt_token_ids

    def generate(self, prompts, sp, lora_request=None, use_tqdm=False):
        from types import SimpleNamespace as NS

        sps = sp if isinstance(sp, list) else [sp] * len(prompts)
        self.sent.append(([p.prompt_token_ids for p in prompts], sps, lora_request))
        outs = []
        for p, s in zip(prompts, sps):
            if s.max_tokens > 1:  # the reasoning: three tokens (the last is the end of the turn)
                outs.append(
                    NS(
                        outputs=[NS(token_ids=[7, 8, 105], logprobs=None)],
                        num_cached_tokens=0,
                        prompt_token_ids=p.prompt_token_ids,
                        metrics=None,
                    )
                )
            else:  # the read: letter ids 11, 12, 13 with log-probabilities that favour the last
                d = {11: NS(logprob=-2.0), 12: NS(logprob=-1.5), 13: NS(logprob=-0.2)}
                outs.append(
                    NS(
                        outputs=[NS(token_ids=[13], logprobs=[d])],
                        num_cached_tokens=0,
                        prompt_token_ids=p.prompt_token_ids,
                        metrics=None,
                    )
                )
        return outs


def test_k11_the_engine_writes_after_the_prefix_and_reads_after_the_suffix(monkeypatch):
    import sys
    import threading
    import types

    fake = FakeVLLM()
    mod = types.ModuleType("vllm")
    mod.SamplingParams = FakeVLLM.SamplingParams
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = FakeVLLM.TokensPrompt
    monkeypatch.setitem(sys.modules, "vllm", mod)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)

    class Tok:
        eos_token_id = 1

        def encode(self, text, add_special_tokens=False):
            return {T.PREFIX: [50, 51], T.SUFFIX: [60]}[text]

        def get_vocab(self):
            return {"<turn|>": 105}

        def convert_tokens_to_ids(self, name):
            return 105

    eng = vllm_engine.LettersEngine.__new__(vllm_engine.LettersEngine)
    eng.llm, eng.tok, eng.adapters, eng._lock = fake, Tok(), {"n1": "LORA"}, threading.Lock()
    lab = ((11,), (12,), (13,))
    eng._prepare_separate = lambda state, questions: ([([1, 2, 3], lab)], 3)
    assert eng.can_think()
    lps, counts, parts = eng.think("state", [{"kind": "choice", "options": ["a", "b", "c"]}], "n1", 64)
    gen, read = fake.sent
    assert gen[0] == [[1, 2, 3, 50, 51]] and gen[1][0].max_tokens == 64 and gen[1][0].temperature == 0.0
    assert gen[1][0].stop_token_ids == [1, 105] and gen[2] == "LORA"  # the named adapter writes
    assert read[0] == [[1, 2, 3, 50, 51, 7, 8, 105, 60]]  # prompt + prefix + what it wrote + suffix
    assert read[2] == "LORA" and counts == [3] and set(parts) == {"generate_ms", "read_ms"}
    p = np.exp(lps[0])
    assert abs(p.sum() - 1) < 1e-9 and p.argmax() == 2
    eng.think("state", [{"kind": "choice", "options": ["a", "b", "c"]}], None, 64)
    assert fake.sent[2][2] is None  # no adapter named: the weights as loaded write (the merged model)
    assert not vllm_engine.LettersEngine.__new__(vllm_engine.LettersEngine).can_think()


def test_k3_the_engine_is_asked_about_the_unsure_questions_among_the_questions_it_read():
    client, so, eng = make(think=True)
    body = wire("0:0.5:x:1")  # q1 is unsure
    body["questions"]["q2"] = question(3)
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200
    # the engine got both questions' rows (so the padding is the first read's) and the position to think about
    state, kinds, adapter, budget, subset = eng.calls[0]
    assert subset == [0, 1] and len(kinds) == 2
    assert len(r.json()["thinking"]["per_question"]) == 2


def test_k11_the_subset_picks_rows_of_the_first_reads_questions(monkeypatch):
    import sys
    import threading
    import types

    fake = FakeVLLM()
    mod = types.ModuleType("vllm")
    mod.SamplingParams = FakeVLLM.SamplingParams
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = FakeVLLM.TokensPrompt
    monkeypatch.setitem(sys.modules, "vllm", mod)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)

    class Tok:
        eos_token_id = 1

        def encode(self, text, add_special_tokens=False):
            return {T.PREFIX: [50], T.SUFFIX: [60]}[text]

        def get_vocab(self):
            return {}

    eng = vllm_engine.LettersEngine.__new__(vllm_engine.LettersEngine)
    eng.llm, eng.tok, eng.adapters, eng._lock = fake, Tok(), {}, threading.Lock()
    lab = ((11,), (12,), (13,))
    eng._prepare_separate = lambda state, questions: ([([1, 2], lab), ([1, 2, 9], lab)], 2)
    lps, counts, _ = eng.think("s", [{}, {}], None, 32, [1])
    assert fake.sent[0][0] == [[1, 2, 9, 50]] and len(lps) == 1 and counts == [3]  # only the second question's row
