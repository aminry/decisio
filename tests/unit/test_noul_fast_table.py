# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A yes/no task's fast table (docs/tasks.md, "A fast table in front of decisio"), the per-task `min_saving`, and the
confidence monitor of a yes/no task with a table under --noul-commit. No model and no download: the fake engine, the
static student's stand-in and the virtual clock of tests/unit/test_fast_table_task.py and
tests/unit/test_fast_table_kinds.py. The state is `top:confidence|text`: decisio puts that confidence on yes (top 0) or
no (top 1); the text says yes or no in words the table can learn, or says nothing it can use.

Y1  a labelled yes/no table: kept, two classes, answers `answered_by: "table"` with its own P(yes), certified gold,
    without asking the model; what it is not sure of goes to decisio
Y2  the table's P(yes) is calibrated by one temperature cross-fitted on the fit examples (never the certification
    examples, never traffic, not refitted by a re-certification): changing only the certification labels leaves the
    temperature and the table's scores bit for bit; the served value is sigmoid(logit / T); the answer and the order of
    the confidences are the uncalibrated table's
Y3  the commit rule on table answers: reported at the band's edge, the answer unchanged; the certificate is read on the
    committed confidences, as decisio's yes/no answers are
Y4  certificate kinds: a learned yes/no table (agreement, calibrated on its fit rows against decisio's answers) and a
    labelled task that learns (gold)
Y5  min_saving: the default 0.10 and its record, the range (422 outside), a per-task bar at registration, the runtime
    fallback under load, and a record without it held to 0.25
Y6  monitor gap (a): a table registration, a re-certification and a registration whose table is not kept make the
    confidence baseline before the commit rule, marked `"reads": "before_noul_commit"`
Y7  monitor gap (b): table, deferred, table-off and learning rows feed the monitor the confidence before the rule (a
    table answer its calibrated P(yes))
Y8  export and import, a task directory, and a replica on a shared store read the table, its calibration and its bar

  uv run pytest -q tests/unit/test_noul_fast_table.py
"""

import json
import math

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("obstore")
pytest.importorskip("tokenizers")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from obstore.store import MemoryStore  # noqa: E402
from test_fast_table_kinds import fake_static_embed, students  # noqa: E402,F401  (the static stand-in: autouse)
from test_fast_table_task import (  # noqa: E402
    FILLER,
    FakeEngine,
    virtual_time,  # noqa: F401  (the virtual clock: autouse)
)

from decisio.readout import certify  # noqa: E402
from decisio.serve import fast_table as FT  # noqa: E402
from decisio.serve import fast_table_learn as FTL  # noqa: E402
from decisio.serve import monitor  # noqa: E402
from decisio.serve import task_backends as TB  # noqa: E402
from decisio.serve import task_store_shared as S  # noqa: E402
from decisio.serve.systemone import COMMIT_HI, COMMIT_LO, SystemOne, SystemOneRequest, add_routes  # noqa: E402
from decisio.serve.tasks import TaskStore, example_digest  # noqa: E402

QUESTION = {"type": "noul", "instructions": "Is the customer angry?"}
YES_WORDS = ["furious", "angry", "outraged", "terrible", "worst", "unacceptable"]
NO_WORDS = ["thanks", "lovely", "great", "happy", "pleased", "wonderful"]


@pytest.fixture(autouse=True)
def learning_by_hand(monkeypatch):
    monkeypatch.setattr(FTL, "BACKGROUND", False)
    monkeypatch.setattr(FTL, "SEED", 0)


def body(yes, rng, words=4):
    """A text that says yes (or no) in `words` words of its list, among six filler words."""
    out = list(rng.choice(YES_WORDS if yes else NO_WORDS, words, replace=False)) if words else []
    out += list(rng.choice(FILLER, 6, replace=False))
    rng.shuffle(out)
    return " ".join(out)


def state(says_yes, conf, text):
    return f"{0 if says_yes else 1}:{conf:.4f}|{text}"


def wire(s):
    return {"state": s, "model": "m", "questions": {"q1": dict(QUESTION)}}


def noul_examples(n, seed=0, accuracy=0.9, clear=0.7, weak=0.0):
    """n labelled yes/no examples: decisio is right with probability `accuracy`, sure (about 0.93) when right and unsure
    (about 0.65) when wrong; `clear` of the texts say the answer in four words, `weak` of them in one word, the rest in
    none."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        yes = bool(rng.random() < 0.5)
        right = bool(rng.random() < accuracy)
        conf = float(np.clip(rng.normal(0.93 if right else 0.65, 0.03), 0.51, 0.995))
        u = rng.random()
        words = 4 if u < clear else 1 if u < clear + weak else 0
        out.append({"request": wire(state(yes if right else not yes, conf, body(yes, rng, words))), "answer": yes})
    return out


def make(commit=False, store=None, tasks_dir=None, name="decisio-test"):
    backend = None
    if store is not None:
        backend = S.ObjectBackend(store, "memory://t")
    elif tasks_dir is not None:
        backend = TB.DirBackend(tasks_dir)
    ts = TaskStore("fp")
    extra = {}
    if backend is not None:
        abstention, report = TB.load_at_start(ts, backend)
        extra = {"abstention_tasks": abstention, "task_backend": backend, "task_report": report}
    so = SystemOne(FakeEngine(), name, task_store=ts, noul_commit=commit, noul_rendering="words", **extra)
    app = FastAPI()
    add_routes(app, so)
    return TestClient(app), so, backend


def register(client, n=800, seed=0, fast_table=True, certify_block=None, **kw):
    req = {
        "id": "angry",
        "examples": noul_examples(n, seed, **kw),
        "certify": certify_block or {"alpha": 0.1, "delta": 0.1},
        "fast_table": fast_table,
    }
    return client.post("/v1/tasks", json=req)


def ask(client, s):
    r = client.post("/v1/systemone", json=wire(s))
    assert r.status_code == 200, r.text
    return r.json()["answers"]["q1"]


def top(a) -> float:
    return max(a["noul"], 1.0 - a["noul"])


def table_p(so, s, calibrated=True):
    """The task's table's own P(yes) for state s, computed from its record (calibrated, or not)."""
    rec = dict(so.task_store.snapshot("angry")["fast_table"])
    if not calibrated:
        rec.pop("calibration", None)
    return float(FT.predict(rec, fake_static_embed("base", [s]))[0, 0])


def watch(m):
    """The confidences monitor m observes from now on."""
    seen = []
    real = m.observe

    def observe(index, certified=False, confidence=None, **kw):
        seen.append(confidence)
        return real(index, certified, confidence, **kw)

    m.observe = observe
    return seen


RNG = np.random.default_rng(123)
CLEAR_YES = state(False, 0.6, " ".join(YES_WORDS[:4] + FILLER[:6]))  # decisio unsure and wrong; the text says yes
CLEAR_NO = state(True, 0.6, " ".join(NO_WORDS[:4] + FILLER[6:12]))
SILENT = state(True, 0.62, "")  # no word at all: the table is not sure, decisio answers (unsure, yes)


# ---- Y1 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("commit", [False, True])
def test_y1_a_labelled_yes_no_table_is_kept_and_answers_what_it_is_sure_of_without_the_model(commit):
    # on the old code: 422 "fast_table: choice questions only"
    client, so, _ = make(commit)
    r = register(client)
    assert r.status_code == 200, r.text
    t = r.json()
    rep, ft, c = t["fast_table_report"], t["fast_table"], t["certificate"]
    assert rep["kept"] and ft["k"] == 2 and ft["binary"] and ft["classes"] == [0, 1]
    assert ft["calibration"]["method"] == "cross_fit" and rep["calibration"] == ft["calibration"]
    assert c["certified"] and c["table_threshold"] is not None and c["stages"]["table"]["certified_n"] > 0
    assert t["status"]["fast_table"]["state"] == "active"
    sc = c["served_check"]  # the serving path reads what the certificate was made on, the commit rule included
    assert sc["max_shift"] == 0 and sc["crossed_threshold"] == 0 and sc["top_answer_changed"] == 0
    calls = FakeEngine.calls
    y, n = ask(client, CLEAR_YES), ask(client, CLEAR_NO)
    assert FakeEngine.calls == calls  # the model was not asked
    for a, yes in ((y, True), (n, False)):
        assert a["answered_by"] == "table" and (a["noul"] > 0.5) is yes and a["certified"] is True
        assert a["certificate"] == c["id"] and a["certificate_kind"] == "gold"
    d = ask(client, SILENT)
    assert d["answered_by"] == "decisio" and FakeEngine.calls == calls + 1 and d["noul"] > 0.5
    if not commit:
        assert d["noul"] == pytest.approx(0.62, abs=1e-6)  # decisio's own read, before any rule


def test_y1_a_yes_no_table_needs_a_certify_block_and_enough_fit_examples():
    client, so, _ = make()
    r = client.post("/v1/tasks", json={"id": "angry", "examples": noul_examples(400), "fast_table": True})
    assert r.status_code == 422 and "certify" in r.json()["detail"]
    r = register(client, n=150)
    assert r.status_code == 422 and "at least 100" in r.json()["detail"] and not so.task_store.by_key


# ---- Y2 -------------------------------------------------------------------------------------------------------------


def parsed(ex):
    """[(question, state, gold)] of wire examples, as a registration reads them."""
    out = []
    for e in ex:
        req = SystemOneRequest(**e["request"])
        out.append((req.questions["q1"], req.state, e["answer"]))
    return out


def test_y2_the_temperature_is_cross_fitted_on_the_fit_examples_only():
    client, so, _ = make()
    ex = noul_examples(800)
    r = client.post(
        "/v1/tasks", json={"id": "angry", "examples": ex, "certify": {"alpha": 0.1, "delta": 0.1}, "fast_table": True}
    )
    assert r.status_code == 200, r.text
    t = so.task_store.snapshot("angry")
    fit_digests = set(t["certificate"]["split"]["fit_digests"])
    fit = [(q, s, g) for q, s, g in parsed(ex) if example_digest(s, q)[:16] in fit_digests]
    assert len(fit) == t["n_examples"] == 400
    cal = t["fast_table"]["calibration"]
    assert cal["method"] == "cross_fit" and cal["folds"] == FT.FOLDS == 5 and cal["fitted_on"] == "fit"
    assert cal["n"] == 400 and cal["reference"] == "gold" and cal["log_loss_after"] <= cal["log_loss_before"]
    # the same function on the fit examples' features and answers gives it, bit for bit: the served table is the one
    # fitted on all of them, the temperature comes from its folds
    X = fake_static_embed("base", [s for _, s, _ in fit])
    y = [0 if g else 1 for _, _, g in fit]
    raw = {k: v for k, v in t["fast_table"].items() if k != "calibration"}
    assert FT.cross_fit_temperature(X, y, raw, reference="gold") == cal
    # traffic does not move it, and neither does a re-certification (it reads no held-out label)
    for i in range(30):
        ask(client, state(True, 0.9, body(bool(i % 2), RNG)))
    assert so.task_store.snapshot("angry")["fast_table"]["calibration"] == cal
    r = client.post("/v1/tasks/angry/certify", json={"examples": noul_examples(400, seed=91)})
    assert r.status_code == 200, r.text
    after = so.task_store.snapshot("angry")
    assert after["certificate"]["id"] != t["certificate"]["id"] and after["fast_table"]["calibration"] == cal


def test_y2_the_certification_labels_do_not_shape_the_temperature_or_the_tables_scores():
    # on the earlier version of this change (T fitted on the held-out examples) both differ
    client, so, _ = make()
    fit, held = parsed(noul_examples(400, seed=3)), parsed(noul_examples(300, seed=4))
    flipped = [(q, s, (not g) if i % 3 == 0 else g) for i, (q, s, g) in enumerate(held)]  # a third of the labels
    a = so.prepare_fast_table(fit, held, request=FT.parse_request(True))["candidates"][0]["record"]
    b = so.prepare_fast_table(fit, flipped, request=FT.parse_request(True))["candidates"][0]["record"]
    assert a["calibration"] == b["calibration"] and a["calibration"]["temperature"] != 1.0
    probe = np.random.default_rng(0).normal(size=(50, 256)).astype(np.float32)
    probe = np.vstack([probe, fake_static_embed("base", [CLEAR_YES, CLEAR_NO, SILENT])])
    assert np.array_equal(FT.predict(a, probe), FT.predict(b, probe))  # bit for bit, on any input
    # end to end: the same fit examples, certification examples that differ only in a few labels
    ex, cert = noul_examples(600, seed=5), noul_examples(400, seed=6)
    changed = [dict(e, answer=(not e["answer"]) if i % 40 == 0 else e["answer"]) for i, e in enumerate(cert)]
    tables = []
    for c in (cert, changed):
        other, so2, _ = make()
        body_ = {"id": "angry", "examples": ex, "certify": {"alpha": 0.1, "delta": 0.1, "examples": c}}
        r = other.post("/v1/tasks", json={**body_, "fast_table": True})
        assert r.status_code == 200 and r.json()["fast_table_report"]["kept"], r.text
        tables.append((so2.task_store.snapshot("angry")["fast_table"], ask(other, CLEAR_YES)))
    (ta, ya), (tb, yb) = tables
    assert ta["calibration"] == tb["calibration"] and ta["coef"] == tb["coef"] and ta["intercept"] == tb["intercept"]
    assert np.array_equal(FT.predict(ta, probe), FT.predict(tb, probe))
    assert ya["answered_by"] == yb["answered_by"] == "table" and ya["noul"] == yb["noul"]


def test_y2_the_served_p_yes_is_the_tables_logit_over_the_temperature_and_its_answer_is_unchanged():
    client, so, _ = make()
    assert register(client).status_code == 200
    T = so.task_store.snapshot("angry")["fast_table"]["calibration"]["temperature"]
    assert T != 1.0
    for s in (CLEAR_YES, CLEAR_NO):
        a = ask(client, s)
        raw = table_p(so, s, calibrated=False)
        z = math.log(raw / (1 - raw))
        assert a["answered_by"] == "table" and a["noul"] == pytest.approx(1 / (1 + math.exp(-z / T)), abs=1e-9)
        assert (a["noul"] > 0.5) == (raw > 0.5)


def test_y2_the_temperature_keeps_the_answer_and_the_order_and_stays_finite_on_separated_examples():
    rng = np.random.default_rng(0)
    py = rng.uniform(0.01, 0.99, 300)
    z0 = np.log(py / (1 - py))
    yes = rng.random(300) < py**3 / (py**3 + (1 - py) ** 3)  # truer than the table says: it is underconfident
    T, before, after = FT.temperature_for(z0, yes)
    assert T < 1.0 and after < before
    z = z0 / T
    pc = 1 / (1 + np.exp(-z))
    assert ((pc > 0.5) == (py > 0.5)).all()
    assert (np.argsort(np.abs(z)) == np.argsort(np.abs(z0))).all()
    # the same certificate on the calibrated confidences as on the table's own: the order is the same
    right = (py > 0.5) == yes
    a, b = np.maximum(py, 1 - py), np.maximum(pc, 1 - pc)
    ta, tb = certify.threshold(a, right, 0.2, 0.1), certify.threshold(b, right, 0.2, 0.1)
    assert ((a >= ta) == (b >= tb)).all()
    # every reference answer is the table's: Platt's targets keep T finite
    T_sep, _, _ = FT.temperature_for(z0, py > 0.5)
    assert FT.TEMPERATURES[0] < T_sep < FT.TEMPERATURES[1]


def test_y2_cross_fitting_uses_fewer_folds_on_a_small_answer_and_none_below_two():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 8))
    y = np.array([0] * 3 + [1] * 117)
    rec = {"C": 1.0}
    assert FT.cross_fit_temperature(X, y, rec)["folds"] == 3
    one = FT.cross_fit_temperature(X, np.array([0] + [1] * 119), rec)
    assert one["folds"] == 0 and one["temperature"] == 1.0


# ---- Y3 -------------------------------------------------------------------------------------------------------------


def test_y3_the_commit_rule_applies_to_table_answers_and_the_certificate_reads_the_committed_confidence():
    # every text says its answer, one in four in a single word: the table is right nearly always, so its threshold
    # can reach down to the band's edge, and a one-word text gets a P(yes) inside the band
    client, so, _ = make(commit=True)
    r = register(client, n=1000, accuracy=0.9, clear=0.75, weak=0.25)
    assert r.status_code == 200, r.text
    t = so.task_store.snapshot("angry")
    thr = t["fast_table"]["thresholds"]["table_threshold"]
    weak = [state(False, 0.6, " ".join([YES_WORDS[i]] + FILLER[30 + 6 * i : 36 + 6 * i])) for i in range(6)]
    weak += [state(True, 0.6, " ".join([NO_WORDS[i]] + FILLER[80 + 6 * i : 86 + 6 * i])) for i in range(6)]
    in_band = [s for s in weak if COMMIT_LO < table_p(so, s) < COMMIT_HI]
    assert in_band, "no one-word text puts the table inside the band"
    assert thr <= COMMIT_HI  # read on the committed confidences: the band's answers are certified as a whole
    for s in in_band:
        a = ask(client, s)
        p = table_p(so, s)
        assert a["answered_by"] == "table" and a["certified"] is True
        assert a["noul"] == (COMMIT_HI if p > 0.5 else COMMIT_LO)  # at the edge, on the table's own side
    # the same table without the rule reports its own P(yes) inside the band
    plain, so2, _ = make(commit=False)
    full = client.get("/v1/tasks?full=1").json()["tasks"]
    assert plain.post("/v1/tasks/import", json={"tasks": full}).status_code == 200
    for s in in_band:
        a = ask(plain, s)
        assert a["noul"] == pytest.approx(table_p(so, s), abs=1e-9) or a["answered_by"] == "decisio"


def test_y3_every_table_answer_under_the_rule_is_outside_the_band_on_the_tables_side():
    client, so, _ = make(commit=True)
    assert register(client).status_code == 200
    rng = np.random.default_rng(5)
    answered = 0
    for i in range(80):
        yes = bool(i % 2)
        s = state(not yes, 0.6, body(yes, rng, words=int(rng.integers(0, 5))))
        a = ask(client, s)
        assert not COMMIT_LO < a["noul"] < COMMIT_HI
        if a["answered_by"] == "table":
            answered += 1
            assert (a["noul"] > 0.5) == (table_p(so, s) > 0.5)
    assert answered > 10


# ---- Y4 -------------------------------------------------------------------------------------------------------------


def learning_traffic(client, n, seed=0, clear=0.7):
    """n requests of a learning yes/no task: decisio is right and sure; `clear` of the texts say the answer."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yes = bool(rng.random() < 0.5)
        words = 4 if rng.random() < clear else 0
        ask(client, state(yes, float(np.clip(rng.normal(0.93, 0.03), 0.6, 0.995)), body(yes, rng, words)))


@pytest.mark.parametrize("commit", [False, True])
def test_y4_a_learned_yes_no_table_carries_an_agreement_certificate_and_a_p_yes_calibrated_on_decisios_answers(commit):
    # on the old code: 422 "fast_table.learn: choice questions only"
    client, so, _ = make(commit)
    r = client.post("/v1/tasks", json={"id": "angry", "question": QUESTION, "fast_table": {"learn": True}})
    assert r.status_code == 200, r.text
    t = r.json()
    assert t["type"] == "noul" and t["options"] == ["yes", "no"]
    assert t["learn"]["request"]["min_saving"] == FT.MIN_SAVING
    learning_traffic(client, 1200)
    so.learning.drain()
    out = so.learning.refit_task("angry")
    assert out["outcome"] == "promoted", out
    L = so.learning.learners["angry"]
    v = L.live
    cal = v.record["calibration"]
    assert cal["method"] == "cross_fit" and cal["reference"] == "decisio" and cal["fitted_on"] == "fit"
    assert v.certificate_kind == "agreement" and cal["n"] == v.d["fit_rows"]  # its fit rows, not its held-out ones
    L.explore = 0.0
    calls = FakeEngine.calls
    a = ask(client, CLEAR_YES)
    assert a["answered_by"] == "table" and a["certificate_kind"] == "agreement" and a["noul"] > 0.5
    assert FakeEngine.calls == calls
    own = float(FT.predict(v.record, fake_static_embed("base", [CLEAR_YES]))[0, 0])
    if commit:
        assert a["noul"] == (own if own >= COMMIT_HI else COMMIT_HI)
    else:
        assert a["noul"] == pytest.approx(own, abs=1e-9)
    # the logged rows hold decisio's distribution before the rule
    rows = L.log.read()
    assert rows["dist"].shape[1] == 2 and set(rows["top"].tolist()) == {0, 1}


def test_y4_a_labelled_yes_no_task_that_learns_has_a_gold_certificate_on_its_labelled_table():
    client, so, _ = make()
    r = register(client, fast_table={"learn": True, "explore": 0.05})
    assert r.status_code == 200, r.text
    t = r.json()
    assert t["status"]["fast_table"]["mode"] == "both" and t["fast_table_report"]["kept"]
    so.learning.learners["angry"].explore = 0.0
    a = ask(client, CLEAR_YES)
    assert a["answered_by"] == "table" and a["certificate_kind"] == "gold" and a["certified"] is True


# ---- Y5 -------------------------------------------------------------------------------------------------------------


def test_y5_the_default_bar_is_ten_percent_and_is_recorded():
    client, so, _ = make()
    t = register(client).json()
    assert FT.MIN_SAVING == 0.10
    assert t["fast_table"]["min_saving"] == 0.10 and t["fast_table_report"]["required"] == 0.10
    assert t["fast_table_report"]["request"]["min_saving"] == 0.10
    assert t["status"]["fast_table"]["min_saving"] == 0.10


@pytest.mark.parametrize("bad", [0.005, 0.95, 0, 1, -0.1, "0.2", True, None])
def test_y5_a_bar_outside_0_01_to_0_90_is_refused_with_422(bad):
    # on the old code: 422 too, but for an unknown key ("not 'min_saving'"), whatever the value
    client, so, _ = make()
    r = register(client, fast_table={"min_saving": bad})
    detail = r.json()["detail"]
    assert r.status_code == 422 and "fast_table.min_saving" in detail and "from 0.01 to 0.9" in detail, r.text
    assert not so.task_store.by_key
    for value in (bad, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="fast_table.min_saving"):
            FT.parse_request({"min_saving": value})


@pytest.mark.parametrize("edge", [0.01, 0.9])
def test_y5_the_range_is_inclusive(edge):
    assert FT.parse_request({"min_saving": edge})["min_saving"] == edge


def test_y5_the_bar_is_per_task_at_registration():
    # the static student takes 35 ms, decisio 60 ms: the table saves about a quarter, kept at the default 10% and not
    # at 40% (on the old code the object form refuses min_saving)
    import test_fast_table_kinds as K

    K.STATIC[0] = 35.0
    client, so, _ = make()
    t = register(client).json()
    saving = t["fast_table_report"]["guard"]["saving"]
    assert 0.10 <= saving < 0.40 and t["fast_table_report"]["kept"]
    other, so2, _ = make()
    t2 = register(other, fast_table={"min_saving": 0.4}).json()
    rep = t2["fast_table_report"]
    assert not rep["kept"] and "under the 40% required" in rep["reason"] and rep["required"] == 0.4
    assert rep["guard"]["saving"] == saving and "fast_table" not in t2 and t2["certificate"]["certified"]


def test_y5_the_runtime_fallback_uses_the_tasks_bar():
    # a table kept under a bar of 70%: when the host's load cuts its saving to about 57%, it falls back to decisio,
    # where the old fixed bar of 25% would have kept it on
    client, so, _ = make()
    t = register(client, fast_table={"min_saving": 0.7}, clear=0.9).json()
    assert t["fast_table_report"]["kept"] and t["fast_table"]["min_saving"] == 0.7, t["fast_table_report"]
    import test_fast_table_kinds as K

    K.STATIC[0] = 20.0  # the static student's median rises to a third of decisio's time
    for i in range(FT.MIN_LATENCY_SAMPLES + 2):
        ask(client, state(False, 0.6, " ".join(YES_WORDS[:4] + FILLER[i : i + 6])))
    st = client.get("/v1/tasks/angry").json()["status"]["fast_table"]
    assert st["state"] == "timing" and "a saving of 70% required" in st["reason"], st


def test_y5_a_record_without_a_bar_is_held_to_the_quarter_it_was_kept_under():
    client, so, _ = make()
    assert register(client).status_code == 200
    full = client.get("/v1/tasks?full=1").json()["tasks"]
    del full[0]["fast_table"]["min_saving"]  # as a table registered before the setting is recorded
    other, so2, _ = make()
    assert other.post("/v1/tasks/import", json={"tasks": full}).status_code == 200
    rt = so2.table_runtime(so2.task_store.snapshot("angry"))
    assert rt.min_saving == FT.LEGACY_MIN_SAVING == 0.25 and rt.facts()["min_saving"] == 0.25
    # the runtime fallback reads it: a median that saves 20% falls back under 0.25 and not under 0.10
    reg = rt.registered_ms
    p50 = 0.8 * reg - (1 - rt.share) * reg  # expected = 0.8 * registered: a saving of 20%
    for _ in range(FT.MIN_LATENCY_SAMPLES):
        rt.table_answered(p50)
    assert rt.state == "timing" and "a saving of 25% required" in rt.reason
    new = FT.Runtime(so.task_store.snapshot("angry")["fast_table"], rt.stack)
    for _ in range(FT.MIN_LATENCY_SAMPLES):
        new.table_answered(p50)
    assert new.min_saving == 0.10 and new.state == "active"
    # a re-certification keeps the bar it was kept under, and now records it
    r = other.post("/v1/tasks/angry/certify", json={"examples": noul_examples(400, seed=77)})
    assert r.status_code == 200, r.text
    assert r.json()["fast_table_report"]["required"] == 0.25
    assert r.json().get("fast_table") is None or r.json()["fast_table"]["min_saving"] == 0.25


def test_y5_a_learning_task_records_its_bar_and_one_without_it_reads_the_quarter():
    client, so, _ = make()
    r = client.post(
        "/v1/tasks", json={"id": "angry", "question": QUESTION, "fast_table": {"learn": True, "min_saving": 0.4}}
    )
    assert r.status_code == 200 and r.json()["learn"]["request"]["min_saving"] == 0.4
    assert so.learning.learners["angry"].min_saving == 0.4
    assert client.get("/v1/tasks/angry").json()["status"]["fast_table"]["min_saving"] == 0.4
    t = dict(so.task_store.snapshot("angry"))
    t["learn"] = {**t["learn"], "request": {k: v for k, v in t["learn"]["request"].items() if k != "min_saving"}}
    assert FTL.Learner(so.learning, t, FTL.MemoryPlace()).min_saving == FT.LEGACY_MIN_SAVING


# ---- Y6 -------------------------------------------------------------------------------------------------------------


def held_out(so, ex):
    """The examples of `ex` the task held out of its fit: [(state, question)]."""
    fit = set(so.task_store.snapshot("angry")["certificate"]["split"]["fit_digests"])
    out = []
    for e in ex:
        req = SystemOneRequest(**e["request"])
        if example_digest(req.state, req.questions["q1"])[:16] not in fit:
            out.append((req.state, req.questions["q1"]))
    return out


def test_y6_a_table_registration_makes_the_baseline_before_the_rule():
    # on the old code: the baseline is made on the committed confidences (never below 0.80) with no "reads" marker
    client, so, _ = make(commit=True)
    ex = noul_examples(800)
    r = client.post(
        "/v1/tasks", json={"id": "angry", "examples": ex, "certify": {"alpha": 0.1, "delta": 0.1}, "fast_table": True}
    )
    assert r.status_code == 200, r.text
    t = so.task_store.snapshot("angry")
    held = held_out(so, ex)
    decisio = np.array([max(so.served_rows(t, [pair], commit=False)[0]) for pair in held])
    P = FT.predict(t["fast_table"], fake_static_embed("base", [s for s, _ in held]))
    by_table = FT.top_confidence(P, (COMMIT_LO, COMMIT_HI)) >= t["fast_table"]["thresholds"]["table_threshold"]
    cascade = np.where(by_table, P.max(axis=1), decisio)  # the table's calibrated P(yes) where it answers
    for name, conf in (("monitor", cascade), ("monitor_alone", decisio)):
        c = t[name]["confidence"]
        assert c["reads"] == monitor.BEFORE_NOUL_COMMIT, name
        assert c["tail_threshold"] == pytest.approx(float(np.quantile(conf, monitor.TAIL_QUANTILE)), abs=1e-12), name
    assert t["monitor_alone"]["confidence"]["tail_threshold"] < COMMIT_HI  # a tail the committed values do not have
    assert r.json()["status"]["monitor"]["confidence"]["reads"] == monitor.BEFORE_NOUL_COMMIT
    assert so.task_store.monitors["angry"].reads_before_noul_commit
    assert so.task_store.monitors_alone["angry"].reads_before_noul_commit


def test_y6_a_re_certification_of_a_table_task_makes_the_baseline_before_the_rule():
    client, so, _ = make(commit=True)
    assert register(client).status_code == 200
    r = client.post("/v1/tasks/angry/certify", json={"examples": noul_examples(400, seed=55)})
    assert r.status_code == 200, r.text
    t = so.task_store.snapshot("angry")
    assert t["fast_table"] and t["monitor"]["confidence"]["reads"] == monitor.BEFORE_NOUL_COMMIT
    assert t["monitor_alone"]["confidence"]["reads"] == monitor.BEFORE_NOUL_COMMIT


def test_y6_a_registration_whose_table_is_not_kept_makes_the_baseline_before_the_rule():
    client, so, _ = make(commit=True)
    t = register(client, fast_table={"min_saving": 0.9}).json()
    assert not t["fast_table_report"]["kept"]
    c = so.task_store.snapshot("angry")["monitor"]["confidence"]
    assert c["reads"] == monitor.BEFORE_NOUL_COMMIT and c["tail_threshold"] < COMMIT_HI


def test_y6_without_the_rule_the_baseline_has_no_marker():
    client, so, _ = make(commit=False)
    assert register(client).status_code == 200
    assert "reads" not in so.task_store.snapshot("angry")["monitor"]["confidence"]


# ---- Y7 -------------------------------------------------------------------------------------------------------------


def test_y7_table_and_deferred_answers_feed_the_monitor_the_confidence_before_the_rule():
    # on the old code: the deferred answer is observed at its committed 0.80, the table's at its committed value
    client, so, _ = make(commit=True)
    assert register(client).status_code == 200
    seen = watch(so.task_store.monitors["angry"])
    a = ask(client, SILENT)
    assert a["answered_by"] == "decisio" and a["noul"] == COMMIT_HI and a["certified_reason"] == "below_threshold"
    assert seen[-1] == pytest.approx(0.62, abs=1e-9)  # decisio's own, before the rule
    weak = state(False, 0.6, " ".join(YES_WORDS[:2] + FILLER[40:46]))
    b = ask(client, weak)
    if b["answered_by"] == "table":
        p = table_p(so, weak)
        assert seen[-1] == pytest.approx(max(p, 1 - p), abs=1e-9)  # the table's calibrated P(yes), before the rule
    c = ask(client, CLEAR_YES)
    p = table_p(so, CLEAR_YES)
    assert c["answered_by"] == "table" and seen[-1] == pytest.approx(max(p, 1 - p), abs=1e-9)


def test_y7_with_the_table_off_decisios_answers_feed_the_alone_monitor_before_the_rule():
    client, so, _ = make(commit=True)
    assert register(client).status_code == 200
    t = so.task_store.snapshot("angry")
    so.table_runtime(t).suspend("a test")
    seen = watch(so.task_store.monitors_alone["angry"])
    a = ask(client, CLEAR_YES)
    assert a["answered_by"] == "decisio" and a["noul"] == COMMIT_LO  # decisio says no at 0.6: reported 0.20
    assert seen[-1] == pytest.approx(0.6, abs=1e-9)


def test_y7_a_learning_tasks_decisio_rows_feed_the_monitor_before_the_rule():
    # a labelled yes/no task that learns, whose labelled table did not pay: its rows are learn:no_table, observed by
    # the alone monitor before the rule (on the old code at the committed 0.80)
    client, so, _ = make(commit=True)
    t = register(client, fast_table={"learn": True, "min_saving": 0.9}).json()
    assert not t["fast_table_report"]["kept"] and t["status"]["fast_table"]["mode"] == "both"
    m = so.task_store.monitors["angry"]  # no cascade: decisio's certificate is the task's
    assert m.reads_before_noul_commit
    seen = watch(m)
    a = ask(client, SILENT)
    assert a["answered_by"] == "decisio" and a["noul"] == COMMIT_HI
    assert seen[-1] == pytest.approx(0.62, abs=1e-9)
    assert so.learning.learners["angry"].counters["no_table"] == 1


# ---- Y8 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("commit", [False, True])
def test_y8_export_and_import_keep_the_calibration_and_the_bar_and_serve_the_same_answers(commit):
    client, so, _ = make(commit)
    t = register(client, fast_table={"min_saving": 0.2}).json()
    full = client.get("/v1/tasks?full=1").json()["tasks"]
    other, so2, _ = make(commit)
    assert other.post("/v1/tasks/import", json={"tasks": full}).status_code == 200
    t2 = so2.task_store.snapshot("angry")
    assert t2["fast_table"]["calibration"] == t["fast_table"]["calibration"] and t2["fast_table"]["min_saving"] == 0.2
    assert so2.table_runtime(t2).min_saving == 0.2
    for s in (CLEAR_YES, CLEAR_NO, SILENT):
        assert ask(client, s) == ask(other, s)


def test_y8_a_task_directory_keeps_the_table_across_a_restart(tmp_path):
    a, so_a, ba = make(commit=True, tasks_dir=tmp_path / "d")
    first = register(a, fast_table={"min_saving": 0.2}).json()
    want = ask(a, CLEAR_YES)
    ba.close()
    b, so_b, bb = make(commit=True, tasks_dir=tmp_path / "d")
    t = so_b.task_store.snapshot("angry")
    assert t["fast_table"]["min_saving"] == 0.2 and t["fast_table"]["calibration"] == first["fast_table"]["calibration"]
    assert t["monitor"]["confidence"]["reads"] == monitor.BEFORE_NOUL_COMMIT
    assert ask(b, CLEAR_YES) == want
    bb.close()


def test_y8_a_replica_on_a_shared_store_reads_the_table_its_calibration_and_its_bar():
    store = MemoryStore()
    a, so_a, _ = make(commit=True, store=store, name="a")
    assert register(a, fast_table={"min_saving": 0.2}).status_code == 200
    b, so_b, bb = make(commit=True, store=store, name="b")
    bb.sync_once()
    t = so_b.task_store.snapshot("angry")
    assert t["fast_table"]["min_saving"] == 0.2 and so_b.table_runtime(t).min_saving == 0.2
    st = b.get("/v1/tasks/angry").json()["status"]["fast_table"]
    assert st["state"] == "active" and st["min_saving"] == 0.2
    x, y = ask(a, CLEAR_YES), ask(b, CLEAR_YES)
    assert x == y and y["answered_by"] == "table" and y["certified"] is True


# ---- review: bad settings are refused, never a 500 or a task that fails every request ----------------------------


def test_y9_a_min_saving_too_large_for_a_float_is_a_422_not_a_500():
    client, _, _ = make()
    r = register(client, fast_table={"min_saving": 10**400})
    assert r.status_code == 422 and "min_saving" in r.json()["detail"]


def test_y9_a_yes_no_table_needs_two_fit_examples_of_each_answer():
    client, _, _ = make()
    ex = noul_examples(300, seed=3)
    yes = [e for e in ex if e["answer"]]
    no = [e for e in ex if not e["answer"]]
    one_no = yes[:180] + no[:1]  # one fit example of no: refused with a clear message, not a fold that cannot fit
    req = {
        "id": "angry",
        "examples": one_no,
        "fast_table": True,
        "certify": {"alpha": 0.1, "delta": 0.1, "examples": yes[180:] + no[1:]},
    }
    r = client.post("/v1/tasks", json=req)
    assert r.status_code == 422 and "two fit examples of each answer" in r.json()["detail"], r.text


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_saving", "fast"),
        ("min_saving", float("nan")),
        ("min_saving", 2.0),
        ("temperature", 0.0),
        ("temperature", -1.0),
    ],
)
def test_y9_an_imported_table_with_a_bad_bar_or_temperature_is_refused_at_import(field, value):
    client, _, _ = make()
    assert register(client).status_code == 200
    full = client.get("/v1/tasks", params={"full": 1}).json()["tasks"]
    t = next(x for x in full if x["id"] == "angry")
    if field == "min_saving":
        t["fast_table"]["min_saving"] = value
    else:
        t["fast_table"]["calibration"]["temperature"] = value
    other, _, _ = make()
    body = json.dumps({"tasks": full}, allow_nan=True)
    r = other.post("/v1/tasks/import", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422 and ("min_saving" in r.text or "temperature" in r.text), r.text
