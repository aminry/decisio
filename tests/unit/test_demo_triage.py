# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The support-ticket triage demo (examples/demos/triage and its tools), on data that needs no server.

TR1  the ticket generator is a function of its seed, and the committed splits are what it writes
TR2  the splits have the stated counts per intent, no ticket text twice, and the training phrasings are their own
TR3  a stream trajectory round-trips through trajectory.read
TR4  rendering the same trajectories twice gives the same frames, one feed and side by side, at the stated size
TR5  accuracy, the paired difference and macro-F1 on a fixed toy input; a failed call counts as wrong
TR6  a limited held-out run still covers every intent evenly; the registration summary says what was kept
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "examples" / "demos" / "tools"
TRIAGE = ROOT / "examples" / "demos" / "triage"
sys.path.insert(0, str(TOOLS))

import measure_triage  # noqa: E402
import render_common  # noqa: E402
import render_triage  # noqa: E402
import trajectory  # noqa: E402

# loaded by path under its own name: examples/tasks has a make_tickets.py too
_spec = importlib.util.spec_from_file_location("triage_make_tickets", TRIAGE / "make_tickets.py")
make_tickets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_tickets)

TAXONOMY = json.loads((TRIAGE / "taxonomy.json").read_text())
KEYS = [i["key"] for i in TAXONOMY["intents"]]


def test_tr1_generator_is_deterministic_and_the_committed_splits_are_its_output():
    a, b = make_tickets.generate(7), make_tickets.generate(7)
    assert a == b
    assert make_tickets.generate(8)["train"] != a["train"]
    committed = make_tickets.generate(make_tickets.SEED)
    for split, tickets in committed.items():
        on_disk = json.loads((TRIAGE / "data" / f"{split}.json").read_text())
        assert on_disk["tickets"] == tickets and on_disk["seed"] == make_tickets.SEED


def test_tr2_splits_have_the_stated_counts_and_are_disjoint():
    assert len(KEYS) == 20 and len(set(KEYS)) == 20
    splits = {s: make_tickets.load_split(s) for s in ("train", "heldout", "stream")}
    expected = {"train": 10, "heldout": 20, "stream": 4}
    for split, tickets in splits.items():
        per = {k: sum(t["intent"] == k for t in tickets) for k in KEYS}
        assert set(per.values()) == {expected[split]}, split
        assert len({t["id"] for t in tickets}) == len(tickets)
    assert [len(splits[s]) for s in ("train", "heldout", "stream")] == [200, 400, 80]
    texts = [make_tickets.normalised(t["text"]) for s in splits.values() for t in s]
    assert len(texts) == len(set(texts))
    assert all("/train" in t["core"] for t in splits["train"])
    assert all("/eval" in t["core"] for t in splits["heldout"] + splits["stream"])
    assert all(t["intent"] in KEYS and t["text"].strip() for s in splits.values() for t in s)
    for t in (t for s in splits.values() for t in s):
        text = next(
            (t["text"][len(g) + 1 :] for g in make_tickets.GREETINGS if t["text"].startswith(g + " ")), t["text"]
        )
        assert 1 <= make_tickets.n_sentences(text) <= 4, t["text"]


def fake_results(tickets, t0=1000.0, step=1.5, latency=0.4):
    """What measure_triage.route returns, for tickets answered in turn: the first ticket right, the rest wrong."""
    out = []
    for i, t in enumerate(tickets):
        chosen = t["intent"] if i % 2 == 0 else KEYS[(KEYS.index(t["intent"]) + 1) % len(KEYS)]
        probs = {k: (0.6 if k == chosen else 0.4 / (len(KEYS) - 1)) for k in KEYS}
        req = measure_triage.request_for(t["text"], measure_triage.question_for(TAXONOMY))
        answer = {"route": {"type": "choice", "choice": chosen, "confidence": 0.5, "probabilities": probs}}
        out.append(
            {
                "request": req,
                "t_sent": t0 + i * step,
                "t_wall": t0 + i * step + latency,
                "latency_ms": latency * 1000,
                "answer": answer,
                "chosen": chosen,
                "tasks": [],
                "server_ms": None,
            }
        )
    return out


def write_pair(tmp_path, n=6):
    tickets = make_tickets.load_split("stream")[:n]
    player = trajectory.player_fields({"label": "a test player", "card": "a card"})
    paths = []
    for arm in ("plain", "taught"):
        run = {"arm": arm, "split": "stream", "pace_ms": 1500, "tickets": n}
        if arm == "taught":
            run["registration"] = {
                "seconds": 12.0,
                "n_examples": 200,
                "head": {"applied": True},
                "calibration": {"applied": False},
            }
        path = tmp_path / f"{arm}.jsonl.gz"
        measure_triage.stream_trajectory(path, tickets, fake_results(tickets), player, run, TAXONOMY)
        paths.append(path)
    return tickets, paths


def test_tr3_trajectory_round_trips(tmp_path):
    tickets, (plain, _) = write_pair(tmp_path)
    head, ticks, end = trajectory.read(plain)
    assert head["format"] == trajectory.FORMAT and head["demo"] == "triage"
    assert head["run"]["arm"] == "plain" and head["run"]["pace_ms"] == 1500
    assert [i["key"] for i in head["intents"]] == KEYS and head["player"]["label"] == "a test player"
    assert len(ticks) == len(tickets) and end == {"tickets": 6, "right": 3, "accuracy": 0.5}
    for tick, t in zip(ticks, tickets):
        assert tick["state"] == {"ticket_id": t["id"], "text": t["text"], "intent": t["intent"]}
        assert tick["decision"]["request"]["state"] == t["text"] and tick["t_wall"] > tick["t_sent"]
    assert len(trajectory.decisions(ticks)) == len(tickets)
    again = tmp_path / "again.jsonl.gz"
    trajectory.write(again, head, ticks, end)
    assert trajectory.read(again) == (head, ticks, end)


def test_tr4_rendering_is_deterministic_and_the_right_size(tmp_path):
    _, (plain, taught) = write_pair(tmp_path)
    summary = {
        "heldout": {
            "n": 400,
            "accuracy_plain": {"est": 0.4, "lo": 0.35, "hi": 0.45},
            "accuracy_taught": {"est": 0.8, "lo": 0.76, "hi": 0.84},
            "difference": {"est": 0.4, "lo": 0.35, "hi": 0.45},
        }
    }
    times = [0.0, 0.7, 1.2, 3.0, 9.5]  # before the first ticket, routing, answered, later, the hold at the end
    for feeds, size in (([plain], (1280, 720 + render_common.STRIP_H)), ([plain, taught], (1280, 720 + 44))):
        digests = []
        for _ in range(2):
            r = render_triage.Renderer([render_triage.Feed(p) for p in feeds], summary)
            frames = [r.frame(t) for t in times]
            assert all(f.size == size for f in frames)
            digests.append([render_common.digest(f) for f in frames])
        assert digests[0] == digests[1]
        assert len(set(digests[0])) == len(times)  # each moment draws something different


def test_tr5_accuracy_paired_difference_and_macro_f1_on_a_toy_input():
    gold = ["a", "a", "b", "b"]
    plain = ["a", "b", "a", "b"]
    taught = ["a", "a", "b", None]  # the last call failed
    out = measure_triage.compare_arms(gold, plain, taught, ["a", "b"])
    assert out["n"] == 4
    assert out["accuracy_plain"]["est"] == pytest.approx(0.5)
    assert out["accuracy_taught"]["est"] == pytest.approx(0.75)
    assert out["difference"]["est"] == pytest.approx(0.25)
    for k in ("accuracy_plain", "accuracy_taught", "difference"):
        b = out[k]
        assert b["lo"] <= b["est"] <= b["hi"]
    assert -1 <= out["difference"]["lo"] and out["difference"]["hi"] <= 1
    assert out["fixed"] == 2 and out["broken"] == 1
    assert out["macro_f1_plain"] == pytest.approx(0.5)
    assert out["macro_f1_taught"] == pytest.approx((1 + 2 / 3) / 2)  # a: F1 1; b: one of two found, F1 2/3
    assert out["per_intent"] == {"a": {"n": 2, "plain": 0.5, "taught": 1.0}, "b": {"n": 2, "plain": 0.5, "taught": 0.5}}
    same = measure_triage.compare_arms(gold, gold, gold, ["a", "b"])
    assert same["difference"] == {"est": 0.0, "lo": 0.0, "hi": 0.0}


def test_tr6_limited_heldout_is_stratified_and_registration_summary_says_what_was_kept():
    held = make_tickets.load_split("heldout")
    few = measure_triage.limit_stratified(held, 60)
    assert len(few) == 60 and {k: sum(t["intent"] == k for t in few) for k in KEYS} == {k: 3 for k in KEYS}
    assert measure_triage.limit_stratified(held, None) == held
    task = {
        "n_examples": 200,
        "per_option_min": 10,
        "calibration": {"applied": False, "reason": "the head applies", "cv_acc_plain": 0.3, "cv_acc_fitted": 0.4},
        "head": {"applied": True, "reason": "cross-validated gain", "lambda": 0.001, "cv_logloss": {"none": 2.0}},
    }
    reg = measure_triage.registration_summary(task, 42.123, 200)
    assert reg["seconds"] == 42.12 and reg["head"]["applied"] and reg["head"]["lambda"] == 0.001
    assert not reg["calibration"]["applied"] and reg["calibration"]["cv_acc_fitted"] == 0.4
    assert measure_triage.registration_text(reg) == "200 examples registered in 42 s; kept: head"
    assert "kept: intent head" in render_triage.registration_line({"registration": reg})
