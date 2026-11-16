# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The header layout's note and counter (docs/running.md, "The header layout"; docs/api.md).

N1  a text request read under the header layout carries x-decisio-note; its answers are the same as without it
N2  a request read as plain carries no note and is not counted: the plain layout, "layout": "plain", a header of 0 lines
N3  /health systemone.header_layout counts the reads and the reads of a state an earlier request read under another
    header; the same header again (the same questions, or one request with all of them) is not a re-read
N5  a request that declares its header (`header_questions`): its own note, and other questions under one list are not
    re-reads
N4  the counter remembers a bounded number of states, the least recently seen out first; it never changes an answer

  uv run pytest -q tests/unit/test_header_note.py
"""

from fastapi.testclient import TestClient
from test_certified_answers import CRIT
from test_header_limit import HeaderEngine, post_many
from test_over_limit import plain_rows  # noqa: F401  (the fixture applies to this module)

from decisio.serve import systemone as so_mod
from decisio.serve.systemone import HEADER_DECLARED_NOTE, HEADER_NOTE, SystemOne
from decisio.serve.tasks import TaskStore
from decisio.serve.vllm_engine import make_app

NOTE = "x-decisio-note"
STATE = "1:0.9:x " + " ".join(f"w{i}" for i in range(60))


def make(layout="header", limit=10_000):
    eng = HeaderEngine(limit=limit, layout=layout)
    so = SystemOne(eng, "decisio-test", task_store=TaskStore("fp"))
    return TestClient(make_app(eng, so)), so, eng


def ask(client, *questions, state=STATE, **extra):
    qs = {f"q{i}": {"type": "choice", "instructions": text, "criteria": CRIT} for i, text in enumerate(questions)}
    return client.post("/v1/systemone", json={"state": {"message": state}, "model": "m", "questions": qs, **extra})


def facts(client):
    return client.get("/health").json()["systemone"]["header_layout"]


# ---- N1 -------------------------------------------------------------------------------------------------------------


def test_n1_a_header_layout_request_carries_the_note():
    client, so, _ = make()
    r = ask(client, "Which queue?")
    assert r.status_code == 200, r.text
    assert r.headers[NOTE] == HEADER_NOTE
    assert "reads the whole state again" in HEADER_NOTE and "in one request" in HEADER_NOTE
    assert '"layout": "plain"' in HEADER_NOTE
    assert "note" not in r.json() and "_timing" not in r.json()


def test_n1_the_note_changes_no_answer():
    client, _, _ = make()
    first = ask(client, "Which queue?", "Which team?")
    ask(client, "Which owner?")  # a read that the counter sees in between
    again = ask(client, "Which queue?", "Which team?")
    assert first.json()["answers"] == again.json()["answers"]


# ---- N2 -------------------------------------------------------------------------------------------------------------


def test_n2_the_plain_layout_carries_no_note_and_is_not_counted():
    client, _, _ = make(layout="plain")
    r = ask(client, "Which queue?")
    assert r.status_code == 200 and NOTE not in r.headers
    ask(client, "Which team?")
    assert facts(client)["reads"] == 0 and facts(client)["state_rereads"] == 0


def test_n2_a_request_that_asks_for_plain_carries_no_note():
    client, _, _ = make()
    r = ask(client, "Which queue?", layout="plain")
    assert r.status_code == 200, r.text
    assert NOTE not in r.headers and facts(client)["reads"] == 0
    assert NOTE in ask(client, "Which queue?").headers


def test_n2_a_header_limited_to_no_line_is_plain_and_carries_no_note():
    plain = HeaderEngine(limit=10_000, layout="plain")
    from test_header_limit import longest, qs_many

    qs = [{"instructions": q["instructions"], "options": q["options"], "kind": "choice"} for q in qs_many(3, words=40)]
    message = "1:0.9:x " + " ".join(f"w{i}" for i in range(150))
    _, so, eng = make()
    eng.limit = longest(plain, {"message": message}, qs) + 3
    client = TestClient(make_app(eng, so))
    r = post_many(client, n=3, words=40)
    assert r.status_code == 200 and r.json()["header_limited"]["listed"] == 0
    assert NOTE not in r.headers and facts(client)["reads"] == 0


# ---- N3 -------------------------------------------------------------------------------------------------------------


def test_n3_a_second_request_with_another_question_is_a_reread():
    client, _, _ = make()
    ask(client, "Which queue?")
    assert facts(client) == {"reads": 1, "state_rereads": 0, "states_remembered": 1, "window": 4096}
    ask(client, "Is it urgent?")
    assert facts(client)["reads"] == 2 and facts(client)["state_rereads"] == 1
    ask(client, "Which queue?")  # the header changed again
    assert facts(client)["state_rereads"] == 2


def test_n3_the_same_header_again_is_not_a_reread():
    client, _, _ = make()
    ask(client, "Which queue?")
    ask(client, "Which queue?")
    ask(client, "Which queue?", "Is it urgent?")  # one request with all the questions: one more header
    ask(client, "Which queue?", "Is it urgent?")
    assert facts(client) == {"reads": 4, "state_rereads": 1, "states_remembered": 1, "window": 4096}


def test_n3_other_states_are_not_rereads():
    client, _, _ = make()
    ask(client, "Which queue?", state=STATE)
    ask(client, "Is it urgent?", state=STATE + " more")
    assert facts(client) == {"reads": 2, "state_rereads": 0, "states_remembered": 2, "window": 4096}


# ---- N4 -------------------------------------------------------------------------------------------------------------


def test_n4_the_window_is_bounded_least_recently_seen_out_first(monkeypatch):
    monkeypatch.setattr(so_mod, "HEADER_SEEN_MAX", 2)
    _, so, _ = make()
    states = [{"message": f"1:0.9:s{i}"} for i in range(3)]
    for s in states:
        so.note_header_read(s, ["Which queue?"])
    assert so.header_layout_facts()["states_remembered"] == 2
    so.note_header_read(states[0], ["Is it urgent?"])  # forgotten: not counted as a re-read
    assert so.header_layout_facts()["state_rereads"] == 0
    so.note_header_read(states[2], ["Is it urgent?"])  # still remembered
    assert so.header_layout_facts()["state_rereads"] == 1


# ---- N5 -------------------------------------------------------------------------------------------------------------


def test_n5_a_declared_header_has_its_own_note_and_no_rereads():
    client, _, _ = make()
    lines = ["Which queue?", "Is it urgent?", "Which team?"]
    r = ask(client, "Which queue?", header_questions=lines)
    assert r.status_code == 200, r.text
    assert r.headers[NOTE] == HEADER_DECLARED_NOTE and "not measured yet" in HEADER_DECLARED_NOTE
    assert '"header_questions"' in HEADER_NOTE
    ask(client, "Is it urgent?", header_questions=lines)
    ask(client, "Which team?", header_questions=lines)
    assert facts(client)["reads"] == 3 and facts(client)["state_rereads"] == 0
    ask(client, "Which team?")  # without the list: another header
    assert facts(client)["state_rereads"] == 1


def test_n5_plain_with_a_declared_list_has_no_note():
    client, _, _ = make()
    r = ask(client, "Which queue?", layout="plain", header_questions=["Which queue?"])
    assert r.status_code == 200 and NOTE not in r.headers and facts(client)["reads"] == 0
