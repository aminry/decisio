# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""What decisio writes carries decisio's name only, and what earlier releases wrote still loads (`decisio.names`).

N1  nothing the server emits carries the project's former name: the task records it writes (format ids of the task, its
    calibration, head and certificate), a response and its headers, `/health` and `/v1/models`, and the served name
N2  a request's `model` is not checked against the served name: the Qwen base's earlier served name is accepted like any
    other and the response echoes the name the request gave; without one it names the served model
N3  a task exported by this release with its format ids rewritten to the earlier spelling loads and applies, and stays
    as written (its fit digest covers those records); a task fitted under the earlier default served name applies on a
    server with the current one

  uv run pytest -q tests/unit/test_names_emitted.py
"""

import json

from test_certified_answers import ask, examples, make, post_task

from decisio import names

FORMER = "rl" + "cd"  # the project's former name; spelled in two parts so that this file does not carry it as one word
OLD_SERVED = f"{FORMER}-qwen3.6-35b-a3b-letters"


def registered(task_id="routing"):
    client, so = make()
    r = post_task(client, {"id": task_id, "examples": examples(400, 0.96), "certify": {}})
    assert r.status_code == 200, r.text
    return client, so, r.json()


def mentions_former_name(obj) -> bool:
    return FORMER in json.dumps(obj, default=str).lower()


def test_n1_nothing_the_server_writes_carries_the_former_name():
    client, so, task = registered()
    exported = client.get("/v1/tasks?full=1").json()
    assert not mentions_former_name(exported) and not mentions_former_name(task)
    t = exported["tasks"][0]
    assert t["format"] == "decisio-task/1" and t["certificate"]["format"] == "decisio-certificate/1"
    assert t["calibration"]["format"] == "decisio-task-prior/1"
    assert not mentions_former_name(t["fingerprint"])
    body = {
        "state": {"message": "1:0.9:x"},
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": "Which queue?",
                "criteria": {"option_0": None, "option_1": None, "option_2": None},
            }
        },
    }
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200
    assert not mentions_former_name(r.json()) and not mentions_former_name(dict(r.headers))
    assert not mentions_former_name(so.stack_facts())
    assert names.SERVED_NAME == "decisio-qwen3.6-35b-a3b-letters" and not mentions_former_name(names.SERVED_NAME)
    for kind in ("task", "task-prior", "intent-head", "abstention", "certificate"):
        assert names.record_format(kind) == f"decisio-{kind}/1"


def test_n2_the_earlier_served_name_in_the_model_field_is_accepted():
    client, so, _ = registered()
    body = {
        "state": {"message": "1:0.9:x"},
        "model": OLD_SERVED,
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": "Which queue?",
                "criteria": {"option_0": None, "option_1": None, "option_2": None},
            }
        },
    }
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["model"] == OLD_SERVED  # the response echoes the name the request gave
    assert r.json()["answers"]["q1"]["choice"] in ("option_0", "option_1", "option_2")
    del body["model"]
    assert client.post("/v1/systemone", json=body).json()["model"] == so.name


def rewrite_formats(obj, to):
    """The same record with every `decisio-<kind>/1` format id spelled the earlier way."""
    if isinstance(obj, dict):
        return {
            k: (v.replace("decisio-", f"{to}-", 1) if k == "format" and isinstance(v, str) else rewrite_formats(v, to))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [rewrite_formats(v, to) for v in obj]
    return obj


def test_n3_records_in_the_earlier_spelling_still_load_and_apply():
    client, so, _ = registered()
    exported = client.get("/v1/tasks?full=1").json()
    old = rewrite_formats(exported, FORMER)
    assert (
        old["tasks"][0]["format"] == f"{FORMER}-task/1"
        and old["tasks"][0]["certificate"]["format"] == f"{FORMER}-certificate/1"
    )
    fresh, _ = make()
    r = fresh.post("/v1/tasks/import", json=old)
    assert r.status_code == 200, r.text
    assert ask(fresh, 1, 0.99)["certified"] is not None  # the imported task applies to a matching question
    # an imported record keeps the spelling it was written with: the fit digest covers its calibration and head records,
    # and the certificate names that digest, so a loaded task is not rewritten
    assert fresh.get("/v1/tasks?full=1").json()["tasks"][0]["format"] == f"{FORMER}-task/1"
    # a task fitted under the earlier default served name applies on a server with the current one
    kw = {"model": "m", "pad_to": 1056, "pad_where": "front", "hide_index_keys": True, "desnake_labels": True}
    assert names.same_fingerprint(
        names.task_fingerprint(served_name=OLD_SERVED, **kw),
        names.task_fingerprint(served_name=names.SERVED_NAME, **kw),
    )
    assert not names.same_fingerprint(
        names.task_fingerprint(served_name="other", **kw), names.task_fingerprint(served_name=names.SERVED_NAME, **kw)
    )
