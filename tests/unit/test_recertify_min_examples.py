# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""`POST /v1/tasks/{id}/certify` and `min_examples`.

Site's re-audit (batch 10): the old certificate's `min_examples` (59, what the default 5% and 10% need) was carried
into a re-certification as if the caller had asked for it, so re-certifying at a looser target (10% and 10% need 29)
was refused below 59 examples. It is carried over now only if the original call set it above what its alpha and delta
need; otherwise the new alpha and delta derive it.

R1  re-certifying at alpha 0.10 with 48 held-out examples succeeds on a task certified at the defaults, and the new
    certificate's minimum is 29
R2  a `min_examples` the original call set explicitly is kept by a re-certification (48 examples are refused), and a
    new explicit value in the call wins over it
R3  the same alpha and delta, no setting named: still 59 (an unchanged target is not loosened); a re-certification
    that tightens the target derives the larger minimum

  uv run pytest -q tests/unit/test_recertify_min_examples.py
"""

from test_certified_answers import examples, make, post_task

from decisio.readout import certify


def registered(certify_block=None, n=400):
    client, so = make()
    r = post_task(client, {"id": "routing", "examples": examples(n, 0.96), "certify": certify_block or {}})
    assert r.status_code == 200, r.text
    return client, so, r.json()


def recertify(client, n, **settings):
    return client.post(
        "/v1/tasks/routing/certify", json={"examples": examples(n, 0.96, seed=9, start=10_000), **settings}
    )


def test_r1_a_looser_target_needs_fewer_examples_than_the_defaults_59():
    assert (certify.minimum_examples(0.1, 0.1), certify.minimum_examples(0.05, 0.1)) == (29, 59)
    client, _, task = registered()
    assert task["certificate"]["min_examples"] == 59
    assert recertify(client, 48).status_code == 422  # the same target still needs 59 (R3)
    r = recertify(client, 48, alpha=0.1)
    assert r.status_code == 200, r.text
    c = r.json()["certificate"]
    assert c["alpha"] == 0.1 and c["min_examples"] == 29 and c["n"] == 48


def test_r2_an_explicit_min_examples_is_kept_and_a_new_one_wins():
    client, so, task = registered({"min_examples": 80})
    assert task["certificate"]["min_examples"] == 80
    r = recertify(client, 48, alpha=0.1)  # 29 would do, but the original call asked for 80
    assert r.status_code == 422 and "80" in r.text
    assert recertify(client, 90, alpha=0.1).status_code == 200
    stored = next(t for t in so.task_store.by_key.values() if t["id"] == "routing")
    assert stored["certificate"]["min_examples"] == 80  # the failed calls left it, and the good one kept it
    r = recertify(client, 48, alpha=0.1, min_examples=30)  # a new explicit value replaces the old one
    assert r.status_code == 200, r.text
    assert r.json()["certificate"]["min_examples"] == 30


def test_r3_the_derived_minimum_follows_the_new_target():
    client, _, _ = registered()
    r = recertify(client, 100)  # unchanged target: 59, as before
    assert r.status_code == 200 and r.json()["certificate"]["min_examples"] == 59
    r = recertify(client, 100, alpha=0.1)  # loosen: 29
    assert r.status_code == 200 and r.json()["certificate"]["min_examples"] == 29
    need = certify.minimum_examples(0.02, 0.1)
    assert need > 59
    r = recertify(client, need - 1, alpha=0.02)  # tighten: derived, not carried
    assert r.status_code == 422 and str(need) in r.text
    assert recertify(client, need + 5, alpha=0.02).json()["certificate"]["min_examples"] == need


def test_r3_a_stored_value_at_the_floor_is_not_an_explicit_ask():
    """A certificate made before this fix (or by a caller who named exactly the floor) stores min_examples as the floor
    of its own alpha and delta; it is derived anew, not carried."""
    client, so, task = registered({"min_examples": 59})
    assert task["certificate"]["min_examples"] == 59
    assert recertify(client, 48, alpha=0.1).status_code == 200
