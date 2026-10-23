# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A base's profile can carry a temperature per question type (yes/no, score) and choice temperatures by band of option
count, beside the global and choice temperatures it always had (decisio.families.Family; no base sets them yet).

P1  the shipped bases set neither, so every served temperature is what it was; the exported hub profile does not
    carry them, so a repository file exported before this still agrees
P2  a malformed profile is refused: choice among the types, a non-positive or non-finite value, bands that overlap or
    leave 2 to 255
P3  the profile is the last word: a flag, then `--temperatures-file`, then a tuned repository's defaults, then it; a
    type is filled only where nothing gave one, and the bands only where nothing gave a choice temperature or bands
P4  through `resolve_base` and a server: the type temperatures and the bands are the ones the server answers at, a
    question outside every band takes the base's choice temperature, and the start-up line source is recorded

  uv run pytest -q tests/unit/test_profile_temperatures.py
"""

import dataclasses
from types import SimpleNamespace

import pytest
from test_hub import args_for
from test_hub_tuned import build, write_repo

from decisio import hub
from decisio.families import BASES, FAMILIES, check_profile_temperatures
from decisio.serve import vllm_engine
from decisio.serve.systemone import SystemOne
from decisio.serve.tasks import TaskStore

FAM = BASES["gemma-4-31b"]
BANDS = {"2-10": 7.0, "11-26": 5.0, "27-255": 4.0}
WITH = dataclasses.replace(FAM, temperatures={"noul": 12.0, "score": 15.0}, choice_bands=BANDS)


# ---- P1 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fam", FAMILIES, ids=lambda f: f.key)
def test_p1_no_shipped_base_sets_them_and_every_profile_passes_the_check(fam):
    assert fam.temperatures == {} and fam.choice_bands == {}
    check_profile_temperatures(fam)


def test_p1_the_exported_profile_is_unchanged_by_them():
    assert hub.profile_of(WITH)["temperatures"] == hub.profile_of(FAM)["temperatures"]
    assert set(hub.profile_of(WITH)["temperatures"]) == {"global", "choice"}


def test_p1_a_bare_base_resolves_to_the_two_numbers_it_always_had():
    args = args_for("gemma-4-31b")
    vllm_engine.resolve_base(args)
    assert (args.temperature, args.temperature_choice) == (FAM.temperature, FAM.choice_temperature)
    assert getattr(args, "temperature_noul", None) is None and getattr(args, "temperature_score", None) is None
    assert args.choice_bands == {} and args.profile_temperatures is None


# ---- P2 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "over, words",
    [
        ({"temperatures": {"choice": 2.0}}, "only noul and score"),
        ({"temperatures": {"noul": 0.0}}, "positive"),
        ({"temperatures": {"score": -1.0}}, "positive"),
        ({"temperatures": {"noul": float("nan")}}, "positive"),
        ({"temperatures": {"noul": True}}, "positive"),
        ({"choice_bands": {"2-10": 7.0, "8-26": 5.0}}, "overlap"),
        ({"choice_bands": {"1-10": 7.0}}, "2"),
        ({"choice_bands": {"2-10": -3.0}}, "positive"),
    ],
)
def test_p2_a_malformed_profile_is_refused(over, words):
    with pytest.raises(ValueError, match=words):
        check_profile_temperatures(dataclasses.replace(FAM, **over))


# ---- P3 -------------------------------------------------------------------------------------------------------------


def args(**kw):
    base = dict(temperature_choice=None, temperature_noul=None, temperature_score=None, choice_bands={})
    return SimpleNamespace(**{**base, **kw})


def test_p3_nothing_given_the_profile_fills_the_types_and_the_bands():
    a = args()
    vllm_engine.apply_profile_temperatures(a, WITH)
    assert (a.temperature_noul, a.temperature_score) == (12.0, 15.0)
    assert a.choice_bands == BANDS
    assert a.profile_temperatures == {"noul": 12.0, "score": 15.0, "choice_bands": BANDS}


def test_p3_a_flag_or_a_file_beats_the_profile_per_type():
    a = args(temperature_noul=3.0)
    vllm_engine.apply_profile_temperatures(a, WITH)
    assert (a.temperature_noul, a.temperature_score) == (3.0, 15.0)
    assert a.profile_temperatures == {"score": 15.0, "choice_bands": BANDS}


def test_p3_a_choice_temperature_from_anywhere_keeps_the_bands_out():
    a = args(temperature_choice=2.0)  # --temperature-choice, or a file or a repository that gave a flat one
    vllm_engine.apply_profile_temperatures(a, WITH)
    assert a.choice_bands == {} and "choice_bands" not in a.profile_temperatures


def test_p3_bands_from_a_file_or_a_repository_are_not_replaced():
    own = {"2-10": 1.0, "11-255": 2.0}
    a = args(choice_bands=dict(own))
    vllm_engine.apply_profile_temperatures(a, WITH)
    assert a.choice_bands == own


def test_p3_a_base_with_neither_contributes_nothing():
    a = args()
    vllm_engine.apply_profile_temperatures(a, FAM)
    assert a.profile_temperatures is None and a.choice_bands == {} and a.temperature_noul is None


def test_p3_a_tuned_repository_keeps_what_it_was_fitted_with(monkeypatch, tmp_path):
    monkeypatch.setitem(BASES, "gemma-4-31b", WITH)
    full = write_repo(tmp_path, build(), name="full")  # noul, score, choice and bands of its own
    a = args_for(str(full))
    vllm_engine.resolve_base(a)
    assert (a.temperature_noul, a.temperature_score) == (4.1932, 6.2068)
    assert a.choice_bands["2-10"] == 2.4454 and a.profile_temperatures is None
    flat = write_repo(tmp_path, build(temperatures={"choice": 1.5}, choice_bands=None), name="flat")
    b = args_for(str(flat))
    vllm_engine.resolve_base(b)
    assert b.temperature_choice == 1.5 and b.choice_bands == {}  # its flat choice temperature; the bands stay out
    assert (b.temperature_noul, b.temperature_score) == (12.0, 15.0)  # what it did not say falls to the base's
    assert b.profile_temperatures == {"noul": 12.0, "score": 15.0}


# ---- P4 -------------------------------------------------------------------------------------------------------------


def test_p4_through_resolve_base_the_server_answers_at_them(monkeypatch):
    monkeypatch.setitem(BASES, "gemma-4-31b", WITH)
    a = args_for("gemma-4-31b")
    fam, _ = vllm_engine.resolve_base(a)
    so = SystemOne(
        object(),
        "decisio-test",
        task_store=TaskStore("fp"),
        temperature=a.temperature,
        temperatures={"choice": a.temperature_choice, "noul": a.temperature_noul, "score": a.temperature_score},
        choice_bands=a.choice_bands,
    )
    assert so.temperature_of("noul") == 12.0 and so.temperature_of("score") == 15.0
    assert [so.temperature_of("choice", options=k) for k in (2, 10, 11, 26, 27, 255)] == [7.0, 7.0, 5.0, 5.0, 4.0, 4.0]
    assert so.temperature_of("choice") == FAM.choice_temperature  # no option count: the base's flat one
    assert fam.choice_bands == BANDS
    health = so.temperatures, so.choice_bands
    assert health[1] == BANDS and health[0]["noul"] == 12.0


def test_p4_they_are_in_the_serving_fingerprint(monkeypatch):
    plain = SystemOne(object(), "decisio-test", task_store=TaskStore("fp"), temperature=FAM.temperature)
    banded = SystemOne(
        object(), "decisio-test", task_store=TaskStore("fp"), temperature=FAM.temperature, choice_bands=BANDS
    )
    typed = SystemOne(
        object(), "decisio-test", task_store=TaskStore("fp"), temperature=FAM.temperature, temperatures={"noul": 12.0}
    )
    from decisio.serve import stack

    ds = stack.differences(plain.stack_facts(), banded.stack_facts()) + stack.differences(
        plain.stack_facts(), typed.stack_facts()
    )
    assert ds, "a certificate made at other temperatures must be suspended"
