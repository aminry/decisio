# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Every base's intent head reads its hidden state in the question's own request (decisio.vllm_plugin.tap).

D1  every base keeps its hidden-readout class (the Hub profiles are unchanged), and that class carries the tap
    (`tap.TapReadout` first in its bases); the tap is on with the vLLM server, an in-process engine and one
    tensor-parallel rank; its engine arguments add the tap's logits processor beside the caller's and its layer set
    (the final state alone) in additional_config, keeping a caller's own layer set
D2  the reader: the tap with an in-process engine on every base, the reserved columns with a separate engine process
    or more than one tensor-parallel rank
D3  a head fitted on the tap's reads records it; on a tap server a kept head fitted before (no such record) is stale:
    the task answers as without it (its calibration too), names the reason and the register command in the listing
    and the store report, and is not certified again; on a reserved-columns server nothing is stale

  uv run pytest -q tests/unit/test_tap_default.py
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from decisio import families
from decisio.families import GEMMA4, GEMMA4_31B, GEMMA4_E4B, QWEN, QWEN35_4B

# every base decisio serves, by its profile's hidden-readout class (the names the Hub profiles carry)
BASES = {
    "GEMMA4": "DecisioGemma4UnifiedHiddenReadout",
    "GEMMA4_FP8": "DecisioGemma4UnifiedHiddenReadout",
    "GEMMA4_31B": "DecisioGemma4HiddenReadout",
    "GEMMA4_31B_BF16": "DecisioGemma4HiddenReadout",
    "GEMMA4_E4B": "DecisioGemma4HiddenReadout",
    "QWEN": "DecisioQwen3_5MoeHiddenReadout",
    "QWEN35_4B": "DecisioQwen3_5HiddenReadout",
}


def test_d1_every_base_serves_its_class_with_the_tap():
    import decisio.vllm_plugin as plugin
    from decisio.serve import vllm_engine as sv
    from decisio.vllm_plugin import tap

    sys.path.insert(0, str(Path(__file__).parent))
    from stub_vllm import stub_vllm

    every = {n for n, v in vars(families).items() if isinstance(v, families.Family)}
    assert every == set(BASES), "a base was added: give its hidden-readout class the tap and list it here"
    with stub_vllm("0.32.0") as registry:
        assert plugin.register()
        for arch in set(BASES.values()):
            assert registry.resolve(arch).__mro__[1].__qualname__ == "TapReadout", arch
    for name, arch in BASES.items():
        fam = getattr(families, name)
        assert fam.classes["hidden-readout"] == arch, name
        on = types.SimpleNamespace(family=fam, model_class="hidden-readout", backend="vllm", engine_process="in")
        assert sv.tap_enabled(on), name
        assert not sv.tap_enabled(types.SimpleNamespace(**{**vars(on), "engine_process": "separate"}))
        assert not sv.tap_enabled(types.SimpleNamespace(**{**vars(on), "backend": "mlx"}))
        assert not sv.tap_enabled(types.SimpleNamespace(**{**vars(on), "model_class": "text-only"}))
        tp2 = types.SimpleNamespace(**{**vars(on), "engine": '{"tensor_parallel_size": 2}'})
        assert not sv.tap_enabled(tp2), name  # the sharded sampler reorders the rows: the reserved columns
        assert sv.tap_enabled(types.SimpleNamespace(**{**vars(on), "engine": '{"tensor_parallel_size": 1}'}))
    kw = sv.tap_engine_kwargs({"compilation_config": {"x": 1}})
    assert kw["logits_processors"] == [tap.PROCESSOR] and kw["additional_config"] == {tap.CONFIG_KEY: []}
    assert kw["compilation_config"] == {"x": 1}
    mine = sv.tap_engine_kwargs({"logits_processors": ["my.mod:Proc"], "additional_config": {tap.CONFIG_KEY: [12]}})
    assert mine["logits_processors"] == ["my.mod:Proc", tap.PROCESSOR] and mine["additional_config"][
        tap.CONFIG_KEY
    ] == [12]
    assert sv.tap_engine_kwargs(mine) == mine  # applied twice: the same


def test_d2_the_reader_follows_the_engine_process(monkeypatch):
    from decisio.serve import hidden_engine as he
    from decisio.serve import vllm_engine as sv

    built = []
    monkeypatch.setattr(he, "TapHidden", lambda engine, layers=None: built.append("tap") or "tap")
    monkeypatch.setattr(he, "SingleEngineHidden", lambda engine, model, revision=None: built.append("reserved") or "r")
    args = lambda fam, process, engine="{}": types.SimpleNamespace(  # noqa: E731
        family=fam,
        model_class="hidden-readout",
        engine_process=process,
        model="m",
        revision=None,
        backend="vllm",
        engine=engine,
    )
    for fam in (GEMMA4, GEMMA4_31B, GEMMA4_E4B, QWEN, QWEN35_4B):
        assert sv.single_engine_reader(object(), args(fam, "in")) == "tap"
    assert sv.single_engine_reader(object(), args(GEMMA4, "separate")) == "r"
    assert sv.single_engine_reader(object(), args(QWEN, "in", '{"tensor_parallel_size": 2}')) == "r"
    assert built == ["tap"] * 5 + ["reserved"] * 2


def _so(reader):
    from decisio.serve.systemone import SystemOne

    so = SystemOne(None, "m", temperature=1.5)
    so.hidden_engine = types.SimpleNamespace(reader=reader, readout=lambda *a, **k: pytest.fail("the head was read"))
    return so


def _task(head_reader=None, certified=False):
    head = {"applied": True, **({"reader": head_reader} if head_reader else {})}
    return {
        "id": "intents",
        "type": "choice",
        "options": ["a", "b", "c"],
        "head": head,
        "calibration": {"applied": True, "K": 3, "bias": [0.3, 0.0, -0.3], "temperature": 1.5},
        **({"certificate": {"id": "c1", "alpha": 0.05, "delta": 0.1}} if certified else {}),
    }


def test_d3_a_head_fitted_before_is_stale_on_a_tap_server():
    from decisio.serve.temperature import apply_temperature

    tap_server, reserved_server = _so("tap"), _so(None)
    old, new = _task(), _task("tap")
    assert tap_server.head_stale(old) and not tap_server.head_stale(new)
    assert not reserved_server.head_stale(old) and reserved_server.head_note(old) is None
    note = tap_server.head_note(old)
    assert note["stale"] and "registered again" in note["why"] and "/v1/tasks" in note["register_again"]
    # answered as without the task: neither the head nor the task's calibration is applied
    tap_server.plain_readout_task = lambda q: old
    p = np.array([0.6, 0.3, 0.1])
    q = types.SimpleNamespace(type="choice")
    assert np.array_equal(tap_server.with_readout_task(None, q, None, p), apply_temperature(p, 1.5))
    # not certified again on the stale fit
    tap_server.task_precheck = lambda kind, task_id: None
    with pytest.raises(ValueError, match="register the task again"):
        tap_server._recertify(_task(certified=True), [], None)


def test_d3_the_store_report_names_the_stale_heads():
    from decisio.serve.tasks import TaskStore

    so = _so("tap")
    so.task_store = types.SimpleNamespace(
        by_key={"k1": _task(), "k2": {**_task("tap"), "id": "fresh"}}, applies=lambda t: True
    )
    so.task_backend = None
    rep = so.task_store_report()["readout"]
    assert rep["head_refit"]["tasks"] == ["intents"] and "/v1/tasks" in rep["head_refit"]["register_again"]
    assert TaskStore  # imported for the record shape the listing reads


def test_d3_registration_records_the_reader():
    from decisio.serve.tasks import TaskStore

    sys.path.insert(0, str(Path(__file__).parent))
    from test_tasks import fake_examples

    ex, P, R = fake_examples(6)
    store = TaskStore("fp")
    tapped, _ = store.register("t", ex, lambda pairs: P, lambda pairs: R, head_reader="tap")
    assert tapped["head"]["applied"] and tapped["head"]["reader"] == "tap"
    plain, _ = TaskStore("fp").register("t", ex, lambda pairs: P, lambda pairs: R)
    assert "reader" not in plain["head"]
