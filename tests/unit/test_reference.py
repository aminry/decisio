# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The plain-transformers reference (`examples/reference/decisio_reference.py`) against the server's own path.

The reference is one file that needs torch and transformers only and imports nothing from decisio, so that it can
check a model repository independently of the code it is meant to check. These tests are its development gate, on
the CPU stand-in (a 0.6B Qwen3 base, `DECISIO_STAND_IN_MODEL`): the same model object answers through `SystemOne` on
`HFLettersEngine` (the served code up to the forward pass) and through the reference, from a repository directory
holding a `decision_config.json`. The conformance check against the real server on each base is a separate run.

R1  the repository: the file is read from the directory, a wrong schema or an unknown value is refused, and a softcap
    the model does not apply is refused;
R2  the prompt: the token ids and the label ids of every case are exactly the rows `HFLettersEngine` scores (padding
    included), under each of the three prompt formats the bases use or could use;
R3  the answer: every field of the wire answer equals the server's, floats within TOLERANCE (fixed before the first
    run: both sides run one float32 forward pass on the same ids, so only the last digits can differ);
R4  the temperatures: a choice question is served at the choice temperature, yes/no and score at the global one;
R5  a tie is broken by the smallest key, and the answer never depends on the order of the keys on the wire;
R6  the command line: one question per call, printed as JSON.

  uv run pytest -q tests/unit/test_reference.py
"""

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODEL = os.environ.get("DECISIO_STAND_IN_MODEL", "Qwen/Qwen3-0.6B-Base")
BLOCK = 64  # the stand-in's block size (the Qwen base's is 1,056)
TOLERANCE = 1e-9


def load_reference():
    spec = importlib.util.spec_from_file_location("decisio_reference", ROOT / "examples/reference/decisio_reference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # the file's dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def hub_config(name):
    return json.loads((ROOT / "hub" / f"decision_config.{name}.json").read_text())


def gemma_like():
    """The Gemma 12B base's profile on the stand-in: its prompt format, no padding, and no softcap (the stand-in has
    none to apply, and the reference refuses a softcap the model does not report)."""
    cfg = hub_config("gemma-4-12b")
    cfg["readout"]["softcap"] = None
    return cfg


def template_single():
    """The Qwen profile read from the template's own answer slot, one bare token per label (label form "bare")."""
    cfg = hub_config("qwen3.6-35b-a3b")
    cfg["prompt"]["format"]["slot"] = "template"
    cfg["prompt"]["label_form"] = "bare"
    return cfg


def compact_tail():
    """The Qwen profile with the compact tail, the served layout before 2026-10-03 (the letters-only line, a score as a
    legend)."""
    cfg = hub_config("qwen3.6-35b-a3b")
    cfg["prompt"]["format"]["tail"] = "compact"
    return cfg


CONFIGS = {
    "qwen": lambda: hub_config("qwen3.6-35b-a3b"),
    "compact": compact_tail,
    "gemma": gemma_like,
    "template-single": template_single,
}

PARAGRAPH = (
    "The parcel left the warehouse on Monday and was scanned at the depot on Tuesday evening. "
    "The customer wrote twice to ask where it was, and the second message was short and angry."
)
LONG_STATE = " ".join([PARAGRAPH] * 3)

CASES = {
    "choice-described": (
        PARAGRAPH,
        {
            "type": "choice",
            "instructions": "What is the message about?",
            "criteria": {"delivery": "Where the parcel is", "billing": "A charge on the account", "other": None},
        },
    ),
    "choice-snake-keys": (
        "I would like to cancel my card.",
        {
            "type": "choice",
            "criteria": {"card_arrival": None, "lost_or_stolen_card": "", "Refund_not_showing_up": None},
        },
    ),
    "choice-index-keys": (
        {"text": "Where is my parcel?", "_hidden": "never shown"},
        {
            "type": "choice",
            "instructions": "Which intent?",
            "criteria": {"option_0": "track a parcel", "option_1": "cancel an order", "option_2": "change an address"},
        },
    ),
    "choice-equal-descriptions": (
        PARAGRAPH,
        {"type": "choice", "instructions": "Pick.", "criteria": {"a": "same words", "b": "same words", "c": None}},
    ),
    "choice-instructions-json": (
        "plain",
        {"type": "choice", "instructions": {"ask": "which", "n": 2}, "criteria": {"x": "first", "y": "second"}},
    ),
    "noul-bare": (PARAGRAPH, {"type": "noul", "instructions": "Is the customer angry?"}),
    "noul-criteria": (
        PARAGRAPH,
        {
            "type": "noul",
            "instructions": "Is the customer angry?",
            "criteria": {"true": "The message is angry", "false": "The message is calm"},
        },
    ),
    "noul-one-side": (PARAGRAPH, {"type": "noul", "criteria": {"true": "The message is angry"}}),
    "noul-sides-read-the-same": (
        PARAGRAPH,
        {"type": "noul", "instructions": "Same?", "criteria": {"true": "equal", "false": "equal"}},
    ),
    "score": (
        PARAGRAPH,
        {"type": "score", "instructions": "How urgent is it?", "criteria": ["not at all", "somewhat", "very", None, 4]},
    ),
    "score-default-instructions": (LONG_STATE, {"type": "score", "criteria": ["low", "high"]}),
    "state-list": (["a", {"b": 1}], {"type": "noul", "instructions": "Is this a list?"}),
    "state-with-blank-lines": (PARAGRAPH + "\n\n" + PARAGRAPH + "\n", {"type": "noul", "instructions": "Twice?"}),
    "long-state": (LONG_STATE, {"type": "choice", "instructions": "Topic?", "criteria": {"a": "parcel", "b": "money"}}),
}


@pytest.fixture(scope="module")
def ref_module():
    return load_reference()


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """(the stand-in engine, a repository directory per profile). The engine's model and tokenizer are shared with the
    references built from it, so a comparison sees one set of weights."""
    from huggingface_hub import snapshot_download

    from decisio.serve.hf_letters import HFLettersEngine

    engine = HFLettersEngine(MODEL, pad_to="block", pad_where="front", block_size=BLOCK)
    source = Path(MODEL) if Path(MODEL).is_dir() else Path(snapshot_download(MODEL))
    repos = {}
    for name, make in CONFIGS.items():
        repo = tmp_path_factory.mktemp(f"repo-{name}")
        for f in source.iterdir():
            (repo / f.name).symlink_to(f.resolve())
        (repo / "decision_config.json").write_text(json.dumps(make(), indent=2))
        repos[name] = repo
    return engine, repos


def serve(engine, cfg, state, question, order=None):
    """The server's answer to one question: SystemOne on the stand-in engine, set up as the profile says."""
    from decisio.readout.letters import PromptFormat
    from decisio.serve.systemone import SystemOne, SystemOneRequest

    fp, pad = cfg["prompt"]["fingerprint"], cfg["serving"]["pad_to"]
    engine.fmt = PromptFormat(**cfg["prompt"]["format"])
    engine.pad_policy, engine.pad_where = "always", fp["pad_where"]
    engine.pad_unit = BLOCK if pad == "block" else None
    temps = cfg["temperatures"]
    so = SystemOne(
        engine,
        "m",
        noul_rendering=cfg["prompt"]["noul_rendering"],
        hide_index_keys=fp["hide_index_keys"],
        desnake_labels=fp["desnake_labels"],
        temperature=temps["global"],
        temperatures={"choice": temps["choice"]},
    )
    if order is not None:
        question = {**question, "criteria": {k: question["criteria"][k] for k in order}}
    before = len(engine.scored_rows)
    answer = so.answer(SystemOneRequest.model_validate({"state": state, "questions": {"q": question}}))["answers"]["q"]
    rows = engine.scored_rows[before:]
    assert len(rows) == 1
    return answer, rows[0]


def same(a, b, path="answer"):
    """Equal, floats within TOLERANCE; the same keys in the same order."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), f"{path}: {list(a)} != {list(b)}"
        for k in a:
            same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, float):
        assert isinstance(b, float) and abs(a - b) <= TOLERANCE, f"{path}: {a!r} != {b!r}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def reference(ref_module, engine, repos, name):
    return ref_module.Reference(
        engine.model, engine.tok, json.loads((repos[name] / "decision_config.json").read_text()), block_size=BLOCK
    )


# ---- R1: the repository ---------------------------------------------------------------------------------------------


def test_r1_load_reads_the_repositorys_file(ref_module, served):
    engine, repos = served
    ref = ref_module.Reference.load(repos["qwen"], dtype="float32", block_size=BLOCK)
    assert ref.config["base"] == "qwen3.6-35b-a3b" and ref.pad_unit == BLOCK
    assert ref.decide("The sky is blue.", {"type": "noul", "instructions": "Is it?"})["type"] == "noul"
    assert ref_module.Reference.load(repos["gemma"], dtype="float32").pad_unit is None


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda c: c.update(schema="decisio.decision_config/2"), "schema"),
        (lambda c: c["prompt"].update(noul_rendering="sideways"), "noul_rendering"),
        (lambda c: c["prompt"]["format"].update(tail="loose"), "tail"),
        (lambda c: c["prompt"].update(label_form="bare"), "label_form"),
        (lambda c: c["serving"].update(pad_to="page"), "pad_to"),
        (lambda c: c["readout"].update(softcap=30.0), "softcap"),
        (lambda c: c["temperatures"].update(**{"global": 0}), "temperature"),
    ],
)
def test_r1_refuses_what_it_cannot_reproduce(ref_module, served, edit, message):
    engine, repos = served
    cfg = json.loads((repos["qwen"] / "decision_config.json").read_text())
    edit(cfg)
    with pytest.raises(ValueError, match=message):
        ref_module.Reference(engine.model, engine.tok, cfg, block_size=BLOCK)


def test_r1_a_directory_without_the_file_is_refused(ref_module, tmp_path):
    with pytest.raises(ValueError, match="decision_config.json"):
        ref_module.read_config(tmp_path)


def test_r1_images_are_refused(ref_module, served):
    engine, repos = served
    ref = reference(ref_module, engine, repos, "qwen")
    with pytest.raises(ValueError, match="image"):
        ref.decide("see data:image/png;base64,AAAA", {"type": "noul"})


# ---- R2, R3: the prompt and the answer ------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("profile", CONFIGS)
def test_r2_r3_the_prompt_and_the_answer_are_the_servers(ref_module, served, profile, case):
    engine, repos = served
    cfg = json.loads((repos[profile] / "decision_config.json").read_text())
    state, question = CASES[case]
    expected, (ids, labels) = serve(engine, cfg, state, question)
    ref = reference(ref_module, engine, repos, profile)
    prepared = ref.prepare(state, question)
    assert prepared.ids == ids
    # the server's labels are one id per label under `single`, a tuple of ids under `summed`; the reference's are tuples
    assert [list(g) for g in prepared.labels] == [list(g) if isinstance(g, tuple) else [g] for g in labels]
    same(ref.decide(state, question), expected)


def test_r2_the_state_is_padded_to_the_block_where_the_profile_says_so(ref_module, served):
    engine, repos = served
    state, question = CASES["choice-described"]
    cfg = json.loads((repos["qwen"] / "decision_config.json").read_text())
    padded = ref_module.Reference(engine.model, engine.tok, cfg, block_size=BLOCK).prepare(state, question)
    cfg["serving"]["pad_to"] = "none"
    bare = ref_module.Reference(engine.model, engine.tok, cfg, block_size=BLOCK).prepare(state, question)
    pads = next(i for i, t in enumerate(padded.ids) if t != ref_module.PAD_TOKEN)
    assert pads > 0 and padded.prefix_tokens % BLOCK == 0 and padded.prefix_tokens == bare.prefix_tokens + pads
    assert padded.ids[pads:] == bare.ids and bare.ids[0] != ref_module.PAD_TOKEN
    # the gemma profile pads nothing, and the "block" size of the Qwen base is vLLM's
    assert reference(ref_module, engine, repos, "gemma").pad_unit is None and ref_module.QWEN_BLOCK == 1056


# ---- R4: the temperatures -------------------------------------------------------------------------------------------


def test_r4_choice_has_its_own_temperature(ref_module, served):
    engine, repos = served
    state, question = CASES["choice-described"]
    cfg = json.loads((repos["qwen"] / "decision_config.json").read_text())
    ref = reference(ref_module, engine, repos, "qwen")
    cooled = json.loads(json.dumps(cfg))
    cooled["temperatures"]["choice"] = None  # a choice question then takes the global one
    other = ref_module.Reference(engine.model, engine.tok, cooled, block_size=BLOCK)
    a, b = ref.decide(state, question), other.decide(state, question)
    assert a["choice"] == b["choice"] and a["probabilities"] != b["probabilities"]
    assert ref.temperature_of("choice") == 1.37 and ref.temperature_of("noul") == 1.506 == ref.temperature_of("score")
    expected, _ = serve(engine, cooled, state, question)
    same(b, expected)


def test_r4_temperature_one_changes_nothing(ref_module):
    p = [0.5, 0.3, 0.2]
    assert ref_module.apply_temperature(p, 1.0) == p
    warm = ref_module.apply_temperature(p, 2.0)
    assert abs(sum(warm) - 1.0) < 1e-12 and warm[0] > warm[1] > warm[2] and warm[0] < p[0]


# ---- R5: ties and the order of the keys -----------------------------------------------------------------------------


def test_r5_a_tie_goes_to_the_smallest_key(ref_module):
    assert ref_module.top_index(["b", "a", "c"], [0.4, 0.4, 0.2]) == 1
    assert ref_module.top_index(["a", "b"], [0.3, 0.7]) == 1
    assert ref_module.top_index(["B", "a"], [0.5, 0.5]) == 0  # code-point order: "B" < "a"


def test_r5_the_order_of_the_keys_on_the_wire_changes_the_prompt_not_the_rule(ref_module, served):
    engine, repos = served
    state, question = CASES["choice-described"]
    ref = reference(ref_module, engine, repos, "qwen")
    reordered = {**question, "criteria": dict(reversed(list(question["criteria"].items())))}
    expected, _ = serve(engine, json.loads((repos["qwen"] / "decision_config.json").read_text()), state, reordered)
    same(ref.decide(state, reordered), expected)


# ---- R6: the command line -------------------------------------------------------------------------


def test_r6_the_command_line_prints_one_answer_per_question(ref_module, served, tmp_path, capsys):
    engine, repos = served
    state, question = CASES["noul-criteria"]
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"state": state, "questions": {"angry": question, "urgent": CASES["score"][1]}}))
    assert (
        ref_module.main(
            [str(repos["qwen"]), "--request", str(request), "--dtype", "float32", "--block-size", str(BLOCK)]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert list(out["answers"]) == ["angry", "urgent"] and out["answers"]["angry"]["type"] == "noul"
    cfg = json.loads((repos["qwen"] / "decision_config.json").read_text())
    for name, q in (("angry", question), ("urgent", CASES["score"][1])):
        expected, _ = serve(engine, cfg, state, q)
        same(out["answers"][name], expected)


def test_r6_the_command_line_takes_a_state_and_a_question(ref_module, served, capsys):
    engine, repos = served
    argv = [
        str(repos["gemma"]),
        "--state",
        "It rained.",
        "--question",
        json.dumps({"type": "noul", "instructions": "Wet?"}),
    ]
    assert ref_module.main([*argv, "--dtype", "float32"]) == 0
    answer = json.loads(capsys.readouterr().out)
    expected, _ = serve(
        engine,
        json.loads((repos["gemma"] / "decision_config.json").read_text()),
        "It rained.",
        {"type": "noul", "instructions": "Wet?"},
    )
    same(answer, expected)
    assert copy.deepcopy(answer)["type"] == "noul"
