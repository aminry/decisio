# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Ten-state conformance of a comparison model's player against the model's own tooling.

Before any game is measured with a model, the player's answers on ten fixed game states (five maze, five Hangman) are
compared with what the model's own code gives for the same question: the same argmax on every state and every
probability within a tolerance. A model that fails is not measured; the failure is the finding.

    python conformance.py decider  --url http://127.0.0.1:8000 --author-dir ~/decider-src --weights /data/decider-4b
    python conformance.py cygnet   --url http://127.0.0.1:8010 --author-dir ~/cygnet-recipe --vllm http://127.0.0.1:8890
    python conformance.py winnow   --url http://127.0.0.1:8091
    python conformance.py jev-omni --weights /data/jev-omni --author-dir /data/jev-omni
    python conformance.py nimble   --weights /data/nimble --author-dir /data/nimble

The references, each the author's own path to the same answer:
  decider   `Decider(weights).decide(state, [{"question", "options"}])`, the library call, against the server
  cygnet    the recipe's `cygnet_shim.answer_for`, called here against the same vLLM, against the decision server
  winnow    the server's own `winnow: {diagnostics: true}` candidate logits, softmaxed at its temperature (1.0),
            against the probabilities the plain request returns
  jev-omni  the author's `verification.json` cases through the loader (its own check: worst difference 0.0194), and
            the loader's `predict` on the ten states with the options written out here, against the player
  nimble    the author's `ParallelScorer.score` with a schema built here from the question, against the player
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gamelib.adapter import question_for, state_text  # noqa: E402
from gamelib.hangman import EntropySolver, Hangman, heldout_words  # noqa: E402
from gamelib.mazechase import MazeChase, teacher_move  # noqa: E402
from gamelib.models import MODELS, option_texts  # noqa: E402
from gamelib.players import make_player  # noqa: E402

TOLERANCE = {"decider": 0.02, "cygnet": 0.02, "winnow": 0.01, "jev-omni": 0.02, "nimble": 0.02}
MAZE_STEPS = (0, 6, 19, 33, 47)  # moves of the teacher before each maze state
HANGMAN_STEPS = (0, 2, 3, 4, 5)  # moves of the solver before each Hangman state


def conformance_states():
    """[(kind, view)]: five maze states (seeds 0 to 4) and five Hangman states (the first five held-out words)."""
    out = []
    for seed, steps in enumerate(MAZE_STEPS):
        game = MazeChase(seed)
        for _ in range(steps):
            if not game.over:
                game.step(teacher_move(game.view()))
        out.append(("maze", game.view()))
    solver = EntropySolver()
    for word, steps in zip(heldout_words(), HANGMAN_STEPS):
        game = Hangman(word)
        for _ in range(steps):
            if not game.over:
                game.step(solver.choose(game.view()))
        out.append(("hangman", game.view()))
    return out


def compare(ours, reference, tolerance):
    """(same argmax, largest |difference|) between two {option: probability} maps over the same options."""
    keys = list(ours)
    same = max(ours, key=ours.get) == max(reference, key=reference.get) and set(ours) == set(reference)
    worst = max(abs(ours[k] - reference.get(k, 0.0)) for k in keys)
    return same, worst


def run(player, reference, tolerance, render=None):
    """[{"state", "argmax_ours", "argmax_reference", "same", "max_abs_diff", "ok"}] over the ten states."""
    rows = []
    for i, (kind, view) in enumerate(conformance_states()):
        player.kind = kind
        ours = player.distribution(view)[0]
        theirs = reference(kind, view)
        same, worst = compare(ours, theirs, tolerance)
        rows.append(
            {
                "state": i,
                "game": kind,
                "argmax_ours": max(ours, key=ours.get),
                "argmax_reference": max(theirs, key=theirs.get),
                "same": same,
                "max_abs_diff": worst,
                "ok": same and worst <= tolerance,
            }
        )
    return rows


# ---- the references --------------------------------------------------------------------------------------------------


def reference_decider(weights):
    sys.path.insert(0, str(Path(weights).parent))  # decider-ai installed, or its source beside the weights
    from decider.infer import Decider

    decider = Decider(weights)

    def ref(kind, view):
        question = question_for(kind)
        options = option_texts(question)
        out = decider.decide(state_text(kind, view), [{"question": question["instructions"], "options": options}])[0]
        return dict(zip(question["criteria"], (out["probs"][o] for o in options)))

    return ref


def reference_cygnet(author_dir, vllm_url):
    import os

    os.environ.setdefault("SHIM_VLLM", vllm_url.rstrip("/") + "/v1/chat/completions")
    os.environ.setdefault("SHIM_MODEL", "cygnet")
    os.environ.setdefault("SHIM_TEMPERATURE", "3.4")
    sys.path.insert(0, str(Path(author_dir) / "shim"))
    import cygnet_shim as shim

    def ref(kind, view):
        question = question_for(kind)
        answer, _usage, err = shim.answer_for(state_text(kind, view), question)
        if err:
            raise RuntimeError(err)
        return dict(answer["probabilities"])

    return ref


def reference_winnow(url):
    from gamelib.teach import call

    def ref(kind, view):
        question = question_for(kind)
        body = {
            "model": MODELS["winnow"]["alias"],
            "state": state_text(kind, view),
            "questions": {"decision": question},
            "winnow": {"diagnostics": True},
        }
        status, out = call(url, "/v1/systemone", body)
        if status != 200:
            raise RuntimeError(f"winnow answered {status}: {out}")
        diag = out["answers"]["decision"].get("diagnostics") or out.get("diagnostics", {})
        logits = diag.get("candidate_logits")
        if not logits:
            raise RuntimeError("the diagnostics carry no candidate logits; check the server's version (docs/API.md)")
        import math

        top = max(logits)
        exps = [math.exp(x - top) for x in logits]
        return dict(zip(question["criteria"], (e / sum(exps) for e in exps)))

    return ref


def reference_jev_omni(author_dir, weights):
    from gamelib.models import JevOmniPlayer

    engine = JevOmniPlayer("maze", path=weights).load()

    def ref(kind, view):
        question = question_for(kind)
        options = option_texts(question)
        out = engine.predict(state=state_text(kind, view), question=question["instructions"], options=options)
        return dict(zip(question["criteria"], out["probabilities"].values()))

    ref.engine = engine
    return ref


def reference_nimble(author_dir, weights):
    from gamelib.models import NimblePlayer

    engine = NimblePlayer("maze", path=weights).load()

    def ref(kind, view):
        question = question_for(kind)
        crit = question["criteria"]
        schema = {
            "decision": {
                "type": "enum",
                "description": question["instructions"],
                "choices": list(crit),
                "choice_descriptions": {k: d if d is not None else k for k, d in crit.items()},
            }
        }
        return dict(engine.score(state_text(kind, view), schema)["fields"]["decision"]["probabilities"])

    ref.engine = engine
    return ref


def verify_jev_omni_cases(engine, author_dir):
    """The author's own check: the loader on `verification.json`'s four cases against its stored reference."""
    spec = json.loads((Path(author_dir) / "verification.json").read_text())
    worst = 0.0
    for case, expected in zip(spec["cases"], spec["reference"]):
        got = engine.predict(state=case["state"], question=case["question"], options=case["options"])
        values = list(got["probabilities"].values())
        ref = (
            expected if isinstance(expected, list) else list(expected.values()) if isinstance(expected, dict) else None
        )
        if ref:
            worst = max(worst, max(abs(a - b) for a, b in zip(values, ref)))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", choices=sorted(TOLERANCE))
    ap.add_argument("--url", default=None, help="the model's server (the models that have one)")
    ap.add_argument("--author-dir", default=None, help="the author's code at the pinned revision")
    ap.add_argument("--weights", default=None, help="a local copy of the weights, where the reference loads them")
    ap.add_argument(
        "--vllm", default="http://127.0.0.1:8890", help="cygnet: the vLLM server behind the decision server"
    )
    ap.add_argument("--out", default=None, help="write the table as JSON here")
    a = ap.parse_args()

    player = make_player("maze", a.model, a.url, path=a.weights)
    extra = {}
    if a.model == "decider":
        ref = reference_decider(a.weights)
    elif a.model == "cygnet":
        ref = reference_cygnet(a.author_dir, a.vllm)
    elif a.model == "winnow":
        ref = reference_winnow(a.url or MODELS["winnow"]["url"])
    elif a.model == "jev-omni":
        ref = reference_jev_omni(a.author_dir, a.weights)
        player.engine = ref.engine  # one copy of the weights serves both
        extra["author_verification_worst_abs_diff"] = verify_jev_omni_cases(ref.engine, a.author_dir)
    else:
        ref = reference_nimble(a.author_dir, a.weights)
        player.engine = ref.engine
    rows = run(player, ref, TOLERANCE[a.model])
    ok = all(r["ok"] for r in rows)
    report = {"model": MODELS[a.model], "tolerance": TOLERANCE[a.model], "pass": ok, "states": rows, **extra}
    for r in rows:
        print(f"state {r['state']} {r['game']:7s} ours {r['argmax_ours']:6s} reference {r['argmax_reference']:6s} "
              f"max|dp| {r['max_abs_diff']:.4f} {'ok' if r['ok'] else 'FAIL'}")  # fmt: skip
    print("CONFORMANCE", "PASS" if ok else "FAIL", a.model)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
