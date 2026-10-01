# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The comparison models: each is asked the way its author's own recipe asks it, one move per request.

Cygnet, Winnow-12B and decider-4b are served by their authors' servers, which speak `/v1/systemone`: the same
`SystemOnePlayer`, with the server's default address. Jev-Omni and Nimble 9B publish library code and no server, so
their players load the author's own classes in this process and call them (Jev-Omni's `predict`, Nimble's
`ParallelScorer.score`), timing the call. Every model sees the same state text and the same question; a choice
question's options are shown the way each recipe shows `criteria` (an option with a description is `key: description`,
as the decider and Decisio renderings do; Nimble takes the description as its own field).

The weights, revisions and licences are in `MODELS`, and a record names them. Nothing here fits, trains or tunes
anything: a model is used as published, at its published temperature.
"""

import sys
import time
from pathlib import Path

from .adapter import question_for, state_text
from .players import Move, Player, SystemOnePlayer

# what a record says about each model (read from the sources on 2026-10-01; docs/games.md lists them for readers)
MODELS = {
    "cygnet": {
        "label": "Cygnet",
        "weights": "google/gemma-4-12B-it",
        "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "recipe": "github.com/blockbrain-ai/cygnet-recipe @ 3cf591c692dec649f7c134449814610307c7bb3a",
        "licence": "Apache-2.0 (weights), MIT (shim)",
        "serving": "vLLM 0.30.0 plus shim/decision_server.py, T = 3.4, thinking off",
        "url": "http://127.0.0.1:8010",
    },
    "winnow": {
        "label": "Winnow-12B",
        "weights": "EldanRing/Winnow-12B gguf/Winnow-12B-Q8_0.gguf",
        "revision": "b6ac22b0d51b69b18200acacb3fbdd98073fffe8",
        "recipe": "github.com/EldanRing/winnow-inference @ 6c2b3c04e248a319f2cb43832628eba03e55fe38",
        "licence": "Apache-2.0 (weights, base), MIT (code)",
        "serving": "winnow-inference server (patched llama.cpp), evaluation profile, T = 1.0",
        "url": "http://127.0.0.1:8091",
        "alias": "Winnow-12B",
    },
    "decider": {
        "label": "decider-4b v2",
        "weights": "Mapika/decider-4b",
        "revision": "7ab294cbdf6be6ac17fc818c10cdead744393d92",
        "recipe": "github.com/Mapika/decider (decider-ai 1.4.0 or later)",
        "licence": "Apache-2.0",
        "serving": "decider.serve (uvicorn), the revision's own temperature 1.935",
        "url": "http://127.0.0.1:8000",
    },
    "jev-omni": {
        "label": "Jev-Omni",
        "weights": "akhilaaa3/Jev-Omni",
        "revision": "5addda86ddee081a68fb067477ea100c221b8917",
        "recipe": "huggingface.co/akhilaaa3/Jev-Omni jev_omni.py (transformers 5.17.0), in this process",
        "licence": "Apache-2.0 (declared in the card)",
        "serving": "the author's JevOmni.predict, bf16, eager",
    },
    "nimble": {
        "label": "Nimble 9B",
        "weights": "bespokelabs/Bespoke-Nimble-9B (adapter) on Qwen/Qwen3.5-9B",
        "revision": "bd792f44ec8e265be861bfcdf4e05967ffe0e858",
        "recipe": "huggingface.co/bespokelabs/Bespoke-Nimble-9B inference.py (ParallelScorer), in this process",
        "licence": "Apache-2.0",
        "serving": "the author's ParallelScorer, bf16, T = 1.0",
    },
}


def option_texts(question):
    """The options of a choice question as text, in order: the key, or `key: description`."""
    return [k if not d else f"{k}: {d}" for k, d in question["criteria"].items()]


class HttpModelPlayer(SystemOnePlayer):
    """A model behind its author's own `/v1/systemone` server."""

    def __init__(self, kind, which, url=None, render=None):
        spec = MODELS[which]
        super().__init__(kind, url or spec["url"], spec.get("alias"), name=spec["label"], render=render)
        self.which = which

    def describe(self):
        return {**MODELS[self.which], "name": self.name, "kind": "http", "url": self.url, "render": self.render}


class InProcessPlayer(Player):
    """A model loaded in this process. Subclasses provide `load()` and `scores(state, question) -> {key: p}`."""

    which = ""

    def __init__(self, kind, render=None, path=None):
        self.kind, self.render, self.path = kind, render, path
        self.name = MODELS[self.which]["label"]
        self.engine = None

    def describe(self):
        return {**MODELS[self.which], "name": self.name, "kind": "in-process", "render": self.render}

    def warm_up(self, requests=3):
        if self.engine is None:
            self.engine = self.load()
        from .players import make_game

        game = make_game(self.kind, 0) if self.kind == "maze" else make_game(self.kind, "planet")
        for _ in range(requests):
            self.scores(state_text(self.kind, game.view(), self.render), question_for(self.kind, self.render))

    def probabilities(self, view):
        """{option key: probability} for a state, with no game behind it."""
        if self.engine is None:
            self.engine = self.load()
        return self.scores(state_text(self.kind, view, self.render), question_for(self.kind, self.render))

    def act(self, game):
        view = game.view()
        t0 = time.perf_counter()
        probs = self.probabilities(view)
        ms = (time.perf_counter() - t0) * 1000
        choice = max(probs, key=probs.get)
        raw = None
        if self.kind == "hangman":
            allowed = {k: p for k, p in probs.items() if k in view.remaining}
            top = max(allowed, key=allowed.get) if allowed and sum(allowed.values()) > 0 else view.remaining[0]
            raw = choice if choice != top else None
            choice = top
        return Move(choice, probs, ms, None, raw)

    def distribution(self, view):
        t0 = time.perf_counter()
        probs = self.probabilities(view)
        return probs, (time.perf_counter() - t0) * 1000, {}


class JevOmniPlayer(InProcessPlayer):
    which = "jev-omni"

    def load(self):
        """The author's classes with the weights fetched at the pinned revision (the loader itself fetches `main`)."""
        import json

        import torch
        import transformers
        from huggingface_hub import snapshot_download

        spec = MODELS[self.which]
        path = Path(
            self.path
            or snapshot_download(
                spec["weights"],
                revision=spec["revision"],
                allow_patterns=[
                    "config.json",
                    "generation_config.json",
                    "model*.safetensors*",
                    "processor_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                    "decision_config.json",
                    "head.pt",
                    "jev_omni.py",
                ],
            )  # fmt: skip
        )
        sys.path.insert(0, str(path))
        import jev_omni as author

        config = transformers.AutoConfig.from_pretrained(path)
        model = (
            getattr(transformers, config.architectures[0])
            .from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
            .eval()
        )
        decision = json.loads((path / "decision_config.json").read_text())
        head = author._Head256(decision["hidden_size"]).to("cuda").eval()
        head.load_state_dict(torch.load(path / "head.pt", map_location="cuda", weights_only=True))
        _, decoder = author._find_backbone(model)
        return author.JevOmni(model, head, transformers.AutoProcessor.from_pretrained(path), decoder, "cuda")

    def scores(self, state, question):
        keys = list(question["criteria"])
        out = self.engine.predict(state=state, question=question["instructions"], options=option_texts(question))
        return dict(zip(keys, out["probabilities"].values()))


class NimblePlayer(InProcessPlayer):
    which = "nimble"

    def load(self):
        """The author's `ParallelScorer` on the adapter fetched at the pinned revision."""
        from huggingface_hub import snapshot_download

        spec = MODELS[self.which]
        path = Path(self.path or snapshot_download("bespokelabs/Bespoke-Nimble-9B", revision=spec["revision"]))
        sys.path.insert(0, str(path))
        import inference as author

        return author.ParallelScorer(path)

    def scores(self, state, question):
        crit = question["criteria"]
        schema = {
            "decision": {
                "type": "enum",
                "description": question["instructions"],
                "choices": list(crit),
                "choice_descriptions": {k: d if d is not None else k for k, d in crit.items()},
            }
        }
        out = self.engine.score(state, schema)["fields"]["decision"]
        return dict(out["probabilities"])


def make_model_player(kind, which, url=None, render=None, path=None):
    if which in ("cygnet", "winnow", "decider"):
        return HttpModelPlayer(kind, which, url, render)
    if which == "jev-omni":
        return JevOmniPlayer(kind, render, path)
    if which == "nimble":
        return NimblePlayer(kind, render, path)
    raise ValueError(f"unknown model {which!r}; choose from {', '.join(MODELS)}")


__all__ = ["MODELS", "make_model_player", "option_texts"]
