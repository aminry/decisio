# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A CPU stand-in for the vLLM letters engine, for the CPU smoke test of the serving pipeline and its routes.

`HFLettersEngine` subclasses `vllm_engine.LettersEngine` and replaces only what needs vLLM: model loading and the
forward pass (`score_prompts`). Everything a request goes through before that (state tokenization, the split into
state and question, padding, label tokens, derived questions, `/v1/answer` and `/v1/systemone`) is the served code.
Scores are exact softmaxes over the label logits at the last position, one prompt at a time, float32 on the CPU; there
is no prefix cache, so `cached_tokens_mean` is always 0. Not for measurement.

    from decisio.serve.hf_letters import HFLettersEngine
    eng = HFLettersEngine("Qwen/Qwen3-0.6B", pad_to="block", pad_where="front", block_size=64)
"""

from __future__ import annotations

import threading
import time

import numpy as np

from decisio.readout.letters import allowed_ids, is_grouped, label_log_softmax
from decisio.serve.engine_health import guarded
from decisio.serve.vllm_engine import PAD_PLACES, PAD_TOKEN, LettersEngine


class HFLettersEngine(LettersEngine):
    def __init__(
        self, model, pad_to=None, pad_token=PAD_TOKEN, pad_where="front", block_size=64, warm_up=True, revision=None
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if pad_where not in PAD_PLACES:
            raise ValueError(f"pad_where must be one of {PAD_PLACES}")
        self.mode, self.pad_token, self.pad_where = "separate", pad_token, pad_where
        self.adapters = {}
        self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32, revision=revision).eval()
        self.model_name, self.revision = model, revision
        self.block_size = self.match_unit = block_size
        self.pad_unit = None if not pad_to else (self.block_size if pad_to == "block" else int(pad_to))
        self._lock = threading.Lock()
        self.scored_rows = []  # every (token ids, label ids) scored, for the smoke test's prompt checks
        if warm_up:
            self._warm_up()

    def probe(self):
        """The model over a one-token prompt (engine_health)."""
        import torch

        with torch.no_grad():
            self.model(torch.tensor([self.tok.encode("ok", add_special_tokens=False)[:1]]), logits_to_keep=1)

    def facts(self):
        import torch
        import transformers

        return {
            "engine": "hf_letters (CPU stand-in, not for measurement)",
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "model": self.model_name,
            "mode": self.mode,
            "block_size": self.block_size,
            "pad_unit": self.pad_unit,
            "pad_where": self.pad_where if self.pad_unit else None,
            "adapters": [],
        }

    def send_warm(self, warm, adapter=None, mm=None, mm_uuids=None):
        """The forward passes a warm-up costs, one prompt at a time; with no prefix cache, nothing is kept (a
        registration deferred until after its response runs through here, decisio.serve.boundary)."""
        import torch

        if adapter:
            raise ValueError("the CPU stand-in serves no adapters")
        t0 = time.perf_counter()
        with torch.no_grad():
            for ids in warm:
                guarded(self, self.model, torch.tensor([list(ids)]), logits_to_keep=1)
        return (time.perf_counter() - t0) * 1000

    def score_prompts(self, rows, adapter=None, warm=(), skip_cache=False, mm=None, mm_uuids=None):
        import torch

        if mm:
            raise ValueError("the text stand-in takes no images (HFImageLettersEngine does)")
        if adapter:
            raise ValueError("the CPU stand-in serves no adapters")
        t0 = time.perf_counter()
        probs, token_lps = [], []
        with torch.no_grad():
            for ids, lab in rows:
                self.scored_rows.append((list(ids), list(lab)))
                # the last position's logits only: the output layer over every position dominated the stand-in's time
                logits = guarded(self, self.model, torch.tensor([ids]), logits_to_keep=1).logits[0, -1].double()
                if is_grouped(lab):  # several forms per label: their probabilities summed (label_log_softmax)
                    flat = allowed_ids(lab)
                    lp = torch.log_softmax(logits[flat], -1).numpy()
                    at = dict(zip(flat, lp.tolist()))
                    token_lps.append([[at[t] for t in g] for g in lab])
                    z = label_log_softmax(logits[flat].numpy(), lab)
                else:
                    z = logits[list(lab)].numpy()
                e = np.exp(z - z.max())
                probs.append(e / e.sum())
        ms = (time.perf_counter() - t0) * 1000
        return probs, {
            "warm_ms": 0.0,
            "cached_tokens_mean": 0.0,
            "engine_timing": [],
            "prompt_tokens_mean": float(np.mean([len(r[0]) for r in rows])) if rows else 0.0,
            "prompt_tokens": sum(len(r[0]) for r in rows),
            "forward_ms": ms,
            **({"label_token_logprobs": token_lps} if token_lps else {}),
        }


class HFImageLettersEngine:
    """The CPU stand-in's image engine: a small checkpoint of the same family under its multimodal class
    (transformers AutoModelForImageTextToText), the same image request path as vLLM's (decisio.serve.image_engine).
    The engine's own expansion of each `<|image_pad|>` is done here, before the forward pass. Not for measurement."""

    def __new__(cls, *args, **kwargs):
        from decisio.serve.image_engine import ImageRequests

        class _Engine(ImageRequests, HFLettersEngine):
            def __init__(self, model, pad_to=None, pad_token=PAD_TOKEN, pad_where="front", block_size=64):
                import torch
                from transformers import AutoModelForImageTextToText, AutoTokenizer

                self.mode, self.pad_token, self.pad_where = "separate", pad_token, pad_where
                self.adapters = {}
                self.tok = AutoTokenizer.from_pretrained(model)
                self.model = AutoModelForImageTextToText.from_pretrained(model, torch_dtype=torch.float32).eval()
                self.model_name = model
                self.block_size = self.match_unit = block_size
                self.pad_unit = None if not pad_to else (self.block_size if pad_to == "block" else int(pad_to))
                self._lock = threading.Lock()
                self.scored_rows = []
                self._image_setup(model)
                self._warm_up()

            def facts(self):
                return {**HFLettersEngine.facts(self), "class": "multimodal (CPU stand-in)"}

            def score_prompts(self, rows, adapter=None, warm=(), skip_cache=False, mm=None, mm_uuids=None):
                import torch

                if not mm:
                    return HFLettersEngine.score_prompts(self, rows, adapter, warm, skip_cache)
                images = mm["image"]
                enc = self.processor.image_processor(images=images, return_tensors="pt")
                counts = [int(g.prod()) // (self.merge**2) for g in enc["image_grid_thw"]]
                t0 = time.perf_counter()
                probs, lens = [], []
                with torch.no_grad():
                    for ids, lab in rows:
                        exp, k = [], 0
                        for t in ids:
                            if t == self.image_pad_id:
                                exp += [t] * counts[k]
                                k += 1
                            else:
                                exp.append(t)
                        self.scored_rows.append((list(ids), list(lab)))
                        # 1 on image tokens, 0 elsewhere: what the processor returns beside input_ids (M-RoPE needs it)
                        types = torch.tensor([[int(t == self.image_pad_id) for t in exp]])
                        out = guarded(
                            self,
                            self.model,
                            input_ids=torch.tensor([exp]),
                            attention_mask=torch.ones(1, len(exp), dtype=torch.long),
                            mm_token_type_ids=types,
                            pixel_values=enc["pixel_values"],
                            image_grid_thw=enc["image_grid_thw"],
                        )
                        z = out.logits[0, -1].double()[list(lab)].numpy()
                        e = np.exp(z - z.max())
                        probs.append(e / e.sum())
                        lens.append(len(exp))
                return probs, {
                    "warm_ms": 0.0,
                    "cached_tokens_mean": 0.0,
                    "engine_timing": [],
                    "prompt_tokens_mean": float(np.mean(lens)),
                    "prompt_tokens": sum(lens),
                    "engine_prompt_tokens": lens,
                    "cached_tokens": [0] * len(lens),
                    "forward_ms": (time.perf_counter() - t0) * 1000,
                }

        return _Engine(*args, **kwargs)
