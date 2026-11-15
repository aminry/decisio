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

import time

import numpy as np

from decisio.readout.letters import allowed_ids, is_grouped, label_log_softmax
from decisio.serve import image_marks as marks
from decisio.serve.combine import FairLock
from decisio.serve.engine_health import guarded
from decisio.serve.gemma_images import DEFAULT_MAX_IMAGES, GemmaImageRequests
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
        self._lock = FairLock()  # as the vLLM engine's
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
        self.refuse_over_window(rows)
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
    (transformers AutoModelForImageTextToText), the same image request path as vLLM's (decisio.serve.image_engine,
    decisio.serve.placed_images).
    The engine's own expansion of each `<|image_pad|>` is done here, before the forward pass. Not for measurement."""

    def __new__(cls, *args, **kwargs):
        from decisio.serve.image_engine import ImageRequests

        class _Engine(ImageRequests, HFLettersEngine):
            def __init__(
                self,
                model,
                pad_to=None,
                pad_token=PAD_TOKEN,
                pad_where="front",
                block_size=64,
                revision=None,
                image_prompt="placed",
            ):
                import torch
                from transformers import AutoModelForImageTextToText, AutoTokenizer

                self.image_prompt = image_prompt

                self.mode, self.pad_token, self.pad_where = "separate", pad_token, pad_where
                self.adapters = {}
                self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model, torch_dtype=torch.float32, revision=revision
                ).eval()
                self.model_name, self.revision = model, revision
                self.block_size = self.match_unit = block_size
                self.pad_unit = None if not pad_to else (self.block_size if pad_to == "block" else int(pad_to))
                self._lock = FairLock()
                self.scored_rows = []
                self._image_setup(model, revision)
                self._warm_up()

            def facts(self):
                return {**HFLettersEngine.facts(self), "class": "multimodal (CPU stand-in)"}

            def score_prompts(
                self,
                rows,
                adapter=None,
                warm=(),
                skip_cache=False,
                mm=None,
                mm_uuids=None,
                row_mm=None,
                row_mm_uuids=None,
            ):
                import torch

                if not mm and row_mm is None:
                    return HFLettersEngine.score_prompts(self, rows, adapter, warm, skip_cache)
                t0 = time.perf_counter()
                probs, lens = [], []
                with torch.no_grad():
                    for i, (ids, lab) in enumerate(rows):
                        pics = list(((row_mm[i] if row_mm is not None else mm) or {}).get("image") or [])
                        enc = self.processor.image_processor(images=pics, return_tensors="pt") if pics else None
                        counts = [int(g.prod()) // (self.merge**2) for g in enc["image_grid_thw"]] if pics else []
                        exp, k = [], 0
                        for t in ids:
                            if t == self.image_pad_id:
                                exp += [t] * counts[k]
                                k += 1
                            else:
                                exp.append(t)
                        if k != len(pics):
                            raise ValueError("a row's placeholders and its images differ")
                        self.scored_rows.append((list(ids), list(lab)))
                        # 1 on image tokens, 0 elsewhere: what the processor returns beside input_ids (M-RoPE needs it)
                        types = torch.tensor([[int(t == self.image_pad_id) for t in exp]])
                        kw = (
                            {"pixel_values": enc["pixel_values"], "image_grid_thw": enc["image_grid_thw"]}
                            if pics
                            else {}
                        )
                        out = guarded(
                            self,
                            self.model,
                            input_ids=torch.tensor([exp]),
                            attention_mask=torch.ones(1, len(exp), dtype=torch.long),
                            mm_token_type_ids=types,
                            **kw,
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


class HFGemmaImageLettersEngine(GemmaImageRequests, HFLettersEngine):
    """The CPU stand-in's Gemma image engine: transformers' Gemma 4 conditional-generation class (the 12B's encoder-free
    `gemma4_unified`, or the 31B's `gemma4` with its vision tower), the same image request path as vLLM's
    (decisio.serve.gemma_images). The expansion of each `<|image|>` placeholder to the begin token, the image's tokens
    and the end token is done here, before the forward pass, as vLLM does it, and the token types the processor returns
    beside the ids are passed to the model, so an image's tokens attend to each other on the sliding layers.

    `model`: a checkpoint (a directory or hub id) of a Gemma 4 conditional-generation model, or None to build a model
    of the class `kind` ("unified" or "tower") from the small `config` (a dict of text and vision sizes) with random
    weights, `seed`: no download, for the tests. `tokenizer` names the tokenizer (the real Gemma 4 one). Not for
    measurement. Every forward pass is logged in `self.forwards`: the expanded ids, the token types, the image count."""

    def __init__(
        self,
        tokenizer,
        model=None,
        kind="unified",
        config=None,
        budget=marks.DEFAULT_BUDGET,
        max_images=DEFAULT_MAX_IMAGES,
        pad_to=None,
        pad_token=PAD_TOKEN,
        pad_where="front",
        block_size=64,
        seed=0,
        revision=None,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        if pad_where not in PAD_PLACES:
            raise ValueError(f"pad_where must be one of {PAD_PLACES}")
        self.mode, self.pad_token, self.pad_where = "separate", pad_token, pad_where
        self.adapters = {}
        self.tok = AutoTokenizer.from_pretrained(tokenizer, revision=revision)
        if model is not None:
            self.model = AutoModelForImageTextToText.from_pretrained(model, torch_dtype=torch.float32).eval()
        else:
            self.model = self._random_model(kind, config or {}, seed)
        self.kind = "unified" if "Unified" in type(self.model).__name__ else "tower"
        self.model_name, self.revision = model or f"random {self.kind}", revision
        self.block_size = self.match_unit = block_size
        self.pad_unit = None if not pad_to else (self.block_size if pad_to == "block" else int(pad_to))
        self._lock = FairLock()
        self.scored_rows = []
        self.forwards = []
        self.adapter_calls = []
        self.prefix_caching = False  # no cache: a warm-up would only repeat a forward pass
        self._image_setup(budget, max_images)
        self.boi = self.tok.convert_tokens_to_ids("<|image>")
        self.eoi = self.tok.convert_tokens_to_ids("<image|>")
        self._warm_up()

    @staticmethod
    def _random_model(kind, config, seed):
        import torch
        from transformers import (
            Gemma4Config,
            Gemma4ForConditionalGeneration,
            Gemma4UnifiedConfig,
            Gemma4UnifiedForConditionalGeneration,
        )

        text = {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "sliding_window": 64,
            "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
            "vocab_size": 262144,
            "final_logit_softcapping": 30.0,
            "use_bidirectional_attention": "vision",
            **config.get("text", {}),
        }
        torch.manual_seed(seed)
        if kind == "unified":
            vision = {"mm_embed_dim": 32, "output_proj_dims": 32, **config.get("vision", {})}
            cfg = Gemma4UnifiedConfig(text_config=text, vision_config=vision, audio_config=None)
            return Gemma4UnifiedForConditionalGeneration(cfg).eval()
        text = {"global_head_dim": 16, "hidden_size_per_layer_input": 0, **text}
        vision = {
            "hidden_size": 24,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 12,
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "position_embedding_size": 64,
            **config.get("vision", {}),
        }
        cfg = Gemma4Config(text_config=text, vision_config=vision, audio_config=None)
        return Gemma4ForConditionalGeneration(cfg).eval()

    def _image_processor(self):
        if self.kind == "unified":
            from transformers.models.gemma4_unified.image_processing_gemma4_unified import Gemma4UnifiedImageProcessor

            return Gemma4UnifiedImageProcessor()
        from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor

        return Gemma4ImageProcessor()

    def add_adapter(self, name, scale=0.5, seed=0):
        """An adapter for the tests: a random change of the language model's MLP output weights (what a merged LoRA
        is, in its place), applied while a request names it. Nothing of the image path is touched, as with vLLM LoRA
        on the language model."""
        import torch

        gen = torch.Generator().manual_seed(seed)
        self.adapters[name] = {
            n: torch.randn(p.shape, generator=gen) * scale * float(p.detach().std())
            for n, p in self.model.named_parameters()
            if "language_model" in n and n.endswith("mlp.down_proj.weight")
        }

    def facts(self):
        return {
            **HFLettersEngine.facts(self),
            "class": f"gemma4 {self.kind} (CPU stand-in)",
            "adapters": sorted(self.adapters),
            "images": self.image_facts(),
        }

    def send_warm(self, warm, adapter=None, mm=None, mm_uuids=None):
        return 0.0

    def score_prompts(
        self, rows, adapter=None, warm=(), skip_cache=False, mm=None, mm_uuids=None, row_mm=None, row_mm_uuids=None
    ):
        import torch

        if adapter and adapter not in self.adapters:
            raise ValueError(f"unknown adapter {adapter!r}")
        if row_mm is None and not mm and not adapter:
            return HFLettersEngine.score_prompts(self, rows, adapter, warm, skip_cache)
        if adapter:
            self.adapter_calls.append(adapter)
            params = dict(self.model.named_parameters())
            with torch.no_grad():
                for n, d in self.adapters[adapter].items():
                    params[n].add_(d)
            try:
                return self._score_rows(rows, mm, row_mm)
            finally:
                with torch.no_grad():
                    for n, d in self.adapters[adapter].items():
                        params[n].sub_(d)
        return self._score_rows(rows, mm, row_mm)

    def generate_rows(self, rows, adapter=None, row_mm=None, row_mm_uuids=None, max_tokens=32, top=20):
        """The stand-in's greedy text (`LettersEngine.generate_rows`): each step runs the whole expanded prompt again,
        float32 on the CPU, no cache; the same figures."""
        import torch

        if adapter and adapter not in self.adapters:
            raise ValueError(f"unknown adapter {adapter!r}")
        if adapter:
            self.adapter_calls.append(adapter)
            params = dict(self.model.named_parameters())
            with torch.no_grad():
                for n, d in self.adapters[adapter].items():
                    params[n].add_(d)
        try:
            t0 = time.perf_counter()
            ip = self._image_processor()
            gens, lens = [], []
            for i, ids in enumerate(rows):
                pics = list(((row_mm[i] if row_mm is not None else None) or {}).get("image") or [])
                enc = ip(images=pics, max_soft_tokens=self.image_budget, return_tensors="pt") if pics else None
                counts = [int(c) for c in enc["num_soft_tokens_per_image"]] if pics else []
                exp, k = [], 0
                for t in ids:
                    if t == self.image_token_id:
                        exp += [self.boi] + [t] * counts[k] + [self.eoi]
                        k += 1
                    else:
                        exp.append(t)
                if k != len(pics):
                    raise ValueError("a row's placeholders and its images differ")
                lens.append(len(exp))
                kw = (
                    {"pixel_values": enc["pixel_values"], "image_position_ids": enc["image_position_ids"]}
                    if pics
                    else {}
                )
                steps = []
                with torch.no_grad():
                    for _ in range(max_tokens):
                        types = torch.tensor([[int(t == self.image_token_id) for t in exp]])
                        self.forwards.append(
                            {"input_ids": list(exp), "mm_token_type_ids": types[0].tolist(), "images": len(pics)}
                        )
                        out = guarded(
                            self,
                            self.model,
                            input_ids=torch.tensor([exp]),
                            attention_mask=torch.ones(1, len(exp), dtype=torch.long),
                            mm_token_type_ids=types,
                            logits_to_keep=1,
                            **kw,
                        )
                        lp = torch.log_softmax(out.logits[0, -1].double(), -1)
                        v, ix = lp.topk(top)
                        names = self.tok.convert_ids_to_tokens(ix.tolist())
                        steps.append((names[0], list(zip(names, v.tolist()))))
                        exp.append(int(ix[0]))
                gens.append(steps)
        finally:
            if adapter:
                with torch.no_grad():
                    for n, d in self.adapters[adapter].items():
                        params[n].sub_(d)
        return gens, {
            "generate_ms": (time.perf_counter() - t0) * 1000,
            "generated_tokens": [len(g) for g in gens],
            "cached_tokens": [0] * len(gens),
            "engine_prompt_tokens": lens,
        }

    def _score_rows(self, rows, mm, row_mm):
        import torch

        t0 = time.perf_counter()
        ip = self._image_processor()
        probs, lens = [], []
        with torch.no_grad():
            for i, (ids, lab) in enumerate(rows):
                images = (row_mm[i] if row_mm is not None else mm) or {}
                pics = list(images.get("image") or [])
                enc = ip(images=pics, max_soft_tokens=self.image_budget, return_tensors="pt") if pics else None
                counts = [int(c) for c in enc["num_soft_tokens_per_image"]] if pics else []
                exp, k = [], 0
                for t in ids:
                    if t == self.image_token_id:
                        exp += [self.boi] + [t] * counts[k] + [self.eoi]  # what vLLM's replacement makes of it
                        k += 1
                    else:
                        exp.append(t)
                if k != len(pics):
                    raise ValueError("a row's placeholders and its images differ")
                self.scored_rows.append((list(ids), list(lab)))
                # 1 on the image tokens, 0 elsewhere (the begin and end tokens are text): what the processor returns
                # beside input_ids; the model builds its image blocks from it
                types = torch.tensor([[int(t == self.image_token_id) for t in exp]])
                kw = (
                    {"pixel_values": enc["pixel_values"], "image_position_ids": enc["image_position_ids"]}
                    if pics
                    else {}
                )
                self.forwards.append(
                    {"input_ids": list(exp), "mm_token_type_ids": types[0].tolist(), "images": len(pics)}
                )
                out = guarded(
                    self,
                    self.model,
                    input_ids=torch.tensor([exp]),
                    attention_mask=torch.ones(1, len(exp), dtype=torch.long),
                    mm_token_type_ids=types,
                    logits_to_keep=1,
                    **kw,
                )
                z = out.logits[0, -1].double()
                if is_grouped(lab):
                    flat = allowed_ids(lab)
                    z = label_log_softmax(z[flat].numpy(), lab)
                else:
                    z = z[list(lab)].numpy()
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
