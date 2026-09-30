# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The readout's hidden state at serving time, for the per-task intent head (decisio.readout.intent_head).

The head needs h, the final-norm hidden state at the readout position (the vector the letter logits are read from),
which vLLM's generate path does not return. Two modes read it (`vllm_engine.resolve_head_mode`; docs/handoffs/tasks.md):

  single-engine   the default (`SingleEngineHidden`): the serving engine itself runs decisio's hidden-readout class and
                  returns h through reserved logit columns (docs/design/hidden-state-readout.md). No second weight copy;
                  three engine requests per head question.
  second-engine   --head-engine (`HiddenEngine`): a second copy of the served model in vLLM's pooling mode
                  (`runner="pooling"`, last-token pooling, no activation) returns h for the exact token rows the served
                  path builds (front padding included). One pooling request per head question, but a second weight
                  copy on the card, and vLLM 0.30's pooling runner has no prefix cache on this hybrid model, so every
                  head question prefills its whole prompt.

In both, the K letter log-probabilities are computed from h and the output layer's K label rows,

    z  = h @ W_labels^T
    lp = log_softmax(z)

so a head is fitted and served on the same (lp, h), and `intent_head.apply_intent_head(lp, h, rec, options)` is the
served arithmetic exactly. The two modes differ only in the dtype of z (`label_logits`).

    eng = HiddenEngine("/models/Qwen3.6-35B-A3B-FP8", pad_to="block", pad_where="front", gpu_memory_utilization=0.47)
    [(lp, h), ...] = eng.readout(state, questions)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from decisio.serve.vllm_engine import LettersEngine


def log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


class HiddenReadout:
    """Mixin: `readout(state, questions)` -> [(lp, h)] per question, from `hidden_rows(token id lists)` and
    `label_rows(label ids)`, both engine-specific."""

    def label_logits(self, h, lab):
        """The label logits from h, in float64 (deterministic, whatever the kernels)."""
        return h.astype(np.float64) @ self.label_rows(lab).T

    def readout(self, state, questions):
        with self._lock:
            t0 = time.perf_counter()
            rows, _ = self._prepare_separate(state, questions)
            H = self.hidden_rows([ids for ids, _ in rows])
            out = []
            for (_, lab), h in zip(rows, H):
                h = np.asarray(h, dtype=np.float32)
                out.append((log_softmax(self.label_logits(h, lab)), h))
            self.last_ms = (time.perf_counter() - t0) * 1000
        return out


class HiddenEngine(HiddenReadout, LettersEngine):
    """The served model in vLLM's pooling mode, returning h at the last prompt position; every prompt setting as the
    served default (the same rows: `_prepare_separate`, front padding to the block)."""

    def __init__(self, model, pad_to="block", pad_where="front", gpu_memory_utilization=0.47, engine_kw=None,
                 max_model_len=32768, max_num_seqs=256):
        from vllm import LLM
        from vllm.config import PoolerConfig
        self.mode, self.pad_token, self.pad_where = "hidden", 198, pad_where
        self.adapters = {}
        kw = dict(model=model, max_model_len=max_model_len, max_num_seqs=max_num_seqs,
                  gpu_memory_utilization=gpu_memory_utilization, limit_mm_per_prompt={"image": 0, "video": 0},
                  runner="pooling", convert="embed",
                  pooler_config=PoolerConfig(task="embed", seq_pooling_type="LAST", use_activation=False))
        kw.update({k: v for k, v in (engine_kw or {}).items() if k != "limit_mm_per_prompt"})
        self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        cfg = self.llm.llm_engine.vllm_config.cache_config
        self.block_size = cfg.block_size
        self.match_unit = self.block_size
        self.pad_unit = None if not pad_to else (self.block_size if pad_to == "block" else int(pad_to))
        self._W = self._load_lm_head(model)
        import threading
        self._lock = threading.Lock()
        for _ in range(2):                                  # warm-up, as the generate engine does
            self.readout("Warm-up. " * 40, [{"kind": "noul", "instructions": "Is this a warm-up request?"}])

    @staticmethod
    def _load_lm_head(model):
        """The output layer's weight (vocabulary x hidden), from the checkpoint's own shards."""
        from safetensors import safe_open
        root = Path(model)
        index = root / "model.safetensors.index.json"
        shard = json.loads(index.read_text())["weight_map"]["lm_head.weight"] if index.exists() else "model.safetensors"
        with safe_open(str(root / shard), framework="pt") as f:
            return f.get_tensor("lm_head.weight")

    def label_rows(self, lab):
        return self._W[list(lab)].double().numpy()

    def hidden_rows(self, token_lists):
        from vllm.inputs import TokensPrompt
        outs = self.llm.encode([TokensPrompt(prompt_token_ids=ids) for ids in token_lists], pooling_task="embed",
                               use_tqdm=False)
        return [o.outputs.data.float().cpu().numpy() for o in outs]

    def facts(self):
        import vllm
        return {"vllm": vllm.__version__, "mode": "hidden (pooling runner, last token)", "block_size": self.block_size,
                "pad_unit": self.pad_unit, "pad_where": self.pad_where if self.pad_unit else None,
                "lm_head": list(self._W.shape)}


class HFHiddenEngine(HiddenReadout):
    """The CPU stand-in: h from the HF model's final-norm hidden state at the last position, the label rows from its
    `lm_head` (float32 model). Not for measurement."""

    def __init__(self, model, pad_to="block", pad_where="front", block_size=64):
        import threading

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.mode, self.pad_token, self.pad_where = "hidden", 198, pad_where
        self.adapters = {}
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32).eval()
        self.block_size = self.match_unit = block_size
        self.pad_unit = None if not pad_to else (self.block_size if pad_to == "block" else int(pad_to))
        self._lock = threading.Lock()

    # the served row builder, unchanged
    _prepare_separate = LettersEngine._prepare_separate
    _template_tail = LettersEngine._template_tail
    _labels = LettersEngine._labels

    def label_rows(self, lab):
        return self.model.lm_head.weight[list(lab)].detach().double().numpy()

    def hidden_rows(self, token_lists):
        import torch
        with torch.no_grad():
            return [self.model.model(input_ids=torch.tensor([ids])).last_hidden_state[0, -1].float().numpy()
                    for ids in token_lists]

    def facts(self):
        return {"engine": "hf hidden (CPU stand-in, not for measurement)", "pad_unit": self.pad_unit}


class SingleEngineHidden(HiddenReadout):
    """h from the serving engine itself (docs/design/hidden-state-readout.md): the text engine runs decisio's
    hidden-readout model class, which writes [0, h] into d + 1 reserved logit columns; a head question is sent with
    exactly those ids allowed, and h is recovered from their log-probabilities (`recover_hidden_chunks`). No second copy
    of the weights. The label log-probabilities are recomputed from h and the output layer's label rows as the engine
    computes them (`label_logits`), so fitting and serving a head are unchanged."""

    def __init__(self, engine, model, start=None):
        from decisio.readout.letters import MAX_LABELS, label_token_ids, letter_labels
        from decisio.vllm_plugin.hidden import check_reserved, reserved_chunks
        self.engine, self.llm, self.tok, self._lock = engine, engine.llm, engine.tok, engine._lock
        self.mode = "hidden (serving engine, reserved logit columns)"
        self._W = HiddenEngine._load_lm_head(model)
        labels = label_token_ids(self.tok, [" yes", " no"] + [" " + c for c in letter_labels(self.tok, MAX_LABELS)])
        self.reserved = check_reserved(self._W.shape[1], self._W.shape[0], labels, start)
        self.chunks = reserved_chunks(self._W.shape[1], self.reserved[0])

    def _prepare_separate(self, state, questions):
        return self.engine._prepare_separate(state, questions)

    def label_rows(self, lab):
        return self._W[list(lab)].double().numpy()

    def label_logits(self, h, lab):
        """The label logits the engine itself computes from h: the product rounded to bfloat16, the dtype of the served
        model's logits (on a card, 2026-09-30: this reproduces the engine's label probabilities within 3.1e-8 on 250
        intent items; in float64 they differ by up to 0.019; runs/2026-09-30_plugin-verification)."""
        import torch
        z = torch.from_numpy(h.astype(np.float64) @ self.label_rows(lab).T)
        return z.float().to(torch.bfloat16).double().numpy()

    def hidden_rows(self, token_lists):
        """One engine request per chunk of reserved ids (vLLM allows at most 1,024 ids per request), sent one at a time:
        identical prompts in one batch are not the same forward on the FP8 stack (their label probabilities differed by
        up to 0.59 on a card, 2026-09-30), so a state stitched from one batch's rows is no forward's state. Sent one at
        a time, the chunks of a prompt are the same forward (they agreed to 1e-7) and share its prefix through the
        cache."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        from decisio.vllm_plugin.hidden import recover_hidden_chunks
        rows = []
        for ids in token_lists:
            lps = []
            for chunk in self.chunks:
                sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(chunk), allowed_token_ids=chunk,
                                    detokenize=False)
                o = self.llm.generate([TokensPrompt(prompt_token_ids=ids)], [sp], use_tqdm=False)[0]
                d = o.outputs[0].logprobs[0]
                lps.append([d[t].logprob for t in chunk])                  # KeyError = a reserved column went missing
            rows.append(recover_hidden_chunks(lps).astype(np.float32))
        return rows

    def facts(self):
        return {"mode": self.mode, "reserved_ids": [self.reserved[0], self.reserved[-1]],
                "lm_head": list(self._W.shape),
                "requests_per_question": len(self.chunks)}


def reserved_logprobs(h, start=0):
    """What vLLM returns for each chunk of reserved ids of a hidden-readout question: the float32 log-softmax of the
    chunk's columns of [0, h] (the masked logits of its allowed set). For the CPU stand-in and the tests."""
    from decisio.vllm_plugin.hidden import reserved_chunks
    full = np.concatenate([np.zeros(1, dtype=np.float32), np.asarray(h, dtype=np.float32)])
    out = []
    for chunk in reserved_chunks(len(full) - 1, start):
        z = full[[i - start for i in chunk]]
        m = z.max()
        out.append((z - (m + np.log(np.exp(z - m, dtype=np.float32).sum(dtype=np.float32)))).astype(np.float32))
    return out


class HFReservedHiddenEngine(HFHiddenEngine):
    """The CPU stand-in of the single-engine route: the HF model's h sent through the same reserved-column
    log-probabilities and recovery as on vLLM. Not for measurement."""

    def hidden_rows(self, token_lists):
        from decisio.vllm_plugin.hidden import recover_hidden_chunks
        return [recover_hidden_chunks(reserved_logprobs(h)).astype(np.float32)
                for h in super().hidden_rows(token_lists)]

    def facts(self):
        return {**super().facts(), "route": "reserved logit columns (emulated)"}
