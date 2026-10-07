# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Letters-readout serving on vLLM: one state, many typed questions, a probability distribution per
question, on vLLM's kernels.

Prompts are byte-identical to evaluation: each question is `readout.letters_prompt` in chat format
(`readout.chat_wrap(..., "chat")`), and the answer is the softmax over the question's label tokens at the
prompt's last position.

Two modes.

separate (default): one engine request per question, the state shared through vLLM's prefix cache.
    The shared prefix is found in token space from the state alone (state_prefix: text-space splits
    change the tokens at the boundary, and a split taken from the questions would move with them). On this
    hybrid, vLLM stores the Gated DeltaNet state only at block boundaries (or, with
    `prefix_match_unit`, at unit boundaries), so a prefix that ends mid-block is re-prefilled by every
    question. `pad_to` inserts pad tokens at the split so the shared prefix ends on a boundary; that
    changes the prompt, and what it does to accuracy is measured, not assumed. For a
    request with more than one question the prefix is prefilled once by a warm-up request of P+1
    tokens, which registers the state at P, before the questions are submitted. Labels are read by
    restricting the single generated token to the label tokens (`allowed_token_ids`) with
    `logprobs_mode="processed_logprobs"`: the K returned log-probabilities are then exactly the
    log-softmax over the label logits. Isolation is structural: every question is its own sequence.
    LoRA adapters are served here (vLLM LoRA; see `check_adapter_names` for the tensor names each class needs).

packed: questions are packed into one prompt as consecutive chat turns, each turn's assistant reply
    left at "Answer:", and the label logits are read at every turn's last position in one prefill,
    through vLLM's pooling runner with a classifier built from the output layer's label rows
    (`convert="classify"`, `classifier_from_token`, token-level pooling). The first question of a pack
    is byte-identical to its separate prompt; later ones see the earlier questions (not their
    answers), so isolation is not structural and must be measured. Packs are capped in questions and
    tokens; the state is repeated in each pack (the pooling runner has no prefix cache on hybrids).

    from decisio.serve.vllm_engine import LettersEngine
    eng = LettersEngine(os.environ["DECISIO_MODEL"], mode="separate", pad_to="block", engine_kw=...)
    out = eng.answer(state, [{"kind": "noul", "instructions": "Is this urgent?"}, ...])

    python -m decisio.serve.vllm_engine --model $DECISIO_MODEL --port 8000
    (the served default: the official checkpoint under decisio's hidden-readout class, front padding to the block,
    detokenize=False, DeepGEMM off, CUDA graphs captured up to 4,096 tokens; on vLLM 0.30.0, optionally with the
    suffix-staging patch series of patches/ and VLLM_SUFFIX_STAGING=1)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from decisio.readout.letters import (  # noqa: E402
    DEFAULT_FORMAT,
    LETTERS_INSTRUCTION,
    PromptFormat,
    allowed_ids,
    chat_turn,
    fmt_state,
    is_grouped,
    label_groups,
    label_logprobs,
    label_token_ids,
    letter_labels,
    question_body,
)


def options_listing(tok, options):
    """The listing a state carries when its questions use `options_in_state`."""
    labs = letter_labels(tok, len(options))
    return "Options:\n" + "\n".join(f"{lab}. {o}" for lab, o in zip(labs, options))


from decisio.names import SERVED_NAME, same_fingerprint, task_fingerprint  # noqa: E402
from decisio.serve.boundary import AfterResponse, Registrar, close_ticket, current_ticket, open_ticket  # noqa: E402
from decisio.serve.engine_health import EngineDead, EngineHealth, InFlight, guarded  # noqa: E402
from decisio.serve.temperature import SERVED_CHOICE_TEMPERATURE  # noqa: E402
from decisio.vllm_plugin.worker import QUALNAME as WORKER_EXTENSION  # noqa: E402

SERVED_ENGINE = {"compilation_config": {"max_cudagraph_capture_size": 4096}}


def run_or_die(main):
    """Run main(); on an exception print it and exit hard. A raised exception otherwise leaves the
    process waiting on vLLM's engine subprocess with the GPU held (seen: a crashed gate hung for
    ten minutes at 88 GB until killed by PID)."""
    import os
    import traceback

    try:
        return main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # os._exit alone orphans vLLM's engine-core child with the GPU held (seen: 88 GB after a crash)
        try:
            import psutil

            kids = psutil.Process().children(recursive=True)
            for k in kids:
                k.kill()
            psutil.wait_procs(kids, timeout=10)
        except Exception:
            pass
        os._exit(1)


PAD_PLACES = ("between", "front", "user")
USER_HEAD = "<|im_start|>user\n"


def pad_prompt(ids, n, k, token, where, head_len):
    """Insert k pad tokens so a prompt whose shared prefix is ids[:n] gets a prefix of n + k tokens.

    between: after the state's blank line, before the question (the first placement; it moved 9.4% of
             top answers against the unpadded prompt).
    front:   at position 0, before the chat template: a fixed-length preamble sized so the state ends
             on a block boundary.
    user:    at the start of the user turn, just before the state.
    The prefix length (and so the cache hit) is the same in all three; only what the model reads differs.
    """
    if not k:
        return ids
    at = {"between": n, "front": 0, "user": head_len}[where]
    return ids[:at] + [token] * k + ids[at:]


TURN_SEP = "<|im_end|>\n"  # closes an assistant turn left at "Answer:" (packed mode)
PAD_TOKEN = 198  # "\n"


def check_adapter_names(model, adapters, arch=None):
    """Refuse an adapter whose tensor names the served class would not map: vLLM then loads it as a
    silent no-op. The multimodal class (Qwen3_5MoeForConditionalGeneration) needs
    `...model.language_model.layers...`; the text classes (Qwen3_5MoeForCausalLM and decisio's) need
    PEFT's original `...model.layers...`."""
    # decisio's registered classes are text classes (the trainer's original names), whatever config.json names
    arch = (
        "ForCausalLM" if arch else json.loads((Path(model) / "config.json").read_text()).get("architectures", [""])[0]
    )
    for name, path in adapters.items():
        with open(Path(path) / "adapter_model.safetensors", "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            keys = [k for k in json.loads(f.read(n)) if k != "__metadata__"]
        converted = all(".language_model.layers." in k for k in keys)
        original = all(k.startswith("base_model.model.model.layers.") for k in keys)
        if arch.endswith("ForConditionalGeneration") and not converted:
            raise ValueError(f"adapter {name}: {arch} needs names under `.language_model.layers.`")
        if arch.endswith("ForCausalLM") and not original:
            raise ValueError(f"adapter {name}: {arch} needs PEFT's original names (`base_model.model.model.layers.`)")


def state_prefix(tok, body, fmt=DEFAULT_FORMAT):
    """The tokens every question about this state starts with, found from the state alone.

    Qwen's pre-tokenizer merges trailing punctuation with the blank line after it (".\n\n" is one
    token), so the state's end is not a token boundary. Two prompts that differ only in the first
    letter of the question share exactly the tokens up to the question; a question's own first
    character never merges back into them unless it is whitespace, which _prepare_separate checks.
    Depending on the state alone matters: a split computed from the request's questions (their longest
    common prefix) would move with the question mix, and so would any padding inserted there.
    """
    a = tok.encode(user_turn(tok, f"{body}\n\nA", fmt), add_special_tokens=False)
    b = tok.encode(user_turn(tok, f"{body}\n\nB", fmt), add_special_tokens=False)
    n = 0
    while a[n] == b[n]:
        n += 1
    return n, a[:n]


def question_text(tok, q, fmt=DEFAULT_FORMAT):
    """A question's own text (no state), as the prompt format renders it after the state (question_body; under the
    default format exactly what letters_prompt writes), and its labels.

    `options_in_state`: the option listing ("Options:\nA. ...", options_listing) is already in the state,
    so the question is its instruction alone; the labels are the same letters (the shared option
    listing layout)."""
    kind = q.get("kind", "choice")
    if q.get("options_in_state"):
        labs = letter_labels(tok, len(q["options"]))
        return f"{q['instructions']}\n{LETTERS_INSTRUCTION}", [" " + x for x in labs], list(q["options"])
    options = ["yes", "no"] if kind == "noul" else list(q["options"])
    text, cands = question_body(tok, kind, q["instructions"], options, fmt)
    if kind == "noul" and q.get("noul_order") == "no_yes":
        # two-order mode's second yes/no branch (systemone.py, after Reflex): the answers named the other way round;
        # the labels and their order are unchanged, so the distribution stays [P(yes), P(no)]
        for said, swapped in NOUL_SWAPS:
            if text.endswith(said):
                text = text[: -len(said)] + swapped
                break
        else:
            raise AssertionError("the yes/no prompt changed; update the second branch")
    return text, cands, options


# the yes/no answer line of each prompt tail, and the same line with the answers named the other way round
NOUL_SWAPS = (
    ("\nAnswer yes or no.", "\nAnswer no or yes."),
    ("\n\nAnswer with yes or no, and nothing else:", "\n\nAnswer with no or yes, and nothing else:"),
)


def user_turn(tok, content, fmt=DEFAULT_FORMAT):
    """One question's chat prompt (decisio.readout.letters.chat_turn); under the default format exactly as evaluation
    renders a letters prompt (thinking disabled, "Answer:")."""
    return chat_turn(tok, content, fmt)


class LettersEngine:
    # the served defaults (--pad-policy, --multi-question); the server sets both from its flags
    pad_policy = "always"
    multi_question = "sequential"
    # where a single question's state boundary is registered on a base that registers it (--register-boundary):
    # after, once the response is out (decisio.serve.boundary); before, ahead of the question, as 0.8.1 did; off, never
    # (a later question about the state reads it again, as before 0.8.1)
    register_boundary = "after"
    # the server's EngineHealth (decisio.serve.engine_health), shared with the other engines; None in library use
    health = None
    fmt = DEFAULT_FORMAT  # the prompt format (decisio.readout.letters.PromptFormat); the server sets it from its flags

    def __init__(
        self,
        model,
        mode="separate",
        pad_to=None,
        pad_token=PAD_TOKEN,
        pad_where="between",
        adapters=None,
        max_labels=77,
        pack=16,
        max_pack_tokens=16384,
        max_model_len=32768,
        max_num_seqs=256,
        gpu_memory_utilization=0.90,
        engine_kw=None,
        fmt=None,
        family=None,
        revision=None,
    ):
        from vllm import LLM

        from decisio.families import family_of

        # where the engine runs (--engine-process), as vLLM will read it when the LLM below is built
        self.engine_process = "in" if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0" else "separate"
        self.family = family or family_of(model, revision)
        self.register_boundary = resolve_register_boundary(self.family, None)
        self.model_name, self.revision = model, revision
        self.fmt = fmt or DEFAULT_FORMAT
        if mode == "packed" and not self.fmt.is_default():
            raise ValueError("packed mode reads the compact layout only (PromptFormat())")
        self.mode, self.pad_token, self.pack, self.max_pack_tokens = mode, pad_token, pack, max_pack_tokens
        if pad_where not in PAD_PLACES:
            raise ValueError(f"pad_where must be one of {PAD_PLACES}")
        self.pad_where = pad_where
        self.adapters = {}
        kw = dict(
            model=model,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt=dict(self.family.limit_mm),
            worker_extension_cls=WORKER_EXTENSION,
            # the base's precision on load (the 31B: FP8); --engine's own "quantization" overrides it
            **({"quantization": self.family.quantization} if self.family.quantization else {}),
            **({"revision": revision, "tokenizer_revision": revision} if revision else {}),
        )
        if mode == "separate":
            # several forms per label (--label-variants summed) read up to 4 x 255 ids per question
            max_lp = 256 if self.fmt.variants == "single" else 1024
            kw.update(enable_prefix_caching=True, max_logprobs=max_lp, logprobs_mode="processed_logprobs")
            if adapters:
                kw.update(enable_lora=True, max_loras=len(adapters), max_lora_rank=64)
        elif mode == "packed":
            from transformers import AutoTokenizer
            from vllm.config import PoolerConfig

            t = AutoTokenizer.from_pretrained(model, revision=revision)
            self.label_ids = sorted(
                set(label_token_ids(t, [" yes", " no"] + [" " + c for c in letter_labels(t, max_labels)]))
            )
            kw.update(
                runner="pooling",
                convert="classify",
                hf_overrides={
                    "text_config": {
                        "classifier_from_token": t.convert_ids_to_tokens(self.label_ids),
                        "method": "no_post_processing",
                    }
                },
                pooler_config=PoolerConfig(
                    task="token_classify", seq_pooling_type="LAST", tok_pooling_type="ALL", use_activation=False
                ),
            )
            self.column = {tid: c for c, tid in enumerate(self.label_ids)}
        else:
            raise ValueError(f"unknown mode {mode!r}")
        kw.update(engine_kw or {})
        self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        cfg = self.llm.llm_engine.vllm_config.cache_config
        self.block_size = cfg.block_size
        self.match_unit = getattr(cfg, "prefix_match_unit", None) or self.block_size
        # what /health reports of the prefix cache (padding keeps match_unit): vLLM's hit unit, the least common
        # multiple of its KV cache groups' block sizes (64 on Gemma 4, whose groups are 16 and 64), and its hash step
        sizes = self.llm.collective_rpc("decisio_kv_block_sizes")[0]
        self.cache_hit_unit, self.hash_unit = sizes or (None, None)
        self.pad_unit = None if not pad_to else (self.block_size if pad_to == "block" else int(pad_to))
        if adapters:
            from vllm.lora.request import LoRARequest

            check_adapter_names(
                model, adapters, arch=((engine_kw or {}).get("hf_overrides") or {}).get("architectures")
            )
            for i, (name, path) in enumerate(adapters.items(), start=1):
                self.adapters[name] = LoRARequest(name, i, str(path))
        self._lock = threading.Lock()  # one request at a time through the engine
        self._warm_up()

    def _warm_up(self):
        """Serve two throwaway requests before any real one. On the H200 the first requests after start-up
        ran on kernels still being JIT-compiled and answered differently from later identical requests
        (answers moved by 5.7e-2 between the first and a later identical request; bit-exact once warm), so no
        caller should ever get a first-request answer. Every loaded adapter too: its LoRA MoE kernel is
        JIT-compiled on the adapter's own first request (`_fused_moe_lora_one_shot_kernel`)."""
        q = [{"kind": "noul", "instructions": "Is this a warm-up request?"}]
        state = "Warm-up. " * 40
        for _ in range(2):
            if self.mode == "separate":
                for adapter in [None, *self.adapters]:
                    self.score_prompts(self._prepare_separate(state, q)[0], adapter)
            else:
                self.score_packed(
                    [[(f"{state}\n\n{question_text(self.tok, q[0])[0]}", label_token_ids(self.tok, [" yes", " no"]))]]
                )

    def probe_ids(self):
        """The probe's prompt (decisio.serve.engine_health): one token."""
        return self.tok.encode("ok", add_special_tokens=False)[:1]

    def probe(self):
        """The smallest call this engine serves, to tell a failed request from a dead engine (engine_health): one
        forward pass over a one-token prompt, as the warm-up's call (a pooling request in packed mode). Prompts shorter
        than a cache block leave the prefix cache as it was."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        prompt = [TokensPrompt(prompt_token_ids=self.probe_ids())]
        if self.mode == "packed":
            self.llm.encode(prompt, pooling_task="token_classify", use_tqdm=False)
        else:
            self.llm.generate(prompt, SamplingParams(max_tokens=1, temperature=0.0, detokenize=False), use_tqdm=False)

    def facts(self):
        import torch
        import vllm

        c = self.llm.llm_engine.vllm_config
        return {
            "vllm": vllm.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "engine_process": self.engine_process,
            "mode": self.mode,
            "block_size": self.block_size,
            "match_unit": self.match_unit,
            "cache_hit_unit": self.cache_hit_unit,
            "hash_unit": self.hash_unit,
            # a single question registers its state's boundary for the next question (decisio.families)
            "registers_state_boundary": facts_register_boundary(self) in ("after", "before"),
            "register_boundary": facts_register_boundary(self),
            **({"repository": self.repository} if getattr(self, "repository", None) else {}),
            "pad_unit": self.pad_unit,
            "pad_where": self.pad_where if self.pad_unit else None,
            "quantization": str(c.model_config.quantization),
            "kv_cache_dtype": str(c.cache_config.cache_dtype),
            "mamba_ssm_cache_dtype": str(c.cache_config.mamba_ssm_cache_dtype),
            "adapters": sorted(self.adapters),
            "VLLM_USE_DEEP_GEMM": os.environ.get("VLLM_USE_DEEP_GEMM", "unset"),
            "prompt_format": self.fmt.facts(),
            "base": self.family.key,
            "attention_backend": str(getattr(getattr(c, "attention_config", None), "backend", None)),
        }

    # ---- the request API ------------------------------------------------------------------------

    def answer(self, state, questions, adapter=None):
        """One distribution per question (numpy arrays, in option order), plus timing and cache facts.

        A yes/no question may carry `"derive": {"from": i, "option": o}` to be answered from Choice
        question i's distribution (P(yes) = P(o)): exact agreement by construction.
        """
        probs, info = self.answer_many([(state, questions)], adapter)
        return probs[0], info

    def answer_many(self, requests, adapter=None):
        """Several independent requests [(state, questions)] through the engine together: every
        request's state is prefilled in one batch, then every question. What a loaded server does with
        concurrent requests, so its throughput is what the cost per token is computed from."""
        todo, derived = [], []
        for _, questions in requests:
            derived.append({i: q["derive"] for i, q in enumerate(questions) if q.get("derive")})
            todo.append([i for i in range(len(questions)) if i not in derived[-1]])
        if self.health is not None:
            self.health.check()  # refused at once when the engine is dead, before any work or the lock
        # the boundary registrations this call defers become due when the server has sent its response, or, without a
        # server, when this call returns (decisio.serve.boundary)
        own = current_ticket() is None
        if own:
            ticket, token = open_ticket()
        try:
            with self._lock:
                t0 = time.perf_counter()
                # registrations already due go before this request's questions, so a question sent after an answer
                # about the same state reads the state from the cache
                registrar = self.__dict__.get("_registrar")
                ran, ran_ms = registrar.run_due() if registrar is not None else (0, 0.0)
                reqs = [(st, [qs[i] for i in td]) for (st, qs), td in zip(requests, todo)]
                if self.mode == "separate":
                    probs, info = self._answer_separate(reqs, adapter)
                else:
                    if adapter:
                        raise ValueError("adapters are served in separate mode")
                    probs, info = self._answer_packed(reqs)
                if ran:
                    info.setdefault("state_boundary", {}).update(ran_before=ran, ran_before_ms=ran_ms)
                info["server_ms"] = (time.perf_counter() - t0) * 1000
        finally:
            if own:
                close_ticket(ticket, token)
        results = []
        for (_, questions), td, der, pr in zip(requests, todo, derived, probs):
            out = dict(zip(td, pr))
            for i, d in der.items():
                src = questions[d["from"]]
                p = float(out[d["from"]][list(src["options"]).index(d["option"])])
                out[i] = np.array([p, 1.0 - p])
            results.append([out[i] for i in range(len(questions))])
        return results, info

    # ---- separate mode --------------------------------------------------------------------------

    def _template_tail(self):
        """The chat-template text after a user turn's content (constant), rendered once."""
        if getattr(self, "_tail", None) is None or getattr(self, "_tail_fmt", None) != self.fmt:
            mark = "\ue000"
            full = user_turn(self.tok, mark, self.fmt)
            self._head_text, self._tail = full.split(mark)
            self._tail_fmt = self.fmt
        return self._tail

    def _labels(self, cands):
        cache = self.__dict__.setdefault("_label_cache", {})
        key = (tuple(cands), self.fmt)
        if key not in cache:
            cache[key] = label_groups(self.tok, cands, self.fmt)
        return cache[key]

    def _prepare_separate(self, state, questions):
        """Token rows for one request. The state is tokenized once per request, not once per question:
        each row is the state prefix (up to and including the blank line after it, state_prefix) plus
        the question's own text and the template tail, tokenized alone. Full-prompt tokenization per
        question cost 0.6 ms (500-token state) to 5.7 ms (8,000) of CPU per question.
        The first row of every request is checked against tokenizing its whole prompt; a mismatch falls
        back to full tokenization for that request, so the rows are always what evaluation scores."""
        enc = lambda s: self.tok.encode(s, add_special_tokens=False)  # noqa: E731
        body = fmt_state(state)
        tail = self._template_tail()
        texts = [question_text(self.tok, q, self.fmt) for q in questions]
        suffixes = [enc(text + tail) for text, _, _ in texts]
        # one tokenization of the state (three cost 87 ms at 8,000 tokens): the first question's
        # whole prompt, split where its own tokens begin. That boundary is the state's (the state_prefix
        # argument: a question's first character never merges back unless it is whitespace); if the whole
        # prompt does not end with the question's own tokens, fall back to the dummy-question split
        full0 = enc(user_turn(self.tok, f"{body}\n\n{texts[0][0]}", self.fmt))
        n = len(full0) - len(suffixes[0])
        if n > 0 and full0[n:] == suffixes[0]:
            prefix = full0[:n]
        else:
            n, prefix = state_prefix(self.tok, body, self.fmt)
        rows = [(prefix + sfx, self._labels(cands)) for sfx, (_, cands, _) in zip(suffixes, texts)]
        if rows[0][0] != full0:
            rows = [
                (enc(user_turn(self.tok, f"{body}\n\n{text}", self.fmt)), self._labels(cands))
                for text, cands, _ in texts
            ]
        for ids, _ in rows:
            if ids[:n] != prefix:
                raise ValueError(
                    "a question's text merges with the state's last token; questions must not start with whitespace"
                )
        k = (-n % self.pad_unit) if self.pad_unit else 0
        if self.pad_policy == "none" or (self.pad_policy == "shared" and len(questions) == 1):
            k = 0  # none: never pad; shared: no second question can reuse the padded boundary
        elif self.pad_policy == "row" and len(questions) == 1 and self.pad_unit:
            # --pad-policy row: the whole row, state and question, ends on a block boundary, so a cold prefill is
            # one engine step (align mode stops a prefill at the last boundary, and a row just past one costs a
            # second step); the state's boundary is then not aligned, which no second question would use anyway
            k = -len(rows[0][0]) % self.pad_unit
        head = len(enc(USER_HEAD))
        rows = [(pad_prompt(ids, n, k, self.pad_token, self.pad_where, head), lab) for ids, lab in rows]
        return rows, n + k

    def score_prompts(self, rows, adapter=None, warm=(), skip_cache=False, mm=None, mm_uuids=None):
        """Label distributions for fully-built prompts [(token ids, label ids)], one engine request each.

        `mm`: multimodal data shared by every row and warm-up ({"image": [PIL images]}), for the image engine
        (decisio.serve.image_engine); each image appears once in the token ids as one `<|image_pad|>`, which the
        engine expands to the image's token count. None for text (the default, unchanged). `mm_uuids`: a content id per
        image ({"image": [str]}), so the engine hashes no image itself (every row carries the same images).

        `warm`: token lists prefilled first, in one batch, so their recurrent states are registered
        before any question is scheduled (a warm-up of P+1 tokens registers the state at P).
        `skip_cache`: the requests do not read the prefix cache (each computes its whole prompt), which is
        how a cold batch is made in an engine that has cached blocks: a cache reset is not enough, as
        vLLM runs one request first and the others then hit its freshly cached prefix."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        lora = self.adapters[adapter] if adapter else None
        extra = {"multi_modal_data": mm, **({"multi_modal_uuids": mm_uuids} if mm_uuids else {})} if mm else {}
        warm_ms = self.send_warm(warm, adapter, mm, mm_uuids) if warm else 0.0
        # SamplingParams defaults leave top-k/top-p/min-p off, so "processed" is the masked logits alone.
        # detokenize=False: labels are read by token id and no text is used; the per-request detokenizer
        # was 73% of the frontend's CPU at an 8,000-token state (profiled)
        sps = [
            SamplingParams(
                max_tokens=1,
                temperature=0.0,
                logprobs=len(allowed_ids(lab)),
                allowed_token_ids=allowed_ids(lab),
                skip_reading_prefix_cache=skip_cache or None,
                detokenize=False,
            )
            for _, lab in rows
        ]
        t = time.perf_counter()
        outs = guarded(
            self,
            self.llm.generate,
            [TokensPrompt(prompt_token_ids=ids, **extra) for ids, _ in rows],
            sps,
            lora_request=lora,
            use_tqdm=False,
        )
        questions_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        probs, token_lps = [], []
        for (_, lab), o in zip(rows, outs):
            d = o.outputs[0].logprobs[0]
            lp = label_logprobs(lab, lambda t: d[t].logprob)  # KeyError = a label went missing
            if is_grouped(lab):  # each form's own log-probability, for the debug readout (verification)
                token_lps.append([[d[t].logprob for t in g] for g in lab])
            p = np.exp(lp - lp.max())
            probs.append(p / p.sum())
        cached = [o.num_cached_tokens or 0 for o in outs]
        engine_len = [len(o.prompt_token_ids or []) for o in outs]
        # per-request engine-core timestamps (monotonic), present when the engine keeps stats
        # (disable_log_stats=False): queue wait and schedule-to-first-token, for latency attribution
        timing = [
            {
                "queue_ms": (m.scheduled_ts - m.queued_ts) * 1000,
                "sched_to_token_ms": (m.first_token_ts - m.scheduled_ts) * 1000,
            }
            for m in (getattr(o, "metrics", None) for o in outs)
            if m is not None and m.scheduled_ts
        ]
        return probs, {
            "warm_ms": warm_ms,
            "questions_ms": questions_ms,
            "readout_ms": (time.perf_counter() - t) * 1000,
            "cached_tokens_mean": float(np.mean(cached)),
            "engine_timing": timing,
            "prompt_tokens_mean": float(np.mean([len(r[0]) for r in rows])),
            "prompt_tokens": sum(len(r[0]) for r in rows) + sum(len(w) for w in warm),
            "engine_prompt_tokens": engine_len,
            "cached_tokens": cached,
            **({"label_token_logprobs": token_lps} if token_lps else {}),
        }

    def send_warm(self, warm, adapter=None, mm=None, mm_uuids=None):
        """Prefill token lists in one batch, one generated token each, so their states are registered in the prefix
        cache (a warm-up of P+1 tokens registers the state at P); the milliseconds it took."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        lora = self.adapters[adapter] if adapter else None
        extra = {"multi_modal_data": mm, **({"multi_modal_uuids": mm_uuids} if mm_uuids else {})} if mm else {}
        t = time.perf_counter()
        guarded(
            self,
            self.llm.generate,
            [TokensPrompt(prompt_token_ids=w, **extra) for w in warm],
            SamplingParams(max_tokens=1, temperature=0.0),
            lora_request=lora,
            use_tqdm=False,
        )
        return (time.perf_counter() - t) * 1000

    # the states whose boundary this engine registered with a warm-up (register_state_boundary), most recent last
    BOUNDARIES_KEPT = 4096

    def boundaries_registered(self, keys):
        """Remember states whose boundary a warm-up registered (the engine's lock held)."""
        for key in keys:
            self._boundaries[key] = True
            self._boundaries.move_to_end(key)
        while len(self._boundaries) > self.BOUNDARIES_KEPT:
            self._boundaries.popitem(last=False)

    @property
    def registrar(self):
        """The boundary registrations deferred until after their responses (decisio.serve.boundary)."""
        if "_registrar" not in self.__dict__:
            self._registrar = Registrar(self)
        return self._registrar

    def _boundary_key(self, ids, adapter):
        data = np.asarray(ids, dtype=np.int64).tobytes() + str(adapter).encode()
        return hashlib.blake2b(data, digest_size=16).digest()

    def _answer_separate(self, requests, adapter):
        rows, warm, spans, shared = [], [], [], []
        unit = min(self.match_unit, self.pad_unit or self.match_unit)
        # a single question registers its state's boundary too, on a base whose requests would not keep it otherwise
        # (decisio.families, register_state_boundary); the hit unit tells whether a later request found it
        family = getattr(self, "family", None)
        hit = getattr(self, "cache_hit_unit", None)
        mode = getattr(self, "register_boundary", "after")
        register = bool(getattr(family, "register_state_boundary", False) and hit and mode != "off")
        after = mode == "after"
        # (key) for the warm-ups sent; (key, row, whole hit units) for the others; (key, warm-up) for those deferred
        registering, checking, deferring = [], [], []
        t = time.perf_counter()
        for state, questions in requests:
            r, P = self._prepare_separate(state, questions)
            # a warm-up pays only when several questions can reuse a registered boundary; --multi-question batch
            # sends the questions without it, each prefilling the state itself, in one engine call
            if len(r) > 1 and P >= unit and self.multi_question != "batch":
                warm.append(r[0][0][: P + 1])
            elif register and len(r) == 1 and P >= unit and self.multi_question != "batch":
                # one question: the warm-up only when this engine has not registered the state's boundary, or a
                # request since found it gone (evicted), so a repeated question costs no extra engine call
                key = self._boundary_key(r[0][0][:P], adapter)
                if key in self._boundaries:
                    self._boundaries.move_to_end(key)
                    checking.append((key, len(rows), (P // hit) * hit))
                elif after and not self.registrar.take(key):
                    # --register-boundary after: answered without the warm-up, which is sent once the response is out
                    deferring.append((key, r[0][0][: P + 1]))
                else:
                    # before (0.8.1), or this state's warm-up is still pending behind a response not yet sent (two
                    # questions about a new state sent together): sent ahead of the question
                    warm.append(r[0][0][: P + 1])
                    registering.append(key)
            spans.append((len(rows), len(rows) + len(r)))
            rows += r
            shared.append(P)
        prepare_ms = (time.perf_counter() - t) * 1000
        if self.multi_question == "sequential" and len(rows) > 1:
            probs, info = self._score_one_at_a_time(rows, adapter, warm)
        else:
            probs, info = self.score_prompts(rows, adapter, warm)
        if register:
            cached = info.get("cached_tokens") or []
            for key, row, whole in checking:
                if row < len(cached) and cached[row] < whole:
                    self._boundaries.pop(key, None)  # evicted: the next request on this state registers it again
            self.boundaries_registered(registering)
            for key, w in deferring:
                self.registrar.defer(key, w, adapter)
            info["state_boundary"] = {
                "registered": len(registering),
                "found": len(checking),
                "deferred": len(deferring),
            }
        info.update(shared_prefix_tokens=shared[0] if len(shared) == 1 else shared, questions=len(rows))
        info["prepare_ms"] = prepare_ms
        return [probs[a:b] for a, b in spans], info

    @property
    def _boundaries(self):
        if not hasattr(self, "_boundary_lru"):
            self._boundary_lru = OrderedDict()
        return self._boundary_lru

    def _score_one_at_a_time(self, rows, adapter, warm):
        """--multi-question sequential: the warm-up, then each question in its own engine call, so no question shares
        a forward pass with another and its answer cannot depend on what else the request asked."""
        probs, infos = [], []
        for i, row in enumerate(rows):
            p, info = self.score_prompts([row], adapter, warm if i == 0 else ())
            probs += p
            infos.append(info)
        merged = dict(infos[0])
        merged["questions_ms"] = sum(x.get("questions_ms", 0.0) for x in infos)
        merged["readout_ms"] = sum(x.get("readout_ms", 0.0) for x in infos)
        for k in ("engine_timing", "engine_prompt_tokens", "cached_tokens", "label_token_logprobs"):
            merged[k] = [v for x in infos for v in x.get(k, [])]
        merged["prompt_tokens"] = sum(x.get("prompt_tokens", 0) for x in infos)
        merged["cached_tokens_mean"] = float(np.mean(merged["cached_tokens"])) if merged["cached_tokens"] else 0.0
        return probs, merged

    # ---- packed mode ----------------------------------------------------------------------------

    def score_packed(self, packs):
        """packs: list of packs, each a list of turns (user content, label ids). Returns, per pack,
        one distribution per turn, read at the turn's final "Answer:" token, in one prefill per pack."""
        from vllm.inputs import TokensPrompt

        enc = lambda s: self.tok.encode(s, add_special_tokens=False)  # noqa: E731
        prompts, reads = [], []
        for turns in packs:
            ids, pos = [], []
            for j, (content, _) in enumerate(turns):
                # every piece after the first starts with a special token, so encoding piecewise is
                # identical to encoding the whole pack, and the first turn is exactly its separate prompt
                ids += enc((TURN_SEP if j else "") + user_turn(self.tok, content))
                pos.append(len(ids) - 1)
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            reads.append(pos)
        outs = guarded(self, self.llm.encode, prompts, pooling_task="token_classify", use_tqdm=False)
        result = []
        for turns, pos, o in zip(packs, reads, outs):
            logits = o.outputs.data  # [prompt length, K classifier tokens]
            pack_probs = []
            for (_, lab), p in zip(turns, pos):
                z = logits[p, [self.column[t] for t in lab]].double().cpu().numpy()
                e = np.exp(z - z.max())
                pack_probs.append(e / e.sum())
            result.append(pack_probs)
        tokens = [len(p["prompt_token_ids"]) for p in prompts]
        return result, {"pack_tokens": tokens, "prompt_tokens": sum(tokens)}

    def _packs_for(self, state, questions):
        body = fmt_state(state)
        state_tokens = len(self.tok.encode(body, add_special_tokens=False))
        packs, cur, cur_tok = [], [], 0
        for q in questions:
            text, cands, _ = question_text(self.tok, q)
            n = len(self.tok.encode(text, add_special_tokens=False)) + 24  # + chat framing
            if cur and (len(cur) >= self.pack or state_tokens + cur_tok + n > self.max_pack_tokens):
                packs.append(cur)
                cur, cur_tok = [], 0
            cur.append((text, label_token_ids(self.tok, cands)))
            cur_tok += n
        if cur:
            packs.append(cur)
        # the state opens each pack's first turn
        return [[(f"{body}\n\n{p[0][0]}", p[0][1])] + p[1:] for p in packs]

    def _answer_packed(self, requests):
        packs, spans = [], []
        for state, questions in requests:
            pk = self._packs_for(state, questions)
            spans.append((len(packs), len(packs) + len(pk)))
            packs += pk
        res, info = self.score_packed(packs)
        info.update(packs=len(packs), questions=sum(len(q) for _, q in requests))
        return [[p for pack in res[a:b] for p in pack] for a, b in spans], info


# ---- HTTP endpoint ------------------------------------------------------------------------------


def make_app(engine, systemone=None, health=None):
    """`/health` and `/v1/answer`; with `systemone` (decisio.serve.systemone.SystemOne) also TypeSafe's wire format,
    `POST /v1/systemone` and `GET /v1/models`, on the same engine. With `health` (decisio.serve.engine_health; by
    default the engine's own), a dead engine makes `/health` answer 503 and every request 503 with the reason."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    health = health if health is not None else getattr(engine, "health", None)

    class Request(BaseModel):
        state: str | dict
        questions: list[dict]
        adapter: str | None = None

    app = FastAPI(title="letters readout")
    # boundary registrations deferred by a request wait for its response (decisio.serve.boundary)
    app.add_middleware(AfterResponse)
    if health is not None:  # the exit after a death waits for the answers in flight (decisio.serve.engine_health)
        app.add_middleware(InFlight, health=health)

    @app.exception_handler(EngineDead)
    def engine_dead(request, exc):
        code = health.exit_code if health is not None else None
        return JSONResponse(
            status_code=503,
            content={
                "detail": f"the engine is dead ({exc}); this server is exiting"
                + (f" with code {code}" if code is not None else "")
                + " so that its restart policy starts a new one"
            },
        )

    @app.get("/health")
    def health_route():
        if health is not None and health.dead:
            return JSONResponse(
                status_code=503,
                content={"ok": False, "engine": "dead", "reason": health.reason, "exit_code": health.exit_code},
            )
        image = getattr(systemone, "image_engine", None) if systemone is not None else None
        so = (
            {
                "systemone": {
                    "hide_index_keys": systemone.hide_index_keys,
                    "desnake_labels": systemone.desnake_labels,
                    "describe_options": systemone.describe_options,
                    "noul_rendering": systemone.noul_rendering,
                    "noul_commit": systemone.noul_commit,
                    "abstention": systemone.abstention,
                    "abstention_tasks": {
                        t["id"]: (t.get("config") or {}).get("threshold") for t in systemone.tasks.values()
                    },
                    "orders": systemone.orders,
                    "abstain_option": systemone.abstain_option,
                    "tasks": systemone.tasks_enabled,
                    "debug_readout": systemone.debug_readout,
                    "temperature": systemone.temperature,
                    "temperatures": systemone.temperatures,
                    "readout_tasks": {
                        t["id"]: {
                            "calibration": bool(t["calibration"].get("applied")),
                            "head": bool(t["head"].get("applied")),
                        }
                        for t in (systemone.task_store.by_key.values() if systemone.task_store else [])
                    },
                }
            }
            if systemone is not None
            else {}
        )
        hidden = getattr(systemone, "hidden_engine", None) if systemone is not None else None
        so.update({"head_engine": hidden.facts()} if hidden is not None else {})
        # registrations deferred until after their responses: pending, and counted (decisio.serve.boundary)
        if facts_register_boundary(engine) == "after" and getattr(engine, "cache_hit_unit", None):
            so["state_boundary"] = engine.registrar.facts()
        return {
            "ok": True,
            **engine.facts(),
            "prompt_format": getattr(engine, "fmt", DEFAULT_FORMAT).facts(),
            **({"base": engine.family.key} if getattr(engine, "family", None) is not None else {}),
            # what this server serves, in one block a run record can quote: the base profile, the temperature each
            # question type gets, and the prompt (decisio.families)
            **({"profile": served_profile(engine, systemone)} if systemone is not None else {}),
            **({"image_engine": image.facts()} if image is not None else {}),
            **so,
        }

    def answer(req: Request):  # sync: FastAPI runs it in a thread; the engine lock serialises
        if req.adapter and req.adapter not in engine.adapters:
            raise HTTPException(404, f"unknown adapter {req.adapter!r}; loaded: {sorted(engine.adapters)}")
        try:
            probs, info = engine.answer(req.state, req.questions, req.adapter)
        except (KeyError, ValueError, AssertionError) as e:
            raise HTTPException(400, str(e))
        answers = []
        for q, p in zip(req.questions, probs):
            opts = ["yes", "no"] if q.get("kind", "choice") == "noul" else list(q["options"])
            answers.append({"options": opts, "probs": [float(x) for x in p]})
        return {"answers": answers, "timing": info}

    # this module has `from __future__ import annotations`, so `req: Request` is the string "Request", which FastAPI
    # cannot resolve for a class local to make_app and reads as a missing query parameter: every POST /v1/answer got
    # a 422 (found by the CPU smoke test; the benches call the engine in-process). Give it the class.
    answer.__annotations__["req"] = Request
    app.post("/v1/answer")(answer)

    if systemone is not None:
        from decisio.serve.systemone import add_routes

        add_routes(app, systemone)
    return app


def deep_gemm_guard(backend, environ, allow=False):
    """The served default runs the FP8 MoE on Triton: vLLM's DeepGEMM path gave wrong FP8 results on a Blackwell card
    (RTX PRO 6000). Before vLLM is imported: set VLLM_USE_DEEP_GEMM=0 when it is unset, and refuse to start when it is
    set to anything else, unless the operator allows it (a card where DeepGEMM was verified)."""
    if backend != "vllm":
        return
    value = environ.get("VLLM_USE_DEEP_GEMM")
    if value is None:
        environ["VLLM_USE_DEEP_GEMM"] = "0"
    elif value != "0" and not allow:
        raise SystemExit(
            f"VLLM_USE_DEEP_GEMM={value}: the served default needs 0 (the FP8 MoE on Triton; DeepGEMM is "
            "wrong on Blackwell cards). Unset it, or pass --allow-deep-gemm on a card where it was "
            "verified."
        )


ENGINE_PROCESSES = ("in", "separate")


def engine_process_guard(backend, choice, environ, second_engines=()):
    """--engine-process: where vLLM's engine runs, resolved once. `in` runs it in the server's process
    (VLLM_ENABLE_V1_MULTIPROCESSING=0), where every question of a request is added before the first engine step, so
    --multi-question warm repeats exactly (vllm-project/vllm#59764; EVAL_CARD.md section 4); `separate` is vLLM's own
    arrangement, a process of its own, which starts a step with whatever has arrived, so a batched request's questions
    do not always share a step and warm can move between repeats.

    The default (choice None) resolves to `in` for a single-engine server and to `separate` when a second engine is
    configured (`second_engines`: two vLLM engines in one process failed every request on the card, so `in` is refused
    with one), or when the environment already asks vLLM for its engine process (VLLM_ENABLE_V1_MULTIPROCESSING=1).
    Gates for the default: runs/2026-10-05_engine-death-gates. Before vLLM reads it, sets the variable for `in`;
    refuses an explicit `separate` when the variable already says 0, so /health never misreports.
    Returns (the arrangement, why), the arrangement None off vLLM (the CPU stand-in and MLX run in the server's
    process anyway)."""
    if backend != "vllm":
        if choice == "in":
            raise SystemExit("--engine-process in is for --backend vllm (the other backends run in the server process)")
        return None, f"--backend {backend} runs in the server's process"
    if choice is None:
        if second_engines:
            choice, why = "separate", f"the default with a second engine ({', '.join(second_engines)})"
        elif environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "1":
            choice, why = "separate", "the default with VLLM_ENABLE_V1_MULTIPROCESSING=1"
        else:
            choice, why = "in", "the default for a single-engine server"
    else:
        why = f"--engine-process {choice}"
    if choice == "in":
        if second_engines:
            raise SystemExit(
                f"--engine-process in runs one vLLM engine in the server's process; {', '.join(second_engines)} would "
                "start a second one there, which fails on the card: use --engine-process separate (the default with "
                "a second engine)"
            )
        environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        return "in", why
    if environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0":
        raise SystemExit(
            "VLLM_ENABLE_V1_MULTIPROCESSING=0 runs the engine in the server's process: pass --engine-process in, or "
            "unset it"
        )
    return "separate", why


def resolve_multi_question(family, choice, engine_process, backend=None):
    """--multi-question's default: the base profile's (gemma-4-31b: warm), when the engine runs in the server's process
    (engine_process `in`, or None off vLLM), where a warm request repeats exactly; sequential otherwise and on the
    other bases. The 31B's warm default passed its gate: no choice changed against sequential on the suite, JevBench
    and travel, the largest probability difference 0.0073 (runs/2026-10-05_engine-death-gates)."""
    if backend == "mlx":  # the MLX engine continues each question from a copy of the shared prefix (mlx_engine)
        return "sequential"
    if choice is not None:
        return choice
    profile = getattr(family, "multi_question", "sequential")
    return profile if engine_process in ("in", None) else "sequential"


def resolve_register_boundary(family, choice):
    """--register-boundary's default: the base profile's (gemma-4-12b: after; gemma-4-31b: before, where registering
    after the response cost it a third of its throughput under load; runs/2026-10-06_latency-585w)."""
    if choice is not None:
        return choice
    return getattr(family, "register_boundary", "after")


MODEL_CLASSES = ("hidden-readout", "text-only", "view")


def resolve_head_mode(model_class, head_engine, one_engine=False):
    """The text engine's model class and where the intent head reads the hidden state from, from the command line.
    Returns (model_class, head_mode), head_mode one of:

      "single-engine"   the default: the text engine runs decisio's hidden-readout class and the head's hidden state
                        is read from it (three extra engine requests per head question, no second weight copy)
      "second-engine"   --head-engine: a second copy of the model in vLLM's pooling mode reads it (one pooling request
                        per head question, a second weight copy, not co-resident with the image engine); the text
                        engine then runs the text-only class unless --model-class says otherwise
      None              no head: --model-class view or text-only without --head-engine, or --one-engine; a registered
                        task then gets calibration only"""
    if one_engine:
        if head_engine:
            raise ValueError("--head-engine serves the text route's model; it does not combine with --one-engine")
        return None, None
    if model_class is None:
        model_class = "text-only" if head_engine else "hidden-readout"
    if model_class not in MODEL_CLASSES:
        raise ValueError(f"--model-class must be one of {MODEL_CLASSES}")
    if model_class == "hidden-readout":
        if head_engine:
            raise ValueError(
                "--model-class hidden-readout reads the head's hidden state from the text engine; "
                "--head-engine reads it from a second engine: choose one"
            )
        return model_class, "single-engine"
    return model_class, "second-engine" if head_engine else None


def resolve_pad_policy(backend, pad_policy=None):
    """--pad-policy's default by backend. vLLM: always, the served prompts (its hybrid prefix cache needs the block
    boundary). MLX: none, gated at 6 bits against the FP8 records (runs/2026-10-02_mlx-backend, 6bit_none): MLX's cache
    needs no padding, so its prompts differ from the vLLM path's by the padding only, a question asked alone and inside
    a request is one prompt, and a single question costs about half the time."""
    if pad_policy is not None:
        return pad_policy
    return "none" if backend == "mlx" else "always"


def engine_kwargs(args) -> dict:
    """`LLM(...)` keyword arguments of the text engine: --engine's JSON, plus decisio's model class when one is chosen
    (registered here for this process; vLLM's engine processes load the plugin through its entry point)."""
    kw = json.loads(args.engine)
    if args.model_class == "view":
        return kw
    import decisio.vllm_plugin as plugin

    if not plugin.installed_entry_point():
        raise SystemExit(
            "--model-class needs decisio's vLLM plugin installed (pip install -e .): vLLM's engine "
            "processes find it through the package's entry point, not through PYTHONPATH"
        )
    if not plugin.register():
        raise SystemExit(f"--model-class needs vllm=={plugin.SUPPORTED_VLLM}; found {plugin.vllm_version()}")
    from decisio.families import QWEN

    family = getattr(args, "family", QWEN)
    if args.model_class == "hidden-readout":
        from decisio.vllm_plugin.hidden import ENV_START

        # the base's reserved range, inherited by vLLM's engine processes (an explicit environment value wins)
        os.environ.setdefault(ENV_START, str(family.hidden_start))
        kw = {**kw, "max_logprobs": 1024}  # a head question reads up to 1,024 columns per request
    arch = family.classes[args.model_class]
    if arch is None:  # the base's own class, unmodified (Gemma's text path)
        return kw
    return {**kw, **plugin.engine_kwargs(arch)}


def facts_register_boundary(engine):
    """Where a single question registers its state's boundary (--register-boundary), or None on a base that does not."""
    fam = getattr(engine, "family", None)
    return getattr(engine, "register_boundary", None) if getattr(fam, "register_state_boundary", False) else None


def served_profile(engine, systemone) -> dict:
    """The base profile in effect and the settings it resolved to: the base, the temperature each question type is
    served at (the global one where a type has none of its own), and the prompt, yes/no rendering, option rendering and
    multi-question scoring."""
    fam = getattr(engine, "family", None)
    fmt = getattr(engine, "fmt", DEFAULT_FORMAT)
    return {
        "base": fam.key if fam is not None else None,
        "checkpoint": getattr(engine, "model_name", None),
        "revision": getattr(engine, "revision", None),
        "quantization_on_load": getattr(fam, "quantization", None),
        "engine_process": getattr(engine, "engine_process", None),
        "temperatures": {q: systemone.temperature_of(q) for q in ("choice", "noul", "score")},
        "prompt": {
            **fmt.facts(),
            "system_prompt": bool(getattr(fmt, "system_prompt", False)),
            "noul_rendering": systemone.noul_rendering,
            "describe_options": systemone.describe_options,
            "multi_question": getattr(engine, "multi_question", None),
            "register_boundary": facts_register_boundary(engine),
            "pad_policy": getattr(engine, "pad_policy", None),
            "pad_unit": getattr(engine, "pad_unit", None),
        },
    }


def resolve_base(args):
    """(family, PromptFormat) from the command line (decisio.families): --base, else the base --model's config.json
    names; the base's checkpoint when --model is not given; its pinned revision whenever the checkpoint is the base's
    own and no --revision was given, however the checkpoint was named; and the base's value for every setting left
    unset (--pad-to, --served-name, --noul-rendering, the prompt format, the temperatures)."""
    from decisio import hub
    from decisio.families import BASES, DEFAULT_BASE, FAMILIES, family_of, pinned_revision, read_config

    repo = None
    if args.base is not None and args.base not in BASES:
        # a decisio repository (decisio.hub): its weights are the checkpoint, its file names the base
        if not hub.is_repository(args.base):
            raise ValueError(
                f"--base {args.base!r} is neither a base ({', '.join(sorted(BASES))}) nor a repository "
                "(owner/name[@revision], or a directory holding decision_config.json)"
            )
        if args.model is not None:
            raise ValueError("--base <repository> serves that repository's weights; do not combine it with --model")
        repo = hub.open_repository(args.base, args.revision)
        args.base, args.model, args.revision = repo.base, repo.model, repo.revision
    elif args.base is None and args.model is None and args.backend == "vllm":
        args.base = DEFAULT_BASE  # neither given: the default base (the CPU stand-in and MLX always name a checkpoint)
    if repo is None and args.backend == "vllm" and args.base in BASES and args.model is None and args.revision is None:
        source = hub.default_source(args.base)  # the base's own key, no checkpoint named: its repository, if it has one
        if source:
            repo = hub.open_repository(source)
            args.base, args.model, args.revision = repo.base, repo.model, repo.revision
    if args.backend == "mlx" and args.model is None:
        raise ValueError("--backend mlx needs --model, an MLX conversion of the base (docs/design/mlx-backend.md)")
    if args.base is not None:
        fam = BASES[args.base]
        given = args.model is not None
        if not given:
            args.model = fam.model
        args.revision = pinned_revision(args.model, args.revision)
        if given:  # a checkpoint that declares another base's model type is refused; an unknown one is taken as given
            mt = read_config(args.model, args.revision).get("model_type")
            other = next((f for f in FAMILIES if mt in f.model_types and f is not fam), None)
            if other is not None:
                raise ValueError(f"--base {args.base}, but --model {args.model} is a {other.key} checkpoint ({mt})")
    else:
        if args.model is None:
            raise ValueError("give --model or --base")
        args.revision = pinned_revision(args.model, args.revision)
        fam = family_of(args.model, args.revision)
    args.repository = None
    if repo is not None:
        hub.verify(repo, fam)
        fam = hub.serve_family(fam, repo)
        # tasks are named for the checkpoint a repository copies, so those fitted on either serve on both
        args.model_identity = os.path.basename(os.path.normpath(repo.config["source"]["repository"]))
        args.repository = repo.facts()
    fmt = PromptFormat(
        tail=args.prompt_tail or fam.prompt_tail,
        slot=args.answer_slot or fam.answer_slot,
        variants=args.label_variants or fam.label_variants,
        system_prompt=fam.system_prompt if args.system_prompt is None else args.system_prompt,
    )
    if args.pad_to is None:
        args.pad_to = fam.pad_to
    if args.served_name is None:
        args.served_name = fam.served_name
    if args.noul_rendering is None:
        args.noul_rendering = fam.noul_rendering
    if args.temperature is None:
        args.temperature = fam.temperature
    if args.temperature_choice is None:
        args.temperature_choice = fam.choice_temperature
    return fam, fmt


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default=None,
        help="the base model and its served settings (decisio.families; README, 'Choosing a base'): gemma-4-31b "
        "(the default; google/gemma-4-31B-it quantized to FP8 on load, or its FP8 repository), qwen3.6-35b-a3b "
        "(Qwen/Qwen3.6-35B-A3B-FP8) or gemma-4-12b (google/gemma-4-12B-it), each at a pinned revision; or a decisio "
        "repository, owner/name[@revision] or a directory holding decision_config.json (docs/running.md). "
        "Every setting below that says 'the base's' takes the base's value unless given. Without --base, the base is "
        "detected from --model's config.json",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="the checkpoint: a local directory or a Hugging Face repository id (default: the base's own checkpoint)",
    )
    ap.add_argument(
        "--revision",
        default=None,
        help="the checkpoint's revision on the Hugging Face hub (default: the base's pinned revision when --model is "
        "the base's own checkpoint)",
    )
    ap.add_argument("--mode", default="separate", choices=["separate", "packed"])
    # served default: the official checkpoint under decisio's hidden-readout class, front padding to the block,
    # detokenize=False (always, in score_prompts), DeepGEMM off (VLLM_USE_DEEP_GEMM=0, deep_gemm_guard)
    ap.add_argument(
        "--pad-to",
        default=None,
        help="'block', a token count, or 'none' (separate mode); default: the base's (Qwen: block; Gemma: none)",
    )
    ap.add_argument("--pad-where", default="front", choices=PAD_PLACES)
    ap.add_argument("--adapter", action="append", default=[], help="name=path of a vLLM-format LoRA adapter")
    ap.add_argument("--pack", type=int, default=16)
    ap.add_argument(
        "--pad-policy",
        default=None,
        choices=["always", "shared", "row", "none"],
        help="always (the default with vLLM): front-pad every state to the block; shared: pad only when the request "
        "has more than one question to share the padded boundary (a single-question request then reads its state "
        "unpadded, a different prompt); row: a single-question request is front-padded so its whole row, state and "
        "question, ends on the block boundary and is prefilled in one engine step (multi-question requests as "
        "always); none (the default with --backend mlx, whose cache needs no padding): never pad, so a question "
        "asked alone and inside a request is one prompt (resolve_pad_policy)",
    )
    ap.add_argument(
        "--multi-question",
        default=None,
        choices=["sequential", "warm", "batch"],
        help="sequential (the served default; gemma-4-31b's is warm with the engine in the server's process): a "
        "request with several questions first prefills the state in a "
        "warm-up request, then sends each question in its own engine call, reading the state from the prefix cache, "
        "so every answer equals the question sent alone; warm (for bulk scoring): the warm-up, then every question "
        "in one batch, faster but each answer depends on the batch; batch: one engine call with every question, "
        "each prefilling the state itself, no warm-up",
    )
    ap.add_argument(
        "--register-boundary",
        default=None,
        choices=["after", "before", "off"],
        help="on a base that registers a state's boundary for the next question (the Gemma bases; /health "
        "registers_state_boundary), where a single question on a new state registers it: after (gemma-4-12b's "
        "default), once the response is out, so the caller does not wait for it and a question sent after the answer "
        "reads the state from the cache; before (gemma-4-31b's default), ahead of the question, as 0.8.1 did; off, "
        "never, so a later question about the state reads it again (docs/running.md)",
    )
    ap.add_argument(
        "--engine-process",
        default=None,
        choices=ENGINE_PROCESSES,
        help="where vLLM's engine runs: in (the server's process, VLLM_ENABLE_V1_MULTIPROCESSING=0), where a request's "
        "questions always share one engine step, so --multi-question warm repeats exactly; separate (vLLM's own "
        "arrangement, a process of its own). The default resolves to in for a single-engine server and to separate "
        "with a second engine (--image-model, --head-engine, --one-engine), with which in is refused",
    )
    # served default: CUDA graphs captured up to 4,096 tokens halve one question's latency at 500-2,000 token
    # states (236 -> 119 ms), answers bit-identical; engine start +105 s (+11 min with an adapter loaded)
    ap.add_argument("--engine", default=json.dumps(SERVED_ENGINE), help="extra LLM(...) keyword arguments as JSON")
    ap.add_argument(
        "--model-class",
        default=None,
        choices=MODEL_CLASSES,
        help="hidden-readout (the default): --model is the official checkpoint, loaded under decisio's "
        "registered text-only class that also returns the hidden state at the answer position, so "
        "registered tasks fit and serve an intent head from this one engine; text-only (the default "
        "with --head-engine): the same class without the hidden state; view: --model is a directory "
        "vLLM loads as it is, the text-only view built by decisio.serve.make_text_only (the fallback "
        "that needs no plugin). The decisio classes need the package installed, so vLLM's engine "
        "processes find the plugin through its entry point",
    )
    ap.add_argument(
        "--allow-deep-gemm",
        action="store_true",
        help="start even when VLLM_USE_DEEP_GEMM is set to something other than 0 (default: refuse)",
    )
    ap.add_argument(
        "--backend",
        default="vllm",
        choices=["vllm", "hf", "mlx"],
        help="hf: the CPU stand-in (decisio.serve.hf_letters), for the CPU smoke test only; mlx: Apple silicon "
        "(decisio.serve.mlx_engine), --model an MLX conversion, the text route with every feature except the image "
        "route, packed mode and adapters",
    )
    ap.add_argument(
        "--prefix-cache-mb",
        type=int,
        default=None,
        help="--backend mlx: the budget of the cross-request prefix cache in MiB (the base's: 2048 for the Qwen base, "
        "7400 for gemma-4-12b; 0 turns it off): a request whose state prefix was seen before continues from its kept "
        "cache instead of prefilling it, with the same answers bit for bit",
    )
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="--backend mlx: the tokenizer the prompts are built with, a directory, a repository or repo@revision "
        "(default: the base's official one, its checkpoint at its revision in decisio.families, so the prompts are "
        "the vLLM path's byte for byte; a conversion's own tokenizer may differ)",
    )
    ap.add_argument(
        "--served-name", default=None, help=f"the name GET /v1/models lists (default: the base's; Qwen: {SERVED_NAME})"
    )
    ap.add_argument(
        "--orders",
        type=int,
        default=1,
        choices=[1, 2],
        help="/v1/systemone: 2 = two-order averaging (after Reflex); the per-question disagreement goes "
        "to --branch-log",
    )
    ap.add_argument("--branch-log", default=None, help="JSONL of two-order branches and their disagreement")
    ap.add_argument(
        "--image-model",
        default=None,
        help="the full multimodal checkpoint: a second engine under the multimodal class, serving only "
        "/v1/systemone requests that carry images (decisio.serve.image_engine); the text route is "
        "unchanged",
    )
    ap.add_argument(
        "--hide-index-keys",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="/v1/systemone: show only the descriptions of options whose keys are a pure enumeration "
        "(option_0, option_1, ...), never the index beside our letters (default on; "
        "--no-hide-index-keys renders `key: description` as before)",
    )
    ap.add_argument(
        "--desnake-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="/v1/systemone: when options are shown as bare labels and every label is snake_case "
        "(card_arrival, ...), show them with spaces (default on; --no-desnake-labels leaves them)",
    )
    ap.add_argument(
        "--describe-options",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="/v1/systemone: show an option that has a description as its description alone, with no key; an "
        "option without one as its key (default on; --no-describe-options shows `key: description`; the keys stay "
        "the answer's keys)",
    )
    ap.add_argument(
        "--noul-rendering",
        default=None,
        choices=["words", "letters", "letters-keys"],
        help="/v1/systemone: how a yes/no question is asked (default: the base's; Qwen letters-keys, Gemma letters). "
        "letters-keys: a two-option "
        "choice, the false side first, the sides named ('No: ...', 'Yes: ...'), read from the letters; letters: as "
        "letters-keys with each side shown as its criteria description alone ('No' and 'Yes' without one); words (the "
        "earlier default): the instructions with 'Yes means' and 'No means' lines, read from the yes and no tokens. "
        "Abstention and two-order requests keep words",
    )
    ap.add_argument(
        "--abstention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="/v1/systemone: apply registered per-task abstention thresholds (POST /v1/abstention/tasks; "
        "decisio.serve.abstention); --no-abstention ignores them (default on; no task, no change)",
    )
    ap.add_argument(
        "--abstention-tasks", default=None, help="a JSON list of tasks to load at start (as the endpoint returns them)"
    )
    ap.add_argument(
        "--abstain-option",
        default=None,
        help='offer this extra option (e.g. "can\'t tell") on every question of requests that use '
        "imajev's extension, and report its probability as unknown_probability / abstained; "
        "untrained, opt-in",
    )
    ap.add_argument(
        "--system-prompt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="a system turn before each question (decisio.readout.system_prompt); default: the base's (Gemma: on, "
        "Qwen: off)",
    )
    ap.add_argument(
        "--prompt-tail",
        default=None,
        choices=["compact", "spaced"],
        help="the question's layout and last line (default: the base's, spaced for both): spaced (a blank line before "
        "and after the "
        "lettered options, then one line asking for the chosen option's letter alone, decisio.readout.spaced), or "
        "compact (the earlier default, for tasks fitted under it; no blank lines, 'Answer with the letter only.'). "
        "Tasks registered under one are not applied under the other",
    )
    ap.add_argument(
        "--answer-slot",
        default=None,
        choices=["prefill", "template"],
        help="where the label is read: after 'Answer:' prefilled in the assistant turn (prefill; the Qwen base's), or "
        "at the chat template's own first assistant position (template; the Gemma base's)",
    )
    ap.add_argument(
        "--label-variants",
        default=None,
        choices=["single", "summed"],
        help="the tokens read per label: single (the Qwen base's; one, the form the slot reads), or summed (the Gemma "
        "base's; every single-token form: ' A', 'A' and a byte-fallback token where there is one; ' yes', 'yes', "
        "' Yes' and 'Yes'; their probabilities summed)",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="/v1/systemone: the global temperature on the text route's plain readout, softmax(log p / T) "
        "(decisio.serve.temperature; default: the value fitted on the suite's served readouts); a "
        "registered task's own correction replaces it; 1 switches it off (the output before it, bit "
        "for bit); never changes the most probable option",
    )
    for qtype in ("choice", "noul", "score"):
        ap.add_argument(
            f"--temperature-{qtype}",
            type=float,
            default=None,
            help=f"/v1/systemone: the temperature for {qtype} questions in place of --temperature (default: "
            + (
                f"the base's; Qwen {SERVED_CHOICE_TEMPERATURE}, fitted on the suite's choice items; Gemma the "
                "global one"
                if qtype == "choice"
                else "the global one"
            )
            + "); 1 switches it off for that type",
        )
    ap.add_argument(
        "--noul-commit",
        action="store_true",
        help="/v1/systemone: an output transform on yes/no answers: P(yes) strictly between 0.20 and 0.80 is reported "
        "as 0.80 above 0.5 and 0.20 at or below it; the answer never changes, its calibration does (the measured cost "
        "in docs/handoffs/tasks.md); off by default",
    )
    ap.add_argument(
        "--tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="/v1/systemone: apply registered per-task calibration and intent heads (POST /v1/tasks; "
        "decisio.serve.tasks); --no-tasks ignores them (default on; no task, no change)",
    )
    ap.add_argument("--tasks-file", default=None, help="a JSON list of tasks to load at start (GET /v1/tasks?full=1)")
    ap.add_argument(
        "--head-engine",
        action="store_true",
        help="read the intent head's hidden state from a second engine "
        "(decisio.serve.hidden_engine.HiddenEngine), a second copy of --model in vLLM's pooling mode, "
        "instead of from the text engine: faster head questions (one request, not three) for "
        "deployments with heavy intent traffic on a dedicated card, at a second weight copy; not "
        "co-resident with --image-model (docs/handoffs/tasks.md)",
    )
    ap.add_argument("--head-gpu-memory-utilization", type=float, default=0.47, help="the hidden-state engine's share")
    ap.add_argument(
        "--debug-readout",
        action="store_true",
        help="honour the x-decisio-debug header (the raw readout in the response; verification only)",
    )
    ap.add_argument(
        "--one-engine",
        action="store_true",
        help="load only the --image-model engine (at --gpu-memory-utilization) and serve text requests on "
        "it too: the fallback for a card that cannot hold both engines; changes the text route's "
        "class",
    )
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="the text engine's share of the card")
    ap.add_argument(
        "--image-gpu-memory-utilization", type=float, default=0.50, help="the image engine's share of the card"
    )
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help="listen address (default: this machine only; put a reverse proxy in front to expose it)",
    )
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    adapters = dict(a.split("=", 1) for a in args.adapter)
    try:
        args.family, fmt = resolve_base(args)
    except ValueError as e:
        ap.error(str(e))
    if args.temperature <= 0:
        ap.error("--temperature must be positive (1 is off)")
    deep_gemm_guard(args.backend, os.environ, args.allow_deep_gemm)
    args.pad_policy = resolve_pad_policy(args.backend, args.pad_policy)
    if args.backend == "mlx":
        refused = [
            flag
            for flag, given in (
                ("--image-model", args.image_model),
                ("--one-engine", args.one_engine),
                ("--head-engine", args.head_engine),
                ("--adapter", args.adapter),
                ("--mode packed", args.mode == "packed"),
                (f"--multi-question {args.multi_question}", args.multi_question not in (None, "sequential")),
            )
            if given
        ]
        if refused:
            ap.error(f"--backend mlx serves the text route in separate mode, without {', '.join(refused)}")
    elif args.tokenizer:
        ap.error("--tokenizer is for --backend mlx (vLLM and the CPU stand-in use the model's own)")
    elif args.prefix_cache_mb is not None:
        ap.error("--prefix-cache-mb is for --backend mlx (vLLM has its own prefix cache)")
    if args.mode == "packed" and not fmt.is_default():
        ap.error(
            "--mode packed reads the compact layout only: add --prompt-tail compact (and no --answer-slot or "
            "--label-variants)"
        )
    one = args.one_engine
    if one and not args.image_model:
        ap.error("--one-engine needs --image-model")
    try:
        args.model_class, head_mode = resolve_head_mode(args.model_class, args.head_engine, one)
    except ValueError as e:
        ap.error(str(e))
    if head_mode == "second-engine" and args.image_model:
        ap.error(
            "--head-engine: the second weight copy leaves no room for --image-model on one card; serve images "
            "from another server, or use the default single-engine head"
        )
    second = [
        f
        for f, given in (
            ("--image-model", args.image_model),
            ("--head-engine", head_mode == "second-engine"),
            ("--one-engine", one),
        )
        if given
    ]
    args.engine_process, why = engine_process_guard(args.backend, args.engine_process, os.environ, second)
    args.multi_question = resolve_multi_question(args.family, args.multi_question, args.engine_process, args.backend)
    print(f"ENGINE PROCESS {args.engine_process} ({why}); multi-question {args.multi_question}", flush=True)
    if one:  # the image engine alone, at the text engine's share of the card, serves both routes
        pad_to = None if args.pad_to == "none" else args.pad_to
        if args.backend == "hf":
            from decisio.serve.hf_letters import HFImageLettersEngine

            engine = HFImageLettersEngine(args.image_model, pad_to=pad_to, pad_where=args.pad_where)
        else:
            from decisio.serve.image_engine import ImageLettersEngine

            engine = ImageLettersEngine(
                args.image_model,
                pad_to=pad_to,
                pad_where=args.pad_where,
                gpu_memory_utilization=args.gpu_memory_utilization,
                engine_kw=json.loads(args.engine),
            )
    elif args.backend == "hf":
        from decisio.serve.hf_letters import HFLettersEngine

        engine = HFLettersEngine(
            args.model,
            pad_to=None if args.pad_to == "none" else args.pad_to,
            pad_where=args.pad_where,
            revision=args.revision,
        )
        engine.fmt, engine.family = fmt, args.family
    elif args.backend == "mlx":
        from decisio.serve.mlx_engine import MLXLettersEngine

        engine = MLXLettersEngine(
            args.model,
            tokenizer=args.tokenizer,
            family=args.family,
            prefix_cache_mb=args.family.mlx_prefix_cache_mb if args.prefix_cache_mb is None else args.prefix_cache_mb,
            pad_to=None if args.pad_to == "none" else args.pad_to,
            pad_where=args.pad_where,
        )
    else:
        engine = LettersEngine(
            args.model,
            mode=args.mode,
            pad_to=None if args.pad_to == "none" else args.pad_to,
            pad_where=args.pad_where,
            adapters=adapters,
            pack=args.pack,
            gpu_memory_utilization=args.gpu_memory_utilization,
            engine_kw=engine_kwargs(args),
            fmt=fmt,
            family=args.family,
            revision=args.revision,
        )
    engine.fmt = fmt
    engine.pad_policy, engine.multi_question = args.pad_policy, args.multi_question
    engine.register_boundary = args.register_boundary = resolve_register_boundary(args.family, args.register_boundary)
    engine.repository = args.repository
    print(
        "ENGINE",
        json.dumps({**engine.facts(), "pad_policy": args.pad_policy, "multi_question": args.multi_question}),
        flush=True,
    )
    image_engine = engine if one else None
    if args.image_model and not one:
        pad_to = None if args.pad_to == "none" else args.pad_to
        if args.backend == "hf":
            from decisio.serve.hf_letters import HFImageLettersEngine

            image_engine = HFImageLettersEngine(args.image_model, pad_to=pad_to, pad_where=args.pad_where)
        else:
            from decisio.serve.image_engine import ImageLettersEngine

            image_engine = ImageLettersEngine(
                args.image_model,
                pad_to=pad_to,
                pad_where=args.pad_where,
                gpu_memory_utilization=args.image_gpu_memory_utilization,
                engine_kw=json.loads(args.engine),
            )
        print("IMAGE ENGINE", json.dumps(image_engine.facts()), flush=True)
    hidden_engine = None
    if head_mode == "single-engine":
        if args.backend == "mlx":
            from decisio.serve.mlx_engine import MLXHiddenReadout

            hidden_engine = MLXHiddenReadout(engine)
        elif args.backend == "hf":
            from decisio.serve.hidden_engine import HFReservedHiddenEngine

            hidden_engine = HFReservedHiddenEngine(
                args.model, pad_to=engine.pad_unit, pad_where=args.pad_where, revision=args.revision
            )
            hidden_engine.family = args.family
            hidden_engine.fmt = fmt
        else:
            from decisio.serve.hidden_engine import SingleEngineHidden

            hidden_engine = SingleEngineHidden(engine, args.model, revision=args.revision)
        print("HEAD ENGINE", json.dumps(hidden_engine.facts()), flush=True)
    if head_mode == "second-engine":
        if args.backend == "vllm" and args.gpu_memory_utilization + args.head_gpu_memory_utilization > 0.95:
            ap.error(
                "--head-engine: the text and hidden-state engines share the card; set --gpu-memory-utilization "
                "and --head-gpu-memory-utilization to at most 0.95 together (e.g. 0.47 and 0.47)"
            )
        pad_to = engine.pad_unit  # the text engine's padding unit, whatever the pooling engine's block size
        if args.backend == "hf":
            from decisio.serve.hidden_engine import HFHiddenEngine

            hidden_engine = HFHiddenEngine(args.model, pad_to=pad_to, pad_where=args.pad_where, revision=args.revision)
        else:
            from decisio.serve.hidden_engine import HiddenEngine

            hidden_engine = HiddenEngine(
                args.model,
                pad_to=pad_to,
                pad_where=args.pad_where,
                gpu_memory_utilization=args.head_gpu_memory_utilization,
                engine_kw=engine_kwargs(args),
                revision=args.revision,
            )
        # the second engine builds its rows with the server's prompt format and base, as the single-engine paths do
        hidden_engine.family = args.family
        hidden_engine.fmt = fmt
        # the head is fitted and served on the text route's exact token rows: refuse to start if they differ
        probe = (
            "A state to check. " * 30,
            [
                {"kind": "choice", "instructions": "Which?", "options": ["alpha", "beta"]},
                {"kind": "noul", "instructions": "Is it a check?"},
            ],
        )
        if hidden_engine._prepare_separate(*probe) != engine._prepare_separate(*probe):
            raise SystemExit("--head-engine: the hidden-state engine builds different token rows than the text engine")
        print("HEAD ENGINE", json.dumps(hidden_engine.facts()), flush=True)
    import uvicorn

    from decisio.serve.systemone import SystemOne
    from decisio.serve.tasks import TaskStore

    # a task is valid only for the model and the rendering it was fitted under
    store = TaskStore(
        fingerprint=task_fingerprint(
            served_name=args.served_name,
            # a decisio repository's copy of a checkpoint is named by its source: tasks fitted on one serve on both
            model=getattr(args, "model_identity", None) or os.path.basename(os.path.normpath(args.model)),
            pad_to=args.pad_to,
            pad_where=args.pad_where,
            hide_index_keys=args.hide_index_keys,
            desnake_labels=args.desnake_labels,
            pad_policy=args.pad_policy,
            prompt_format=None if fmt.is_default() else fmt.facts(),
            # a base other than Qwen (whose fingerprints predate bases)
            base=None if args.family.key == "qwen3.6-35b-a3b" else args.family.key,
        )
    )
    if args.tasks_file:
        data = json.loads(Path(args.tasks_file).read_text())
        store.load(data["tasks"] if isinstance(data, dict) else data)
        stale = [t["id"] for t in store.by_key.values() if not same_fingerprint(t["fingerprint"], store.fingerprint)]
        if stale:
            print(f"WARNING: tasks fitted under another model or rendering are not applied: {stale}", flush=True)
    so = SystemOne(
        engine,
        args.served_name,
        orders=args.orders,
        branch_log=args.branch_log,
        image_engine=image_engine,
        abstain_option=args.abstain_option,
        hide_index_keys=args.hide_index_keys,
        describe_options=args.describe_options,
        noul_rendering=args.noul_rendering,
        noul_commit=args.noul_commit,
        desnake_labels=args.desnake_labels,
        abstention=args.abstention,
        abstention_tasks=(json.loads(Path(args.abstention_tasks).read_text()) if args.abstention_tasks else None),
        tasks_enabled=args.tasks,
        task_store=store,
        hidden_engine=hidden_engine,
        debug_readout=args.debug_readout,
        temperature=args.temperature,
        temperatures={
            "choice": args.temperature_choice,
            "noul": args.temperature_noul,
            "score": args.temperature_score,
        },
    )
    # one engine state for every engine of this server: a dead engine refuses requests, turns /health to 503 and ends
    # the process with code 70, so that a restart policy starts a new server (decisio.serve.engine_health)
    health = EngineHealth()
    engines = [e for e in dict.fromkeys([engine, image_engine, hidden_engine]) if e is not None]
    for e in engines:
        e.health = health
    health.watch(engines)
    probe_ms = health.calibrate(engines)  # one timed probe per engine: the first request's probe allows for a slow one
    print("ENGINE PROBE", json.dumps({"ms": probe_ms, "deadline_s": health.probe_deadline_s()}), flush=True)
    uvicorn.run(make_app(engine, so), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    run_or_die(main)
