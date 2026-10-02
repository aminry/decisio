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

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from decisio.readout.letters import (  # noqa: E402
    LETTERS_INSTRUCTION,
    chat_wrap,
    fmt_state,
    label_token_ids,
    letter_labels,
    letters_prompt,
)


def options_listing(tok, options):
    """The listing a state carries when its questions use `options_in_state`."""
    labs = letter_labels(tok, len(options))
    return "Options:\n" + "\n".join(f"{lab}. {o}" for lab, o in zip(labs, options))


from decisio.names import SERVED_NAME, same_fingerprint  # noqa: E402
from decisio.serve.temperature import SERVED_TEMPERATURE  # noqa: E402

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


def state_prefix(tok, body):
    """The tokens every question about this state starts with, found from the state alone.

    Qwen's pre-tokenizer merges trailing punctuation with the blank line after it (".\n\n" is one
    token), so the state's end is not a token boundary. Two prompts that differ only in the first
    letter of the question share exactly the tokens up to the question; a question's own first
    character never merges back into them unless it is whitespace, which _prepare_separate checks.
    Depending on the state alone matters: a split computed from the request's questions (their longest
    common prefix) would move with the question mix, and so would any padding inserted there.
    """
    a = tok.encode(user_turn(tok, f"{body}\n\nA"), add_special_tokens=False)
    b = tok.encode(user_turn(tok, f"{body}\n\nB"), add_special_tokens=False)
    n = 0
    while a[n] == b[n]:
        n += 1
    return n, a[:n]


def question_text(tok, q):
    """A question's own text (no state), as letters_prompt renders it after the state, and its labels.

    `options_in_state`: the option listing ("Options:\nA. ...", options_listing) is already in the state,
    so the question is its instruction alone; the labels are the same letters (the shared option
    listing layout)."""
    kind = q.get("kind", "choice")
    if q.get("options_in_state"):
        labs = letter_labels(tok, len(q["options"]))
        return f"{q['instructions']}\n{LETTERS_INSTRUCTION}", [" " + x for x in labs], list(q["options"])
    options = ["yes", "no"] if kind == "noul" else list(q["options"])
    full, cands = letters_prompt(tok, kind, "", q["instructions"], options)
    assert full.startswith("\n\n") and full.endswith("\nAnswer:")
    text = full[2 : -len("\nAnswer:")]
    if kind == "noul" and q.get("noul_order") == "no_yes":
        # two-order mode's second yes/no branch (systemone.py, after Reflex): the answers named the other way round;
        # the labels and their order are unchanged, so the distribution stays [P(yes), P(no)]
        assert text.endswith("\nAnswer yes or no."), "the yes/no prompt changed; update the second branch"
        text = text[: -len("Answer yes or no.")] + "Answer no or yes."
    return text, cands, options


def user_turn(tok, content):
    """One chat turn exactly as evaluation renders a letters prompt (thinking disabled, "Answer:")."""
    return chat_wrap(tok, content + "\nAnswer:", "chat")


class LettersEngine:
    # the served defaults (--pad-policy, --multi-question); the server sets both from its flags
    pad_policy = "always"
    multi_question = "sequential"

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
    ):
        from vllm import LLM

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
            limit_mm_per_prompt={"image": 0, "video": 0},
        )
        if mode == "separate":
            kw.update(enable_prefix_caching=True, max_logprobs=256, logprobs_mode="processed_logprobs")
            if adapters:
                kw.update(enable_lora=True, max_loras=len(adapters), max_lora_rank=64)
        elif mode == "packed":
            from transformers import AutoTokenizer
            from vllm.config import PoolerConfig

            t = AutoTokenizer.from_pretrained(model)
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

    def facts(self):
        import torch
        import vllm

        c = self.llm.llm_engine.vllm_config
        return {
            "vllm": vllm.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "mode": self.mode,
            "block_size": self.block_size,
            "match_unit": self.match_unit,
            "pad_unit": self.pad_unit,
            "pad_where": self.pad_where if self.pad_unit else None,
            "quantization": str(c.model_config.quantization),
            "kv_cache_dtype": str(c.cache_config.cache_dtype),
            "mamba_ssm_cache_dtype": str(c.cache_config.mamba_ssm_cache_dtype),
            "adapters": sorted(self.adapters),
            "VLLM_USE_DEEP_GEMM": os.environ.get("VLLM_USE_DEEP_GEMM", "unset"),
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
        with self._lock:
            t0 = time.perf_counter()
            reqs = [(st, [qs[i] for i in td]) for (st, qs), td in zip(requests, todo)]
            if self.mode == "separate":
                probs, info = self._answer_separate(reqs, adapter)
            else:
                if adapter:
                    raise ValueError("adapters are served in separate mode")
                probs, info = self._answer_packed(reqs)
            info["server_ms"] = (time.perf_counter() - t0) * 1000
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
        if getattr(self, "_tail", None) is None:
            mark = "\ue000"
            full = user_turn(self.tok, mark)
            self._head_text, self._tail = full.split(mark)
        return self._tail

    def _labels(self, cands):
        cache = self.__dict__.setdefault("_label_cache", {})
        key = tuple(cands)
        if key not in cache:
            cache[key] = label_token_ids(self.tok, cands)
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
        texts = [question_text(self.tok, q) for q in questions]
        suffixes = [enc(text + tail) for text, _, _ in texts]
        # one tokenization of the state (three cost 87 ms at 8,000 tokens): the first question's
        # whole prompt, split where its own tokens begin. That boundary is the state's (the state_prefix
        # argument: a question's first character never merges back unless it is whitespace); if the whole
        # prompt does not end with the question's own tokens, fall back to the dummy-question split
        full0 = enc(user_turn(self.tok, f"{body}\n\n{texts[0][0]}"))
        n = len(full0) - len(suffixes[0])
        if n > 0 and full0[n:] == suffixes[0]:
            prefix = full0[:n]
        else:
            n, prefix = state_prefix(self.tok, body)
        rows = [(prefix + sfx, self._labels(cands)) for sfx, (_, cands, _) in zip(suffixes, texts)]
        if rows[0][0] != full0:
            rows = [(enc(user_turn(self.tok, f"{body}\n\n{text}")), self._labels(cands)) for text, cands, _ in texts]
        for ids, _ in rows:
            if ids[:n] != prefix:
                raise ValueError(
                    "a question's text merges with the state's last token; questions must not start with whitespace"
                )
        k = (-n % self.pad_unit) if self.pad_unit else 0
        if self.pad_policy == "shared" and len(questions) == 1:
            k = 0  # --pad-policy shared: no second question can reuse the padded boundary
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
        warm_ms = 0.0
        if warm:
            t = time.perf_counter()
            self.llm.generate(
                [TokensPrompt(prompt_token_ids=w, **extra) for w in warm],
                SamplingParams(max_tokens=1, temperature=0.0),
                lora_request=lora,
                use_tqdm=False,
            )
            warm_ms = (time.perf_counter() - t) * 1000
        # SamplingParams defaults leave top-k/top-p/min-p off, so "processed" is the masked logits alone.
        # detokenize=False: labels are read by token id and no text is used; the per-request detokenizer
        # was 73% of the frontend's CPU at an 8,000-token state (profiled)
        sps = [
            SamplingParams(
                max_tokens=1,
                temperature=0.0,
                logprobs=len(lab),
                allowed_token_ids=lab,
                skip_reading_prefix_cache=skip_cache or None,
                detokenize=False,
            )
            for _, lab in rows
        ]
        t = time.perf_counter()
        outs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=ids, **extra) for ids, _ in rows], sps, lora_request=lora, use_tqdm=False
        )
        questions_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        probs = []
        for (_, lab), o in zip(rows, outs):
            d = o.outputs[0].logprobs[0]
            lp = np.array([d[t].logprob for t in lab], dtype=np.float64)  # KeyError = a label went missing
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
        }

    def _answer_separate(self, requests, adapter):
        rows, warm, spans, shared = [], [], [], []
        unit = min(self.match_unit, self.pad_unit or self.match_unit)
        t = time.perf_counter()
        for state, questions in requests:
            r, P = self._prepare_separate(state, questions)
            # a warm-up pays only when several questions can reuse a registered boundary; --multi-question batch
            # sends the questions without it, each prefilling the state itself, in one engine call
            if len(r) > 1 and P >= unit and self.multi_question != "batch":
                warm.append(r[0][0][: P + 1])
            spans.append((len(rows), len(rows) + len(r)))
            rows += r
            shared.append(P)
        prepare_ms = (time.perf_counter() - t) * 1000
        if self.multi_question == "sequential" and len(rows) > 1:
            probs, info = self._score_one_at_a_time(rows, adapter, warm)
        else:
            probs, info = self.score_prompts(rows, adapter, warm)
        info.update(shared_prefix_tokens=shared[0] if len(shared) == 1 else shared, questions=len(rows))
        info["prepare_ms"] = prepare_ms
        return [probs[a:b] for a, b in spans], info

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
        for k in ("engine_timing", "engine_prompt_tokens", "cached_tokens"):
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
        outs = self.llm.encode(prompts, pooling_task="token_classify", use_tqdm=False)
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


def make_app(engine, systemone=None):
    """`/health` and `/v1/answer`; with `systemone` (decisio.serve.systemone.SystemOne) also TypeSafe's wire format,
    `POST /v1/systemone` and `GET /v1/models`, on the same engine."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class Request(BaseModel):
        state: str | dict
        questions: list[dict]
        adapter: str | None = None

    app = FastAPI(title="letters readout")

    @app.get("/health")
    def health():
        image = getattr(systemone, "image_engine", None) if systemone is not None else None
        so = (
            {
                "systemone": {
                    "hide_index_keys": systemone.hide_index_keys,
                    "desnake_labels": systemone.desnake_labels,
                    "describe_options": systemone.describe_options,
                    "noul_rendering": systemone.noul_rendering,
                    "abstention": systemone.abstention,
                    "abstention_tasks": {
                        t["id"]: (t.get("config") or {}).get("threshold") for t in systemone.tasks.values()
                    },
                    "orders": systemone.orders,
                    "abstain_option": systemone.abstain_option,
                    "tasks": systemone.tasks_enabled,
                    "debug_readout": systemone.debug_readout,
                    "temperature": systemone.temperature,
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
        return {"ok": True, **engine.facts(), **({"image_engine": image.facts()} if image is not None else {}), **so}

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
    arch = {"text-only": plugin.TEXT_ONLY, "hidden-readout": plugin.HIDDEN_READOUT}[args.model_class]
    if args.model_class == "hidden-readout":
        from decisio.vllm_plugin.hidden import DEFAULT_START, ENV_START

        os.environ.setdefault(ENV_START, str(DEFAULT_START))  # inherited by vLLM's engine processes
        kw = {**kw, "max_logprobs": 1024}  # a head question reads up to 1,024 columns per request
    return {**kw, **plugin.engine_kwargs(arch)}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="separate", choices=["separate", "packed"])
    # served default: the official checkpoint under decisio's hidden-readout class, front padding to the block,
    # detokenize=False (always, in score_prompts), DeepGEMM off (VLLM_USE_DEEP_GEMM=0, deep_gemm_guard)
    ap.add_argument("--pad-to", default="block", help="'block', a token count, or 'none' (separate mode)")
    ap.add_argument("--pad-where", default="front", choices=PAD_PLACES)
    ap.add_argument("--adapter", action="append", default=[], help="name=path of a vLLM-format LoRA adapter")
    ap.add_argument("--pack", type=int, default=16)
    ap.add_argument(
        "--pad-policy",
        default="always",
        choices=["always", "shared", "row"],
        help="always (the served default): front-pad every state to the block; shared: pad only when the request has "
        "more than one question to share the padded boundary (a single-question request then reads its state "
        "unpadded, a different prompt); row: a single-question request is front-padded so its whole row, state "
        "and question, ends on the block boundary and is prefilled in one engine step (multi-question requests "
        "as always)",
    )
    ap.add_argument(
        "--multi-question",
        default="sequential",
        choices=["sequential", "warm", "batch"],
        help="sequential (the served default): a request with several questions first prefills the state in a "
        "warm-up request, then sends each question in its own engine call, reading the state from the prefix cache, "
        "so every answer equals the question sent alone; warm (for bulk scoring): the warm-up, then every question "
        "in one batch, faster but each answer depends on the batch; batch: one engine call with every question, "
        "each prefilling the state itself, no warm-up",
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
        "--tokenizer",
        default=None,
        help="--backend mlx: the tokenizer the prompts are built with (default: the official "
        "Qwen/Qwen3.6-35B-A3B-FP8, so the prompts are the vLLM path's byte for byte; a conversion's own tokenizer "
        "may differ)",
    )
    ap.add_argument("--served-name", default=SERVED_NAME, help="the name GET /v1/models lists")
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
        default="words",
        choices=["words", "letters", "letters-keys"],
        help="/v1/systemone: how a yes/no question is asked. words (default): the instructions with 'Yes means' and "
        "'No means' lines, read from the yes and no tokens; letters: a two-option choice, the false side first, each "
        "shown as its criteria description ('No' and 'Yes' without one), read from the letters; letters-keys: as "
        "letters with the sides named ('No: ...', 'Yes: ...'). Abstention and two-order requests keep words",
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
        "--temperature",
        type=float,
        default=SERVED_TEMPERATURE,
        help="/v1/systemone: the global temperature on the text route's plain readout, softmax(log p / T) "
        "(decisio.serve.temperature; default: the value fitted on the suite's served readouts); a "
        "registered task's own correction replaces it; 1 switches it off (the output before it, bit "
        "for bit); never changes the most probable option",
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
    if args.temperature <= 0:
        ap.error("--temperature must be positive (1 is off)")
    deep_gemm_guard(args.backend, os.environ, args.allow_deep_gemm)
    if args.backend == "mlx":
        refused = [
            flag
            for flag, given in (
                ("--image-model", args.image_model),
                ("--one-engine", args.one_engine),
                ("--head-engine", args.head_engine),
                ("--adapter", args.adapter),
                ("--mode packed", args.mode == "packed"),
            )
            if given
        ]
        if refused:
            ap.error(f"--backend mlx serves the text route in separate mode, without {', '.join(refused)}")
    elif args.tokenizer:
        ap.error("--tokenizer is for --backend mlx (vLLM and the CPU stand-in use the model's own)")
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
            args.model, pad_to=None if args.pad_to == "none" else args.pad_to, pad_where=args.pad_where
        )
    elif args.backend == "mlx":
        from decisio.serve.mlx_engine import OFFICIAL_TOKENIZER, MLXLettersEngine

        engine = MLXLettersEngine(
            args.model,
            tokenizer=args.tokenizer or OFFICIAL_TOKENIZER,
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
        )
    engine.pad_policy, engine.multi_question = args.pad_policy, args.multi_question
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

            hidden_engine = HFReservedHiddenEngine(args.model, pad_to=engine.pad_unit, pad_where=args.pad_where)
        else:
            from decisio.serve.hidden_engine import SingleEngineHidden

            hidden_engine = SingleEngineHidden(engine, args.model)
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

            hidden_engine = HFHiddenEngine(args.model, pad_to=pad_to, pad_where=args.pad_where)
        else:
            from decisio.serve.hidden_engine import HiddenEngine

            hidden_engine = HiddenEngine(
                args.model,
                pad_to=pad_to,
                pad_where=args.pad_where,
                gpu_memory_utilization=args.head_gpu_memory_utilization,
                engine_kw=engine_kwargs(args),
            )
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
        fingerprint=json.dumps(
            {
                "served_name": args.served_name,
                "model": os.path.basename(os.path.normpath(args.model)),
                "pad_to": args.pad_to,
                "pad_where": args.pad_where,
                "hide_index_keys": args.hide_index_keys,
                "desnake_labels": args.desnake_labels,
                # only when not the default, so the fingerprints of tasks registered under the default are unchanged
                **({"pad_policy": args.pad_policy} if args.pad_policy != "always" else {}),
                # --describe-options is not here: it enters the task key of the questions it changes (tasks.task_key)
            },
            sort_keys=True,
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
        desnake_labels=args.desnake_labels,
        abstention=args.abstention,
        abstention_tasks=(json.loads(Path(args.abstention_tasks).read_text()) if args.abstention_tasks else None),
        tasks_enabled=args.tasks,
        task_store=store,
        hidden_engine=hidden_engine,
        debug_readout=args.debug_readout,
        temperature=args.temperature,
    )
    uvicorn.run(make_app(engine, so), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    run_or_die(main)
