# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Image requests on the Gemma 4 bases (`--images on`): the same engine that reads text reads the pictures.

A request's text carries its images as marks (`decisio.serve.image_marks`): one where each image's reference stands,
in the state or in one question's own text. This module renders the text exactly as the text route does (system turn,
layout header, spaced tail, answer slot, label forms), cuts it at the marks and puts the model's image placeholder
`<|image|>` there, so a request with images and the same request without them differ by the image tokens only.
vLLM expands each placeholder to the begin-of-image token, the image's soft tokens and the end-of-image token.

Scope. An image in the state sits in the shared prefix, which every question reads from the cache. An image in a
question's instructions or criteria sits in that question's own part of the prompt: no other question reads it and no
other question's tokens count it. The header layout lists every question's instructions before the state; a question's
image appears there as the text "[image N]" and as pixels only in that question's own part.

Cache boundary. The prefix ends after the state's last image's end token, never inside an image block: the engine's
cache hits whole blocks, and an image's tokens attend to each other (below), so a block boundary inside an image block
is not a state to resume from. Where the state's own text after its last image is too short to carry the last whole
block past the image, the prefix is extended by newline tokens up to the block boundary (`_prepare_images`).

Bidirectional image blocks. On the sliding-window layers the tokens of one image attend to each other, not only to
the tokens before them (the model's `use_bidirectional_attention = "vision"`, from the token types the processor
returns beside the ids). vLLM 0.31.0 does this itself for the image tokens it expands; the CPU stand-in
(`decisio.serve.hf_letters.HFGemmaImageLettersEngine`) passes the token types to the model.

    eng = GemmaImageLettersEngine(os.environ["DECISIO_MODEL"], image_budget=280, max_images=16)
    probs, info = eng.answer(f"{mark(1, tag)} The photo.", [question], images=[img], tag=tag)
"""

from __future__ import annotations

import time

from decisio.readout import box as boxmod
from decisio.readout.letters import check_thinking_closed, fmt_state, layout_header
from decisio.serve import image_marks as marks
from decisio.serve import layout as lay
from decisio.serve.image_engine import ImageRequests
from decisio.serve.vllm_engine import LettersEngine, pad_prompt, question_text, user_turn

IMAGE_TOKEN = "<|image|>"
# the most images one request may carry unless `--max-images` says otherwise: vLLM sizes its start-up profile of the
# image path from it and from the budget, and that cost is not measured yet
DEFAULT_MAX_IMAGES = 16
# a Box reply: at most this many new tokens, each with its top logprobs (the box_2d list is 19 tokens)
BOX_MAX_TOKENS = 32
BOX_TOP_LOGPROBS = 20
# branch expansion (the estimator's fix for digit-count ambiguity, decisio.readout.box.expand_branches): a branch
# worth a real decode is at least this likely, and at most this many are decoded per Box question
BOX_BRANCH_MIN_PROB = 0.05
BOX_MAX_BRANCHES = 16
BOX_BRANCH_MAX_TOKENS = 6  # a branch's continuation: the rest of its digits (at most 3 more) and its terminator
# the exact-tree debug readout (decisio.readout.box.exact_coordinate_tree): a card session's own check, never on by
# itself (the server needs --debug-box-tree, and a request needs its debug field)
BOX_TREE_MIN_PROB = 1e-4
BOX_TREE_MAX_DECODES = 64
BOX_TREE_TOP_LOGPROBS = 64


class GemmaImageRequests:
    """Mixin for a LettersEngine on a Gemma 4 base: `answer(..., images=[...])` for a text that carries marks. Needs
    `self.tok` and the LettersEngine request-building attributes; the engine's `score_prompts` takes `row_mm`."""

    images_enabled = True
    image_budget = marks.DEFAULT_BUDGET
    max_images = DEFAULT_MAX_IMAGES

    def _image_setup(self, budget, max_images):
        if budget not in marks.BUDGETS:
            raise ValueError(f"--image-budget: one of {', '.join(map(str, marks.BUDGETS))}, not {budget}")
        if max_images < 1:
            raise ValueError("--max-images must be at least 1")
        self.image_budget, self.max_images = int(budget), int(max_images)
        self.image_token_id = self.tok.convert_tokens_to_ids(IMAGE_TOKEN)
        if not isinstance(self.image_token_id, int) or self.image_token_id < 0:
            raise ValueError(f"the tokenizer has no {IMAGE_TOKEN}: this checkpoint reads no images")

    def image_facts(self) -> dict:
        facts = {"enabled": True, "budget": self.image_budget, "max_images": self.max_images, "route": "same engine"}
        window = getattr(self, "image_window", None)
        if window is not None:  # the budgets whose largest image block fits the base's sliding window
            facts["allowed_budgets"] = list(marks.allowed_budgets(window))
        return facts

    def image_tokens(self, img) -> int:
        """What one image adds to a prompt on this engine: its soft tokens at the server's budget, plus the begin and
        end tokens (the processor's arithmetic, `decisio.serve.image_marks.soft_tokens`)."""
        return marks.image_tokens(img.height, img.width, self.image_budget)

    def _enc_placed(self, text, tag):
        """(token ids, the image numbers in order): `text` cut at its marks, each part tokenized alone, the placeholder
        token at each mark."""
        enc = lambda s: self.tok.encode(s, add_special_tokens=False)  # noqa: E731
        ids, found, at = [], [], 0
        for m in marks.pattern(tag).finditer(text):
            if m.start() > at:
                ids += enc(text[at : m.start()])
            ids.append(self.image_token_id)
            found.append(int(m.group(1)))
            at = m.end()
        if at < len(text):
            ids += enc(text[at:])
        if ids.count(self.image_token_id) != len(found):
            raise ValueError(f"the text contains {IMAGE_TOKEN}, which stands for an image; send the image instead")
        return ids, found

    def _prepare_images(self, state, images, questions, tag):
        """The rows of a request whose text carries marks: (rows, the prefix's sent length, its length once the engine
        has expanded the images, the state's image numbers, each row's image numbers, the images' token counts, where
        the state's last image ends in the expanded prefix or 0). A row is the shared prefix (the header, the state)
        and the question's own tokens; every image number is counted from 1."""
        fmt = self.fmt
        if fmt.read_twice_under:
            raise ValueError("--read-twice-under is not served with images")
        body = fmt_state(state)
        if fmt.layout == "header":
            body = layout_header([marks.marks_as_text(q["instructions"], tag) for q in questions]) + body
        tail = self._template_tail()
        texts = [question_text(self.tok, q, fmt) for q in questions]
        suffixes = [self._enc_placed(t + tail, tag) for t, _, _ in texts]
        # the state's boundary as the text route finds it (`_rows`): the first question's whole prompt, split where its
        # own tokens begin; a question's first character never merges back unless it is whitespace
        full0, found0 = self._enc_placed(user_turn(self.tok, f"{body}\n\n{texts[0][0]}", fmt), tag)
        n = len(full0) - len(suffixes[0][0])
        if n > 0 and full0[n:] == suffixes[0][0]:
            prefix = full0[:n]
        else:
            a, _ = self._enc_placed(user_turn(self.tok, f"{body}\n\nA", fmt), tag)
            b, _ = self._enc_placed(user_turn(self.tok, f"{body}\n\nB", fmt), tag)
            n = 0
            while a[n] == b[n]:
                n += 1
            prefix = a[:n]
        state_found = found0[: prefix.count(self.image_token_id)]
        rows = [(prefix + sfx, self._labels(cands)) for (sfx, _), (_, cands, _) in zip(suffixes, texts)]
        for ids, _ in rows:
            if ids[:n] != prefix:
                raise ValueError(
                    "a question's text merges with the state's last token; questions must not start with whitespace"
                )
        row_found = [state_found + f for _, f in suffixes]
        for found in row_found:
            if len(set(found)) != len(found) or any(not 1 <= i <= len(images) for i in found):
                raise ValueError("the request's image marks do not match its images")
        counts = [self.image_tokens(img) for img in images]
        extra = sum(counts[i - 1] - 1 for i in state_found)  # what the engine's expansion adds to the prefix
        n_exp = n + extra
        k = (-n_exp % self.pad_unit) if self.pad_unit else 0
        if self.pad_policy == "none" or (self.pad_policy == "shared" and len(questions) == 1):
            k = 0
        head = len(self._enc_placed(self._template_head(), tag)[0])
        end = 0
        if state_found:
            last = max(i for i, t in enumerate(prefix) if t == self.image_token_id)
            end = last + 1 + extra + (k if self.pad_where in ("front", "user") else 0)
        rows = [(pad_prompt(ids, n, k, self.pad_token, self.pad_where, head), lab) for ids, lab in rows]
        if state_found and k == 0:
            # the cache hits whole blocks: where the last whole block of the prefix ends inside the last image, pad
            # with newlines up to the block boundary, which is then after the image's end token
            unit = self.match_unit
            if (n_exp // unit) * unit < end:
                nl = self.tok.encode("\n", add_special_tokens=False)
                if len(nl) != 1:
                    raise ValueError("the tokenizer has no single newline token to pad the prefix with")
                k = -n_exp % unit
                rows = [(pad_prompt(ids, n, k, nl[0], "between", head), lab) for ids, lab in rows]
        return rows, n + k, n_exp + k, state_found, row_found, counts, end

    def warm_images(self, rounds: int = 2):
        """Read one blank picture `rounds` times, through the same request path as a served one (a fresh tag, a mark,
        the image at its place), so the multimodal path is compiled before the first request arrives."""
        from PIL import Image

        blank = Image.new("RGB", (512, 512), (128, 128, 128))
        tag = marks.new_tag()
        for _ in range(rounds):
            self.answer(
                f"{marks.mark(1, tag)} Warm-up.",
                [{"kind": "noul", "instructions": "Is this a warm-up request?"}],
                images=[blank],
                tag=tag,
            )

    def answer(self, state, questions, adapter=None, images=None, tag=None):
        if not images:
            return super().answer(state, questions, adapter)
        if tag is None:
            raise ValueError("a request with images needs the tag of its marks")
        if len(images) > self.max_images:
            raise ValueError(f"at most {self.max_images} images per request")
        with self._lock:
            t0 = time.perf_counter()
            rows, p_sent, p_exp, state_found, row_found, counts, end = self._prepare_images(
                state, images, questions, tag
            )
            ident = [ImageRequests.image_id(img) for img in images]
            state_mm = {"image": [images[i - 1] for i in state_found]} if state_found else None
            state_ids = {"image": [ident[i - 1] for i in state_found]} if state_found else None
            row_mm = [{"image": [images[i - 1] for i in f]} if f else None for f in row_found]
            row_ids = [{"image": [ident[i - 1] for i in f]} if f else None for f in row_found]
            unit = min(self.match_unit, self.pad_unit or self.match_unit)
            # as for text: a warm-up of the prefix plus one token registers the state (and its images) in the cache
            keeps = len(rows) > 1 or getattr(getattr(self, "family", None), "register_state_boundary", False)
            warm = [rows[0][0][: p_sent + 1]] if keeps and p_exp >= unit else []
            expected = [
                len(ids) - len(f) + sum(counts[i - 1] for i in f) for (ids, _), f in zip(rows, row_found)
            ]  # the prompt the engine builds from each row: every placeholder expanded
            prep_ms = (time.perf_counter() - t0) * 1000
            probs, info = self.score_prompts(
                rows, adapter, warm, mm=state_mm, mm_uuids=state_ids, row_mm=row_mm, row_mm_uuids=row_ids
            )
            seen = info.get("engine_prompt_tokens")
            if seen and list(seen) != expected:
                raise RuntimeError(
                    f"the engine's prompts are {list(seen)} tokens, the image arithmetic made {expected}: the image "
                    "token counts do not match this checkpoint's processor"
                )
            info.update(
                shared_prefix_tokens=p_exp,
                state_image_end=end,
                # state images once, each question's own images once for the question that carries them
                image_tokens=[counts[i - 1] for i in state_found]
                + [counts[i - 1] for f in row_found for i in f[len(state_found) :]],
                expected_prompt_tokens=expected,
                prepare_ms=prep_ms,
                questions=len(rows),
                server_ms=(time.perf_counter() - t0) * 1000,
            )
        return probs, info

    # ---- Box -------------------------------------------------------------------------------------------------------

    box_enabled = True

    def _box_tail(self):
        """The chat template's text after a user turn's content, with the model turn opened and thinking closed (the
        empty thought channel), and no answer slot: the reply that follows is the forced `box_2d` prefix."""
        cache = self.__dict__.setdefault("_box_tails", {})
        sysp = bool(self.fmt.system_prompt)
        if sysp not in cache:
            mark = "\ue000"
            cache[sysp] = self._box_turn(mark).split(mark)[1]
        return cache[sysp]

    def _box_turn(self, content):
        messages = [{"role": "user", "content": content}]
        if self.fmt.system_prompt:
            from decisio.readout.system_prompt import SYSTEM_PROMPT

            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        out = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        check_thinking_closed(out, content)
        return out

    def box_blocked(self) -> str | None:
        """Why this engine cannot answer a Box question, or None. The layout is not one of the reasons: a Box is read
        under the plain layout whatever the server's default (`box` puts `decisio.serve.layout.using("plain")`
        around its reads)."""
        return None

    def _box_branch_prompts(self, pairs, offsets, full_ids, gen_ids):
        """The continuation prompts of the branches worth a real decode off these four coordinates' greedy paths
        (at most `BOX_MAX_BRANCHES`, by probability): the generation replayed up to each branch's own digit, then
        the branch's digit instead of the greedy one. Returns (the branches asked for, their prompts, whether more
        branches had enough probability than the cap allowed)."""
        cands = boxmod.box_branches(pairs, BOX_BRANCH_MIN_PROB)
        capped = len(cands) > BOX_MAX_BRANCHES
        chosen = cands[:BOX_MAX_BRANCHES]
        prompts = []
        for b in chosen:
            idx = boxmod.branch_step_index(offsets[b.coordinate], b.position)
            prompts.append(full_ids + gen_ids[:idx] + [self.tok.convert_tokens_to_ids(str(b.digit))])
        return chosen, prompts, capped

    def _box_exact_tree(self, pairs, offsets, full_ids, gen_ids, adapter, mm, mm_ids, fast):
        """The debug readout: per coordinate, a real recursive expansion of the digit tree (`exact_coordinate_tree`),
        beside the fast estimator's own value (`fast`, already computed) and the plain greedy value."""
        out = []
        for ci, (ds, ts) in enumerate(pairs):
            base = full_ids + gen_ids[: offsets[ci][0]]

            def decode(prefix, base=base):
                ids = base + [self.tok.convert_tokens_to_ids(str(d)) for d in prefix]
                gens, _ = self.generate_rows(
                    [ids], adapter, row_mm=[mm], row_mm_uuids=[mm_ids], max_tokens=1, top=BOX_TREE_TOP_LOGPROBS
                )
                return boxmod.step_distribution(gens[0][0], len(prefix), min_prob=BOX_TREE_MIN_PROB)

            expected, covered, decodes = boxmod.exact_coordinate_tree(
                ds, ts, decode, min_prob=BOX_TREE_MIN_PROB, max_decodes=BOX_TREE_MAX_DECODES
            )
            greedy = min(int("".join(s[0] for s in ds)), boxmod.BOX_GRID)
            out.append(
                {
                    "exact": round(expected / boxmod.BOX_GRID, 4),
                    "fast": round(fast[ci] / boxmod.BOX_GRID, 4),
                    "greedy": round(greedy / boxmod.BOX_GRID, 4),
                    "tree_mass": round(covered, 4),
                    "decodes": decodes,
                }
            )
        return out

    def box(self, state, questions, adapter=None, images=None, tag=None, presence=None, exact_tree=False):
        """One greedy generation per Box question, then its box from the branch-expanded estimator (real decodes
        of the likeliest alternatives off the greedy path, `decisio.readout.box.expand_branches`; a terminator
        branch is exact already and needs none). `questions`: dicts with `instructions` (the text carries its
        marks); `presence`: the yes/no questions read beside them (their prompts share the state's prefix, so the
        rows here share it with them, and the first of them fixes where it ends). `exact_tree`: also read the full
        digit tree by real decodes, for the debug readout (`--debug-box-tree`); never on without it.

        Returns (`results`, `info`): `results` is a `decisio.readout.box.BoxReadout` per question, or the
        `BoxParseError` the model's reply raised (never a made-up box); `info["generated_tokens"]` is the main
        reply's tokens per question, `info["branch_tokens"]` the branch decodes' tokens in all (both count toward
        usage), and `info["branches"]` is, per question, how many branches were decoded and whether the cap was
        hit; `info["tree"]`, present when `exact_tree`, is the exact readout per question (None for a question
        that had no readable box)."""
        if not images:
            raise ValueError("a Box question needs an image")
        reason = self.box_blocked()
        if reason:
            raise ValueError(reason)
        with self._lock, lay.using("plain"):
            t0 = time.perf_counter()
            probe = presence or [{"kind": "noul", "instructions": "Is this a picture?"}]
            rows, p_sent, p_exp, state_found, row_found, counts, end = self._prepare_images(state, images, probe, tag)
            prefix = rows[0][0][:p_sent]
            tail = self._box_tail()
            suffixes = [
                self._enc_placed(boxmod.render_box_prompt(q.get("instructions")) + tail + boxmod.BOX_REPLY_PREFIX, tag)
                for q in questions
            ]
            full = [prefix + ids for ids, _ in suffixes]
            found = [state_found + f for _, f in suffixes]
            for f in found:
                if len(set(f)) != len(f) or any(not 1 <= i <= len(images) for i in f):
                    raise ValueError("the request's image marks do not match its images")
            row_mm = [{"image": [images[i - 1] for i in f]} for f in found]
            ident = [ImageRequests.image_id(img) for img in images]
            row_ids = [{"image": [ident[i - 1] for i in f]} for f in found]
            prep_ms = (time.perf_counter() - t0) * 1000
            gens, info = self.generate_rows(
                full, adapter, row_mm=row_mm, row_mm_uuids=row_ids, max_tokens=BOX_MAX_TOKENS, top=BOX_TOP_LOGPROBS
            )

            results, branches_info, tree_info = [], [], []
            branch_tokens = 0
            for qi, steps in enumerate(gens):
                try:
                    pairs, _, offsets = boxmod.parse_coordinates(steps, prefix_in_steps=False, with_offsets=True)
                except boxmod.BoxParseError as e:
                    results.append(e)
                    branches_info.append({"decoded": 0, "capped": False})
                    tree_info.append(None)
                    continue
                gen_ids = [self.tok.convert_tokens_to_ids(tok) for tok, _ in steps]
                chosen, prompts, capped = self._box_branch_prompts(pairs, offsets, full[qi], gen_ids)
                decoded = {}
                if prompts:
                    br_gens, br_info = self.generate_rows(
                        prompts,
                        adapter,
                        row_mm=[row_mm[qi]] * len(prompts),
                        row_mm_uuids=[row_ids[qi]] * len(prompts),
                        max_tokens=BOX_BRANCH_MAX_TOKENS,
                        top=1,
                    )
                    branch_tokens += sum(br_info["generated_tokens"])
                    for b, continuation in zip(chosen, br_gens):
                        try:
                            decoded[b] = boxmod.branch_value(boxmod.branch_prefix_digits(pairs, b), continuation)
                        except boxmod.BoxParseError:
                            pass  # this one branch keeps the plain estimator's approximation
                by_coord: dict[int, dict] = {}
                for b, v in decoded.items():
                    by_coord.setdefault(b.coordinate, {})[(b.position, b.digit)] = float(v)
                est = [boxmod.estimate_coordinate(ds, ts, by_coord.get(ci)) for ci, (ds, ts) in enumerate(pairs)]
                results.append(boxmod.assemble_box(est))
                branches_info.append({"decoded": len(decoded), "capped": capped})
                tree_info.append(
                    self._box_exact_tree(
                        pairs, offsets, full[qi], gen_ids, adapter, row_mm[qi], row_ids[qi], [e.expected for e in est]
                    )
                    if exact_tree
                    else None
                )

            info.update(
                prepare_ms=prep_ms,
                server_ms=(time.perf_counter() - t0) * 1000,
                branch_tokens=branch_tokens,
                branches=branches_info,
                **({"tree": tree_info} if exact_tree else {}),
            )
            info["box_input_tokens"] = [len(ids) - len(f[len(state_found) :]) for (ids, _), f in zip(suffixes, found)]
        return results, info


class GemmaImageLettersEngine(GemmaImageRequests, LettersEngine):
    """The vLLM letters engine of a Gemma 4 base with image input on: the multimodal class, `max_images` images per
    prompt, the soft-token budget `image_budget`; every text-path setting as the served default."""

    def __init__(self, model, image_budget=marks.DEFAULT_BUDGET, max_images=DEFAULT_MAX_IMAGES, engine_kw=None, **kw):
        from decisio.families import family_of

        if image_budget not in marks.BUDGETS:
            raise ValueError(f"--image-budget: one of {', '.join(map(str, marks.BUDGETS))}, not {image_budget}")
        from decisio.families import read_config

        family = kw.get("family") or family_of(model, kw.get("revision"))
        self.image_window = marks.sliding_window_of(read_config(model, kw.get("revision")))
        marks.check_budget_fits_window(image_budget, self.image_window)
        # video and audio stay off; the image limit is the one place a request's pictures are bounded in memory
        engine_kw = {
            "limit_mm_per_prompt": {**family.limit_mm, "image": int(max_images)},
            "mm_processor_kwargs": {"max_soft_tokens": int(image_budget)},
            **(engine_kw or {}),
        }
        LettersEngine.__init__(self, model, engine_kw=engine_kw, **kw)
        self._image_setup(image_budget, max_images)
        # the multimodal path compiles on its first picture: warm it as the text path is
        self.warm_images()

    def facts(self):
        return {**super().facts(), "images": self.image_facts()}
