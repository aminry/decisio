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

from decisio.readout.letters import fmt_state, layout_header
from decisio.serve import image_marks as marks
from decisio.serve.image_engine import ImageRequests
from decisio.serve.vllm_engine import LettersEngine, pad_prompt, question_text, user_turn

IMAGE_TOKEN = "<|image|>"
# the most images one request may carry unless `--max-images` says otherwise: vLLM sizes its start-up profile of the
# image path from it and from the budget, and that cost is not measured yet
DEFAULT_MAX_IMAGES = 16


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
        return {"enabled": True, "budget": self.image_budget, "max_images": self.max_images, "route": "same engine"}

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


class GemmaImageLettersEngine(GemmaImageRequests, LettersEngine):
    """The vLLM letters engine of a Gemma 4 base with image input on: the multimodal class, `max_images` images per
    prompt, the soft-token budget `image_budget`; every text-path setting as the served default."""

    def __init__(self, model, image_budget=marks.DEFAULT_BUDGET, max_images=DEFAULT_MAX_IMAGES, engine_kw=None, **kw):
        from decisio.families import family_of

        if image_budget not in marks.BUDGETS:
            raise ValueError(f"--image-budget: one of {', '.join(map(str, marks.BUDGETS))}, not {image_budget}")
        family = kw.get("family") or family_of(model, kw.get("revision"))
        # video and audio stay off; the image limit is the one place a request's pictures are bounded in memory
        engine_kw = {
            "limit_mm_per_prompt": {**family.limit_mm, "image": int(max_images)},
            "mm_processor_kwargs": {"max_soft_tokens": int(image_budget)},
            **(engine_kw or {}),
        }
        LettersEngine.__init__(self, model, engine_kw=engine_kw, **kw)
        self._image_setup(image_budget, max_images)
        # the multimodal path compiles on its first picture: warm it as the text path is
        from PIL import Image

        blank = Image.new("RGB", (512, 512), (128, 128, 128))
        for _ in range(2):
            self.answer(
                f"{marks.mark(1)} Warm-up.",
                [{"kind": "noul", "instructions": "Is this a warm-up request?"}],
                images=[blank],
            )

    def facts(self):
        return {**super().facts(), "images": self.image_facts()}
