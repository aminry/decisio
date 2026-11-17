# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Images at their places in a request's prompt, for every base that reads them (the model-independent part).

A request's text carries its images as marks (`decisio.serve.image_marks`): one where each image's reference stands, in
the state or in one question's own text. This module renders the text exactly as the text route does (system turn,
layout header, spaced tail, answer slot, label forms), cuts it at the marks and writes the model's placeholder there,
so a request with images and the same request without them differ by the image tokens only. The engine expands each
placeholder to the image's tokens.

Scope. An image in the state sits in the shared prefix, which every question reads from the cache. An image in a
question's instructions or criteria sits in that question's own part of the prompt: no other question reads it and no
other question's tokens count it. The header layout lists every question's instructions before the state; a question's
image appears there as the text "[image N]" and as pixels only in that question's own part.

Cache boundary. The prefix ends after the state's last image's end token, never inside an image block: the engine's
cache hits whole blocks, and an image's tokens belong together, so a block boundary inside an image is not a state to
resume from. Where the state's own text after its last image is too short to carry the last whole block past the
image, the prefix is extended by newline tokens up to the block boundary (`_prepare_images`).

What differs between bases is small and is a hook of the engine class: the placeholder (`placeholder_ids`, the token
that stands for the picture is `image_token_id`), what a mark expands to (`expanded_tokens`), what a picture costs in
`usage` (`image_tokens`), the format its prompts are written in (`image_fmt`) and whether a single question registers
the state's boundary with a warm-up (`warms_single_question`). `decisio.serve.gemma_images` and
`decisio.serve.image_engine` (the Qwen base) are the two.
"""

from __future__ import annotations

import hashlib
import time

from decisio.readout.letters import fmt_state, layout_header
from decisio.serve import image_marks as marks
from decisio.serve.vllm_engine import pad_prompt, question_text, user_turn


def image_id(img) -> str:
    """A content id for vLLM's multimodal caches: the decoded pixels, so the same picture sent as PNG or as a data URL
    shares its cache entries; computed once per request instead of once per question by the engine."""
    return f"{img.mode}:{img.width}x{img.height}:" + hashlib.sha256(img.tobytes()).hexdigest()


class PlacedImageRequests:
    """Mixin for a LettersEngine: `answer(..., images=[...], tag=...)` for a text that carries marks. Needs `self.tok`
    and the LettersEngine request-building attributes; the engine's `score_prompts` takes `row_mm`."""

    images_placed = True
    image_token_id: int  # the token the engine expands to an image's tokens
    placeholder_tail = 0  # tokens after `image_token_id` in one placeholder (the end-of-image token on Qwen)
    max_images = 0

    # ---- what a base says ------------------------------------------------------------------------

    @property
    def placeholder_ids(self) -> list[int]:
        return [self.image_token_id]

    def expanded_tokens(self, img) -> int:
        """The tokens one placeholder becomes in the engine's prompt for this picture."""
        raise NotImplementedError

    def image_tokens(self, img) -> int:
        """What one picture adds to `usage.input_tokens`."""
        raise NotImplementedError

    def image_fmt(self):
        """The prompt format this route writes its text in."""
        return self.fmt

    def warms_single_question(self) -> bool:
        return bool(getattr(getattr(self, "family", None), "register_state_boundary", False))

    # ---- the prompt ------------------------------------------------------------------------------

    def _enc_placed(self, text, tag):
        """(token ids, the image numbers in order): `text` cut at its marks, each part tokenized alone, the placeholder
        at each mark."""
        enc = lambda s: self.tok.encode(s, add_special_tokens=False)  # noqa: E731
        ids, found, at = [], [], 0
        for m in marks.pattern(tag).finditer(text):
            if m.start() > at:
                ids += enc(text[at : m.start()])
            ids += self.placeholder_ids
            found.append(int(m.group(1)))
            at = m.end()
        if at < len(text):
            ids += enc(text[at:])
        if ids.count(self.image_token_id) != len(found):
            name = self.tok.convert_ids_to_tokens(self.image_token_id)
            raise ValueError(f"the text contains {name}, which stands for an image; send the image instead")
        return ids, found

    def _prepare_images(self, state, images, questions, tag):
        """The rows of a request whose text carries marks: (rows, the prefix's sent length, its length once the engine
        has expanded the images, the state's image numbers, each row's image numbers, the images' token counts, where
        the state's last image ends in the expanded prefix or 0). A row is the shared prefix (the header, the state)
        and the question's own tokens; every image number is counted from 1."""
        fmt = self.image_fmt()
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
        grown = len(self.placeholder_ids)
        extra = sum(self.expanded_tokens(images[i - 1]) - grown for i in state_found)  # what the expansion adds
        n_exp = n + extra
        k = (-n_exp % self.pad_unit) if self.pad_unit else 0
        if self.pad_policy == "none" or (self.pad_policy == "shared" and len(questions) == 1):
            k = 0
        head = len(self._enc_placed(self._template_head(), tag)[0])
        end = 0
        if state_found:
            last = max(i for i, t in enumerate(prefix) if t == self.image_token_id)
            end = last + 1 + self.placeholder_tail + extra + (k if self.pad_where in ("front", "user") else 0)
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
        the image at its place; the deprecated legacy layout has no marks), so the multimodal path is compiled before
        the first request arrives."""
        from PIL import Image

        blank = Image.new("RGB", (512, 512), (128, 128, 128))
        question = [{"kind": "noul", "instructions": "Is this a warm-up request?"}]
        tag = marks.new_tag() if self.images_placed else None
        for _ in range(rounds):
            state = f"{marks.mark(1, tag)} Warm-up." if tag else "Warm-up."
            self.answer(state, question, images=[blank], **({"tag": tag} if tag else {}))

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
            ident = [image_id(img) for img in images]
            state_mm = {"image": [images[i - 1] for i in state_found]} if state_found else None
            state_ids = {"image": [ident[i - 1] for i in state_found]} if state_found else None
            row_mm = [{"image": [images[i - 1] for i in f]} if f else None for f in row_found]
            row_ids = [{"image": [ident[i - 1] for i in f]} if f else None for f in row_found]
            unit = min(self.match_unit, self.pad_unit or self.match_unit)
            # as for text: a warm-up of the prefix plus one token registers the state (and its images) in the cache
            keeps = len(rows) > 1 or self.warms_single_question()
            warm = [rows[0][0][: p_sent + 1]] if keeps and p_exp >= unit else []
            grown = len(self.placeholder_ids)
            expected = [
                len(ids) + sum(self.expanded_tokens(images[i - 1]) - grown for i in f)
                for (ids, _), f in zip(rows, row_found)
            ]  # the prompt the engine builds from each row: every placeholder expanded
            prep_ms = (time.perf_counter() - t0) * 1000
            probs, info = self.score_prompts(
                rows,
                adapter,
                warm,
                mm=state_mm,
                mm_uuids=state_ids,
                row_mm=row_mm,
                row_mm_uuids=row_ids,
                # an engine with its own prefix cache (the Mac's) is told where the shared prefix ends
                **({"prefix": p_sent} if getattr(self, "score_takes_prefix", False) else {}),
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
