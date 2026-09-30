# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Image requests on the letters engine: photos go into the state, questions are read exactly as for text.

The full checkpoint is served under the multimodal class (Qwen3_5MoeForConditionalGeneration) as a second engine beside
the text-only default, which stays the route for every request without images (the multimodal class builds M-RoPE
positions for every prompt token, a per-request cost the text route does not pay).

A request with images is rendered as the model's chat template renders one: every image as
`<|vision_start|><|image_pad|><|vision_end|>` at the start of the user turn, then the state and the question exactly as
the text route renders them (`vllm_engine._prepare_separate`). The cached prefix (images plus state) is front-padded
to the block like a text state, so every question after the first reads the images and the state from the prefix cache;
the vision tower runs once per distinct image (vLLM keys its encoder cache and the prefix blocks by the image's hash).

Each image stays one `<|image_pad|>` in the token ids sent to the engine, which expands it to the image's token count
(the processor's grid, 16-pixel patches merged 2x2 on Qwen3.5/3.6); the padding and the cached-prefix length are
computed in expanded units, and the engine-side prompt length is returned so the expansion can be checked (gate G0).

    eng = ImageLettersEngine("/models/Qwen3.6-35B-A3B-FP8", pad_to="block", pad_where="front")
    probs, info = eng.answer(state, questions, images=[PIL.Image, ...])
"""
from __future__ import annotations

import hashlib
import io
import time

from decisio.readout.letters import fmt_state
from decisio.serve.vllm_engine import USER_HEAD, LettersEngine, pad_prompt, question_text, state_prefix, user_turn

VISION = "<|vision_start|><|image_pad|><|vision_end|>"
MAX_IMAGES = 2
MAX_BYTES = 20 * 1024 * 1024          # imajev's server limits (github.com/mohit67890/imajev)
MAX_PIXELS = 20_000_000


def load_image(data: bytes):
    """Bytes -> an RGB PIL image; PNG, JPEG or WebP; refuses oversize inputs (the imajev limits)."""
    from PIL import Image
    if len(data) > MAX_BYTES:
        raise ValueError("image exceeds 20 MiB")
    img = Image.open(io.BytesIO(data))
    if img.format not in ("PNG", "JPEG", "WEBP"):
        raise ValueError(f"unsupported image format {img.format}")
    if img.width * img.height > MAX_PIXELS:
        raise ValueError("image exceeds 20 megapixels")
    return img.convert("RGB")


class ImageRequests:
    """Mixin for a LettersEngine: `answer(..., images=[...])`. Needs `self.tok`, `self.processor` (the model's
    AutoProcessor) and the LettersEngine request-building attributes; the engine's `score_prompts` takes `mm`."""

    def _image_setup(self, model):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model)
        self.image_pad_id = self.tok.convert_tokens_to_ids("<|image_pad|>")
        self.merge = self.processor.image_processor.merge_size

    @staticmethod
    def image_id(img) -> str:
        """A content id for vLLM's multimodal caches: the decoded pixels, so the same picture sent as PNG or as a
        data URL shares its cache entries; computed once per request instead of once per question by the engine."""
        return f"{img.mode}:{img.width}x{img.height}:" + hashlib.sha256(img.tobytes()).hexdigest()

    def image_tokens(self, img) -> int:
        """The engine's expansion of one image: its processor's resize (`smart_resize`, the processor's own function
        and pixel bounds) in merged patches, without running the processor (tests/unit/test_image_route.py, I1, checks
        the two agree)."""
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        ip = self.processor.image_processor
        unit = ip.patch_size * ip.merge_size
        h, w = smart_resize(img.height, img.width, factor=unit, min_pixels=ip.size["shortest_edge"],
                            max_pixels=ip.size["longest_edge"])
        return (h // unit) * (w // unit)

    def _prepare_images(self, state, images, questions):
        """(rows with unexpanded image placeholders, P in expanded units, P in sent units, per-image token counts)."""
        enc = lambda s: self.tok.encode(s, add_special_tokens=False)          # noqa: E731
        body = fmt_state(state)
        vision = VISION * len(images)
        tail = self._template_tail()
        texts = [question_text(self.tok, q) for q in questions]
        suffixes = [enc(t + tail) for t, _, _ in texts]
        full0 = enc(user_turn(self.tok, f"{vision}{body}\n\n{texts[0][0]}"))
        n = len(full0) - len(suffixes[0])
        if not (n > 0 and full0[n:] == suffixes[0]):
            n, _ = state_prefix(self.tok, vision + body)
        prefix = full0[:n]
        rows = [(prefix + sfx, self._labels(cands)) for sfx, (_, cands, _) in zip(suffixes, texts)]
        for ids, _ in rows:
            if ids[:n] != prefix:
                raise ValueError("a question's text merges with the state's last token")
        counts = [self.image_tokens(img) for img in images]
        if prefix.count(self.image_pad_id) != len(images):
            raise ValueError("the image placeholders are not all in the cached prefix")
        extra = sum(c - 1 for c in counts)                       # what the engine's expansion adds to the prefix
        n_exp = n + extra
        k = (-n_exp % self.pad_unit) if self.pad_unit else 0
        head = len(enc(USER_HEAD))
        rows = [(pad_prompt(ids, n, k, self.pad_token, self.pad_where, head), lab) for ids, lab in rows]
        return rows, n_exp + k, n + k, counts

    def answer(self, state, questions, adapter=None, images=None):
        if not images:
            return super().answer(state, questions, adapter)
        if len(images) > MAX_IMAGES:
            raise ValueError(f"at most {MAX_IMAGES} images per request")
        with self._lock:
            t0 = time.perf_counter()
            rows, p_exp, p_sent, counts = self._prepare_images(state, images, questions)
            unit = min(self.match_unit, self.pad_unit or self.match_unit)
            # as for text: one warm-up request of the prefix plus one token registers the state (and the images)
            warm = [rows[0][0][:p_sent + 1]] if len(rows) > 1 and p_exp >= unit else []
            ids = [self.image_id(img) for img in images]
            prep_ms = (time.perf_counter() - t0) * 1000
            probs, info = self.score_prompts(rows, adapter, warm, mm={"image": list(images)}, mm_uuids={"image": ids})
            info.update(shared_prefix_tokens=p_exp, image_tokens=counts, prepare_ms=prep_ms, questions=len(rows),
                        server_ms=(time.perf_counter() - t0) * 1000)
        return probs, info


class ImageLettersEngine(ImageRequests, LettersEngine):
    """The multimodal class on vLLM with up to two images per request; every text-path setting as the served default."""

    def __init__(self, model, engine_kw=None, **kw):
        engine_kw = {"limit_mm_per_prompt": {"image": MAX_IMAGES, "video": 0}, **(engine_kw or {})}
        LettersEngine.__init__(self, model, engine_kw=engine_kw, **kw)
        self._image_setup(model)
        # warm the image path too: the vision tower and the multimodal prefill compile on their first image request
        from PIL import Image
        blank = Image.new("RGB", (512, 512), (128, 128, 128))
        for _ in range(2):
            self.answer("Warm-up.", [{"kind": "noul", "instructions": "Is this a warm-up request?"}], images=[blank])

    def facts(self):
        return {**super().facts(), "class": "multimodal", "max_images": MAX_IMAGES}
