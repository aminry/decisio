# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Image requests on the letters engine: pictures go where the request puts them, and a question reads only its own.

The full checkpoint is served under the multimodal class (Qwen3_5MoeForConditionalGeneration) as a second engine beside
the text-only default, which stays the route for every request without images (the multimodal class builds M-RoPE
positions for every prompt token, a per-request cost the text route does not pay).

A request with images is rendered as the text route renders it, with each image written as the model's chat template
writes one, `<|vision_start|><|image_pad|><|vision_end|>`, where its reference stands: in the state, or in one
question's own instructions or criteria (`decisio.serve.placed_images`, the same rules as the Gemma bases). An image in
the state is in the cached prefix, which every question after the first reads from the prefix cache; an image in a
question is in that question's own part of the prompt only, and no other question's prompt, tokens or usage hold it.
The cached prefix is front-padded to the block like a text state and ends after the state's last image. The vision
tower runs once per distinct image (vLLM keys its encoder cache and the prefix blocks by the image's hash).

Each image stays one `<|image_pad|>` in the token ids sent to the engine, which expands it to the image's token count
(the processor's grid, 16-pixel patches merged 2x2 on Qwen3.5/3.6); the padding and the cached-prefix length are
computed in expanded units, and the engine-side prompt length is returned so the expansion can be checked.

The earlier layout (`--image-prompt legacy`, deprecated): every image at the start of the user turn, "[image N]" in the
text, all images in the shared prefix and so visible to every question. It stays for a notice period
(`_answer_legacy`) and is removed in a later release.

    eng = ImageLettersEngine(os.environ["DECISIO_MODEL"], pad_to="block", pad_where="front")
    probs, info = eng.answer(f"{mark(1, tag)} The photo.", [question], images=[PIL.Image], tag=tag)
"""

from __future__ import annotations

import io
import time

from decisio.readout.letters import DEFAULT_FORMAT, fmt_state
from decisio.serve import image_marks as marks
from decisio.serve.placed_images import PlacedImageRequests, image_id
from decisio.serve.vllm_engine import USER_HEAD, LettersEngine, pad_prompt, question_text, state_prefix, user_turn

VISION = "<|vision_start|><|image_pad|><|vision_end|>"
# Images per request: no cap of its own, the context bounds a request (System One's rule). vLLM wants a number at
# start-up for its multimodal profiling: 128 images of 512 by 512 pixels (256 tokens each) fill the 32,768-token
# context, so for photos the context, not this number, ends a request; `--max-images` sets it (smaller images fit more).
DEFAULT_MAX_IMAGES = 128
MAX_BYTES = 20 * 1024 * 1024  # imajev's server limits (github.com/mohit67890/imajev)
MAX_PIXELS = 20_000_000


def load_image(data: bytes):
    """Bytes -> an RGB PIL image; PNG, JPEG or WebP; refuses oversize inputs (the imajev limits) and any image with more
    than one frame (an animated PNG or WebP)."""
    from PIL import Image

    if len(data) > MAX_BYTES:
        raise ValueError("image exceeds 20 MiB")
    try:  # only these formats are even parsed: Pillow never opens, let alone decodes, any other (EPS, for one)
        img = Image.open(io.BytesIO(data), formats=("PNG", "JPEG", "WEBP"))
    except Image.UnidentifiedImageError:
        raise ValueError("unsupported image format: PNG, JPEG or WebP") from None
    if img.format not in ("PNG", "JPEG", "WEBP"):
        raise ValueError(f"unsupported image format {img.format}")
    if getattr(img, "n_frames", 1) > 1:  # an animated PNG or WebP: System One refuses it, and so does this route
        raise ValueError("animated images are not supported")
    if img.width * img.height > MAX_PIXELS:
        raise ValueError("image exceeds 20 megapixels")
    return img.convert("RGB")


PROMPTS = ("placed", "legacy")


class ImageRequests(PlacedImageRequests):
    """Mixin for a LettersEngine: `answer(..., images=[...], tag=...)`. Needs `self.tok`, `self.processor` (the model's
    AutoProcessor) and the LettersEngine request-building attributes; the engine's `score_prompts` takes `mm` and
    `row_mm`. `image_prompt`: "placed" (images where the request puts them, the default) or "legacy" (deprecated)."""

    image_prompt = "placed"
    max_images = DEFAULT_MAX_IMAGES

    @property
    def images_placed(self) -> bool:
        return self.image_prompt != "legacy"

    def _image_setup(self, model, revision=None):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model, revision=revision)
        to_id = self.tok.convert_tokens_to_ids
        self.vision_start_id, self.image_token_id, self.vision_end_id = (
            to_id(t) for t in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")
        )
        self.image_pad_id = self.image_token_id
        self.merge = self.processor.image_processor.merge_size

    # what this base says to the shared image path (`decisio.serve.placed_images`)
    placeholder_tail = 1  # the end-of-vision token follows the pad token

    @property
    def placeholder_ids(self) -> list[int]:
        return [self.vision_start_id, self.image_token_id, self.vision_end_id]

    image_id = staticmethod(image_id)

    def pad_tokens(self, img) -> int:
        """The pad tokens the engine expands one image to: its processor's resize (`smart_resize`, the processor's own
        function and pixel bounds) in merged patches, without running the processor (tests/unit/test_image_route.py,
        I1, checks the two agree)."""
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        ip = self.processor.image_processor
        unit = ip.patch_size * ip.merge_size
        h, w = smart_resize(
            img.height, img.width, factor=unit, min_pixels=ip.size["shortest_edge"], max_pixels=ip.size["longest_edge"]
        )
        return (h // unit) * (w // unit)

    def image_tokens(self, img) -> int:
        """What one image adds to `usage.input_tokens`: its pad tokens and the two vision delimiter tokens around them,
        what the engine receives (the Gemma bases count their begin and end tokens the same way)."""
        return self.pad_tokens(img) + 2

    def expanded_tokens(self, img) -> int:
        """One placeholder (`<|vision_start|><|image_pad|><|vision_end|>`) becomes the pad token repeated for the
        image's tokens, with the two vision tokens around them."""
        return self.image_tokens(img)

    def image_fmt(self):
        """The image route writes its prompts in the default format (it has no layout, and `--prompt-tail` is not
        applied to a photo prompt: the format its figures were measured with)."""
        return DEFAULT_FORMAT

    def warms_single_question(self) -> bool:
        return False  # the route has warmed a state's prefix for requests of two or more questions only

    def answer(self, state, questions, adapter=None, images=None, tag=None):
        if not images:
            return PlacedImageRequests.answer(self, state, questions, adapter)  # the text route, unchanged
        if self.image_prompt == "legacy":
            return self._answer_legacy(state, questions, adapter, images)
        if tag is None:
            # a library call with no marks in its text: the images open the state, as the server writes the images a
            # request lists without a place
            tag = marks.new_tag()
            head = "".join(marks.mark(i, tag) for i in range(1, len(images) + 1))
            state = f"{head}\n{fmt_state(state)}"
        return PlacedImageRequests.answer(self, state, questions, adapter, images, tag)

    # ---- the legacy layout (deprecated) -------------------------------------------------------------

    def _prepare_images_legacy(self, state, images, questions):
        """(rows with unexpanded image placeholders, P in expanded units, P in sent units, per-image token counts)."""
        enc = lambda s: self.tok.encode(s, add_special_tokens=False)  # noqa: E731
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
        counts = [self.pad_tokens(img) for img in images]  # the deprecated layout counted the pad tokens only
        if prefix.count(self.image_pad_id) != len(images):
            raise ValueError("the image placeholders are not all in the cached prefix")
        extra = sum(c - 1 for c in counts)  # what the engine's expansion adds to the prefix
        n_exp = n + extra
        k = (-n_exp % self.pad_unit) if self.pad_unit else 0
        head = len(enc(USER_HEAD))
        rows = [(pad_prompt(ids, n, k, self.pad_token, self.pad_where, head), lab) for ids, lab in rows]
        return rows, n_exp + k, n + k, counts

    def _answer_legacy(self, state, questions, adapter, images):
        limit = getattr(self, "max_images", DEFAULT_MAX_IMAGES)
        if len(images) > limit:
            raise ValueError(f"at most {limit} images per request")
        with self._lock:
            t0 = time.perf_counter()
            rows, p_exp, p_sent, counts = self._prepare_images_legacy(state, images, questions)
            unit = min(self.match_unit, self.pad_unit or self.match_unit)
            # as for text: one warm-up request of the prefix plus one token registers the state (and the images)
            warm = [rows[0][0][: p_sent + 1]] if len(rows) > 1 and p_exp >= unit else []
            ids = [self.image_id(img) for img in images]
            prep_ms = (time.perf_counter() - t0) * 1000
            probs, info = self.score_prompts(rows, adapter, warm, mm={"image": list(images)}, mm_uuids={"image": ids})
            info.update(
                shared_prefix_tokens=p_exp,
                image_tokens=counts,
                prepare_ms=prep_ms,
                questions=len(rows),
                server_ms=(time.perf_counter() - t0) * 1000,
            )
        return probs, info


class ImageLettersEngine(ImageRequests, LettersEngine):
    """The multimodal class on vLLM with as many images per request as the context holds (up to `max_images`, 128 unless
    `--max-images` says otherwise); every text-path setting as the served default."""

    def __init__(self, model, engine_kw=None, max_images=DEFAULT_MAX_IMAGES, image_prompt="placed", **kw):
        if image_prompt not in PROMPTS:
            raise ValueError(f"--image-prompt: one of {', '.join(PROMPTS)}, not {image_prompt}")
        self.image_prompt = image_prompt
        self.max_images = int(max_images)
        engine_kw = {"limit_mm_per_prompt": {"image": self.max_images, "video": 0}, **(engine_kw or {})}
        LettersEngine.__init__(self, model, engine_kw=engine_kw, **kw)
        self._image_setup(model, kw.get("revision"))
        # warm the image path too: the vision tower and the multimodal prefill compile on their first image request
        self.warm_images()

    def facts(self):
        return {
            **super().facts(),
            "class": "multimodal",
            "max_images": self.max_images,
            "image_prompt": self.image_prompt,
        }
