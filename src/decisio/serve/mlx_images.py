# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Images on a Mac (`--backend mlx --images on`): the Gemma 4 12B reads pictures with the engine that reads text.

The request path is the vLLM server's (`decisio.serve.gemma_images`): an image stands where its reference stands, in
the state or in one question's own text; a state's images sit in the shared prefix, which every question reads from the
prefix cache, and a question's images in that question's suffix alone; the usage counts come from the processor's
arithmetic (`decisio.serve.image_marks`). This engine replaces the part that needs vLLM: the vision embedder and the
forward pass.

The model is loaded with mlx-vlm (the `mlx-images` extra), which holds the vision embedder of the 12B's encoder-free
`gemma4_unified` architecture. The pictures are prepared by transformers' own image processor, the one the vLLM path's
token counts are checked against; each `<|image|>` placeholder is expanded here to the begin-of-image token, the image's
soft tokens and the end-of-image token, and the soft tokens' embeddings come from the embedder.

Bidirectional image blocks. On the sliding-window layers the tokens of one image attend to each other, not only to the
tokens before them (`use_bidirectional_attention = "vision"`, from the token types the processor returns beside the
ids). mlx-vlm 0.7.12's `Model.__call__` does not pass the token types to the language model, so called as it is, every
image stays causal and the model loses the vertical axis (a square in the top corners is read as "bottom"). And its
language model builds the bidirectional overlay only where the keys are the query's own tokens, so an image that
follows a cached prefix (a question's image after the state) would stay causal even with the token types passed.
This engine therefore calls the language model's backbone itself: the embeddings are built here, and for the call the
layers' cache entries make their attention masks with the image blocks' overlay (`_vision_masks`: the mask the cache
makes, or-ed with "same image block" over the call's own tokens). It is the cache entries of one call that are changed,
for the call only; nothing in mlx-vlm is replaced. A prompt is fed in chunks that never cut an image block, so the
overlay never needs keys from an earlier call. `tests/unit/test_mlx_images.py` pins it: the token types make a
difference, and a cached read equals a cold read.

The prefix cache keeps a state with its images (`PrefixCache`, keyed by the expanded token ids and the images'
content); a second question about the same state continues from a copy, as for text. The cache boundary is after the
state's last image's end token by construction: the prefix ends where the first question's own tokens begin.

Memory. MLX's cache limit (`--mlx-cache-limit-gb`, 2 GiB) bounds the buffers it keeps for reuse, and the engine clears
them after every request that carried images; the family's `mlx_working_images_gb` sizes the memory limit.

Box (a bounding box read) is not served on the Mac yet. `generate` is what a Mac Box will build on: a greedy decode
on the image prompt that returns the top-k log-probabilities at every step.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import numpy as np

from decisio.readout.letters import allowed_ids, is_grouped, label_log_softmax
from decisio.serve import image_marks as marks
from decisio.serve.engine_health import guarded
from decisio.serve.gemma_images import DEFAULT_MAX_IMAGES, GemmaImageRequests
from decisio.serve.mlx_engine import MLXLettersEngine, _cache_arrays, copy_cache
from decisio.serve.mlx_memory import MemoryRefused

BEGIN_IMAGE, END_IMAGE = "<|image>", "<image|>"
# the keys of the vision embedder's weights in a checkpoint: the embedder and the projection into the text model
VISION_KEYS = ("vision_embedder", "embed_vision")
BOX_REFUSED = (
    "Box is not served with --backend mlx yet: the Mac reads images for noul, choice and score questions only. "
    "Serve the base with the vLLM server for Box"
)


class ImagesRefused(MemoryRefused):
    """The conversion cannot read images (the server refuses to start, with the reason, as it does for memory)."""


def weight_names(model_dir) -> set[str]:
    """The tensor names of a conversion: its index file's, or the header of its safetensors file."""
    d = Path(model_dir)
    index = d / "model.safetensors.index.json"
    if index.exists():
        return set(json.loads(index.read_text())["weight_map"])
    names = set()
    for f in sorted(d.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            names |= {k for k in json.loads(fh.read(n)) if k != "__metadata__"}
    return names


def vision_refusal(model_dir) -> str | None:
    """Why `--images on` cannot serve the conversion in `model_dir`, or None when it can: it must be the 12B's
    architecture (`gemma4_unified`) and carry the vision embedder and its projection."""
    d = Path(model_dir)
    config = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    kind = config.get("model_type")
    names = weight_names(d)
    have = {k: any(k in n for n in names) for k in VISION_KEYS}
    missing = [k for k, ok in have.items() if not ok]
    if kind != "gemma4_unified":
        if kind == "gemma4":
            if missing or not config.get("vision_config"):
                return (
                    f"--images on: this conversion ({d.name}) is the 31B's architecture without its vision tower "
                    "(no vision tensors in it, no vision_config), which an MLX conversion made for text drops. "
                    "A Mac reads images with a conversion that keeps the tower; none is published yet"
                )
            return "--images on on a Mac serves the 12B (gemma4_unified) only: the 31B's vision tower is not served yet"
        return f"--images on is for the Gemma 4 bases, and this conversion's model_type is {kind!r}"
    if missing or not config.get("vision_config"):
        what = ", ".join(missing) if missing else "vision_config in its config.json"
        return (
            f"--images on: this conversion ({d.name}) has no vision weights (missing: {what}); a conversion made for "
            "text drops the 12B's vision embedder. Serve google/gemma-4-12B-it (bf16) or an MLX conversion that keeps "
            "vision_embedder and embed_vision, such as mlx-community/gemma-4-12B-it-6bit"
        )
    return None


class MLXGemmaImageLettersEngine(GemmaImageRequests, MLXLettersEngine):
    """The MLX letters engine of the Gemma 4 12B with image input on.

    `model`: a directory or Hub repository of a conversion with the vision weights (`vision_refusal`). `random`: a
    small config (`{"text": {...}, "vision": {...}}`) to build a model of the same architecture with random weights
    instead of loading `model`, for the tests (no download of weights)."""

    score_takes_prefix = True
    box_enabled = False  # Box is not served on a Mac yet (`box_blocked`)

    def __init__(self, model, image_budget=marks.DEFAULT_BUDGET, max_images=DEFAULT_MAX_IMAGES, random=None, **kw):
        if image_budget not in marks.BUDGETS:
            raise ValueError(f"--image-budget: one of {', '.join(map(str, marks.BUDGETS))}, not {image_budget}")
        self._random = random
        MLXLettersEngine.__init__(self, model, **kw)
        self._image_setup(image_budget, max_images)
        self.boi = self.tok.convert_tokens_to_ids(BEGIN_IMAGE)
        self.eoi = self.tok.convert_tokens_to_ids(END_IMAGE)
        # the match unit is the vLLM cache's block; this cache hits at any token, so the block rule (a prefix that
        # ends inside an image block is padded to a block boundary) has nothing to do here
        self.match_unit = 1
        self._processor = None
        lm = self.model.language_model.model
        if getattr(lm, "hidden_size_per_layer_input", 0):
            raise ValueError("--images on: a checkpoint with per-layer inputs (the small Gemma 4 models) is not served")
        self._embed = lm.embed_tokens
        self._embed_scale = lm.embed_scale
        self._warm_images()

    # ---- loading -------------------------------------------------------------------------------

    def _check_model_dir(self, model_dir):
        if self._random is None and (why := vision_refusal(model_dir)):
            raise ImagesRefused(why)

    def _working_gb(self):
        return getattr(self.family, "mlx_working_images_gb", 0) or None

    def _load_model(self, model_dir):
        if self._random is not None:
            return self._random_model(self._random)
        from mlx_vlm import load

        return load(str(model_dir))[0]

    @staticmethod
    def _random_model(config):
        """A model of the 12B's architecture, small, with random weights."""
        import mlx.core as mx
        from mlx.utils import tree_map
        from mlx_vlm.models.gemma4_unified import Model, ModelConfig

        text = {
            "model_type": "gemma4_text",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "global_head_dim": 16,
            "sliding_window": 64,
            "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
            "vocab_size": 262144,
            "vocab_size_per_layer_input": 262144,
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
            "final_logit_softcapping": 30.0,
            "use_bidirectional_attention": "vision",
            **config.get("text", {}),
        }
        vision = {
            "model_type": "gemma4_unified_vision",
            "mm_embed_dim": 32,
            "output_proj_dims": 32,
            "mm_posemb_size": 128,
            "model_patch_size": 48,
            "patch_size": 16,
            "pooling_kernel_size": 3,
            **config.get("vision", {}),
        }
        cfg = ModelConfig.from_dict(
            {
                "model_type": "gemma4_unified",
                "text_config": text,
                "vision_config": vision,
                "image_token_id": 258880,
                "boi_token_id": 255999,
                "eoi_token_id": 258882,
            }
        )
        mx.random.seed(int(config.get("seed", 0)))
        model = Model(cfg)
        # random.normal weights: the default initialisation is zeros for some tensors, which would hide every effect
        model.update(tree_map(lambda p: mx.random.normal(p.shape) * 0.4 if p.ndim else p, model.parameters()))
        mx.eval(model.parameters())
        return model

    def _image_processor(self):
        if self._processor is None:
            from transformers.models.gemma4_unified.image_processing_gemma4_unified import Gemma4UnifiedImageProcessor

            self._processor = Gemma4UnifiedImageProcessor()
        return self._processor

    def _warm_images(self):
        """The first picture compiles the vision path: do it before the first request."""
        from PIL import Image

        blank = Image.new("RGB", (512, 512), (128, 128, 128))
        self.answer(
            f"{marks.mark(1, 'warm')} Warm-up.",
            [{"kind": "noul", "instructions": "Is this a warm-up request?"}],
            images=[blank],
            tag="warm",
        )

    def facts(self):
        from importlib.metadata import version

        return {**super().facts(), "mlx_vlm": version("mlx-vlm"), "images": self.image_facts()}

    def box_blocked(self):
        return BOX_REFUSED

    # ---- the forward pass with images --------------------------------------------------------------

    @contextlib.contextmanager
    def _vision_masks(self, cache, block):
        """For one call: every cache entry makes its attention mask with the image blocks of the call's own tokens
        attending to each other (`block`: one id per token, the image's number or -1). The entries' `make_mask` is the
        library's; the overlay is or-ed onto the mask it makes (key columns: the cache's kept keys, then the call's
        own tokens, so the block's columns are the last N)."""
        import mlx.core as mx

        b = mx.array(block)
        same = (b[:, None] == b[None, :]) & (b[:, None] >= 0)
        n = len(block)

        def overlaying(make):
            def make_mask(N, *args, **kw):
                kw["return_array"] = True
                m = make(N, *args, **kw)
                if m is None or N != n:
                    return m
                k = m.shape[-1]
                return m | mx.pad(same.astype(mx.uint8), [(0, 0), (k - n, 0)]).astype(mx.bool_)

            return make_mask

        entries = [c for c in cache if hasattr(c, "make_mask")]
        for c in entries:
            c.make_mask = overlaying(c.make_mask)
        try:
            yield
        finally:
            for c in entries:
                del c.make_mask  # back to the class's own

    def _processed(self, img):
        """(pixel values, position ids, token count) of one picture by the Gemma 4 image processor."""
        import mlx.core as mx

        enc = self._image_processor()(images=[img], max_soft_tokens=self.image_budget, return_tensors="np")
        return (
            mx.array(enc["pixel_values"]),
            mx.array(enc["image_position_ids"]),
            int(enc["num_soft_tokens_per_image"][0]),
        )

    def _chunks(self, ids):
        """[(start, end)] of the prompt's chunks: at most `prefill_step` tokens, never ending inside an image block."""
        out, a, n, tid = [], 0, len(ids), self.image_token_id
        while a < n:
            b = min(a + self.prefill_step, n)
            while b < n and ids[b - 1] == tid and ids[b] == tid:
                b += 1
            out.append((a, b))
            a = b
        return out

    def _feed_images(self, ids, cache, images):
        """Feed the expanded `ids` into `cache`; the final-norm hidden state at the last position. `images`: the PIL
        pictures of the image blocks in `ids`, in order."""
        import mlx.core as mx

        tid, h, used = self.image_token_id, None, 0
        for a, b in self._chunks(ids):
            part = ids[a:b]
            runs, i = [], 0
            while i < len(part):
                if part[i] == tid:
                    j = i
                    while j < len(part) and part[j] == tid:
                        j += 1
                    runs.append((i, j))
                    i = j
                else:
                    i += 1
            emb = self._embed(mx.array([part])) * self._embed_scale
            pieces, at, block = [], 0, np.full(len(part), -1, dtype=np.int32)
            for r, (i, j) in enumerate(runs):
                px, pos, n = self._processed(images[used + r])
                if n != j - i:
                    raise RuntimeError(
                        f"the processor made {n} tokens for an image the prompt has {j - i} for: the image token "
                        "counts do not match this checkpoint's processor"
                    )
                feats = self.model.get_image_features(px, pos).astype(emb.dtype)
                pieces += [emb[:, at:i], feats[None]]
                block[i:j] = r
                at = j
            used += len(runs)
            pieces.append(emb[:, at:])
            x = mx.concatenate(pieces, axis=1) if runs else emb
            with self._vision_masks(cache, block) if runs else contextlib.nullcontext():
                out = self._backbone(None, inputs_embeds=x, cache=cache)
            h = out[0, -1]
            mx.eval(h, *_cache_arrays(cache))
        if used != len(images):
            raise ValueError("the prompt's image blocks and its pictures differ")
        return h

    def _expand(self, ids, counts):
        """`ids` with each `<|image|>` placeholder expanded to the begin token, the image's soft tokens and the end
        token."""
        out, k = [], 0
        for t in ids:
            if t == self.image_token_id:
                out += [self.boi] + [t] * counts[k] + [self.eoi]
                k += 1
            else:
                out.append(t)
        if k != len(counts):
            raise ValueError("a row's placeholders and its images differ")
        return out

    def _count(self, img):
        return marks.soft_tokens(img.height, img.width, self.image_budget)

    def _logits(self, h, ids=None):
        z = self._head(h[None])[0]
        if ids is not None:
            z = z[self._mx.array(ids)]
        if self.softcap is not None:
            z = self._mx.tanh(z / self.softcap) * self.softcap
        return z

    def _prefix_of(self, rows, prefix, state_images, state_ids=None):
        """(the prefix's cache, its time in ms, whether it came from the prefix cache, the expanded prefix)."""
        t0 = time.perf_counter()
        pre = self._expand(rows[0][0][:prefix], [self._count(i) for i in state_images])
        salt = "|".join(state_ids if state_ids is not None else [marks_id(i) for i in state_images])
        cache = self.prefix_cache
        base = cache.get(pre, 0, salt=salt) if cache is not None else None
        hit = base is not None
        if not hit:
            base = self.model.make_cache()
            self._feed_images(pre, base, state_images)
            if cache is not None:
                cache.put(pre, base, 0, salt=salt)
        return base, (time.perf_counter() - t0) * 1000, hit, pre

    def _score_images(self, rows, prefix, row_images, state_n, state_ids=None):
        """[(label log-probabilities float64, hidden state float32)] per row, the prompts' expanded lengths, the
        prefix's time, whether it came from the cache. `row_images[i]`: row i's pictures, the state's first."""
        mx = self._mx
        state_images = list(row_images[0][:state_n]) if rows else []
        if any(ids[:prefix] != rows[0][0][:prefix] for ids, _ in rows):
            raise ValueError("the rows do not share their first `prefix` tokens")
        lens, expanded = [], []
        for (ids, _), imgs in zip(rows, row_images):
            e = self._expand(ids, [self._count(i) for i in imgs])
            expanded.append(e)
            lens.append(len(e))
        self.refuse_over_window([(e, lab) for e, (_, lab) in zip(expanded, rows)])
        base, prefix_ms, hit, pre = self._prefix_of(rows, prefix, state_images, state_ids)
        out = []
        for e, (ids, lab), imgs in zip(expanded, rows, row_images):
            h = self._feed_images(e[len(pre) :], copy_cache(base), imgs[state_n:])
            z = np.array(self._logits(h, allowed_ids(lab)).astype(mx.float32), dtype=np.float64)
            if is_grouped(lab):
                lp = label_log_softmax(z, lab)
            else:
                lp = z - z.max()
                lp -= np.log(np.exp(lp).sum())
            out.append((lp, np.array(h.astype(mx.float32))))
        return out, lens, prefix_ms, hit, len(pre)

    # ---- the engine interface ---------------------------------------------------------------------

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
        prefix=None,
        context=None,
    ):
        """As the MLX engine's, plus `row_mm`: per row {"image": [PIL images]} (the state's images first), whose
        placeholders in the rows' ids the engine expands. `prefix` (the sent prefix length) is the request's."""
        if adapter:
            raise ValueError("the MLX engine serves no adapters")
        if row_mm is None:
            return MLXLettersEngine.score_prompts(
                self, rows, adapter, warm, skip_cache, mm, mm_uuids, prefix=prefix, context=context
            )
        if prefix is None:
            raise ValueError("a request with images gives its shared prefix length")
        t0 = time.perf_counter()
        row_images = [list((m or {}).get("image") or []) for m in row_mm]
        state_n = rows[0][0][:prefix].count(self.image_token_id)
        state_ids = list((row_mm_uuids[0] or {}).get("image") or [])[:state_n] if row_mm_uuids else None
        scored, lens, prefix_ms, hit, p_exp = guarded(
            self, self._score_images, rows, prefix, row_images, state_n, state_ids
        )
        self._mx.clear_cache()  # the buffers of the vision path: the cache limit keeps what is reused, no more
        probs = [np.exp(lp) / np.exp(lp).sum() for lp, _ in scored]
        return probs, {
            "warm_ms": 0.0,
            "prefix_ms": prefix_ms,
            "prefix_cache_hit": hit,
            "cached_tokens_mean": float(p_exp),
            "engine_timing": [],
            "prompt_tokens_mean": float(np.mean(lens)),
            "prompt_tokens": sum(lens),
            "engine_prompt_tokens": lens,
            "cached_tokens": [p_exp] * len(rows),
            "forward_ms": (time.perf_counter() - t0) * 1000,
        }

    def generate(self, ids, images=(), max_new_tokens=32, top_k=20, prefix=None, stop_ids=()):
        """A greedy decode on a prompt with images, for a reader that needs the model's own words (Box): the steps as
        [{"token": id, "top": [(id, log-probability), ...]}], `top_k` candidates at every step with the softcap applied
        and the log-softmax over the whole vocabulary. `ids`: the sent ids (one placeholder per image) and `images`
        their pictures; `prefix`: the shared prefix length, whose cache entry is kept and reused."""
        mx = self._mx
        with self._lock:
            e = self._expand(ids, [self._count(i) for i in images])
            cache = self.model.make_cache()
            k = ids[:prefix].count(self.image_token_id) if prefix else 0
            if prefix:
                base, _, _, pre = self._prefix_of([(ids, None)], prefix, list(images[:k]))
                cache = copy_cache(base)
            else:
                pre = []
            h = self._feed_images(e[len(pre) :], cache, list(images[k:]))
            steps = []
            for _ in range(max_new_tokens):
                z = self._logits(h).astype(mx.float32)
                lp = z - mx.logsumexp(z)
                top = mx.argsort(-lp)[:top_k]
                tl = np.array(lp[top])
                ti = [int(t) for t in np.array(top)]
                steps.append({"token": ti[0], "top": list(zip(ti, [float(x) for x in tl]))})
                if ti[0] in stop_ids:
                    break
                h = self._feed(ti[:1], cache)
            mx.clear_cache()
        return steps


def marks_id(img) -> str:
    """A content id of a picture, for the prefix cache's key: the decoded pixels."""
    from decisio.serve.image_engine import ImageRequests

    return ImageRequests.image_id(img)
