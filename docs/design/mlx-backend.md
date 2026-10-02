<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Design: an MLX engine for the text route on a Mac

`--backend mlx` serves decisio's text route on Apple silicon from an MLX conversion of the served checkpoint.
Every route and feature of the text route works unchanged:
- the letters readout;
- the state prefix shared across a request's questions;
- the global temperature;
- task registration with calibration and the intent head;
- abstention;
- the rendering rules and the tie-break.

The image route, packed mode, LoRA adapters and the second-engine head are refused with this backend.

    uv sync --extra mlx
    uv run python -m decisio.serve.vllm_engine --backend mlx --model mlx-community/Qwen3.6-35B-A3B-4bit

## What is replaced, and what is not

`MLXLettersEngine` (`decisio.serve.mlx_engine`) subclasses `vllm_engine.LettersEngine`, as the CPU stand-in does, and replaces only model loading and the forward pass.
The request path before that is the served code: state tokenization, the split into state and question, the padding, the label tokens, derived questions, both routes, the temperature, tasks and abstention.

The prompts are the vLLM path's, byte for byte:
- **Tokenizer:** the official one (`--tokenizer`, default `Qwen/Qwen3.6-35B-A3B-FP8`). The mlx-community conversions ship a different `tokenizer.json` and `tokenizer_config.json`; the chat template and the vocabulary are the same. `GET /health` reports the tokenizer file's sha256 and whether it is the official one.
- **Padding:** the served default's, front padding to vLLM's 1,056-token block. MLX's cache does not need it, but the served prompts carry it, and the accuracy records were made with it; `--pad-to none` leaves it out.

## The forward pass

For each request, the state prefix (the tokens every question shares, padding included) is prefilled once into a fresh prompt cache, in chunks of 2,048 tokens, and evaluated.
Each question then continues from a copy of that cache, one question at a time.

Copying the cache depends on the layer type:
- **Gated DeltaNet layers:** their cache entries (the convolution and recurrent states) are replaced by each forward, never written into, so a copy shares them.
- **Attention layers:** they write their keys and values into a buffer in place, so a copy slices the buffer to the prefix, and its first write allocates a new one.

Every question goes through the prefix, single-question requests included:
- **Isolation:** a question's answer does not depend on what else is in its request. Each question alone, the same question beside 16 others, and a repeat all agree bit for bit (`tests/mlx/serving_gates.py`, G2).
- **No batch effect:** the vLLM engine scores a request's questions in one batch and cannot make that promise (EVAL_CARD.md, the batch-forward finding).

The letters are read at the last position: the softmax over the label logits (bf16 out of the output layer, then float64).
The final-norm hidden state at the same position comes out of the same forward, so the intent head (`MLXHiddenReadout`) fits and serves on exactly the served readout, with no extra request and no second weight copy.

## The prefix path against the whole prompt

Scoring a question from the prefix cache is not bit-identical to running its whole prompt in one pass.
The bf16 arithmetic is chunked differently, which is the same effect vLLM's cached-against-cold gate (G1) bounds.
The MLX engine always serves the prefix path, so it never serves the other one; the difference is recorded, not gated (numbers below).

## Gates

Measured on an Apple M5 Pro (64 GB) against the FP8 served default's records (`runs/2026-09-30_plugin-verification`), pre-registered before any measurement.
The record is in the RLCD experiment `2026-10-02_t2_mlx_backend`.

(Numbers to be filled from the records when the gates are in.)
