<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Images on Gemma 4

`--images on` lets a Gemma base read pictures with the engine that reads its text (`decisio.serve.gemma_images`).
This note says how a request's images get into the prompt, who sees them, where the cache ends and what the model does with them.
The wire format is in `docs/api.md`; the flags are in `docs/running.md`.

## One engine

The 12B (`gemma4_unified`) has no vision tower: its image path is a linear projection of raw 48 by 48 pixel patches, already part of vLLM's multimodal class, which the text route runs under with every multimodal input off.
`--images on` raises the image limit and nothing else.
The 31B (`gemma4`) has a SigLIP-style tower that its text route leaves out by loading vLLM's text class.
`--images on` loads the multimodal class instead, so one set of weights serves text and images.
A second engine, as the Qwen route has, would load the 31B twice.
The text-only class's weight rule (`decisio.vllm_plugin.weights.without_vision`) matches Qwen's `visual.` names only, and the Gemma classes do not use it: vLLM's own Gemma 4 text class skips the checkpoint's `vision_tower`, `embed_vision`, `audio_tower` and `embed_audio` tensors itself, so nothing in the 31B path depends on that rule.

## Where an image stands

`bind_assets` and the `data:image` extraction write a mark (`decisio.serve.image_marks`) where each reference stood, in the state or in one question's own text, and keep the image's number.
A mark is printable, because a nested state is rendered with `str`, which escapes a character that is not.
It also carries a tag made fresh for each request (16 random hex characters, `⟦image:<tag>:N⟧`) that only the server knows, and the engine matches that request's exact tag.
Text a user writes that looks like a mark, with another tag or none, is plain text and is never refused.
The request's tag is never part of what the model reads.
The engine renders the text exactly as the text route does (system turn, header, spaced tail, answer slot, labels), cuts the rendered text at the marks, tokenizes each part and puts the model's single `<|image|>` placeholder token at each mark.
vLLM expands each placeholder to the begin-of-image token, the image's soft tokens and the end-of-image token.
Because only the marks differ, a request with images and the same request without them have the same tokens apart from the images.

## Who sees an image

An image in the state is in the shared prefix.
An image in a question's instructions or criteria is in that question's row after the prefix.
Each row is handed to the engine with the images its own tokens carry: the state's, then its own.
The header layout lists every question's instructions in the prefix, so a question's image is written there as `[image N]` and is pixels only in its own row.
`usage.input_tokens` is the request's text without the marks plus the images' tokens: the state's once, each question's own once.

## Where the cache ends

The prefix is the header, the state and the blank line before the question, so it ends after the state's last image's end token whenever any text follows the image, which the blank line always is.
vLLM's cache hits whole blocks, and the state's cache entry is the last whole block of the prefix.
If that block ended inside the last image, a later request would resume from a state the image's own bidirectional attention had not finished.
So when the last whole block of the prefix lies before the end of the last state image, the prefix is extended with newline tokens up to the block boundary (the state's own text is not changed, and a state with enough text after its image needs none).
A prefix hit that begins or ends inside an image of a question's own text is left to vLLM and is among the things the first card read checks.

## Bidirectional image blocks

Gemma 4's text config says `use_bidirectional_attention = "vision"`: on the sliding-window layers the tokens of one image attend to each other, not only to the tokens before them; the global layers stay causal.
The model builds the blocks from `mm_token_type_ids`, 1 on image tokens and 0 elsewhere (the begin and end tokens are text), and each image is its own block.
vLLM 0.31.0 does this for the tokens it expands, through its multimodal prefix ranges, clamped to the sliding window.
A runtime that drops the token types reads an image causally and loses its vertical axis; a Mac port has to pass them itself.
The CPU stand-in passes them, a test asserts what reaches its forward pass, and another shows that dropping them changes the readout.
A real checkpoint test (a square in each corner, answering top or bottom and left or right) is kept beside the others and runs when the local Gemma 4 E2B is named by an environment variable.

## Temperature

The text temperatures were fitted on text.
A base's `image_temperature` replaces them for requests with images and defaults to none, which reads images at the text temperatures.
An adapter keeps its own temperatures on every request.
