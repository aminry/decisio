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
The rule on a sliding layer is the window AND (causal OR the same image block), as in the reference implementation: the window is not lifted inside an image, so tokens of a block more than one window apart do not see each other.
A block longer than the window would therefore leave its later tokens unable to see its top rows, and the readout would lose the top of the picture.
That matters only where the language model's attention is all that links the parts of a picture: on the 12B, which embeds patches straight into the language model.
The 31B reads pictures through its own 27-layer vision encoder, which attends over the whole picture, so every image token already carries all of it and a block longer than the window loses nothing (measured: at 1120 the 31B reads a square in each corner correctly, where the 12B reads the top ones as bottom).
So on a base without a vision encoder (`vision_config` declares no `num_hidden_layers`), the server refuses a budget whose largest block (the budget's soft tokens plus the begin and end tokens) is longer than the sliding window, read from the checkpoint's `text_config.sliding_window`: 1120 (up to 1,122 tokens) against the 12B's 1,024-token window, while 560 (up to 562) fits; the 31B takes every budget.
It refuses rather than lowers the budget, so the operator sees what the server does; a base with a longer window allows more.
The default budget (280) is still under measurement.
A runtime that drops the token types reads an image causally and loses its vertical axis; a Mac port has to pass them itself.
The CPU stand-in passes them, a test asserts what reaches its forward pass, and another shows that dropping them changes the readout.
A real checkpoint test (a square in each corner, answering top or bottom and left or right) is kept beside the others and runs when the local Gemma 4 E2B is named by an environment variable.

## Temperature

The text temperatures were fitted on text.
A base's `image_temperature` replaces them for requests with images and defaults to none, which reads images at the text temperatures.
An adapter keeps its own temperatures on every request.

## Box questions

A Box question asks where a target is, and the answer is one box.
The wire is in `docs/api.md`; this section is what the server does.

**The read.**
Each Box is one greedy generation on its own prompt: the state with its images (the cached prefix), then the question's own text and images, rendered with the model's chat template, the thought channel closed, and a reply that is forced to begin `{"box_2d": [`.
The instruction asks for one box as JSON in Gemma's documented format, `[y_min, x_min, y_max, x_max]` on a 0-1000 grid.
Forcing the start means the first tokens are the digits of `y_min`, and a reply is one box, or `null` for no box.
At most 32 tokens are generated, at temperature 0, with the top 20 log-probabilities of each step.
The Boxes of a request are one batch, and their prefix is the one the request's yes/no reads use, so it is read once.
Under the header layout the prefix carries a list of the request's questions, which a generation does not want, so every Box and its presence read is read under the plain layout, whatever the server's default.
A request that asks for `"layout": "header"` together with a Box is refused.
A mixed request on a header-layout server reads the state twice, once for the other questions under the header layout and once for the Box and its presence read under the plain one; the prefix cache keeps each read to its own prefix.

**The expected value.**
Gemma writes each digit as its own token.
For each coordinate the server follows the greedy path and, at every digit, takes the distribution over the next digits and the end of the number from the logprobs, with mass under 0.01 dropped.
A branch off the greedy path with probability at least 0.05 is decoded for real: one short extra greedy generation from that branch's digit to its terminator, in the same batch as the request's other branches, on the cached prefix (`decisio.readout.box.expand_branches`).
A branch below that probability keeps the cheaper approximation, the greedy path's own expected lower digits, which is exact when only the last digit and the end are uncertain and otherwise has a real but small error; the real decodes fix the larger errors, a digit-count ambiguity (a `9,` read as the end against a `95,` that keeps going).
At most 16 branches are decoded per Box question; past the cap, the rest keep the approximation and the server's timing says so (`branches_capped`).
The edge is the expected value, clipped to the grid and ordered; a degenerate side is widened to one grid unit.
The probability that an edge lies within 0.02 of the image of the reported value comes from the same distribution, and the raw confidence is the geometric mean of that over the four edges.
The estimator sits behind one function (`estimate_coordinate`, branch by branch), so a decoded branch's real value and the approximation share one code path; `assemble_box` turns four edges into the box, flags and raw confidence, whichever estimator produced them.

**The exact-tree debug readout.**
A card session needs to check the branch-expanded estimator itself against the full digit tree: `--debug-box-tree` on the server, `x-decisio-debug: box_tree` on a request, turn on one more read per edge, a real recursive expansion of the tree (`exact_coordinate_tree`).
Along the greedy path the digit and end probabilities are already known from the main generation's own logprobs, read at a finer cut (1e-4) than the served estimator keeps (0.01); off it, each node visited is one more real decode, at most 64 per coordinate.
The response's debug object then carries, per edge, the exact expected value, the served (branch-expanded) estimator's own value, the plain greedy value, and the tree mass actually covered (under 1.0 only when the cap cut a branch short).
This is an operator's tool for one check, not part of the public answer: it costs a real decode per tree node and is never on by itself.

**The confidence.**
The raw edge confidence alone separates a present target from an absent one poorly, because the forced reply is confident in its digits whatever the picture shows.
The server therefore reads presence separately: one yes/no question, "Is this in the picture: <the Box's instructions>?", in the same request on the same prefix, through the ordinary noul readout and temperature, which costs one forward pass.
The confidence is the presence probability times the raw edge confidence.
A calibration, when the served model has one, is an isotonic map (pool adjacent violators) from that product to the rate at which a box has IoU of 0.5 or more with the truth on a calibration set of present and absent targets; it is a file of knots loaded with `--box-confidence` or found beside the model's weights.
Until then the product is served as it is and the `x-decisio-box` header says `calibrated=false`.
Isotonic was chosen over a two-feature logistic fit because it needs no assumption about how the two factors combine and is monotone by construction.

**A target the model calls absent.**
The wire has no "not found" form: a Box answer always carries a box.
When the reply holds `null` where the list, or its first coordinate, would be (`{"box_2d": null`, `[null`), with no coordinate before it, the readout returns an absent result instead of an error.
The answer is then the whole image at confidence 0.0, `absent` is added to the `x-decisio-box` header and to the question's `timing.box` entry, the presence read is read as usual, and the tokens the reply wrote count in usage.
Branch expansion and the exact-tree debug readout skip the question; the debug readout says `{"absent": true}` for it.
Any other unreadable reply is still a 502.

**Cost and counts.**
`usage.output_tokens` adds the tokens a Box wrote: the main reply, and every decoded branch's continuation.
The batch's generation time is the `box` entry of `x-decisio-stages`; `x-decisio-box` carries whether the confidence is calibrated, the tokens written and the readout's flags (`clipped`, `swapped_x`, `swapped_y`, `degenerate_x`, `degenerate_y`); under a debug readout the presence probability, the raw confidence, how many branches were decoded and whether the cap was hit are added per question.
How accurate the boxes are and what a Box costs are not measured yet.
