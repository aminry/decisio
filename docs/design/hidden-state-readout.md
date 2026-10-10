<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Design: the intent head's hidden state from the serving engine

The intent head serves `softmax(lp + h @ A + c)`, where `lp` are the label log-probabilities and `h` is the final-norm hidden state (2,048 numbers) at the answer position.
vLLM's generate path returns log-probabilities only.
This note is how decisio gets `h` from the serving engine itself, with no second copy of the weights and no change to vLLM's source.
Verified on a card (`runs/2026-09-30_plugin-verification`); the alternative, a second engine in pooling mode, stays available as `--head-engine` (docs/handoffs/tasks.md).

## Options considered

| Option | vLLM source change | `h` exact? | Verdict |
| --- | --- | --- | --- |
| Serve everything from a pooling engine | none | yes | Rejected: the pooling path has no prefix cache on this hybrid model, and the prefix cache is the latency design |
| Recover `h` from the label log-probabilities | none | no | Impossible: K label scores do not determine 2,048 numbers |
| **A registered model class that returns `h` through reserved logit columns** | **none (plugin only)** | to float32 rounding | **Shipped as the default** |
| A patch that adds the last hidden state to the generate output | several files | bit for bit | The clean end state, for upstream vLLM (vllm-project/vllm#59543); carried for 0.31.0 in `patches/vllm-0.31.0/return-last-hidden-states`, not applied by the image and not used by the readout yet |

## The mechanism

`DecisioQwen3_5MoeHiddenReadout` subclasses vLLM's `Qwen3_5MoeForCausalLM` (through the text-only class) and overrides only `compute_logits`: it computes the normal logits, then overwrites the columns of d + 1 reserved token ids with `[0, h_1, ..., h_d]`, a zero reference column and `h` itself (`decisio.vllm_plugin.hidden.write_hidden_columns`).
The reserved ids start at 100,000 (`DECISIO_HIDDEN_READOUT_START`); in the Qwen3.6 tokenizer no label token (the letter codes, " yes", " no") is among them, and the server checks that at start-up.

A head question is sent with only reserved ids allowed and their log-probabilities requested.
vLLM returns the log-softmax over the allowed set, so `h_j = lp_j - lp_0`, to float32 rounding at the normaliser's magnitude.
Every other question is sent exactly as before, with its K labels only; the reserved columns are never in its allowed set, so its answer does not change by a bit.

The label log-probabilities of a head question are recomputed from `h` and the output layer's label rows, so fitting and serving the head use one `(lp, h)`.

## What the card taught (three fixes)

1. **vLLM 0.30.0 and 0.31.0 allow at most 1,024 ids per request** (`MAX_NUM_ALLOWED_TOKEN_IDS`), and an oversized request kills the engine instead of being refused.
   `h` is read in chunks of at most 1,024 ids, each with the reference column: 2,048 dimensions take three requests (`reserved_chunks`).
2. **Identical requests in one batch are not the same forward on this FP8 stack.**
   The three chunk requests sent together gave label probabilities that differed across the batch's rows by up to 0.59, so a state stitched from them was no forward's state.
   Sent one at a time, the chunks agree to 1.2e-7 and share the prompt's prefix through the cache (`SingleEngineHidden.hidden_rows`).
3. **The engine computes its logits in bf16.**
   Label logits recomputed from `h` in float64 differ from the engine's by up to 0.019 in probability; rounded to bf16 they agree within 3e-8 (`SingleEngineHidden.label_logits`).

## The gate

The single engine cannot be bit-identical to a second engine's numbers (different runners, kernels and batching), so the gate is in three parts, each exact where exactness is achievable.
All three passed on the card:

1. **Exact arithmetic**: every stored head equals `fit_intent_head` on its registration's own readout, and every served head answer equals `apply_intent_head` on its own readout, bit for bit (6 of 6 fits, 750 of 750 answers).
2. **Same forward**: on 250 intent items, the label probabilities recomputed from the recovered `h` match the engine's own for the same question within 1e-4 and pick the same option (max difference 3.1e-8, 250 of 250).
3. **No change elsewhere**: the 1,400-item suite bit-identical to the reference view under the same request history, and C1 to C4 pass.

## Costs

- A head question takes three extra engine requests, one at a time: about 127 ms of server time against 46 ms for the same question without a task.
  A direct hidden-state return in vLLM would remove the extra requests.
- One engine and one weight copy: the text engine at 0.90 of the card, or beside the image engine at 0.47 and 0.49.
- The class is for serving only: a text generation that sampled a reserved token would be wrong, so only the decision server's text engine uses it.

## Tests

Without a GPU (`tests/unit/test_hidden_readout.py`, `test_head_modes.py`): the class writes only the reserved columns; recovery from vLLM's float32 log-probabilities returns every stored intent-readout vector within 1e-4; the whole head path through `SingleEngineHidden` on an emulated engine equals the reference arithmetic; the two head modes declare the same choice on the stored readouts.
On a card (`tests/gpu/test_model_classes.py`): the same-forward check on fresh 77-option questions.
