<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# The bf16 merge on the real 12B

What this shows: `python -m decisio.hub_bf16 merge` run on the real Gemma 4 12B (bf16, 677 tensors) with a real LoRA adapter of it (rank 16, alpha 32, 328 module pairs and 8 `v_proj` copies of `k_proj` pairs for the full-attention layers), on a CPU, and what the one rounding to bf16 keeps of the adapter's change in weight space.
It measures weights, not answers: how the merged model answers is the card check (docs/running.md, "Shipping a tuned model").

`weight_space.py` redoes the exact fp32 merge of every adapted tensor and compares the two roundings; `output.txt` is its output.
The merge took one minute and 22 GB of disk; the report (`MERGE_BF16.json`) says 328 pairs applied, 8 copies left out, seed 0, weight-space effect kept 0.999998, rounding bias over the change 1e-5, and that half of the adapted entries changed.
