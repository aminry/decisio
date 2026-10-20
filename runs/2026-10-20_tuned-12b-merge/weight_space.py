# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
# ruff: noqa: E501  (a script that prints sentences)
"""How much of an adapter's change a bf16 merge keeps, in weight space, on the real 12B: rounding to nearest against the
stochastic rounding of `decisio.hub_bf16`.

    uv run python runs/2026-10-20_tuned-12b-merge/weight_space.py SOURCE_DIR ADAPTER_DIR MERGED_DIR   > output.txt

SOURCE_DIR: google/gemma-4-12B-it (bf16); ADAPTER_DIR: a PEFT adapter of the 12B; MERGED_DIR: `hub_bf16 merge` of the two.
For every adapted tensor the exact fp32 merge W + scale * B @ A is made again; the share kept is the slope of
(rounded - W) on (exact - W), over all entries of all adapted tensors (1 is the whole change, 0 is none)."""

import sys
from pathlib import Path

import torch

from decisio import hub_fp8

src, adapter, merged = (Path(p) for p in sys.argv[1:4])
S, M = hub_fp8.Source(src), hub_fp8.Source(merged)
deltas, scale, sha, layout, orphans = hub_fp8.load_adapter(adapter, S.map)
num_n = num_s = den = 0.0
same_n = same_s = total = 0
for name, ab in sorted(deltas.items()):
    w = S.get(name)
    exact = hub_fp8.merged32(w, ab, scale, "cpu")
    d = (exact - w.float()).double()
    nearest = exact.to(torch.bfloat16)
    stoch = M.get(name)
    num_n += float(((nearest.float() - w.float()).double() * d).sum())
    num_s += float(((stoch.float() - w.float()).double() * d).sum())
    den += float((d * d).sum())
    same_n += int((nearest == w).sum())
    same_s += int((stoch == w).sum())
    total += w.numel()
print(
    f"adapter {sha[:12]}, scale {scale}, {layout} layout: {len(deltas)} adapted tensors, {len(orphans)} v_proj copies of k_proj pairs left out"
)
print(f"{total:,} entries in the adapted tensors")
print(f"entries unchanged from the base:   rounding to nearest {same_n / total:.4f}   stochastic {same_s / total:.4f}")
print(f"share of the adapter's change kept: rounding to nearest {num_n / den:.4f}   stochastic {num_s / den:.4f}")
