# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A LoRA adapter merged into a bf16 Gemma 4 checkpoint (the 12B), written as bf16.

    python -m decisio.hub_bf16 merge --source <bf16 dir> --adapter <peft dir> --out <dir> \
        [--seed 0] [--device cuda] [--null]

The 31B is stored in FP8, where the merge is quantized once (`decisio.hub_fp8`). The 12B is served in bf16, so its
merged repository is bf16 too, and the one rounding the merge needs is the one from fp32 back to bf16.
Rounding to nearest cannot be used: the base weights are already on the bf16 grid and an adapter's change is a small
fraction of a grid step, so `(W + delta).to(bf16)` rounds back to W for most entries and keeps about half of the adapter
(0.44 to 0.51 measured on a tuned 31B). **Stochastic rounding** is unbiased: each entry goes to one of its two bf16
neighbours with the probability that makes the expected value the fp32 value, so the merged weights carry the whole
change in expectation (1.02 kept, measured). The noise is seeded per tensor (`sha256(seed | tensor name)`), so the merge
is repeatable and does not depend on the order the tensors are processed; the seed is in the report and in the
repository's file.

Every other tensor is copied byte for byte; a pair of the adapter must land on a tensor of the base (never a merge of
nothing). A `v_proj` pair of a full-attention layer with `attention_k_eq_v` (no `v_proj` in the base, V is K) is left
out and counted when it is bit for bit the layer's `k_proj` pair, and refused when it differs (as `hub_fp8 merge` does).
`--null` merges an adapter of scale 0, whose result must equal the source byte for byte (exact values survive the
rounding). Writes `MERGE_BF16.json`, which `decisio.hub tuned-config` reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

from decisio.hub_fp8 import (
    INDEX,
    PROJ,
    WEIGHTS,
    Source,
    Writer,
    k_copy_difference,
    load_adapter,
    merged32,
)

KIND = "fp32 merge to bf16"


def tensor_seed(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{name}".encode()).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF


def stochastic_bf16(w32: torch.Tensor, seed: int, name: str) -> torch.Tensor:
    """`w32` (float32) rounded to bfloat16 stochastically: uniform noise is added below the bf16 mantissa and the low 16
    bits dropped, so the result is one of the two neighbours with probabilities that make it unbiased. Exact bf16 values
    (low 16 bits zero) map to themselves whatever the noise. CPU, seeded by tensor."""
    g = torch.Generator().manual_seed(tensor_seed(seed, name))
    noise = torch.randint(0, 1 << 16, w32.shape, generator=g, dtype=torch.int32)
    bits = (w32.contiguous().view(torch.int32) + noise) & torch.tensor(-65536, dtype=torch.int32)
    return bits.view(torch.float32).to(torch.bfloat16)


def merge(
    source: Path,
    adapter: Path,
    out: Path,
    seed: int = 0,
    device: str = "cpu",
    null: bool = False,
    shard_bytes: int = 4 << 30,
    progress=None,
) -> dict:
    """Write the bf16 checkpoint of `source` with the adapter merged in fp32 and rounded once, stochastically; the
    report (`MERGE_BF16.json`). Raises ValueError if an adapter pair finds no tensor."""
    src = Source(source)
    deltas, scale, sha, layout, orphans = load_adapter(adapter, src.map)
    differ = {n: d for n, ab in orphans.items() if (d := k_copy_difference(n, ab, deltas)) is not None}
    if differ:
        raise ValueError(
            f"{len(differ)} adapter pairs are v_proj of full-attention layers that ship no v_proj (attention_k_eq_v: "
            "V is K) and are not copies of the layer's k_proj pair, so they ask for a V that differs from K, which "
            "this checkpoint layout cannot hold: " + "; ".join(f"{n}: {d}" for n, d in sorted(differ.items())[:5])
        )
    projections = {n for n in src.map if PROJ.match(n)}
    uncovered = [n for n in deltas if n not in projections]
    if uncovered:
        raise ValueError(f"{len(uncovered)} adapter weights are not projections of the text model, e.g. {uncovered[0]}")
    scale = 0.0 if null else scale
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    writer = Writer(out, shard_bytes)
    # in weight space: the share of the adapter's change the rounded weights carry (1 is all of it), the entries that
    # changed, and the bias of the rounding relative to the change (0 is unbiased)
    num = den = bias = absdelta = 0.0
    changed = total = 0
    done = 0
    names = sorted(src.map)
    for i, name in enumerate(names):
        w = src.get(name)
        if name not in deltas:
            writer.add(name, w)
            continue
        if w.dtype != torch.bfloat16 or w.dim() != 2:
            raise ValueError(f"{name}: expected a 2-D bfloat16 weight, found {w.dtype}")
        exact = merged32(w, deltas[name], scale, device).cpu()
        rounded = stochastic_bf16(exact, seed, name)
        base = w.to(torch.float32)
        d_exact, d_round = (exact - base).double(), (rounded.to(torch.float32) - base).double()
        num += float((d_round * d_exact).sum())
        den += float((d_exact * d_exact).sum())
        bias += float((d_round - d_exact).sum())
        absdelta += float(d_exact.abs().sum())
        changed += int((rounded != w).sum())
        total += w.numel()
        writer.add(name, rounded)
        done += 1
        if progress and done % 40 == 0:
            progress(f"{done}/{len(deltas)} adapted tensors")
    if done != len(deltas):
        raise ValueError(f"{len(deltas)} adapter pairs, {done} applied: the merge would drop some of the adapter")
    weight_map = writer.finish()
    (out / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2, sort_keys=True))
    for f in src.dir.iterdir():  # config, tokenizer, chat template, generation and processor files: the base's own
        if f.is_file() and not WEIGHTS.match(f.name) and f.name != INDEX:
            shutil.copy2(f, out / f.name)
    meta = {
        "kind": KIND,
        "rounding": "stochastic",
        "seed": seed,
        "adapter_sha256": sha,
        "adapter_scale": scale,
        "adapter_layout": layout,
        "adapted_tensors": len(deltas),
        "dropped_k_eq_v_pairs": len(orphans),
        "null": null,
        "tensors": len(weight_map),
        "weight_space_effect_kept": (num / den) if den > 0 else None,
        "rounding_bias_over_change": (bias / absdelta) if absdelta > 0 else None,
        "changed_share_of_adapted_entries": (changed / total) if total else None,
    }
    (out / "MERGE_BF16.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m decisio.hub_bf16", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mg = sub.add_parser("merge", help="merge a LoRA adapter into a bf16 checkpoint, rounding once, stochastically")
    mg.add_argument("--source", required=True, help="the bf16 base directory")
    mg.add_argument("--adapter", required=True, help="the PEFT adapter directory")
    mg.add_argument("--out", required=True)
    mg.add_argument("--seed", type=int, default=0, help="the rounding's seed (recorded; default 0)")
    mg.add_argument(
        "--device", default="cpu", help="where the fp32 merge runs (cuda for speed); the rounding is on the CPU"
    )
    mg.add_argument(
        "--null", action="store_true", help="merge an adapter of scale 0: must equal the source byte for byte"
    )
    a = ap.parse_args(argv)
    meta = merge(
        Path(a.source), Path(a.adapter), Path(a.out), a.seed, a.device, a.null, progress=lambda m: print(m, flush=True)
    )
    kept = meta["weight_space_effect_kept"]
    print(
        f"{meta['adapted_tensors']} adapter pairs applied ({meta['adapter_layout']} layout, "
        f"{meta['dropped_k_eq_v_pairs']} v_proj copies of k_proj pairs left out); {meta['tensors']} tensors; "
        f"seed {meta['seed']}; weight-space effect kept " + ("n/a (null merge)" if kept is None else f"{kept:.4f}")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
