# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A text-only view of the Qwen3.6-35B-A3B checkpoint, for vLLM's Qwen3_5MoeForCausalLM class.

The official checkpoint is the multimodal model; vLLM serves it as Qwen3_5MoeForConditionalGeneration,
which builds M-RoPE positions for every prompt token of every request (13% of the engine thread at an
8,000-token state, profiled). The text-only class uses plain RoPE, but its loader refuses the
vision tower's tensors ("no module or parameter named 'visual'"). This writes a directory where every
file is a symlink to the original except the shard(s) holding `visual` tensors, rewritten without them,
the weight index without them, and config.json naming the text-only architecture. No language-model
tensor is changed or copied. The original directory is not touched.

The fallback for `--model-class view`: decisio's registered text-only class (`decisio.vllm_plugin`) loads the official
checkpoint directly and serves it bit-identically (runs/2026-09-30_plugin-verification).

    python -m decisio.serve.make_text_only $DECISIO_MODEL $DECISIO_VIEW
"""
import json
import os
import sys
from pathlib import Path

from decisio.vllm_plugin.weights import is_vision_weight


def main():
    src, dst = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    visual = {k for k in wm if is_vision_weight(k)}
    shards_with_visual = {wm[k] for k in visual}
    print(f"{len(visual)} vision tensors in {len(shards_with_visual)} shard(s): {sorted(shards_with_visual)}")
    from safetensors import safe_open
    from safetensors.torch import save_file
    for shard in sorted(shards_with_visual):
        keep = {}
        with safe_open(str(src / shard), framework="pt") as f:
            meta = f.metadata()
            for k in f.keys():
                if k not in visual:
                    keep[k] = f.get_tensor(k)
        save_file(keep, str(dst / shard), metadata=meta or {"format": "pt"})
        print(f"  rewrote {shard}: kept {len(keep)} tensors")
    for p in src.iterdir():
        if p.name in shards_with_visual or p.name in ("config.json", "model.safetensors.index.json", ".fetched"):
            continue
        link = dst / p.name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(p.resolve(), link)
    index["weight_map"] = {k: v for k, v in wm.items() if k not in visual}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    cfg = json.loads((src / "config.json").read_text())
    cfg["architectures"] = ["Qwen3_5MoeForCausalLM"]
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"TEXT_ONLY_DONE {dst}")


if __name__ == "__main__":
    main()
