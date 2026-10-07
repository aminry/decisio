# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Read-only queries the server makes to vLLM's worker, mixed into it with `LLM(worker_extension_cls=...)` and called
by method name through `LLM.collective_rpc` (a name, not a pickled callable, so vLLM's insecure serialization stays
off).

Imports nothing at module level: vLLM imports this module in every worker it starts.
"""

from __future__ import annotations

QUALNAME = "decisio.vllm_plugin.worker.DecisioWorkerExtension"


class DecisioWorkerExtension:
    def decisio_kv_block_sizes(self) -> list[int] | None:
        """vLLM's own (scheduler block size, hash block size) for the engine's KV cache groups, or None where this
        vLLM cannot say.

        The scheduler block size is the unit a prefix-cache hit comes in: with several KV cache groups it is the least
        common multiple of their block sizes, since a hit must hold in every group (Gemma 4 on vLLM 0.30.0: five groups
        of 16 tokens and one of 64, so 64). The hash block size is the step prefixes are hashed at (there, 16)."""
        try:
            from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
        except ImportError:
            return None
        cfg = getattr(getattr(self, "model_runner", None), "kv_cache_config", None)
        if cfg is None:
            return None
        return [int(x) for x in resolve_kv_cache_block_sizes(cfg, self.vllm_config)]

    def decisio_fp8_fingerprints(self) -> dict | None:
        """Per quantized linear layer of the text model: its fused FP8 weight's shape as [out, in], its scale and the
        sha256 of its bytes (row-major), keyed `layers.N.<module>.<proj>`, or None where this vLLM cannot say.

        The identity gate compares these between a server that quantizes the bf16 checkpoint on load and one that loads
        the stored FP8 checkpoint (decisio.hub_fp8: `expected` says what each should hold, `compare` the difference).
        vLLM keeps a quantized weight transposed ([in, out]); it is put back to [out, in] before it is hashed, so the
        bytes are the ones the checkpoint stores."""
        import hashlib
        import re

        import torch

        runner = getattr(self, "model_runner", None)
        model = runner.get_model() if runner is not None and hasattr(runner, "get_model") else None
        if model is None:
            return None
        key = re.compile(r"layers\.(\d+)\.(self_attn|mlp)\.(qkv_proj|o_proj|gate_up_proj|down_proj)$")
        out = {}
        for name, module in model.named_modules():
            m = key.search(name)
            weight, scale = getattr(module, "weight", None), getattr(module, "weight_scale", None)
            if m is None or weight is None or scale is None or weight.dtype != torch.float8_e4m3fn:
                continue
            n_out, n_in = getattr(module, "output_size", None), getattr(module, "input_size", None)
            if tuple(weight.shape) == (n_in, n_out):
                weight = weight.t()
            digest = hashlib.sha256()
            flat = weight.contiguous().view(torch.uint8).reshape(-1)
            for i in range(0, flat.numel(), 1 << 28):
                digest.update(flat[i : i + (1 << 28)].cpu().numpy().tobytes())
            out[f"layers.{m[1]}.{m[2]}.{m[3]}"] = {
                "shape": list(weight.shape),
                "scale": float(scale.reshape(-1).max()),
                "sha256": digest.hexdigest(),
            }
        return out
