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
