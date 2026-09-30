# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The vLLM plugin of decisio: model classes registered through vLLM's `vllm.general_plugins` entry point
(docs/design/vllm-plugin.md).

    [project.entry-points."vllm.general_plugins"]
    decisio = "decisio.vllm_plugin:register"

vLLM calls `register()` in every process it starts (front end, engine core, workers), possibly more than once, so it
is re-entrant and imports nothing heavy: the model classes are registered in vLLM's lazy `<module>:<class>` form and
are imported only when vLLM builds a model of that architecture.

The plugin is written against one vLLM version. On any other version it registers nothing (one log line), so stock
vLLM behaves exactly as without it. It never patches vLLM's own code.
"""
from __future__ import annotations

import logging

SUPPORTED_VLLM = "0.30.0"
# architecture name -> "<module>:<class>" (vLLM's lazy registration form)
TEXT_ONLY = "DecisioQwen3_5MoeTextOnly"
HIDDEN_READOUT = "DecisioQwen3_5MoeHiddenReadout"
MODELS: dict[str, str] = {
    TEXT_ONLY: "decisio.vllm_plugin.models:DecisioQwen3_5MoeTextOnly",
    HIDDEN_READOUT: "decisio.vllm_plugin.models:DecisioQwen3_5MoeHiddenReadout",
}
BASE_ARCH = "Qwen3_5MoeForCausalLM"        # vLLM's class every decisio class subclasses; its config hook is reused
ENTRY_POINT_GROUP, ENTRY_POINT_NAME = "vllm.general_plugins", "decisio"

logger = logging.getLogger(__name__)
_registered = False


def vllm_version() -> str | None:
    try:
        import vllm
    except ImportError:
        return None
    return getattr(vllm, "__version__", None)


def register() -> bool:
    """Register decisio's model classes with vLLM. Returns whether they are registered in this process.

    Re-entrant: a second call in the same process does nothing. On a vLLM version other than `SUPPORTED_VLLM` (or
    without vLLM) nothing is registered."""
    global _registered
    if _registered:
        return True
    version = vllm_version()
    if version != SUPPORTED_VLLM:
        logger.warning("decisio's vLLM plugin supports vllm==%s; found %s: nothing registered, stock vLLM unchanged",
                       SUPPORTED_VLLM, version)
        return False
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    # vLLM keys its per-architecture config hook by architecture name (for the stock text class: the recurrent-state
    # cache dtype from the checkpoint's config, and the M-RoPE fields removed). A registered class under a new name
    # gets no hook unless it is entered here, and then runs with another block size and other positions than the
    # class it subclasses (found on a card, 2026-09-30: 544-token blocks instead of 1,056).
    hook = MODELS_CONFIG_MAP.get(BASE_ARCH)
    if hook is None:
        logger.warning("decisio: vLLM has no config hook for %s: nothing registered", BASE_ARCH)
        return False
    for arch, target in MODELS.items():
        ModelRegistry.register_model(arch, target)
        MODELS_CONFIG_MAP[arch] = hook
    _registered = True
    logger.info("decisio registered %d model class(es) with vLLM %s: %s", len(MODELS), version, sorted(MODELS))
    return True


def engine_kwargs(arch: str) -> dict:
    """`LLM(...)` keyword arguments that make vLLM build the checkpoint under one of decisio's classes."""
    if arch not in MODELS:
        raise KeyError(f"{arch!r} is not one of decisio's model classes: {sorted(MODELS)}")
    return {"hf_overrides": {"architectures": [arch]}}


def installed_entry_point() -> bool:
    """Whether vLLM's other processes (engine core, workers) will find the plugin: they discover it through the
    installed package's entry point, which a bare PYTHONPATH does not provide."""
    from importlib.metadata import entry_points
    return any(ep.name == ENTRY_POINT_NAME and ep.value == "decisio.vllm_plugin:register"
               for ep in entry_points(group=ENTRY_POINT_GROUP))
