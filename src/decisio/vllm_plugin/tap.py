# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The hidden-state tap (vLLM 0.32.0; every base's head reader in an in-process engine): a question's hidden
states at its answer position, the final-norm state and a configured set of middle layers, read in the same engine
request as its letter probabilities, with no extra request and no second engine.

How:
  - every base's hidden-readout class (`TapReadout` first in its bases: the Gemma 4 12B's, the Gemma 4 text class of
    the 31B and E4B, and both Qwen classes) asks vLLM's model for the residual stream after the tapped layers
    (`engine_config`, below; vLLM's EAGLE-3 auxiliary-state mechanism:
    layer L is the output of decoder layer L, before the final norm; 0 is the embeddings) and returns
    [final | layer a | layer b | ...] for every token;
  - vLLM's runner takes the answer-position rows of that tensor and hands them to the class's `compute_logits`, which
    computes the logits from the final part alone (contiguous, as before) and keeps the rows for this step (`stash`);
  - a logits processor (`HiddenTap`, vLLM's Model Runner V2 interface) runs right after, in the same step: it knows
    each row's request slot, and for the requests that asked (`SamplingParams.extra_args["decisio_tap"]` = a key) it
    puts the row into the store under that key, without touching the logits;
  - the caller pops the rows by key once the request has finished (`pop`). A prompt read in several chunks stores a
    row at each chunk's step; the last, at the answer position, is what remains.

The store is a module-level dict, so the caller must be in the engine's process (decisio's in-process engine,
`--engine-process in`).

The tapped layers are set in the engine's `additional_config` (`engine_config(layers)`, read by `layers_of`), never
in the environment: vLLM keys its compiled graph on that config (VllmConfig.compute_hash), not on the model class, so
a layer set set anywhere else would load a graph compiled for another set (seen on a card: the graph compiled without
the layers was loaded for a start that asked for five, and served none of them).

  DECISIO_TAP_LAYERS="12,24,30,36,48"   (tap_layers) the launcher's way to write that config: layers tapped beside the
                                         final-norm state (empty: the final state only)
"""

from __future__ import annotations

import os
import threading

TAP_KEY = "decisio_tap"
PROCESSOR = "decisio.vllm_plugin.tap:HiddenTap"  # the logits processor's name, as vLLM's logits_processors take it
_lock = threading.Lock()
_store: dict = {}
# this step's answer-position rows: written by the model's compute_logits and read by the processor in the same engine
# step (one engine core per process; a module variable, not thread-local, as vLLM may sample on another thread)
_rows = None


def tap_layers(env=None) -> tuple[int, ...]:
    """The tapped middle layers, sorted and unique, from DECISIO_TAP_LAYERS (e.g. "12,24,30,36,48")."""
    raw = (os.environ if env is None else env).get("DECISIO_TAP_LAYERS", "")
    layers = sorted({int(x) for x in raw.replace(" ", "").split(",") if x})
    if any(x < 0 for x in layers):
        raise ValueError(f"DECISIO_TAP_LAYERS: layer indices are 0 or more, got {raw!r}")
    return tuple(layers)


CONFIG_KEY = "decisio_tap_layers"


def engine_config(layers) -> dict:
    """`LLM(...)` keyword arguments that give the tap class its layers, inside the compiled graph's cache key."""
    return {"additional_config": {CONFIG_KEY: [int(x) for x in sorted(set(layers))]}}


def layers_of(vllm_config) -> tuple[int, ...]:
    """The tapped layers an engine was built with (`engine_config`), or () when none were given."""
    extra = getattr(vllm_config, "additional_config", None) or {}
    layers = extra.get(CONFIG_KEY, ()) if isinstance(extra, dict) else ()
    return tuple(sorted({int(x) for x in layers}))


def stash(rows) -> None:
    """Keep this step's answer-position rows (rows x (1 + k) * d), for the processor that runs after the logits."""
    global _rows
    _rows = rows


def take():
    global _rows
    rows, _rows = _rows, None
    return rows


def put(key, row) -> None:
    with _lock:
        _store[key] = row


def pop(key):
    """The row stored under `key` (a torch tensor on the engine's device), or None; removed from the store."""
    with _lock:
        return _store.pop(key, None)


def pending() -> int:
    with _lock:
        return len(_store)


def split(row, d: int, layers: tuple[int, ...]):
    """A stored row as (final-norm state, {layer: state}), float32 numpy."""
    import numpy as np

    v = row.float().cpu().numpy() if hasattr(row, "cpu") else np.asarray(row, dtype=np.float32)
    if v.shape[-1] != d * (1 + len(layers)):
        raise ValueError(f"a tapped row of {v.shape[-1]} values is not {1 + len(layers)} states of {d}")
    return v[:d], {layer: v[d * (1 + i) : d * (2 + i)] for i, layer in enumerate(layers)}


def route(rows, row_slots, keys) -> int:
    """Put the rows whose request asked into the store: `row_slots[i]` is row i's request slot, `keys` slot -> key.
    Returns how many were stored."""
    n = 0
    for i, slot in enumerate(row_slots):
        key = keys.get(int(slot))
        if key is not None:
            put(key, rows[i].detach().clone())
            n += 1
    return n


def _processor_base():
    try:
        from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor
    except ImportError:  # no vLLM (the tests give the interface)
        return object
    return LogitsProcessor


class HiddenTap(_processor_base()):
    """vLLM Model Runner V2 logits processor: stores the tapped rows of the requests that asked, by their key.

    It never changes the logits. It accepts a request (`add_request` True) only when the request carries a key, so a
    request without one is processed exactly as before; decisio's readout requests already pass through the logits
    stage (their allowed-token mask is one), so accepting them adds no copy."""

    def __init__(self, vllm_config=None, req_states=None):
        self.keys: dict[int, str] = {}

    @classmethod
    def validate_params(cls, sampling_params) -> None:
        key = (getattr(sampling_params, "extra_args", None) or {}).get(TAP_KEY)
        if key is not None and not isinstance(key, str):
            raise ValueError(f"extra_args[{TAP_KEY!r}] must be a string key")

    def add_request(self, req_idx: int, sampling_params) -> bool:
        key = (getattr(sampling_params, "extra_args", None) or {}).get(TAP_KEY)
        if key is None:
            self.keys.pop(int(req_idx), None)
            return False
        self.keys[int(req_idx)] = key
        return True

    def apply(self, logits, ctx):
        rows = take()
        if rows is None or not self.keys:
            return logits
        # one logits row per request without speculative decoding: row i is batch position i
        route(rows, ctx.idx_mapping_np[: rows.shape[0]], self.keys)
        return logits


class TapReadout:
    """The tap's part of decisio's hidden-readout classes, first in their bases (every base: decisio.vllm_plugin.gemma,
    decisio.vllm_plugin.models). With no layers asked for (`layers_of`), forward is the base class's, as it is; with
    some, it returns [final | layers] (vLLM's EAGLE-3 auxiliary states), concatenated after the compiled region.
    compute_logits keeps the answer-position rows for `HiddenTap`, computes the logits from the contiguous final state
    and writes [0, h] into the reserved columns, so a server reads the head's state either way."""

    def __init__(self, *, vllm_config=None, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        cfg = getattr(self.config, "text_config", self.config)
        self.tap_d = int(cfg.hidden_size)
        self.tap_layers = layers_of(vllm_config)  # in the compiled graph's cache key (engine_config)
        n = getattr(cfg, "num_hidden_layers", None)
        if n is not None and any(layer > n for layer in self.tap_layers):
            raise ValueError(f"the tapped layers {self.tap_layers}: this model has {n} layers")
        if self.tap_layers:
            self.set_aux_hidden_state_layers(self.tap_layers)

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        if isinstance(out, tuple):  # (final-norm state, [layer states]) once auxiliary layers are set
            import torch

            final, layers = out
            return torch.cat([final, *layers], dim=-1)
        return out

    def compute_logits(self, hidden_states):
        from decisio.vllm_plugin.hidden import write_hidden_columns

        stash(hidden_states)  # the answer-position rows, for the tap's logits processor when it runs
        final = hidden_states
        if hidden_states.shape[-1] != self.tap_d:
            final = hidden_states[..., : self.tap_d].contiguous()
        logits = super().compute_logits(final)
        return logits if logits is None else write_hidden_columns(logits, final)
