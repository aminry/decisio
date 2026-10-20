# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The serving-stack fingerprint of a certificate: everything about the server that can move an answer.

A certified task is a promise about answers read through one stack (docs/tasks.md, "Certified answers"). The
fingerprint is
taken when the certificate is made and compared with the live server's at every answer; any difference suspends the
certificate (the answers carry `certified: false`) until it is made again with an explicit call.

What it holds, and where each part comes from:
  decisio       the installed package's version
  base          the base's key (`gemma-4-31b`, ...) and the checkpoint: repository name and revision, or the model's
  name
  engine        the engine and its version (vLLM, or MLX and mlx-lm, or transformers and torch), the card or device, the
                dtype or quantisation, the KV cache dtype, the context length served (`max_model_len`: where it follows
                the card's fit, another length is another compiled path, and it moved short answers by restart-sized
                amounts on one card class), the attention backend, the block and cache-hit units, and the loaded
                adapters: their names and the sha256 of each one's weights and configuration
  prompt        a hash of the token rows the server builds for a fixed probe question (so the tokenizer, the chat
  template,
                the system turn and the rendering are all in it), and the prompt format's own facts
  calibration   the temperatures served for the questions without a task correction (and each loaded adapter's own),
                and the task's own fit (the prior's and the head's arrays) as a digest
  serving       the flags that change what is asked or returned: orders, multi-question mode, pad policy, option
  rendering,
                abstention and --noul-commit

Standard library only; `engine_facts` is `engine.facts()` and the rest are plain values.
"""

from __future__ import annotations

import hashlib
import json
import os

# the keys of an engine's facts() that move an answer; counters and cache sizes are left out on purpose
ENGINE_KEYS = (
    "engine",
    "vllm",
    "mlx",
    "mlx_lm",
    "transformers",
    "torch",
    "gpu",
    "device",
    "machine",
    "model",
    "model_revision",
    "base",
    "quantization",
    "kv_cache_dtype",
    "max_model_len",
    "attention_backend",
    "engine_process",
    "mode",
    "block_size",
    "match_unit",
    "cache_hit_unit",
    "hash_unit",
    "register_boundary",
    "pad_unit",
    "pad_where",
    "prefill_step",
    "tokenizer",
    "tokenizer_revision",
    "tokenizer_json_sha256",
    "final_logit_softcap",
    "adapters",
    "adapter_identity",
)

PROBE_STATE = "A state to check. " * 30
PROBE_QUESTIONS = [
    {"kind": "choice", "instructions": "Which?", "options": ["alpha", "beta"]},
    {"kind": "noul", "instructions": "Is it a check?"},
]


def _plain(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    if isinstance(x, (tuple, list)):
        return [_plain(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    return x


def sha256(obj) -> str:
    return hashlib.sha256(json.dumps(_plain(obj), sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _name(value):
    """A checkpoint given as an absolute path is named by its last component (a snapshot's revision folder), so the same
    weights under another cache directory are the same stack and a local path does not travel in a certificate."""
    if isinstance(value, str) and os.path.isabs(value):
        return os.path.basename(os.path.normpath(value))
    return value


def probe_digest(engine) -> str | None:
    """A hash of the token rows the engine builds for the fixed probe request, or None where it cannot say."""
    try:
        return sha256(engine._prepare_separate(PROBE_STATE, PROBE_QUESTIONS))
    except Exception:  # an engine without the method, or a stand-in: the rest of the fingerprint still stands
        return None


def stack_facts(
    engine,
    *,
    decisio_version: str,
    serving: dict,
    temperatures: dict,
    task_fingerprint: str | None,
    probe_sha256: str | None = None,
) -> dict:
    """The fingerprint's parts for a running server; see the module docstring."""
    facts = engine.facts() if hasattr(engine, "facts") else {}
    repo = facts.get("repository") if isinstance(facts.get("repository"), dict) else None
    return {
        "decisio": decisio_version,
        "base": facts.get("base"),
        "repository": (
            {
                "name": repo.get("name"),
                "revision": repo.get("revision"),
                # a tuned repository (decisio.hub, schema /2): the digest of its tuned and defaults sections, so that
                # a directory (no revision) and a changed default are told apart too
                **({"defaults_sha256": repo["defaults_sha256"]} if repo.get("defaults_sha256") else {}),
            }
            if repo
            else None
        ),
        "engine": {k: _name(facts[k]) for k in ENGINE_KEYS if k in facts},
        "prompt": {
            "probe_sha256": probe_sha256 if probe_sha256 is not None else probe_digest(engine),
            "format": facts.get("prompt_format") or getattr(getattr(engine, "fmt", None), "facts", lambda: None)(),
        },
        "calibration": {"temperatures": temperatures},
        "serving": serving,
        "task_fingerprint": task_fingerprint,
    }


def differences(a, b, path="") -> list[str]:
    """The dotted paths at which two fingerprints differ (a key present on one side only counts)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            sub = f"{path}.{k}" if path else str(k)
            if k not in a or k not in b:
                out.append(sub)
            else:
                out += differences(a[k], b[k], sub)
        return out
    return [] if _plain(a) == _plain(b) else [path]
