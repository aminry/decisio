# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The hidden state at the answer position, returned through the generate path's own log-probabilities
(docs/design/hidden-state-readout.md). No vLLM or torch import: usable anywhere.

The model class (`models.DecisioQwen3_5MoeHiddenReadout`) overwrites the logits of d + 1 reserved token ids with
[0, h_1, ..., h_d]: a zero reference column and the final-norm hidden state h the logits were computed from. A request
that allows the reference id and some of the others and asks for their log-probabilities gets the log-softmax of those
columns back, and

    h_j = lp_j - lp_0

recovers those dimensions (to float32 rounding at the normaliser's magnitude). vLLM's sampler takes at most 1,024
allowed ids per request, so h is read in chunks (`reserved_chunks`), one request each, every chunk with the reference.
A request that does not allow the reserved ids never sees them: its masked log-probabilities are what vLLM's own class
gives, bit for bit.
"""
from __future__ import annotations

import os

ENV_START = "DECISIO_HIDDEN_READOUT_START"
DEFAULT_START = 100_000          # Qwen3.6's tokenizer: no letter label and no " yes" / " no" in [100000, 102048]


def readout_start() -> int:
    """The first reserved token id; read from the environment so the front end and vLLM's engine processes agree."""
    return int(os.environ.get(ENV_START, DEFAULT_START))


def reserved_ids(hidden_size: int, start: int | None = None) -> list[int]:
    """The reserved token ids, in order: the zero reference column, then one per hidden dimension."""
    start = readout_start() if start is None else start
    return list(range(start, start + hidden_size + 1))


MAX_ALLOWED = 1024               # vLLM 0.30.0's sampler: at most 1,024 allowed token ids per request
                                 # (vllm/v1/worker/gpu/sample/logit_bias.py, MAX_NUM_ALLOWED_TOKEN_IDS)


def reserved_chunks(hidden_size: int, start: int | None = None) -> list[list[int]]:
    """The reserved ids as allowed sets of at most `MAX_ALLOWED` ids, one request each: every chunk starts with the
    zero reference column and carries up to MAX_ALLOWED - 1 hidden dimensions, in order (2,048 dimensions: 3 chunks)."""
    ids = reserved_ids(hidden_size, start)
    ref, dims = ids[0], ids[1:]
    step = MAX_ALLOWED - 1
    return [[ref, *dims[i:i + step]] for i in range(0, len(dims), step)]


def recover_hidden_chunks(chunk_logprobs):
    """h from the log-probabilities of each chunk's ids (in `reserved_chunks` order): each chunk's values minus its own
    reference column, concatenated, in float64."""
    import numpy as np
    return np.concatenate([recover_hidden(lp) for lp in chunk_logprobs])


def check_reserved(hidden_size: int, vocab_size: int, label_ids, start: int | None = None) -> list[int]:
    """The reserved ids, after checking that they are inside the vocabulary and that no label token is among them."""
    ids = reserved_ids(hidden_size, start)
    if ids[0] < 0 or ids[-1] >= vocab_size:
        raise ValueError(f"reserved ids {ids[0]}..{ids[-1]} are outside the vocabulary (size {vocab_size})")
    clash = sorted(set(ids) & set(label_ids))
    if clash:
        raise ValueError(f"reserved ids {ids[0]}..{ids[-1]} contain label tokens {clash[:5]}; set {ENV_START}")
    return ids


def write_hidden_columns(logits, hidden_states, start: int | None = None):
    """Overwrite, in place, the reserved columns of `logits` (rows x vocab) with [0, hidden_states] (rows x d).
    Works on torch tensors; every other column is untouched."""
    start = readout_start() if start is None else start
    d = hidden_states.shape[-1]
    if start + d + 1 > logits.shape[-1]:
        raise ValueError(f"reserved ids {start}..{start + d} are outside the vocabulary (size {logits.shape[-1]})")
    logits[..., start] = 0
    logits[..., start + 1:start + 1 + d] = hidden_states.to(logits.dtype)
    return logits


def recover_hidden(logprobs):
    """h from the log-probabilities of the reserved ids, in `reserved_ids` order: lp[1:] - lp[0], in float64."""
    import numpy as np
    lp = np.asarray(logprobs, dtype=np.float64)
    return lp[..., 1:] - lp[..., :1]
