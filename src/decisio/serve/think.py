# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Think when unsure: a second read after the served model has written its own reasoning.

The procedure is Chon's (github.com/xoai/chon, Apache-2.0, `chon/infer.py`), as tested on the 12B and the
adapter writing its own reasoning:

1. A question is read as always, and served confidence is the top probability after the type's temperature (yes/no:
   max(P(yes), P(no))). It is **unsure** when that is below the type's threshold times a factor: choice 0.75, yes/no
   0.92, score 0.70, each times `factor`, and it has at most 16 options.
2. An unsure question is sent again with `PREFIX` written inside the reply; the **same served model** (the adapter named
   in the request, or the merged weights) writes up to `budget` tokens greedily; then `SUFFIX` follows and the options
   are read again from the next token.
3. The two reads are mixed in log space (the mean of the two log-probabilities, normalised), and the mix goes through
   the same temperature as a single read (the type's, the band's for a choice question). The commit rule of a tuned
   repository applies after.

No history decides anything: whether a question thinks depends on that question and the settings alone, so the same
request thinks the same way whatever came before it. Off unless asked (`--think`, a request's `think` field, or a
tuned repository's `defaults.think`).
"""

from __future__ import annotations

import numpy as np

PREFIX = "Let me check the deciding facts first.\n"
SUFFIX = "\n\nFinal answer:"
TAU = {"choice": 0.75, "noul": 0.92, "score": 0.70}
MAX_OPTIONS = 16
TYPES = ("choice", "noul", "score")  # the question types a server can think on, in the order the files list them
DEFAULT_FACTOR = 1.0
DEFAULT_BUDGET = 512
MAX_BUDGET = 4096

# why a question did not think, as the response says it
SURE = "sure"
TOO_MANY_OPTIONS = f"more than {MAX_OPTIONS} options"
NOT_THINKING_TYPE = "its type is not one this server thinks on"
REGISTERED_TASK = "a registered task answers this question"
NOT_COMBINED = "not combined with two-order averaging, abstention or the image route"


class ThinkUnavailable(ValueError):
    """Thinking was asked for where it cannot happen (an engine that cannot write reasoning); the server answers 422."""


def check_factor(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < float(value) <= 1.5:
        raise ValueError(f"{where} must be a number above 0 and at most 1.5, got {value!r}")
    return float(value)


def check_budget(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 16 <= value <= MAX_BUDGET:
        raise ValueError(f"{where} must be a whole number of tokens from 16 to {MAX_BUDGET}, got {value!r}")
    return int(value)


def check_types(value, where: str) -> list[str]:
    """The question types a server thinks on (`--think-types`, `defaults.think.types`): a non-empty list of choice, noul
    and score without repeats, returned in canonical order."""
    if isinstance(value, str) or not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{where} must be a non-empty list of question types ({', '.join(TYPES)})")
    bad = [v for v in value if v not in TYPES]
    if bad or len(set(value)) != len(value):
        raise ValueError(f"{where} must name each of {', '.join(TYPES)} at most once, got {list(value)!r}")
    return [t for t in TYPES if t in value]


def check_settings(value, where: str) -> dict:
    """A repository's `defaults.think`: {"factor": number, "budget": tokens (optional), "types": the question types
    that think (optional: all)}."""
    if not isinstance(value, dict) or "factor" not in value:
        raise ValueError(f"{where} must be an object with a factor")
    unknown = sorted(set(value) - {"factor", "budget", "types"})
    if unknown:
        raise ValueError(f"{where} has unknown keys {unknown}")
    out = {"factor": check_factor(value["factor"], f"{where}.factor")}
    if "budget" in value:
        out["budget"] = check_budget(value["budget"], f"{where}.budget")
    if "types" in value:
        out["types"] = check_types(value["types"], f"{where}.types")
    return out


def confidence(p: np.ndarray, qtype: str) -> float:
    """The served confidence the unsure test reads: the top probability; for yes/no max(P(yes), P(no))."""
    return float(max(p[0], 1.0 - p[0])) if qtype == "noul" else float(np.max(p))


def is_unsure(p: np.ndarray, qtype: str, factor: float) -> bool:
    return confidence(p, qtype) < TAU[qtype] * factor


def mix(p1: np.ndarray, lp2: np.ndarray) -> np.ndarray:
    """The plain read and the second read's log-probabilities mixed: their mean, normalised."""
    z = (np.log(np.maximum(np.asarray(p1, dtype=np.float64), 1e-300)) + np.asarray(lp2, dtype=np.float64)) / 2.0
    e = np.exp(z - z.max())
    return e / e.sum()


def stop_ids(tok) -> list[int]:
    """The token ids that end the reasoning: the end of sequence and the end of the model's turn."""
    eos = tok.eos_token_id
    out = {eos} if isinstance(eos, int) else set(eos or [])
    vocab = tok.get_vocab()
    for name in ("<turn|>", "<end_of_turn>"):
        if name in vocab:
            out.add(tok.convert_tokens_to_ids(name))
    return sorted(out)
