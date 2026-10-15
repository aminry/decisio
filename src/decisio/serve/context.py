# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The context length of a vLLM server that was not given `--max-model-len`: the largest that fits this card, up to the
base's ceiling.

A fixed length can fail to start on a smaller card (a 31B at 98,304 tokens needs about 33 GiB of KV cache, and an
80 GB card has about 35 GiB after the weights), and a failed start loses the whole run. So the server asks vLLM for the
length that fits (`max_model_len=-1`: vLLM profiles the card, takes the KV memory it reports and searches the longest
single request that fits, up to the model's own maximum) and serves the largest clean step of that, never above the
base's ceiling and never below the 32,768 tokens every record was measured at.

Standard library only; `engine` is whatever reports the fit.
"""

from __future__ import annotations

from dataclasses import dataclass

FLOOR = 32768  # the length every record was measured at; below it the server does not start, as before
STEP = 8192  # the lengths the server serves are multiples of it


class ContextDoesNotFit(ValueError):
    """The card holds less than the 32,768 tokens of context every record was measured at."""


def round_down(n: int, step: int = STEP) -> int:
    return (int(n) // step) * step


@dataclass(frozen=True)
class AutoContext:
    """The default context of a base whose ceiling is above 32,768: what fits this card, up to the ceiling."""

    ceiling: int
    floor: int = FLOOR
    step: int = STEP

    def choose(self, fit: int) -> int:
        """The length to serve when vLLM says `fit` tokens of one request fit: the largest multiple of `step` not above
        `fit` and not above the ceiling. Raises ContextDoesNotFit when the floor itself does not fit."""
        fit = int(fit)
        if fit < self.floor:
            raise ContextDoesNotFit(
                f"the KV cache memory vLLM reports on this card holds a context of {fit:,} tokens, less than the "
                f"{self.floor:,} every record was measured at. Start a smaller base, raise --gpu-memory-utilization, "
                f"or set --max-model-len to a length that fits (docs/running.md, 'The context length')"
            )
        return max(self.floor, min(self.ceiling, round_down(fit, self.step)))

    def describe(self, chosen: int, fit: int) -> str:
        """Why this length, for /health and the start-up line."""
        if chosen >= self.ceiling:
            return f"auto: the base's ceiling of {self.ceiling:,} fits on this card"
        if chosen == self.floor and int(fit) < self.floor + self.step:
            return f"auto: fits {chosen:,} of {self.ceiling:,} on this card (the floor)"
        return f"auto: fits {chosen:,} of {self.ceiling:,} on this card"


def fixed_source(n: int, ceiling: int, reason: str) -> str:
    """The source line of a length that was not chosen by fit."""
    return f"{reason}: {n:,} (the base's ceiling is {ceiling:,})" if ceiling > n else f"{reason}: {n:,}"
