# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The answer-mix alert of a certified task: per task, how often each option was chosen, never the texts.

The server keeps, for each certified task, the option each of its last 500 answers chose (an integer index into the
task's
options) and counts. Every 100 answers once the window is full it compares the window's mix of chosen options with
the mix of
the task's certification examples, by total variation distance, and raises the alert when the distance is above a
line fixed
when the certificate was made:

  line   the 99.9th percentile of the same distance between two independent resamples of the certification examples, one
         the size of the examples and one the size of a window (2,000 draws, a fixed seed). The examples' own mix is a
         sample, so a window of the same traffic is not expected to match it exactly.

The alert says the traffic changed, not that the certificate is broken: it does not change answers. A change of the
serving
stack, which can break it, is not this monitor's but the fingerprint's (`decisio.serve.stack`). The monitor's
counters live
in memory and start again with the server; the baseline and the line are stored with the task.

Replay of record-based streams (Lab 3, 2026-10-13): the alert fired on at most 0.5 percent of full-window checks
without a
shift and detected a rise of the out-of-scope share from 1.5 to 18 percent in a median 350 to 450 answers.
Standard library and numpy.
"""

from __future__ import annotations

import threading
import zlib
from collections import deque

import numpy as np

WINDOW, STEP, LINE_QUANTILE, DRAWS = 500, 100, 0.999, 2000


def mix(counts) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    return counts / counts.sum()


def distance(a, b) -> float:
    """Total variation distance between two mixes (or two count vectors)."""
    return 0.5 * float(np.abs(mix(a) - mix(b)).sum())


def alert_line(codes, k: int, seed: str, window: int = WINDOW, draws: int = DRAWS) -> float:
    """The alert line for certification examples whose chosen options are `codes` (indices below k)."""
    codes = np.asarray(codes, dtype=int)
    rng = np.random.default_rng(zlib.crc32(seed.encode()))
    w = np.stack([np.bincount(r, minlength=k) for r in codes[rng.integers(0, len(codes), (draws, window))]]) / window
    c = np.stack([np.bincount(r, minlength=k) for r in codes[rng.integers(0, len(codes), (draws, len(codes)))]]) / len(
        codes
    )
    return float(np.quantile(0.5 * np.abs(w - c).sum(1), LINE_QUANTILE))


def baseline(codes, k: int, seed: str) -> dict:
    """The record stored with a certified task."""
    return {
        "window": WINDOW,
        "step": STEP,
        "line_quantile": LINE_QUANTILE,
        "draws": DRAWS,
        "seed": seed,
        "option_counts": np.bincount(np.asarray(codes, dtype=int), minlength=k).tolist(),
        "line": alert_line(codes, k, seed),
    }


class MixMonitor:
    """The window of one task. `base` is a `baseline` record."""

    def __init__(self, base: dict):
        self.base = base
        self.k = len(base["option_counts"])
        self._ring: deque[int] = deque(maxlen=base["window"])
        self._counts = np.zeros(self.k, dtype=int)
        self.answers = 0
        self.certified_answers = 0
        self.checked_at = 0
        self.alerts = 0
        self.distance: float | None = None
        self.alerting = False
        self._lock = threading.Lock()

    def observe(self, index: int, certified: bool = False) -> None:
        with self._lock:
            if len(self._ring) == self._ring.maxlen:
                self._counts[self._ring[0]] -= 1
            self._ring.append(int(index))
            self._counts[int(index)] += 1
            self.answers += 1
            self.certified_answers += int(certified)
            if len(self._ring) == self._ring.maxlen and self.answers - self.checked_at >= self.base["step"]:
                self.checked_at = self.answers
                self.distance = distance(self._counts, self.base["option_counts"])
                self.alerting = self.distance > self.base["line"]
                self.alerts += int(self.alerting)

    def facts(self) -> dict:
        with self._lock:
            return {
                "state": "traffic_changed" if self.alerting else "ok",
                "window_answers": len(self._ring),
                "answers": self.answers,
                "certified_answers": self.certified_answers,
                "checked_at": self.checked_at or None,
                "distance": self.distance,
                "line": self.base["line"],
                "alerts": self.alerts,
                "option_counts": self._counts.tolist(),
            }
