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

The alert says the traffic changed, not that the certificate is broken: it never changes an answer's option. The
confidence signal's alert stops the task certifying (`suspend_on_alert`, on by default); the answer mix reports only. A
change of the serving stack, which can break the certificate, is not this monitor's but the fingerprint's
(`decisio.serve.stack`). The monitor's counters live in memory and start again with the server; the baseline and the
line are stored with the task.

A second signal watches the confidences themselves (added after the card replay of 2026-10-16, below): a CUSUM on how
often an answer's top probability falls below the certification examples' 5th percentile. The answer mix cannot see a
shift among many options (out-of-scope messages are spread over a 150-option task's labels and barely move the mix),
and its line is set high by the same many options; the confidences of such messages are visibly lower.

  tail      the 5th percentile of the certification examples' top probabilities, and `p0`, the share of them below it
  normal    up to `OK_FACTOR` x p0 (1.3): traffic from another split of the same data sat at 1.2 to 1.3 x p0
  alarm     `BAD_FACTOR` x p0 (2.5) is a shift worth an alarm
  statistic S = max(0, S + log-likelihood ratio of "bad" against "normal") after every answer, which needs no window
  limit     the smallest h at which a stream with the normal share alarms with probability at most 1 percent in 5,000
            answers (4,000 simulated streams, a fixed seed), so a quiet task is not near-alarm
  clears    the alert lasts until S falls under h / 2; S is held at 1.5 h, so that a long shift does not delay the
            clearing (about 300 normal answers after the traffic returns)

On the card replay (the 31B, CLINC150 with 10 examples per intent, 5,363 messages, the out-of-scope share rising from
message 1,000) the answer mix alerted after message 5,200 and 976 wrong certified answers; this signal alerts near
message 1,700 after 55, and stays quiet on the replay's in-scope messages alone (`runs/2026-10-17_drift-monitor/`).

Replay of record-based streams (Lab 3, 2026-10-13): the alert fired on at most 0.5 percent of full-window checks
without a
shift and detected a rise of the out-of-scope share from 1.5 to 18 percent in a median 350 to 450 answers.
Standard library and numpy.
"""

from __future__ import annotations

import math
import threading
import zlib
from collections import deque

import numpy as np

WINDOW, STEP, LINE_QUANTILE, DRAWS = 500, 100, 0.999, 2000
TAIL_QUANTILE, OK_FACTOR, BAD_FACTOR, HORIZON, FALSE_ALARM, RUNS, CAP = 0.05, 1.3, 2.5, 5000, 0.01, 4000, 1.5


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


def cusum_limit(p_ok: float, p_bad: float, seed: str) -> float:
    """The smallest limit at which a stream whose tail share is `p_ok` alarms with probability at most FALSE_ALARM
    within HORIZON answers (RUNS simulated streams, a fixed seed)."""
    up, down = math.log(p_bad / p_ok), math.log((1 - p_bad) / (1 - p_ok))
    rng = np.random.default_rng(zlib.crc32(("cusum:" + seed).encode()))
    s, top = np.zeros(RUNS), np.zeros(RUNS)
    for _ in range(HORIZON):
        s = np.maximum(0.0, s + np.where(rng.random(RUNS) < p_ok, up, down))
        top = np.maximum(top, s)
    return float(np.quantile(top, 1 - FALSE_ALARM))


def confidence_baseline(conf, seed: str) -> dict | None:
    """The confidence signal's record for certification examples whose top probabilities are `conf`, or None when the
    examples cannot support it (a tail share too large to tell a rise from normal)."""
    conf = np.asarray(conf, dtype=float)
    if len(conf) < 20:
        return None
    thr = float(np.quantile(conf, TAIL_QUANTILE))
    p0 = max(float((conf < thr).mean()), 1.0 / len(conf))
    p_ok, p_bad = OK_FACTOR * p0, BAD_FACTOR * p0
    if p_bad > 0.5:
        return None
    return {
        "tail_quantile": TAIL_QUANTILE,
        "tail_threshold": thr,
        "tail_share": p0,
        "p_ok": p_ok,
        "p_bad": p_bad,
        "limit": cusum_limit(p_ok, p_bad, seed),
        "false_alarm": FALSE_ALARM,
        "horizon": HORIZON,
    }


def baseline(codes, k: int, seed: str, conf=None) -> dict:
    """The record stored with a certified task: the answer mix's, and the confidence signal's when `conf` (the
    certification examples' top probabilities) supports one."""
    out = _mix_baseline(codes, k, seed)
    cb = confidence_baseline(conf, seed) if conf is not None else None
    if cb is not None:
        out["confidence"] = cb
    return out


def _mix_baseline(codes, k: int, seed: str) -> dict:
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
    """The window and the confidence statistic of one task. `base` is a `baseline` record. A record without a
    `confidence` block (a task made before it existed) runs the answer mix alone."""

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
        self.mix_alerting = False
        self.first_alert_at: int | None = None
        c = base.get("confidence")
        self._conf = c
        self.statistic = 0.0
        self.conf_alerting = False
        self.conf_alerts = 0
        if c:
            self._up = math.log(c["p_bad"] / c["p_ok"])
            self._down = math.log((1 - c["p_bad"]) / (1 - c["p_ok"]))
        self.shared_skipped = 0  # answers scored in a shared engine call (--concurrency batched), not observed
        self._lock = threading.Lock()

    @property
    def alerting(self) -> bool:
        return self.mix_alerting or self.conf_alerting

    def observe(
        self, index: int, certified: bool = False, confidence: float | None = None, shared: bool = False
    ) -> None:
        """Count one answer of the task. `shared`: it was scored in an engine call shared with other requests
        (--concurrency batched), a function of the traffic, while the baseline was read one request at a time: it is
        counted as skipped, not observed."""
        with self._lock:
            if shared:
                self.shared_skipped += 1
                return
            if len(self._ring) == self._ring.maxlen:
                self._counts[self._ring[0]] -= 1
            self._ring.append(int(index))
            self._counts[int(index)] += 1
            self.answers += 1
            self.certified_answers += int(certified)
            if len(self._ring) == self._ring.maxlen and self.answers - self.checked_at >= self.base["step"]:
                self.checked_at = self.answers
                self.distance = distance(self._counts, self.base["option_counts"])
                self.mix_alerting = self.distance > self.base["line"]
                self.alerts += int(self.mix_alerting)
            if self._conf and confidence is not None:
                c = self._conf
                step = self._up if confidence < c["tail_threshold"] else self._down
                self.statistic = min(CAP * c["limit"], max(0.0, self.statistic + step))
                if self.statistic > c["limit"]:
                    if not self.conf_alerting:
                        self.conf_alerts += 1
                    self.conf_alerting = True
                elif self.statistic < c["limit"] / 2:
                    self.conf_alerting = False
            if self.alerting and self.first_alert_at is None:
                self.first_alert_at = self.answers

    def facts(self) -> dict:
        with self._lock:
            out = {
                "state": "traffic_changed" if self.alerting else "ok",
                "window_answers": len(self._ring),
                "answers": self.answers,
                "certified_answers": self.certified_answers,
                "checked_at": self.checked_at or None,
                "distance": self.distance,
                "line": self.base["line"],
                "alerts": self.alerts,
                "option_counts": self._counts.tolist(),
                "signals": [
                    n for n, on in (("answer_mix", self.mix_alerting), ("confidence", self.conf_alerting)) if on
                ],
                "first_alert_at": self.first_alert_at,
                **({"shared_skipped": self.shared_skipped} if self.shared_skipped else {}),
            }
            if self._conf:
                out["confidence"] = {
                    "statistic": self.statistic,
                    "limit": self._conf["limit"],
                    "tail_threshold": self._conf["tail_threshold"],
                    "tail_share": self._conf["tail_share"],
                    "alerts": self.conf_alerts,
                }
            return out
