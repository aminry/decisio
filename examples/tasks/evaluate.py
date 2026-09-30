# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Measure what registration does on your own held-out tickets: paired accuracy before and after, with an interval.

It removes any task registered under --id, answers every held-out ticket (before), registers the training tickets,
answers the same tickets again (after), and reports both accuracies and the paired difference with a bootstrap
interval over tickets. Keep the held-out tickets out of the training file, or the "after" number means nothing.

    python examples/tasks/evaluate.py --url http://127.0.0.1:8000 --id ticket-routing \\
        --train examples/tasks/train.csv --heldout examples/tasks/heldout.csv
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ask, call, load_question, load_rows  # noqa: E402
from register import body_for, summary  # noqa: E402


def answers(url, rows, question):
    out, tasks = [], set()
    for text, _ in rows:
        choice, _, headers = ask(url, text, question)
        out.append(choice)
        tasks.update(t for t in headers.get("x-decisio-tasks", "").split(",") if t)
    return out, tasks


def paired_interval(before, after, resamples=10000, seed=0):
    """95% bootstrap interval of mean(after) - mean(before) over items resampled with replacement, pairs kept."""
    rng = random.Random(seed)
    n = len(before)
    diffs = sorted(
        sum(after[i] - before[i] for i in (rng.randrange(n) for _ in range(n))) / n for _ in range(resamples)
    )
    return diffs[int(0.025 * resamples)], diffs[int(0.975 * resamples) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--id", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--question", default=None)
    ap.add_argument("--resamples", type=int, default=10000)
    a = ap.parse_args()
    question, train, held = load_question(a.question), load_rows(a.train), load_rows(a.heldout)
    overlap = {t for t, _ in train} & {t for t, _ in held}
    if overlap:
        raise SystemExit(f"{len(overlap)} held-out tickets are also training tickets; keep the two files apart")
    call(a.url, f"/v1/tasks/{a.id}", method="DELETE")  # a 404 (nothing registered) is fine
    gold = [label for _, label in held]
    plain, tasks = answers(a.url, held, question)
    if tasks:
        raise SystemExit(f"tasks {sorted(tasks)} still apply to this question; delete them first")
    t0 = time.perf_counter()
    status, _, task = call(a.url, "/v1/tasks", body_for(a.id, train, question))
    if status != 200:
        raise SystemExit(f"registration refused ({status}): {task}")
    print(summary(task, time.perf_counter() - t0))
    registered, tasks = answers(a.url, held, question)
    before = [int(c == g) for c, g in zip(plain, gold)]
    after = [int(c == g) for c, g in zip(registered, gold)]
    lo, hi = paired_interval(before, after, a.resamples)
    n = len(gold)
    print(f"held-out tickets: {n}; matched task: {', '.join(sorted(tasks)) or 'none (served as before)'}")
    print(
        f"accuracy before {sum(before) / n:.3f}, after {sum(after) / n:.3f}; "
        f"difference {100 * (sum(after) - sum(before)) / n:+.1f} points, "
        f"95% interval [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    )
    print(
        f"changed answers: {sum(b != x for b, x in zip(plain, registered))} of {n} "
        f"({sum(x > b for b, x in zip(before, after))} fixed, {sum(x < b for b, x in zip(before, after))} broken)"
    )


if __name__ == "__main__":
    main()
