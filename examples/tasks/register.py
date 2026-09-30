# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Register a recurring question with a decisio server from labelled examples, and print what the server fitted.

The examples file is a CSV with `text` and `label` columns, or a JSONL of {"text", "label"}; every label must be one of
the question's option keys (question.json). The request body is written with --out (examples.json in this directory is
train.csv in that form), and --dry-run stops before sending it.

    python examples/tasks/register.py --url http://127.0.0.1:8000 --id ticket-routing --data examples/tasks/train.csv
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import call, load_question, load_rows, request_for  # noqa: E402


def body_for(task_id, rows, question):
    keys = list(question["criteria"])
    unknown = sorted({label for _, label in rows} - set(keys))
    if unknown:
        raise SystemExit(f"labels that are not option keys of the question: {unknown}")
    return {
        "id": task_id,
        "examples": [{"request": request_for(text, question), "answer": label} for text, label in rows],
    }


def summary(task, seconds=None):
    """The fit, in words: what the server will apply to this question from now on, and why."""
    lines = [
        f"task {task['id']!r}: {task['n_examples']} examples, {len(task['options'])} options, "
        f"at least {task['per_option_min']} per option" + (f", registered in {seconds:.0f} s" if seconds else "")
    ]
    cal = task["calibration"]
    line = f"  calibration: {'applied' if cal.get('applied') else 'not applied'} ({cal.get('reason')})"
    if "cv_logloss_plain" in cal:
        line += (
            f"; cross-validated log loss {cal['cv_logloss_plain']} -> {cal['cv_logloss_fitted']}, "
            f"accuracy {cal['cv_acc_plain']} -> {cal['cv_acc_fitted']}"
        )
    lines.append(line)
    head = task["head"]
    line = f"  head: {'applied' if head.get('applied') else 'not applied'} ({head.get('reason')})"
    if head.get("applied"):
        line += f"; penalty {head['lambda']}"
    if head.get("cv_logloss"):
        line += "; cross-validated log loss by penalty " + json.dumps(head["cv_logloss"])
    lines.append(line)
    if not cal.get("applied") and not head.get("applied"):
        lines.append("  nothing applied: this question is served exactly as before registration")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--id", required=True, help="the task's name (registering the same id again replaces it)")
    ap.add_argument("--data", required=True, help="CSV or JSONL of text and label")
    ap.add_argument("--question", default=None, help="the question (default: question.json here)")
    ap.add_argument("--out", default=None, help="also write the request body to this file")
    ap.add_argument("--dry-run", action="store_true", help="build (and --out) the body, send nothing")
    a = ap.parse_args()
    rows = load_rows(a.data)
    body = body_for(a.id, rows, load_question(a.question))
    if a.out:
        Path(a.out).write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n")
    counts = collections.Counter(label for _, label in rows)
    print(f"{len(rows)} examples, {len(counts)} labels, {min(counts.values())} to {max(counts.values())} per label")
    if a.dry_run:
        return
    t0 = time.perf_counter()
    status, _, task = call(a.url, "/v1/tasks", body)
    if status != 200:
        raise SystemExit(f"registration refused ({status}): {task}")
    print(summary(task, time.perf_counter() - t0))


if __name__ == "__main__":
    main()
