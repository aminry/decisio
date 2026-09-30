# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Ask the recurring question about one or more tickets and show what the server did.

For each ticket: the choice, the most probable options, and the response headers that say which route answered and
which registered task the question matched (x-decisio-tasks; absent when it matched none).

    python examples/tasks/ask.py --url http://127.0.0.1:8000 "My card keeps getting declined when I renew."
    python examples/tasks/ask.py --url http://127.0.0.1:8000 --file examples/tasks/heldout.csv --limit 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ask, load_question, load_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tickets", nargs="*", help="ticket texts")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--file", default=None, help="CSV or JSONL of tickets (labels, if present, are shown)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top", type=int, default=3, help="how many options to show")
    ap.add_argument("--question", default=None)
    a = ap.parse_args()
    items = [(t, None) for t in a.tickets] + (load_rows(a.file) if a.file else [])
    if not items:
        ap.error("give ticket texts or --file")
    question = load_question(a.question)
    for text, label in items[: a.limit]:
        choice, probs, headers = ask(a.url, text, question)
        top = sorted(probs.items(), key=lambda kv: -kv[1])[: a.top]
        print(f"{text[:90]}")
        print(
            f"  choice {choice}"
            + (f" (label {label})" if label else "")
            + f"; x-decisio-tasks: {headers.get('x-decisio-tasks', 'none')}; route {headers.get('x-decisio-route')}, "
            f"{headers.get('x-decisio-server-ms')} ms"
        )
        print("  " + ", ".join(f"{k} {p:.3f}" for k, p in top))


if __name__ == "__main__":
    main()
