#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The items file of the conformance gates C3 and C4 (`decisio.serve.systemone_conformance`), from two public sets.

  BoolQ (validation, CC BY-SA 3.0)  yes/no: a passage and a question, answered with the noul type
  BANKING77 (test, CC BY 4.0)       a choice among 77 intents, each shown by its words

Each item is written in both request forms the gates compare, `/v1/answer` and `/v1/systemone`. The labels are shown as
words in both ("card arrival", not `card_arrival`), because the served renderer shows bare snake_case labels as words
and C3 requires the two forms to give the same prompt. Nothing is redistributed: the script downloads the two sets,
draws a fixed sample (random.Random(27)), and prints the sha256 of what it downloaded, which belongs in the run record.

    python benchmarks/make_conformance_items.py --out conformance_items.json [--boolq 100] [--banking77 100]

Standard library only. The file holds benchmark item text: it is an input of the gates, not a record, and is not
committed (the records in runs/ keep ids, labels and probabilities only).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BOOLQ_URL = "https://datasets-server.huggingface.co/rows?dataset=google/boolq&config=default&split=validation"
BANKING77_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
SEED = 27
BOOLQ_INSTRUCTIONS = "Answer the question using only the passage."
BANKING77_INSTRUCTIONS = "What is the customer's intent in this banking support message? Pick the closest intent."


def served_name() -> str:
    try:
        from decisio.names import SERVED_NAME

        return SERVED_NAME
    except ImportError:  # run from a bare checkout
        return "decisio-qwen3.6-35b-a3b-letters"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "decisio-conformance-items"})
    for attempt in range(7):
        try:
            with urllib.request.urlopen(request, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if (e.code < 500 and e.code != 429) or attempt == 6:
                raise
            time.sleep(5 * (attempt + 1))  # the datasets server answers 502 now and then, and 429 to a quick burst
    raise AssertionError("unreachable")


def fetch_boolq() -> tuple[list[dict], bytes]:
    """The validation split of google/boolq through the Hugging Face datasets server, 100 rows per request."""
    rows, raw, offset = [], b"", 0
    while True:
        page = fetch(f"{BOOLQ_URL}&offset={offset}&length=100")
        data = json.loads(page)
        rows += [r["row"] for r in data["rows"]]
        raw += page
        offset += 100
        time.sleep(0.7)
        if offset >= data["num_rows_total"]:
            return rows, raw


def boolq_item(i: int, row: dict, model: str) -> dict:
    state = {"passage": row["passage"], "question": row["question"]}
    return {
        "task": "boolq",
        "i": i,
        "label": int(bool(row["answer"])),
        "state": state,
        "answer": {"state": state, "questions": [{"kind": "noul", "instructions": BOOLQ_INSTRUCTIONS}]},
        "systemone": {
            "state": state,
            "model": model,
            "questions": {"q": {"type": "noul", "instructions": BOOLQ_INSTRUCTIONS}},
        },
    }


def banking77_item(i: int, text: str, category: str, categories: list[str], model: str) -> dict:
    options = [c.replace("_", " ") for c in categories]
    return {
        "task": "banking77",
        "i": i,
        "label": categories.index(category),
        "state": text,
        "answer": {
            "state": text,
            "questions": [{"kind": "choice", "instructions": BANKING77_INSTRUCTIONS, "options": options}],
        },
        "systemone": {
            "state": text,
            "model": model,
            "questions": {
                "q": {"type": "choice", "instructions": BANKING77_INSTRUCTIONS, "criteria": {o: None for o in options}}
            },
        },
    }


def build(
    boolq_rows: list[dict], banking_rows: list[tuple[str, str]], n_boolq: int, n_banking: int, model: str
) -> list[dict]:
    rng = random.Random(SEED)
    categories = sorted({c for _, c in banking_rows})
    items = [boolq_item(i, boolq_rows[i], model) for i in sorted(rng.sample(range(len(boolq_rows)), n_boolq))]
    items += [
        banking77_item(i, banking_rows[i][0], banking_rows[i][1], categories, model)
        for i in sorted(rng.sample(range(len(banking_rows)), n_banking))
    ]
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--boolq", type=int, default=100, help="BoolQ items to draw")
    ap.add_argument("--banking77", type=int, default=100, help="BANKING77 items to draw")
    a = ap.parse_args()
    boolq_rows, raw_boolq = fetch_boolq()
    raw_banking = fetch(BANKING77_URL)
    banking_rows = [(r["text"], r["category"]) for r in csv.DictReader(io.StringIO(raw_banking.decode()))]
    items = build(boolq_rows, banking_rows, a.boolq, a.banking77, served_name())
    Path(a.out).write_text(json.dumps(items, ensure_ascii=False))
    print(f"{len(items)} items -> {a.out}")
    for name, url, raw in (("BoolQ validation", BOOLQ_URL, raw_boolq), ("BANKING77 test", BANKING77_URL, raw_banking)):
        print(f"{name}: sha256 {hashlib.sha256(raw).hexdigest()}  {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
