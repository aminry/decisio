# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Add the calculator block to rows the way serving does: `python -m decisio.calculator render IN --out OUT`.

  python -m decisio.calculator render IN.jsonl[.gz] --out OUT.jsonl[.gz] [--field state] [--context-field context]
      [--version calc-v1]

Each row's state (a string or an object, rendered as the prompt renders it) is read with the row's context, if a
context field is named and present; where the calculator gives a block, the field becomes the rendered state, a
newline and the block (`decisio.calculator.attach`); where it abstains, the row is written unchanged. The counts per
outcome go to stdout as JSON.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys

from decisio.calculator import VERSIONS, Annotation, annotate, attach


def _open(path: str, mode: str):
    return gzip.open(path, mode + "t", encoding="utf-8") if path.endswith(".gz") else open(path, mode, encoding="utf-8")


def render_rows(src: str, dst: str, field: str = "state", context_field: str | None = None, version: str = "calc-v1"):
    from decisio.readout.letters import fmt_state

    counts: collections.Counter = collections.Counter()
    with _open(src, "r") as fin, _open(dst, "w") as fout:
        for n, line in enumerate(fin, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if field not in row:
                raise SystemExit(f"row {n}: no {field!r} field")
            context = row.get(context_field) if context_field else None
            ann = annotate(row[field], context=context, version=version)
            counts["rows"] += 1
            if isinstance(ann, Annotation):
                row[field] = attach(fmt_state(row[field]), ann)
                counts["shown"] += 1
            else:
                counts[f"none: {ann.reason}"] += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dict(counts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m decisio.calculator", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="add the calculator block to each row's state")
    r.add_argument("src")
    r.add_argument("--out", required=True)
    r.add_argument("--field", default="state")
    r.add_argument("--context-field", default=None)
    r.add_argument("--version", default=VERSIONS[-1], choices=VERSIONS)
    a = ap.parse_args(argv)
    counts = render_rows(a.src, a.out, a.field, a.context_field, a.version)
    print(json.dumps(counts, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
