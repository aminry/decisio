# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Fail when a fenced code block in the rendered documentation has no language.

The project's website renders README.md, docs/*.md and examples/tasks/README.md, and guesses the highlighting of a
fence that names no language. Every opening fence (``` or ~~~, indented or not) must name one; closing fences are bare.

  python3 scripts/check_code_fences.py            the default files
  python3 scripts/check_code_fences.py FILE ...   the files given
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")


def default_files() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), ROOT / "examples" / "tasks" / "README.md"]


def bare_fences(text: str) -> list[int]:
    """1-based line numbers of opening fences without a language."""
    bare, opener = [], None
    for n, line in enumerate(text.splitlines(), 1):
        m = FENCE.match(line)
        if not m:
            continue
        if opener is None:
            opener = m.group(2)
            if not m.group(3).strip():
                bare.append(n)
        elif m.group(2)[0] == opener[0] and len(m.group(2)) >= len(opener) and not m.group(3).strip():
            opener = None
    return bare


def main(argv: list[str]) -> int:
    files = [Path(a) for a in argv] or default_files()
    found = [(f, n) for f in files for n in bare_fences(f.read_text(encoding="utf-8"))]
    for f, n in found:
        name = f.relative_to(ROOT) if f.is_absolute() and f.is_relative_to(ROOT) else f
        print(f"{name}:{n}: code fence without a language (write e.g. ```bash, ```json or ```text)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
