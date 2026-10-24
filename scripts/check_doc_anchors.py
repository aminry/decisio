# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Fail when a link in the documentation points at a file that is not there or at a heading that is not in it.

The project's website renders README.md, EVAL_CARD.md and docs/*.md (and docs/design/*.md); a heading that is reworded
changes its anchor (GitHub's rule: lower case, punctuation dropped, spaces to hyphens), and a link written before the
change breaks without any test noticing. Every relative link `[text](file.md#anchor)` or `[text](#anchor)` is checked:
the file must exist, and the anchor must be a heading of it (or an explicit `<a name=...>`). Links inside code fences
are skipped, as are web links.

  python3 scripts/check_doc_anchors.py            the default files
  python3 scripts/check_doc_anchors.py FILE ...   the files given
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
LINK = re.compile(r"\]\(([^)\s]+)\)")


def default_files() -> list[Path]:
    return [
        ROOT / "README.md",
        ROOT / "EVAL_CARD.md",
        *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / "docs" / "design").glob("*.md")),
        ROOT / "examples" / "tasks" / "README.md",
    ]


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: markup removed, lower case, punctuation dropped, spaces to hyphens."""
    h = re.sub(r"`", "", heading)
    h = re.sub(r"<[^>]+>", "", h)
    h = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", h)
    return re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-")


def anchors(text: str) -> set[str]:
    """The anchors of a document: its headings (a repeated heading gets -1, -2, ...) and its explicit names."""
    seen: dict[str, int] = {}
    out: set[str] = set()
    fence = False
    for line in text.splitlines():
        if line.startswith(("```", "~~~")):
            fence = not fence
            continue
        m = None if fence else HEADING.match(line)
        if m:
            s = slug(m.group(2))
            n = seen.get(s, 0)
            seen[s] = n + 1
            out.add(s if n == 0 else f"{s}-{n}")
    out.update(re.findall(r'<a (?:name|id)="([^"]+)"', text))
    return out


def broken(path: Path, cache: dict[Path, set[str] | None]) -> list[tuple[int, str, str]]:
    """(line, link, why) for each link of the file that does not resolve."""
    found, fence = [], False
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith(("```", "~~~")):
            fence = not fence
            continue
        if fence:
            continue
        for m in LINK.finditer(line):
            link = m.group(1)
            if link.startswith(("http://", "https://", "mailto:")):
                continue
            target, _, frag = link.partition("#")
            dest = path if not target else Path(os.path.normpath(path.parent / target))
            if not dest.exists():
                found.append((n, link, "no such file"))
                continue
            if frag and dest.suffix == ".md":
                if dest not in cache:
                    cache[dest] = anchors(dest.read_text(encoding="utf-8"))
                if frag not in (cache[dest] or set()):
                    found.append((n, link, "no such heading"))
    return found


def main(argv: list[str]) -> int:
    files = [Path(a).resolve() for a in argv] or default_files()
    cache: dict[Path, set[str] | None] = {}
    count = 0
    for f in files:
        for n, link, why in broken(f, cache):
            name = f.relative_to(ROOT) if f.is_relative_to(ROOT) else f
            print(f"{name}:{n}: {link}: {why}")
            count += 1
    return 1 if count else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
