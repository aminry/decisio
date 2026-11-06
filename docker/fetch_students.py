# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Fetch the fast table's students into an image at build time (docker/Dockerfile.table).

    python3 fetch_students.py OUT [--encoder] [--licence LICENSES/MIT.txt]

The files are the ones decisio pins (decisio.serve.fast_table.pinned_files: the static students potion-base-8M and
potion-multilingual-128M, and with --encoder bge-base-en-v1.5's ONNX file and tokenizer), each fetched at its pinned
revision, checked against its pinned sha256 and written to OUT/<repo>/<revision>/<file>, where the server reads them
when DECISIO_STUDENTS_DIR=OUT (fast_table.student_file). A file whose sha256 is not the pin stops the build.

Each repository's model card (README.md) at the pinned revision is kept beside its files, and the build stops unless
the card declares the MIT licence; the MIT text goes to OUT/LICENSES/MIT.txt with --licence. MANIFEST.json lists every
file with its sha256 and size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

LICENCE = "mit"  # every student decisio pins is MIT (their model cards); another licence stops the build


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def card_licence(text: str) -> str | None:
    """The `license:` of a model card's front matter, or None."""
    if not text.startswith("---"):
        return None
    for line in text.split("\n---", 1)[0].splitlines()[1:]:
        if line.strip().lower().startswith("license:"):
            return line.split(":", 1)[1].strip().strip("'\"").lower()
    return None


def fetch(pins: list[dict], out, cards: bool = False) -> list[dict]:
    """Fetch, check and lay out the pinned files under `out`; with `cards`, each repository's model card too, whose
    licence must be MIT. SystemExit on a sha256 that is not the pin or another licence."""
    from huggingface_hub import hf_hub_download

    out = Path(out)
    manifest = []
    for p in pins:
        src = hf_hub_download(p["repo"], p["file"], revision=p["revision"])
        digest = sha256_of(src)
        if digest != p["sha256"]:
            raise SystemExit(
                f"{p['repo']} {p['file']} at {p['revision']}: sha256 {digest}, not the pinned {p['sha256']}"
            )
        dst = out / p["repo"] / p["revision"] / p["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        manifest.append({**p, "bytes": dst.stat().st_size})
    if cards:
        for repo, revision in sorted({(p["repo"], p["revision"]) for p in pins}):
            src = hf_hub_download(repo, "README.md", revision=revision)
            text = Path(src).read_text(encoding="utf-8")
            if card_licence(text) != LICENCE:
                raise SystemExit(f"{repo} at {revision}: its model card declares {card_licence(text)!r}, not MIT")
            dst = out / repo / revision / "MODEL_CARD.md"
            shutil.copyfile(src, dst)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--encoder", action="store_true", help="also bge-base-en-v1.5's ONNX file and tokenizer (436 MB)")
    ap.add_argument("--licence", help="the MIT licence text to copy to OUT/LICENSES/MIT.txt")
    a = ap.parse_args(argv)
    from decisio.serve.fast_table import pinned_files

    manifest = fetch(pinned_files(encoder=a.encoder), a.out, cards=True)
    if a.licence:
        (Path(a.out) / "LICENSES").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.licence, Path(a.out) / "LICENSES" / "MIT.txt")
    total = sum(m["bytes"] for m in manifest)
    print(f"students: {len(manifest)} files, {total / 1e6:.0f} MB in {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
