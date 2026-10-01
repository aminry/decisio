# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Rebuild data/words.txt, data/heldout_words.txt and data/dev_words.txt: the Hangman dictionary, its 200 held-out words
and 100 development words.

The dictionary is the ENABLE word list (public domain) restricted to the 20,000 most frequent words of Peter Norvig's
count_1w.txt (MIT licence; counts from the Google Web Trillion Word Corpus), lowercase letters only, 4 to 12 letters.
The held-out words are 200 drawn from it with a fixed seed; the registration examples never use them.

    curl -O https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt
    curl -O https://norvig.com/ngrams/count_1w.txt
    python gamelib/build_words.py --enable enable1.txt --counts count_1w.txt
    python gamelib/build_words.py --check          # the committed files are what the sources give (needs the two files)
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data"
TOP = 20000
LENGTHS = (4, 12)
HELDOUT = 200
SEED = 20261002
DEV = 100  # words for iterating on prompts: neither held out nor a source of a teaching example
DEV_SEED = 20261003


def build(enable_path, counts_path):
    enable = {w.strip() for w in open(enable_path)}
    words = []
    for i, line in enumerate(open(counts_path)):
        if i >= TOP:
            break
        w = line.split("\t")[0]
        if w in enable and w.isascii() and w.isalpha() and w.islower() and LENGTHS[0] <= len(w) <= LENGTHS[1]:
            words.append(w)
    words = sorted(words)
    heldout = sorted(random.Random(SEED).sample(words, HELDOUT))
    taught = {r["word"] for r in json.loads((DATA / "hangman_teaching.json").read_text())}
    pool = sorted(set(words) - set(heldout) - taught)
    dev = sorted(random.Random(DEV_SEED).sample(pool, DEV))
    return words, heldout, dev


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--enable", required=True, help="enable1.txt")
    ap.add_argument("--counts", required=True, help="count_1w.txt")
    ap.add_argument("--check", action="store_true", help="compare with the committed files instead of writing")
    a = ap.parse_args()
    words, heldout, dev = build(a.enable, a.counts)
    outputs = {
        "words.txt": "\n".join(words) + "\n",
        "heldout_words.txt": "\n".join(heldout) + "\n",
        "dev_words.txt": "\n".join(dev) + "\n",
    }
    for name, text in outputs.items():
        path = DATA / name
        if a.check:
            if path.read_text() != text:
                sys.exit(f"{name} differs from what the sources give")
        else:
            path.write_text(text)
    print(
        f"{len(words)} words, {len(heldout)} held out; sources enable1.txt {sha256(a.enable)[:12]}, "
        f"count_1w.txt {sha256(a.counts)[:12]}"
    )


if __name__ == "__main__":
    main()
