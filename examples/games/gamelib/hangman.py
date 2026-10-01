# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Hangman: the engine, the dictionary, and the entropy teacher.

The engine knows the secret word; everything that plays sees only a `HangmanView` (the revealed pattern, the wrong
guesses so far, the word length), which is also what is rendered to text for the model. The teacher is the classic
information-theoretic solver: among the dictionary words still consistent with the view, guess the letter whose answer
(the positions it occupies, or none) has the highest entropy over those words.
"""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
MAX_WRONG = 6
DATA = Path(__file__).parent / "data"


def load_words(name="words.txt"):
    return [w for w in (DATA / name).read_text().split() if w]


def heldout_words():
    """The 200 words that no registration example and no teaching state ever uses."""
    return load_words("heldout_words.txt")


def dev_words():
    """100 words for iterating on prompts: not held out, not a source of a teaching example."""
    return load_words("dev_words.txt")


@dataclass(frozen=True)
class HangmanView:
    """What a player sees: the pattern with `_` for hidden letters, the wrong guesses in order, the allowed misses."""

    pattern: str
    wrong: tuple
    max_wrong: int = MAX_WRONG

    @property
    def length(self):
        return len(self.pattern)

    @property
    def guessed(self):
        return frozenset(self.pattern.replace("_", "")) | frozenset(self.wrong)

    @property
    def remaining(self):
        return [c for c in ALPHABET if c not in self.guessed]


class Hangman:
    """One game. `step(letter)` returns True on a hit; `status` is playing, won or lost."""

    def __init__(self, word, max_wrong=MAX_WRONG):
        if not word.isalpha() or not word.islower() or not word.isascii():
            raise ValueError(f"word must be lowercase ascii letters: {word!r}")
        self.word, self.max_wrong = word, max_wrong
        self.pattern = ["_"] * len(word)
        self.wrong = []
        self.moves = []

    @classmethod
    def from_seed(cls, seed, words=None, max_wrong=MAX_WRONG):
        return cls(random.Random(seed).choice(words or load_words()), max_wrong)

    @property
    def status(self):
        if "_" not in self.pattern:
            return "won"
        return "lost" if len(self.wrong) >= self.max_wrong else "playing"

    @property
    def over(self):
        return self.status != "playing"

    def view(self):
        return HangmanView("".join(self.pattern), tuple(self.wrong), self.max_wrong)

    def step(self, letter):
        if self.over:
            raise ValueError("the game is over")
        if letter not in ALPHABET or len(letter) != 1:
            raise ValueError(f"not a letter: {letter!r}")
        if letter in self.view().guessed:
            raise ValueError(f"already guessed: {letter!r}")
        self.moves.append(letter)
        hit = letter in self.word
        if hit:
            for i, c in enumerate(self.word):
                if c == letter:
                    self.pattern[i] = letter
        else:
            self.wrong.append(letter)
        return hit


class EntropySolver:
    """The teacher. `candidates(view)` are the dictionary words the view allows; `scores(view)` the entropy in bits of
    each unguessed letter's answer over them; `choose(view)` the best letter (ties: more candidates contain it, then
    alphabetical order). When no word fits (a word outside the dictionary), it falls back to letter frequency over the
    words of that length."""

    def __init__(self, words=None):
        self.by_length = {}
        for w in words or load_words():
            self.by_length.setdefault(len(w), []).append(w)
        # one row per word, one column per position, letters as 0..25
        self.matrix = {
            n: np.array([[ord(c) - 97 for c in w] for w in ws], dtype=np.int8) for n, ws in self.by_length.items()
        }

    def _mask(self, view):
        """The rows of the view's length matrix that the view allows."""
        m = self.matrix.get(len(view.pattern))
        if m is None:
            return None, np.zeros(0, dtype=bool)
        ok = np.ones(len(m), dtype=bool)
        banned = [ord(c) - 97 for c in set(view.wrong) | (set(view.pattern) - {"_"})]
        for i, c in enumerate(view.pattern):
            if c != "_":
                ok &= m[:, i] == ord(c) - 97
            else:  # a letter already guessed cannot sit at a hidden position
                for b in banned:
                    ok &= m[:, i] != b
        return m, ok

    def candidates(self, view):
        _, ok = self._mask(view)
        words = self.by_length.get(len(view.pattern), [])
        return [words[i] for i in np.flatnonzero(ok)]

    def _letter_stats(self, view):
        """({letter: entropy}, {letter: share of candidates containing it}) for the unguessed letters."""
        m, ok = self._mask(view)
        rows = m[ok] if m is not None else np.zeros((0, 0), dtype=np.int8)
        n = len(rows)
        entropy, presence = {}, {}
        weights = 1 << np.arange(rows.shape[1], dtype=np.int64) if n else None
        for letter in view.remaining:
            if n == 0:
                pool = self.matrix.get(len(view.pattern))
                share = float((pool == ord(letter) - 97).any(axis=1).mean()) if pool is not None and len(pool) else 0.0
                entropy[letter], presence[letter] = share, 0.0
                continue
            hit = rows == ord(letter) - 97
            _, counts = np.unique((hit * weights).sum(axis=1), return_counts=True)
            p = counts / n
            entropy[letter] = float(-(p * np.log2(p)).sum())
            presence[letter] = float(hit.any(axis=1).mean())
        return entropy, presence

    def scores(self, view):
        return self._letter_stats(view)[0]

    def presence(self, view):
        """The share of the candidate words that contain each unguessed letter."""
        return self._letter_stats(view)[1]

    def choose(self, view):
        scores, presence = self._letter_stats(view)
        return max(sorted(scores), key=lambda c: (round(scores[c], 9), presence[c]))


def teacher_letter(view, solver):
    return solver.choose(view)


def random_letter(view, rng):
    return rng.choice(view.remaining)
