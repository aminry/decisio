# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Box readout: one bounding box and a confidence from the logprobs of a generated `box_2d` reply.

The model is asked for one box in the format Gemma documents for detection,
`{"box_2d": [y_min, x_min, y_max, x_max], "label": ...}`, with coordinates on a 0-1000 grid and y first. The
reply is forced to start with `{"box_2d": [` (BOX_REPLY_PREFIX), so the first generated tokens are the digits
of y_min. Gemma's tokenizer writes every digit as its own token ("1", "2", "3"), a space before a number is a
separate "▁" token, and the end of a number is a token that starts with "," or "]" (",", "],", "]}"). A number
of one to four digits is therefore a run of digit steps followed by one terminator step.

Each step of the generation is `(chosen token, top-k [(token, logprob), ...])`. The readout follows the greedy
path. At each digit position it takes the distribution over the digit tokens (and, after the first digit, the
terminators, which compete with a further digit) given the greedy prefix, renormalised over those tokens,
with mass under 0.01 pruned. Branches off the greedy path are valued with the greedy path's own expected lower
digits, so the estimate is exact when only the last digit and the end of the number are uncertain, and
first-order accurate otherwise (the test suite measures the gap against a full digit tree).

The expected value of each edge is the box edge; the probability that an edge lies within 0.02 of the reported
value comes from the same distribution, and the raw confidence is the geometric mean of that over the four edges.
`ConfidenceMap` turns the raw confidence into a rate of IoU >= 0.5 on a calibration split.

Nothing here invents a box: a generation without four numbers inside a `box_2d` list raises BoxParseError.
The one reply that is not an error is the model saying there is no box: `null` where the list, or its first
coordinate, would be. `parse_coordinates` raises `BoxAbsentSignal` for it and `read_box` returns a `BoxAbsent`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

BOX_GRID = 1000  # the model's coordinate grid
MAX_DIGITS = 4  # 0 to 1000
PRUNE_BELOW = 0.01  # probability mass under this, within a position, is dropped before the expectation
CONFIDENCE_WINDOW = 0.02  # an edge counts as right when within this fraction of the image of the reported value
MIN_BOX_SIDE = 1 / BOX_GRID  # a degenerate box is widened to one grid unit

BOX_REPLY_PREFIX = '{"box_2d": ['
BOX_INSTRUCTION = (
    'Detect {target}. Output a single bounding box as JSON: {{"box_2d": [y_min, x_min, y_max, x_max], '
    '"label": "..."}} with coordinates normalised to a 0-1000 grid. Output only the JSON.'
)
BOX_DEFAULT_TARGET = "the main subject of the image"
BOX_CONFIDENCE_FILE = "box_confidence.json"

_WS = " \n\t\r"
_DIGITS = "0123456789"

# a step as given by an engine: the chosen token and the top-k (token, logprob) pairs at that step
RawStep = tuple[str, Sequence[tuple[str, float]]]


class BoxParseError(ValueError):
    """The generation has no readable `box_2d` list of four numbers. The route turns this into a clear failure."""

    def __init__(self, reason: str, text: str = ""):
        super().__init__(f"{reason}: {text[:120]!r}" if text else reason)
        self.reason = reason
        self.text = text


class BoxAbsentSignal(Exception):
    """The generation says there is no box: `null` as the `box_2d` value or as its first element, with no number
    before it. Not a parse error; `parse_coordinates` raises it and the readout turns it into a `BoxAbsent`."""

    def __init__(self, text: str = ""):
        super().__init__(f"the model wrote no box: {text[:120]!r}")
        self.text = text


@dataclass(frozen=True)
class BoxAbsent:
    """The readout of a reply that says the target is not there. The route answers the whole image at confidence 0."""

    text: str = ""
    tokens: int = 0


def _starts_null(text: str) -> bool:
    """Whether `text`, past whitespace, starts with the word `null` (not `nullable`, not a number)."""
    t = text.lstrip(_WS)
    if not t.startswith("null"):
        return False
    rest = t[4:]
    return rest == "" or not (rest[0].isalnum() or rest[0] == "_")


def box_target(instructions) -> str:
    """What the question looks for, as text: a string (as written), an object or an array (compact JSON), or the main
    subject when the question carries nothing."""
    if instructions is None:
        return BOX_DEFAULT_TARGET
    if isinstance(instructions, str):
        return instructions.strip().rstrip(".") or BOX_DEFAULT_TARGET
    return json.dumps(instructions, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def render_box_prompt(instructions) -> str:
    """The Box instruction text for one question. The image itself is placed by the image path, not here.

    `instructions` is what the question carries: a string (used as the target), an object or an array
    (written out as compact JSON), or None (the main subject).
    """
    return BOX_INSTRUCTION.format(target=box_target(instructions))


def presence_instructions(instructions) -> str:
    """The yes/no question that reads whether a Box question's target is in the picture at all."""
    return f"Is this in the picture: {box_target(instructions)}?"


# ---------------------------------------------------------------------------------------------------------------
# token classes


def _kind(token: str) -> str:
    """'digit', 'term' (ends a number), 'ws' or 'other'."""
    s = token.replace("▁", " ").strip(_WS)
    if s == "":
        return "ws"
    if len(s) == 1 and s in _DIGITS:
        return "digit"
    if s[0] in ",]":
        return "term"
    return "other"


def _digit_value(token: str) -> int:
    return int(token.replace("▁", " ").strip(_WS))


@dataclass(frozen=True)
class _Dist:
    """Distribution at one position of a number, renormalised over the allowed tokens and pruned."""

    digits: tuple[float, ...]  # 10 probabilities
    term: float  # probability that the number ends here


def _position_dist(step: RawStep, n_before: int, prune_below: float = PRUNE_BELOW) -> _Dist:
    """Digits (if fewer than four are written) and terminators (if at least one is) from a step's top-k.

    `prune_below`: the production default (PRUNE_BELOW, 0.01); the exact-tree debug readout reads the same
    top-k at a finer cut (down to its own `min_prob`), since a step's top-k can carry more resolution than the
    served estimator keeps."""
    chosen, top = step
    allow_digit = n_before < MAX_DIGITS
    allow_term = n_before >= 1
    digits = [0.0] * 10
    term = 0.0
    seen_chosen = False
    for tok, lp in top:
        k = _kind(tok)
        p = math.exp(lp)
        if tok == chosen:
            seen_chosen = True
        if k == "digit" and allow_digit:
            digits[_digit_value(tok)] += p
        elif k == "term" and allow_term:
            term += p
    ck = _kind(chosen)
    if not top:
        # a forced token has no alternatives
        if ck == "digit":
            digits[_digit_value(chosen)] = 1.0
        else:
            term = 1.0
    elif not seen_chosen:
        # the chosen token fell outside the top-k: it cannot have more mass than the smallest listed
        floor = math.exp(min(lp for _, lp in top))
        if ck == "digit":
            digits[_digit_value(chosen)] += floor
        else:
            term += floor
    total = sum(digits) + term
    digits = [d / total for d in digits]
    term /= total
    keep_digit = _digit_value(chosen) if ck == "digit" else -1
    digits = [d if (d >= prune_below or i == keep_digit) else 0.0 for i, d in enumerate(digits)]
    if term < prune_below and ck != "term":
        term = 0.0
    total = sum(digits) + term
    return _Dist(tuple(d / total for d in digits), term / total)


# ---------------------------------------------------------------------------------------------------------------
# one coordinate


@dataclass(frozen=True)
class EdgeEstimate:
    """One coordinate on the 0-1000 grid."""

    greedy: int  # the greedy reading
    expected: float  # the expected value, before clipping
    p_within: float  # probability that the edge is within CONFIDENCE_WINDOW of `expected`
    p_greedy: float  # probability the greedy digits are the whole answer under the pruned distributions
    n_digits: int
    support: tuple[tuple[float, float], ...] = field(default=(), repr=False)  # (value, probability), by value


def estimate_coordinate(
    digit_steps: Sequence[RawStep],
    term_step: RawStep | None,
    branch_values: dict[tuple[int, int], float] | None = None,
) -> EdgeEstimate:
    """Expected value of a number from the steps of its digits and of the step that ended it (None: not seen).

    `branch_values`, keyed `(position, digit)`: for a branch off the greedy path at that position (position `n`
    is the terminator step reading one more digit instead), the real value of that branch (`expand_branches`'s
    decode, exact), used in place of the greedy path's own expected lower digits (the plain estimator's
    approximation, kept when a branch is not in `branch_values`, including when `branch_values` is None)."""
    n = len(digit_steps)
    if not 1 <= n <= MAX_DIGITS:
        raise BoxParseError(f"a coordinate has 1 to {MAX_DIGITS} digits, found {n}")
    dists = [_position_dist(s, j) for j, s in enumerate(digit_steps)]
    greedy_digits = [_digit_value(s[0]) for s in digit_steps]
    final = _position_dist(term_step, n) if term_step is not None else _Dist((0.0,) * 10, 1.0)
    branch_values = branch_values or {}

    # mean of each position's digit, given the greedy prefix and that the number goes on
    means = []
    for d in dists:
        z = sum(d.digits)
        means.append(sum(i * p for i, p in enumerate(d.digits)) / z)
    tail = [0.0] * (n + 1)  # tail[j] = expected value of positions j.. (as their own place values)
    for j in range(n - 1, -1, -1):
        tail[j] = 10 ** (n - 1 - j) * means[j] + tail[j + 1]

    points: dict[float, float] = {}

    def add(value: float, prob: float) -> None:
        if prob > 0:
            value = min(float(BOX_GRID), value)
            points[value] = points.get(value, 0.0) + prob

    reach = 1.0
    prefix = 0
    for j in range(n):
        d = dists[j]
        for dd in range(10):
            if dd != greedy_digits[j] and d.digits[dd] > 0:
                v = branch_values.get((j, dd))
                if v is None:
                    v = (prefix * 10 + dd) * 10 ** (n - j - 1) + tail[j + 1]
                add(v, reach * d.digits[dd])
        if j > 0:
            add(prefix, reach * d.term)
        reach *= d.digits[greedy_digits[j]]
        prefix = prefix * 10 + greedy_digits[j]
    p_greedy = reach * final.term
    add(prefix, reach * final.term)
    for dd in range(10):
        v = branch_values.get((n, dd))
        if v is None:
            v = prefix * 10 + dd
        add(v, reach * final.digits[dd])

    z = sum(points.values())
    items = sorted((v, p / z) for v, p in points.items())
    expected = sum(v * p for v, p in items)
    p_within = sum(p for v, p in items if abs(v - expected) <= CONFIDENCE_WINDOW * BOX_GRID + 1e-9)
    return EdgeEstimate(
        greedy=prefix,
        expected=expected,
        p_within=min(1.0, p_within),
        p_greedy=p_greedy / z,
        n_digits=n,
        support=tuple(items),
    )


# ---------------------------------------------------------------------------------------------------------------
# parsing a generation


def _normalise_steps(steps: Iterable) -> list[RawStep]:
    out = []
    for s in steps:
        chosen, top = s
        out.append((chosen, [(t, float(lp)) for t, lp in top]))
    return out


def parse_coordinates(
    steps: Sequence[RawStep], prefix_in_steps: bool = True, with_offsets: bool = False
) -> (
    tuple[list[tuple[list[RawStep], RawStep | None]], int]
    | tuple[list[tuple[list[RawStep], RawStep | None]], int, list[list[int]]]
):
    """Locate the first `box_2d` list and split it into four (digit steps, terminator step) pairs.

    Returns the pairs and the index of the step after the last one used. With `prefix_in_steps` the steps include
    the text up to the list's "[" (the list start is found in the text of the chosen tokens); without it the steps
    are the tokens after a forced BOX_REPLY_PREFIX and the numbers start at the first step.

    `with_offsets`: also return, per coordinate, the index in `steps` of each of its digits (`offsets[ci][j]` is
    where `digit_steps[j]` sits in `steps`; the coordinate's terminator step, when it has one, sits right after
    the last of them). What a branch's continuation (`expand_branches`) replays from: the chosen tokens of
    `steps` up to a branch's own position, then the branch's digit instead of the greedy one.
    """
    text = "".join(c for c, _ in steps)
    flat = text.replace("▁", " ")
    start = 0
    if prefix_in_steps:
        at = flat.find("box_2d")
        if at < 0:
            raise BoxParseError("no box_2d in the generation", flat)
        lb = flat.find("[", at)
        between = flat[at + len("box_2d") : lb if lb >= 0 else len(flat)].lstrip('" ' + _WS)
        if between.startswith(":") and _starts_null(between[1:]):
            raise BoxAbsentSignal(flat)  # `{"box_2d": null`
        if lb < 0:
            raise BoxParseError("no list after box_2d", flat)
        # the first step that starts after the "["
        pos = 0
        start = len(steps)
        for i, (c, _) in enumerate(steps):
            if pos >= lb + 1:
                start = i
                break
            pos += len(c)
    pairs: list[tuple[list[RawStep], RawStep | None]] = []
    offsets: list[list[int]] = []
    cur: list[RawStep] = []
    cur_off: list[int] = []
    i = start
    while i < len(steps) and len(pairs) < 4:
        step = steps[i]
        if not pairs and not cur and _starts_null("".join(c for c, _ in steps[i:]).replace("▁", " ")):
            raise BoxAbsentSignal(flat)  # `[null`: no coordinate before it
        k = _kind(step[0])
        if k == "ws":
            if cur:
                raise BoxParseError("whitespace inside a number", flat)
        elif k == "digit":
            cur.append(step)
            cur_off.append(i)
            if len(cur) > MAX_DIGITS:
                raise BoxParseError("a number with more than four digits", flat)
        elif k == "term":
            if not cur:
                raise BoxParseError("a number is missing from the box_2d list", flat)
            pairs.append((cur, step))
            offsets.append(cur_off)
            cur, cur_off = [], []
        else:
            raise BoxParseError(f"unexpected token {step[0]!r} in the box_2d list", flat)
        i += 1
    if cur and len(pairs) == 3:
        pairs.append((cur, None))  # generation stopped right after the last digit
        offsets.append(cur_off)
        cur, cur_off = [], []
    if len(pairs) < 4:
        raise BoxParseError(f"box_2d has {len(pairs)} numbers, 4 needed", flat)
    return (pairs, i, offsets) if with_offsets else (pairs, i)


def branch_step_index(offsets_ci: Sequence[int], position: int) -> int:
    """Where a branch at `position` sits in the generation's steps: one of its coordinate's own digits
    (`offsets_ci[position]`), or, at `position == len(offsets_ci)`, the terminator step right after the last of
    them (an alternative read of it as one more digit instead of the number ending)."""
    return offsets_ci[position] if position < len(offsets_ci) else offsets_ci[-1] + 1


@dataclass(frozen=True)
class Box:
    top: float
    bottom: float
    left: float
    right: float


@dataclass(frozen=True)
class BoxReadout:
    box: Box  # fractions of the image, clipped and ordered, not rounded
    edges: dict[str, EdgeEstimate]  # y_min, x_min, y_max, x_max as generated, on the 0-1000 grid
    raw_confidence: float
    flags: tuple[str, ...]  # "clipped", "swapped_x", "swapped_y", "degenerate_x", "degenerate_y"
    diagnostics: dict

    def summary(self) -> dict:
        """JSON-friendly per-edge summary (no full supports)."""
        return {
            k: {
                "greedy": e.greedy,
                "expected": round(e.expected, 3),
                "p_within": round(e.p_within, 4),
                "p_greedy": round(e.p_greedy, 4),
            }
            for k, e in self.edges.items()
        }


EDGE_NAMES = ("y_min", "x_min", "y_max", "x_max")


def assemble_box(est: Sequence[EdgeEstimate]) -> BoxReadout:
    """The box, its flags and its raw confidence from four already-estimated edges (y_min, x_min, y_max, x_max):
    what `read_box` and a branch-expanded read (`expand_branches`) both finish with. `diagnostics` is empty; the
    caller fills in what it has (the generation's text and token counts)."""
    edges = dict(zip(EDGE_NAMES, est, strict=True))
    flags: list[str] = []

    vals = [e.expected for e in est]
    if any(v < 0 or v > BOX_GRID for v in vals):
        flags.append("clipped")
    y0, x0, y1, x1 = (min(max(v, 0.0), float(BOX_GRID)) / BOX_GRID for v in vals)
    if y0 > y1:
        y0, y1 = y1, y0
        flags.append("swapped_y")
    if x0 > x1:
        x0, x1 = x1, x0
        flags.append("swapped_x")

    def widen(lo: float, hi: float, name: str) -> tuple[float, float]:
        if hi - lo < MIN_BOX_SIDE:
            flags.append(f"degenerate_{name}")
            if lo + MIN_BOX_SIDE <= 1.0:
                return lo, lo + MIN_BOX_SIDE
            return 1.0 - MIN_BOX_SIDE, 1.0
        return lo, hi

    y0, y1 = widen(y0, y1, "y")
    x0, x1 = widen(x0, x1, "x")
    conf = math.prod(max(e.p_within, 0.0) for e in est) ** 0.25
    return BoxReadout(
        box=Box(top=y0, bottom=y1, left=x0, right=x1),
        edges=edges,
        raw_confidence=conf,
        flags=tuple(flags),
        diagnostics={},
    )


def read_box(steps: Iterable, prefix_in_steps: bool = True) -> BoxReadout | BoxAbsent:
    """The box and raw confidence of one generation, by the plain estimator (no branch expansion). Raises
    BoxParseError when there is no readable box; returns a BoxAbsent when the model wrote `null` for it.

    `prefix_in_steps=False` reads steps that start after the forced reply prefix, at the first digit.
    """
    steps = _normalise_steps(steps)
    try:
        pairs, used = parse_coordinates(steps, prefix_in_steps)
    except BoxAbsentSignal as a:
        return BoxAbsent(a.text, len(steps))
    est = [estimate_coordinate(d, t) for d, t in pairs]
    rd = assemble_box(est)
    rd.diagnostics.update(
        steps=len(steps),
        steps_used=used,
        digits=[e.n_digits for e in est],
        text="".join(c for c, _ in steps).replace("▁", " "),
    )
    return rd


# ---------------------------------------------------------------------------------------------------------------
# branch expansion: real continuations off the greedy path, for the estimator's larger errors


@dataclass(frozen=True)
class BranchCandidate:
    """One alternative digit off a Box's greedy path, worth a real decode: `coordinate` (0: y_min, 1: x_min,
    2: y_max, 3: x_max), `position` (one of its digits, or `len(digit_steps)` for the terminator step reading
    one more digit instead), the alternative `digit` and its probability under the pruned distribution there."""

    coordinate: int
    position: int
    digit: int
    prob: float


def coordinate_branches(
    digit_steps: Sequence[RawStep], term_step: RawStep | None, min_prob: float = 0.05
) -> list[tuple[int, int, float]]:
    """`(position, digit, probability)` of every digit off one coordinate's greedy path with probability at least
    `min_prob`, highest first. A terminator branch (the number ending earlier) needs no decode, is already exact
    in `estimate_coordinate`, and is not returned here."""
    n = len(digit_steps)
    dists = [_position_dist(s, j) for j, s in enumerate(digit_steps)]
    greedy_digits = [_digit_value(s[0]) for s in digit_steps]
    out = []
    for j in range(n):
        d = dists[j]
        for dd in range(10):
            if dd != greedy_digits[j] and d.digits[dd] >= min_prob:
                out.append((j, dd, d.digits[dd]))
    final = _position_dist(term_step, n) if term_step is not None else _Dist((0.0,) * 10, 1.0)
    for dd in range(10):
        if final.digits[dd] >= min_prob:
            out.append((n, dd, final.digits[dd]))
    return sorted(out, key=lambda t: -t[2])


def box_branches(
    pairs: Sequence[tuple[Sequence[RawStep], RawStep | None]], min_prob: float = 0.05
) -> list[BranchCandidate]:
    """Every branch of a Box's four coordinates worth a real decode, highest probability first."""
    out = [
        BranchCandidate(ci, j, dd, p)
        for ci, (ds, ts) in enumerate(pairs)
        for j, dd, p in coordinate_branches(ds, ts, min_prob)
    ]
    return sorted(out, key=lambda b: -b.prob)


def branch_prefix_digits(
    pairs: Sequence[tuple[Sequence[RawStep], RawStep | None]], branch: BranchCandidate
) -> tuple[int, ...]:
    """The digits a branch's continuation completes: its coordinate's greedy digits up to `branch.position`, then
    `branch.digit` itself."""
    digit_steps, _ = pairs[branch.coordinate]
    greedy = tuple(_digit_value(s[0]) for s in digit_steps[: branch.position])
    return greedy + (branch.digit,)


def branch_value(prefix_digits: Sequence[int], continuation: Iterable) -> int:
    """The coordinate value of one decoded branch: `prefix_digits`, then more digits from `continuation`'s own
    chosen tokens (a greedy decode; its alternatives are not read) until a terminator token, or its end."""
    digits = list(prefix_digits)
    for step in continuation:
        tok = step[0] if isinstance(step, (tuple, list)) else step
        k = _kind(tok)
        if k == "term":
            break
        if k == "ws":
            continue
        if k != "digit" or len(digits) >= MAX_DIGITS:
            raise BoxParseError(f"a branch's continuation is not a number: {tok!r}")
        digits.append(_digit_value(tok))
    return min(int("".join(map(str, digits))), BOX_GRID)


def expand_branches(
    pairs: Sequence[tuple[Sequence[RawStep], RawStep | None]],
    decode: Callable[[Sequence[BranchCandidate]], dict[BranchCandidate, float]],
    min_prob: float = 0.05,
    max_branches: int = 16,
) -> tuple[list[EdgeEstimate], int, bool]:
    """The four coordinates' estimates with their highest-probability branches decoded exactly in place of the
    plain estimator's greedy-tail approximation (the fix for its larger errors, a digit-count ambiguity: a `9,`
    read as the end against a `95,` that keeps going).

    `decode(branches)` resolves a batch of branches (an engine call, real generations) to their coordinate values,
    keyed by the branch; at most `max_branches` are asked for, by probability, over the whole box (not per
    coordinate). Returns (the four estimates, how many branches were decoded, whether more branches had enough
    probability than the cap allowed, so the rest fell back to the approximation)."""
    cands = box_branches(pairs, min_prob)
    capped = len(cands) > max_branches
    chosen = cands[:max_branches]
    values = decode(chosen) if chosen else {}
    by_coord: dict[int, dict[tuple[int, int], float]] = {}
    for b in chosen:
        if b in values:
            by_coord.setdefault(b.coordinate, {})[(b.position, b.digit)] = float(values[b])
    est = [estimate_coordinate(ds, ts, by_coord.get(ci)) for ci, (ds, ts) in enumerate(pairs)]
    expanded = sum(len(v) for v in by_coord.values())
    return est, expanded, capped


def wire_box(box: Box) -> dict[str, float]:
    """The box as the answer carries it: two decimals, with left < right and top < bottom kept strict."""

    def pair(lo: float, hi: float) -> tuple[float, float]:
        lo, hi = round(lo, 2), round(hi, 2)
        if hi <= lo:
            if hi + 0.01 <= 1.0:
                hi = round(lo + 0.01, 2)
            else:
                lo = round(hi - 0.01, 2)
        return lo, hi

    top, bottom = pair(box.top, box.bottom)
    left, right = pair(box.left, box.right)
    return {"top": top, "bottom": bottom, "left": left, "right": right}


# ---------------------------------------------------------------------------------------------------------------
# exact reference (for tests and offline checks)

# conditional next-token distribution given the digits so far: (10 digit probabilities, probability of ending)
NextDist = Callable[[tuple[int, ...]], tuple[Sequence[float], float]]


def step_distribution(step: RawStep, n_before: int, min_prob: float = PRUNE_BELOW) -> tuple[tuple[float, ...], float]:
    """The digit probabilities and the end probability at one generation step, pruned at `min_prob`: a public
    entry point to `_position_dist`, for a caller outside this module that has decoded a fresh step (a Box
    engine's `exact_coordinate_tree` decode)."""
    d = _position_dist(step, n_before, prune_below=min_prob)
    return d.digits, d.term


def exact_coordinate_distribution(next_dist: NextDist) -> dict[int, float]:
    """The full digit tree: probability of every value 0..1000 under `next_dist`, no pruning, no approximation.

    `next_dist(prefix)` gives the digit probabilities and the end probability for a prefix of digits; they are
    renormalised here, the end is not allowed on an empty prefix and a digit is not allowed after four.
    """
    out: dict[int, float] = {}

    def rec(prefix: tuple[int, ...], prob: float) -> None:
        digits, end = next_dist(prefix)
        digits = list(digits)
        if len(prefix) >= MAX_DIGITS:
            digits = [0.0] * 10
        if not prefix:
            end = 0.0
        z = sum(digits) + end
        value = int("".join(map(str, prefix))) if prefix else 0
        if end > 0:
            v = min(value, BOX_GRID)
            out[v] = out.get(v, 0.0) + prob * end / z
        for d, p in enumerate(digits):
            if p > 0:
                rec(prefix + (d,), prob * p / z)

    rec((), 1.0)
    return out


DecodeStep = Callable[[tuple[int, ...]], tuple[Sequence[float], float]]


def exact_coordinate_tree(
    digit_steps: Sequence[RawStep],
    term_step: RawStep | None,
    decode: DecodeStep,
    min_prob: float = 1e-4,
    max_decodes: int = 64,
) -> tuple[float, float, int]:
    """The debug readout's exact expected value of one coordinate, from a real recursive expansion of its digit
    tree: along the greedy path the digit and end probabilities are already known (the generation's own top-k,
    read at `min_prob` rather than the served estimator's coarser prune), so no new call is made there. Off the
    greedy path, `decode(prefix)` is a real decode (one short extra generation): the digit probabilities and the
    end probability that follow that exact prefix. It is called once per prefix visited, only for a prefix
    reached with at least `min_prob` of the coordinate's total probability, and at most `max_decodes` times in
    all (a prefix beyond the cap is left out).

    Returns (the expected value over the mass actually covered, that covered mass, how many real decodes it
    took). The covered mass is 1.0 exactly when nothing was left out; `box()`'s debug readout reports it so a
    reader can tell a true exact value from one the cap cut short."""
    n = len(digit_steps)
    greedy = tuple(_digit_value(s[0]) for s in digit_steps)

    def known(prefix: tuple[int, ...]) -> tuple[Sequence[float], float] | None:
        if prefix != greedy[: len(prefix)]:
            return None
        if len(prefix) < n:
            d = _position_dist(digit_steps[len(prefix)], len(prefix), prune_below=min_prob)
            return d.digits, d.term
        d = _position_dist(term_step, n, prune_below=min_prob) if term_step is not None else _Dist((0.0,) * 10, 1.0)
        return d.digits, d.term

    points: dict[int, float] = {}
    decodes = 0

    def rec(prefix: tuple[int, ...], prob: float) -> None:
        nonlocal decodes
        got = known(prefix)
        if got is None:
            if decodes >= max_decodes:
                return
            decodes += 1
            got = decode(prefix)
        digits = [float(x) for x in got[0]]
        end = float(got[1])
        if len(prefix) >= MAX_DIGITS:
            digits = [0.0] * 10
        if not prefix:
            end = 0.0
        z = sum(digits) + end
        if z <= 0:
            return
        value = int("".join(map(str, prefix))) if prefix else 0
        if end > 0:
            points[min(value, BOX_GRID)] = points.get(min(value, BOX_GRID), 0.0) + prob * end / z
        for d, p in enumerate(digits):
            if p <= 0:
                continue
            pp = prob * p / z
            if pp >= min_prob:
                rec(prefix + (d,), pp)

    rec((), 1.0)
    covered = min(1.0, sum(points.values()))
    greedy_value = min(int("".join(map(str, greedy))) if greedy else 0, BOX_GRID)
    expected = sum(v * p for v, p in points.items()) / covered if covered > 0 else float(greedy_value)
    return expected, covered, decodes


# ---------------------------------------------------------------------------------------------------------------
# confidence calibration


@dataclass(frozen=True)
class ConfidenceMap:
    """Monotone map from raw confidence to P(IoU >= 0.5): knots (raw, calibrated), linear between them."""

    knots: tuple[tuple[float, float], ...] = ()

    @property
    def calibrated(self) -> bool:
        return bool(self.knots)

    def apply(self, raw: float) -> float:
        raw = min(max(float(raw), 0.0), 1.0)
        k = self.knots
        if not k:
            return raw
        if raw <= k[0][0]:
            return k[0][1]
        if raw >= k[-1][0]:
            return k[-1][1]
        for (x0, y0), (x1, y1) in zip(k, k[1:], strict=False):
            if x0 <= raw <= x1:
                if x1 == x0:
                    return y1
                return y0 + (y1 - y0) * (raw - x0) / (x1 - x0)
        return raw  # unreachable for sorted knots

    def to_json(self) -> str:
        return json.dumps({"knots": [list(p) for p in self.knots]})

    @classmethod
    def from_json(cls, text: str) -> ConfidenceMap:
        data = json.loads(text)
        raw = data["knots"] if isinstance(data, dict) else data
        knots = tuple((float(a), float(b)) for a, b in raw)
        for (x0, y0), (x1, y1) in zip(knots, knots[1:], strict=False):
            if x1 < x0 or y1 < y0:
                raise ValueError("confidence map knots must be non-decreasing in both coordinates")
        return cls(knots)


def load_confidence_map(path: str | Path | None) -> ConfidenceMap:
    """The map stored at `path` (a file, or a model folder holding BOX_CONFIDENCE_FILE); identity when none exists."""
    if path is None:
        return ConfidenceMap()
    p = Path(path)
    if p.is_dir():
        p = p / BOX_CONFIDENCE_FILE
    if not p.is_file():
        return ConfidenceMap()
    return ConfidenceMap.from_json(p.read_text())


def fit_confidence_map(raw: Sequence[float], hit: Sequence[float]) -> ConfidenceMap:
    """Isotonic fit (pool adjacent violators) of the hit rate (IoU >= 0.5, 0 or 1) on the raw confidence."""
    if len(raw) != len(hit):
        raise ValueError("raw and hit differ in length")
    if not raw:
        return ConfidenceMap()
    groups: dict[float, list[float]] = {}
    for r, h in zip(raw, hit, strict=True):
        groups.setdefault(float(r), []).append(float(h))
    # blocks: [sum of hits, weight, min raw, max raw]
    blocks: list[list[float]] = []
    for r in sorted(groups):
        blocks.append([sum(groups[r]), float(len(groups[r])), r, r])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop()
            blocks[-1][0] += b[0]
            blocks[-1][1] += b[1]
            blocks[-1][3] = b[3]
    knots: list[tuple[float, float]] = []
    for s, w, lo, hi in blocks:
        v = s / w
        knots.append((lo, v))
        if hi > lo:
            knots.append((hi, v))
    return ConfidenceMap(tuple(knots))


def box_confidence(p_presence: float, raw: float, cmap: ConfidenceMap | None = None) -> tuple[float, bool]:
    """The confidence the answer carries and whether it is calibrated: the presence probability times the raw edge
    confidence, through the fitted map when there is one (isotonic on the product, fitted against IoU >= 0.5 on
    present and absent items alike, so that an absent target pulls it down), else the product itself."""
    x = min(max(float(p_presence), 0.0), 1.0) * min(max(float(raw), 0.0), 1.0)
    cmap = cmap or ConfidenceMap()
    return min(max(cmap.apply(x), 0.0), 1.0), cmap.calibrated
