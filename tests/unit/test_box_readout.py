# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Box readout: parsing, expectation, confidence, calibration map and the wire models."""

from __future__ import annotations

import math
import random

import pytest
from pydantic import ValidationError

from decisio.readout import box as B
from decisio.serve.systemone import BoxAnswer, BoxQuestion

PREFIX_STEPS = [('{"', []), ("box", []), ("_", []), ("2", []), ("d", []), ('":', []), ("▁[", [])]


def lp(p):
    return math.log(p)


def sure(tok):
    return (tok, [(tok, 0.0)])


def num_steps(digits, end=",", sep=True, dist=None):
    """Steps for one number: optional space token, digits (each certain unless dist gives {pos: {tok: p}}), end."""
    out = [sure("▁")] if sep else []
    dist = dist or {}
    for i, d in enumerate(digits):
        if i in dist:
            out.append((str(d), [(t, lp(p)) for t, p in dist[i].items()]))
        else:
            out.append(sure(str(d)))
    if "end" in dist:
        out.append((end, [(t, lp(p)) for t, p in dist["end"].items()]))
    else:
        out.append(sure(end))
    return out


def gen(nums, prefix=True, dists=None, tail=True):
    """A whole generation for four numbers, e.g. [613, 24, 947, 591]."""
    dists = dists or {}
    steps = list(PREFIX_STEPS) if prefix else []
    for i, n in enumerate(nums):
        s = num_steps(str(n), end="]," if i == 3 else ",", sep=i > 0, dist=dists.get(i))
        steps += s
    if tail:
        steps += [sure("▁"), sure('"'), sure("label"), sure('":'), sure("▁"), sure('"'), sure("x"), sure('"}')]
    return steps


# ---------------------------------------------------------------------------------------------- prompt


def test_prompt_pieces():
    assert B.BOX_REPLY_PREFIX == '{"box_2d": ['
    p = B.render_box_prompt("the coffee mug in `photo`.")
    assert p.startswith("Detect the coffee mug in `photo`. Output a single bounding box as JSON")
    assert '{"box_2d": [y_min, x_min, y_max, x_max], "label": "..."}' in p
    assert p.endswith("Output only the JSON.")
    assert B.BOX_DEFAULT_TARGET in B.render_box_prompt(None)
    assert B.render_box_prompt({"b": 1, "a": ["x"]}).startswith('Detect {"a":["x"],"b":1}.')
    assert B.render_box_prompt(["a", "b"]).startswith('Detect ["a","b"].')


# ---------------------------------------------------------------------------------------------- parsing


def test_parse_basic_certain():
    r = B.read_box(gen([613, 24, 947, 591]))
    # y0=613 x0=24 y1=947 x1=591
    assert r.box.top == pytest.approx(0.613)
    assert r.box.left == pytest.approx(0.024)
    assert r.box.bottom == pytest.approx(0.947)
    assert r.box.right == pytest.approx(0.591)
    assert r.raw_confidence == pytest.approx(1.0)
    assert r.flags == ()
    assert r.diagnostics["digits"] == [3, 2, 3, 3]


def test_parse_without_forced_prefix_in_steps_and_with():
    a = B.read_box(gen([10, 20, 30, 40], prefix=True))
    b = B.read_box(gen([10, 20, 30, 40], prefix=False), prefix_in_steps=False)
    assert a.box == b.box and a.raw_confidence == b.raw_confidence
    assert a.box.top == pytest.approx(0.01)
    # steps after a forced prefix have no box_2d text: reading them as if the prefix were there is an error
    with pytest.raises(B.BoxParseError):
        B.read_box(gen([10, 20, 30, 40], prefix=False))


def test_parse_lengths_and_1000():
    r = B.read_box(gen([5, 50, 500, 1000]))
    assert r.diagnostics["digits"] == [1, 2, 3, 4]
    assert r.box.top == pytest.approx(0.005)
    assert r.box.right == pytest.approx(1.0)


def test_parse_leading_zeros():
    r = B.read_box(gen(["007", "050", 400, 900]))
    assert r.diagnostics["digits"] == [3, 3, 3, 3]
    assert r.box.top == pytest.approx(0.007)
    assert r.box.left == pytest.approx(0.05)


def test_parse_newlines_and_spaces():
    steps = list(PREFIX_STEPS) + [sure("\n"), sure("▁▁")]
    for i, n in enumerate(["12", "345", "400", "800"]):
        for d in n:
            steps.append(sure(d))
        steps.append(sure(",") if i < 3 else sure("]}"))
        if i < 3:
            steps += [sure("\n"), sure("▁")]
    r = B.read_box(steps)
    assert r.box.top == pytest.approx(0.012)
    assert r.box.left == pytest.approx(0.345)
    assert r.box.right == pytest.approx(0.8)


def test_extra_boxes_and_label_ignored():
    steps = gen([100, 200, 300, 400]) + [sure(","), sure("▁"), sure('{"')] + gen([900, 900, 950, 950], prefix=False)
    r = B.read_box(steps)
    assert r.box.top == pytest.approx(0.1)
    assert r.box.bottom == pytest.approx(0.3)


def test_generation_stopping_right_after_last_digit():
    steps = gen([100, 200, 300, 400], tail=False)[:-1]  # drop the final terminator
    r = B.read_box(steps)
    assert r.box.right == pytest.approx(0.4)


@pytest.mark.parametrize(
    "steps",
    [
        [sure("Sorry"), sure(","), sure("▁"), sure("no")],
        list(PREFIX_STEPS) + [sure("▁"), sure("1"), sure(","), sure("▁"), sure("2"), sure("]}")],
        list(PREFIX_STEPS) + [sure("null"), sure("]}")],
        list(PREFIX_STEPS) + [sure("▁"), sure("1"), sure("2"), sure("."), sure("5")],
        list(PREFIX_STEPS) + [sure("▁"), sure(","), sure("▁"), sure("1")],
        list(PREFIX_STEPS),
        [],
        list(PREFIX_STEPS) + [sure(d) for d in "12345"] + [sure(",")],
    ],
)
def test_malformed_raises(steps):
    with pytest.raises(B.BoxParseError):
        B.read_box(steps)


def test_no_list_after_box_2d():
    with pytest.raises(B.BoxParseError, match="no list"):
        B.read_box([sure("box_2d"), sure("x")])


# ---------------------------------------------------------------------------------------------- expectation


def test_expectation_last_digit_two_way():
    # 6 1 [3 or 5]: greedy 613 (0.6), alternative 615 (0.4)
    dists = {0: {2: {"3": 0.6, "5": 0.4}}}
    r = B.read_box(gen([613, 24, 947, 591], dists=dists))
    assert r.edges["y_min"].expected == pytest.approx(613 * 0.6 + 615 * 0.4)


def test_expectation_first_digit_uncertain_uses_greedy_tail():
    # first digit 6 (0.7) or 4 (0.3), then 1 and 3 certain: the off-path branch takes the greedy tail (13)
    dists = {0: {0: {"6": 0.7, "4": 0.3}}}
    r = B.read_box(gen([613, 24, 947, 591], dists=dists))
    assert r.edges["y_min"].expected == pytest.approx(0.7 * 613 + 0.3 * 413)


def test_expectation_renormalises_over_digits_and_prunes():
    # the other tokens take 40% of the mass; "5" has 0.5% of the digit mass and is pruned
    dists = {0: {2: {"3": 0.595, "4": 0.4, "5": 0.005, "x": 0.4}}}
    r = B.read_box(gen([613, 24, 947, 591], dists=dists))
    z = 0.595 + 0.4
    assert r.edges["y_min"].expected == pytest.approx(610 + (3 * 0.595 + 4 * 0.4) / z)


def test_terminator_competes_with_a_further_digit():
    # two digits "45" then "," (0.7) or a third digit 6 (0.3): values 45 and 456
    dists = {0: {"end": {",": 0.7, "6": 0.3}}}
    r = B.read_box(gen([45, 24, 947, 591], dists=dists))
    assert r.edges["y_min"].greedy == 45
    assert r.edges["y_min"].expected == pytest.approx(0.7 * 45 + 0.3 * 456)


def test_terminator_after_first_digit_competes_with_second_digit():
    # "4" then "," (0.8) or "5" (0.2) at the second position; the greedy has 4 then 5? greedy is "4,"
    dists = {0: {"end": {",": 0.8, "5": 0.2}}}
    r = B.read_box(gen([4, 24, 947, 591], dists=dists))
    assert r.edges["y_min"].expected == pytest.approx(0.8 * 4 + 0.2 * 45)


def test_terminator_alternatives_all_end_the_number():
    # "],", "]}" and "," all end a number: they pool
    dists = {0: {"end": {",": 0.4, "],": 0.3, "5": 0.3}}}
    r = B.read_box(gen([45, 24, 947, 591], dists=dists))
    assert r.edges["y_min"].p_greedy == pytest.approx(0.7)


def test_early_end_off_the_greedy_path():
    # at the second position of "45": "5" 0.8, "," 0.2 (a number of one digit)
    dists = {0: {1: {"5": 0.8, ",": 0.2}}}
    r = B.read_box(gen([45, 24, 947, 591], dists=dists))
    assert r.edges["y_min"].expected == pytest.approx(0.8 * 45 + 0.2 * 4)


def test_value_over_1000_is_clipped():
    dists = {0: {"end": {",": 0.5, "5": 0.5}}}
    r = B.read_box(gen([1000, 24, 947, 591], dists=dists))
    assert r.edges["y_min"].expected <= 1000.0


# ---------------------------------------------------------------------------------------------- clip, order, degenerate


def test_clip_and_order():
    r = B.read_box(gen([900, 800, 100, 200]))
    assert (r.box.top, r.box.bottom) == pytest.approx((0.1, 0.9))
    assert (r.box.left, r.box.right) == pytest.approx((0.2, 0.8))
    assert set(r.flags) == {"swapped_x", "swapped_y"}
    assert r.box.top < r.box.bottom and r.box.left < r.box.right


def test_degenerate_widened():
    r = B.read_box(gen([300, 500, 300, 500]))
    assert r.box.bottom - r.box.top == pytest.approx(0.001)
    assert r.box.right - r.box.left == pytest.approx(0.001)
    assert set(r.flags) == {"degenerate_x", "degenerate_y"}


def test_degenerate_at_the_far_edge_widens_inward():
    r = B.read_box(gen([1000, 1000, 1000, 1000]))
    assert r.box.bottom == pytest.approx(1.0) and r.box.top == pytest.approx(0.999)
    assert r.box.right == pytest.approx(1.0) and r.box.left == pytest.approx(0.999)


def test_wire_box_two_decimals_and_strict_order():
    r = B.read_box(gen([300, 500, 300, 500]))
    w = B.wire_box(r.box)
    assert w["top"] < w["bottom"] and w["left"] < w["right"]
    assert B.wire_box(B.Box(0.123456, 0.987654, 0.5, 0.5049)) == {
        "top": 0.12,
        "bottom": 0.99,
        "left": 0.5,
        "right": 0.51,
    }
    far = B.wire_box(B.Box(0.999, 1.0, 0.999, 1.0))
    assert far == {"top": 0.99, "bottom": 1.0, "left": 0.99, "right": 1.0}


# ---------------------------------------------------------------------------------------------- confidence


def test_confidence_formula():
    # x1 is split 855 / 983 style: one edge at p_within 0.6 (greedy 100 at 0.6, 500 at 0.4), the others certain
    dists = {0: {0: {"1": 0.6, "5": 0.4}}}
    r = B.read_box(gen([100, 24, 947, 591], dists=dists))
    e = r.edges["y_min"]
    assert e.expected == pytest.approx(0.6 * 100 + 0.4 * 500)
    # |100 - 260| and |500 - 260| are both over 20, so nothing is within the window of the expectation
    assert e.p_within == 0.0
    assert r.raw_confidence == 0.0
    # tight split: 100 vs 110, both within 20 of 106
    dists = {0: {2: {"0": 0.6, "9": 0.4}}}
    r = B.read_box(gen([100, 24, 947, 591], dists=dists))
    assert r.edges["y_min"].p_within == pytest.approx(1.0)
    assert r.raw_confidence == pytest.approx(1.0)


def test_confidence_geometric_mean():
    # two edges at p_within 0.5 and 0.8 by construction (a far alternative outside the 20-unit window of E)
    # E of edge with greedy 100 (p) and 700 (1-p): within window only when one side dominates
    # edge y_min: 100 (0.97) or 700 (0.03), E = 118: only the greedy value is within 20 of it
    # edge x_min: 200 (0.98) or 300 (0.02), E = 202: only the greedy value is within 20 of it
    dists = {0: {0: {"1": 0.97, "7": 0.03}}, 1: {0: {"2": 0.98, "3": 0.02}}}
    r = B.read_box(gen([100, 200, 947, 591], dists=dists))
    assert r.edges["y_min"].p_within == pytest.approx(0.97)
    assert r.edges["x_min"].p_within == pytest.approx(0.98)
    assert r.raw_confidence == pytest.approx((0.97 * 0.98) ** 0.25)
    # an edge whose two values straddle its expectation outside the window scores zero
    dists = {0: {0: {"1": 0.6, "5": 0.4}}}
    assert B.read_box(gen([100, 200, 947, 591], dists=dists)).raw_confidence == 0.0


def test_determinism():
    dists = {0: {2: {"3": 0.6, "5": 0.4}}, 2: {1: {"4": 0.7, "2": 0.3}}}
    a = B.read_box(gen([613, 24, 947, 591], dists=dists))
    b = B.read_box(gen([613, 24, 947, 591], dists=dists))
    assert a == b


def test_summary_is_json_friendly():
    import json

    r = B.read_box(gen([613, 24, 947, 591]))
    json.dumps(r.summary())
    assert set(r.summary()) == set(B.EDGE_NAMES)


# ---------------------------------------------------------------------------------------------- exact reference


def scripted(rng, certain_leading, realistic=False):
    """A random conditional digit-tree distribution, memoised per prefix.

    `realistic`: the number of digits is settled (an early end is rare, below 5%, and the end is certain at the
    target length), as when the model knows the magnitude of a coordinate; otherwise the end is likely anywhere.
    """
    memo = {}
    target_len = rng.randint(1, 4)

    def nd(prefix):
        if prefix in memo:
            return memo[prefix]
        w = [rng.random() ** 3 for _ in range(10)]
        z = sum(w)
        digits = [x / z for x in w]
        end = rng.random() * 0.8 if prefix else 0.0
        if realistic and prefix:
            n = len(prefix)
            end = rng.random() * 0.05 if n < target_len else 1.0
            if n >= target_len:
                digits = [0.0] * 10
        if certain_leading is not None:
            # the leading digits are certain; only the last digit and the end are uncertain
            n = len(prefix)
            if n < certain_leading - 1:
                k = rng.randrange(10)
                digits = [1.0 if i == k else 0.0 for i in range(10)]
                end = 0.0
            elif n == certain_leading - 1:
                end = rng.choice([0.0, 0.3, 0.6]) if n > 0 else 0.0
            else:
                digits = [0.0] * 10
                end = 1.0
        if len(prefix) >= B.MAX_DIGITS:
            digits, end = [0.0] * 10, 1.0
        # drop probabilities below the pruning line, then normalise
        tot = sum(digits) + end
        digits = [d / tot for d in digits]
        end /= tot
        digits = [d if d >= 0.011 else 0.0 for d in digits]
        end = end if end >= 0.011 else 0.0
        tot = sum(digits) + end
        out = ([d / tot for d in digits], end / tot)
        memo[prefix] = out
        return out

    return nd


def steps_along_greedy(nd, noise_tok=True):
    """Digit steps and terminator step along the greedy path of a scripted distribution."""
    prefix = ()
    digit_steps = []
    while True:
        digits, end = nd(prefix)
        if not prefix:
            end = 0.0
        if len(prefix) >= B.MAX_DIGITS:
            digits = [0.0] * 10
        z = sum(digits) + end
        cands = [(str(i), p / z) for i, p in enumerate(digits) if p > 0]
        if end > 0:
            cands.append((",", end / z))
        top = [(t, math.log(p * 0.9)) for t, p in cands] + [("other", math.log(0.1))] if noise_tok else None
        top = top or [(t, math.log(p)) for t, p in cands]
        best = max(cands, key=lambda c: (c[1], c[0]))[0]
        if best == ",":
            return digit_steps, (",", top)
        digit_steps.append((best, top))
        prefix += (int(best),)


def reference_stats(nd):
    dist = B.exact_coordinate_distribution(nd)
    e = sum(v * p for v, p in dist.items())
    pw = sum(p for v, p in dist.items() if abs(v - e) <= 20 + 1e-9)
    return e, pw


def test_exact_tree_sums_to_one_and_known_case():
    def nd(prefix):
        if not prefix:
            return [0, 1] + [0] * 8, 0.0
        if len(prefix) == 1:
            return [0.5, 0.5] + [0] * 8, 0.0
        return [0.0] * 10, 1.0

    dist = B.exact_coordinate_distribution(nd)
    assert dist == {10: pytest.approx(0.5), 11: pytest.approx(0.5)}


def test_estimator_exact_when_leading_digits_certain():
    errs = []
    for seed in range(300):
        rng = random.Random(seed)
        n = rng.randint(1, 4)
        nd = scripted(rng, certain_leading=n)
        ds, ts = steps_along_greedy(nd)
        est = B.estimate_coordinate(ds, ts)
        e, pw = reference_stats(nd)
        errs.append((abs(est.expected - e), abs(est.p_within - pw)))
    assert max(a for a, _ in errs) < 1e-6
    assert max(b for _, b in errs) < 1e-6


def measure(realistic, n=400, seed=2026):
    rng = random.Random(seed)
    abs_err, pw_err = [], []
    for _ in range(n):
        nd = scripted(rng, certain_leading=None, realistic=realistic)
        ds, ts = steps_along_greedy(nd)
        est = B.estimate_coordinate(ds, ts)
        e, pw = reference_stats(nd)
        abs_err.append(abs(est.expected - e))
        pw_err.append(abs(est.p_within - pw))
    return abs_err, pw_err


def test_estimator_error_bounded_when_the_digit_count_is_settled():
    abs_err, pw_err = measure(realistic=True)
    n = len(abs_err)
    print(
        f"estimator vs full tree, {n} settled-length distributions: expected-value error mean "
        f"{sum(abs_err) / n:.2f}, max {max(abs_err):.2f} grid units; p_within error mean {sum(pw_err) / n:.3f}, "
        f"max {max(pw_err):.3f}"
    )
    # measured on this seed: mean 15.9, max 84 grid units (0.016 and 0.084 of the image); the digit distributions
    # here are far flatter than a model's, so this is a loose bound. The error comes from branches that leave the
    # greedy path being valued with the greedy path's own lower digits
    assert max(abs_err) < 110.0
    assert sum(abs_err) / n < 25.0


def test_estimator_error_when_the_digit_count_is_open_is_reported_not_bounded_tightly():
    abs_err, pw_err = measure(realistic=False)
    n = len(abs_err)
    print(
        f"estimator vs full tree, {n} open-length distributions: expected-value error mean {sum(abs_err) / n:.2f}, "
        f"max {max(abs_err):.2f} grid units; p_within error mean {sum(pw_err) / n:.3f}, max {max(pw_err):.3f}"
    )
    # branches that end at a different length than the greedy path change a value by a factor of ten, which the
    # first-order estimate does not see; the error stays under half the grid on this seed (max 532)
    assert max(abs_err) < 600.0


# ---------------------------------------------------------------------------------------------- calibration map


def test_identity_without_a_map(tmp_path):
    m = B.load_confidence_map(None)
    assert not m.calibrated and m.apply(0.37) == 0.37
    assert not B.load_confidence_map(tmp_path).calibrated
    assert not B.load_confidence_map(tmp_path / "missing.json").calibrated


def test_fit_pav_known_case():
    raw = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    hit = [0, 1, 0, 1, 1, 1]
    m = B.fit_confidence_map(raw, hit)
    # 0.2 and 0.3 violate order and pool to 0.5
    assert m.apply(0.1) == 0.0
    assert m.apply(0.2) == pytest.approx(0.5)
    assert m.apply(0.3) == pytest.approx(0.5)
    assert m.apply(0.6) == 1.0
    assert m.apply(0.35) == pytest.approx(0.5 + 0.5 * 0.5)  # halfway between (0.3, 0.5) and (0.4, 1.0)
    ys = [y for _, y in m.knots]
    assert ys == sorted(ys)


def test_fit_ties_and_extremes():
    m = B.fit_confidence_map([0.5, 0.5, 0.5, 0.9], [1, 0, 0, 1])
    assert m.apply(0.5) == pytest.approx(1 / 3)
    assert m.apply(0.0) == pytest.approx(1 / 3)  # clamps below the first knot
    assert m.apply(1.0) == 1.0
    assert not B.fit_confidence_map([], []).calibrated
    with pytest.raises(ValueError):
        B.fit_confidence_map([0.1], [1, 0])


def test_fit_is_monotone_on_random_data():
    rng = random.Random(5)
    raw = [rng.random() for _ in range(500)]
    hit = [1 if rng.random() < r else 0 for r in raw]
    m = B.fit_confidence_map(raw, hit)
    grid = [i / 200 for i in range(201)]
    vals = [m.apply(g) for g in grid]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:], strict=False))
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert m.calibrated


def test_map_round_trip_and_load(tmp_path):
    m = B.fit_confidence_map([0.1, 0.5, 0.9], [0, 1, 1])
    (tmp_path / B.BOX_CONFIDENCE_FILE).write_text(m.to_json())
    assert B.load_confidence_map(tmp_path) == m
    assert B.load_confidence_map(tmp_path / B.BOX_CONFIDENCE_FILE) == m
    with pytest.raises(ValueError):
        B.ConfidenceMap.from_json('{"knots": [[0.1, 0.9], [0.2, 0.1]]}')


# ---------------------------------------------------------------------------------------------- wire models


def test_box_question():
    q = BoxQuestion.model_validate({"type": "box", "instructions": "the mug", "extra": 1})
    assert q.instructions == "the mug"
    assert BoxQuestion.model_validate({"type": "box"}).instructions is None
    assert BoxQuestion.model_validate({"type": "box", "instructions": {"a": 1}}).instructions == {"a": 1}
    assert BoxQuestion.model_validate({"type": "box", "instructions": ["a", {"b": 2}]}).instructions == ["a", {"b": 2}]
    with pytest.raises(ValidationError):
        BoxQuestion.model_validate({"type": "noul"})


def test_box_answer_rounds_to_two_decimals():
    a = BoxAnswer(box={"top": 0.123456, "bottom": 0.987, "left": 0.5, "right": 0.7049}, confidence=0.8123)
    assert a.type == "box"
    assert a.model_dump()["box"] == {"top": 0.12, "bottom": 0.99, "left": 0.5, "right": 0.7}
    assert a.confidence == 0.8123
    with pytest.raises(ValidationError):
        BoxAnswer(box={"top": 0.1}, confidence=0.5)


def test_box_is_not_in_the_served_union_yet():
    from decisio.serve import systemone

    members = (
        {getattr(t, "__name__", "") for t in systemone.Question.__origin__.__args__}
        if hasattr(systemone.Question, "__origin__")
        else set()
    )
    assert "BoxQuestion" not in members
