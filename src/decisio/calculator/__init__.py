# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Exact arithmetic support for decisions over numbers: the calculator block.

A decision about a bill, a payslip or a freight charge often turns on a line value that a model has to work out in
its head: 89 minutes at $41.00 an hour, 980 lb priced per kg, tax on a base a credit came off. The calculator
reads the arithmetic a state states, computes every line value exactly (`fractions.Fraction`, formatted only at the
end) and appends a fixed-format block after the state, which the model reads in the same forward pass:

  Calculator (exact values, shown to six decimals):
  - Loading time: 89 min x $41.00/h = $60.816667
  - Cargo weight: 980 lb = 444.528 kg; 444.528 x $0.35 = $155.5848
  - Haul distance: 212 mi = 341.1716 km; 341.1716 x $1.10 = $375.28876
  - Cold storage: 2 x $14.25 x 7 days = $199.50
  - Missed slot: $40.00 x 2 days = $80.00 (taken off)
  - Handling fee: 2 x $8.50 = $17.00
  - Claims credit: $25.00 (taken off)
  - Taxed base: $703.190227
  - Tax charged: $105.478534

It never writes the final total, a verdict, a comparison with a claim or an option. It is all or nothing: if any part
of the arithmetic is not understood with certainty, there is no block at all, and the reason is one short code
(`REASONS`). A missing block is the safe failure; a wrong number is the one failure that is not allowed.

  annotate(state, context=None, version="calc-v1") -> Annotation | Abstain
      read the rendered state (`decisio.readout.letters.fmt_state`), with the task's context before it if there is
      one: rules may sit in the context and amounts in the state; `normalized=True` (keyword) reads the text through
      `decisio.calculator.normalize` first (decimal commas, marks after amounts, numbers in words; off by default);
      `reader="cascade"` (keyword, with `tagger=` a loaded tagger or a model directory, loaded once, and
      `thresholds=`) reads through `decisio.calculator.cascade` instead: the same block bytes, with the path it took
      in `.path` (`rules_agree`, `stated_figures`, `threshold`, `rules_only`, `abstain`), which takes no part in
      equality; its abstain reasons may be `cascade-<code>`. The default reader is `rules`, unchanged.
  from_input(data, version="calc-v1") -> Annotation
      the same block from structured amounts a client already has; raises CalculatorInputError naming the field
      (the schema is in `decisio.calculator.inputs`)
  attach(rendered_state, annotation) -> str
      the state as the model reads it: the rendered state, a newline, the block

Format `calc-v1` (one heading line, then `- ` lines; the order is the priced items as the state lists them, the
percentage adjustments, the fees, the credits, `Taxed base`, then `Tax charged` or `Tax withheld`):

  `Attachments: 4 x $299.46 = $1,197.84`                       a quantity at a unit price
  `Hire charge: 1 x $52.81 x 6 days = $316.86`                 a quantity at a price per day
  `Operator time: 776 min x $92.74/h = $1,199.437333`          time at a rate, after any stated rounding
  `Cut pieces: 9 m x $42.94/m = $386.46`                       lengths at a rate, after any stated rounding
  `Goods weight: 1605 lb = 728.028 kg; 728.028 x $0.20 = $145.6056`   a conversion with the factor the text states
  `Late delivery: $54.35 x 5 days = $271.75 (taken off)`       a credit per day
  `Volume discount: 12% of Linehaul = -$130.6212`              a percentage of one item, before tax
  `Handling fee: 7 x $11.31 = $79.17`                          a fee, counted
  `Account credit: $63.15 (taken off)`                         a credit

Money is written in the state's currency with thousands separators and two to six decimals; quantities with up to
six decimals and no trailing zeros. A value with more decimals is rounded at the sixth, half to even. Any byte change to
this format is a new version.

  python -m decisio.calculator render IN.jsonl.gz --out OUT.jsonl.gz [--field state]

adds the block to every row's state exactly as serving does (a row the calculator abstains on is left as it is) and
prints the counts per reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from decisio.calculator import compute, parse, render
from decisio.calculator.parse import REASONS, Unsure

VERSIONS = ("calc-v1",)
HEADING = render.HEADING

__all__ = [
    "HEADING",
    "REASONS",
    "VERSIONS",
    "Abstain",
    "Annotation",
    "CalculatorInputError",
    "annotate",
    "attach",
    "from_input",
]


@dataclass(frozen=True)
class Annotation:
    version: str
    text: str  # the block: the heading line, then the "- " lines, joined by "\n"; no leading or trailing newline
    lines: int  # the number of "- " lines
    path: str = field(default="", compare=False)  # the cascade reader's path; "" from the rule reader and from_input


@dataclass(frozen=True)
class Abstain:
    reason: str  # one of REASONS (the cascade reader: also `cascade-<code>`)
    path: str = field(default="", compare=False)


class CalculatorInputError(ValueError):
    """A structured input that is invalid or inconsistent; the message names the field."""


def _check_version(version: str) -> None:
    if version not in VERSIONS:
        raise ValueError(f"unknown calculator version {version!r}; this decisio implements {', '.join(VERSIONS)}")


def _annotation(case: compute.Case, version: str) -> Annotation:
    lines = compute.lines(case)
    return Annotation(version=version, text=render.block(lines), lines=len(lines))


def annotate(
    state,
    context: str | None = None,
    version: str = "calc-v1",
    *,
    normalized: bool | None = None,
    reader: str = "rules",
    tagger=None,
    thresholds=None,
) -> Annotation | Abstain:
    """The block for a state (text or a dict, rendered as the prompt renders it), or why there is none. `normalized`
    reads the text through `decisio.calculator.normalize` first (default: `parse.NORMALIZED`, off). `reader="cascade"`
    reads through the cascade (`decisio.calculator.cascade.read`) with `tagger` (a `Tagger`, or a model directory or
    pinned Hub id, loaded once and cached) and `thresholds` (a `cascade.Thresholds`, a dict of its fields, or None:
    the threshold rule off); the result carries its path in `.path`."""
    if reader not in ("rules", "cascade"):
        raise ValueError(f"unknown calculator reader {reader!r}; use 'rules' or 'cascade'")
    rules = _rules(state, context, version, normalized)
    if reader == "rules":
        return rules
    from decisio.calculator import cascade

    tg = cascade.tagger_for(tagger)
    th = cascade.thresholds_for(thresholds)
    try:
        out = cascade.read(state, context, rules_case=rules, tagger=tg, thresholds=th, version=version)
    except Exception:  # never a partial or a guessed block, and never a failed request
        return Abstain("internal-error", path="abstain")
    if isinstance(out, Annotation):
        return Annotation(out.version, out.text, out.lines, path=out.path)
    return Abstain(out.reason, path=out.path)


def _rules(state, context: str | None, version: str, normalized: bool | None) -> Annotation | Abstain:
    _check_version(version)
    from decisio.readout.letters import fmt_state

    parts = ([context] if context else []) + [fmt_state(state)]
    try:
        case = parse.parse(parts) if normalized is None else parse.parse(parts, normalized=normalized)
        return _annotation(case, version)
    except Unsure as e:
        return Abstain(e.reason)
    except Exception:  # never a partial or a guessed block, and never a failed request
        return Abstain("internal-error")


def from_input(data: dict, version: str = "calc-v1") -> Annotation:
    """The block for a client's structured amounts (`decisio.calculator.inputs`); raises CalculatorInputError."""
    _check_version(version)
    from decisio.calculator.inputs import case_from_input

    return _annotation(case_from_input(data, CalculatorInputError), version)


def attach(rendered_state: str, ann: Annotation) -> str:
    """The state the model reads: the rendered state, a newline, the block."""
    return f"{rendered_state}\n{ann.text}"
