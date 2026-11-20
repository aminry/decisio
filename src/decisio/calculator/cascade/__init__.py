# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The cascade reader: a role tagger, the rule reader and acceptance gates, for states the rule reader cannot read.

Not wired into serving: no server setting reads through it yet, and nothing here changes the calculator's default.

  read(state, context=None, rules_case=None, *, tagger, thresholds=Thresholds()) -> Shown | Refused

1. The tagger (`decisio.calculator.cascade.tagger`) marks the spans of the text with roles (a quantity, a price, a
   rate, a tax rate, a distractor...), groups them into lines and sets the flags (tax charged or withheld, what sits
   after tax, rounding modes), each decision with a probability.
2. The assembler (`assembler.assemble`) turns the tags into a `calculator_input`; `from_input` renders the block, so
   values and bytes are the ones the rule reader and the structured input give (format `calc-v1`, unchanged).
3. The gates (`gates`) decide: provenance, every number read, units, stated figures; then the rule reader gives the
   same block (`rules_agree`), or the stated figures reproduce (`stated_figures`), or every decision clears its
   threshold outside the margin band (`threshold`). Anything else is no block.

When the tagger's tags give no case but the rule reader reads the state, the rule reader's block stands (`rules_only`:
the cascade never shows less than the rule reader); when both give a case and they differ, there is no block.

`Shown` is an `Annotation` and `Refused` an `Abstain` (`reason` is `cascade-<code>`), each with the `path` it took,
for the report. Optional dependencies: the tagger's model needs `torch` and `transformers`, imported only when a model
is loaded; the stub tagger, the assembler and the gates need nothing beyond the standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from decisio.calculator import Abstain, Annotation
from decisio.calculator import normalize as N
from decisio.calculator.cascade import gates
from decisio.calculator.cascade.assembler import REFUSALS, Assembly, Refusal, assemble
from decisio.calculator.cascade.gates import Check, Thresholds
from decisio.calculator.cascade.tagger import StubTagger, Tagger
from decisio.calculator.cascade.tags import Tags, TagSpan

__all__ = [
    "PATHS",
    "REFUSALS",
    "Assembly",
    "Check",
    "Refusal",
    "Refused",
    "Shown",
    "StubTagger",
    "TagSpan",
    "Tagger",
    "Tags",
    "Thresholds",
    "assemble",
    "read",
    "tagger_for",
    "thresholds_for",
]

PATHS = ("rules_agree", "stated_figures", "threshold", "rules_only", "abstain")


@dataclass(frozen=True)
class Shown(Annotation):
    checks: tuple[Check, ...] = ()


@dataclass(frozen=True)
class Refused(Abstain):
    path: str = field(default="abstain", compare=False)
    gate: str = ""  # the gate or step that refused: tagger, assemble, provenance, numbers, units, stated_figures,
    # readers, accept
    detail: str = ""
    checks: tuple[Check, ...] = ()


@lru_cache(maxsize=4)
def _loaded(path: str, revision: str | None):
    from decisio.calculator.cascade.tagger import HFTagger

    return HFTagger.load(path, revision)


def tagger_for(tagger, revision: str | None = None) -> Tagger:
    """A tagger as given, or the model at a directory (or a Hub id at a pinned `revision`), loaded once per process."""
    if tagger is None:
        raise ValueError("the cascade reader needs a tagger: a Tagger, or a model directory")
    if isinstance(tagger, (str, Path)):
        return _loaded(str(tagger), revision)
    if not isinstance(tagger, Tagger):
        raise TypeError("tagger must have predict(state, context) -> Tags, or be a model directory")
    return tagger


def thresholds_for(thresholds) -> Thresholds:
    """Thresholds as given, from a dict of their fields, or None: the threshold rule off."""
    if thresholds is None:
        return Thresholds()
    if isinstance(thresholds, Thresholds):
        return thresholds
    return Thresholds(**{k: thresholds[k] for k in ("role", "link", "flag", "band") if k in thresholds})


# what every block the cascade shows needs in its text: an amount of money (a price or rate is read only from a value
# with a currency mark: `$`, `€`, `£`, `¥`, a currency code, or dollars, euros, pounds) and a tax rate (a percentage:
# `%`, percent, per cent); the assembler refuses a case without either. A wider, case-blind match of the normalizer's
# own marks, so the pre-gate skips only texts on which the tagger could not lead to a block.
_MONEY_MARK = re.compile(r"[$€£¥]|\b(?:" + "|".join(N.CODES) + r")\b|\b(?:dollars?|euros?|pounds?)\b", re.I)
_RATE_MARK = re.compile(r"%|\bper ?cent\b", re.I)


def may_hold_case(text: str, context: str | None = None) -> bool:
    """False when the text (with its context) has no currency mark or no percentage: no cascade block is possible, so
    the tagger is not run."""
    if "pregate" in _DISABLED:
        return True
    whole = (context or "") + "\n" + text
    return bool(_MONEY_MARK.search(whole)) and bool(_RATE_MARK.search(whole))


# switched off by name for audits ("pregate": the tagger runs on every text); empty in serving
_DISABLED: set[str] = set()


def _rules_block(state, context, rules_case, version, normalized=None):
    from decisio.calculator import annotate, compute, render

    if rules_case is None:
        rules_case = annotate(state, context, version, normalized=normalized)
    if isinstance(rules_case, Annotation):
        return rules_case
    if isinstance(rules_case, compute.Case):
        lines = compute.lines(rules_case)
        return Annotation(version=version, text=render.block(lines), lines=len(lines))
    return None  # an Abstain: the rule reader has no case


def read(
    state,
    context: str | None = None,
    rules_case=None,
    *,
    tagger: Tagger,
    thresholds: Thresholds = Thresholds(),
    version: str = "calc-v1",
    also=None,
    normalized: bool | None = None,
) -> Shown | Refused:
    """The block for a state through the cascade, or why there is none. `rules_case` is the rule reader's result for
    the same text (an `Annotation`, a `compute.Case` or an `Abstain`); None runs the rule reader here (with
    `normalized`), on the CPU while a GPU tagger's encoder computes. `also` (no arguments) is more CPU work run there
    too, once, whatever the outcome."""
    from decisio.readout.letters import fmt_state

    text = fmt_state(state)
    got: dict = {}

    def run_rules():  # the rule reader (and `also`), on the CPU while a GPU tagger's encoder runs
        got["rules"] = _rules_block(state, context, rules_case, version, normalized)
        if also is not None:
            also()

    if not may_hold_case(text, context):  # no block can come from the tagger: it is not run (about 0 ms)
        run_rules()
        rules = got["rules"]
        if rules is not None:
            return Shown(rules.version, rules.text, rules.lines, path="rules_only")
        return Refused("cascade-no-amount-or-rate", gate="pregate")
    tags, error = None, None
    try:
        if getattr(tagger, "overlaps", False):
            tags = tagger.predict(text, context, during=run_rules)
        else:
            run_rules()
            tags = tagger.predict(text, context)
    except Exception as e:  # a tagger failure is no block from the tagger, never a failed request
        error = e
    if "rules" not in got:
        run_rules()
    rules = got["rules"]

    def rules_only(gate: str, code: str, detail: str = "", checks=()) -> Shown | Refused:
        if rules is not None:
            return Shown(rules.version, rules.text, rules.lines, path="rules_only", checks=tuple(checks))
        return Refused(f"cascade-{code}", gate=gate, detail=detail, checks=tuple(checks))

    if error is not None:
        return rules_only("tagger", "tagger-error", type(error).__name__)
    if tags.note:
        return rules_only("tagger", tags.note)
    if tags.abstain_p >= 0.5:
        return rules_only("tagger", "tagger-abstain")
    try:
        asm = assemble(tags, text, context)
        ann = asm.block(version)
    except Refusal as e:
        return rules_only("assemble", e.code, e.detail)
    except Exception as e:
        return rules_only("assemble", "internal-error", type(e).__name__)

    checks = []
    for check in (
        gates.provenance(asm),
        gates.numbers(tags, text, context),
        gates.units(asm),
        gates.rule_forms(text, context),
        gates.wording(asm),
    ):
        checks.append(check)
        if not check.ok:
            return rules_only(check.gate, check.code, check.detail, checks)
    consistent = gates.consistency(asm)
    checks.append(consistent)
    if not consistent.ok:
        return rules_only(consistent.gate, consistent.code, consistent.detail, checks)
    stated = gates.stated_figures(asm)
    checks.append(stated)
    if stated.code in ("stated-mismatch",):
        return Refused(f"cascade-{stated.code}", gate=stated.gate, detail=stated.detail, checks=tuple(checks))
    if rules is not None:
        agree = gates.agreement(ann.text, rules.text)
        checks.append(agree)
        if not agree.ok:
            return Refused("cascade-readers-disagree", gate="readers", checks=tuple(checks))
        return Shown(ann.version, ann.text, ann.lines, path="rules_agree", checks=tuple(checks))
    if stated.ok:
        return Shown(ann.version, ann.text, ann.lines, path="stated_figures", checks=tuple(checks))
    conf = gates.confidence(tags, thresholds)
    checks.append(conf)
    if conf.ok:
        return Shown(ann.version, ann.text, ann.lines, path="threshold", checks=tuple(checks))
    return Refused(f"cascade-{conf.code}", gate="accept", detail=conf.detail, checks=tuple(checks))
