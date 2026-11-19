# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The acceptance gates: pure functions over an assembled case, each returning a `Check` with a named failure.

A block is shown only when every required gate passes:

  provenance      every operand is a value read verbatim inside a span of the state or the context (re-read here from
                  the source text, independently of the assembler), or derived from such values in one stated way
                  (a duration in minutes, the days of a date range, a step converted with an exact unit factor)
  numbers         every number in the text (each value the normalizer finds, each run of digits) lies in a span: a
                  role, a label or a distractor; nothing is left unread
  units           the units type-check with the normalizer's dimensions: a time rate is per hour or per minute, a
                  conversion's two sides are the same kind of unit and its stated factor is within 2% of the definition
                  when the definition is exact, a length rate is per the lengths' unit, a bare number takes its unit
                  from its own line, a price per a unit matches the quantity's unit
  stated figures  every total, subtotal or tax the text states equals a figure of the case to the cent; a mismatch
                  vetoes the block whatever else passes (never auto-corrected)
  rule forms      no sentence states a rule the calculator cannot express, whatever the tagger made of it: rounding
                  with a direction (up, down, a ceiling or a floor; a tie-break such as "half a cent up" is not one),
                  a minimum or maximum charge, a cap, or "whichever is greater"
  wording         the text's own words agree with the case: no line label is cut out of a longer phrase (such as
                  "days" out of "Daily rate days"); every line taken off (a credit, a day credit, a
                  deducted adjustment) is named in a sentence that says something comes off; and no sentence naming
                  a fee or credit (with the sentences after it on its line that go on about it: "It is added once
                  the tax has been worked out") puts it on the other side of the tax from the case

and then one of the acceptance rules:

  rules_agree     the rule reader gives the same block, byte for byte
  stated_figures  the figures the text states reproduce, to the cent, and between them they cover every line
  threshold       every decision the tagger made (each span's role, each link, each flag) has probability at least
                  its threshold plus the margin band; a decision inside the band [t - band, t + band) abstains

Thresholds are inputs, fitted on held-out data at zero errors; the band is sized from the spread of probabilities
measured across devices. With every threshold at 0.5 or more, a decision that clears `t + band` on one device clears
`t` on any device whose spread is under `band`, so devices can differ only in block or no block, never in two
different blocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from decisio.calculator import normalize as N
from decisio.calculator.cascade import assembler as A
from decisio.calculator.cascade.tags import Tags

# rules the calculator's format cannot express: a tagger may read such a sentence as one it can (rounding up as
# rounding to the nearest step), so the text itself vetoes the block (`rule_forms`)
_TIE = re.compile(r"\bhal(?:f|ves)\b[^.\n]{0,40}?\b(?:up|down|upward|downward)\b", re.I)
_DIRECTION = re.compile(
    r"\bround(?:s|ed|ing)?\b[^.\n]{0,80}?\b(?:up|down|upward|upwards|downward|downwards)\b"
    r"|\b(?:up|down) to the (?:next|nearest|previous|following)\b|\bceil(?:ing)?\b|\bfloor(?:ed)?\b"
    r"|\b(?:part|partial|started) (?:hour|day|unit|metre|meter|week)s? "
    r"(?:count|counts|is|are) (?:charged )?as (?:a )?full\b",
    re.I,
)
_LIMITS = re.compile(
    r"\b(?:minimum|maximum) (?:charge|fee|amount|of|billing|hire)\b"
    r"|\bwhichever is (?:greater|higher|larger|lower|smaller|less)\b"
    r"|\bcapped at\b|\bup to a (?:maximum|limit) of\b|\bno (?:more|less) than\b",
    re.I,
)

# `wording`: words that say an amount comes off the bill, and that place an amount before or after the tax
_SUBTRACT = re.compile(
    r"\b(?:taken off|takes? off|deduct\w*|credit\w*|discount\w*|allowances?|less|minus|refund\w*|rebate\w*|"
    r"vouchers?|deposits?|prepa(?:id|y\w*)|reduc\w*|subtract\w*|waive\w*|removed|comes? off|"
    r"off the (?:bill|total|invoice|amount))\b",
    re.I,
)
_NOT = r"(?<!not )(?<!n't )(?<!never )"
_AFTER_TAX = re.compile(
    r"\bafter (?:the )?(?:tax|vat|gst)\b|\bonce (?:the )?(?:tax|vat|gst)\b|\bpost-tax\b"
    r"|\b(?:tax|vat|gst) has been (?:worked out|added|charged|applied|calculated)\b"
    r"|\b(?:not|never) (?:be )?(?:part of|counted in|included in|inside|in|within) (?:the )?(?:amount|taxed|tax|base)"
    r"|\boutside (?:the )?(?:tax|taxed)|\bexcluded from (?:the )?(?:tax|taxed)",
    re.I,
)
_BEFORE_TAX = re.compile(
    r"\bbefore (?:the )?(?:tax|vat|gst)\b|\bpre-tax\b"
    rf"|{_NOT}\b(?:inside|included in|counted in|part of|within) "
    r"(?:the )?(?:amount the tax|taxed|tax base|amount taxed)",
    re.I,
)
# a label is not cut out of a longer phrase: a label that starts with a small letter follows a function word,
# punctuation or the start of its line ("days" out of "Daily rate days" is refused). The words after a label are not
# read: a head noun ("the Wiper blades line", "the Goods weight charge") is written either way.
_LABEL_BEFORE_OK = frozenset(
    "the a an under for of and or your our this that these those each every per to in on at with by as from also "
    "plus is are was were has have its their one any no called named labelled marked item line".split()
)
_WORD_BEFORE = re.compile(r"([A-Za-z][\w'-]*)[ \t]+$")


@dataclass(frozen=True)
class Check:
    gate: str
    ok: bool
    code: str = ""  # the failure: a short code
    detail: str = ""  # where (offsets, roles, paths; never the text)


def _pass(gate: str) -> Check:
    return Check(gate, True)


def _fail(gate: str, code: str, detail: str = "") -> Check:
    return Check(gate, False, code, detail)


@dataclass(frozen=True)
class Thresholds:
    """Per decision kind: `role` (each span's role), `link` (grouping and targets), `flag`; None turns the threshold
    rule off (it never accepts). `band` is the margin band around each threshold."""

    role: float | None = None
    link: float | None = None
    flag: float | None = None
    band: float = 0.0

    def __post_init__(self):
        if not 0 <= self.band < 0.5:
            raise ValueError("band must be in [0, 0.5)")
        for name in ("role", "link", "flag"):
            t = getattr(self, name)
            if t is not None and not (0.5 <= t and t + self.band <= 1):
                raise ValueError(f"{name} threshold must be at least 0.5 and leave room for the band below 1")

    @property
    def enabled(self) -> bool:
        return None not in (self.role, self.link, self.flag)


# --- required gates ----------------------------------------------------------------------------------------------


def provenance(asm: A.Assembly) -> Check:
    gate = "provenance"
    texts = dict(asm.texts)
    docs = N.document(((texts["context"] + "\n") if texts.get("context") else "") + texts["state"])
    values = {w: N.normalize(t, docs).values for w, t in texts.items() if t}
    for op in asm.operands:
        if not op.spans:
            return _fail(gate, "no-span", op.path)
        read = []
        for s in op.spans:
            src = texts.get(s.where, "")
            if src[s.start : s.end] != s.text:
                return _fail(gate, "not-verbatim", f"{op.path} at {s.key}")
            inside = [v for v in values.get(s.where, ()) if v.span[0] < s.end and v.span[1] > s.start]
            if len(inside) != 1:
                return _fail(gate, "not-one-value", f"{op.path} at {s.key}")
            read.append(inside[0])
        if op.how == "range-days":
            ranges = {id(v) for v in read}
            if len(ranges) != 1 or read[0].kind != "date-range":
                return _fail(gate, "derivation", op.path)
            if op.value not in {Fraction(read[0].days(True) or 0), Fraction(read[0].days(False) or 0)}:
                return _fail(gate, "derivation", op.path)
            continue
        for v in read:
            if v.kind in ("date", "date-range", "time-of-day", "ambiguous"):
                return _fail(gate, "derivation", f"{op.path}: a {v.kind}")
            x = Fraction(v.value)
            if op.how == "unit-factor":
                f = N.factor(v.unit, _length_unit(asm, op.path))
                x = None if f is None else x * f
            elif op.how == "minutes" and v.kind == "quantity" and v.dimension == "time":
                f = N.factor(v.unit, "min")
                x = None if f is None else x * f
            if x != op.value:
                return _fail(gate, "value", op.path)
    return _pass(gate)


def _length_unit(asm: A.Assembly, path: str) -> str:
    i = int(re.match(r"items\[(\d+)\]", path).group(1))
    return asm.input["items"][i]["unit"]


_DIGITS = re.compile(r"\d+")


def numbers(tags: Tags, state: str, context: str | None = None) -> Check:
    """Every run of digits lies inside one span, and every value the normalizer reads (numbers in words too) meets a
    span; a span of any role counts, a distractor included."""
    gate = "numbers"
    texts = {"state": state, "context": context or ""}
    doc = N.document(((context + "\n") if context else "") + state)
    for where, text in texts.items():
        if not text:
            continue
        spans = sorted((s.start, s.end) for s in tags.spans if s.where == where)

        def inside(a: int, b: int) -> bool:
            return any(s <= a and b <= e for s, e in spans)

        def meets(a: int, b: int) -> bool:
            return any(s < b and a < e for s, e in spans)

        for m in _DIGITS.finditer(text):
            if not inside(m.start(), m.end()):
                return _fail(gate, "untagged-digits", f"{where}:{m.start()}-{m.end()}")
        for v in N.normalize(text, doc).values:
            if not meets(*v.span):
                return _fail(gate, "untagged-value", f"{where}:{v.span[0]}-{v.span[1]} ({v.kind})")
    return _pass(gate)


# words that are units but more often plain English next to a number ("in", "a", "t" for tonne)
_NOT_UNITS = frozenset({"in", "a", "t", "s", "g", "l", "at"})


def _unit_word(word: str) -> str | None:
    """A measured unit (mass, length, volume, area, time); count nouns (`entries`, `shifts`) are not units here."""
    u = None if word.lower() in _NOT_UNITS else A.unit_of(word)
    return u if u and N.dimension(u) else None


_SENTENCE_END = re.compile(r"[.;!?](?=\s)")


def governing_unit(text: str, start: int, end: int, skip=()) -> str | None:
    """The unit a bare number takes from its line: the nearest unit word before it on the line, else the nearest after
    it, within its sentence (`minutes: 86, 356`, `lengths in metres (2.75, 11.05)`, `2.089 | yd`); a unit joined to it
    by a hyphen comes first (`a 15-minute step`). Words inside `skip` ranges (the line
    labels: `Packing hours` names a line, it is no unit) are passed over."""
    glued = re.match(r"-([A-Za-z]+)", text[end : end + 16])  # `a 15-minute step`
    if glued and _unit_word(glued.group(1)):
        return _unit_word(glued.group(1))
    a = text.rfind("\n", 0, start) + 1
    b = text.find("\n", end)
    b = len(text) if b < 0 else b
    ends = [m.end() for m in _SENTENCE_END.finditer(text, a, start)]  # the sentence the number is in
    a = ends[-1] if ends else a
    m = _SENTENCE_END.search(text, end, b)
    b = m.start() if m else b

    def units_in(lo: int, hi: int) -> list[str]:
        out = []
        for m in re.finditer(r"[A-Za-z]+", text[lo:hi]):
            s, e = lo + m.start(), lo + m.end()
            if any(x <= s and e <= y for x, y in skip):
                continue
            u = _unit_word(m.group())
            if u:
                out.append(u)
        return out

    before = units_in(a, start)
    if before:
        return before[-1]
    after = units_in(end, b)
    return after[0] if after else None


def units(asm: A.Assembly) -> Check:
    gate = "units"
    texts = dict(asm.texts)
    labels = {w: [(s.start, s.end) for s in asm.spans if s.where == w and s.role == "item_label"] for w in texts}
    for g in asm.groups:
        vals = [(r, s, v) for r, s, v in g.values]

        def per_of(role: str) -> set[str]:
            return {A.per_unit_near(texts[s.where], s.start, s.end) for r, s, _ in vals if r == role} - {None}

        if g.kind == "time":
            if g.per not in ("h", "min") or per_of("rate") - {g.per}:
                return _fail(gate, "rate-unit", g.id)
            for r, s, v in vals:
                if r in ("duration", "rounding_step") and A.bare(v):
                    if governing_unit(texts[s.where], s.start, s.end, labels[s.where]) != "min":
                        return _fail(gate, "bare-minutes", f"{g.id} {r} at {s.key}")
                elif r in ("duration", "rounding_step") and v.kind == "quantity" and v.dimension != "time":
                    return _fail(gate, "not-time", f"{g.id} {r} at {s.key}")
        elif g.kind == "length":
            unit = next(it for it in asm.input["items"] if it["label"] == g.label)["unit"]
            if N.dimension(unit) in (None, "time"):
                return _fail(gate, "length-unit", g.id)
            if per_of("rate") - {unit}:
                return _fail(gate, "rate-unit", g.id)
            for r, s, v in vals:
                if r == "quantity" and A.bare(v):
                    if governing_unit(texts[s.where], s.start, s.end, labels[s.where]) != unit:
                        return _fail(gate, "bare-length", f"{g.id} at {s.key}")
        elif g.kind == "convert":
            it = next(x for x in asm.input["items"] if x["label"] == g.label)
            u, to = it["unit"], it["to_unit"]
            du, dt = N.dimension(u), N.dimension(to)
            if du is None or du != dt or du == "time":
                return _fail(gate, "convert-dimension", g.id)
            ff = [v for r, _, v in vals if r == "factor_from"]
            if any(v.unit != u for v in ff):
                return _fail(gate, "factor-unit", g.id)
            exact = N.factor(u, to)
            stated = Fraction(it["factor"])
            if exact is not None and abs(stated - exact) > exact / 50:
                return _fail(gate, "factor-size", g.id)
            if per_of("price") - {to}:
                return _fail(gate, "price-unit", g.id)
        elif g.kind in ("count", "days"):
            q = [v for r, _, v in vals if r == "quantity"]
            measured = {v.unit for v in q if v.kind == "quantity" and v.dimension not in ("count",)}
            pers = {p for p in per_of("price") if N.dimension(p) not in (None, "count", "time")}
            if pers and measured and pers != measured:
                return _fail(gate, "price-unit", g.id)
            if measured and g.kind == "days":
                return _fail(gate, "days-quantity", g.id)
            if pers and not measured:
                return _fail(gate, "price-unit", f"{g.id}: priced per a unit the quantity does not state")
        elif g.kind == "fee":
            q = [v for r, _, v in vals if r == "quantity"]
            if any(v.kind == "quantity" and v.dimension != "count" for v in q):
                return _fail(gate, "fee-count", g.id)
    return _pass(gate)


# --- stated figures ----------------------------------------------------------------------------------------------


def _cents(x: Fraction) -> set[Fraction]:
    """The value to the cent, rounded half to even and half up (a text may use either)."""
    y = abs(x) * 100
    lo = y.numerator // y.denominator
    frac = y - lo
    up = lo + 1 if frac >= Fraction(1, 2) else lo
    even = round(y)
    return {Fraction(up, 100), Fraction(even, 100)}


def figures(asm: A.Assembly) -> list[tuple[str, Fraction, frozenset[str]]]:
    """The case's figures a text may state, each with the lines it covers: every line's value, the items' subtotal,
    the taxed base, the tax and the total."""
    from decisio.calculator import CalculatorInputError, compute
    from decisio.calculator.inputs import case_from_input

    case = case_from_input(asm.input, CalculatorInputError)
    by = {i.label: i for i in case.items}
    out = []
    for i in case.items:
        out.append((f"line:{i.label}", i.value, frozenset({i.label})))
    for a in case.adjustments:
        out.append((f"line:{a.label}", a.percent / 100 * by[a.target].value, frozenset({a.label})))
    for f in case.fees:
        out.append((f"line:{f.label}", f.value, frozenset({f.label})))
    for c in case.credits:
        out.append((f"line:{c.label}", c.amount, frozenset({c.label})))
    inside = frozenset(
        [i.label for i in case.items if i.label not in case.outside]
        + [a.label for a in case.adjustments]
        + [f.label for f in case.fees if f.label not in case.outside]
        + [c.label for c in case.credits if c.label not in case.outside]
    )
    everything = frozenset([i.label for i in case.items] + [a.label for a in case.adjustments])
    everything |= frozenset([f.label for f in case.fees] + [c.label for c in case.credits])
    out.append(("subtotal", sum((i.signed for i in case.items), Fraction(0)), frozenset(i.label for i in case.items)))
    base = compute.taxed_base(case)
    tax = base * case.tax.rate / 100
    out.append(("taxed-base", base, inside))
    out.append(("tax", tax, inside))
    after = sum((f.value for f in case.fees if f.label in case.outside), Fraction(0))
    after -= sum((c.amount for c in case.credits if c.label in case.outside), Fraction(0))
    after -= sum((i.value for i in case.items if i.kind == "daycredit" and i.label in case.outside), Fraction(0))
    total = base + (tax if case.tax.kind == "charged" else -tax) + after
    out.append(("total", total, everything))
    return out


def stated_figures(asm: A.Assembly) -> Check:
    """`no-stated-figure` when the text states none (the rule does not apply); `stated-mismatch` (a veto) when one
    equals no figure of the case; `partial` when the matched figures leave a line unchecked."""
    gate = "stated_figures"
    if not asm.stated:
        return _fail(gate, "no-stated-figure")
    figs = figures(asm)
    lines = set().union(*(cover for _, _, cover in figs))
    covered: set[str] = set()
    for span, value in asm.stated:
        hits = [(name, cover) for name, x, cover in figs if abs(value) in _cents(x)]
        if not hits:
            return _fail(gate, "stated-mismatch", f"at {span.key}")
        for _, cover in hits:
            covered |= cover
    if covered != lines:
        return _fail(gate, "partial", f"{len(lines - covered)} lines not covered")
    return _pass(gate)


# --- acceptance --------------------------------------------------------------------------------------------------


def agreement(block: str, rules_block: str | None) -> Check:
    gate = "rules_agree"
    if rules_block is None:
        return _fail(gate, "no-rules-block")
    return _pass(gate) if block == rules_block else _fail(gate, "disagree")


def confidence(tags: Tags, th: Thresholds) -> Check:
    gate = "threshold"
    if not th.enabled:
        return _fail(gate, "no-thresholds")
    decisions = [("role", s.key, s.p) for s in tags.spans]
    for k, p in tags.confidences.items():
        kind = k.split(":", 1)[0]
        if kind not in ("link", "flag"):
            return _fail(gate, "unknown-decision", k)
        decisions.append((kind, k, p))
    decisions.append(("flag", "flag:outcome", 1.0 - tags.abstain_p))
    band = False
    for kind, key, p in decisions:
        t = getattr(th, kind)
        if p >= t + th.band:
            continue
        if p >= t - th.band:
            band = True
            continue
        return _fail(gate, "below-threshold", f"{kind} {key}")
    return _fail(gate, "margin-band") if band else _pass(gate)


def rule_forms(text: str, context: str | None = None) -> Check:
    """A sentence of the state or the context that states a rule the calculator cannot express (a rounding direction,
    a minimum, maximum or cap) refuses the block: `rounding-direction` or `limit-rule`."""
    for source in (context or "", text):
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", source):
            if _DIRECTION.search(_TIE.sub(" ", sentence)):
                return _fail("rule_forms", "rounding-direction")
            if _LIMITS.search(sentence):
                return _fail("rule_forms", "limit-rule")
    return _pass("rule_forms")


def _sentences(text: str):
    """(start, end) of each sentence of the text: split after . ! ? and a space, and at line breaks."""
    start = 0
    for m in re.finditer(r"(?<=[.!?])\s+|\n+", text):
        yield start, m.start()
        start = m.end()
    yield start, len(text)


# a sentence that goes on about the line the previous sentence named ("Booking fee: EUR 38.81 is charged once. It is
# added once the tax has been worked out")
_CONTINUES = re.compile(
    r"\s*(?:it|this|that|they|these|those|its|their|the (?:same )?"
    r"(?:fee|fees|credit|credits|charge|charges|amount|deposit|voucher|discount|allowance|payment))\b",
    re.I,
)


def _naming(asm: A.Assembly, label: str):
    """The passages (of the state and the context) that name a label (case and spacing ignored): each sentence that
    names it, with the sentences right after it on the same line that go on about it (they open with a pronoun or
    "the fee", "the credit" ...)."""
    key = A.norm_label(label)
    for _, text in asm.texts:
        sents = list(_sentences(text))
        for j, (a, b) in enumerate(sents):
            if key not in A.norm_label(text[a:b]):
                continue
            end, k = b, j + 1
            while (
                k < len(sents)
                and "\n" not in text[end : sents[k][0]]
                and _CONTINUES.match(text[sents[k][0] : sents[k][1]])
            ):
                end, k = sents[k][1], k + 1
            yield text[a:end]


def wording(asm: A.Assembly) -> Check:
    """The text's words against the case: `label-cut` (a label span starts or ends inside a phrase), `no-subtract-word`
    (a line the case takes off is named in no sentence that says something comes off), `tax-side` (a sentence naming a
    fee or credit places it on the other side of the tax)."""
    gate = "wording"
    texts = dict(asm.texts)
    for s in asm.spans:
        if s.role != "item_label":
            continue
        src = texts.get(s.where, "")
        before = _WORD_BEFORE.search(src[: s.start])
        if s.text[:1].islower() and before and before.group(1).casefold() not in _LABEL_BEFORE_OK:
            return _fail(gate, "label-cut", s.key)
    deducted = {A.norm_label(a["label"]) for a in asm.input.get("adjustments", ()) if a.get("direction") == "deduct"}
    outside = {A.norm_label(x) for x in asm.input.get("tax", {}).get("outside", ())}
    for g in asm.groups:
        off = g.kind in ("credit", "daycredit") or (g.kind == "adjustment" and A.norm_label(g.label) in deducted)
        named = list(_naming(asm, g.label))
        if off and not any(_SUBTRACT.search(t) for t in named):
            return _fail(gate, "no-subtract-word", g.id)
        if g.kind in ("fee", "credit", "daycredit"):
            is_after = A.norm_label(g.label) in outside
            for t in named:
                said = "after" if _AFTER_TAX.search(t) else "before" if _BEFORE_TAX.search(t) else None
                if said is not None and (said == "after") != is_after:
                    return _fail(gate, "tax-side", g.id)
    return _pass(gate)
