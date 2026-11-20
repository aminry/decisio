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
  wording         the text's own words agree with the case: no line label is cut out of a longer phrase ("days"
                  out of "Daily rate days", or "Fuel levy" out of the heading "Fuel levy units:"); every line taken
                  off (a credit, a day credit, a deducted adjustment) is named in a sentence that says something
                  comes off; no line the case adds is named with words that say it comes off the bill ("is a credit
                  of", "comes off the bill"); no adjustment is named as added when the case takes it off, or the
                  other way round; no sentence naming a fee or credit (with the sentences after it on
                  its line that go on about it: "It is added once the tax has been worked out") puts it on the
                  other side of the tax from the case, and every fee and credit has its side stated (by name, in a
                  list of what the tax is charged on, or in a sentence about every line); every fee is said to be
                  charged; one count is not read as both a quantity and a day count; a restated total is not read
                  as an entry; an adjustment's target is named where its percent is. Sentences are read over
                  wrapped lines.

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

import functools
import re
from dataclasses import dataclass
from fractions import Fraction

from decisio.calculator import normalize as N
from decisio.calculator.cascade import assembler as A
from decisio.calculator.cascade.tags import Tags

# rules the calculator's format cannot express: a tagger may read such a sentence as one it can (rounding up as
# rounding to the nearest step), so the text itself vetoes the block (`rule_forms`)
_TIE = re.compile(r"\bhal(?:f|ves)\b[^.\n]{0,40}?\b(?:up|down|upward|downward)\b", re.I)
# "up" and "down" in phrasal verbs that say nothing about rounding ("the rounded entries are added up", "these make up
# the taxed amount"): removed before the direction is looked for
_PHRASAL = re.compile(
    r"\b(?:add(?:ed|s|ing)?|sum(?:med|s|ming)?|total(?:l?ed|s|l?ing)?|make[s]?|made|making|set[s]?|pick(?:ed|s)?|"
    r"back(?:ed|s)?|end(?:ed|s)?|line[sd]?|pa(?:y|ys|id)|top(?:ped|s)?|break[s]?|broken|broke|write[s]?|written|"
    r"wrote|mark(?:ed|s)?|scale[sd]?|count(?:ed|s)?|bring[s]?|brought|go(?:es)?|went|come[s]?|came) (?:up|down)\b",
    re.I,
)
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
# an adjustment's direction in the text: added to the bill or a charge, or taken off it (`adjustment-direction`)
_ADDS = re.compile(
    r"\b(?:is|are|gets?|being|then) added\b|\badded (?:to|on|before|after)\b|\bon top of\b"
    r"|\badds? (?:an? |the )?(?:extra |further )?\d[\d.,]*\s?%",
    re.I,
)
# a rounding step is read only where its sentence speaks of rounding ("converted to hours at 60 minutes per hour"
# is no step)
_ROUNDING = re.compile(r"\bround|\bnearest\b|\bstep\b|\bwhole (?:hour|minute|metre|meter|unit)s?\b", re.I)
_TAKES = re.compile(
    r"\b(?:deduct(?:ed|s)?|subtract(?:ed|s)?|taken off|takes? off|comes? off|knocked off|reduc(?:es|ed|ing))\b", re.I
)
# words that say a line comes off the bill itself (not a percentage taken from one charge): a line the case adds,
# named with them, is refused (`comes-off-word`)
_COMES_OFF = re.compile(
    r"\b(?:comes?|taken|takes?|knocked) off (?:the |your )?(?:bill|total|invoice|amount due|balance)\b"
    r"|\bis an? (?:credit|refund|rebate) (?:of|to|for)\b|\bas an? (?:credit|refund)\b|\bcredited (?:to|back)\b",
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
    r"(?:the )?(?:amount the tax|taxed|tax base|amount taxed|amount (?:that is |being )?taxed)",
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
# ... nor stops short of the end of a heading, a table cell or a quoted name: "Fuel levy" out of "Fuel levy units: 44
# at ...", of "| Standing charge days |" or of '"Standing charge days"' (one or two more words, then the colon, the
# cell's border or the closing quote)
_HEADING_REST = re.compile(r"^[ \t]+[A-Za-z][\w'-]*(?:[ \t]+[A-Za-z][\w'-]*)?[ \t]*(?::|\||[\"\u201d'](?![A-Za-z]))")


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
            inside = A.one_restated(inside, src)
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
            if op.how == "magnitude":
                x = abs(x)
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
        for a, b in _sentences(source):
            sentence = _flat(source[a:b])
            if _DIRECTION.search(_PHRASAL.sub(" ", _TIE.sub(" ", sentence))):
                return _fail("rule_forms", "rounding-direction")
            if _LIMITS.search(sentence):
                return _fail("rule_forms", "limit-rule")
    return _pass("rule_forms")


# a line that starts something new rather than going on with the line before it: a list item, a bullet, a table row
# or a "Label:" line
_LINE_OPENS = re.compile(r"(?:\d{1,2}[.)]|[a-z][.)]|[-*\u2022]|[A-Za-z][\w '()&/-]{0,40}:(?!\S))(?:\s|$)|[|{}\[\]\"<=]")
_COLUMNS = re.compile(r"\S[ \t]{3,}\S|\|")


def _wraps(line: str, nxt: str) -> bool:
    """A line break inside a sentence (a wrapped line): the line ends with no sentence or field end, and neither line
    is a table row, a heading or the start of a list item."""
    a, b = line.strip(), nxt.strip()
    if "wrapped-lines" in DISABLED or not a or not b or a[-1] in ".!?:;|" or _COLUMNS.search(a) or _COLUMNS.search(b):
        return False
    if _LINE_OPENS.match(b) or (a.isupper() and len(a) > 3) or (b.isupper() and len(b) > 3):
        return False
    return True


def _sentences(text: str):
    """(start, end) of each sentence of the text: split after . ! ? and a space, at blank lines, and at line breaks
    that are not wrapped lines (a sentence wrapped over lines is one sentence). Cached per text (every check reads the
    same sentences), so a read splits each text once."""
    return _sentences_of(text, "wrapped-lines" in DISABLED)


@functools.lru_cache(maxsize=64)
def _sentences_of(text: str, no_wraps: bool) -> tuple[tuple[int, int], ...]:
    return tuple(_split_sentences(text))


def _split_sentences(text: str):
    breaks = []  # the line breaks that end a sentence
    lines = text.split("\n")
    at = 0
    for i, line in enumerate(lines[:-1]):
        at += len(line)
        if not _wraps(line, lines[i + 1]):
            breaks.append(at)
        at += 1
    start = 0
    for m in re.finditer(r"(?<=[.!?])\s+|\n+", text):
        if m.group() == "\n" and text[m.start() - 1 : m.start()] not in (".", "!", "?") and m.start() not in breaks:
            continue  # a wrapped line
        yield start, m.start()
        start = m.end()
    yield start, len(text)


@functools.lru_cache(maxsize=4096)
def _flat(text: str) -> str:
    """Text with its runs of spaces and line breaks read as one space (cues read over wrapped lines)."""
    return " ".join(text.split())


# a sentence that goes on about the line the previous sentence named ("Booking fee: EUR 38.81 is charged once. It is
# added once the tax has been worked out")
_CONTINUES = re.compile(
    r"\s*(?:it|this|that|they|these|those|its|their|the (?:same )?"
    r"(?:fee|fees|credit|credits|charge|charges|amount|deposit|voucher|discount|allowance|payment))\b",
    re.I,
)


# an enumeration of lines after a sentence that introduces them: "Two further items are applied after the tax ...
# First, a Damage waiver fee ... Second, a Breakdown allowance ..."
_ENUMERATES = re.compile(r"\s*(?:first|second|third|fourth|fifth|next|then|finally|lastly|also)\b", re.I)
_INTRODUCES = re.compile(r"\b(?:items|charges|lines|fees|credits|amounts|two|three|four|both)\b", re.I)


def _same_item(gap: str) -> bool:
    """Two sentences are in one item: on one line, or the second on an indented line under the first (a wrapped list
    item: "5. The Connection fee is applied AFTER tax.\n   It is outside the taxed amount")."""
    return "\n" not in gap or ("wrapped-lines" not in DISABLED and bool(re.fullmatch(r"[ \t]*\n[ \t]+", gap)))


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
                and _same_item(text[end : sents[k][0]])
                and _CONTINUES.match(text[sents[k][0] : sents[k][1]])
            ):
                end, k = sents[k][1], k + 1
            start = a
            if "enumeration" not in DISABLED and _ENUMERATES.match(text[a:b]):
                k = j - 1
                while (
                    k >= 0
                    and "\n\n" not in text[sents[k][1] : start]
                    and _ENUMERATES.match(text[sents[k][0] : sents[k][1]])
                ):
                    k -= 1
                if (
                    k >= 0
                    and "\n\n" not in text[sents[k][1] : start]
                    and _INTRODUCES.search(text[sents[k][0] : sents[k][1]])
                ):
                    start = sents[k][0]
            yield _flat(text[start:end])


# a label after a heading keyword on its line ("CHARGE: Site visits"): stopping one or two words before the line ends
# is a cut (a line with no keyword, such as a shop's title, is not read)
_HEADING_LINE = re.compile(r"^[ \t]*[A-Z][A-Z ]{1,20}:[ \t]*$")
_LINE_REST = re.compile(r"^[ \t]+[A-Za-z][\w'-]*(?:[ \t]+[A-Za-z][\w'-]*)?[ \t]*(?=\n|$)")
# words that may stand between a label and its heading's colon without cutting it ("Gardener time entries:",
# "Excavator hire charge only:"), and "and" before another line's label ("Theatre time and Consultation:")
_CONTINUATIONS = frozenset({"charge", "charges", "only", "entries", "entry", "applies", "apply"})


def _continues_label(src: str, end: int, rest: str, asm: A.Assembly) -> bool:
    words = re.findall(r"[A-Za-z][\w'-]*", rest)
    if words and all(w.casefold() in _CONTINUATIONS for w in words):
        return True
    if len(words) == 2 and words[0].casefold() == "and":
        at = end + rest.index(words[1], rest.casefold().index("and") + 3)
        return any(x.role == "item_label" and x.start == at for x in asm.spans)
    return False


# a line priced by the day says so where its price is stated ("per unit per day", "a day", "daily"); "$388.10 each ...
# for each of the 17 days" is a count of days, not a day count beside a quantity
_PER_DAY = re.compile(
    r"\bper (?:unit )?(?:per )?(?:day|night|week)\b|\b(?:a|each|every) (?:day|night)\b|\bdaily\b|/ ?day\b", re.I
)
# ... and a line priced both per unit and per day says per unit ("$378.73 per unit per day", "each unit a day")
_PER_UNIT = re.compile(
    r"\bper (?:unit|item|piece|person|head|machine|vehicle)\b|\b(?:each|a|every) (?:unit|item|piece|machine|vehicle)\b"
    r"|/ ?unit\b",
    re.I,
)


def _whole_sentence(text: str, at: int) -> str:
    """The sentence around a position, over line wraps: from the last sentence end or blank line before it to the
    next one after it."""
    a = max(text.rfind(". ", 0, at), text.rfind("\n\n", 0, at), text.rfind(".\n", 0, at))
    a = 0 if a < 0 else a + 1
    ends = [i for i in (text.find(". ", at), text.find("\n\n", at), text.find(".\n", at)) if i >= 0]
    return text[a : min(ends) if ends else len(text)]


# checks switched off by name, for audits that measure what each one blocks; empty in serving
DISABLED: set[str] = set()


def wording(asm: A.Assembly) -> Check:
    """The text's words against the case: `label-cut` (a label span starts or ends inside a phrase), `no-subtract-word`
    (a line the case takes off is named in no sentence that says something comes off), `comes-off-word` (a line the
    case adds is named as coming off the bill), `adjustment-direction` (an adjustment named as added when the case
    takes it off, or the other way, with no word for the case's direction), `rounding-without-rule` (a rounding step
    whose sentence does not speak of rounding), `count-as-days` (a line priced by the day whose price is nowhere stated
    per day: "$388.10 each ... for each of the 17 days"), `count-is-days` (one count read as both the quantity and the
    days: "34 at $96.35 each ... for each day, for 34 days"), `restated-total` (one of three or more entries is the sum
    of the others: "Total time 268 min" after its entries), `adjustment-target` (the percent's sentence does not name
    the adjustment's target, or names another line), `tax-side` (the text places a fee or credit on the other side of
    the tax), `tax-side-unstated` (the text never states a fee's or credit's side), `fee-unstated` (a fee the text only
    shows beside its amount, never saying it is charged), `adjustment-direction-unstated` (no verb near the
    adjustment's name, nor its name's own word, gives the case's direction), `label-direction` (a name that states both
    directions: "Rush surcharge deduction"), `tax-side-ambiguous` (wording that leaves a side of the tax open: "handled
    together with the tax"), `unaccounted-line` (a fee or credit the text names that is no line of the case),
    `credit-direction` (a line taken off that the text adds by name: "plus the Standing charge days"),
    `identifier-number` (a number that names something read as a count: "Length section 1"), `negative-total` (a case
    whose taxed base or total is below zero), `subtract-word-outside-name` (a line taken off whose only word for it is
    its own name: "Site access deposit: $150.00"), `adjustment-side-unstated` (an adjustment whose side of the tax the
    text never states). An every-line statement needs a line noun ("every charge"), and an adjustment's target named
    after "not", "except" or "excluding" refuses."""
    gate = "wording"
    texts = dict(asm.texts)
    for s in asm.spans:
        if s.role != "item_label":
            continue
        src = texts.get(s.where, "")
        before = _WORD_BEFORE.search(src[: s.start])
        if s.text[:1].islower() and before and before.group(1).casefold() not in _LABEL_BEFORE_OK:
            return _fail(gate, "label-cut", s.key)
        # a label that starts or ends inside a word ("haul" out of "Linehaul")
        in_word = (s.start > 0 and src[s.start - 1].isalnum() and s.text[:1].isalnum()) or (
            s.end < len(src) and src[s.end].isalnum() and s.text[-1:].isalnum()
        )
        if "label-cut-in-word" not in DISABLED and in_word:
            return _fail(gate, "label-cut", s.key)
        if "label-cut-hyphen" not in DISABLED and _hyphen_cut(src, s.start, s.end):
            return _fail(gate, "label-cut", s.key)
        m = _HEADING_REST.match(src[s.end :])
        if m and not _continues_label(src, s.end, m.group(0), asm):
            return _fail(gate, "label-cut", s.key)
        m = None if "label-cut-cell" in DISABLED else _CELL_REST.match(src[s.end :])
        if m and _starts_cell(src, s.start) and not _continues_label(src, s.end, m.group(0), asm):
            return _fail(gate, "label-cut", s.key)
        line_start = src.rfind("\n", 0, s.start) + 1
        if "label-cut-heading-line" not in DISABLED and _HEADING_LINE.match(src[line_start : s.start]):
            m = _LINE_REST.match(src[s.end :])
            if m and not _continues_label(src, s.end, m.group(0), asm):
                return _fail(gate, "label-cut", s.key)
    for x in asm.spans:
        if x.role == "rounding_step":
            src = texts.get(x.where, "")
            if "rounding-without-rule" not in DISABLED and not _ROUNDING.search(_whole_sentence(src, x.start)):
                return _fail(gate, "rounding-without-rule", x.key)
    for g in asm.groups:
        if g.kind in ("days", "daycredit") and "count-as-days" not in DISABLED:
            prices = [x for r, x, _ in g.values if r == "price"]
            if prices and not any(_PER_DAY.search(_whole_sentence(texts.get(x.where, ""), x.start)) for x in prices):
                return _fail(gate, "count-as-days", g.id)
        if g.kind == "days" and "count-is-days" not in DISABLED:
            # "34 at $96.35 each ... $96.35 for each day, for 34 days": one count read as both the quantity and the
            # days; a line priced per unit per day says so
            q = {Fraction(v.value) for r, _, v in g.values if r == "quantity"}
            d = {Fraction(v.value) for r, _, v in g.values if r == "day_count"}
            near = [_flat(_whole_sentence(texts.get(x.where, ""), x.start)) for _, x, _ in g.values]
            if q & d and not any(_PER_UNIT.search(t) for t in near):
                return _fail(gate, "count-is-days", g.id)
    if "restated-total" not in DISABLED:
        for it in asm.input.get("items", ()):
            entries = [Fraction(str(x)) for x in it.get("minutes") or it.get("lengths") or ()]
            # "Entry 1 90 min, Entry 2 120 min, Entry 3 58 min, Total time 268 min": the total is no entry
            if len(entries) >= 3 and any(2 * e == sum(entries) for e in entries):
                return _fail(gate, "restated-total", A.norm_label(it["label"]))
    if "adjustment-target" not in DISABLED:
        bad = _adjustment_target(asm)
        if bad:
            return _fail(gate, "adjustment-target", bad)
    deducted = {A.norm_label(a["label"]) for a in asm.input.get("adjustments", ()) if a.get("direction") == "deduct"}
    outside = {A.norm_label(x) for x in asm.input.get("tax", {}).get("outside", ())}
    if "identifier-number" not in DISABLED:
        for x in asm.spans:
            if x.role in _COUNTED and _IDENTIFIER.search(texts.get(x.where, "")[max(0, x.start - 16) : x.start]):
                return _fail(gate, "identifier-number", x.key)
    if "negative-total" not in DISABLED:
        bad = _negative_figure(asm)
        if bad:
            return _fail(gate, "negative-total", bad)
    sides = _stated_sides(asm)
    if "tax-side-ambiguous" not in DISABLED:
        for _, text in asm.texts:
            for a, b in _sentences(text):
                if _TAX_AMBIGUOUS.search(_flat(text[a:b])):
                    return _fail(gate, "tax-side-ambiguous")
    if "unaccounted-line" not in DISABLED:
        extra = _unaccounted(asm)
        if extra:
            return _fail(gate, "unaccounted-line", extra)
    for g in asm.groups:
        off = g.kind in ("credit", "daycredit") or (g.kind == "adjustment" and A.norm_label(g.label) in deducted)
        named = list(_naming(asm, g.label))
        if off and not any(_SUBTRACT.search(t) for t in named):
            return _fail(gate, "no-subtract-word", g.id)
        if off and "subtract-word-outside-name" not in DISABLED:
            # the word that takes the line off is not only the line's own name ("Site access deposit: $150.00")
            key = re.compile(re.escape(A.norm_label(g.label)), re.I)
            if not any(_SUBTRACT.search(key.sub(" ", t)) for t in named):
                return _fail(gate, "subtract-word-outside-name", g.id)
        if not off and g.kind != "adjustment" and any(_COMES_OFF.search(t) for t in named):
            return _fail(gate, "comes-off-word", g.id)
        if g.kind == "adjustment":
            deduct = A.norm_label(g.label) in deducted
            same, other = (_TAKES, _ADDS) if deduct else (_ADDS, _TAKES)
            if any(other.search(t) for t in named) and not any(same.search(t) for t in named):
                return _fail(gate, "adjustment-direction", g.id)
            # the direction is read from words of the text for this line ("added", "deducts", "plus the ..."), never
            # from the tagger alone: "Rush surcharge: 8% of the Pallet handling charge only" states none
            if "adjustment-direction-unstated" not in DISABLED:
                said = _stated_direction(g.label, named)
                if said != ("deduct" if deduct else "add"):
                    return _fail(gate, "adjustment-direction-unstated", g.id)
        if g.kind in ("credit", "daycredit") and "credit-direction" not in DISABLED:
            # a line taken off that the text adds by name ("The taxed amount is Gas used plus Standing charge days")
            key = re.escape(A.norm_label(g.label))
            if any(re.search(rf"\bplus (?:the |a |an )?{key}\b", t.casefold()) for t in named):
                return _fail(gate, "credit-direction", g.id)
        if "label-direction" not in DISABLED and _label_contradicts(g, A.norm_label(g.label) in deducted):
            return _fail(gate, "label-direction", g.id)
        if g.kind == "fee" and "fee-unstated" not in DISABLED and not any(_CHARGED.search(t) for t in named):
            # an amount the text only shows ("Parking reimbursement | - | $35.00"), never saying it is charged
            return _fail(gate, "fee-unstated", g.id)
        if g.kind == "adjustment" and "adjustment-side-unstated" not in DISABLED:
            said = any(_BEFORE_TAX.search(t) or _AFTER_TAX.search(t) for t in named)
            said = said or A.norm_label(g.label) in sides.named or "before" in sides.every
            if not said:
                return _fail(gate, "adjustment-side-unstated", g.id)
        if g.kind in ("fee", "credit", "daycredit"):
            code = _side_check(asm, g, named, A.norm_label(g.label) in outside, sides)
            if code:
                return _fail(gate, code, g.id)
    return _pass(gate)


# a table cell's label that stops one or two words before the cell ends (the column's padding or a tab before the next
# cell): "Standing charge" out of "Standing charge days   18   £193.89 each"
_CELL_REST = re.compile(r"^[ \t][A-Za-z][\w'()-]*(?:[ \t][A-Za-z][\w'()-]*)?(?=[ \t]{2,}\S|\t)")


def _starts_cell(src: str, at: int) -> bool:
    """The label opens its line's first cell (only spaces before it on the line) or a cell after a column gap."""
    line = src[src.rfind("\n", 0, at) + 1 : at]
    return not line.strip() or bool(re.search(r"(?:[ \t]{2,}|\t|\|)[ \t]*$", line))


# words in a label that name a direction: a line added to the bill is not called a deduction, nor a line taken off a
# surcharge ("Rush surcharge deduction" on an added line, "Rush surcharge" read as deducted)
_ADD_WORDS = re.compile(r"\b(?:surcharges?|premiums?|supplements?|uplifts?|mark-?ups?|levy|levies)\b", re.I)
_OFF_WORDS = re.compile(
    r"\b(?:discounts?|deductions?|rebates?|reductions?|refunds?|credits?|vouchers?|deposits?|prepayments?|"
    r"retainers?)\b",
    re.I,
)


# the verbs that state an adjustment's direction ("added", "deducts", "plus the Rush surcharge", "minus the ...")
_ADD_VERB = re.compile(_ADDS.pattern + r"|\badd(?:s|ed|ing)?\b|\bon top\b|\bincreas(?:es|ed|ing)\b", re.I)
_OFF_VERB = re.compile(_TAKES.pattern + r"|\bdeduct(?:s|ed|ing)?\b|\bsubtract(?:s|ed|ing)?\b|\btaken? off\b", re.I)
# "plus" and "minus" belong to the name after them ("..., plus the Rush surcharge, minus the Account credit")
_PRE_ADD = re.compile(r"\bplus (?:the |a |an )?$", re.I)
_PRE_OFF = re.compile(r"\b(?:minus|less) (?:the |a |an )?$", re.I)


def _stated_direction(label: str, named: list[str]) -> str | None:
    """The direction the text states for an adjustment: the verb nearest a mention of its name, over the passages
    that name it ("The taxed amount is the Pallet handling charge, plus the Rush surcharge, minus the Account credit":
    add); with no verb, the name's own word if it has one direction ("Volume discount": deduct); else None."""
    key = A.norm_label(label)
    best = None  # (distance, direction)
    for t in named:
        low = t.casefold()
        at = [i for i in range(len(low)) if low.startswith(key, i)]
        if not at:
            continue
        spans = [(i, i + len(key)) for i in at]
        for i in at:
            for rx, d in ((_PRE_ADD, "add"), (_PRE_OFF, "deduct")):
                if rx.search(t[max(0, i - 12) : i]):
                    best = (0, d) if best is None or best[0] > 0 else best
        for rx, d in ((_ADD_VERB, "add"), (_OFF_VERB, "deduct")):
            for m in rx.finditer(t):
                if any(a <= m.start() < b for a, b in spans):
                    continue  # a word of the name itself
                dist = min(abs(m.start() - b) if m.start() >= b else abs(a - m.end()) for a, b in spans)
                if best is None or dist < best[0]:
                    best = (dist, d)
    if best is not None:
        return best[1]
    add, off = bool(_ADD_WORDS.search(label)), bool(_OFF_WORDS.search(label))
    return "add" if add and not off else "deduct" if off and not add else None


def _label_contradicts(g: A.Group, deducted: bool) -> bool:
    """A name that states both directions ("Rush surcharge deduction"): a name cut together with a word of its
    sentence, or a line whose direction its own name leaves open. (A name against the line's direction alone, such
    as a "Volume discount" the text adds, is the text's own wording and is read from its verbs.)"""
    return g.kind == "adjustment" and bool(_ADD_WORDS.search(g.label) and _OFF_WORDS.search(g.label))


# wording that leaves an amount's side of the tax open ("handled together with the tax", "applied in line with the tax
# step", "when the invoice is finalised")
_TAX_AMBIGUOUS = re.compile(
    r"\b(?:does not|doesn't|do not|did not) (?:say|state|specify)\b[^.]{0,60}\b(?:before or after|tax)\b|"
    r"\b(?:together|along|alongside|in line|in step|at the same time|in parallel)\b[^.]{0,20}\b(?:with )?the tax\b"
    r"|\bwith the tax step\b|\bwhen the (?:invoice|bill) is finali[sz]ed\b",
    re.I,
)

# a fee, credit or adjustment the text names that is no line of the case ("Any loyalty credit that is confirmed later"):
# its amount or side is not known, so no block
_KIND_NOUN = (
    r"(?:credits?|fees?|discounts?|deposits?|vouchers?|surcharges?|rebates?|refunds?|deductions?|premiums?|"
    r"supplements?|prepayments?|retainers?)"
)
_NAMED_KIND = re.compile(rf"\b([A-Za-z][\w-]*)\s+{_KIND_NOUN}\b")
_GENERIC = frozenset(
    "a an the any each every all no other this that these those its their one some such further extra additional "
    "listed above below and or of per flat fixed one-time single second first same your our is are be".split()
)


def _unaccounted(asm: A.Assembly) -> str:
    labels = [A.norm_label(g.label) for g in asm.groups]
    for _, text in asm.texts:
        for a, b in _sentences(text):
            t = _flat(text[a:b])
            if re.search(r"\bno (?:other|further|additional)\b", t, re.I):
                continue
            low = t.casefold()
            for m in _NAMED_KIND.finditer(t):
                if m.group(1).casefold() in _GENERIC:
                    continue
                phrase = A.norm_label(m.group(0))
                if not any(phrase in lab or lab in low[max(0, m.start() - 40) : m.end() + 5] for lab in labels):
                    return phrase
    return ""


# a number that names something rather than counts it ("Length section 1", "Entry 2", "No. 3", "#4"): never a
# quantity, length, duration or day count
_COUNTED = frozenset({"quantity", "duration", "day_count", "rounding_step", "factor_from", "factor_to"})
_IDENTIFIER = re.compile(
    r"(?:\b(?:section|entry|item|part|line|row|step|stage|page|ref|reference|order|invoice|account|no|number)\.?|#)"
    r"[ \t]*$",
    re.I,
)


def _negative_figure(asm: A.Assembly) -> str:
    """A case whose taxed base or total comes out below zero is no bill the text describes: refused."""
    try:
        for name, x, _ in figures(asm):
            if name in ("taxed-base", "total") and x < 0:
                return name
    except Exception:
        return "figures"
    return ""


def _hyphen_cut(src: str, start: int, end: int) -> bool:
    """A label cut at a hyphen inside a word: "-booking sur" or "booking" out of "Pre-booking", "Pre" out of
    "Pre-booking"."""
    if src[start:end][:1] == "-" and start > 0 and src[start - 1].isalnum():
        return True
    if src[start:end][-1:] == "-" and end < len(src) and src[end].isalnum():
        return True
    if start > 1 and src[start - 1] == "-" and src[start - 2].isalnum() and src[start : start + 1].isalnum():
        return True
    return end + 1 < len(src) and src[end] == "-" and src[end + 1].isalnum() and src[end - 1 : end].isalnum()


_PRICED = ("count", "days", "time", "length", "convert")


def _adjustment_target(asm: A.Assembly) -> str:
    """An adjustment whose percent is stated in a sentence that does not name its target: the target must be named
    where the percent is ("It deducts 15% of the Night shifts charge only"), and a percent's sentence that names
    another priced line but not the target refuses. Returns the adjustment's id, or ""."""
    texts = dict(asm.texts)
    priced = {A.norm_label(g.label) for g in asm.groups if g.kind in _PRICED}
    targets = {A.norm_label(a["label"]): A.norm_label(a.get("of") or "") for a in asm.input.get("adjustments", ())}
    for g in asm.groups:
        if g.kind != "adjustment":
            continue
        target = targets.get(A.norm_label(g.label), "")
        seen = False
        for r, x, _ in g.values:
            if r != "percent":
                continue
            src = texts.get(x.where, "")
            sent = next((A.norm_label(src[a:b]) for a, b in _sentences(src) if a <= x.start < b), "")
            if target and target in sent:
                seen = True
                at = sent.find(target)
                if "target-negated" not in DISABLED and re.search(
                    r"\b(?:not|except|excluding|other than|but not)\b(?: (?:the|to|on|of))?[ '\"]*$",
                    sent[max(0, at - 20) : at],
                ):
                    return g.id
            elif any(lab in sent for lab in priced - {target} if not (target and lab in target)):
                return g.id
        if not seen:
            return g.id
    return ""


# words that say a fee is charged: a fee the text names only beside its amount is not guessed (`fee-unstated`)
_CHARGED = re.compile(
    r"\b(?:charg\w*|fees?|add(?:s|ed|ing)?|appl(?:y|ies|ied)|payable|paid|pays?|plus|on top|surcharges?|levy|costs?|"
    r"each|once|twice|times|per)\b",
    re.I,
)


# --- the side of the tax a fee or credit is on, as the text states it ----------------------------------------------

# a sentence that lists what the tax is charged on: "The taxed amount includes Meter reads, Gas used and Connection
# fee", "The tax applies to the whole amount, which includes ...", "Tax ... is calculated on the Site attendance and
# Disposal fee charges", "It applies to every charge and fee listed above, including the Delivery fee"
_LISTS = re.compile(
    r"\b(?:taxed (?:amount|base)|amount (?:that is |being )?taxed|tax base|amount the tax is (?:charged|calculated) on)"
    r"\b,?(?: which)?(?: is)? (?:includes?|including|contains?|covers?|holds?|made up of|is|are)\b"
    r"|\b(?:applies|apply|is charged|is calculated|is worked out)(?: only)? (?:to|on)\b(?! top\b)"
    r"|\b(?:covers?|includes?|contains?)\b",
    re.I,
)
_TAX_WORD = re.compile(r"\b(?:tax\w*|vat|gst)\b", re.I)
_NOT_TAX = re.compile(r"\b(?:discount|adjustment|surcharge|premium|supplement|deducts?|reduction|rebate)\b", re.I)
_EXCEPT = re.compile(r"\b(?:except|other than|apart from|but not|excluding|not including)\b", re.I)
_NEGATED = re.compile(r"\b(?:not|never|no)\b|n't\b", re.I)
# the list ends where a new clause starts; a name right before a verb is that clause's subject ("A and B, and the
# Booking fee is added afterwards"), not one of the list's names
_VERB = re.compile(
    r"\b(?:is|are|was|were|be|been|has|have|had|added|applied|charged|taken|comes?|deducted|subtracted)\b", re.I
)
_JOIN = re.compile(r"\b(?:so|but|while|whereas|because|then)\b|;", re.I)
_HEAD_NOUN = re.compile(r"[ \t]*(?:fee|credit|charge|amount|deduction|allowance)?s?[ \t]*$", re.I)
# the list covers every line ("every charge", "all of the above", "everything")
_ALL_LINES = re.compile(
    r"\b(?:every|all|each)\b(?: \w+){0,2} (?:lines?|charges?|items?|amounts?)\b|\ball of (?:this|these|them|it)\b"
    r"|\b(?:all|everything)(?: of the| listed)? above\b|\beverything\b",
    re.I,
)
# a sentence that says nothing is on the after-tax side
_NOTHING_OUTSIDE = re.compile(
    r"\bnothing (?:is |falls |sits |else is )?(?:applied |placed )?(?:outside|after (?:the )?tax)\b"
    r"|\bno (?:fees?|credits?|lines?)(?: or (?:fees?|credits?))? (?:is|are|sits?|falls?|lies?) (?:applied )?"
    r"(?:outside|after)",
    re.I,
)
_EVERY = re.compile(r"\b(?:every|all|each|everything)\b", re.I)
# a fee or a credit named by its kind: read as that line when the case has one line of the kind
_KIND_NOUNS = {
    "fee": re.compile(r"\b(?:the|this|every|each|all|and) (?:fees?)\b", re.I),
    "credit": re.compile(r"\b(?:the|this|every|each|all|and) (?:credits?|reductions?|deductions?)\b", re.I),
}


@dataclass
class _Sides:
    named: dict  # a label (normalised) -> the sides the text puts it on by name ("after", "before")
    kinds: dict  # "fee" / "credit" -> the sides stated for the kind ("the fee and the reduction are inside ...")
    every: set  # the sides stated for every line


def _lists(t: str, prev: str):
    """The places a sentence lists what the tax is charged on. "applies to", "is charged on", "covers" make a list
    only when the sentence speaks of the tax before them, or opens with "It" right after a sentence about the tax (not
    a discount's "It does not apply to the meal allowances")."""
    for m in _LISTS.finditer(t):
        if m.group().casefold().startswith(("taxed", "amount", "tax base")):
            yield m
            continue
        about_tax = _TAX_WORD.search(t[: m.start()]) and not _NOT_TAX.search(t[: m.start()])
        it_tax = re.match(r"(?:it|this)\b", t, re.I) and _TAX_WORD.search(prev) and not _NOT_TAX.search(prev)
        if about_tax or it_tax:
            yield m


def _stated_sides(asm: A.Assembly) -> _Sides:
    movable = [g for g in asm.groups if g.kind in ("fee", "credit", "daycredit")]
    labels = {A.norm_label(g.label) for g in movable}
    out = _Sides({}, {}, set())
    for _, text in asm.texts:
        sents = [_flat(text[a:b]) for a, b in _sentences(text)]
        for j, t in enumerate(sents):
            low = A.norm_label(t)
            nothing = bool(_NOTHING_OUTSIDE.search(t))
            if nothing:
                out.every.add("before")
            for m in () if "tax-lists" in DISABLED else _lists(t, sents[j - 1] if j else ""):
                negated = bool(_NEGATED.search(t[max(0, m.start() - 25) : m.start()]))  # "It does not apply to"
                region = low[len(A.norm_label(t[: m.end()])) :]
                j_end = _JOIN.search(region)
                v_end = _VERB.search(region)
                end = min(x.start() for x in (j_end, v_end, re.search(r"$", region)) if x)
                subject = v_end is not None and v_end.start() == end
                region = region[:end]
                ex = _EXCEPT.search(region)
                cut = ex.start() if ex else len(region)
                for lab in labels:
                    at = region.find(lab)
                    if at < 0 or (subject and _HEAD_NOUN.fullmatch(region[at + len(lab) :])):
                        continue
                    side = "after" if negated or at >= cut else "before"
                    out.named.setdefault(lab, set()).add(side)
                incl = region[:cut]
                if not negated and _ALL_LINES.search(incl):
                    out.every.add("before")
                for kind, noun in _KIND_NOUNS.items():
                    if not negated and noun.search(incl):
                        out.kinds.setdefault(kind, set()).add("before")
            said = "after" if _AFTER_TAX.search(t) else "before" if _BEFORE_TAX.search(t) else None
            if said is None or nothing:
                continue
            every = _ALL_LINES.search(t) if "every-needs-lines" not in DISABLED else _EVERY.search(t)
            if every and not re.search(r"\bno other\b", t, re.I):
                out.every.add(said)
            for kind, noun in _KIND_NOUNS.items():
                if noun.search(t):
                    out.kinds.setdefault(kind, set()).add(said)
    return out


def _side_check(asm: A.Assembly, g: A.Group, named: list[str], is_after: bool, sides: _Sides) -> str:
    """ "" when the text states the line's side as the case has it; "tax-side" when the text puts it on the other side
    by name; "tax-side-unstated" when the text never states its side."""
    want = "after" if is_after else "before"
    explicit = set()
    for t in named:
        said = "after" if _AFTER_TAX.search(t) else "before" if _BEFORE_TAX.search(t) else None
        if said is not None:
            explicit.add(said)
    explicit |= sides.named.get(A.norm_label(g.label), set())
    if explicit - {want}:
        return "tax-side"
    if explicit:
        return ""
    kind = "fee" if g.kind == "fee" else "credit"
    one = sum(1 for x in asm.groups if x.kind in (("fee",) if kind == "fee" else ("credit", "daycredit")))
    stated = (sides.kinds.get(kind, set()) if one == 1 else set()) | sides.every
    if "tax-side-unstated" in DISABLED or want in stated:
        return ""
    return "tax-side-unstated"
