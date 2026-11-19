# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Tagged spans and flags -> a `calculator_input`, and the block `from_input` renders from it.

Deterministic, with no model. Every operand is a value the normalizer reads inside one span
(`decisio.calculator.normalize`, with the decimal convention of the whole text), so values, units and the block bytes
are exactly the ones `from_input` gives for the same amounts. Anything inconsistent raises `Refusal` with a code from
`REFUSALS`; nothing is guessed and nothing partial is returned.

The kind of a priced item follows from the roles in its group:

  factor_from + factor_to + quantity + price          convert   (`1 lb = 0.4536 kg`; the price is per the right side)
  duration (one per entry) + rate [+ rounding_step]   time      (the rate's unit, hour or minute, is read next to it)
  quantity (one per entry) + rate [+ rounding_step]   length    (every entry in one unit)
  quantity + price + day_count (or a date range)      days
  price + day_count (or a date range)                 day credit (taken off)
  quantity + price                                    count

Fees (`f`) have a price and an optional count, credits (`c`) a price (the amount), adjustments (`a`) a percent with
the item it applies to; the tax rate is the `tax` group. A value mentioned more than once in one role must read the
same every time; entries (durations, lengths) are every span of the role, in text order, and two entries of the same
value in one line are refused (a restatement and a second entry read the same). Lines are ordered as the text first
mentions them (the context before the state).
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from fractions import Fraction

from decisio.calculator import normalize as N
from decisio.calculator.cascade.tags import FREE_ROLES, ROLES, WORD_ROLES, Tags, TagSpan, thaw

REFUSALS = {
    "span-text": "a span's text is not the source text at its offsets",
    "span-overlap": "two spans overlap",
    "span-value": "a span does not hold exactly one value of the kind its role needs",
    "ambiguous-value": "a value inside a span can be read two ways",
    "unknown-role": "a role the assembler does not know",
    "role-without-item": "a span of a line role names no line, or a free role names one",
    "unknown-item": "a line id that is not i, f, c, a, tax followed by a number",
    "no-label": "a line has no label span",
    "conflicting-labels": "the label spans of one line read differently",
    "conflicting-values": "one role of one line holds two different values (two prices for one item)",
    "item-roles": "a line's roles match no line kind (a required role is missing, or one does not belong)",
    "repeated-entry": "two entries of one line have the same value (a restatement or a second entry)",
    "unit": "a unit that is missing, not stated or not of the right kind",
    "target": "an adjustment's target is not a priced item",
    "two-tax-rates": "more than one tax rate",
    "no-tax": "no tax rate",
    "missing-flag": "a flag the case needs is absent",
    "flag-conflict": "a flag names no line of the right kind, or two flags disagree",
    "two-currencies": "more than one currency in the text",
    "no-currency": "no money amount",
    "range": "a date range whose day count is not certain",
    "inexact": "a value that cannot be written with six decimals",
    "invalid-input": "the assembled input is refused by from_input",
}


class Refusal(Exception):
    """No case: `code` is one of `REFUSALS`, `detail` says where (offsets and roles, never the text)."""

    def __init__(self, code: str, detail: str = ""):
        assert code in REFUSALS, code
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class Operand:
    """One number of the input, with the spans it was read from and how (`same`: the value read in the span;
    `minutes`: a duration in minutes; `range-days`: the days of a date range; `unit-factor`: a step converted with an
    exact definitional factor; `magnitude`: a credit's amount, a minus sign on a mention being its direction)."""

    path: str
    value: Fraction
    spans: tuple[TagSpan, ...]
    how: str = "same"


@dataclass(frozen=True)
class Group:
    """A line as the assembler read it: its id, kind, label and the values of its spans by role."""

    id: str
    kind: str  # count days time length convert daycredit fee credit adjustment tax
    label: str
    values: tuple[tuple[str, TagSpan, object], ...]  # (role, span, the normalizer's value or None)
    per: str = ""  # time: "h" or "min"


@dataclass(frozen=True)
class Assembly:
    input: dict  # the calculator_input
    operands: tuple[Operand, ...]
    groups: tuple[Group, ...]
    stated: tuple[tuple[TagSpan, Fraction], ...]  # the figures the text states
    spans: tuple[TagSpan, ...]
    texts: tuple[tuple[str, str], ...]  # ("state", text), ("context", text)

    def block(self, version: str = "calc-v1"):
        from decisio.calculator import from_input

        return from_input(self.input, version)


_ID = re.compile(r"(i|f|c|a)([1-9]\d{0,3})|tax2?")
_KIND_ROLES = {  # kind: (required, optional)
    "count": ({"quantity", "price"}, {"unit", "per_unit"}),
    "days": ({"quantity", "price"}, {"unit", "per_unit"}),  # + day_count or range_start/range_end
    "time": ({"duration", "rate"}, {"rounding_step", "per_unit"}),
    "length": ({"quantity", "rate"}, {"rounding_step", "unit", "per_unit"}),
    "convert": ({"quantity", "factor_from", "factor_to", "price"}, {"unit", "per_unit"}),
    "daycredit": ({"price"}, {"per_unit"}),  # + day_count or range_start/range_end
    "fee": ({"price"}, {"quantity", "per_unit", "unit"}),
    "credit": ({"price"}, set()),
    "adjustment": ({"percent"}, set()),
}
_MULTI = {"time": "duration", "length": "quantity"}
_PREFIX_KIND = {"f": "fee", "c": "credit", "a": "adjustment"}
_AFTER_PER = re.compile(r"\s*(?:/\s*|per\s+|an?\s+|each\s+|every\s+)(\w+)", re.I)
_AFTER_HOURLY = re.compile(r"\s*(hourly)\b", re.I)
_BEFORE_PER = re.compile(
    r"(?<![A-Za-z])(hourly|per[-_ ]hour|per[-_ ]minute)(?:[\W_]*rate)?(?:\s+(?:of|is|at))?[\W_]*$", re.I
)


@functools.lru_cache(maxsize=8192)
def norm_label(text: str) -> str:
    return " ".join(text.split()).casefold()


def pick_label(texts: list[str]) -> str:
    """The label a line is written with: its spans must read the same ignoring case and spacing; the first that starts
    with a capital and is not all capitals (a heading), else the first that is not all capitals, else the first."""
    keys = {norm_label(t) for t in texts}
    if len(keys) != 1:
        raise Refusal("conflicting-labels", f"{len(keys)} readings")
    clean = [" ".join(t.split()) for t in texts]
    for ok in (lambda t: t[:1].isupper() and t != t.upper(), lambda t: t != t.upper()):
        for t in clean:
            if ok(t):
                return t
    return clean[0]


def unit_of(word: str) -> str | None:
    """The normalizer's canonical unit for a unit word (`metres` -> `m`, `hour` -> `h`, `pallets` -> `pallet`)."""
    word = word.strip()
    if not word or not re.fullmatch(r"[A-Za-z][A-Za-z ]*", word):
        return None
    vals = N.normalize(f"1 {word}").values
    if len(vals) != 1 or vals[0].span != (0, len(word) + 2):
        return None
    v = vals[0]
    if v.kind == "duration":  # a duration's value is in minutes: the word's size names the unit
        return {Fraction(60): "h", Fraction(1): "min"}.get(Fraction(v.value))
    return v.unit if v.kind == "quantity" else None


def per_unit_word(text: str) -> str | None:
    """The canonical unit of a `per_unit` span (`per kg`, `an hour`, `/m`)."""
    return unit_of(re.sub(r"^(?:per|an?|each|every|/)\s*", "", text.strip(), flags=re.I))


def per_unit_near(text: str, start: int, end: int) -> str | None:
    """The unit a money amount at `text[start:end]` is per, as written next to it (`$59.44 per hour`, `$1.10/km`,
    `hourly rate: $41.00`), canonical; None when none is written."""
    after = text[end : end + 24]
    m = _AFTER_PER.match(after)
    if m:
        u = unit_of(m.group(1))
        if u:
            return u
    if _AFTER_HOURLY.match(after):
        return "h"
    line_start = text.rfind("\n", 0, start) + 1
    b = _BEFORE_PER.search(text[max(line_start, start - 40) : start])
    if b:
        return "min" if "minute" in b.group(1).lower() else "h"
    return None


def _dec(x: Fraction, path: str) -> str:
    for places in range(0, 7):
        scaled = x * 10**places
        if scaled.denominator == 1:
            n = scaled.numerator
            if places == 0:
                return str(n)
            sign, n = ("-", -n) if n < 0 else ("", n)
            whole, frac = divmod(n, 10**places)
            return f"{sign}{whole}.{frac:0{places}d}"
    raise Refusal("inexact", path)


class _Reader:
    """The normalizer's values for the state and the context, with the decimal convention of both together."""

    def __init__(self, state: str, context: str | None):
        self.texts = {"state": state, "context": context or ""}
        doc = N.document(((context + "\n") if context else "") + state)
        self.values = {w: N.normalize(t, doc).values if t else () for w, t in self.texts.items()}

    def check(self, span: TagSpan) -> None:
        src = self.texts.get(span.where)
        if src is None or not (0 <= span.start < span.end <= len(src)) or src[span.start : span.end] != span.text:
            raise Refusal("span-text", f"{span.role} at {span.key}")

    def value(self, span: TagSpan):
        """The one value inside the span; a value cut by the span's ends, or two values, is refused."""
        inside = []
        for v in self.values[span.where]:
            s, e = v.span
            if e <= span.start or s >= span.end:
                continue
            if span.role in ("range_start", "range_end") and v.kind == "date-range":
                inside.append(v)
                continue
            if s < span.start or e > span.end:
                raise Refusal("span-value", f"{span.role} at {span.key}: a value crosses the span's end")
            inside.append(v)
        inside = one_restated(inside, self.texts[span.where])
        if len(inside) != 1:
            raise Refusal("span-value", f"{span.role} at {span.key}: {len(inside)} values")
        v = inside[0]
        if v.kind == "ambiguous":
            raise Refusal("ambiguous-value", f"{span.role} at {span.key}: {v.why}")
        return v


def one_restated(values: list, text: str) -> list:
    """A number in words restated in digits right after it, in parentheses ("thirty-nine (39)"), is one value when the
    two agree; any other pair stays two values."""
    if len(values) == 2:
        a, b = sorted(values, key=lambda v: v.span[0])
        between = text[a.span[1] : b.span[0]]
        after = text[b.span[1] : b.span[1] + 1]
        if (
            a.kind == b.kind == "number"
            and Fraction(a.value) == Fraction(b.value)
            and re.fullmatch(r"\s*\(\s*", between)
            and after == ")"
            and not any(ch.isdigit() for ch in text[a.span[0] : a.span[1]])
        ):
            return [a]
    return values


_ACCEPTS = {
    "quantity": ("number", "quantity"),
    "price": ("money",),
    "rate": ("money",),
    "stated_total": ("money", "number"),
    "duration": ("duration", "number", "quantity"),
    "day_count": ("number", "quantity"),
    "factor_from": ("quantity",),
    "factor_to": ("quantity",),
    "percent": ("percent",),
    "tax_rate": ("percent",),
    "rounding_step": ("duration", "number", "quantity"),
    "range_start": ("date-range",),
    "range_end": ("date-range",),
}


def bare(v) -> bool:
    """A number with no unit of its own (`95`, or a count noun the normalizer attached: `68 added`); its unit comes
    from its line, which the units gate checks."""
    return v.kind == "number" or (v.kind == "quantity" and v.dimension == "count")


def _minutes(v, path: str) -> Fraction:
    if v.kind == "duration" or bare(v):
        return Fraction(v.value)
    f = N.factor(v.unit, "min") if v.dimension == "time" else None
    if f is None:
        raise Refusal("unit", f"{path}: not a time")
    return v.value * f


def _whole_days(v, path: str) -> int:
    if v.kind == "quantity" and v.unit != "day":
        raise Refusal("unit", f"{path}: a day count in {v.dimension}")
    x = Fraction(v.value)
    if x.denominator != 1 or x < 1:
        raise Refusal("span-value", f"{path}: not a whole number of days")
    return int(x)


def _flag_for(flag, group: Group, ids_ok: bool = True):
    """A per-line flag's entry for a group: keyed by the line's label (any case) or its id."""
    if flag is None:
        return None
    if not isinstance(flag, dict):
        raise Refusal("flag-conflict", "a per-line flag is not a mapping")
    hits = [v for k, v in flag.items() if norm_label(str(k)) == norm_label(group.label) or (ids_ok and k == group.id)]
    if len(hits) > 1:
        raise Refusal("flag-conflict", f"{group.id}: named twice")
    return hits[0] if hits else None


def _labels_known(flag, groups: list[Group], kinds: set[str], name: str) -> None:
    if flag is None:
        return
    keys = flag if isinstance(flag, (list, tuple)) else list(flag)
    for k in keys:
        if not any(g.kind in kinds and (norm_label(str(k)) == norm_label(g.label) or k == g.id) for g in groups):
            raise Refusal("flag-conflict", f"{name} names no line of the right kind")


def assemble(tags: Tags, state: str, context: str | None = None) -> Assembly:
    """The `calculator_input` the tags describe, with where every operand came from; or `Refusal`."""
    rd = _Reader(state, context)
    spans = sorted(tags.spans, key=lambda s: (0 if s.where == "context" else 1, s.start, s.end))
    for s in spans:
        if s.role not in ROLES:
            raise Refusal("unknown-role", f"at {s.key}")
        rd.check(s)
    for a, b in zip(spans, spans[1:]):
        if a.where == b.where and b.start < a.end:
            raise Refusal("span-overlap", f"{a.key} and {b.key}")

    spans = _without_lone_range_ends(spans, rd)
    by_group: dict[str, list[tuple[str, TagSpan, object]]] = {}
    stated: list[tuple[TagSpan, Fraction]] = []
    money_marks: set[str] = set()
    for s in spans:
        if s.role in FREE_ROLES:
            if s.item is not None:
                raise Refusal("role-without-item", f"{s.role} at {s.key} names a line")
            if s.role == "stated_total":
                v = rd.value(s)
                if v.kind not in _ACCEPTS["stated_total"]:
                    raise Refusal("span-value", f"stated_total at {s.key}: a {v.kind}")
                if v.kind == "money":
                    money_marks.add(v.currency)
                stated.append((s, Fraction(v.value)))
            continue
        if not s.item:
            raise Refusal("role-without-item", f"{s.role} at {s.key}")
        if not _ID.fullmatch(s.item):
            raise Refusal("unknown-item", f"at {s.key}")
        v = None
        if s.role not in WORD_ROLES:
            v = rd.value(s)
            if v.kind not in _ACCEPTS[s.role]:
                raise Refusal("span-value", f"{s.role} at {s.key}: a {v.kind}")
            if v.kind == "money":
                money_marks.add(v.currency)
        by_group.setdefault(s.item, []).append((s.role, s, v))

    # every money amount in the text, inside a span or not, is in one currency
    for w in rd.values:
        for v in rd.values[w]:
            if v.kind == "money":
                money_marks.add(v.currency)
    if len(money_marks) > 1:
        raise Refusal("two-currencies", f"{len(money_marks)} marks")
    if not money_marks:
        raise Refusal("no-currency")
    currency = money_marks.pop()

    if "tax2" in by_group:
        raise Refusal("two-tax-rates", "a second rate is tagged")
    tax_vals = by_group.pop("tax", None)
    if not tax_vals:
        raise Refusal("no-tax")
    if any(role != "tax_rate" for role, _, _ in tax_vals):
        raise Refusal("item-roles", "the tax group holds another role")
    if any(role == "tax_rate" for g in by_group.values() for role, _, _ in g):
        raise Refusal("role-without-item", "a tax rate inside a line")
    if len({v.value for _, _, v in tax_vals}) != 1:
        raise Refusal("two-tax-rates", f"{len({v.value for _, _, v in tax_vals})} values")
    tax_rate = tax_vals[0][2].value

    flags = thaw(tags.flags)
    groups: list[Group] = []
    for gid, vals in sorted(by_group.items(), key=lambda kv: _first(kv[1])):
        labels = [s.text for role, s, _ in vals if role == "item_label"]
        if not labels:
            raise Refusal("no-label", gid)
        label = pick_label(labels)
        roles = [(role, s, v) for role, s, v in vals if role != "item_label"]
        kind = _kind(gid, {r for r, _, _ in roles})
        groups.append(Group(gid, kind, label, tuple(roles)))

    operands: list[Operand] = [Operand("tax.rate", Fraction(tax_rate), tuple(s for _, s, _ in tax_vals))]
    items, fees, credits, adjustments = [], [], [], []
    out_groups = []
    for g in groups:
        g, entry = _line(g, flags, rd, operands)
        out_groups.append(g)
        if g.kind == "fee":
            entry_path_fix(entry, operands, f"fees[{len(fees)}]")
            fees.append(entry)
        elif g.kind == "credit":
            entry_path_fix(entry, operands, f"credits[{len(credits)}]")
            credits.append(entry)
        elif g.kind == "adjustment":
            entry_path_fix(entry, operands, f"adjustments[{len(adjustments)}]")
            adjustments.append(entry)
        else:
            entry_path_fix(entry, operands, f"items[{len(items)}]")
            items.append(entry)
    groups = out_groups

    # adjustments: the target is a priced item, named by id
    for adj, g in zip(adjustments, [g for g in groups if g.kind == "adjustment"]):
        targets = {s.target for role, s, _ in g.values if role == "percent"}
        if len(targets) != 1 or None in targets:
            raise Refusal("target", f"{g.id}: {len(targets - {None})} targets")
        tid = targets.pop()
        hit = [x for x in groups if x.id == tid and x.kind in ("count", "days", "time", "length", "convert")]
        if not hit:
            raise Refusal("target", f"{g.id}: {tid} is not a priced item")
        adj["of"] = hit[0].label
        direction = _flag_for(flags.get("adjustment_direction"), g)
        if direction not in ("add", "deduct"):
            raise Refusal("missing-flag", f"adjustment_direction for {g.id}")
        adj["direction"] = direction
    _labels_known(flags.get("adjustment_direction"), groups, {"adjustment"}, "adjustment_direction")

    tax_kind = flags.get("tax_kind")
    if tax_kind not in ("charged", "withheld"):
        raise Refusal("missing-flag", "tax_kind")
    if "tax_after" not in flags or not isinstance(flags["tax_after"], list):
        raise Refusal("missing-flag", "tax_after")
    movable = {"fee", "credit", "daycredit"}
    _labels_known(flags["tax_after"], groups, movable, "tax_after")
    outside = [g.label for g in groups if g.kind in movable and _named(flags["tax_after"], g)]
    cp = flags.get("credit_position")
    if cp is not None:
        _labels_known(cp, groups, movable, "credit_position")
        for g in groups:
            pos = _flag_for(cp, g)
            if pos is None:
                continue
            if pos not in ("before_tax", "after_tax") or (pos == "after_tax") != (g.label in outside):
                raise Refusal("flag-conflict", f"credit_position and tax_after disagree for {g.id}")
    _labels_known(flags.get("time_rounding"), groups, {"time"}, "time_rounding")
    _labels_known(flags.get("length_rounding"), groups, {"length"}, "length_rounding")

    data = {"currency": currency, "items": items}
    if adjustments:
        data["adjustments"] = adjustments
    if fees:
        data["fees"] = fees
    if credits:
        data["credits"] = credits
    data["tax"] = {"rate": _dec(Fraction(tax_rate), "tax.rate"), "kind": tax_kind}
    if outside:
        data["tax"]["outside"] = outside
    asm = Assembly(
        input=data,
        operands=tuple(operands),
        groups=tuple(groups),
        stated=tuple(stated),
        spans=tuple(spans),
        texts=tuple((w, t) for w, t in rd.texts.items() if t or w == "state"),
    )
    from decisio.calculator import CalculatorInputError

    try:
        asm.block()
    except CalculatorInputError as e:
        raise Refusal("invalid-input", str(e).split(":", 1)[0]) from None
    return asm


_PER_DAY = re.compile(r"\b(?:per|a|each) day\b|\bdaily\b|\bper (?:night|week)\b", re.I)


def _line_of(text: str, start: int) -> tuple[int, int]:
    a = text.rfind("\n", 0, start) + 1
    b = text.find("\n", start)
    return a, len(text) if b < 0 else b


def _without_lone_range_ends(spans, rd):
    """A range start or end with no partner in its line, on a text line of its own (a header's "Date issued: 14 March
    2025"), where no line of the group's other spans prices by the day, prices nothing: it is left out before any value
    is read (its digits stay covered by the span for the numbers gate, which reads the tags). Anything else is kept,
    and refused later if it does not make a range."""
    groups: dict[str, list[TagSpan]] = {}
    for x in spans:
        if x.item:
            groups.setdefault(x.item, []).append(x)
    drop = set()
    for members in groups.values():
        ends = [x for x in members if x.role in ("range_start", "range_end")]
        rest = [x for x in members if x.role not in ("range_start", "range_end")]
        if len(ends) != 1 or not rest:
            continue
        lone = ends[0]
        own = _line_of(rd.texts[lone.where], lone.start)
        keep = False
        for x in rest:
            line = _line_of(rd.texts[x.where], x.start)
            if (x.where == lone.where and line == own) or _PER_DAY.search(rd.texts[x.where][line[0] : line[1]]):
                keep = True
                break
        if not keep:
            drop.add(id(lone))
    return [x for x in spans if id(x) not in drop]


def _first(vals) -> tuple:
    s = min((sp for _, sp, _ in vals), key=lambda sp: (0 if sp.where == "context" else 1, sp.start))
    return (0 if s.where == "context" else 1, s.start)


def _named(names, g: Group) -> bool:
    return any(norm_label(str(n)) == norm_label(g.label) or n == g.id for n in names)


def _kind(gid: str, roles: set[str]) -> str:
    if gid[0] in _PREFIX_KIND:
        kind = _PREFIX_KIND[gid[0]]
    elif "factor_from" in roles or "factor_to" in roles:
        kind = "convert"
    elif "duration" in roles:
        kind = "time"
    elif "rate" in roles:
        kind = "length"
    elif "day_count" in roles or "range_start" in roles or "range_end" in roles:
        kind = "days" if "quantity" in roles else "daycredit"
    else:
        kind = "count"
    required, optional = _KIND_ROLES[kind]
    allowed = required | optional
    if kind in ("days", "daycredit"):
        days = {"day_count"} if "day_count" in roles else {"range_start", "range_end"}
        required, allowed = required | days, allowed | days
    if not required <= roles or not roles <= allowed:
        raise Refusal(
            "item-roles", f"{gid} as {kind}: {sorted(required - roles)} missing, {sorted(roles - allowed)} extra"
        )
    return kind


def entry_path_fix(entry: dict, operands: list[Operand], path: str) -> None:
    """Operands of a line are recorded under a placeholder path until the line's place in the input is known."""
    for i, op in enumerate(operands):
        if op.path.startswith("@."):
            operands[i] = Operand(path + op.path[1:], op.value, op.spans, op.how)


def _one(g: Group, role: str):
    """The single value of a role in a line: every mention must read the same (value and unit)."""
    got = [(s, v) for r, s, v in g.values if r == role]
    if not got:
        return None, ()
    values = {Fraction(v.value) for _, v in got}
    units = {v.unit for _, v in got if v.kind == "quantity" and v.dimension != "count"}
    if len(values) != 1 or len(units) > 1:
        raise Refusal("conflicting-values", f"{g.id} {role}: {len(values)} values, {len(units)} units")
    best = next((v for _, v in got if v.kind == "quantity" and v.dimension != "count"), got[0][1])
    return best, tuple(s for s, _ in got)


def _magnitude(g: Group):
    """A credit's amount: every mention must read the same magnitude; a minus sign on one of them is the direction the
    line already has (taken off), so the amount is the magnitude (`magnitude`, re-read so by the provenance gate)."""
    got = [(s, v) for r, s, v in g.values if r == "price"]
    if not got:
        return None, (), "same"
    mags = {abs(Fraction(v.value)) for _, v in got}
    if len(mags) != 1:
        raise Refusal("conflicting-values", f"{g.id} price: {len(mags)} values")
    signed = any(Fraction(v.value) < 0 for _, v in got)
    best = next((v for _, v in got if Fraction(v.value) > 0), got[0][1])
    return best, tuple(s for s, _ in got), "magnitude" if signed else "same"


def _entries(g: Group, role: str):
    got = [(s, v) for r, s, v in g.values if r == role]
    vals = [Fraction(v.value) for _, v in got]
    if len(set(vals)) != len(vals):
        raise Refusal("repeated-entry", f"{g.id} {role}")
    return got


def _range_days(g: Group, flags: dict, path: str, operands: list[Operand]) -> int:
    starts = [(s, v) for r, s, v in g.values if r == "range_start"]
    ends = [(s, v) for r, s, v in g.values if r == "range_end"]
    if len(starts) != 1 or len(ends) != 1 or starts[0][1] is not ends[0][1]:
        raise Refusal("range", f"{g.id}: the two ends are not one stated range")
    rule = flags.get("range_last_day")
    if isinstance(rule, dict):
        rule = _flag_for(rule, g)
    if rule not in ("counts", "excluded"):
        raise Refusal("missing-flag", f"range_last_day for {g.id}")
    days = starts[0][1].days(inclusive=rule == "counts")
    if days is None or days < 1:
        raise Refusal("range", f"{g.id}: the day count depends on the year or is not positive")
    operands.append(Operand(path, Fraction(days), (starts[0][0], ends[0][0]), "range-days"))
    return days


def _line(g: Group, flags: dict, rd: _Reader, operands: list[Operand]):
    """One line's input entry; its operands are recorded with the placeholder path `@.field`."""

    def add(field: str, v, spans, how="same", value=None) -> Fraction:
        x = Fraction(v.value) if value is None else value
        operands.append(Operand(f"@.{field}", x, tuple(spans), how))
        return x

    k = g.kind
    entry: dict = {"label": g.label}
    if k in ("count", "days", "convert", "daycredit"):
        entry = {"kind": k, "label": g.label}
    if k in ("count", "days", "convert"):
        q, qs = _one(g, "quantity")
        entry["quantity"] = _dec(add("quantity", q, qs), f"{g.id}.quantity")
        p, ps = _one(g, "price")
        entry["price"] = _dec(add("price", p, ps), f"{g.id}.price")
        if k == "convert":
            if q.kind != "quantity":
                raise Refusal("unit", f"{g.id}: the converted quantity has no unit")
            ff, ffs = _one(g, "factor_from")
            ft, fts = _one(g, "factor_to")
            if Fraction(ff.value) != 1:
                raise Refusal("span-value", f"{g.id}: the factor's left side is not 1")
            add("factor_from", ff, ffs)
            entry["unit"], entry["to_unit"] = q.unit, ft.unit
            entry["factor"] = _dec(add("factor", ft, fts), f"{g.id}.factor")
        if k == "days":
            entry["days"] = _days(g, flags, operands)
    elif k == "daycredit":
        p, ps = _one(g, "price")
        entry["price"] = _dec(add("price", p, ps), f"{g.id}.price")
        entry["days"] = _days(g, flags, operands)
    elif k in ("time", "length"):
        role = _MULTI[k]
        got = _entries(g, role)
        rate, rs = _one(g, "rate")
        entry = {"kind": k, "label": g.label}
        if k == "time":
            mins = []
            for i, (s, v) in enumerate(got):
                m = _minutes(v, f"{g.id}.minutes[{i}]")
                operands.append(Operand(f"@.minutes[{i}]", m, (s,), "minutes"))
                mins.append(_dec(m, f"{g.id}.minutes[{i}]"))
            entry["minutes"] = mins
            entry["rate"] = _dec(add("rate", rate, rs), f"{g.id}.rate")
            pers = {per_unit_near(rd.texts[s.where], s.start, s.end) for s in rs} - {None}
            pers |= {per_unit_word(s.text) for r, s, _ in g.values if r == "per_unit"} - {None}
            if len(pers) != 1 or next(iter(pers)) not in ("h", "min"):
                raise Refusal("unit", f"{g.id}: the rate's time unit is not stated once")
            entry["per"] = next(iter(pers))
            g = Group(g.id, g.kind, g.label, g.values, entry["per"])
        else:
            units = {v.unit for _, v in got if not bare(v)}
            if any(bare(v) for _, v in got):  # bare lengths are in the unit the rate is per
                units |= {per_unit_near(rd.texts[s.where], s.start, s.end) for s in rs} - {None}
            if len(units) != 1:
                raise Refusal("unit", f"{g.id}: the lengths are not all in one stated unit")
            unit = units.pop()
            lens = []
            for i, (s, v) in enumerate(got):
                operands.append(Operand(f"@.lengths[{i}]", Fraction(v.value), (s,)))
                lens.append(_dec(Fraction(v.value), f"{g.id}.lengths[{i}]"))
            entry["lengths"], entry["unit"] = lens, unit
            entry["rate"] = _dec(add("rate", rate, rs), f"{g.id}.rate")
        rounding = _flag_for(flags.get(f"{k}_rounding"), g)
        step_v, step_s = _one(g, "rounding_step")
        if (rounding is None) != (step_v is None):
            raise Refusal("missing-flag" if rounding is None else "flag-conflict", f"{k}_rounding for {g.id}")
        if rounding is not None:
            if not isinstance(rounding, dict) or rounding.get("mode") not in ("each", "total"):
                raise Refusal("missing-flag", f"{k}_rounding mode for {g.id}")
            if k == "time":
                step, how = _minutes(step_v, f"{g.id}.round.step"), "minutes"
                stated = rounding.get("step_min")
            else:
                step, how = _length_step(step_v, entry["unit"], g.id)
                stated = rounding.get("step")
            if stated is not None and Fraction(str(stated)) != step:
                raise Refusal("flag-conflict", f"{k}_rounding step for {g.id}")
            operands.append(Operand("@.round.step", step, step_s, how))
            entry["round"] = {"step": _dec(step, f"{g.id}.round.step"), "mode": rounding["mode"]}
    elif k == "fee":
        p, ps = _one(g, "price")
        entry["price"] = _dec(add("price", p, ps), f"{g.id}.price")
        c, cs = _one(g, "quantity")
        if c is not None:
            entry["count"] = _dec(add("count", c, cs), f"{g.id}.count")
    elif k == "credit":
        p, ps, how = _magnitude(g)
        mag = None if p is None or Fraction(p.value) > 0 else abs(Fraction(p.value))  # the positive mention as written
        entry["amount"] = _dec(add("amount", p, ps, how=how, value=mag), f"{g.id}.amount")
    elif k == "adjustment":
        p, ps = _one(g, "percent")
        entry["percent"] = _dec(add("percent", p, ps), f"{g.id}.percent")
        entry["of"], entry["direction"] = None, None
    return g, entry


def _days(g: Group, flags: dict, operands: list[Operand]) -> int:
    d, ds = _one(g, "day_count")
    if d is None:
        return _range_days(g, flags, "@.days", operands)
    days = _whole_days(d, f"{g.id}.days")
    operands.append(Operand("@.days", Fraction(days), ds))
    return days


def _length_step(v, unit: str, gid: str) -> tuple[Fraction, str]:
    if v.kind != "quantity":
        raise Refusal("unit", f"{gid}: the rounding step has no length unit")
    if v.unit == unit:
        return Fraction(v.value), "same"
    f = N.factor(v.unit, unit)
    if f is None:
        raise Refusal("unit", f"{gid}: the rounding step's unit does not convert exactly")
    return v.value * f, "unit-factor"
