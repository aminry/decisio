# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Typed values in text, with their character offsets: money, percentages, numbers, durations, quantities, dates.

A deterministic reader with no model and no dependency beyond the standard library. It finds every value it
recognises in a text and returns it as a frozen dataclass with exact `Fraction` values and its `(start, end)` span in
the original text, in text order. A form that can be read two ways is never guessed: it is returned as `Ambiguous`,
with the readings it could have.

  normalize(text, doc=None) -> Normalized              the values in text order; `.ambiguous` lists the unsure ones
  document(text) -> Context                            what a whole document settles for its parts: the decimal
                                                       convention and the currencies next to numbers
  decimal_convention(text) -> "point" | "comma" | "mixed" | None
  canonical(text, doc=None, words=False) -> (text, ambiguous)
      the text with numbers spelled the way the calculator's reader reads them (below)
  dimension(unit) -> "mass" | "length" | "volume" | "area" | "time" | None
  factor(from_unit, to_unit) -> Fraction | None        an exact definitional factor, or None (none, or not certain)

Value kinds (`kind`):

  money       `$1,234.56`, `EUR 12.50`, `12,50 €`, `€1.234,56`, `1.234,56 EUR`, `-$40.00`, `$-40.00`, `$1.2M`,
              `$3 million`, `twenty dollars`; `currency` is the mark as written (`$`, `€`, `EUR`)
  percent     `12.5%`, `12,5 %`, `20 percent`, `twenty per cent`
  number      `1,234`, `0.4536`, `seven`, `twenty-one`, `one hundred and five`, `a dozen`, `two dozen`, `half`,
              `one and a half`, `three quarters`
  duration    `9 h 28 min`, `1 h 52 min`, `45 minutes`, `1.5 hours`, `2 hours and 15 minutes`, `9h28`, `1:30 h`,
              `half an hour`, `quarter hour`; `value` is in minutes
  quantity    a number with a unit: mass, length, volume, area and time units (`980 lb`, `3.57 m`, `26.1 L`,
              `5 days`), or a count noun (`4 pallets`, `two crews`); `unit` is canonical (`kg`, `L`, `day`) or the
              singular noun, `dimension` is the unit's dimension or `count`
  time-of-day `10:30 am`, `at 14:00`; `value` is minutes after midnight
  date        `2026-11-18`, `18 November 2026`, `18th Nov`, `Nov 18`, `November 18, 2026`, `13/04/2026`;
              `year` is None when the text gives none
  date-range  `27 Mar to 2 Apr`, `27-30 March`, `Mar 27 to Apr 2`, `between 3 May and 9 May`
  ambiguous   a form with two readings or none; `why` says which, `readings` holds the candidates

The decimal convention. `1,234` is 1234 with a decimal point and 1.234 with a decimal comma, and `1.234` the other way
round; `12,50`, `1.234,56`, `0,5` and `1234,567` can only be comma decimals, and `12.50`, `1,234.56` and `0.125` only
point decimals. A text that has unambiguous numbers of one convention only settles its ambiguous ones by it; a text
with none, or with both (`mixed`), leaves them `Ambiguous`.

Other forms that are never guessed:
  - a numeric date whose day and month could swap (`03/04/2026`; `13/04/2026` and `04/04/2026` are certain), a
    two-digit year, a slash form with no year (`3/4`: a fraction or a date), a date that is not in the calendar;
  - `1:30` with no unit and no clock word (a duration or a time of day);
  - a currency mark between two numbers (`5 € 10`), an amount in parentheses (`($12.50)`: negative, or an aside),
    `pounds` after a number (money or mass), unless the document's money is in another currency only (a mass then);
  - a range or a difference of numbers (`3-5`), a run of separators that is no number (`1.2.3`, `8,10,12`);
  - units whose size is not certain (`gal`, `ton`, `fl oz`) keep their dimension, but `factor` gives None for them.

`canonical` is what the calculator's reader uses (`decisio.calculator.parse`, `normalized=True`): with a decimal
comma, every number with a separator is written with a decimal point and no thousands separator (`1.234,56` becomes
`1234.56`); a currency mark written after an amount moves before it (`12,50 €` becomes `€12.50`, `12.50 EUR`
becomes `EUR 12.50`); `20 percent` becomes `20%`; with `words=True`, whole numbers in words become digits (`seven`
becomes `7`, `a dozen` becomes `12`). Negative amounts, amounts with a multiplier (`$1.2M`) and fractions in words are
left as written.

Units: the table here is the source of truth, so the output does not depend on what is installed. `pint_check()`
compares it with `pint` when `pint` is installed (the `calculator` extra) and returns the disagreements.
"""

from __future__ import annotations

import datetime as dt
import functools
import re
from dataclasses import dataclass, field
from fractions import Fraction

__all__ = [
    "Ambiguous",
    "Date",
    "DateRange",
    "Duration",
    "Money",
    "Normalized",
    "Context",
    "Number",
    "Percent",
    "Quantity",
    "TimeOfDay",
    "canonical",
    "decimal_convention",
    "document",
    "dimension",
    "factor",
    "normalize",
    "pint_check",
]

Span = tuple[int, int]


# --- value kinds ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Money:
    value: Fraction
    currency: str  # the mark as written: "$", "€", "EUR"
    span: Span
    kind: str = field(default="money", init=False)


@dataclass(frozen=True)
class Percent:
    value: Fraction  # 12.5 for 12.5%
    span: Span
    kind: str = field(default="percent", init=False)


@dataclass(frozen=True)
class Number:
    value: Fraction
    span: Span
    words: bool = False  # written in words
    kind: str = field(default="number", init=False)


@dataclass(frozen=True)
class Duration:
    value: Fraction  # minutes
    span: Span
    unit: str = field(default="min", init=False)
    kind: str = field(default="duration", init=False)


@dataclass(frozen=True)
class Quantity:
    value: Fraction
    unit: str  # canonical unit ("kg", "L", "day") or the singular count noun ("pallet")
    dimension: str  # "mass", "length", "volume", "area", "time" or "count"
    span: Span
    kind: str = field(default="quantity", init=False)


@dataclass(frozen=True)
class TimeOfDay:
    value: Fraction  # minutes after midnight
    span: Span
    kind: str = field(default="time-of-day", init=False)


@dataclass(frozen=True)
class Date:
    year: int | None
    month: int
    day: int
    span: Span
    kind: str = field(default="date", init=False)


@dataclass(frozen=True)
class DateRange:
    start: Date
    end: Date
    span: Span
    kind: str = field(default="date-range", init=False)

    def days(self, inclusive: bool) -> int | None:
        """The number of days the range covers, or None when it depends on a year the text does not give (a 29
        February may fall inside) or the range ends before it starts."""
        a, b = self.start, self.end

        def count(ya: int, yb: int) -> int | None:
            try:
                da, db = dt.date(ya, a.month, a.day), dt.date(yb, b.month, b.day)
            except ValueError:
                return None
            return None if db < da else (db - da).days + (1 if inclusive else 0)

        def years(guess: int | None) -> tuple[int, int]:
            ya, yb = a.year, b.year
            if ya is None and yb is None:
                ya = guess
            if ya is None:
                ya = yb if (a.month, a.day) <= (b.month, b.day) else yb - 1
            if yb is None:
                yb = ya if (a.month, a.day) <= (b.month, b.day) else ya + 1
            return ya, yb

        if a.year is not None or b.year is not None:
            return count(*years(None))
        got = {count(*years(g)) for g in (2023, 2024)}
        return got.pop() if len(got) == 1 else None


@dataclass(frozen=True)
class Ambiguous:
    span: Span
    why: str
    readings: tuple = ()  # the candidate values, each with the same span; empty when the form reads as nothing
    kind: str = field(default="ambiguous", init=False)


Value = Money | Percent | Number | Duration | Quantity | TimeOfDay | Date | DateRange | Ambiguous


@dataclass(frozen=True)
class Context:
    """What the whole document settles for each of its parts: the decimal convention and the currencies it uses."""

    convention: str | None  # "point", "comma", "mixed" or None
    currencies: frozenset[str]  # marks next to numbers: "$", "€", "EUR"; "£" also for `sterling`


@dataclass(frozen=True)
class Normalized:
    text: str
    values: tuple  # Value, in text order, spans never overlapping
    convention: str | None  # "point", "comma", "mixed" or None

    @property
    def ambiguous(self) -> tuple[Ambiguous, ...]:
        return tuple(v for v in self.values if isinstance(v, Ambiguous))


# --- units ------------------------------------------------------------------------------------------------------

# canonical unit -> (dimension, size in the dimension's base unit, or None when the size is not certain)
# bases: g, m, L, m2, min. The sizes are the definitions (1 lb = 0.45359237 kg, 1 in = 2.54 cm, 1 mi = 1609.344 m).
_UNITS: dict[str, tuple[str, Fraction | None]] = {
    "mg": ("mass", Fraction(1, 1000)),
    "g": ("mass", Fraction(1)),
    "kg": ("mass", Fraction(1000)),
    "t": ("mass", Fraction(10**6)),
    "lb": ("mass", Fraction("453.59237")),
    "oz": ("mass", Fraction("453.59237") / 16),
    "ton": ("mass", None),  # short, long or metric
    "mm": ("length", Fraction(1, 1000)),
    "cm": ("length", Fraction(1, 100)),
    "m": ("length", Fraction(1)),
    "km": ("length", Fraction(1000)),
    "in": ("length", Fraction("0.0254")),
    "ft": ("length", Fraction("0.3048")),
    "yd": ("length", Fraction("0.9144")),
    "mi": ("length", Fraction("1609.344")),
    "ml": ("volume", Fraction(1, 1000)),
    "cl": ("volume", Fraction(1, 100)),
    "L": ("volume", Fraction(1)),
    "m3": ("volume", Fraction(1000)),
    "gal": ("volume", None),  # US or imperial
    "fl oz": ("volume", None),  # US or imperial
    "m2": ("area", Fraction(1)),
    "s": ("time", Fraction(1, 60)),
    "min": ("time", Fraction(1)),
    "h": ("time", Fraction(60)),
    "day": ("time", Fraction(1440)),
    "week": ("time", Fraction(10080)),
    "month": ("time", None),  # calendar lengths vary
    "year": ("time", None),
}
_UNIT_NAMES = {
    "mg": "mg milligram milligrams",
    "g": "g gram grams",
    "kg": "kg kgs kilogram kilograms kilo kilos",
    "t": "t tonne tonnes",
    "lb": "lb lbs pound pounds",
    "oz": "oz ounce ounces",
    "ton": "ton tons",
    "mm": "mm millimetre millimetres millimeter millimeters",
    "cm": "cm centimetre centimetres centimeter centimeters",
    "m": "m metre metres meter meters",
    "km": "km kilometre kilometres kilometer kilometers",
    "in": "inch inches",
    "ft": "ft foot feet",
    "yd": "yd yard yards",
    "mi": "mi mile miles",
    "ml": "ml mL millilitre millilitres milliliter milliliters",
    "cl": "cl cL centilitre centilitres centiliter centiliters",
    "L": "L l litre litres liter liters",
    "m3": "m3 m³",
    "gal": "gal gallon gallons",
    "m2": "m2 m²",
    "s": "s sec secs second seconds",
    "min": "min mins minute minutes",
    "h": "h hr hrs hour hours",
    "day": "day days",
    "week": "week weeks wk wks",
    "month": "month months",
    "year": "year years yr yrs",
}
_UNIT = {name: unit for unit, names in _UNIT_NAMES.items() for name in names.split()}


def dimension(unit: str) -> str | None:
    """The dimension of a unit name or canonical unit (`"lb"`, `"pounds"`, `"L"`), or None for one the table lacks."""
    u = unit if unit in _UNITS else _UNIT.get(unit) or ("fl oz" if unit == "fl oz" else None)
    return _UNITS[u][0] if u else None


def factor(from_unit: str, to_unit: str) -> Fraction | None:
    """How many `to_unit` one `from_unit` is, exactly; None across dimensions or when either size is not certain."""
    a, b = (u if u in _UNITS else _UNIT.get(u) for u in (from_unit, to_unit))
    if a is None or b is None or _UNITS[a][0] != _UNITS[b][0]:
        return None
    sa, sb = _UNITS[a][1], _UNITS[b][1]
    if sa is None or sb is None:
        return None
    return sa / sb


_PINT_NAMES = {
    "mg": "milligram",
    "g": "gram",
    "kg": "kilogram",
    "t": "metric_ton",
    "lb": "pound",
    "oz": "ounce",
    "mm": "millimeter",
    "cm": "centimeter",
    "m": "meter",
    "km": "kilometer",
    "in": "inch",
    "ft": "foot",
    "yd": "yard",
    "mi": "mile",
    "ml": "milliliter",
    "cl": "centiliter",
    "L": "liter",
    "m3": "meter**3",
    "m2": "meter**2",
    "s": "second",
    "min": "minute",
    "h": "hour",
    "day": "day",
    "week": "week",
}
_BASES = {"mass": "gram", "length": "meter", "volume": "liter", "area": "meter**2", "time": "minute"}


def pint_check() -> list[str] | None:
    """The units on which the table disagrees with `pint` (dimension, or size beyond 1e-12 relative), or None when
    `pint` is not installed. The table stays the source of truth either way."""
    try:
        import pint
    except ImportError:
        return None
    ureg = pint.UnitRegistry()
    bad = []
    for unit, (dim, size) in sorted(_UNITS.items()):
        name = _PINT_NAMES.get(unit)
        if name is None or size is None:
            continue
        q = ureg.Quantity(1, name)
        base = ureg.Quantity(1, _BASES[dim])
        if q.dimensionality != base.dimensionality:
            bad.append(f"{unit}: dimension")
            continue
        got = q.to(_BASES[dim]).magnitude
        if abs(got - float(size)) > 1e-12 * float(size):
            bad.append(f"{unit}: {got} vs {float(size)}")
    return bad


# --- lexical tables ---------------------------------------------------------------------------------------------

SYMBOLS = "$€£¥"
CODES = (
    "USD EUR GBP JPY CHF CAD AUD NZD SEK NOK DKK ISK INR CNY HKD SGD ZAR MXN BRL PLN CZK HUF RON TRY ILS AED SAR KRW"
).split()
_MINUS = "-−"
_DASHES = "-–"  # hyphen and en dash between the ends of a range
_CUR_WORDS = {"dollar": "$", "dollars": "$", "euro": "€", "euros": "€"}
_MULTIPLIERS = {"k": 1000, "K": 1000, "m": 10**6, "M": 10**6, "bn": 10**9}
_MULTIPLIER_WORDS = {"thousand": 1000, "million": 10**6, "billion": 10**9}

_UNITS_0 = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen"
).split()
_SMALL = {w: i for i, w in enumerate(_UNITS_0)}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"thousand": 1000, "million": 10**6}
_FRACTION_WORDS = {"half": 2, "halves": 2, "third": 3, "thirds": 3, "quarter": 4, "quarters": 4}

_MONTHS = {}
for _i, _names in enumerate(
    [
        "Jan January",
        "Feb February",
        "Mar March",
        "Apr April",
        "May",
        "Jun June",
        "Jul July",
        "Aug August",
        "Sep Sept September",
        "Oct October",
        "Nov November",
        "Dec December",
    ]
):
    for _n in _names.split():
        _MONTHS[_n] = _i + 1
_MONTH = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"

# words after a number that are not its count noun
_NOT_NOUNS = set(
    """at a an the and or nor of for to in on per each x by from with is are was were be been as than more less if but
    so then plus minus times percent off out over under up down into onto it its this that these those which who
    every all any no not only about around after before between until through extra other also same such their his
    her our your my we they he she you i will would can could may might must should has have had do does did per
    cent hundred thousand million billion dozen half quarter third am pm""".split()
)
_CLOCK_WORDS = {"at", "from", "until", "till", "by", "before", "after", "opens", "closes", "starts", "ends", "to"}
_CLOCK_WORDS |= set(
    "monday tuesday wednesday thursday friday saturday sunday mon tue tues wed weds thu thur thurs".split()
)
_CLOCK_WORDS |= {"fri", "sat", "sun"}


# --- number atoms -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Atom:
    start: int
    end: int
    readings: tuple[Fraction, ...]  # one reading; or (point reading, comma reading); empty: no number
    words: bool = False  # written (partly) in words
    integer_words: bool = False  # words that spell a whole number, which `canonical` may write as digits
    evidence: str | None = None  # "point" or "comma" when the form settles the convention
    why: str = ""  # why there is not exactly one reading
    separators: bool = False  # digits written with "." or ","


def _classify_digits(s: str) -> tuple[tuple[Fraction, ...], str | None, str]:
    """A digit run's readings: (readings, evidence, why). Two readings are (point reading, comma reading)."""
    if "." not in s and "," not in s:
        return (Fraction(s),), None, ""
    if s.startswith("."):
        return (Fraction("0" + s),), "point", ""
    seps = [c for c in s if c in ".,"]
    parts = re.split(r"[.,]", s)
    if len(set(seps)) == 2:
        dec, group = seps[-1], seps[0]
        if any(c != group for c in seps[:-1]):
            return (), None, "separators that form no number"
        if not (1 <= len(parts[0]) <= 3 and all(len(g) == 3 for g in parts[1:-1])):
            return (), None, "separators that form no number"
        return (Fraction("".join(parts[:-1]) + "." + parts[-1]),), ("point" if dec == "." else "comma"), ""
    sep = seps[0]
    if len(seps) > 1:  # only thousands groups: 1,234,567 or 1.234.567
        if 1 <= len(parts[0]) <= 3 and all(len(g) == 3 for g in parts[1:]):
            return (Fraction("".join(parts)),), ("point" if sep == "," else "comma"), ""
        return (), None, "separators that form no number"
    a, b = parts
    as_decimal = Fraction(a + "." + b)
    if len(b) != 3 or len(a) > 3 or a.startswith("0"):
        return (as_decimal,), ("point" if sep == "." else "comma"), ""
    grouped = Fraction(a + b)
    if sep == ",":
        return (grouped, as_decimal), None, "a separator that groups thousands or marks decimals"
    return (as_decimal, grouped), None, "a separator that groups thousands or marks decimals"


_DIGITS = re.compile(r"(?<![\w.,])(?:\d+(?:[.,]\d+)*|\.\d+)")
_WORD = re.compile(r"[A-Za-z]+")
_DOZEN = re.compile(r" dozen(?![\w])")


def _digit_atoms(text: str, taken: list[Span]) -> list[_Atom]:
    out = []
    for m in _DIGITS.finditer(text):
        s, e = m.span()
        if _inside(s, e, taken):
            continue
        if re.match(r"(?:st|nd|rd|th)(?![\w])", text[e:]):
            continue  # an ordinal
        if re.match(r"[A-Za-z]*\d", text[e:]):
            continue  # part of a code such as 4x4
        if s >= 2 and text[s - 1] in _DASHES and text[s - 2].isalnum():
            continue  # inside an identifier
        raw = m.group(0)
        readings, evidence, why = _classify_digits(raw)
        sep = "." in raw or "," in raw
        dz = _DOZEN.match(text, e)
        if dz and len(readings) == 1 and readings[0].denominator == 1:
            out.append(_Atom(s, dz.end(), (readings[0] * 12,), True, True, evidence, "", sep))
            continue
        out.append(_Atom(s, e, readings, evidence=evidence, why=why, separators=sep))
    return out


def _under_100(toks: list[str], i: int) -> tuple[int | None, int]:
    t = toks[i] if i < len(toks) else None
    if t in _SMALL:
        return _SMALL[t], i + 1
    if t in _TENS:
        if i + 1 < len(toks) and toks[i + 1] in _SMALL and 1 <= _SMALL[toks[i + 1]] <= 9:
            return _TENS[t] + _SMALL[toks[i + 1]], i + 2
        return _TENS[t], i + 1
    return None, i


def _under_1000(toks: list[str], i: int) -> tuple[int | None, int]:
    """`seven`, `twenty-one`, `three hundred and five`, `fifteen hundred`, `a hundred`."""
    v, j = _under_100(toks, i)
    if v is None and toks[i : i + 2] == ["a", "hundred"]:
        v, j = 1, i + 1
    if v is None:
        return None, i
    if toks[j : j + 1] == ["hundred"] and v >= 1:
        v, j = v * 100, j + 1
        k = j + 1 if toks[j : j + 1] == ["and"] else j
        r, m = _under_100(toks, k)
        if r:
            v, j = v + r, m
    return v, j


def _cardinal(toks: list[str]) -> tuple[int | None, int]:
    """The whole number spelled at the start of toks, and how many tokens it takes."""
    total, j, last = 0, 0, None
    while j < len(toks):
        if j == 0 and toks[0] == "a" and toks[1:2] and toks[1] in _SCALES:
            v, k = 1, 1
        else:
            v, k = _under_1000(toks, j)
        if v is None:
            break
        if toks[k : k + 1] and toks[k] in _SCALES and v and (last is None or _SCALES[toks[k]] < last):
            total += v * _SCALES[toks[k]]
            last, j = _SCALES[toks[k]], k + 1
            if toks[j : j + 1] == ["and"] and _under_1000(toks, j + 1)[0]:
                j += 1
            continue
        if last is not None and v >= last:
            break
        total, j = total + v, k
        break
    if j == 0 or (j and toks[j - 1] == "and"):
        return None, 0
    return total, j


_PRONOUN_BEFORE_ONE = {"no", "each", "every", "any", "the", "this", "that", "which", "another", "only"}
_HALF_NOT_BEFORE = {"first", "second", "other", "in", "the", "a", "an"}


def _word_number(text: str, run: list[tuple[str, int, int]]) -> _Atom | None:
    """The number spelled at the start of a run of words (joined by single spaces or hyphens), or None."""
    toks = [w.lower() for w, _, _ in run]
    n = len(toks)
    integer = False
    if toks[:3] == ["half", "a", "dozen"]:
        value, j = Fraction(6), 3
    elif toks[:2] == ["a", "dozen"]:
        value, j, integer = Fraction(12), 2, True
    elif toks[:2] == ["a", "half"]:
        value, j = Fraction(1, 2), 2
    elif toks[:3] in (["a", "quarter", "of"], ["a", "third", "of"]):
        value, j = Fraction(1, _FRACTION_WORDS[toks[1]]), 2
    elif toks[0] == "half":
        prev = re.search(r"([A-Za-z]+)\W*$", text[: run[0][1]])
        if toks[1:2] in (["past"], ["an"], ["a"]) or (prev and prev.group(1).lower() in _HALF_NOT_BEFORE):
            return None
        value, j = Fraction(1, 2), 1
    else:
        c, j = _cardinal(toks)
        if c is None:
            return None
        value, integer = Fraction(c), True
        if toks[j : j + 1] == ["dozen"]:
            value, j = value * 12, j + 1
        elif toks[j : j + 3] == ["and", "a", "half"]:
            value, j, integer = value + Fraction(1, 2), j + 3, False
        elif j < n and toks[j] in _FRACTION_WORDS and toks[j + 1 : j + 2] == ["of"]:
            plural = toks[j].endswith("s")
            if plural == (c >= 2) and c >= 1:
                value, j, integer = value / _FRACTION_WORDS[toks[j]], j + 1, False
    start, end = run[0][1], run[j - 1][2]
    # hyphen-joined to a word outside the number: a compound such as one-off, two-way, twenty one-hour
    if end + 1 < len(text) and text[end] == "-" and text[end + 1].isalpha():
        return _word_number(text, run[: j - 1]) if j > 1 else None
    if start >= 2 and text[start - 1] == "-" and text[start - 2].isalpha():
        return None
    if toks[:j] == ["one"]:
        prev = re.search(r"([A-Za-z]+)\W*$", text[:start])
        p = prev.group(1).lower() if prev else ""
        if p in _PRONOUN_BEFORE_ONE or toks[1:2] == ["another"] or toks[1:3] == ["by", "one"]:
            return None
        if p == "by" and re.search(r"\bone by\W*$", text[:start], re.I):
            return None
    return _Atom(start, end, (value,), words=True, integer_words=integer)


def _word_atoms(text: str, taken: list[Span]) -> list[_Atom]:
    words = [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(text)]
    out: list[_Atom] = []
    i = 0
    while i < len(words):
        j = i + 1
        while j < len(words) and j - i < 12 and text[words[j - 1][2] : words[j][1]] in (" ", "-"):
            j += 1
        got = None if _inside(words[i][1], words[i][2], taken) else _word_number(text, words[i:j])
        if got is None or _inside(got.start, got.end, taken):
            i += 1
            continue
        out.append(got)
        while i < len(words) and words[i][2] <= got.end:
            i += 1
    return out


def _inside(s: int, e: int, taken: list[Span]) -> bool:
    return any(s < te and ts < e for ts, te in taken)


# --- dates and other forms found first --------------------------------------------------------------------------


def _date_form(tag: str) -> str:
    d, mo, y = f"d{tag}", f"m{tag}", f"y{tag}"
    a = rf"(?P<{d}a>\d{{1,2}})(?:st|nd|rd|th)? (?:of )?(?P<{mo}a>{_MONTH})(?:,? (?P<{y}a>\d{{4}}))?"
    b = rf"(?P<{mo}b>{_MONTH}) (?P<{d}b>\d{{1,2}})(?:st|nd|rd|th)?(?:,? (?P<{y}b>\d{{4}}))?"
    return rf"(?:{a}|{b})"


_TO = rf" ?(?:to|until|till|through|thru|[{_DASHES}]) ?"
_ISO = re.compile(r"(?<![\w-])(\d{4})-(\d{2})-(\d{2})(?![\w-])")
_NAMED = re.compile(rf"(?<![\w]){_date_form('1')}(?![\w])")
_RANGE = re.compile(rf"(?<![\w]){_date_form('1')}{_TO}{_date_form('2')}(?![\w])")
_BETWEEN = re.compile(rf"(?<![\w])between {_date_form('1')} and {_date_form('2')}(?![\w])")
_DAY_RANGE = re.compile(
    rf"(?<![\w.,])(?P<d1>\d{{1,2}}){_TO}(?P<d2>\d{{1,2}}) (?P<m>{_MONTH})(?:,? (?P<y>\d{{4}}))?(?![\w])"
    rf"|(?<![\w])(?P<mb>{_MONTH}) (?P<d3>\d{{1,2}}){_TO}(?P<d4>\d{{1,2}})(?:,? (?P<yb>\d{{4}}))?(?![\w])"
)
_NUMERIC_DATE = re.compile(r"(?<![\w./-])(\d{1,4})([/.])(\d{1,2})\2(\d{2,4})(?![\w/]|[.,]\d)")
_SLASH = re.compile(r"(?<![\w./])(\d{1,2})/(\d{1,2})(?![\w/]|[.,]\d)")
_CLOCK = re.compile(r"(?<![\w:.])(\d{1,2})(?::(\d{2}))? ?(am|pm|a\.m\.|p\.m\.)(?![\w])", re.I)
_HMM = re.compile(r"(?<![\w:.,])(\d{1,3}):(\d{2})(?::(\d{2}))?(?![\w:]|[.,]\d)")
_GLUED = re.compile(r"(?<![\w.,])(\d{1,3})h(\d{2})(?![\w.,])")  # 9h28
_ID = re.compile(r"(?<![\w-])[A-Za-z]{1,6}-\d[\w-]*|#\d+")
_NUM_RANGE = re.compile(rf"(?<![\w.,])\d+(?:[.,]\d+)* ?[{_DASHES}] ?\d+(?:[.,]\d+)*(?![\w.,]?\d)")
_PART_UNIT = re.compile(
    r"(?<![\w-])(?P<f>half|a quarter of|a third of) an? (?!dozen\b)(?P<u>[a-z]+)(?![\w])"
    r"|(?<![\w-])(?P<g>half|quarter)[- ](?P<v>hour|day)(?![\w])",
    re.I,
)


def _month(s: str) -> int:
    return _MONTHS[s.rstrip(".")]


def _mk_date(y: int | None, m: int, d: int, span: Span) -> Date | Ambiguous:
    try:
        dt.date(2024 if y is None else y, m, d)  # with no year: a 29 February is a date
    except ValueError:
        return Ambiguous(span, "not a calendar date")
    return Date(y, m, d, span)


def _named(m: re.Match, tag: str, span: Span) -> Date | Ambiguous:
    side = "a" if m.group(f"m{tag}a") else "b"
    y = m.group(f"y{tag}{side}")
    return _mk_date(int(y) if y else None, _month(m.group(f"m{tag}{side}")), int(m.group(f"d{tag}{side}")), span)


def _range(a: Date | Ambiguous, b: Date | Ambiguous, span: Span) -> DateRange | Ambiguous:
    if isinstance(a, Ambiguous) or isinstance(b, Ambiguous):
        return Ambiguous(span, "not a calendar date")
    r = DateRange(a, b, span)
    if a.year is not None and b.year is not None and (b.year, b.month, b.day) < (a.year, a.month, a.day):
        return Ambiguous(span, "a date range that ends before it starts", (r,))
    return r


def _numeric_date(m: re.Match) -> Date | Ambiguous:
    span = m.span()
    p1, sep, p2, p3 = m.group(1), m.group(2), m.group(3), m.group(4)
    if len(p1) == 4:  # 2026/11/18: year, month, day
        if sep == "." or len(p3) > 2:
            return Ambiguous(span, "not a date form")
        return _mk_date(int(p1), int(p2), int(p3), span)
    if len(p1) > 2 or len(p3) == 3:
        return Ambiguous(span, "not a date form")
    if len(p3) == 2:
        return Ambiguous(span, "a two-digit year")
    y, a, b = int(p3), int(p1), int(p2)
    ok = [x for x in (_mk_date(y, b, a, span), _mk_date(y, a, b, span)) if isinstance(x, Date)]  # d/m, m/d
    if not ok:
        return Ambiguous(span, "not a calendar date")
    if len(ok) == 1 or ok[0] == ok[1]:
        return ok[0]
    return Ambiguous(span, "day and month could swap", tuple(ok))


def _slash(text: str, m: re.Match) -> Value:
    span = m.span()
    a, b = int(m.group(1)), int(m.group(2))
    nxt = re.match(r" ?([A-Za-z]+)(?![\w])", text[m.end() :])
    if b and nxt and nxt.group(1) in _UNIT:
        u = _UNIT[nxt.group(1)]
        whole = re.search(r"(?<![\w.,])(\d+) $", text[: span[0]])
        start, val = (whole.start(1), int(whole.group(1)) + Fraction(a, b)) if whole else (span[0], Fraction(a, b))
        return Quantity(val, u, _UNITS[u][0], (start, m.end() + nxt.end()))
    readings: list[Value] = [Number(Fraction(a, b), span)] if b else []
    for mo, d in ((a, b), (b, a)):
        got = _mk_date(None, mo, d, span)
        if isinstance(got, Date) and got not in readings:
            readings.append(got)
    return Ambiguous(span, "a fraction or a date with no year", tuple(readings))


def _hmm(text: str, m: re.Match) -> Value:
    span = m.span()
    h, mi, se = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    if mi >= 60 or se >= 60:
        return Ambiguous(span, "not a duration or a time of day")
    minutes = Fraction(h * 60 + mi) + Fraction(se, 60)
    unit = re.match(r" ?(?:hours|hour|hrs|hr|h)(?![\w])", text[m.end() :])
    if unit:
        return Duration(minutes, (span[0], m.end() + unit.end()))
    prev = re.search(r"([A-Za-z]+)\W*$", text[: span[0]])
    if prev is not None and prev.group(1).lower() in _CLOCK_WORDS and h < 24 and not m.group(3):
        return TimeOfDay(minutes, span)
    if h >= 24:
        return Duration(minutes, span)
    return Ambiguous(span, "a duration or a time of day", (Duration(minutes, span), TimeOfDay(minutes, span)))


def _glued(m: re.Match) -> Value:
    h, mi = int(m.group(1)), int(m.group(2))
    return Duration(Fraction(h * 60 + mi), m.span()) if mi < 60 else Ambiguous(m.span(), "not a duration")


def _part_unit(m: re.Match) -> Value:
    span = m.span()
    f = (m.group("f") or m.group("g")).lower()
    size = {"half": Fraction(1, 2), "a quarter of": Fraction(1, 4), "quarter": Fraction(1, 4)}.get(f, Fraction(1, 3))
    word = m.group("u") or m.group("v")
    unit = _UNIT.get(word)
    if unit in ("h", "min", "s"):
        return Duration(size * _UNITS[unit][1], span)
    if unit:
        return Quantity(size, unit, _UNITS[unit][0], span)
    return Quantity(size, _sing(word), "count", span)


@functools.lru_cache(maxsize=4096)  # pure in the text; a reader asks for the same parts several times
def _claimed(text: str) -> tuple[tuple[Value, ...], tuple[Span, ...]]:
    """Identifiers, dates, date ranges, clock times, number ranges and part units (`half an hour`): found before any
    number, so that their digits and words are not read again as numbers."""
    vals: list[Value] = []
    taken: list[Span] = []

    def scan(pattern: re.Pattern, make) -> None:
        for m in pattern.finditer(text):
            if _inside(*m.span(), taken):
                continue
            v = make(m)
            span = m.span() if v is None else (min(m.start(), v.span[0]), max(m.end(), v.span[1]))
            if v is not None and span != m.span() and _inside(*span, taken):
                v, span = Ambiguous(m.span(), "overlapping forms"), m.span()
            taken.append(span)
            if v is not None:
                vals.append(v)

    scan(_ID, lambda m: None)
    scan(_ISO, lambda m: _mk_date(int(m.group(1)), int(m.group(2)), int(m.group(3)), m.span()))
    scan(_BETWEEN, lambda m: _range(_named(m, "1", m.span()), _named(m, "2", m.span()), m.span()))
    scan(_RANGE, lambda m: _range(_named(m, "1", m.span()), _named(m, "2", m.span()), m.span()))
    scan(_DAY_RANGE, _day_range)
    scan(_NAMED, lambda m: _named(m, "1", m.span()))
    scan(_NUMERIC_DATE, _numeric_date)
    scan(_CLOCK, _clock)
    scan(_HMM, lambda m: _hmm(text, m))
    scan(_GLUED, _glued)
    scan(_SLASH, lambda m: _slash(text, m))
    scan(_NUM_RANGE, lambda m: Ambiguous(m.span(), "a range or a difference of numbers"))
    scan(_PART_UNIT, _part_unit)
    return tuple(vals), tuple(taken)


def _day_range(m: re.Match) -> Value:
    if m.group("m"):
        mo, d1, d2, y = _month(m.group("m")), int(m.group("d1")), int(m.group("d2")), m.group("y")
    else:
        mo, d1, d2, y = _month(m.group("mb")), int(m.group("d3")), int(m.group("d4")), m.group("yb")
    yy, span = (int(y) if y else None), m.span()
    return _range(_mk_date(yy, mo, d1, span), _mk_date(yy, mo, d2, span), span)


def _clock(m: re.Match) -> Value:
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower().replace(".", "")
    if not (1 <= h <= 12 and mi < 60):
        return Ambiguous(m.span(), "not a time of day")
    return TimeOfDay(Fraction((h % 12 + (12 if ap == "pm" else 0)) * 60 + mi), m.span())


# --- typing the atoms -------------------------------------------------------------------------------------------

_SYM = "[" + re.escape(SYMBOLS) + "]"
_CODE = r"(?:" + "|".join(CODES) + r")"
_PRE_MARK = re.compile(
    rf"(?:(?P<sign1>[{_MINUS}]) ?)?(?P<mark>{_SYM}|(?<![A-Za-z]){_CODE}) ?(?:(?P<sign2>[{_MINUS}]))?$"
)
_MUL = r"(?P<mul>k|K|m|M|bn)?(?: (?P<mulw>thousand|million|billion))?"
_AFTER_PRE = re.compile(rf"{_MUL}(?![\w])")
_POST_MARK = re.compile(
    rf"{_MUL}(?: ?(?P<sym>{_SYM})| ?(?P<code>{_CODE})(?![\w])| (?P<word>dollars?|euros?|pounds?)(?![\w]))"
)
_PERCENT = re.compile(r" ?%| (?:percent|per cent)(?![\w])")
_HOURS = re.compile(r" ?(?:hours|hour|hrs|hr|h)(?![\w])")
_MINS = re.compile(r" ?(?:minutes|minute|mins|min|m)(?![\w])")
_MINS_ALONE = re.compile(r" ?(?:minutes|minute|mins|min)(?![\w])")  # a bare `m` is a metre unless hours came first
_SECS = re.compile(r" ?(?:seconds|second|secs|sec|s)(?![\w])")
_JOIN = re.compile(r",? (?:and )?")
_UNIT_AFTER = re.compile(r" ?(?P<u>fl oz|[A-Za-z]+[23²³]?)(?![\w])")
_NOUN_AFTER = re.compile(r" (?P<w>[a-z][a-z'-]*)(?![\w])")


def _sing(word: str) -> str:
    w = word.lower()
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith(("ches", "shes", "sses", "xes")):
        return w[:-2]
    if len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _readings(atom: _Atom, convention: str | None) -> tuple[tuple[Fraction, ...], str]:
    """The atom's readings under the text's convention: one, or several with the reason they are not one."""
    if not atom.readings:
        return (), atom.why or "not a number"
    if len(atom.readings) == 1:
        return atom.readings, ""
    if convention == "point":
        return atom.readings[:1], ""
    if convention == "comma":
        return atom.readings[1:], ""
    return atom.readings, atom.why


def _one(span: Span, rs: tuple[Fraction, ...], why: str, make) -> Value:
    """A value when there is one reading, else Ambiguous with every reading."""
    if len(rs) == 1:
        return make(rs[0], span)
    return Ambiguous(span, why, tuple(make(r, span) for r in rs))


def _convention(atoms: list[_Atom]) -> str | None:
    seen = {a.evidence for a in atoms if a.evidence}
    return "mixed" if len(seen) > 1 else (seen.pop() if seen else None)


_CURRENCY_SEEN = re.compile(
    rf"(?P<a>{_SYM}) ?[{_MINUS}]?\d|\d ?(?P<b>{_SYM})|(?<![A-Za-z])(?P<c>{_CODE}) ?[{_MINUS}]?\d"
    rf"|\d ?(?P<d>{_CODE})(?![\w])|\d (?P<e>dollars?|euros?)(?![\w])|(?P<f>\bsterling\b)"
)


@functools.lru_cache(maxsize=1024)
def document(text: str) -> Context:
    """What a whole document settles for its parts: the decimal convention and the currencies next to numbers."""
    _, taken = _claimed(text)
    seen = set()
    for m in _CURRENCY_SEEN.finditer(text):
        k = m.lastgroup
        seen.add("£" if k == "f" else _CUR_WORDS[m.group(k)] if k == "e" else m.group(k))
    return Context(_convention(_digit_atoms(text, taken)), frozenset(seen))


def decimal_convention(text: str) -> str | None:
    """ "point", "comma", "mixed" (both seen) or None (no number settles it), from the forms in the text."""
    return document(text).convention


@functools.lru_cache(maxsize=4096)
def _atoms(text: str) -> tuple[tuple[Value, ...], tuple[_Atom, ...]]:
    claimed, taken = _claimed(text)
    return claimed, tuple(sorted(_digit_atoms(text, taken) + _word_atoms(text, taken), key=lambda a: a.start))


def normalize(text: str, doc: Context | None = None) -> Normalized:
    """Every value recognised in `text`, in text order. `doc` is what the whole document settles (`document`), for a
    caller that reads a document part by part; by default the text is its own document."""
    return _normalize(text, document(text) if doc is None else doc)


@functools.lru_cache(maxsize=4096)  # a pure function of the text and the document's context, both immutable
def _normalize(text: str, doc: Context) -> Normalized:
    convention = doc.convention
    claimed, atoms = _atoms(text)
    claimed_spans = [v.span for v in claimed]
    by_start = {a.start: a for a in atoms}
    values = list(claimed)
    used: list[Span] = []
    for atom in atoms:
        if _inside(atom.start, atom.end, used):
            continue
        v = _type(text, atom, by_start, doc)
        if _inside(*v.span, claimed_spans):  # a unit or a mark that belongs to a date: the number alone
            rs, why = _readings(atom, convention)
            v = _one((atom.start, atom.end), rs, why, lambda r, sp: Number(r, sp, atom.words)) if rs else v
        if _inside(*v.span, used):  # a form that reaches back into the one before: neither is read
            prev = values.pop()
            v = Ambiguous((min(prev.span[0], v.span[0]), max(prev.span[1], v.span[1])), "overlapping forms")
            used.pop()
        values.append(v)
        used.append(v.span)
    values.sort(key=lambda v: v.span)
    return Normalized(text, tuple(values), convention)


def _multiplier(m: re.Match | None) -> int:
    if m is None:
        return 1
    return _MULTIPLIERS.get(m.group("mul") or "", 1) * _MULTIPLIER_WORDS.get(m.group("mulw") or "", 1)


def _type(text: str, atom: _Atom, by_start: dict[int, _Atom], doc: Context) -> Value:
    s, e = atom.start, atom.end
    rs, why = _readings(atom, doc.convention)
    if not rs:
        return Ambiguous((s, e), why)

    # money with its mark before the number: $12.50, EUR 12.50, -$40, $-40, ($12.50), $1.2M
    pre = None if atom.words else _PRE_MARK.search(text, max(0, s - 8), s)
    if pre:
        ps, mark = pre.start(), pre.group("mark")
        sign = -1 if (pre.group("sign1") or pre.group("sign2")) else 1
        if pre.group("sign1") and re.search(r"[\w)] ?$", text[:ps]):
            sign, ps = 1, pre.start("mark")  # $12 - $4 and 3-$4: a minus between two amounts, no sign
        mul = _AFTER_PRE.match(text, e)
        end = mul.end() if mul else e
        span = (ps, end)
        if re.match(rf" ?(?:{_SYM}|{_CODE}(?![\w]))", text[end:]):
            return Ambiguous(span, "two currency marks on one amount")
        if re.search(r"\d ?$", text[:ps]):
            return Ambiguous(span, "a currency mark between two numbers")
        size = _multiplier(mul)
        if ps > 0 and text[ps - 1] == "(" and text[end : end + 1] == ")":
            span = (ps - 1, end + 1)
            vals = [Money(r * size * k, mark, span) for r in rs for k in (1, -1)]
            return Ambiguous(span, "an amount in parentheses: negative, or an aside", tuple(vals))
        return _one(span, rs, why, lambda r, sp: Money(sign * r * size, mark, sp))

    # money with its mark after the number: 12,50 €, 12.50 EUR, twenty dollars, 50 pounds (money or mass)
    post = _POST_MARK.match(text, e)
    if post and (post.group("word") or not atom.words):
        word = post.group("word") or ""
        span_s = s - 1 if s > 0 and text[s - 1] in _MINUS and (s < 2 or not text[s - 2].isalnum()) else s
        span = (span_s, post.end())
        sign = -1 if span_s < s else 1
        size = _multiplier(post)
        if word.startswith("pound") and doc.currencies and "£" not in doc.currencies and "GBP" not in doc.currencies:
            return _one(span, rs, why, _lb)  # the document's money is in another currency: a mass
        if word.startswith("pound"):
            return Ambiguous(
                span, "pounds: money or mass", tuple(x for r in rs for x in (Money(r, "£", span), _lb(r, span)))
            )
        nxt = re.match(rf" ?[{_MINUS}]?(?=\d)", text[post.end() :])
        if nxt and not word:
            a2 = by_start.get(post.end() + nxt.end())
            return Ambiguous((span[0], a2.end if a2 else post.end()), "a currency mark between two numbers")
        mark = post.group("sym") or post.group("code") or _CUR_WORDS[word]
        return _one(span, rs, why, lambda r, sp: Money(sign * r * size, mark, sp))

    # a percentage
    pm = _PERCENT.match(text, e)
    if pm:
        return _one((s, pm.end()), rs, why, lambda r, sp: Percent(r, sp))

    # a duration: hours, then minutes and seconds; minutes, then seconds; seconds
    first = _HOURS.match(text, e) or _MINS_ALONE.match(text, e) or _SECS.match(text, e)
    if first:
        unit = " " + first.group(0).strip()
        base = 60 if _HOURS.fullmatch(unit) else 1 if _MINS.fullmatch(unit) else Fraction(1, 60)
        scale, end, extra = base, first.end(), Fraction(0)
        for next_scale, pat in ((1, _MINS), (Fraction(1, 60), _SECS)):
            if scale <= next_scale:
                continue
            j = _JOIN.match(text, end)
            a2 = by_start.get(j.end() if j else end)
            u2 = pat.match(text, a2.end) if a2 is not None and not a2.separators else None
            if u2 is None or len(a2.readings) != 1:
                continue
            extra += a2.readings[0] * next_scale
            end, scale = u2.end(), next_scale
        return _one((s, end), rs, why, lambda r, sp: Duration(r * base + extra, sp))

    # a quantity with a unit, or with a count noun
    um = _UNIT_AFTER.match(text, e)
    if um and (um.group("u") in _UNIT or um.group("u") == "fl oz"):
        u = "fl oz" if um.group("u") == "fl oz" else _UNIT[um.group("u")]
        return _one((s, um.end()), rs, why, lambda r, sp: Quantity(r, u, _UNITS[u][0], sp))
    nm = _NOUN_AFTER.match(text, e)
    if nm and not ({nm.group("w")} & (_NOT_NOUNS | set(_SMALL) | set(_TENS))):
        noun = _sing(nm.group("w"))
        return _one((s, nm.end()), rs, why, lambda r, sp: Quantity(r, noun, "count", sp))

    return _one((s, e), rs, why, lambda r, sp: Number(r, sp, atom.words))


def _lb(r: Fraction, span: Span) -> Quantity:
    return Quantity(r, "lb", "mass", span)


# --- canonical spelling for the calculator's reader -------------------------------------------------------------


def canonical(text: str, doc: Context | None = None, words: bool = False) -> tuple[str, tuple[Ambiguous, ...]]:
    """The text with values spelled as the calculator's reader reads them, and every ambiguous form in it.

    - with a decimal comma (the document's convention): every number with a separator is written with a point and no
      thousands separator;
    - a currency mark after an amount moves before it (`12.50 €` to `€12.50`, `12.50 EUR` to `EUR 12.50`);
    - `20 percent` and `20 per cent` become `20%`;
    - `words`: whole numbers in words become digits (`seven` to `7`, `a dozen` to `12`, `two dozen` to `24`).
    Negative amounts, amounts with a multiplier and fractions in words keep their spelling.
    """
    doc = document(text) if doc is None else doc
    convention = doc.convention
    n = normalize(text, doc)
    _, atoms = _atoms(text)
    edits: list[tuple[int, int, str]] = []

    def number(a: _Atom) -> str | None:
        """The atom's canonical spelling, or None to keep it."""
        if not a.words and a.separators and convention == "comma" and a.evidence != "point" and a.readings:
            return text[a.start : a.end].replace(".", "").replace(",", ".")
        if a.words and words and a.integer_words:
            return str(a.readings[0].numerator)
        return None

    for v in n.values:
        if isinstance(v, Ambiguous):
            continue
        s, e = v.span
        inner = [a for a in atoms if s <= a.start and a.end <= e]
        if isinstance(v, Money):
            lead = inner[0] if inner else None
            if lead is None or lead.words or v.value < 0:
                continue
            if re.match(r"(?:[kKmM]|bn| thousand| million| billion)", text[lead.end : e]):
                continue  # an amount with a multiplier keeps its spelling
            spelled = number(lead) or text[lead.start : lead.end]
            if lead.start == s:  # the mark comes after: write it first
                mark = v.currency
                edits.append((s, e, (mark if len(mark) == 1 else f"{mark} ") + spelled))
            else:
                edits.append((lead.start, lead.end, spelled))
            continue
        if isinstance(v, Percent) and inner and not inner[0].words:
            edits.append((s, e, (number(inner[0]) or text[inner[0].start : inner[0].end]) + "%"))
            continue
        for a in inner:
            got = number(a)
            if got is not None:
                edits.append((a.start, a.end, got))
    out = text
    for s, e, rep in sorted(set(edits), reverse=True):
        out = out[:s] + rep + out[e:]
    return out, n.ambiguous
