# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Read a state's arithmetic into a `Case`, or say why not (`Unsure`).

The parser knows no item names. It reads structure: a state is lines (plain `Label: detail` lines, bullets, two-column
table rows, or a JSON object's fields and lists of sentences), and every line is either an item, a rule sentence or
harmless text. All of them have to be understood; one line that is not gives no case at all.

Items (the detail after the label):
  `4 attachments at $299.46 each`, `2 trips at $978.15 a trip`, `9 h at $23.50 an hour`, `1 item at $48.00`
  `1450 kg at $0.56 per lb`, `1605 lb, $0.20/kg`, `26.1 L; price $49.22 per gal`  (a unit conversion: the factor has
      to be stated in a rule sentence)
  `1 machine at $52.81 a day for 6 days`, `3 machines at $10.78 each a day, 15 Aug to 29 Aug`,
      `5 h a day at $8.80 an hour, 17 Aug to 1 Sep`  (a date range needs a rule saying whether its last day counts)
  `9 h 28 min and 7 h 10 min; paid at $59.44 per hour`, `3.57 m, 4.11 m and 1.59 m; cutting $42.94 a metre`
  `31 Jan to 4 Feb; credit $54.35 a day`, `7 Aug to 12 Aug; loses a $129.81 attendance bonus each day`
  `$58.31 a week, 4 weeks`, `$235.29 per visit` (counted from the list item of that name), `$65.37`  (amounts whose
      sign a rule gives: added as a fee or taken off as a credit)
  Not read: a count followed by a word that stands for a number of things or by a number in words (`2 dozen eggs at
      $3.00 each`, `1 gross of pens at $20.00`, `2 packs of six at $3.00 each`), unless the price is per that same word
      (`2 dozen at $3.00 a dozen`); a unit factor between two dimensions (`1 lb = 0.4536 m`).

Rule sentences: a definition of the figure asked about, the final rounding to the cent (read, not applied), unit
factors, whether a date range counts its last day, time or length rounding to a stated step (per entry or once on
the sum), a percentage of one named item added or taken off before tax, the tax rule (its rate, charged or withheld,
and which fees and credits are added or taken off before or after it), and when a credit comes off.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from fractions import Fraction

from decisio.calculator.compute import Adjustment, Case, Credit, Fee, Item, Tax

# every reason the calculator gives no block, in the order the checks run
REASONS = {
    "no-arithmetic": "the text states no money amount, or no priced item",
    "not-english": "the text is not in English (v1 reads English only)",
    "mixed-currency": "more than one currency",
    "unparsed-item": "an item line with an amount does not parse completely, or its count is not certain (a word "
    "such as dozen, pair or hundred, or a number in words, between the count and the price)",
    "unparsed-rule": "a sentence with a number, an amount or a rule word matches no rule kind",
    "unsupported-rule": "a rule v1 recognises but does not compute (an adjustment after tax, rounding money before "
    "the final figure, rounding up or down, quantities priced as printed with no conversion, an adjustment on a fee "
    "or on several items)",
    "unmatched-label": "a rule names something that matches no item",
    "ambiguous-label": "a rule names something that matches more than one item, or two lines share a name",
    "ambiguous-rule": "a rule's meaning is not certain (its sign, its place before or after tax, whether tax is "
    "charged or withheld, or a rounding that falls exactly half way)",
    "conflicting-rules": "two rules disagree, the same thing is placed twice, or a unit factor joins units of "
    "different dimensions (1 lb = 0.4536 m)",
    "missing-rule": "an item needs a rule the text does not give (a unit factor, a day-count rule, a credit's place)",
    "unaccounted-amount": "an amount no rule places: added or taken off is not stated",
    "ambiguous-date": "a date or a date range cannot be read with certainty",
    "several-records": "the text holds more than one record (more than one line with an identifier such as INV-204)",
    "no-tax-rule": "no tax rule, so the taxed base and the tax line are not defined",
    "internal-error": "the calculator failed on this input; no block (a bug to report)",
}


class Unsure(Exception):
    """The calculator gives no block; `reason` is one of `REASONS`, `detail` says where (never logged with text)."""

    def __init__(self, reason: str, detail: str = ""):
        assert reason in REASONS, reason
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# --- lexical pieces ---------------------------------------------------------------------------------------------

SYMBOLS = "$€£¥"
CODES = (
    "USD EUR GBP JPY CHF CAD AUD NZD SEK NOK DKK ISK INR CNY HKD SGD ZAR MXN BRL PLN CZK HUF RON TRY ILS AED SAR KRW"
).split()
_AMT = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_NUM = _AMT  # a quantity reads like an amount
_ANY_MONEY = re.compile(rf"[{re.escape(SYMBOLS)}] ?\d|\b(?:{'|'.join(CODES)}) ?\d")
_ID = re.compile(r"\b[A-Z]{1,5}-\d{1,8}\b")
_WORD = r"[A-Za-z][A-Za-z'-]*"
_WORDS = rf"{_WORD}(?: {_WORD}){{0,2}}"
_MONTHS = {
    m: i + 1
    for i, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ]
    )
    for m in names
}
_DATE = r"\d{1,2} [A-Z][a-z]+(?: \d{4})?|[A-Z][a-z]+ \d{1,2}(?:, \d{4})?"
_RANGE = rf"(?P<da>{_DATE}) to (?P<db>{_DATE})"

# measure units and the names they go by; any other noun is a count noun ("sheets", "trips")
_UNIT_NAMES = {
    "h": ("h", "hr", "hrs", "hour", "hours"),
    "min": ("min", "mins", "minute", "minutes"),
    "m": ("m", "metre", "metres", "meter", "meters"),
    "cm": ("cm", "centimetre", "centimetres", "centimeter", "centimeters"),
    "km": ("km", "kilometre", "kilometres", "kilometer", "kilometers"),
    "mi": ("mi", "mile", "miles"),
    "ft": ("ft", "foot", "feet"),
    "yd": ("yd", "yard", "yards"),
    "kg": ("kg", "kilogram", "kilograms", "kilo", "kilos"),
    "g": ("g", "gram", "grams"),
    "lb": ("lb", "lbs", "pound", "pounds"),
    "oz": ("oz", "ounce", "ounces"),
    "t": ("t", "tonne", "tonnes"),
    "L": ("L", "l", "litre", "litres", "liter", "liters"),
    "ml": ("ml", "mL", "millilitre", "millilitres", "milliliter", "milliliters"),
    "gal": ("gal", "gallon", "gallons"),
}
_UNIT = {name: unit for unit, names in _UNIT_NAMES.items() for name in names}
# the dimension of each measure unit: a unit factor has to join two units of one dimension
_DIMENSION = {
    **dict.fromkeys(("h", "min"), "time"),
    **dict.fromkeys(("m", "cm", "km", "mi", "ft", "yd"), "length"),
    **dict.fromkeys(("kg", "g", "lb", "oz", "t"), "mass"),
    **dict.fromkeys(("L", "ml", "gal"), "volume"),
}
_LENGTH_UNITS = {"m", "cm", "km", "mi", "ft", "yd"}
_PERIODS = {"day", "days", "week", "weeks", "month", "months", "night", "nights", "year", "years"}
# words that stand for a number of things (2 dozen eggs are 24 eggs, 2 packs of six are 12 cans): after an item's
# count, they leave what the count and the price are counted in uncertain
_MULTIPLIERS = set(
    """dozen dozens doz gross score scores pair pairs brace couple couples hundred hundreds thousand thousands
    million millions half halves third thirds quarter quarters""".split()
)
_NUMBER_WORDS = set(
    """zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen
    eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety""".split()
)

# words that change what an amount means; a line or sentence carrying one is not read as a plain form
_RISKY = set(
    """minimum maximum max cap capped extra plus additional discount discounted surcharge premium free first after
    before over under above below beyond double triple half times tax taxed fee credit except not no only less minus
    off deposit waived waive flat fixed total instead unless if when per""".split()
)
# words that make a sentence a rule; a sentence with one has to match a rule kind, or there is no block
_RULE_WORDS = re.compile(
    r"\b(?:tax|taxed|taxes|taxable|vat|gst|discount\w*|surcharge\w*|premium\w*|credit\w*|round\w*|fee|fees|before|"
    r"after|per|each|every|deduct\w*|subtract\w*|withh\w*|convert\w*|percent\w*|cents?|minus|plus|add|added|adding|"
    r"charge|charged|charges|days?|dates?|price[sd]?|pric\w+|rates?|nearest|exclud\w*|includ\w*|except|waive\w*|free|"
    r"minimum|maximum|cap|capped|instead|ignore\w*|only|unless|refund\w*|rebate\w*|allowance|bonus|levy|commission|"
    r"exempt\w*|zero-rated|void\w*|cancel\w*|return\w*|revers\w*|double\w*|twice|half|extra|less|reduc\w*|"
    r"increas\w*|limit\w*|split|share[sd]?|deposit\w*|prepaid|already|advance|instal\w*|outstanding|interest|"
    r"penalt\w*|overdue|tip|tips|gratuity|subtotal|multiply|multiplied|times|sum|summed)\b",
    re.I,
)
_EN_WORDS = set(
    "the a an and of to is are at on for in each per with by from then before after this these what all every as be "
    "it its or that which only one".split()
)
_OTHER_WORDS = set(
    """der die das und ist mit für nicht von den dem le la les et est des du pour avec une un sur dans el los las y es
    para con una por del il lo gli della di che è het een en van niet och är att det og er af ikke ja ei että
    ve bir bu ile da""".split()
)


def _fr(s: str) -> Fraction:
    return Fraction(s.replace(",", ""))


def _sing(word: str) -> str:
    w = word.lower()
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith(("ches", "shes", "sses", "xes")):
        return w[:-2]
    if len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _words(s: str) -> list[str]:
    w = re.sub(r"[-_]", " ", s.lower()).split()
    if w and w[0] == "the":
        w = w[1:]
    return [_sing(x) for x in w]


def _norm(s: str) -> str:
    return " ".join(_words(s))


def _contains(phrase: str, label: str) -> bool:
    p, lab = _words(phrase), _words(label)
    return bool(lab) and any(p[i : i + len(lab)] == lab for i in range(len(p) - len(lab) + 1))


def _cap(label: str) -> str:
    return label[:1].upper() + label[1:]


def _has_number(s: str) -> bool:
    return bool(re.search(r"\d", _ID.sub("", s)))


# --- what the text holds ----------------------------------------------------------------------------------------


@dataclass
class _Raw:
    """An item line as read, before the rules place it."""

    label: str
    order: int
    form: str  # count, convert, days, list, daycredit, feecount, feeper, lone
    from_key: bool = False  # a JSON field name, whose spaces and hyphens both became underscores
    q: Fraction = Fraction(0)
    unit: str = ""
    price: Fraction = Fraction(0)
    to_unit: str = ""
    days: int | None = None
    span: tuple[str, str] | None = None
    dim: str = ""  # list: "time" or "length"
    entries: tuple[Fraction, ...] = ()
    per: str = ""  # feeper: the noun the fee is per


@dataclass
class _Rules:
    factors: dict[tuple[str, str], Fraction] = field(default_factory=dict)
    day_count: set[str] = field(default_factory=set)
    roundings: list[tuple[str, Fraction, bool]] = field(default_factory=list)  # (dimension, step, each)
    percents: list[tuple[str, Fraction, str, int]] = field(default_factory=list)  # (name, percent, target, sign)
    taxes: list[tuple[Fraction, str | None, list[str], list[str], str | None]] = field(default_factory=list)
    timings: list[tuple[str, str]] = field(default_factory=list)  # (phrase, "inside" | "outside")


class _Text:
    def __init__(self, parts: list[str]):
        self.full = "\n".join(parts)
        found = set()
        for m in re.finditer(rf"([{re.escape(SYMBOLS)}]) ?\d", self.full):
            found.add(m.group(1))
        for m in re.finditer(rf"\b({'|'.join(CODES)}) ?\d", self.full):
            found.add(m.group(1))
        self.currencies = found
        self.currency = next(iter(found)) if len(found) == 1 else ""
        self.mark = re.escape(self.currency) + " ?"

    def money(self, name: str) -> str:
        return rf"{self.mark}(?P<{name}>{_AMT})"


def _english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    if sum(1 for c in letters if ord(c) > 0x24F) > 0.05 * len(letters):
        return False
    words = re.findall(r"[^\W\d_]+", text.lower())
    en = sum(1 for w in words if w in _EN_WORDS)
    other = sum(1 for w in words if w in _OTHER_WORDS)
    return en > other and (en > 0 or len(words) < 5)


# --- lines ------------------------------------------------------------------------------------------------------


@dataclass
class _Line:
    raw: str
    label: str | None = None
    detail: str | None = None
    from_key: bool = False


def _json_lines(obj) -> list[_Line] | None:
    if not isinstance(obj, dict):
        return None
    out: list[_Line] = []
    for key, val in obj.items():
        if isinstance(val, str):
            out.append(_Line(f"{key}: {val}", str(key).replace("_", " "), val, True))
        elif isinstance(val, list) and all(isinstance(x, str) for x in val):
            out.extend(_Line(x) for x in val)
        elif isinstance(val, dict):
            for k, v in val.items():
                if not isinstance(v, str):
                    raise Unsure("unparsed-item", f"field {k!r} is not text")
                out.append(_Line(f"{k}: {v}", str(k).replace("_", " "), v, True))
        else:
            raise Unsure("unparsed-item", f"field {key!r} holds records or a number")
    return out


def _text_lines(text: str) -> list[_Line]:
    s = text.strip()
    if s.startswith("{"):
        try:
            obj = json.loads(s)
        except ValueError:
            obj = None
        if obj is not None:
            got = _json_lines(obj)
            if got is not None:
                return got
    out = []
    for ln in text.split("\n"):
        t = ln.strip()
        if not t:
            continue
        if t.startswith("|"):
            cells = [c.strip() for c in t.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?|", c) for c in cells):
                continue
            if len(cells) == 2:
                out.append(_Line(t, cells[0], cells[1]))
            else:
                out.append(_Line(t))
            continue
        t = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", t)
        if ": " in t:
            label, detail = t.split(": ", 1)
            out.append(_Line(t, label.strip(), detail.strip()))
        elif t.endswith(":"):
            out.append(_Line(t, t[:-1].strip(), ""))
        else:
            out.append(_Line(t))
    return out


# --- items ------------------------------------------------------------------------------------------------------


def _risky(words: str | None) -> bool:
    return bool(words) and any(w.lower() in _RISKY for w in words.split())


def _duration(s: str) -> Fraction | None:
    m = re.fullmatch(
        r"(?:(?P<h>\d+(?:\.\d+)?) ?(?:h|hr|hrs|hours?)(?: (?P<m>\d+) ?(?:min|mins|minutes?))?|"
        r"(?P<m2>\d+(?:\.\d+)?) ?(?:min|mins|minutes?))",
        s,
    )
    if not m:
        return None
    if m.group("m2"):
        return _fr(m.group("m2"))
    return _fr(m.group("h")) * 60 + (int(m.group("m")) if m.group("m") else 0)


def _entries(s: str) -> tuple[str, tuple[Fraction, ...], str] | None:
    """A list of durations (in minutes) or of lengths in one unit: `9 h 28 min and 7 h 10 min`, `3.57 m and 1.59 m`."""
    parts = re.split(r", (?:and )?| and ", s)
    durs = [_duration(p) for p in parts]
    if all(d is not None for d in durs):
        return "time", tuple(durs), "min"
    vals, units = [], set()
    for p in parts:
        m = re.fullmatch(rf"(?P<v>{_NUM}) ?(?P<u>[A-Za-z]+)", p)
        if not m or _UNIT.get(m.group("u")) not in _LENGTH_UNITS:
            return None
        vals.append(_fr(m.group("v")))
        units.add(_UNIT[m.group("u")])
    if len(units) != 1:
        return None
    return "length", tuple(vals), units.pop()


def _count_unsure(words: str | None, per: str | None) -> bool:
    """The words between an item's count and its price hold a multiplier (`2 dozen eggs`, `1 gross of pens`) or a
    number in words (`2 packs of six`): what the count means is not certain. One multiplier priced per that same
    multiplier (`2 dozen at $3.00 a dozen`) is certain."""
    if not words:
        return False
    toks = [w.lower() for w in words.split()]
    if len(toks) == 1 and toks[0] in _MULTIPLIERS and per is not None and _sing(per) == _sing(toks[0]):
        return False
    return any(t in _MULTIPLIERS or all(p in _NUMBER_WORDS for p in t.split("-")) for t in toks)


def _same_noun(a: str, b: str) -> bool:
    ua, ub = _UNIT.get(a), _UNIT.get(b)
    if ua or ub:
        return ua is not None and ua == ub
    return _sing(a.split()[-1]) == _sing(b.split()[-1])


def _quantity_price(label, order, q, u, price, pu, each) -> _Raw | None:
    """`q u at price per pu`: a count, or a conversion when both are measure units that differ."""
    if _count_unsure(u, None if each else pu):
        return None
    if each or pu is None:
        if u is not None and _UNIT.get(u.split()[-1]) is not None and len(u.split()) == 1 and not each:
            return None  # `5 kg at $3` names no price unit
        return _Raw(label, order, "count", q=q, price=price, unit=u or "")
    if pu.lower() in _PERIODS:
        return None  # a price per day with no day count
    if u is None:
        return None
    ua, ub = _UNIT.get(u), _UNIT.get(pu)
    if ua and ub and ua != ub:
        return _Raw(label, order, "convert", q=q, unit=ua, price=price, to_unit=ub)
    if _same_noun(u, pu):
        return _Raw(label, order, "count", q=q, price=price, unit=u)
    return None


def _item(text: _Text, label: str, detail: str, order: int) -> _Raw | None:
    d = detail.strip().rstrip(".").strip()
    if not d or len(d) > 400 or not text.currency:
        return None
    M = text.money
    per = rf"(?:(?P<each>each)|(?:a|an|per) (?P<pu>{_WORD})|/(?P<su>{_WORD}))"

    if m := re.fullmatch(M("p"), d):
        return _Raw(label, order, "lone", price=_fr(m.group("p")))
    if m := re.fullmatch(rf"{M('p')} (?:a|an|per) (?P<pu>{_WORD}), (?P<n>{_NUM}) (?P<u>{_WORD})", d):
        if not _same_noun(m.group("u"), m.group("pu")):
            return None
        return _Raw(label, order, "feecount", price=_fr(m.group("p")), q=_fr(m.group("n")), unit=m.group("u"))
    if m := re.fullmatch(rf"{M('p')} (?:a|an|per) (?P<pu>{_WORD})", d):
        if m.group("pu").lower() in _PERIODS or m.group("pu") in _UNIT:
            return None
        return _Raw(label, order, "feeper", price=_fr(m.group("p")), per=m.group("pu"))
    if m := re.fullmatch(rf"{_RANGE}; credit {M('p')} (?:a|per) day", d):
        return _Raw(label, order, "daycredit", price=_fr(m.group("p")), span=(m.group("da"), m.group("db")))
    if m := re.fullmatch(rf"{_RANGE}; loses (?:a|an) {M('p')} (?P<w>[a-z]+(?: [a-z]+){{0,3}}) (?:each|a|per) day", d):
        if _risky(m.group("w")):
            return None
        return _Raw(label, order, "daycredit", price=_fr(m.group("p")), span=(m.group("da"), m.group("db")))

    tail = rf"(?: for (?P<n>\d+) days?|, {_RANGE})"
    if m := re.fullmatch(rf"(?P<q>{_NUM}) (?P<u>{_WORDS}) at {M('p')} (?:each )?(?:a|per) day{tail}", d):
        if _risky(m.group("u")) or _UNIT.get(m.group("u")) is not None or _count_unsure(m.group("u"), None):
            return None
        return _days(label, order, m, unit=m.group("u"))
    if m := re.fullmatch(rf"(?P<q>{_NUM}) (?P<u>{_WORD}) a day at {M('p')} (?:a|an|per) (?P<pu>{_WORD}){tail}", d):
        if _UNIT.get(m.group("u")) is None or _UNIT.get(m.group("u")) != _UNIT.get(m.group("pu")):
            return None
        return _days(label, order, m, unit=m.group("u"))

    if m := re.fullmatch(rf"(?P<q>{_NUM}) (?P<u>{_WORD}), {M('p')}/(?P<pu>{_WORD})", d):
        return _quantity_price(label, order, _fr(m.group("q")), m.group("u"), _fr(m.group("p")), m.group("pu"), False)
    if m := re.fullmatch(rf"(?P<q>{_NUM}) (?P<u>{_WORD}); price {M('p')} per (?P<pu>{_WORD})", d):
        return _quantity_price(label, order, _fr(m.group("q")), m.group("u"), _fr(m.group("p")), m.group("pu"), False)
    if m := re.fullmatch(rf"(?P<q>{_NUM})(?: (?P<u>{_WORDS}))? at {M('p')}(?: {per})?", d):
        if _risky(m.group("u")):
            return None
        pu = m.group("pu") or m.group("su")
        return _quantity_price(
            label, order, _fr(m.group("q")), m.group("u"), _fr(m.group("p")), pu, bool(m.group("each"))
        )

    if m := re.fullmatch(
        rf"(?P<list>[^;]+); (?P<w>(?:[a-z]+ ){{0,3}}){M('p')}(?: (?:an|a|per) (?P<pu>{_WORD})|/(?P<su>{_WORD}))", d
    ):
        if _risky(m.group("w")):
            return None
        got = _entries(m.group("list"))
        if got is None:
            return None
        dim, vals, unit = got
        rate_unit = _UNIT.get(m.group("pu") or m.group("su") or "")
        if dim == "time" and rate_unit not in ("h", "min"):
            return None
        if dim == "length" and rate_unit != unit:
            return None
        return _Raw(label, order, "list", dim=dim, entries=vals, unit=rate_unit, price=_fr(m.group("p")))
    return None


def _days(label, order, m, unit) -> _Raw:
    raw = _Raw(label, order, "days", q=_fr(m.group("q")), price=_fr(m.group("p")), unit=unit)
    if m.group("n"):
        raw.days = int(m.group("n"))
    else:
        raw.span = (m.group("da"), m.group("db"))
    return raw


def _date(tok: str) -> tuple[int, int, int | None]:
    m = re.fullmatch(r"(\d{1,2}) ([A-Za-z]+)(?: (\d{4}))?", tok) or None
    if m:
        day, mon, year = int(m.group(1)), m.group(2), m.group(3)
    else:
        m = re.fullmatch(r"([A-Za-z]+) (\d{1,2})(?:, (\d{4}))?", tok)
        if not m:
            raise Unsure("ambiguous-date", tok)
        mon, day, year = m.group(1), int(m.group(2)), m.group(3)
    if mon.lower() not in _MONTHS:
        raise Unsure("ambiguous-date", tok)
    return _MONTHS[mon.lower()], day, int(year) if year else None


def _span_days(span: tuple[str, str], inclusive: bool) -> int:
    (ma, da, ya), (mb, db, yb) = _date(span[0]), _date(span[1])

    def count(year_a: int, year_b: int) -> int:
        try:
            a, b = dt.date(year_a, ma, da), dt.date(year_b, mb, db)
        except ValueError:
            raise Unsure("ambiguous-date", f"{span[0]} to {span[1]}") from None
        if b < a:
            raise Unsure("ambiguous-date", f"{span[0]} to {span[1]} ends before it starts")
        return (b - a).days + (1 if inclusive else 0)

    def years(y_known_a, y_known_b, guess):
        ya_, yb_ = y_known_a, y_known_b
        if ya_ is None and yb_ is None:
            ya_ = guess
        if ya_ is None:
            ya_ = yb_ if (ma, da) <= (mb, db) else yb_ - 1
        if yb_ is None:
            yb_ = ya_ if (ma, da) <= (mb, db) else ya_ + 1
        return ya_, yb_

    if ya is not None or yb is not None:
        return count(*years(ya, yb, None))
    # no year: the count must not depend on whether a 29 February falls inside
    got = {count(*years(None, None, g)) for g in (2023, 2024)}
    if len(got) != 1:
        raise Unsure("ambiguous-date", f"{span[0]} to {span[1]} needs a year")
    return got.pop()


# --- rule sentences ---------------------------------------------------------------------------------------------

_PCT = r"(?P<pct>\d+(?:\.\d+)?)%"
_NP = r"[A-Za-z][A-Za-z' -]*?"
_WHEN = r"(?P<when>before|ahead of|after)"
_ONLY = r"(?:only|alone)"
_PERCENT_FORMS = [
    rf"(?:a|an) {_PCT} (?P<name>{_NP}) (?P<verb>comes off|is taken off|is deducted from|is subtracted from|applies to|"
    rf"is applied to|is charged on|is paid on|is added to|is added on) the (?P<target>{_NP}) {_ONLY}, {_WHEN} tax",
    rf"(?:a|an) (?P<name>{_NP}) of {_PCT} on the (?P<target>{_NP}) {_ONLY} is (?P<verb>added|deducted|charged|paid|"
    rf"taken off|subtracted) {_WHEN} tax",
    rf"(?P<verb>add|charge|deduct|take|subtract|pay) {_PCT} of the (?P<target>{_NP}) {_ONLY} as the (?P<name>{_NP}), "
    rf"{_WHEN} tax",
    rf"(?P<verb>add|charge|deduct|subtract|pay) (?:a|an) {_PCT} (?P<name>{_NP}) on the (?P<target>{_NP}) {_ONLY}, "
    rf"{_WHEN} tax",
    rf"{_WHEN} tax, (?P<verb>add|deduct|charge|take off|subtract|pay) (?:a|an) (?P<name>{_NP}) of {_PCT} of the "
    rf"(?P<target>{_NP}) {_ONLY}",
    rf"(?P<verb>take) (?:a|an) {_PCT} (?P<name>{_NP}) off the (?P<target>{_NP}) {_ONLY}, {_WHEN} tax",
    rf"the (?P<name>{_NP}) is {_PCT} of the (?P<target>{_NP}) and nothing else, (?P<verb>added|deducted|charged|paid|"
    rf"taken off) {_WHEN} tax",
    rf"there is (?:a|an) {_PCT} (?P<name>{_NP}) on the (?P<target>{_NP}) {_ONLY}, (?P<verb>added|deducted|charged|paid|"
    rf"taken off) {_WHEN} tax",
]
_PERCENT_FORMS = [re.compile(p, re.I) for p in _PERCENT_FORMS]
_PLUS_VERBS = {"add", "added", "charge", "charged", "is charged on", "is paid on", "paid", "pay", "is added to"}
_PLUS_VERBS |= {"is added on"}
_MINUS_VERBS = {"deduct", "deducted", "comes off", "is taken off", "take", "take off", "taken off", "subtract"}
_MINUS_VERBS |= {"subtracted", "is deducted from", "is subtracted from"}
_PLUS_NOUNS = {"surcharge", "premium", "markup", "uplift", "loading", "supplement"}
_MINUS_NOUNS = {"discount", "rebate", "reduction"}

_TAX_RATE = re.compile(
    r"\d(?:\.\d+)?% (?:income )?tax\b|\btax(?:ed)? (?:of|at) \d|\btax \(\d|\btax: \d|\bwithhold(?:ing)? \d|"
    r"\bsubject to \d+(?:\.\d+)?% (?:income )?tax|\bat \d+(?:\.\d+)?%(?:[,;]|$)",
    re.I,
)
_TAX_OUT = re.compile(
    r"leaving out|everything but|all but|all pay but|out of the taxed|not taxed|outside|the rest\b|untaxed|"
    r"after tax,|once tax is|\bexcept\b|\bexcluding\b",
    re.I,
)
_TAX_IN = re.compile(
    r"first[,;] then|after adding|into the taxed|are taxed:|inside tax|sits? inside|before adding|before withholding",
    re.I,
)
_TAX_MENTION = re.compile(
    r"\bthe ([a-z][a-z' -]*?)(?=\s+(?:and|at|are|is|sit|sits|out|into|first|before|then|after|once|untaxed|from|to|"
    r"on)\b|[,;:.]|$)"
)
_TAX_WHOLE = {
    _norm(x)
    for x in (
        "rest",
        "rest of the bill",
        "rest of the pay",
        "bill",
        "whole bill",
        "pay",
        "whole pay",
        "full pay",
        "total pay",
        "taxed pay",
        "result",
        "charge",
        "whole charge",
        "full charge",
        "total",
        "taxed amount",
        "end",
    )
}
_NAMED_END = r"(?= and take| and taking| and the |, take|, then|;| first| before| then|, untaxed|, after|$)"
_ADDS = re.compile(rf"\badd(?:ing)? the ({_NP}){_NAMED_END}", re.I)
_TAKES = re.compile(rf"\btak(?:e|ing) off the ({_NP}){_NAMED_END}", re.I)

_TIMING_FORMS = [
    (r"(?:deduct|subtract|take) the (?P<x>{np}) (?:off )?(?:before|ahead of) (?:tax|working out tax)", "inside"),
    (r"the (?P<x>{np}) (?:comes off|is taken off|is deducted|is subtracted) (?:before|ahead of) tax", "inside"),
    (r"the (?P<x>{np}) reduces the amount that is taxed", "inside"),
    (r"tax is worked out after the (?P<x>{np}) comes off", "inside"),
    (r"(?:deduct|subtract|take) the (?P<x>{np}) (?:off )?after tax", "outside"),
    (r"the (?P<x>{np}) (?:comes off|is taken off|is deducted|is subtracted) after tax", "outside"),
    (r"the (?P<x>{np}) comes off the taxed amount", "ambiguous"),
]
_TIMING_FORMS = [(re.compile(p.format(np=_NP), re.I), where) for p, where in _TIMING_FORMS]


def _sentences(line: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", line.strip()) if s]


def _classify(sentence: str, rules: _Rules, text: _Text) -> None:
    s = sentence.strip()
    body = s.rstrip(".").strip()
    low = body.lower()
    if text.currency and re.search(text.money("x"), body):
        raise Unsure("unparsed-rule", "an amount inside a rule sentence")
    digits = _has_number(body)

    # a definition of the figure asked about
    if (
        not digits
        and re.search(r"\b(?:these rules|the rules below|rules (?:below )?apply|rules are applied)\b", low)
        and re.search(r"\b(?:means?|we mean|is what|is the amount)\b", low)
        and not re.search(
            r"\b(?:tax\w*|discount\w*|surcharge\w*|credit\w*|fee|fees|round\w*|percent\w*|less|minus|plus|deposit\w*|"
            r"except|exclud\w*|includ\w*|without|refund\w*|already|unless|but)\b",
            low,
        )
    ):
        return
    # the final rounding to the cent: read, not applied
    if re.search(r"\bcents?\b", low):
        if (
            not digits
            and re.search(r"\b(?:final|last|end result|end|the result)\b", low)
            and not re.search(r"\b(?:after|for) (?:each|every)\b|\b(?:each|every) (?:step|line|item|amount)\b", low)
        ):
            return
        raise Unsure("unsupported-rule", "money rounded before the final figure")
    # prices applied to quantities as printed, with no conversion: no line form for it in v1
    if re.search(r"\bconver\w*", low) and re.search(r"\b(?:no|not|never|without)\b|as printed", low):
        raise Unsure("unsupported-rule", "quantities priced as printed, no conversion")
    # unit factors: `(1 kg = 2.2046 lb; 1 L = 0.2642 gal)`
    if m := re.search(r"\(([^()]*=[^()]*)\)", body):
        outer = (body[: m.start()] + body[m.end() :]).lower()
        if _has_number(outer) or not re.search(r"\b(?:convert\w*|price[sd]?|priced|quoted)\b", outer):
            raise Unsure("unparsed-rule", "unit factors")
        named = set()
        for part in m.group(1).split(";"):
            fm = re.fullmatch(rf"\s*({_NUM}) ?([A-Za-z]+) = ({_NUM}) ?([A-Za-z]+)\s*", part)
            if not fm or fm.group(2) not in _UNIT or fm.group(4) not in _UNIT:
                raise Unsure("unparsed-rule", "unit factor")
            a, ua, b, ub = _fr(fm.group(1)), _UNIT[fm.group(2)], _fr(fm.group(3)), _UNIT[fm.group(4)]
            if a == 0 or b == 0 or ua == ub:
                raise Unsure("unparsed-rule", "unit factor")
            if _DIMENSION[ua] != _DIMENSION[ub]:
                raise Unsure("conflicting-rules", f"a factor from {ua} ({_DIMENSION[ua]}) to {ub} ({_DIMENSION[ub]})")
            f = b / a
            if rules.factors.get((ua, ub), f) != f:
                raise Unsure("conflicting-rules", f"two factors for {ua} to {ub}")
            rules.factors[(ua, ub)] = f
            named |= {ua, ub}
        for word in re.findall(r"\b(?:per|for one|one) ([a-z]+)", outer):
            if _UNIT.get(word) not in named and word not in ("unit", "item"):
                raise Unsure("unparsed-rule", f"prices per {word} with no factor to it")
        return
    # whether a date range counts its last day
    if not digits and re.search(r"\b(?:date|dates|range|ranges|days are|day counts?)\b", low) and "tax" not in low:
        if re.search(r"\bneither\b", low):
            raise Unsure("unsupported-rule", "neither date counts")
        if re.search(
            r"\b(?:weekends?|weekdays?|working|business|holidays?|sundays?|saturdays?|except|unless|half|only|"
            r"minimum|maximum|least|most)\b",
            low,
        ):
            raise Unsure("unparsed-rule", "a day count with a condition")
        if re.search(r"\bnot\b|n't\b|left out|\bminus\b|\bexclud|\bexclusive", low):
            rules.day_count.add("exclusive")
            return
        if re.search(
            r"\b(?:both|inclusive(?:ly)?|through)\b|\bfirst and (?:its |the )?last\b|\bstart and (?:its |the )?end\b",
            low,
        ):
            rules.day_count.add("inclusive")
            return
        raise Unsure("unparsed-rule", "day count")
    # time or length rounding to a stated step
    if "nearest" in low:
        rules.roundings.append(_rounding(low))
        return
    # a percentage of one named item, before or after tax
    if "%" in body:
        for pat in _PERCENT_FORMS:
            if pm := pat.fullmatch(body):
                rules.percents.append(_percent(pm))
                return
        if _TAX_RATE.search(body) and not re.search(r"\b(?:discount|surcharge|premium|sacrifice|contribution)", low):
            rules.taxes.append(_tax(body, text))
            return
        raise Unsure("unparsed-rule", "a percentage")
    # when a credit comes off
    if not digits:
        for pat, where in _TIMING_FORMS:
            if tm := pat.fullmatch(body):
                if where == "ambiguous":
                    raise Unsure("ambiguous-rule", "comes off the taxed amount: before or after tax")
                rules.timings.append((tm.group("x"), where))
                return
    if digits or _RULE_WORDS.search(body):
        raise Unsure("unparsed-rule", "no rule kind")


def _rounding(low: str) -> tuple[str, Fraction, bool]:
    m = re.search(
        rf"nearest (?:(?P<n>{_NUM}) ?(?P<u>minutes?|mins?|hours?|h|m|metres?|meters?|cm|ft|feet)\b|"
        r"(?P<w>quarter[- ]hour|half[- ]hour|hour|whole hour|whole metre|whole meter|metre|meter|"
        r"quarter (?:of a )?metre|half (?:a )?metre|tenth of a metre))",
        low,
    )
    if not m or len(re.findall(r"\d+(?:\.\d+)?", _ID.sub("", low))) != (1 if m.group("n") else 0):
        raise Unsure("unparsed-rule", "rounding step")
    words = {
        "quarter hour": ("time", Fraction(15)),
        "quarter-hour": ("time", Fraction(15)),
        "half hour": ("time", Fraction(30)),
        "half-hour": ("time", Fraction(30)),
        "hour": ("time", Fraction(60)),
        "whole hour": ("time", Fraction(60)),
        "whole metre": ("length", Fraction(1)),
        "whole meter": ("length", Fraction(1)),
        "metre": ("length", Fraction(1)),
        "meter": ("length", Fraction(1)),
        "quarter metre": ("length", Fraction(1, 4)),
        "quarter of a metre": ("length", Fraction(1, 4)),
        "half metre": ("length", Fraction(1, 2)),
        "half a metre": ("length", Fraction(1, 2)),
        "tenth of a metre": ("length", Fraction(1, 10)),
    }
    if m.group("w"):
        dim, step = words[m.group("w")]
    else:
        n, u = _fr(m.group("n")), m.group("u")
        if u.startswith(("min", "hour")) or u == "h":
            dim, step = "time", n * (60 if u.startswith("hour") or u == "h" else 1)
        elif u in ("m", "metre", "metres", "meter", "meters"):
            dim, step = "length", n
        else:
            raise Unsure("unsupported-rule", f"rounding to {u}")
    if step <= 0:
        raise Unsure("unparsed-rule", "rounding step")
    if re.search(r"\bround(?:ed|ing)? (?:up|down)\b|\b(?:up|down) to the\b", low):
        raise Unsure("unsupported-rule", "rounding up or down")
    if re.search(
        r"\b(?:minimum|maximum|least|most|except|unless|if|but|cap|capped|free|plus|extra|additional|partial|started|"
        r"charge|charged|tax\w*|discount\w*|credit\w*|fee|fees)\b",
        low,
    ):
        raise Unsure("unparsed-rule", "a rounding with a condition")
    total = bool(re.search(r"\b(?:total|summed|combined|once|sum)\b", low))
    each = bool(re.search(r"\b(?:each|every|one by one|on its own|per)\b", low))
    if not (total or each):
        raise Unsure("unparsed-rule", "rounding: each or once")
    return dim, step, not total


def _percent(m: re.Match) -> tuple[str, Fraction, str, int]:
    if m.group("when").lower() == "after":
        raise Unsure("unsupported-rule", "a percentage adjustment after tax")
    verb = m.group("verb").lower()
    vs = 1 if verb in _PLUS_VERBS else -1 if verb in _MINUS_VERBS else 0
    noun = m.group("name").split()[-1].lower()
    ns = 1 if noun in _PLUS_NOUNS else -1 if noun in _MINUS_NOUNS else 0
    if vs and ns and vs != ns:
        raise Unsure("ambiguous-rule", f"{verb} a {noun}")
    sign = vs or ns
    if not sign:
        raise Unsure("ambiguous-rule", f"is the {noun} added or taken off")
    target = m.group("target")
    if re.search(r"\b(?:and|or)\b|,", target):
        raise Unsure("unsupported-rule", "a percentage of several items")
    return m.group("name").strip(), _fr(m.group("pct")), target.strip(), sign


def _tax(body: str, text: _Text) -> tuple[Fraction, str | None, list[str], list[str], str | None]:
    pcts = re.findall(r"(\d+(?:\.\d+)?)%", body)
    if len(pcts) != 1 or len(re.findall(r"\d+(?:\.\d+)?", _ID.sub("", body))) != 1:
        raise Unsure("unparsed-rule", "tax rate")
    adds = [a.strip() for a in _ADDS.findall(body)]
    takes = [t.strip() for t in _TAKES.findall(body)]
    # what the tax is charged on or withheld from, read with the named fees and credits taken out ("parking charge")
    low = body.lower()
    for phrase in sorted(set(adds + takes), key=len, reverse=True):
        low = low.replace(phrase.lower(), "_")
    kind = None
    if re.search(r"\bwithh", low):
        kind = "withheld"
    if re.search(
        r"\badd(?:ed|ing)? (?:\d+(?:\.\d+)?% )?(?:income )?tax\b|\btax\b[^;:]*?\badded\b|\bon top\b|\bcarries\b", low
    ):
        if kind is not None:
            raise Unsure("ambiguous-rule", "tax both added and withheld")
        kind = "charged"
    if kind is None:
        pay = bool(re.search(r"\b(?:the|all|whole|full|total|taxed) pay\b|\bincome tax comes off\b", low))
        bill = bool(re.search(r"\b(?:charge|charges|bill)\b", low))
        if pay != bill:
            kind = "withheld" if pay else "charged"
    named = {_norm(x) for x in adds + takes}
    for np in _TAX_MENTION.findall(body.lower()):
        if _norm(np) not in named and _norm(np) not in _TAX_WHOLE:
            raise Unsure("unparsed-rule", "tax names a part of the bill that is no fee or credit")
    if re.search(
        r"\b(?:except|exclud\w*|apart|other than|only|exempt\w*|zero|half|reduced|unless|if|when|not on)\b", low
    ):
        raise Unsure("unparsed-rule", "a tax rule with a condition")
    out, inn = bool(_TAX_OUT.search(body)), bool(_TAX_IN.search(body))
    if not adds and not takes:
        if out or inn:
            raise Unsure("unparsed-rule", "tax names a part of the bill but no fee or credit")
        where = None
    elif out == inn:
        raise Unsure("ambiguous-rule", "fees and credits before or after tax")
    else:
        where = "outside" if out else "inside"
    return _fr(pcts[0]), kind, adds, takes, where


# --- putting it together ----------------------------------------------------------------------------------------


def _round_to(x: Fraction, step: Fraction) -> Fraction:
    q = x / step
    lo = q.numerator // q.denominator
    rest = q - lo
    if rest == Fraction(1, 2):
        raise Unsure("ambiguous-rule", "a rounding falls exactly half way")
    return (lo + (1 if rest > Fraction(1, 2) else 0)) * step


def _match(raws: list[_Raw], phrase: str, forms: set[str] | None = None) -> list[_Raw]:
    key = _norm(phrase)
    return [r for r in raws if _norm(r.label) == key and (forms is None or r.form in forms)]


def _one(found: list[_Raw], phrase: str) -> _Raw:
    if not found:
        raise Unsure("unmatched-label", phrase)
    if len(found) > 1:
        raise Unsure("ambiguous-label", phrase)
    return found[0]


def _respell(raw: _Raw, phrase: str, spell: dict[int, str]) -> None:
    """A JSON field name lost its hyphens; take them back from a rule that names the item in the same words."""
    if not raw.from_key:
        return
    pw = re.split(r"([ -])", phrase.strip())
    kw = raw.label.split()
    words, seps = pw[0::2], pw[1::2]
    if words and words[0].lower() == "the":
        words, seps = words[1:], seps[1:]
    if len(words) != len(kw) or [_sing(w) for w in words] != [_sing(w) for w in kw]:
        return
    spell[raw.order] = "".join(k + (seps[i] if i < len(seps) else "") for i, k in enumerate(kw))


_PRICED = {"count", "convert", "days", "list"}
# names that say an amount is not a plain charge, whatever form it is written in (a rule has to place it)
_NOT_A_CHARGE = {
    _sing(w)
    for w in """discount discounts credit credits refund refunds rebate rebates deduction deductions deposit deposits
    voucher vouchers return returns reduction reductions waiver adjustment adjustments tax taxes vat gst total
    subtotal balance payment payments prepaid advance paid saving savings offset""".split()
}


def parse(parts: list[str]) -> Case:
    """The case a text states (the task's context, if any, then the rendered state), or `Unsure`."""
    text = _Text(parts)
    if not _ANY_MONEY.search(text.full):
        raise Unsure("no-arithmetic")
    if not _english(text.full):
        raise Unsure("not-english")
    if len(text.currencies) > 1:
        raise Unsure("mixed-currency", ", ".join(sorted(text.currencies)))

    raws: list[_Raw] = []
    rules = _Rules()
    headers = 0
    for part in parts:
        for line in _text_lines(part):
            candidate = line.label is not None and line.detail is not None
            if candidate and len(line.label) <= 60 and len(line.label.split()) <= 6:
                raw = _item(text, line.label, line.detail, len(raws))
                if raw is not None:
                    raw.from_key = line.from_key
                    raws.append(raw)
                    continue
            if _ID.search(line.raw):
                headers += 1
                if headers > 1:
                    raise Unsure("several-records")
            try:
                for s in _sentences(line.raw):
                    _classify(s, rules, text)
            except Unsure:
                if candidate and re.search(text.money("x"), line.detail):
                    raise Unsure("unparsed-item", line.label) from None
                raise
    return _resolve(text, raws, rules)


def _resolve(text: _Text, raws: list[_Raw], rules: _Rules) -> Case:
    if not any(r.form in _PRICED or r.form == "daycredit" for r in raws):
        raise Unsure("no-arithmetic", "no priced item")
    norms = [_norm(r.label) for r in raws]
    if len(set(norms)) != len(norms):
        raise Unsure("ambiguous-label", "two items share a name")
    for r in raws:
        if r.form in _PRICED and set(_words(r.label)) & _NOT_A_CHARGE:
            raise Unsure("unaccounted-amount", f"{r.label} is priced but its name says it is not a plain charge")
    spell: dict[int, str] = {}

    # time and length rounding
    rounding: dict[int, tuple[Fraction, bool]] = {}
    for dim, step, each in rules.roundings:
        cands = [r for r in raws if r.form == "list" and r.dim == dim]
        target = _one(cands, f"the {dim} to round")
        if target.order in rounding and rounding[target.order] != (step, each):
            raise Unsure("conflicting-rules", "two roundings for one item")
        rounding[target.order] = (step, each)

    # the tax rule
    if not rules.taxes:
        raise Unsure("no-tax-rule")
    if len({t[:2] + (tuple(t[2]), tuple(t[3]), t[4]) for t in rules.taxes}) > 1:
        raise Unsure("conflicting-rules", "two tax rules")
    rate, kind, adds, takes, where = rules.taxes[0]
    if kind is None:
        pay = bool(
            re.search(r"\b(?:payslip|pay slip|employee|wages?|salary|take-home|takes home|net pay)\b", text.full, re.I)
        )
        bill = bool(
            re.search(
                r"\b(?:invoice|bill|customer|client|shipper|consignment|booking|order|job|quote|owe[sd]?|owing)\b",
                text.full,
                re.I,
            )
        )
        if pay == bill:
            raise Unsure("ambiguous-rule", "tax charged or withheld")
        kind = "withheld" if pay else "charged"

    role: dict[int, str] = {}  # order -> "fee" | "credit"
    place: dict[int, str] = {}  # order -> "inside" | "outside"

    def put(raw: _Raw, where_: str) -> None:
        if place.get(raw.order, where_) != where_:
            raise Unsure("conflicting-rules", f"{raw.label} both before and after tax")
        place[raw.order] = where_

    for phrase in adds:
        raw = _one(_match(raws, phrase), phrase)
        _respell(raw, phrase, spell)
        if raw.form not in ("feecount", "feeper", "lone", "count") or role.get(raw.order, "fee") != "fee":
            raise Unsure("unsupported-rule", f"{raw.label} added as a fee")
        role[raw.order] = "fee"
        put(raw, where)
    for phrase in takes:
        raw = _one(_match(raws, phrase), phrase)
        _respell(raw, phrase, spell)
        if raw.form not in ("lone", "daycredit") or role.get(raw.order, "credit") != "credit":
            raise Unsure("unsupported-rule", f"{raw.label} taken off as a credit")
        if raw.form == "lone":
            role[raw.order] = "credit"
        put(raw, where)

    # when a credit comes off
    for phrase, where_ in rules.timings:
        exact = _match(raws, phrase, {"lone", "daycredit"})
        cands = exact or [r for r in raws if r.form == "daycredit" and _contains(phrase, r.label)]
        raw = _one(cands, phrase)
        if exact:
            _respell(raw, phrase, spell)
        if raw.form == "lone":
            if role.get(raw.order, "credit") != "credit":
                raise Unsure("conflicting-rules", f"{raw.label} both added and taken off")
            role[raw.order] = "credit"
        put(raw, where_)

    # percentages of named items
    adjustments = []
    for name, pct, target_phrase, sign in rules.percents:
        found = _match(raws, target_phrase)
        raw = _one(found, target_phrase)
        if raw.form not in _PRICED or raw.order in role:
            raise Unsure("unsupported-rule", f"a percentage of {raw.label}")
        _respell(raw, target_phrase, spell)
        adjustments.append((name, pct, raw, sign))

    # every amount accounted for
    day_rule = None
    if len(rules.day_count) > 1:
        raise Unsure("conflicting-rules", "two day-count rules")
    if rules.day_count:
        day_rule = next(iter(rules.day_count)) == "inclusive"
    for raw in raws:
        if raw.form in ("feecount", "feeper", "lone") and raw.order not in role:
            raise Unsure("unaccounted-amount", raw.label)
        if raw.form == "daycredit" and raw.order not in place:
            raise Unsure("missing-rule", f"when {raw.label} comes off")
        if raw.span is not None and day_rule is None:
            raise Unsure("missing-rule", "whether a date range counts its last day")
        if raw.form == "convert" and (raw.unit, raw.to_unit) not in rules.factors:
            if (raw.to_unit, raw.unit) not in rules.factors:
                raise Unsure("missing-rule", f"no factor from {raw.unit} to {raw.to_unit}")
        if raw.form in _PRICED and raw.order in place and raw.order not in role:
            raise Unsure("unsupported-rule", f"{raw.label} placed after tax")

    def label(raw: _Raw) -> str:
        return _cap(spell.get(raw.order, raw.label))

    items, fees, credits, outside = [], [], [], set()
    for raw in raws:
        lab = label(raw)
        if place.get(raw.order) == "outside":
            outside.add(lab)
        if role.get(raw.order) == "fee":
            if raw.form == "feeper":
                lists = [r for r in raws if r.form == "list" and _norm(r.label) == _norm(raw.per)]
                count = Fraction(len(_one(lists, f"{raw.per} to count").entries))
            elif raw.form in ("feecount", "count"):
                count = raw.q
            else:
                count = None
            fees.append(Fee(lab, raw.price, count))
        elif role.get(raw.order) == "credit":
            credits.append(Credit(lab, raw.price))
        elif raw.form == "count":
            items.append(Item(lab, "count", quantity=raw.q, price=raw.price))
        elif raw.form == "convert":
            f = rules.factors.get((raw.unit, raw.to_unit))
            if f is None:
                f = 1 / rules.factors[(raw.to_unit, raw.unit)]
            items.append(
                Item(lab, "convert", quantity=raw.q, unit=raw.unit, price=raw.price, to_unit=raw.to_unit, factor=f)
            )
        elif raw.form == "days":
            days = raw.days if raw.span is None else _span_days(raw.span, day_rule)
            items.append(Item(lab, "days", quantity=raw.q, price=raw.price, days=days))
        elif raw.form == "daycredit":
            items.append(Item(lab, "daycredit", price=raw.price, days=_span_days(raw.span, day_rule)))
        elif raw.form == "list":
            vals = raw.entries
            if raw.order in rounding:
                step, each = rounding[raw.order]
                total = sum((_round_to(v, step) for v in vals), Fraction(0)) if each else _round_to(sum(vals), step)
            else:
                total = sum(vals, Fraction(0))
            if raw.dim == "time":
                items.append(Item(lab, "time", quantity=total, price=raw.price, unit=raw.unit))
            else:
                items.append(Item(lab, "length", quantity=total, price=raw.price, unit=raw.unit))
    by_order = {r.order: label(r) for r in raws}
    adj = tuple(Adjustment(_cap(n), p, by_order[r.order], s) for n, p, r, s in adjustments)
    taken = (
        {i.label.casefold() for i in items} | {f.label.casefold() for f in fees} | {c.label.casefold() for c in credits}
    )
    if any(a.label.casefold() in taken for a in adj) or len({a.label.casefold() for a in adj}) != len(adj):
        raise Unsure("ambiguous-label", "an adjustment shares a name")
    if not items:
        raise Unsure("no-arithmetic", "no priced item")
    return Case(
        currency=text.currency,
        items=tuple(items),
        tax=Tax(rate, kind),
        adjustments=adj,
        fees=tuple(fees),
        credits=tuple(credits),
        outside=frozenset(outside),
    )
