# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The structured client input (`calculator_input`): amounts a client already holds, rendered as the same block.

No text is parsed; the input is validated strictly and is all or nothing like the parser: any invalid or inconsistent
field raises `CalculatorInputError` with the field's path (`items[2].price`), and no block is made.

Numbers are JSON integers or decimal strings (`"299.46"`, `"12.5"`); a JSON float is refused, since it is not exact.
Unknown fields are refused. Labels are the client's own words, written as they should appear (1 to 80 characters, no
colon, no line break, unique ignoring case; `Taxed base`, `Tax charged` and `Tax withheld` are reserved).
Bounds (each a CalculatorInputError naming the field): at most 64 items, 32 adjustments, 32 fees, 32 credits and 32
labels in tax.outside; at most 64 minutes or lengths in one item; a number has at most 14 digits before the point and
6 after (a JSON integer is under 10^14); a day count is 1 to 3,660.

  {
    "currency": "$",                    a symbol ($ € £ ¥) or a three-letter code ("EUR"), written before amounts
    "items": [                          at least one; the priced items in the order the block lists them
      {"kind": "count", "label": "Attachments", "quantity": "4", "price": "299.46"},
      {"kind": "days", "label": "Hire charge", "quantity": 1, "price": "52.81", "days": 6},
      {"kind": "time", "label": "Operator time", "minutes": [41, 418, 317], "rate": "92.74", "per": "h",
       "round": {"step": 15, "mode": "each"}},         per: "h" or "min"; round is optional, its step in minutes
      {"kind": "length", "label": "Cut pieces", "lengths": ["4.21", "4.70"], "unit": "m", "rate": "38.87",
       "round": {"step": "0.25", "mode": "total"}},    the rate is per `unit`; the step is in `unit`
      {"kind": "convert", "label": "Goods weight", "quantity": "1605", "unit": "lb", "to_unit": "kg",
       "factor": "0.4536", "price": "0.20"},           one `unit` is `factor` `to_unit`; the price is per `to_unit`
      {"kind": "daycredit", "label": "Late delivery", "price": "54.35", "days": 5}   taken off: price x days
    ],
    "adjustments": [                    optional; a percentage of one priced item, before tax
      {"label": "Volume discount", "percent": "12", "of": "Linehaul", "direction": "deduct"}   or "add"
    ],
    "fees": [                           optional; added: price x count, or the price once without a count
      {"label": "Handling fee", "price": "11.31", "count": 7}
    ],
    "credits": [                        optional; taken off
      {"label": "Account credit", "amount": "63.15"}
    ],
    "tax": {
      "rate": "15",                     a percentage, 0 to 100
      "kind": "charged",                "charged" (added to a bill) or "withheld" (taken from pay)
      "outside": ["Handling fee", "Account credit"]   optional: fees, credits and day credits applied after tax;
    }                                                 everything else is inside the taxed base
  }

Rounding (`round`) goes to the nearest step, per entry (`"mode": "each"`) or once on the sum (`"total"`); a value
exactly half way is refused, since the direction is not stated.
"""

from __future__ import annotations

import re
from fractions import Fraction

from decisio.calculator.compute import Adjustment, Case, Credit, Fee, Item, Tax

_DEC = re.compile(r"-?(\d+)(?:\.(\d+))?")
# hard bounds, so that every accepted input is computed and written in bounded time and size: list lengths, and the
# digits a number may have (an amount up to 10^14 with six decimals; a JSON integer under 10^14)
MAX_ITEMS = 64
MAX_LIST = 32  # adjustments, fees, credits, tax.outside
MAX_ENTRIES = 64  # minutes or lengths of one item
MAX_INT_DIGITS, MAX_DEC_DIGITS = 14, 6
MAX_DAYS = 3660
_RESERVED = {"taxed base", "tax charged", "tax withheld"}
_ITEM_FIELDS = {
    "count": ({"kind", "label", "quantity", "price"}, set()),
    "days": ({"kind", "label", "quantity", "price", "days"}, set()),
    "time": ({"kind", "label", "minutes", "rate", "per"}, {"round"}),
    "length": ({"kind", "label", "lengths", "unit", "rate"}, {"round"}),
    "convert": ({"kind", "label", "quantity", "unit", "to_unit", "factor", "price"}, set()),
    "daycredit": ({"kind", "label", "price", "days"}, set()),
}


class _Reader:
    def __init__(self, err):
        self.err = err
        self.labels: set[str] = set()

    def fail(self, path: str, why: str):
        raise self.err(f"calculator_input.{path}: {why}")

    def obj(self, x, path: str, required: set[str], optional: set[str] = frozenset()) -> dict:
        if not isinstance(x, dict):
            self.fail(path, "must be an object")
        for k in x:
            if k not in required and k not in optional:
                self.fail(f"{path}.{k}" if path else k, "unknown field")
        for k in sorted(required):
            if k not in x:
                self.fail(f"{path}.{k}" if path else k, "required")
        return x

    def num(self, x, path: str, *, min_: Fraction | None = Fraction(0), positive: bool = False) -> Fraction:
        if isinstance(x, bool) or not isinstance(x, (int, str)):
            self.fail(path, "must be an integer or a decimal string (a float is not exact)")
        if isinstance(x, int) and abs(x) >= 10**MAX_INT_DIGITS:
            self.fail(path, f"must be under 10^{MAX_INT_DIGITS}")
        if isinstance(x, str):
            m = _DEC.fullmatch(x) if len(x) <= MAX_INT_DIGITS + MAX_DEC_DIGITS + 2 else None
            if not m:
                self.fail(
                    path, f"is not a decimal number of at most {MAX_INT_DIGITS} digits and {MAX_DEC_DIGITS} decimals"
                )
            if len(m.group(1)) > MAX_INT_DIGITS or len(m.group(2) or "") > MAX_DEC_DIGITS:
                self.fail(
                    path, f"must have at most {MAX_INT_DIGITS} digits before the point and {MAX_DEC_DIGITS} after"
                )
        v = Fraction(x)
        if positive and v <= 0:
            self.fail(path, "must be greater than 0")
        if min_ is not None and v < min_:
            self.fail(path, f"must be at least {min_}")
        return v

    def whole(self, x, path: str, min_: int = 0, max_: int = MAX_DAYS) -> int:
        if isinstance(x, bool) or not isinstance(x, int) or not min_ <= x <= max_:
            self.fail(path, f"must be a whole number from {min_} to {max_}")
        return x

    def text(self, x, path: str) -> str:
        if not isinstance(x, str) or not x.strip():
            self.fail(path, "must be non-empty text")
        return x.strip()

    def label(self, x, path: str) -> str:
        lab = self.text(x, path)
        if len(lab) > 80 or ":" in lab or any(ord(c) < 32 for c in lab):
            self.fail(path, "a label is 1 to 80 characters with no colon or line break")
        if lab.casefold() in _RESERVED:
            self.fail(path, f"{lab!r} is reserved for the block's own lines")
        if lab.casefold() in self.labels:
            self.fail(path, f"{lab!r} is used twice")
        self.labels.add(lab.casefold())
        return lab

    def unit(self, x, path: str) -> str:
        u = self.text(x, path)
        if not re.fullmatch(r"[A-Za-z]{1,12}", u):
            self.fail(path, "a unit is 1 to 12 letters")
        return u

    def seq(self, x, path: str, *, nonempty: bool = False, most: int = MAX_LIST) -> list:
        if not isinstance(x, list) or (nonempty and not x):
            self.fail(path, "must be a non-empty list" if nonempty else "must be a list")
        if len(x) > most:
            self.fail(path, f"has {len(x)} entries; at most {most}")
        return x


def _rounded(rd: _Reader, values: list[Fraction], spec, path: str) -> Fraction:
    if spec is None:
        return sum(values, Fraction(0))
    rd.obj(spec, path, {"step", "mode"})
    step = rd.num(spec["step"], f"{path}.step", positive=True)
    mode = spec["mode"]
    if mode not in ("each", "total"):
        rd.fail(f"{path}.mode", 'must be "each" or "total"')

    def near(x: Fraction) -> Fraction:
        q = x / step
        lo = q.numerator // q.denominator
        if q - lo == Fraction(1, 2):
            rd.fail(path, "a value falls exactly half way between two steps; the direction is not stated")
        return (lo + (1 if q - lo > Fraction(1, 2) else 0)) * step

    return sum((near(v) for v in values), Fraction(0)) if mode == "each" else near(sum(values, Fraction(0)))


def _item(rd: _Reader, x, path: str) -> Item:
    if not isinstance(x, dict):
        rd.fail(path, "must be an object")
    kind = x.get("kind")
    if not isinstance(kind, str) or kind not in _ITEM_FIELDS:
        rd.fail(f"{path}.kind", f"must be one of {', '.join(_ITEM_FIELDS)}")
    required, optional = _ITEM_FIELDS[kind]
    rd.obj(x, path, required, optional)
    lab = rd.label(x["label"], f"{path}.label")
    if kind == "count":
        return Item(
            lab, kind, quantity=rd.num(x["quantity"], f"{path}.quantity"), price=rd.num(x["price"], f"{path}.price")
        )
    if kind == "days":
        return Item(
            lab,
            kind,
            quantity=rd.num(x["quantity"], f"{path}.quantity"),
            price=rd.num(x["price"], f"{path}.price"),
            days=rd.whole(x["days"], f"{path}.days", 1),
        )
    if kind == "time":
        mins = [
            rd.num(v, f"{path}.minutes[{i}]")
            for i, v in enumerate(rd.seq(x["minutes"], f"{path}.minutes", nonempty=True, most=MAX_ENTRIES))
        ]
        if x["per"] not in ("h", "min"):
            rd.fail(f"{path}.per", 'must be "h" or "min"')
        total = _rounded(rd, mins, x.get("round"), f"{path}.round")
        return Item(lab, kind, quantity=total, price=rd.num(x["rate"], f"{path}.rate"), unit=x["per"])
    if kind == "length":
        lens = [
            rd.num(v, f"{path}.lengths[{i}]")
            for i, v in enumerate(rd.seq(x["lengths"], f"{path}.lengths", nonempty=True, most=MAX_ENTRIES))
        ]
        unit = rd.unit(x["unit"], f"{path}.unit")
        total = _rounded(rd, lens, x.get("round"), f"{path}.round")
        return Item(lab, kind, quantity=total, price=rd.num(x["rate"], f"{path}.rate"), unit=unit)
    if kind == "convert":
        unit, to_unit = rd.unit(x["unit"], f"{path}.unit"), rd.unit(x["to_unit"], f"{path}.to_unit")
        if unit == to_unit:
            rd.fail(f"{path}.to_unit", "is the same unit; use a count item")
        return Item(
            lab,
            kind,
            quantity=rd.num(x["quantity"], f"{path}.quantity"),
            unit=unit,
            to_unit=to_unit,
            factor=rd.num(x["factor"], f"{path}.factor", positive=True),
            price=rd.num(x["price"], f"{path}.price"),
        )
    return Item(lab, kind, price=rd.num(x["price"], f"{path}.price"), days=rd.whole(x["days"], f"{path}.days", 1))


def case_from_input(data, err) -> Case:
    """The case a structured input describes; `err` is the exception class raised on any invalid field."""
    rd = _Reader(err)
    rd.obj(data, "", {"currency", "items", "tax"}, {"adjustments", "fees", "credits"})
    cur = data["currency"]
    if not isinstance(cur, str) or not (cur in ("$", "€", "£", "¥") or re.fullmatch(r"[A-Z]{3}", cur)):
        rd.fail("currency", 'must be "$", "€", "£", "¥" or a three-letter code')
    items = [
        _item(rd, x, f"items[{i}]") for i, x in enumerate(rd.seq(data["items"], "items", nonempty=True, most=MAX_ITEMS))
    ]
    priced = {i.label: i for i in items if i.kind != "daycredit"}

    adjustments = []
    for i, x in enumerate(rd.seq(data.get("adjustments", []), "adjustments")):
        p = f"adjustments[{i}]"
        rd.obj(x, p, {"label", "percent", "of", "direction"})
        lab = rd.label(x["label"], f"{p}.label")
        target = rd.text(x["of"], f"{p}.of")
        if target not in priced:
            rd.fail(f"{p}.of", f"{target!r} is not a priced item's label")
        if x["direction"] not in ("add", "deduct"):
            rd.fail(f"{p}.direction", 'must be "add" or "deduct"')
        pct = rd.num(x["percent"], f"{p}.percent", positive=True)
        adjustments.append(Adjustment(lab, pct, target, 1 if x["direction"] == "add" else -1))

    fees = []
    for i, x in enumerate(rd.seq(data.get("fees", []), "fees")):
        p = f"fees[{i}]"
        rd.obj(x, p, {"label", "price"}, {"count"})
        lab = rd.label(x["label"], f"{p}.label")
        count = rd.num(x["count"], f"{p}.count", positive=True) if "count" in x else None
        fees.append(Fee(lab, rd.num(x["price"], f"{p}.price"), count))

    credits = []
    for i, x in enumerate(rd.seq(data.get("credits", []), "credits")):
        p = f"credits[{i}]"
        rd.obj(x, p, {"label", "amount"})
        credits.append(Credit(rd.label(x["label"], f"{p}.label"), rd.num(x["amount"], f"{p}.amount")))

    tax = rd.obj(data["tax"], "tax", {"rate", "kind"}, {"outside"})
    rate = rd.num(tax["rate"], "tax.rate")
    if rate > 100:
        rd.fail("tax.rate", "must be at most 100")
    if tax["kind"] not in ("charged", "withheld"):
        rd.fail("tax.kind", 'must be "charged" or "withheld"')
    movable = {f.label for f in fees} | {c.label for c in credits} | {i.label for i in items if i.kind == "daycredit"}
    outside = set()
    for i, x in enumerate(rd.seq(tax.get("outside", []), "tax.outside")):
        lab = rd.text(x, f"tax.outside[{i}]")
        if lab not in movable:
            rd.fail(f"tax.outside[{i}]", f"{lab!r} is not a fee, credit or day credit label")
        if lab in outside:
            rd.fail(f"tax.outside[{i}]", f"{lab!r} is listed twice")
        outside.add(lab)
    return Case(
        currency=cur,
        items=tuple(items),
        tax=Tax(rate, tax["kind"]),
        adjustments=tuple(adjustments),
        fees=tuple(fees),
        credits=tuple(credits),
        outside=frozenset(outside),
    )
