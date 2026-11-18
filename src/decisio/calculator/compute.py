# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The calculator's case: what a state's arithmetic is, as exact numbers, and the block's lines it gives.

The parser (`decisio.calculator.parse`) and the structured input (`decisio.calculator.inputs`) both build a `Case`;
`lines` is the only code that turns one into the block, so the two routes cannot drift.

Order of the lines: the priced items in the order the state lists them (a day credit, which is taken off, keeps its
place among them), then the percentage adjustments, then the fees, then the credits, then `Taxed base` and the tax
line. The final total is never computed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from decisio.calculator import render

# item kinds and the line each one gives
#   count      `Attachments: 4 x $299.46 = $1,197.84`
#   days       `Hire charge: 1 x $52.81 x 6 days = $316.86`
#   time       `Operator time: 776 min x $92.74/h = $1,199.437333` (minutes after any stated rounding)
#   length     `Cut pieces: 9 m x $42.94/m = $386.46` (the summed length after any stated rounding)
#   convert    `Goods weight: 1605 lb = 728.028 kg; 728.028 x $0.20 = $145.6056`
#   daycredit  `Late delivery: $54.35 x 5 days = $271.75 (taken off)`
ITEM_KINDS = ("count", "days", "time", "length", "convert", "daycredit")


@dataclass(frozen=True)
class Item:
    label: str
    kind: str
    quantity: Fraction = Fraction(0)  # count, days, convert: the quantity; time: minutes; length: the summed length
    price: Fraction = Fraction(0)  # the unit price, the rate per hour or per minute, or the credit per day
    unit: str = ""  # time: the rate's unit ("h" or "min"); length: the length unit; convert: the stated unit
    days: int = 0  # days, daycredit
    to_unit: str = ""  # convert: the price's unit
    factor: Fraction = Fraction(1)  # convert: how many `to_unit` one `unit` is

    @property
    def converted(self) -> Fraction:
        return self.quantity * self.factor

    @property
    def value(self) -> Fraction:
        """The line's value, positive; a day credit is taken off the taxed base (`signed`)."""
        if self.kind == "count":
            return self.quantity * self.price
        if self.kind == "days":
            return self.quantity * self.price * self.days
        if self.kind == "time":
            return self.quantity * self.price / (60 if self.unit == "h" else 1)
        if self.kind == "length":
            return self.quantity * self.price
        if self.kind == "convert":
            return self.converted * self.price
        if self.kind == "daycredit":
            return self.price * self.days
        raise ValueError(f"unknown item kind {self.kind!r}")

    @property
    def signed(self) -> Fraction:
        return -self.value if self.kind == "daycredit" else self.value


@dataclass(frozen=True)
class Adjustment:
    """A percentage of one priced item, added (`sign` +1) or taken off (-1) before tax."""

    label: str
    percent: Fraction
    target: str  # the priced item's label
    sign: int


@dataclass(frozen=True)
class Fee:
    """An amount added, once (`count` None) or a number of times."""

    label: str
    price: Fraction
    count: Fraction | None = None

    @property
    def value(self) -> Fraction:
        return self.price if self.count is None else self.price * self.count


@dataclass(frozen=True)
class Credit:
    """An amount taken off."""

    label: str
    amount: Fraction


@dataclass(frozen=True)
class Tax:
    rate: Fraction  # a percentage: 15 is 15%
    kind: str  # "charged" (added to a bill) or "withheld" (taken from pay)


@dataclass(frozen=True)
class Case:
    currency: str
    items: tuple[Item, ...]
    tax: Tax
    adjustments: tuple[Adjustment, ...] = ()
    fees: tuple[Fee, ...] = ()
    credits: tuple[Credit, ...] = ()
    outside: frozenset[str] = field(default_factory=frozenset)  # labels of fees and credits applied after tax


def check(case: Case) -> None:
    """The invariants every case meets, whichever route built it; a broken one is a bug, not an input error."""
    labels = [i.label for i in case.items] + [a.label for a in case.adjustments]
    labels += [f.label for f in case.fees] + [c.label for c in case.credits]
    if len({lab.casefold() for lab in labels}) != len(labels):
        raise ValueError("two lines share a label")
    if not case.items:
        raise ValueError("no priced item")
    priced = {i.label for i in case.items if i.kind != "daycredit"}
    for adj in case.adjustments:
        if adj.target not in priced or adj.sign not in (1, -1):
            raise ValueError(f"adjustment {adj.label!r}")
    movable = (
        {f.label for f in case.fees}
        | {c.label for c in case.credits}
        | {i.label for i in case.items if i.kind == "daycredit"}
    )
    if not case.outside <= movable:
        raise ValueError("only fees and credits can sit outside the taxed base")
    if case.tax.kind not in ("charged", "withheld"):
        raise ValueError(f"tax kind {case.tax.kind!r}")


def taxed_base(case: Case) -> Fraction:
    by_label = {i.label: i for i in case.items}
    base = sum((i.signed for i in case.items if i.label not in case.outside), Fraction(0))
    base += sum((a.sign * a.percent / 100 * by_label[a.target].value for a in case.adjustments), Fraction(0))
    base += sum((f.value for f in case.fees if f.label not in case.outside), Fraction(0))
    base -= sum((c.amount for c in case.credits if c.label not in case.outside), Fraction(0))
    return base


def lines(case: Case) -> list[str]:
    """The block's lines, without the heading and the `- ` markers."""
    check(case)
    cur = case.currency

    def m(x: Fraction) -> str:
        return render.money(x, cur)

    q = render.quantity
    out = []
    for it in case.items:
        if it.kind == "count":
            body = f"{q(it.quantity)} x {m(it.price)} = {m(it.value)}"
        elif it.kind == "days":
            body = f"{q(it.quantity)} x {m(it.price)} x {it.days} days = {m(it.value)}"
        elif it.kind == "time":
            body = f"{q(it.quantity)} min x {m(it.price)}/{it.unit} = {m(it.value)}"
        elif it.kind == "length":
            body = f"{q(it.quantity)} {it.unit} x {m(it.price)}/{it.unit} = {m(it.value)}"
        elif it.kind == "convert":
            conv = q(it.converted)
            body = f"{q(it.quantity)} {it.unit} = {conv} {it.to_unit}; {conv} x {m(it.price)} = {m(it.value)}"
        else:
            body = f"{m(it.price)} x {it.days} days = {m(it.value)} (taken off)"
        out.append(f"{it.label}: {body}")
    by_label = {i.label: i for i in case.items}
    for adj in case.adjustments:
        val = adj.sign * adj.percent / 100 * by_label[adj.target].value
        out.append(f"{adj.label}: {q(adj.percent)}% of {adj.target} = {m(val)}")
    for fee in case.fees:
        body = m(fee.price) if fee.count is None else f"{q(fee.count)} x {m(fee.price)} = {m(fee.value)}"
        out.append(f"{fee.label}: {body}")
    for cr in case.credits:
        out.append(f"{cr.label}: {m(cr.amount)} (taken off)")
    base = taxed_base(case)
    out.append(f"Taxed base: {m(base)}")
    out.append(f"Tax {case.tax.kind}: {m(base * case.tax.rate / 100)}")
    return out
