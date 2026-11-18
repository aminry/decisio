# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The block's number formats and lines (format `calc-v1`), from exact values.

Every value arrives as a `fractions.Fraction` and is only turned into text here:

  money      the state's currency (a symbol written before the amount, or a three-letter code and a space),
             thousands separators, at least two decimals and at most six, trailing zeros past the second
             dropped; a negative value is written `-$130.6212`
  quantity   at most six decimals, no trailing zeros, no thousands separators (`728.028`, `1002`)

A value with more than six decimals is rounded at the sixth, half to even (626.5306666... is shown as 626.530667,
262.2930625 as 262.293062): the values are exact and only shown to six decimals, as the heading says.
"""

from __future__ import annotations

from fractions import Fraction

HEADING = "Calculator (exact values, shown to six decimals):"
PLACES = 6
_SCALE = 10**PLACES


def _rounded(x: Fraction) -> tuple[bool, int]:
    """The value's sign and its magnitude in millionths, rounded at the sixth decimal, half to even (exact: a
    Fraction's round)."""
    neg = x < 0
    mag = -x if neg else x
    return neg, round(mag * _SCALE)


def _digits(millionths: int, min_places: int) -> tuple[str, str]:
    whole, frac = divmod(millionths, _SCALE)
    frac_s = f"{frac:0{PLACES}d}".rstrip("0")
    if len(frac_s) < min_places:
        frac_s = frac_s.ljust(min_places, "0")
    return str(whole), frac_s


def quantity(x: Fraction) -> str:
    """A quantity, a count or a percentage: up to six decimals, no trailing zeros (`728.028`, `1002`, `12.5`)."""
    neg, m = _rounded(Fraction(x))
    whole, frac = _digits(m, 0)
    text = f"{whole}.{frac}" if frac else whole
    return f"-{text}" if neg and m else text


def money(x: Fraction, currency: str) -> str:
    """A money value in the state's currency: `$1,088.51`, `$992.648`, `-$130.6212`, `EUR 12.50`."""
    neg, m = _rounded(Fraction(x))
    whole, frac = _digits(m, 2)
    grouped = f"{int(whole):,}"
    mark = currency if len(currency) == 1 else f"{currency} "
    return f"{'-' if neg and m else ''}{mark}{grouped}.{frac}"


def block(lines: list[str]) -> str:
    """The block's text: the heading, then one `- ` line per entry, joined by newlines."""
    return "\n".join([HEADING, *(f"- {line}" for line in lines)])
