# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The date and time calculator (format `time-v1`): the deciding facts about the dates and clock times a state
states, written as a block the model reads with the state. Facts only, never an answer: what the text's own values
come to, so that a model reading a rota, a warranty clause or a travel log does not have to do the time arithmetic.

Each line is derived from values written in the text, by fixed rules, with exact arithmetic:

  zoned times     a clock time with an explicit zone ("01:30 CEST", "18:30 UTC-4", "13:00 Chicago time") is written
                  in UTC, with the day it falls on when the text gives its date ("01:30 CEST, 4 Sep 2026 = 23:30 UTC,
                  3 Sep 2026"); a time without a date gives no line
  real time       between dated zoned times (every pair when there are at most five, else successive ones): the real
                  time between them, across zones and clock changes ("13:00 Chicago time, 3 Sep 2026 to 01:30 CEST,
                  4 Sep 2026: 5 h 30 min"; "minus" when the second is earlier)
  calendar        "N days / weeks after <date>": the date it comes to ("23 Jan 2026 + 9 weeks: 27 Mar 2026");
                  "N months / years after <date>": the date, or, when that day does not exist, that it does not
                  and how many days the month has (the text's own rule decides what happens then)

Zone abbreviations with one meaning are fixed offsets (CEST is UTC+2 whatever the date); abbreviations with more than
one meaning (CST, IST, AST, ...) are not read. A major city's time is read with its zone's rules on the stated date
(daylight saving included), only when the date is given and the time is not one a clock change skips or repeats. A
value the rules cannot read in one way is left out: the block holds only what is certain. With no line, there is no
block (`Abstain("no-time-facts")`).
"""

from __future__ import annotations

import calendar
import functools
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

VERSION = "time-v1"
HEADING = "Calculator, times (exact):"
MAX_LINES = 24

_MONTHS = {
    m.lower(): i
    for i, names in enumerate(
        [(), ("January", "Jan"), ("February", "Feb"), ("March", "Mar"), ("April", "Apr"), ("May",), ("June", "Jun"),
         ("July", "Jul"), ("August", "Aug"), ("September", "Sep", "Sept"), ("October", "Oct"), ("November", "Nov"),
         ("December", "Dec")]
    )
    for m in names
}  # fmt: skip
_MONTH_RX = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAY_RX = r"(?:Mon|Tue|Tues|Wed|Thu|Thur|Thurs|Fri|Sat|Sun)(?:day|sday|nesday|rsday|urday)?\.?"
_MON3 = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# zone abbreviations with one meaning, as minutes east of UTC
ZONES = {
    "UTC": 0, "GMT": 0, "Z": 0, "WET": 0, "WEST": 60, "BST": 60, "CET": 60, "CEST": 120, "EET": 120, "EEST": 180,
    "MSK": 180, "EST": -300, "EDT": -240, "CDT": -300, "MST": -420, "MDT": -360, "PST": -480, "PDT": -420,
    "AKST": -540, "AKDT": -480, "HST": -600, "JST": 540, "KST": 540, "SGT": 480, "HKT": 480, "AWST": 480,
    "ACST": 570, "ACDT": 630, "AEST": 600, "AEDT": 660, "NZST": 720, "NZDT": 780,
}  # fmt: skip
_ZONE_RX = r"(?:UTC|GMT)\s?[+\-−]\s?\d{1,2}(?::?\d{2})?|" + "|".join(sorted(ZONES, key=len, reverse=True))

_DATE = re.compile(
    rf"(?:(?P<wd1>{_WEEKDAY_RX}),?\s+)?"
    rf"(?:(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<m1>{_MONTH_RX})\.?(?:,?\s+(?P<y1>\d{{4}}))?"
    rf"|(?P<m2>{_MONTH_RX})\.?\s+(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(?P<y2>\d{{4}}))?"
    rf"|(?P<y3>\d{{4}})-(?P<m3>\d{{2}})-(?P<d3>\d{{2}}))\b",
    re.I,
)
# places with one time zone, read with the zone's rules on the stated date (daylight saving included)
CITIES = {
    "chicago": "America/Chicago", "new york": "America/New_York", "los angeles": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles", "seattle": "America/Los_Angeles", "denver": "America/Denver",
    "phoenix": "America/Phoenix", "houston": "America/Chicago", "dallas": "America/Chicago",
    "atlanta": "America/New_York", "boston": "America/New_York", "miami": "America/New_York",
    "toronto": "America/Toronto", "vancouver": "America/Vancouver", "mexico city": "America/Mexico_City",
    "sao paulo": "America/Sao_Paulo", "s\u00e3o paulo": "America/Sao_Paulo", "london": "Europe/London",
    "dublin": "Europe/Dublin", "lisbon": "Europe/Lisbon", "madrid": "Europe/Madrid", "barcelona": "Europe/Madrid",
    "paris": "Europe/Paris", "brussels": "Europe/Brussels", "amsterdam": "Europe/Amsterdam",
    "rotterdam": "Europe/Amsterdam", "berlin": "Europe/Berlin", "frankfurt": "Europe/Berlin", "munich": "Europe/Berlin",
    "zurich": "Europe/Zurich", "vienna": "Europe/Vienna", "rome": "Europe/Rome", "milan": "Europe/Rome",
    "stockholm": "Europe/Stockholm", "oslo": "Europe/Oslo", "copenhagen": "Europe/Copenhagen",
    "helsinki": "Europe/Helsinki", "warsaw": "Europe/Warsaw", "prague": "Europe/Prague", "athens": "Europe/Athens",
    "istanbul": "Europe/Istanbul", "moscow": "Europe/Moscow", "dubai": "Asia/Dubai", "mumbai": "Asia/Kolkata",
    "delhi": "Asia/Kolkata", "new delhi": "Asia/Kolkata", "singapore": "Asia/Singapore", "hong kong": "Asia/Hong_Kong",
    "shanghai": "Asia/Shanghai", "beijing": "Asia/Shanghai", "tokyo": "Asia/Tokyo", "seoul": "Asia/Seoul",
    "sydney": "Australia/Sydney", "melbourne": "Australia/Melbourne", "brisbane": "Australia/Brisbane",
    "perth": "Australia/Perth", "auckland": "Pacific/Auckland", "johannesburg": "Africa/Johannesburg",
    "cairo": "Africa/Cairo", "nairobi": "Africa/Nairobi", "lagos": "Africa/Lagos",
}  # fmt: skip
_CITY_RX = "|".join(sorted((re.escape(c) for c in CITIES), key=len, reverse=True))
_TIME = re.compile(
    r"(?<![\d:.,$\u20ac\u00a3])(?P<h>[01]?\d|2[0-4])(?P<sep>[:.])(?P<mi>[0-5]\d)(?![\d:])(?:\s?(?P<ap>[ap]\.?m\.?)(?![a-z]))?"
    rf"(?:\s*\(?(?P<z>{_ZONE_RX})\)?(?![A-Za-z/])|\s+(?:[A-Za-z]+\s+){{0,4}}\(?(?P<city>{_CITY_RX})\s+time\b\)?)?",
    re.I,
)
_SHIFT = re.compile(
    rf"\b(?P<n>\d{{1,4}})\s+(?P<u>days?|weeks?|months?|years?)\s+(?P<dir>after|from|following|before|prior to)\s+"
    rf"(?:the\s+)?(?P<date>(?:{_WEEKDAY_RX},?\s+)?(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTH_RX})\.?(?:,?\s+\d{{4}})?"
    rf"|(?:{_MONTH_RX})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?|\d{{4}}-\d{{2}}-\d{{2}}))",
    re.I,
)

# "1 week after the Start date", the base given by its label elsewhere in the text
_SHIFT_LABEL = re.compile(
    r"\b(?P<n>\d{1,4})\s+(?P<u>days?|weeks?|months?|years?)\s+(?P<dir>after|from|following|before|prior to)\s+"
    r"the\s+[\"\u201c*]*(?P<label>[A-Za-z][\w'-]*(?:\s+[A-Za-z][\w'-]*){0,4}?)[\"\u201d*]*"
    r"(?=\s*(?:[.,;:|)]|$|\s+(?:listed|shown|given|above|below|in|on|of|and|for|under|as|unless)\b))",
    re.I | re.M,
)
# what may stand between a label and its date: "Start date: ...", "| Start date | ...", "Start date is ...",
# "Start date of ..."
_LABEL_LINK = r"[\"\u201d*]*\s*(?:\||:|=|\bis\b|\bwas\b|\bof\b)?\s*[\"\u201c*|]*\s*"

# a table row's time cell followed by a cell that holds one zone and nothing else: "| 08:00 | JST |"
_CELL_ZONE = re.compile(
    rf"\|\s*(?P<h>[01]?\d|2[0-4]):(?P<mi>[0-5]\d)\s*\|\s*(?P<z>{_ZONE_RX})\s*\|",
)

# what a text needs before any rule runs: a month name, an ISO date or a clock time
_PREGATE = re.compile(rf"\b(?:{_MONTH_RX})\b|\d{{4}}-\d{{2}}-\d{{2}}|\b\d{{1,2}}[:.][0-5]\d\b", re.I)


@dataclass(frozen=True)
class Stamp:
    text: str  # as written
    start: int
    end: int
    minutes: int  # minutes since midnight, local
    offset: int | None  # minutes east of UTC, None when no zone
    day: date | None  # the date the text gives for it, if exactly one


def may_hold_facts(text: str) -> bool:
    return bool(_PREGATE.search(text))


def _year_of(text: str) -> int | None:
    years = {int(y) for y in re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", text)}
    return years.pop() if len(years) == 1 else None


def _dates(text: str, year: int | None) -> list[tuple[int, int, date, str]]:
    out = []
    for m in _DATE.finditer(text):
        try:
            if m.group("y3"):
                d = date(int(m.group("y3")), int(m.group("m3")), int(m.group("d3")))
            else:
                mon = _MONTHS[(m.group("m1") or m.group("m2")).lower()]
                day = int(m.group("d1") or m.group("d2"))
                y = m.group("y1") or m.group("y2")
                y = int(y) if y else year
                if y is None:
                    continue
                d = date(y, mon, day)
        except ValueError:
            continue
        wd = m.group("wd1")
        if wd and not d.strftime("%a").lower().startswith(wd[:3].lower()):
            continue  # the text's weekday and date disagree: neither is read
        out.append((m.start(), m.end(), d, m.group(0)))
    return out


def _offset(z: str) -> int | None:
    z = z.upper().replace("−", "-").replace(" ", "")
    if z in ZONES:
        return ZONES[z]
    m = re.fullmatch(r"(?:UTC|GMT)([+\-])(\d{1,2})(?::?(\d{2}))?", z)
    if not m:
        return None
    h, mi = int(m.group(2)), int(m.group(3) or 0)
    if h > 14 or mi >= 60:
        return None
    return (1 if m.group(1) == "+" else -1) * (h * 60 + mi)


def _sentence_bounds(text: str, at: int) -> tuple[int, int]:
    a = max(text.rfind(". ", 0, at), text.rfind("\n", 0, at)) + 1
    ends = [i for i in (text.find(". ", at), text.find("\n", at)) if i >= 0]
    return a, (min(ends) if ends else len(text))


def _stamps(text: str, dates: list[tuple[int, int, date, str]]) -> list[Stamp]:
    out = []
    for m in _TIME.finditer(text):
        h, mi, ap = int(m.group("h")), int(m.group("mi")), (m.group("ap") or "").lower().replace(".", "")
        if m.group("sep") == "." and not ap and not m.group("z"):
            continue  # "12.41" alone may be an amount: a dotted time is read only with am/pm or a zone
        if ap:
            if not 1 <= h <= 12:
                continue
            h = (0 if h == 12 else h) + (12 if ap == "pm" else 0)
        if h == 24 and mi:
            continue
        off = _offset(m.group("z")) if m.group("z") else None
        if m.group("z") and off is None:
            continue
        a, b = _sentence_bounds(text, m.start())
        near = [d for s, e, d, _ in dates if a <= s and e <= b]
        day = near[0] if len(set(near)) == 1 else None
        if m.group("city"):
            off = _city_offset(CITIES[m.group("city").lower()], day, h * 60 + mi) if day else None
            if off is None:
                continue  # a place's offset depends on the date (and a clock change hour is not read)
        out.append(Stamp(_balanced(m.group(0).strip()), m.start(), m.end(), h * 60 + mi, off, day))
    for m in _CELL_ZONE.finditer(text):
        h, mi = int(m.group("h")), int(m.group("mi"))
        off = _offset(m.group("z"))
        if off is None or (h == 24 and mi):
            continue
        a, b = _sentence_bounds(text, m.start())
        near = [d for s, e, d, _ in dates if a <= s and e <= b]
        day = near[0] if len(set(near)) == 1 else None
        out.append(Stamp(f"{m.group('h')}:{m.group('mi')} {m.group('z')}", m.start(), m.end(), h * 60 + mi, off, day))
    out.sort(key=lambda s: s.start)
    return out


def _balanced(t: str) -> str:
    """A stamp as written, without a closing parenthesis it does not open ("05:45 UTC+3)")."""
    while t.endswith(")") and t.count(")") > t.count("("):
        t = t[:-1].rstrip()
    return t


_ZONE_NAME = re.compile(r"[A-Za-z_]+(?:/[A-Za-z0-9_+-]+)*")


@functools.lru_cache(maxsize=128)
def _zone(name: str):
    """The zone's rules from the `tzdata` package, never the host's zone files, so that every host (a Linux box reads
    /usr/share/zoneinfo first, a Mac its own copy) applies the same rules for the same decisio install; None when the
    name is not an IANA zone."""
    if not _ZONE_NAME.fullmatch(name):
        return None
    try:
        from importlib.resources import files
        from zoneinfo import ZoneInfo

        with files("tzdata.zoneinfo").joinpath(*name.split("/")).open("rb") as f:
            return ZoneInfo.from_file(f, key=name)
    except Exception:
        return None


def _city_offset(zone: str, day: date, minutes: int) -> int | None:
    """The zone's offset (minutes east of UTC) at that local date and time; None when the zone is unknown, or the
    local time is skipped or repeated by a clock change."""
    z = _zone(zone)
    if z is None:
        return None
    if minutes >= 24 * 60:
        return None
    local = datetime(day.year, day.month, day.day, minutes // 60, minutes % 60)
    a, b = local.replace(tzinfo=z, fold=0), local.replace(tzinfo=z, fold=1)
    if a.utcoffset() != b.utcoffset():
        return None  # repeated or skipped by a clock change
    return int(a.utcoffset().total_seconds() // 60)


def _date_for_label(text: str, label: str, year: int | None) -> date | None:
    """The one date the text gives a label: every mention of the label followed straight away (through ":", "|", "is",
    "was", "of" only) by a date; None when no mention or two different dates. A possessive before the label ("the
    employee's Start date") is dropped."""
    words = label.split()
    cands = [label] + ([" ".join(words[1:])] if len(words) > 1 and words[0].lower().endswith("'s") else [])
    for cand in cands:
        found = set()
        for m in re.finditer(rf"(?<![\w-]){re.escape(cand)}(?![\w-]){_LABEL_LINK}", text, re.I):
            ds = _dates(text[m.end() : m.end() + 40], year)
            if ds and ds[0][0] == 0:
                found.add(ds[0][2])
        if len(found) == 1:
            return found.pop()
        if found:
            return None
    return None


def _hm(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{h:02d}:{m:02d}"


def _dur(minutes: int) -> str:
    """A real duration, minutes always shown; "minus" when the second time is earlier than the first."""
    h, m = divmod(abs(minutes), 60)
    return ("minus " if minutes < 0 else "") + f"{h} h {m} min"


def _day(d: date) -> str:
    return f"{d.day} {_MON3[d.month]} {d.year}"


def _stamp(s: Stamp) -> str:
    """A dated zoned time as the block writes it: the time and zone as the text writes them, then its date."""
    return f"{s.text}, {_day(s.day)}"


def facts(state: str, context: str | None = None) -> list[str]:
    """The block's lines for a state (and its context), in the text's order; [] when nothing is certain."""
    text = ((context + "\n") if context else "") + state
    if not may_hold_facts(text):
        return []
    year = _year_of(text)
    dates = _dates(text, year)
    lines: list[str] = []
    stamps = _stamps(text, dates)
    # zoned times in UTC
    zoned = [s for s in stamps if s.offset is not None]
    for s in zoned:
        utc = s.minutes - s.offset
        if s.day is not None:
            when = datetime(s.day.year, s.day.month, s.day.day) + timedelta(minutes=utc)
            lines.append(f"{_stamp(s)} = {_hm(when.hour * 60 + when.minute)} UTC, {_day(when.date())}")
    # elapsed time between successive zoned, dated times
    dated = [s for s in zoned if s.day is not None]
    # every pair when there are few (a dose and the next one may have other events between them), else successive ones
    pairs = (
        [(dated[i], dated[j]) for i in range(len(dated)) for j in range(i + 1, len(dated))]
        if len(dated) <= 5
        else list(zip(dated, dated[1:]))
    )
    for a, b in pairs:
        ta = datetime(a.day.year, a.day.month, a.day.day) + timedelta(minutes=a.minutes - a.offset)
        tb = datetime(b.day.year, b.day.month, b.day.day) + timedelta(minutes=b.minutes - b.offset)
        mins = int((tb - ta).total_seconds() // 60)
        lines.append(f"{_stamp(a)} to {_stamp(b)}: {_dur(mins)}")
    # calendar shifts
    shifts = [(m, _dates(m.group("date"), year)) for m in _SHIFT.finditer(text)]
    for m in _SHIFT_LABEL.finditer(text):
        d = _date_for_label(text, m.group("label"), year)
        if d is not None:
            shifts.append((m, [(0, 0, d, "")]))
    shifts.sort(key=lambda x: x[0].start())
    for m, ds in shifts:
        if len(ds) != 1:
            continue
        d0, n, unit = ds[0][2], int(m.group("n")), m.group("u").lower().rstrip("s")
        sign = -1 if m.group("dir").lower() in ("before", "prior to") else 1
        what = f"{_day(d0)} {'+' if sign > 0 else '-'} {n} {unit}{'s' if n != 1 else ''}"
        if unit in ("day", "week"):
            d1 = d0 + timedelta(days=sign * n * (7 if unit == "week" else 1))
            lines.append(f"{what}: {_day(d1)}")
            continue
        months = sign * n * (12 if unit == "year" else 1)
        y, mo = divmod(d0.year * 12 + (d0.month - 1) + months, 12)
        mo += 1
        last = calendar.monthrange(y, mo)[1]
        if d0.day <= last:
            d1 = date(y, mo, d0.day)
            lines.append(f"{what}: {_day(d1)}")
        else:
            lines.append(
                f"{what}: {d0.day} {_MON3[mo]} {y} does not exist; {calendar.month_name[mo]} {y} has {last} days"
            )
    out, seen_lines = [], set()
    for ln in lines:
        if ln not in seen_lines:
            seen_lines.add(ln)
            out.append(ln)
    return out[:MAX_LINES]


def block(lines: list[str]) -> str:
    return "\n".join([HEADING, *(f"- {ln}" for ln in lines)])
