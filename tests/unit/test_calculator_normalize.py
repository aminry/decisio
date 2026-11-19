# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The calculator's normalizer (`decisio.calculator.normalize`) and the normalized reading it gives the parser.

N1  value kinds: money (marks before and after, codes, decimal commas, negatives, multipliers, words), percentages,
    numbers in digits and in words, durations, quantities with units and count nouns, dates, date ranges, times of
    day; exact Fractions and spans that slice the original text;
N2  forms that are never guessed: a separator nothing settles, a mixed document, day and month that could swap, a
    slash form with no year, `1:30`, a mark between two numbers, an amount in parentheses, `pounds`, a range;
N3  determinism, text order, spans that never overlap;
N4  units: dimensions, exact factors, units whose size is not certain, the check against pint when it is installed;
N5  `canonical`: the spelling the reader reads;
N6  the parser with `normalized=True`: off by default and identical without it, wider where it is certain (decimal
    commas, marks after amounts, numbers in words, JSON and table states), and no block where a form is unsure.

  uv run pytest -q tests/unit/test_calculator_normalize.py
"""

from fractions import Fraction as F

import pytest

from decisio.calculator import Abstain, Annotation, annotate, normalize, parse
from decisio.calculator.normalize import (
    Ambiguous,
    Date,
    DateRange,
    Duration,
    Money,
    Number,
    Percent,
    Quantity,
    TimeOfDay,
    canonical,
    decimal_convention,
    dimension,
    document,
    factor,
)


def values(text, doc=None):
    return normalize.normalize(text, doc).values


def one(text):
    """The single value in a text, checked to span the whole text."""
    vs = values(text)
    assert len(vs) == 1, vs
    assert vs[0].span == (0, len(text)), (vs[0].span, text)
    return vs[0]


# --- N1 value kinds -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, value, currency",
    [
        ("$1,234.56", F("1234.56"), "$"),
        ("$ 12.50", F("12.5"), "$"),
        ("€1.234,56", F("1234.56"), "€"),
        ("1.234,56 €", F("1234.56"), "€"),
        ("12,50 €", F("12.5"), "€"),
        ("12.50€", F("12.5"), "€"),
        ("EUR 12.50", F("12.5"), "EUR"),
        ("12.50 EUR", F("12.5"), "EUR"),
        ("1.234,56 EUR", F("1234.56"), "EUR"),
        ("-$40.00", F(-40), "$"),
        ("$-40.00", F(-40), "$"),
        ("-12,50 €", F("-12.5"), "€"),
        ("$1.2M", F(1200000), "$"),
        ("$5k", F(5000), "$"),
        ("$3 million", F(3000000), "$"),
        ("£0.99", F("0.99"), "£"),
        ("twenty dollars", F(20), "$"),
        ("45 euros", F(45), "€"),
    ],
)
def test_money(text, value, currency):
    v = one(text)
    assert isinstance(v, Money) and v.kind == "money"
    assert (v.value, v.currency) == (value, currency)
    assert isinstance(v.value, F)


@pytest.mark.parametrize(
    "text, value",
    [("12.5%", F("12.5")), ("12,5 %", F("12.5")), ("20 percent", F(20)), ("twenty per cent", F(20)), ("7%", F(7))],
)
def test_percent(text, value):
    v = one(text)
    assert isinstance(v, Percent) and v.value == value


@pytest.mark.parametrize(
    "text, value",
    [
        ("seven", F(7)),
        ("Seven", F(7)),
        ("twenty-one", F(21)),
        ("twenty one", F(21)),
        ("one hundred and five", F(105)),
        ("three hundred", F(300)),
        ("fifteen hundred", F(1500)),
        ("two thousand and twelve", F(2012)),
        ("a thousand", F(1000)),
        ("a dozen", F(12)),
        ("two dozen", F(24)),
        ("half a dozen", F(6)),
        ("half", F(1, 2)),
        ("a half", F(1, 2)),
        ("one and a half", F(3, 2)),
        ("0.4536", F("0.4536")),
        ("1,234,567", F(1234567)),
        ("1.234.567", F(1234567)),
        ("1234,5", F("1234.5")),
        (".5", F(1, 2)),
        ("2026", F(2026)),
    ],
)
def test_numbers(text, value):
    v = one(text)
    assert isinstance(v, Number) and v.value == value


def test_fractions_in_words_need_of():
    vs = values("three quarters of the bill")
    assert isinstance(vs[0], Number) and vs[0].value == F(3, 4)
    # `two quarters` with no `of` is a count of quarters
    q = values("rent for two quarters")[0]
    assert isinstance(q, Quantity) and (q.value, q.unit) == (2, "quarter")


@pytest.mark.parametrize(
    "text, minutes",
    [
        ("9 h 28 min", 568),
        ("1 h 52 min", 112),
        ("45 minutes", 45),
        ("1.5 hours", 90),
        ("1,5 h", 90),
        ("2 hours and 15 minutes", 135),
        ("1 hr, 5 mins", 65),
        ("9h28", 568),
        ("1:30 h", 90),
        ("30:00", 1800),
        ("90 seconds", F(3, 2)),
        ("half an hour", 30),
        ("quarter hour", 15),
        ("a quarter of an hour", 15),
        ("two hours", 120),
        ("one and a half hours", 90),
    ],
)
def test_durations(text, minutes):
    v = one(text)
    assert isinstance(v, Duration) and v.value == minutes and v.unit == "min"


@pytest.mark.parametrize(
    "text, value, unit, dim",
    [
        ("980 lb", F(980), "lb", "mass"),
        ("12kg", F(12), "kg", "mass"),
        ("3.57 m", F("3.57"), "m", "length"),
        ("212 miles", F(212), "mi", "length"),
        ("26.1 L", F("26.1"), "L", "volume"),
        ("750 ml", F(750), "ml", "volume"),
        ("40 m2", F(40), "m2", "area"),
        ("5 days", F(5), "day", "time"),
        ("2 weeks", F(2), "week", "time"),
        ("10 gal", F(10), "gal", "volume"),
        ("4 pallets", F(4), "pallet", "count"),
        ("seven crates", F(7), "crate", "count"),
        ("twenty-one boxes", F(21), "box", "count"),
        ("a dozen bolts", F(12), "bolt", "count"),
        ("1/2 kg", F(1, 2), "kg", "mass"),
        ("2 1/2 kg", F(5, 2), "kg", "mass"),
        ("half a metre", F(1, 2), "m", "length"),
    ],
)
def test_quantities(text, value, unit, dim):
    v = one(text)
    assert isinstance(v, Quantity) and (v.value, v.unit, v.dimension) == (value, unit, dim)


@pytest.mark.parametrize(
    "text, ymd",
    [
        ("2026-11-18", (2026, 11, 18)),
        ("18 November 2026", (2026, 11, 18)),
        ("18th Nov", (None, 11, 18)),
        ("Nov 18", (None, 11, 18)),
        ("November 18, 2026", (2026, 11, 18)),
        ("Sept 3", (None, 9, 3)),
        ("13/04/2026", (2026, 4, 13)),
        ("04/13/2026", (2026, 4, 13)),
        ("04/04/2026", (2026, 4, 4)),
        ("2026/11/18", (2026, 11, 18)),
        ("29 Feb", (None, 2, 29)),
    ],
)
def test_dates(text, ymd):
    v = one(text)
    assert isinstance(v, Date) and (v.year, v.month, v.day) == ymd


@pytest.mark.parametrize(
    "text, a, b",
    [
        ("27 Mar to 2 Apr", (3, 27), (4, 2)),
        ("27 Mar - 2 Apr", (3, 27), (4, 2)),
        ("Mar 27 to Apr 2", (3, 27), (4, 2)),
        ("27-30 March", (3, 27), (3, 30)),
        ("Mar 27-30", (3, 27), (3, 30)),
        ("between 3 May and 9 May", (5, 3), (5, 9)),
        ("28 Dec to 3 Jan", (12, 28), (1, 3)),
    ],
)
def test_date_ranges(text, a, b):
    v = one(text)
    assert isinstance(v, DateRange)
    assert (v.start.month, v.start.day) == a and (v.end.month, v.end.day) == b


def test_date_range_days():
    r = one("27 Mar to 2 Apr")
    assert (r.days(inclusive=True), r.days(inclusive=False)) == (7, 6)
    assert one("28 Dec to 3 Jan").days(inclusive=True) == 7
    assert one("27 Feb to 2 Mar").days(inclusive=True) is None  # a leap day may fall inside
    assert one("27 Feb 2026 to 2 Mar 2026").days(inclusive=True) == 4


def test_times_of_day():
    assert one("10:30 am") == TimeOfDay(F(630), (0, 8))
    assert one("2 pm").value == 14 * 60
    v = values("opens at 14:00")[0]
    assert isinstance(v, TimeOfDay) and v.value == 840


def test_a_mixed_sentence():
    text = "Labour: 2 h 15 min at $48.00 an hour, 27 Mar to 2 Apr; 12% off; 3 pallets of 980 lb."
    got = [(v.kind, text[v.span[0] : v.span[1]]) for v in values(text)]
    assert got == [
        ("duration", "2 h 15 min"),
        ("money", "$48.00"),
        ("date-range", "27 Mar to 2 Apr"),
        ("percent", "12%"),
        ("quantity", "3 pallets"),
        ("quantity", "980 lb"),
    ]


def test_not_numbers():
    # identifiers, ordinals, codes, compounds, pronouns
    assert values("Order GW-204 is the 3rd of its kind, a 4x4 frame.") == ()
    assert values("A one-off fee, done one by one, and the one we chose.") == ()
    assert values("No one asked for another one.") == ()
    v = values("twenty one-hour sessions")
    assert len(v) == 1 and v[0].value == 20  # `one-hour` is a compound, not 1


# --- N2 never guessed ---------------------------------------------------------------------------------------------


def test_separator_alone_is_ambiguous():
    v = one("$1,234")
    assert isinstance(v, Ambiguous) and "thousands or marks decimals" in v.why
    assert {r.value for r in v.readings} == {F(1234), F("1.234")}
    assert isinstance(one("1.234"), Ambiguous)


def test_separator_settled_by_the_text():
    point = values("$1,234 and $9.50")
    assert [v.value for v in point] == [F(1234), F("9.5")]
    comma = values("1,234 € and 9,50 €")
    assert [v.value for v in comma] == [F("1.234"), F("9.5")]
    assert decimal_convention("$1,234 and $9.50") == "point"
    assert decimal_convention("1.234 € and 9,50 €") == "comma"
    assert decimal_convention("$1,234") is None


def test_mixed_document_settles_nothing():
    text = "$1,234.50, then 12,5 kg, then 1.234"
    assert decimal_convention(text) == "mixed"
    vs = values(text)
    assert vs[0].value == F("1234.5") and vs[1].value == F("12.5")  # each certain on its own form
    assert isinstance(vs[2], Ambiguous)


def test_a_part_reads_with_its_document():
    doc = document("Total 9,50 €.\nShipping: 1.250 kg")
    assert doc.convention == "comma" and doc.currencies == {"€"}
    v = values("1.250 kg", doc)[0]
    assert isinstance(v, Quantity) and v.value == 1250
    assert isinstance(values("1.250 kg")[0], Ambiguous)  # alone, nothing settles it


@pytest.mark.parametrize(
    "text, why",
    [
        ("03/04/2026", "day and month could swap"),
        ("03/04/26", "a two-digit year"),
        ("3/4", "a fraction or a date with no year"),
        ("30 Feb", "not a calendar date"),
        ("1:30", "a duration or a time of day"),
        ("5 € 10", "a currency mark between two numbers"),
        ("($12.50)", "an amount in parentheses: negative, or an aside"),
        ("50 pounds", "pounds: money or mass"),
        ("3-5", "a range or a difference of numbers"),
        ("1.2.3", "separators that form no number"),
        ("8,10,12", "separators that form no number"),
        ("2 Mar 2026 to 1 Mar 2026", "a date range that ends before it starts"),
    ],
)
def test_ambiguous_forms(text, why):
    v = one(text)
    assert isinstance(v, Ambiguous) and v.why == why, v
    for r in v.readings:
        assert r.span == v.span


def test_ambiguous_readings():
    assert {(r.month, r.day) for r in one("03/04/2026").readings} == {(4, 3), (3, 4)}
    kinds = {r.kind for r in one("1:30").readings}
    assert kinds == {"duration", "time-of-day"}
    assert {r.kind for r in one("50 pounds").readings} == {"money", "quantity"}
    assert {r.value for r in one("($12.50)").readings} == {F("12.5"), F("-12.5")}


def test_pounds_settled_by_the_document():
    doc = document("Freight at $0.35 per kg; 980 pounds")
    v = values("980 pounds", doc)[0]
    assert isinstance(v, Quantity) and (v.unit, v.dimension) == ("lb", "mass")
    assert isinstance(values("980 pounds", document("£12.00 and 980 pounds"))[0], Ambiguous)


def test_a_minus_between_amounts_is_no_sign():
    vs = values("$40.00 - $15.15 = $24.85")
    assert [v.value for v in vs] == [F(40), F("15.15"), F("24.85")]


def test_numbers_joined_by_to_are_two_numbers():
    vs = values("months 1 to 18")
    assert [v.value for v in vs] == [1, 18]


# --- N3 determinism -----------------------------------------------------------------------------------------------

MIXED = (
    "Invoice GW-310, 18 November 2026. Labour: 9 h 28 min at €41,00 an hour; 1.250,00 € deposit (twenty per cent); "
    "three crews, a dozen bolts, 26,1 L of oil, 27 Mar to 2 Apr, 2 pm, 03/04/2026, 5 € 10, 12,5 %."
)


def test_deterministic_and_ordered():
    a, b = normalize.normalize(MIXED), normalize.normalize(MIXED)
    assert a == b
    spans = [v.span for v in a.values]
    assert spans == sorted(spans)
    assert all(e1 <= s2 for (_, e1), (s2, _) in zip(spans, spans[1:]))  # never overlapping
    for v in a.values:
        assert 0 <= v.span[0] < v.span[1] <= len(MIXED)
        for name in ("value",):
            if hasattr(v, name):
                assert isinstance(getattr(v, name), F)


def test_values_are_frozen():
    v = one("$12.50")
    with pytest.raises(AttributeError):
        v.value = F(1)


# --- N4 units -----------------------------------------------------------------------------------------------------


def test_dimensions():
    assert [dimension(u) for u in ("lb", "pounds", "kg", "mi", "L", "gal", "h", "minutes", "m2")] == [
        "mass",
        "mass",
        "mass",
        "length",
        "volume",
        "volume",
        "time",
        "time",
        "area",
    ]
    assert dimension("pallet") is None


def test_factors_are_exact_definitions():
    assert factor("lb", "kg") == F("0.45359237")
    assert factor("mi", "km") == F("1.609344")
    assert factor("kg", "lb") == 1 / F("0.45359237")
    assert factor("h", "min") == 60
    assert factor("week", "day") == 7
    assert factor("ft", "in") == 12
    assert factor("kg", "m") is None  # across dimensions
    assert factor("gal", "L") is None  # US or imperial
    assert factor("ton", "kg") is None  # short, long or metric
    assert factor("month", "day") is None


def test_pint_check_when_installed():
    pytest.importorskip("pint")
    assert normalize.pint_check() == []


def test_pint_check_without_pint():
    try:
        import pint  # noqa: F401
    except ImportError:
        assert normalize.pint_check() is None
    else:
        pytest.skip("pint is installed")


# --- N5 canonical -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, words, out",
    [
        ("3 bins at 95,40 € each", False, "3 bins at €95.40 each"),
        ("2 pallets at 1.250,00 € each", False, "2 pallets at €1250.00 each"),
        ("1,5 h at €40,00 per hour", False, "1.5 h at €40.00 per hour"),
        ("12 bags at 4.50 EUR each", False, "12 bags at EUR 4.50 each"),
        ("Tax of 20 percent is added", False, "Tax of 20% is added"),
        ("Tax of 12,5 per cent is added", False, "Tax of 12.5% is added"),
        ("seven bins at 9,50 € each for three days", True, "7 bins at €9.50 each for 3 days"),
        ("seven bins at 9,50 € each", False, "seven bins at €9.50 each"),
        ("a dozen bolts at 0,40 € per bolt", True, "12 bolts at €0.40 per bolt"),
        ("one and a half hours at 9,50 €", True, "one and a half hours at €9.50"),
        ("a credit of -12,50 € and 1,2M €", False, "a credit of -12,50 € and 1,2M €"),
    ],
)
def test_canonical(text, words, out):
    got, unsure = canonical(text, document(text + " 9,50 €"), words=words)
    assert (got, unsure) == (out, ())


def test_canonical_point_text_keeps_its_numbers():
    text = "4 attachments at $1,299.46 each; 3.57 m"
    assert canonical(text) == (text, ())


def test_canonical_reports_ambiguity():
    got, unsure = canonical("4 boxes at $1,250 each")
    assert got == "4 boxes at $1,250 each" and len(unsure) == 1


# --- N6 the normalized reading ------------------------------------------------------------------------------------

BASE = """The amount due is what the client pays once the rules below are applied.
A 5% early-booking discount comes off the hedge trimming only, before tax.
Tax of 10% is added to the whole charge.
Only the final money figure is rounded to the cent, half a cent going up.
Work order GW-118 for Ilse Morrow:
- Hedge trimming: 3 hedges at $42.50 each
- Mower hire: 1 mower at $30.00 a day for 2 days
- Paving stone: 2 pallets at $1,250.00 each
- Green waste: 4 bags at $6.25 each"""

BASE_BLOCK = """Calculator (exact values, shown to six decimals):
- Hedge trimming: 3 x $42.50 = $127.50
- Mower hire: 1 x $30.00 x 2 days = $60.00
- Paving stone: 2 x $1,250.00 = $2,500.00
- Green waste: 4 x $6.25 = $25.00
- Early-booking discount: 5% of Hedge trimming = -$6.375
- Taxed base: $2,706.125
- Tax charged: $270.6125"""

EURO = (
    BASE.replace("$42.50", "42,50 €")
    .replace("$30.00", "30,00 €")
    .replace("$1,250.00", "1.250,00 €")
    .replace("$6.25", "6,25 €")
)
WORDS = (
    BASE.replace("3 hedges", "three hedges")
    .replace("1 mower", "one mower")
    .replace("for 2 days", "for two days")
    .replace("4 bags", "four bags")
)


def text_of(ann):
    assert isinstance(ann, Annotation), ann
    return ann.text


def test_off_by_default():
    assert parse.NORMALIZED is False
    for state in (BASE, EURO, WORDS):
        assert annotate(state) == annotate(state, normalized=False)
    assert annotate(EURO) == Abstain("no-arithmetic")
    assert annotate(WORDS) == Abstain("unparsed-item")


def test_on_reads_a_plain_state_the_same():
    assert text_of(annotate(BASE)) == BASE_BLOCK
    assert text_of(annotate(BASE, normalized=True)) == BASE_BLOCK


def test_on_reads_decimal_commas_and_marks_after():
    assert text_of(annotate(EURO, normalized=True)) == BASE_BLOCK.replace("$", "€")


def test_on_reads_item_counts_in_words():
    assert text_of(annotate(WORDS, normalized=True)) == BASE_BLOCK


def test_the_module_switch(monkeypatch):
    monkeypatch.setattr(parse, "NORMALIZED", True)
    assert text_of(annotate(EURO)) == BASE_BLOCK.replace("$", "€")
    assert annotate(EURO, normalized=False) == Abstain("no-arithmetic")


def test_on_reads_json_and_table_states():
    state = (
        '{"rules": ["The amount due is what the client pays once the rules below are applied.", '
        '"Tax of 10% is added to the whole charge."], '
        '"order": {"id": "GW-119", "tiles": "12 boxes at 18,40 € each", "grout": "three tubs at 7,95 € each"}}'
    )
    assert text_of(annotate(state, normalized=True)) == (
        "Calculator (exact values, shown to six decimals):\n"
        "- Tiles: 12 x €18.40 = €220.80\n"
        "- Grout: 3 x €7.95 = €23.85\n"
        "- Taxed base: €244.65\n"
        "- Tax charged: €24.465"
    )
    table = (
        "The amount due is what the client pays once the rules below are applied.\n"
        "Tax of 10% is added to the whole charge.\n"
        "| Item | Detail |\n|---|---|\n"
        "| Fence panels | six panels at 55.00 EUR each |\n"
        "| Posts | 7 posts at 12.00 EUR each |"
    )
    assert text_of(annotate(table, normalized=True)) == (
        "Calculator (exact values, shown to six decimals):\n"
        "- Fence panels: 6 x EUR 55.00 = EUR 330.00\n"
        "- Posts: 7 x EUR 12.00 = EUR 84.00\n"
        "- Taxed base: EUR 414.00\n"
        "- Tax charged: EUR 41.40"
    )


def test_on_leaves_labels_as_written():
    state = BASE.replace("- Green waste: 4 bags", "- Two-way radio: 4 radios").replace(
        "- Mower hire", "- Seven-day mower hire"
    )
    block = text_of(annotate(state, normalized=True))
    assert "- Two-way radio: 4 x $6.25 = $25.00" in block
    assert "- Seven-day mower hire: 1 x $30.00 x 2 days = $60.00" in block


def test_on_reads_a_rounding_step_in_words_only_in_an_item():
    # number words are digits only in an item's detail: a rule sentence keeps its words, so v1's rule kinds decide
    state = BASE.replace("Tax of 10%", "Tax of ten percent")
    assert annotate(state, normalized=True) == annotate(state) == Abstain("unparsed-rule")


@pytest.mark.parametrize(
    "change, reason",
    [
        # a separator nothing in the text settles
        (
            lambda s: (
                s.replace("$42.50", "$42")
                .replace("$30.00", "$30")
                .replace("$1,250.00", "$1,250")
                .replace("$6.25", "$6")
            ),
            "unparsed-item",
        ),
        # a dozen priced each: per piece or per dozen
        (lambda s: s.replace("4 bags at $6.25 each", "a dozen bags at $6.25 each"), "unparsed-item"),
        # a numeric date whose day and month could swap
        (lambda s: s.replace("Work order GW-118 for Ilse Morrow:", "Visit date: 03/04/2026"), "ambiguous-date"),
        # an amount in parentheses
        (lambda s: s.replace("$6.25 each", "($6.25) each"), "unparsed-item"),
    ],
)
def test_on_abstains_where_a_form_is_unsure(change, reason):
    assert annotate(change(BASE), normalized=True) == Abstain(reason)


def test_on_keeps_v1_reasons_first():
    assert annotate("Nothing here costs anything; it opened at 1:30.", normalized=True) == Abstain("no-arithmetic")
    two = BASE.replace("$6.25", "6,25 €")
    assert annotate(two, normalized=True) == Abstain("mixed-currency")


FREIGHT = """By total charge we mean what the shipper owes under these rules.
Prices are per kilogram (1 lb = 0.4536 kg).
Add the handling fee first, then add 15% tax to everything.
Shipment SH-77 for Pell Arden:
- Cargo weight: 980 lb at $0.35 per kg
- Handling fee: 2 at $8.50 each"""


def test_on_rejects_a_factor_across_dimensions():
    ok = annotate(FREIGHT, normalized=True)
    assert text_of(ok) == text_of(annotate(FREIGHT))
    bad = FREIGHT.replace("per kilogram (1 lb = 0.4536 kg)", "per metre (1 lb = 0.4536 m)").replace("per kg", "per m")
    # a factor across dimensions gives no block with the switch off and on (the reader checks it since 0.50.2)
    assert annotate(bad) == Abstain("conflicting-rules")
    assert annotate(bad, normalized=True) == Abstain("conflicting-rules")


def test_on_is_deterministic():
    a = [annotate(EURO, normalized=True) for _ in range(3)]
    assert a[0] == a[1] == a[2]


def test_padding_before_a_time_unit_written_out_is_read_as_one_duration():
    from decisio.calculator import normalize as N

    vals = N.normalize("rounded to the nearest 15    minutes", N.document("x")).values
    assert [(v.kind, v.value) for v in vals] == [("duration", 15)]
    vals = N.normalize(
        "Rate per h:   45    m", N.document("x")
    ).values  # a bare unit letter is not reached over padding
    assert all(v.kind != "duration" for v in vals)
