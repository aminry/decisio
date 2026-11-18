# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The calculator block (`decisio.calculator`): exact line values for states about bills, payslips and freight.

C1  golden blocks (format calc-v1) for a plain-text invoice, a JSON payslip and a freight table;
C2  number formats: money with a currency, separators and two to six decimals, rounded half to even at the sixth;
    quantities without trailing zeros; a value that does not terminate;
C3  every line kind and the rules that shape it (rounding per entry or once, day counts, unit factors, percentages,
    fees and credits before or after tax, charged or withheld);
C4  every abstain reason, and no block ever states the final total;
C5  dict states, JSON text, a task context read with the state, determinism, `attach`;
C6  the renderer CLI: rows with a block, rows left unchanged, counts per reason.

  uv run pytest -q tests/unit/test_calculator.py
"""

import gzip
import json
from fractions import Fraction

import pytest

import decisio.calculator as calc
from decisio.calculator import REASONS, Abstain, Annotation, annotate, attach, render
from decisio.calculator.__main__ import main as cli_main
from decisio.readout.letters import fmt_state

HIRE = """The amount due is what the customer pays once the rules below are applied.
Each site visit is rounded on its own to the nearest 15 minutes; the rounded figures are added.
A 10% loyalty discount comes off the skip bins only, before tax.
Tax of 20% is added to the whole charge.
Only the final money figure is rounded to the cent, half a cent going up.
Invoice HX-204 for Marla Quent:
- Site visits: 1 h 7 min and 52 min; labour at $64.00 per hour
- Scaffold hire: 2 frames at $18.75 each a day for 3 days
- Skip bins: 3 bins at $95.40 each"""

HIRE_BLOCK = """Calculator (exact values, shown to six decimals):
- Site visits: 105 min x $64.00/h = $112.00
- Scaffold hire: 2 x $18.75 x 3 days = $112.50
- Skip bins: 3 x $95.40 = $286.20
- Loyalty discount: 10% of Skip bins = -$28.62
- Taxed base: $482.08
- Tax charged: $96.416"""

PAYSLIP = {
    "rules": [
        "Here the net pay means what the employee takes home under these rules.",
        "Leave the travel allowance and the locker fee out of the taxed pay: withhold 22% tax, then add the travel "
        "allowance and take off the locker fee.",
    ],
    "payslip": {
        "id": "PS-77",
        "employee": "Ondre Vask",
        "shift_hours": "7 h 13 min, 6 h 49 min and 8 h 2 min; paid at $27.30 per hour",
        "weekend_pay": "6.5 h at $31.15 an hour",
        "travel_allowance": "$12.40 a day, 5 days",
        "locker_fee": "$9.00",
    },
}

PAYSLIP_BLOCK = """Calculator (exact values, shown to six decimals):
- Shift hours: 1324 min x $27.30/h = $602.42
- Weekend pay: 6.5 x $31.15 = $202.475
- Travel allowance: 5 x $12.40 = $62.00
- Locker fee: $9.00 (taken off)
- Taxed base: $804.895
- Tax withheld: $177.0769"""

FREIGHT = """By total charge we mean what the shipper owes under these rules.
Prices are per kilogram and per kilometre (1 lb = 0.4536 kg; 1 mi = 1.6093 km).
Every date range counts its first and its last day.
Take the missed-slot credit off before tax.
Add the handling fees and take off the claims credit first, then add 15% tax to everything.
Round only the final money figure to the cent, half a cent going up.
Shipment SH-310, shipper: Teo Barrand.
| Item | Detail |
| Loading time | 51 min and 38 min; rate $41.00/h |
| Cargo weight | 980 lb at $0.35 per kg |
| Haul distance | 212 mi, $1.10/km |
| Cold storage | 2 pallets at $14.25 a day, 27 Mar to 2 Apr |
| Missed slot | 3 Apr to 4 Apr; credit $40.00 a day |
| Handling fee | $8.50 a pallet, 2 pallets |
| Claims credit | $25.00 |"""

FREIGHT_BLOCK = """Calculator (exact values, shown to six decimals):
- Loading time: 89 min x $41.00/h = $60.816667
- Cargo weight: 980 lb = 444.528 kg; 444.528 x $0.35 = $155.5848
- Haul distance: 212 mi = 341.1716 km; 341.1716 x $1.10 = $375.28876
- Cold storage: 2 x $14.25 x 7 days = $199.50
- Missed slot: $40.00 x 2 days = $80.00 (taken off)
- Handling fee: 2 x $8.50 = $17.00
- Claims credit: $25.00 (taken off)
- Taxed base: $703.190227
- Tax charged: $105.478534"""


def block(state, context=None) -> str:
    ann = annotate(state, context=context)
    assert isinstance(ann, Annotation), ann
    return ann.text


def reason(state, context=None) -> str:
    ann = annotate(state, context=context)
    assert isinstance(ann, Abstain), ann
    return ann.reason


# C1 golden blocks


def test_golden_invoice():
    ann = annotate(HIRE)
    assert ann == Annotation(version="calc-v1", text=HIRE_BLOCK, lines=6)


def test_golden_json_payslip():
    assert block(json.dumps(PAYSLIP)) == PAYSLIP_BLOCK


def test_golden_freight_table():
    ann = annotate(FREIGHT)
    assert ann.text == FREIGHT_BLOCK and ann.lines == 9


def test_block_shape():
    text = block(FREIGHT)
    assert text.split("\n")[0] == calc.HEADING == "Calculator (exact values, shown to six decimals):"
    assert all(line.startswith("- ") for line in text.split("\n")[1:])
    assert not text.startswith("\n") and not text.endswith("\n")
    assert calc.VERSIONS == ("calc-v1",)


# C2 number formats


@pytest.mark.parametrize(
    "value, text",
    [
        (Fraction("1088.51"), "$1,088.51"),
        (Fraction("992.648"), "$992.648"),
        (Fraction("211.5"), "$211.50"),
        (Fraction(5), "$5.00"),
        (Fraction("-130.6212"), "-$130.6212"),
        (Fraction(1, 3), "$0.333333"),
        (Fraction(2, 3), "$0.666667"),  # rounded at the sixth decimal
        (Fraction(2622930625, 10**7), "$262.293062"),  # an exact tie goes to the even digit
        (Fraction(2622930635, 10**7), "$262.293064"),
        (Fraction("1234567.8912345"), "$1,234,567.891234"),
        (Fraction(-1, 10**7), "$0.00"),  # rounds to zero: no negative zero
        (Fraction(0), "$0.00"),
    ],
)
def test_money_format(value, text):
    assert render.money(value, "$") == text


def test_money_other_currencies():
    assert render.money(Fraction("12.5"), "€") == "€12.50"
    assert render.money(Fraction("-1500"), "EUR") == "-EUR 1,500.00"


@pytest.mark.parametrize(
    "value, text",
    [
        (Fraction(1002), "1002"),
        (Fraction("728.028"), "728.028"),
        (Fraction("3196.6700"), "3196.67"),
        (Fraction(1, 3), "0.333333"),
        (Fraction(5, 7), "0.714286"),
        (Fraction("12.5"), "12.5"),
    ],
)
def test_quantity_format(value, text):
    assert render.quantity(value) == text


def test_values_stay_exact_until_formatted():
    # 1 min at $1/h three times: a third of a cent short each time if each line were cut before adding
    state = """Tax of 10% is added to the whole charge.
- Call A: 1 min; rate $1.00/h
- Call B: 1 min; rate $1.00/h
- Call C: 1 min; rate $1.00/h"""
    lines = block(state).split("\n")
    assert lines[1] == "- Call A: 1 min x $1.00/h = $0.016667"
    assert lines[4] == "- Taxed base: $0.05"  # 3/60 exactly, not 3 x 0.016666
    assert lines[5] == "- Tax charged: $0.005"


# C3 line kinds and rules


def test_rounding_once_on_the_total():
    state = HIRE.replace(
        "Each site visit is rounded on its own to the nearest 15 minutes; the rounded figures are added.",
        "The visits are added first, and only the total is rounded to the nearest 15 minutes.",
    )
    # 67 + 52 = 119 min, to the nearest 15: 120
    assert "- Site visits: 120 min x $64.00/h = $128.00" in block(state)


def test_no_rounding_rule_sums_exactly():
    state = HIRE.replace(
        "Each site visit is rounded on its own to the nearest 15 minutes; the rounded figures are added.\n", ""
    )
    assert "- Site visits: 119 min x $64.00/h = $126.933333" in block(state)


def test_lengths_rounded_per_piece():
    state = """Tax of 10% is added to the whole charge.
Round every cut piece to the nearest 0.5 m first, then add them up.
- Cut pieces: 2.2 m, 3.9 m and 1.6 m; cutting $10.00 a metre"""
    # 2.0 + 4.0 + 1.5
    assert "- Cut pieces: 7.5 m x $10.00/m = $75.00" in block(state)


def test_day_count_rule_decides_the_last_day():
    excl = FREIGHT.replace(
        "Every date range counts its first and its last day.", "Every date range counts its first day but not its last."
    )
    text = block(excl)
    assert "- Cold storage: 2 x $14.25 x 6 days = $171.00" in text
    assert "- Missed slot: $40.00 x 1 days = $40.00 (taken off)" in text


def test_date_range_across_a_year_end():
    state = FREIGHT.replace("27 Mar to 2 Apr", "30 Dec to 2 Jan")
    assert "- Cold storage: 2 x $14.25 x 4 days = $114.00" in block(state)


def test_conversion_with_the_factor_the_other_way():
    state = FREIGHT.replace("(1 lb = 0.4536 kg; 1 mi = 1.6093 km)", "(1 kg = 2.5 lb; 1 mi = 1.6093 km)")
    # 980 lb / 2.5 = 392 kg exactly
    assert "- Cargo weight: 980 lb = 392 kg; 392 x $0.35 = $137.20" in block(state)


def test_fees_and_credits_after_tax():
    state = FREIGHT.replace(
        "Add the handling fees and take off the claims credit first, then add 15% tax to everything.",
        "Add 15% tax to everything but the handling fees and the claims credit, then add the handling fees and take "
        "off the claims credit.",
    )
    text = block(state)
    assert "- Handling fee: 2 x $8.50 = $17.00" in text  # the lines are the same
    assert "- Taxed base: $711.190227" in text  # the base leaves them out: 703.190227 - 17 + 25


def test_surcharge_adds_and_names_follow_the_state():
    state = HIRE.replace(
        "A 10% loyalty discount comes off the skip bins only, before tax.",
        "Charge 12.5% of the scaffold hire alone as the weekend surcharge, before tax.",
    )
    assert "- Weekend surcharge: 12.5% of Scaffold hire = $14.0625" in block(state)


def test_fee_counted_per_listed_visit():
    state = """Here the balance due means what the client pays after these rules.
Add the call-out fees and take off the voucher first; then add 10% tax on the result.
Job KW-12 - Pell Arden
Visits: 30 min, 45 min and 1 h; labour $60.00 an hour
Call-out fee: $20.00 per visit
Voucher: $15.00"""
    text = block(state)
    assert "- Visits: 135 min x $60.00/h = $135.00" in text
    assert "- Call-out fee: 3 x $20.00 = $60.00" in text
    assert "- Voucher: $15.00 (taken off)" in text
    assert "- Taxed base: $180.00" in text


def test_json_field_names_take_their_hyphens_from_the_rules():
    pay = json.loads(json.dumps(PAYSLIP))
    pay["payslip"]["night_shift_pay"] = "10 h at $30.00 an hour"
    pay["rules"].append("A 10% shift premium is paid on the night-shift pay only, before tax.")
    text = block(json.dumps(pay))
    assert "- Night-shift pay: 10 x $30.00 = $300.00" in text
    assert "- Shift premium: 10% of Night-shift pay = $30.00" in text


def test_neutral_tax_sentence_reads_the_document():
    state = HIRE.replace("Tax of 20% is added to the whole charge.", "Work out 20% tax on everything.")
    assert block(state).endswith("- Tax charged: $96.416")


# C4 abstain reasons


ABSTAIN_CASES = {
    "no-arithmetic": "Sort the parcels by size before the van leaves.",
    "not-english": "Die Rechnung ist fällig.\nDer Betrag wird mit der Steuer berechnet.\n- Teile: 3 Stück zu $4.00 je",
    "mixed-currency": HIRE.replace("3 bins at $95.40 each", "3 bins at €95.40 each"),
    "unparsed-item": HIRE.replace("3 bins at $95.40 each", "3 bins at $95.40 each, the third one free"),
    "unparsed-rule": HIRE.replace("Tax of 20%", "Orders over 3 bins ship free. Tax of 20%"),
    "unsupported-rule": HIRE.replace("skip bins only, before tax", "skip bins only, after tax"),
    "unmatched-label": HIRE.replace("off the skip bins only", "off the ladders only"),
    "ambiguous-label": HIRE + "\n- Skip bin: 1 bin at $95.40 each",
    "ambiguous-rule": HIRE.replace("A 10% loyalty discount comes off", "A 10% service levy applies to"),
    "conflicting-rules": HIRE + "\nTax of 15% is added to the whole charge.",
    "missing-rule": FREIGHT.replace("Every date range counts its first and its last day.\n", ""),
    "unaccounted-amount": HIRE + "\n- Booking deposit: $50.00",
    "ambiguous-date": FREIGHT.replace("27 Mar to 2 Apr", "27 Feb to 2 Mar"),
    "several-records": HIRE + "\nInvoice HX-205 for Dov Remy:\n- Ladders: 2 ladders at $9.00 each",
    "no-tax-rule": HIRE.replace("Tax of 20% is added to the whole charge.\n", ""),
}


@pytest.mark.parametrize("why", sorted(ABSTAIN_CASES))
def test_abstain_reasons(why):
    assert reason(ABSTAIN_CASES[why]) == why


def test_every_reason_is_tested():
    assert set(ABSTAIN_CASES) | {"internal-error"} == set(REASONS)


def test_internal_error_abstains(monkeypatch):
    def boom(parts):
        raise ZeroDivisionError

    monkeypatch.setattr(calc.parse, "parse", boom)
    assert annotate(HIRE) == Abstain("internal-error")


def test_ambiguous_credit_sentence_abstains():
    state = FREIGHT.replace(
        "Take the missed-slot credit off before tax.", "The missed-slot credit comes off the taxed amount."
    )
    assert reason(state) == "ambiguous-rule"


def test_rounding_tie_abstains():
    tie = HIRE.replace("nearest 15 minutes", "nearest 10 minutes").replace("1 h 7 min", "1 h 5 min")
    assert reason(tie) == "ambiguous-rule"  # 65 min is half way between 60 and 70


@pytest.mark.parametrize(
    "old, new",
    [
        ("Tax of 20% is added to the whole charge.", "Tax of 20% is added to the whole charge except the skip bins."),
        ("Tax of 20% is added to the whole charge.", "Tax of 20% is added to everything but the skip bins."),
        ("the rounded figures are added.", "the rounded figures are added, with a minimum of one hour."),
        ("once the rules below are applied.", "once the rules below are applied, less any deposit."),
    ],
)
def test_a_clause_the_rule_kind_does_not_read_abstains(old, new):
    assert reason(HIRE.replace(old, new)) == "unparsed-rule"


def test_a_day_count_with_a_condition_abstains():
    state = FREIGHT.replace("counts its first and its last day.", "counts its first and its last day, weekends aside.")
    assert reason(state) == "unparsed-rule"


@pytest.mark.parametrize("label", ["Discount", "Deposit paid", "Tax", "Subtotal"])
def test_a_priced_line_whose_name_says_it_is_no_charge_abstains(label):
    assert reason(HIRE + f"\n- {label}: 2 units at $5.00 each") == "unaccounted-amount"


def test_no_conversion_rule_is_recognised_not_computed():
    state = FREIGHT.replace(
        "Prices are per kilogram and per kilometre (1 lb = 0.4536 kg; 1 mi = 1.6093 km).",
        "Quantities are used exactly as printed; do not convert them.",
    )
    assert reason(state) == "unsupported-rule"


def test_a_claimed_amount_in_the_state_abstains():
    state = HIRE + "\nThe invoice states a total of $580.00."
    assert reason(state) == "unparsed-rule"


# a count followed by a word that stands for a number of things, or by a number in words: what is counted is not
# certain (2 dozen eggs are 24 eggs), so no block; one such word priced per that same word is certain

SUPPLY = """The amount due is what the buyer pays once the rules below are applied.
Tax of 8% is added to the whole bill.
Supply order SO-41 for Wren Tallis:
- Notebooks: 5 notebooks at $2.40 each
- {label}: {detail}"""


@pytest.mark.parametrize(
    "detail",
    [
        "2 dozen eggs at $3.00 each",
        "2 dozen at $3.00 each",
        "2 dozen eggs at $0.25 per egg",
        "1.5 dozen rolls at $0.40 per roll",
        "2 doz eggs at $3.00 each",
        "1 gross of pens at $20.00",
        "2 gross pens at $0.10 each",
        "2 hundred bolts at $0.10 each",
        "3 thousand sheets at $0.01 each",
        "2 score nails at $0.05 each",
        "3 pairs of gloves at $4.00 each",
        "3 pairs at $4.00 each",
        "2 brace of birds at $9.00 each",
        "2 couple of chairs at $9.00 each",
        "2 quarters at $500.00 each",
        "2 packs of six at $3.00 each",
        "2 boxes of ten at $5.00 each",
        "3 cases of twenty-four at $20.00 each",
        "2 sets of three at $5.00 each",
        "2 dozen, $0.25/egg",
        "2 dozen eggs at $3.00 a day for 3 days",
        "2 pairs of skis at $15.00 each a day for 3 days",
        # forms the reader already left alone; kept here so that they stay so
        "2 x 3 boxes at $5.00 each",
        "2 packs of 6 at $3.00 each",
        "3-5 boxes at $4.00 each",
        "3 to 5 boxes at $4.00 each",
        "2 or 3 boxes at $4.00 each",
        "1/2 kg at $3.00 per kg",
    ],
)
def test_a_count_that_is_not_certain_abstains(detail):
    assert reason(SUPPLY.format(label="Goods", detail=detail)) == "unparsed-item"


@pytest.mark.parametrize(
    "detail, line",
    [
        ("2 dozen at $3.00 a dozen", "2 x $3.00 = $6.00"),
        ("3 pairs at $4.00 a pair", "3 x $4.00 = $12.00"),
        ("2 dozen, $3.00/dozen", "2 x $3.00 = $6.00"),
        ("3 six-packs at $9.00 each", "3 x $9.00 = $27.00"),
        ("2 half-pallets at $40.00 each", "2 x $40.00 = $80.00"),
        ("4 reams at $5.00 each", "4 x $5.00 = $20.00"),
    ],
)
def test_a_count_in_the_unit_it_is_priced_in_reads(detail, line):
    assert f"- Goods: {line}" in block(SUPPLY.format(label="Goods", detail=detail)).split("\n")


# a unit factor has to join two units of one dimension

CARRIAGE = """By total charge we mean what the sender owes under these rules.
Prices are per kilogram (1 lb = 0.4536 kg).
Add 12% tax to the whole charge.
Consignment CN-9 for Odile Brask:
- Parcel weight: 50 lb at $1.20 per kg"""


def test_a_unit_factor_within_one_dimension_reads():
    assert block(CARRIAGE).split("\n")[1] == "- Parcel weight: 50 lb = 22.68 kg; 22.68 x $1.20 = $27.216"


@pytest.mark.parametrize(
    "rule, item",
    [
        ("Prices are per metre (1 lb = 0.4536 m).", "50 lb at $1.20 per m"),
        ("Prices are per litre (1 kg = 1.25 L).", "50 kg at $1.20 per L"),
        ("Prices are per hour (1 km = 0.75 h).", "50 km at $1.20 per h"),
    ],
)
def test_a_unit_factor_across_dimensions_abstains(rule, item):
    state = CARRIAGE.replace("Prices are per kilogram (1 lb = 0.4536 kg).", rule).replace("50 lb at $1.20 per kg", item)
    assert reason(state) == "conflicting-rules"


def test_a_unit_factor_across_dimensions_abstains_even_unused():
    state = CARRIAGE.replace("(1 lb = 0.4536 kg)", "(1 lb = 0.4536 kg; 1 mi = 2.2 kg)")
    assert reason(state) == "conflicting-rules"


def test_a_unit_factor_with_a_unit_the_reader_does_not_know_is_unchanged():
    state = CARRIAGE.replace("(1 lb = 0.4536 kg)", "(1 lb = 0.4536 kg; 1 stone = 6.35 kg)")
    assert reason(state) == "unparsed-rule"


@pytest.mark.parametrize("state", [HIRE, json.dumps(PAYSLIP), FREIGHT])
def test_never_the_final_total(state):
    from decisio.calculator import compute
    from decisio.calculator.parse import parse

    case = parse([state])
    base = compute.taxed_base(case)
    tax = base * case.tax.rate / 100
    total = base + tax if case.tax.kind == "charged" else base - tax
    total += sum((f.value for f in case.fees if f.label in case.outside), Fraction(0))
    total -= sum((c.amount for c in case.credits if c.label in case.outside), Fraction(0))
    text = block(state)
    cents = (total * 100 + Fraction(1, 2)).__floor__()
    assert render.money(Fraction(cents, 100), case.currency) not in text
    assert render.money(total, case.currency) not in text


def test_unknown_version_is_a_programming_error():
    with pytest.raises(ValueError, match="calc-v9"):
        annotate(HIRE, version="calc-v9")


# C5 state forms, context, determinism, attach


def test_dict_state_reads_as_rendered():
    state = {
        "Site visits": "1 h 7 min and 52 min; labour at $64.00 per hour",
        "Scaffold hire": "2 frames at $18.75 each a day for 3 days",
        "Skip bins": "3 bins at $95.40 each",
        "_source": "ignored, never shown",
    }
    rules = "\n".join(HIRE.split("\n")[:5])
    text = block(state, context=rules)
    assert text == HIRE_BLOCK
    assert annotate(state, context=rules) == annotate(fmt_state(state), context=rules)


def test_rules_in_the_context_amounts_in_the_state():
    lines = HIRE.split("\n")
    context, state = "\n".join(lines[:5]), "\n".join(lines[5:])
    assert block(state, context=context) == HIRE_BLOCK
    assert reason(state) == "no-tax-rule"  # without the context the rules are missing


def test_deterministic():
    assert {annotate(FREIGHT).text for _ in range(5)} == {FREIGHT_BLOCK}


def test_attach():
    ann = annotate(HIRE)
    assert attach(HIRE, ann) == HIRE + "\n" + HIRE_BLOCK


# C6 renderer CLI


def test_render_cli(tmp_path, capsys):
    src, dst = tmp_path / "in.jsonl.gz", tmp_path / "out.jsonl.gz"
    rows = [
        {"id": "a", "state": HIRE, "gold": 0},
        {"id": "b", "state": "Sort the parcels by size.", "gold": 1},
        {"id": "c", "state": json.dumps(PAYSLIP), "gold": 0},
    ]
    with gzip.open(src, "wt") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    assert cli_main(["render", str(src), "--out", str(dst)]) == 0
    counts = json.loads(capsys.readouterr().out)
    assert counts == {"rows": 3, "shown": 2, "none: no-arithmetic": 1}
    out = [json.loads(ln) for ln in gzip.open(dst, "rt")]
    assert out[0] == {**rows[0], "state": HIRE + "\n" + HIRE_BLOCK}
    assert out[1] == rows[1]
    assert out[2]["state"] == json.dumps(PAYSLIP) + "\n" + PAYSLIP_BLOCK
