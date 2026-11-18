# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The structured calculator input (`decisio.calculator.from_input`, `calculator_input`).

I1  the same block as the parser gives for the same amounts, byte for byte (one renderer);
I2  every item kind, rounding, outside-the-base fees and credits, withheld tax;
I3  strict validation: every invalid or inconsistent field raises CalculatorInputError naming its path; nothing
    partial is ever returned.

  uv run pytest -q tests/unit/test_calculator_input.py
"""

import copy

import pytest

from decisio.calculator import Annotation, CalculatorInputError, annotate, from_input

FREIGHT = """By total charge we mean what the shipper owes under these rules.
Prices are per kilogram and per kilometre (1 lb = 0.4536 kg; 1 mi = 1.6093 km).
Every date range counts its first and its last day.
Take the missed-slot credit off before tax.
A 5% fuel surcharge applies to the haul distance only, before tax.
Add 15% tax to everything but the handling fees and the claims credit, then add the handling fees and take off the
claims credit.
Shipment SH-310, shipper: Teo Barrand.
| Item | Detail |
| Loading time | 51 min and 38 min; rate $41.00/h |
| Cargo weight | 980 lb at $0.35 per kg |
| Haul distance | 212 mi, $1.10/km |
| Cold storage | 2 pallets at $14.25 a day, 27 Mar to 2 Apr |
| Missed slot | 3 Apr to 4 Apr; credit $40.00 a day |
| Handling fee | $8.50 a pallet, 2 pallets |
| Claims credit | $25.00 |""".replace("the\nclaims", "the claims")

FREIGHT_INPUT = {
    "currency": "$",
    "items": [
        {"kind": "time", "label": "Loading time", "minutes": [51, 38], "rate": "41.00", "per": "h"},
        {
            "kind": "convert",
            "label": "Cargo weight",
            "quantity": 980,
            "unit": "lb",
            "to_unit": "kg",
            "factor": "0.4536",
            "price": "0.35",
        },
        {
            "kind": "convert",
            "label": "Haul distance",
            "quantity": "212",
            "unit": "mi",
            "to_unit": "km",
            "factor": "1.6093",
            "price": "1.10",
        },
        {"kind": "days", "label": "Cold storage", "quantity": 2, "price": "14.25", "days": 7},
        {"kind": "daycredit", "label": "Missed slot", "price": "40.00", "days": 2},
    ],
    "adjustments": [{"label": "Fuel surcharge", "percent": 5, "of": "Haul distance", "direction": "add"}],
    "fees": [{"label": "Handling fee", "price": "8.50", "count": 2}],
    "credits": [{"label": "Claims credit", "amount": "25.00"}],
    "tax": {"rate": "15", "kind": "charged", "outside": ["Handling fee", "Claims credit"]},
}


def test_same_block_as_the_parser():
    parsed = annotate(FREIGHT)
    assert isinstance(parsed, Annotation)
    assert from_input(FREIGHT_INPUT) == parsed
    assert "- Fuel surcharge: 5% of Haul distance = $18.764438" in parsed.text


def test_count_length_rounding_and_withheld_tax():
    data = {
        "currency": "EUR",
        "items": [
            {"kind": "count", "label": "Training pay", "quantity": "9", "price": "23.50"},
            {
                "kind": "time",
                "label": "Day shifts",
                "minutes": [568, 430],
                "rate": "59.44",
                "per": "h",
                "round": {"step": 6, "mode": "each"},
            },
            {
                "kind": "length",
                "label": "Cut pieces",
                "lengths": ["3.57", "4.11", "1.59"],
                "unit": "m",
                "rate": "42.94",
                "round": {"step": "0.25", "mode": "total"},
            },
        ],
        "adjustments": [{"label": "Pension contribution", "percent": "5", "of": "Training pay", "direction": "deduct"}],
        "fees": [{"label": "Union fee", "price": "12"}],
        "tax": {"rate": 18, "kind": "withheld"},
    }
    assert from_input(data).text == (
        "Calculator (exact values, shown to six decimals):\n"
        "- Training pay: 9 x EUR 23.50 = EUR 211.50\n"
        "- Day shifts: 1002 min x EUR 59.44/h = EUR 992.648\n"
        "- Cut pieces: 9.25 m x EUR 42.94/m = EUR 397.195\n"
        "- Pension contribution: 5% of Training pay = -EUR 10.575\n"
        "- Union fee: EUR 12.00\n"
        "- Taxed base: EUR 1,602.768\n"
        "- Tax withheld: EUR 288.49824"
    )


def test_input_is_not_mutated():
    data = copy.deepcopy(FREIGHT_INPUT)
    from_input(data)
    assert data == FREIGHT_INPUT


def bad(mutate):
    data = copy.deepcopy(FREIGHT_INPUT)
    mutate(data)
    return data


@pytest.mark.parametrize(
    "mutate, field",
    [
        (lambda d: d["items"][0].update(rate=41.0), "items[0].rate"),  # a float is not exact
        (lambda d: d["items"][0].update(rate=True), "items[0].rate"),
        (lambda d: d["items"][0].update(rate="41,00"), "items[0].rate"),
        (lambda d: d["items"][3].update(price="-1"), "items[3].price"),
        (lambda d: d["items"][3].update(days=0), "items[3].days"),
        (lambda d: d["items"][3].update(colour="red"), "items[3].colour"),
        (lambda d: d["items"][3].pop("price"), "items[3].price"),
        (lambda d: d["items"][3].update(kind="bundle"), "items[3].kind"),
        (lambda d: d["items"][0].update(per="day"), "items[0].per"),
        (lambda d: d["items"][0].update(minutes=[]), "items[0].minutes"),
        (lambda d: d["items"][0].update(round={"step": 0, "mode": "each"}), "items[0].round.step"),
        (lambda d: d["items"][0].update(round={"step": 15, "mode": "nearest"}), "items[0].round.mode"),
        (lambda d: d["items"][0].update(minutes=[45], round={"step": 30, "mode": "each"}), "items[0].round"),
        (lambda d: d["items"][1].update(to_unit="lb"), "items[1].to_unit"),
        (lambda d: d["items"][1].update(factor="0"), "items[1].factor"),
        (lambda d: d["items"][1].update(label="Loading time"), "items[1].label"),  # used twice
        (lambda d: d["items"][1].update(label="Taxed base"), "items[1].label"),  # reserved
        (lambda d: d["items"][1].update(label="Weight: gross"), "items[1].label"),
        (lambda d: d["items"][1].update(label="  "), "items[1].label"),
        (lambda d: d.update(items=[]), "items"),
        (lambda d: d.update(currency="dollars"), "currency"),
        (lambda d: d.update(extra=1), "extra"),
        (lambda d: d.pop("tax"), "tax"),
        (lambda d: d["adjustments"][0].update(of="Ladders"), "adjustments[0].of"),
        (lambda d: d["adjustments"][0].update(of="Missed slot"), "adjustments[0].of"),  # a credit, not priced
        (lambda d: d["adjustments"][0].update(direction="up"), "adjustments[0].direction"),
        (lambda d: d["adjustments"][0].update(percent=0), "adjustments[0].percent"),
        (lambda d: d["fees"][0].update(count=0), "fees[0].count"),
        (lambda d: d["credits"][0].update(amount=None), "credits[0].amount"),
        (lambda d: d["tax"].update(rate="120"), "tax.rate"),
        (lambda d: d["tax"].update(kind="vat"), "tax.kind"),
        (lambda d: d["tax"].update(outside=["Cargo weight"]), "tax.outside[0]"),  # a priced item
        (lambda d: d["tax"].update(outside=["Claims credit", "Claims credit"]), "tax.outside[1]"),
        (lambda d: d.update(fees={"label": "x"}), "fees"),
    ],
)
def test_invalid_input_names_the_field(mutate, field):
    with pytest.raises(CalculatorInputError) as e:
        from_input(bad(mutate))
    assert str(e.value).startswith(f"calculator_input.{field}:"), str(e.value)


def test_not_an_object():
    with pytest.raises(CalculatorInputError, match="must be an object"):
        from_input(["items"])


def test_unknown_version():
    with pytest.raises(ValueError, match="calc-v2"):
        from_input(FREIGHT_INPUT, version="calc-v2")


# ---- unhashable kinds and hard bounds --------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", [["count"], {"k": 1}, 3, None, True])
def test_a_kind_of_any_type_is_a_named_error(kind):
    data = copy.deepcopy(FREIGHT_INPUT)
    data["items"][0]["kind"] = kind
    with pytest.raises(CalculatorInputError, match=r"calculator_input\.items\[0\]\.kind"):
        from_input(data)


@pytest.mark.parametrize(
    ("path", "edit"),
    [
        ("items", lambda d: d.__setitem__("items", [dict(d["items"][3], label=f"I{i}") for i in range(65)])),
        ("fees", lambda d: d.__setitem__("fees", [{"label": f"F{i}", "price": "1"} for i in range(33)])),
        ("credits", lambda d: d.__setitem__("credits", [{"label": f"C{i}", "amount": "1"} for i in range(33)])),
        ("items[0].minutes", lambda d: d["items"][0].__setitem__("minutes", [1] * 65)),
        ("items[3].price", lambda d: d["items"][3].__setitem__("price", "1" * 5000)),
        ("items[3].price", lambda d: d["items"][3].__setitem__("price", "1" * 15)),
        ("items[3].price", lambda d: d["items"][3].__setitem__("price", "1.1234567")),
        ("items[3].quantity", lambda d: d["items"][3].__setitem__("quantity", 10**14)),
        ("items[3].days", lambda d: d["items"][3].__setitem__("days", 10**9)),
    ],
)
def test_bounds_are_named_errors(path, edit):
    data = copy.deepcopy(FREIGHT_INPUT)
    edit(data)
    with pytest.raises(CalculatorInputError) as e:
        from_input(data)
    assert f"calculator_input.{path}" in str(e.value)


def test_the_largest_accepted_numbers_still_render():
    data = copy.deepcopy(FREIGHT_INPUT)
    data["items"][3].update(quantity="9" * 14 + ".999999", price="9" * 14 + ".999999", days=3660)
    assert isinstance(from_input(data), Annotation)
