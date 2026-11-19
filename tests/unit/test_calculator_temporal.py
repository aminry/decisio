# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The time and unit calculator (`time-v1`): facts from the text's own dates, clock times and quantities, or none."""

import pytest

from decisio.calculator import Abstain, Annotation, annotate
from decisio.calculator import temporal as T


def lines(text):
    return T.facts(text)


def test_a_zoned_time_with_its_date_is_written_in_utc():
    text = "The claim was submitted at 18:30 UTC-4 on 29 February 2028."
    assert "18:30 UTC-4, 29 Feb 2028 = 22:30 UTC, 29 Feb 2028" in lines(text)


def test_a_zoned_time_without_a_date_is_written_with_its_day_shift():
    assert "01:30 CEST = 23:30 UTC the previous day" in lines("The dose was taken at 01:30 CEST.")
    assert "13:00 CDT = 18:00 UTC" in lines("The last dose was at 13:00 CDT.")


def test_the_real_time_between_two_dated_zoned_times():
    text = "Previous dose: 13:00 CDT on 3 Sep 2026. On-board dose: 01:30 CEST on 4 Sep 2026."
    assert "From 13:00 CDT, 3 Sep 2026 to 01:30 CEST, 4 Sep 2026: 5 h 30 min of real time" in lines(text)


def test_a_clock_change_is_counted_when_both_ends_carry_their_zone():
    text = "Shift ends 18:00 CEST on 24 Oct 2026. Next shift starts 04:30 CET on 25 Oct 2026."
    assert "From 18:00 CEST, 24 Oct 2026 to 04:30 CET, 25 Oct 2026: 11 h 30 min of real time" in lines(text)


def test_a_clock_span_is_said_to_be_by_the_clock():
    assert "21:00 to 07:30: 10 h 30 min by the clock (into the next day)" in lines("Rest from 21:00 to 07:30.")


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("It ends 18 months after 31 August 2026.",
         "31 Aug 2026 + 18 months: 31 Feb 2028 does not exist; February 2028 has 29 days"),
        ("Due 2 months after 15 January 2027.", "15 Jan 2027 + 2 months = 15 Mar 2027 (Monday)"),
        ("Payable 30 days after 3 Sep 2026.", "3 Sep 2026 + 30 days = 3 Oct 2026 (Saturday)"),
        ("Notice 2 weeks before 1 March 2028.", "1 Mar 2028 - 2 weeks = 16 Feb 2028 (Wednesday)"),
    ],
)  # fmt: skip
def test_calendar_shifts_state_month_ends_and_do_not_resolve_them(text, line):
    assert line in lines(text)


def test_weekdays_of_full_dates_and_a_wrong_weekday_is_not_read():
    assert "29 Feb 2028 is a Tuesday" in lines("Deadline: 29 February 2028, in business days.")
    assert lines("Deadline: 29 February 2028.") == []  # no talk of weekdays: no weekday line
    # the text's weekday is wrong: neither is read
    assert not any("29 Feb 2028" in x for x in lines("Deadline: Monday 29 February 2028."))


def test_a_date_without_a_year_is_read_only_when_the_text_has_one_year():
    assert "3 Sep 2026 is a Thursday" in lines("Year 2026. The visit is on 3 Sep, a weekday.")
    assert not any("Sep" in x for x in lines("In 2026 and 2027 the visit is on 3 Sep, a weekday."))


@pytest.mark.parametrize(
    "text",
    [
        "The meeting is at 15:00 CST.",  # CST: US Central or China Standard Time
        "The call is at 09:00 IST.",  # IST: India, Ireland or Israel
        "Claims close at 17:00 Rotterdam time.",  # a place name, not an abbreviation
    ],
)
def test_ambiguous_zones_are_not_read(text):
    assert not any("UTC" in x for x in lines(text))


def test_amounts_are_not_read_as_times():
    assert lines("Lunch boxes: 36 at $12.41 each, delivery 10.30 EUR.") == []


def test_units_are_converted_only_when_the_text_uses_both_of_a_pair():
    text = "Net weight 5,280 lb. The position limit is 2,500 kg."
    assert "5,280 lb = 2394.967714 kg" in lines(text)
    assert "2,500 kg = 5511.556555 lb" in lines(text)
    assert lines("Net weight 5,280 lb.") == []


def test_no_date_time_or_unit_means_no_block():
    assert lines("Which of these is a mammal? A whale or a trout.") == []
    out = annotate("Which of these is a mammal? A whale or a trout.", version="time-v1")
    assert isinstance(out, Abstain) and out.reason == "no-time-facts"


def test_annotate_writes_the_time_block_and_both_blocks_together():
    text = "The dose was taken at 01:30 CEST."
    out = annotate(text, version="time-v1")
    assert isinstance(out, Annotation)
    assert out.text.split("\n")[0] == T.HEADING == "Calculator, times and units (exact):"
    both = annotate(text, version="calc-v1+time-v1")
    assert isinstance(both, Annotation) and both.text == out.text  # no billing case: the time block alone


def test_a_citys_time_is_read_with_its_daylight_saving_rule_on_the_stated_date():
    text = "Year 2026.\n- 3 Sep, 13:00 Chicago time: took dose.\n- 4 Sep, 01:30 by my watch (Madrid time): next dose."
    out = lines(text)
    assert "13:00 Chicago time, 3 Sep 2026 = 18:00 UTC, 3 Sep 2026" in out  # CDT, UTC-5
    assert "01:30 by my watch (Madrid time), 4 Sep 2026 = 23:30 UTC, 3 Sep 2026" in out  # CEST, UTC+2
    assert any(x.endswith(": 5 h 30 min of real time") and x.startswith("From 13:00 Chicago time") for x in out)
    winter = lines("Year 2026.\n- 3 Dec, 13:00 Chicago time: took dose.")
    assert "13:00 Chicago time, 3 Dec 2026 = 19:00 UTC, 3 Dec 2026" in winter  # CST, UTC-6


def test_a_citys_time_needs_its_date_and_is_not_read_in_a_clock_change_hour():
    assert lines("The call is at 13:00 Chicago time.") == []
    # 25 Oct 2026, 02:30 in Berlin happens twice (clocks go back at 03:00)
    assert not any("UTC" in x for x in lines("On 25 Oct 2026, 02:30 Berlin time the job ran."))
