# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The date and time calculator (`time-v1`): the deciding facts from the text's own dates and clock times, or none."""

import pytest

from decisio.calculator import Abstain, Annotation, annotate
from decisio.calculator import temporal as T


def lines(text):
    return T.facts(text)


def test_a_dated_zoned_time_is_written_in_utc():
    text = "The claim was submitted at 18:30 UTC-4 on 29 February 2028."
    assert lines(text) == ["18:30 UTC-4, 29 Feb 2028 = 22:30 UTC, 29 Feb 2028"]


def test_a_zoned_time_without_a_date_gives_no_line():
    assert lines("The dose was taken at 01:30 CEST.") == []


def test_the_real_time_between_two_dated_zoned_times_with_minutes_always_shown():
    text = "Previous dose: 13:00 CDT on 3 Sep 2026. On-board dose: 01:30 CEST on 4 Sep 2026."
    assert "13:00 CDT, 3 Sep 2026 to 01:30 CEST, 4 Sep 2026: 5 h 30 min" in lines(text)
    text = "Out at 09:00 UTC on 1 Mar 2026. Back at 11:00 UTC on 1 Mar 2026."
    assert "09:00 UTC, 1 Mar 2026 to 11:00 UTC, 1 Mar 2026: 2 h 0 min" in lines(text)
    text = "Landed 10:00 UTC on 2 Mar 2026. Departed 08:00 UTC on 2 Mar 2026."
    assert "10:00 UTC, 2 Mar 2026 to 08:00 UTC, 2 Mar 2026: minus 2 h 0 min" in lines(text)


def test_a_clock_change_is_counted_when_both_ends_carry_their_zone():
    text = "Shift ends 18:00 CEST on 24 Oct 2026. Next shift starts 04:30 CET on 25 Oct 2026."
    assert "18:00 CEST, 24 Oct 2026 to 04:30 CET, 25 Oct 2026: 11 h 30 min" in lines(text)


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("It ends 18 months after 31 August 2026.",
         "31 Aug 2026 + 18 months: 31 Feb 2028 does not exist; February 2028 has 29 days"),
        ("Due 2 months after 15 January 2027.", "15 Jan 2027 + 2 months: 15 Mar 2027"),
        ("Due 1 month after 15 January 2027.", "15 Jan 2027 + 1 month: 15 Feb 2027"),
        ("Payable 30 days after 3 Sep 2026.", "3 Sep 2026 + 30 days: 3 Oct 2026"),
        ("Notice 9 weeks after 23 Jan 2026.", "23 Jan 2026 + 9 weeks: 27 Mar 2026"),
        ("Notice 2 weeks before 1 March 2028.", "1 Mar 2028 - 2 weeks: 16 Feb 2028"),
    ],
)  # fmt: skip
def test_calendar_shifts_state_month_ends_and_do_not_resolve_them(text, line):
    assert lines(text) == [line]


def test_a_date_without_a_year_is_read_only_when_the_text_has_one_year():
    assert lines("Year 2026. Payable 30 days after 3 Sep.") == ["3 Sep 2026 + 30 days: 3 Oct 2026"]
    assert lines("In 2026 and 2027: payable 30 days after 3 Sep.") == []


def test_a_wrong_weekday_makes_its_date_unread():
    assert lines("Payable 30 days after Monday 3 Sep 2026.") == []  # 3 Sep 2026 is a Thursday


@pytest.mark.parametrize(
    "text",
    [
        "The meeting is at 15:00 CST on 3 Sep 2026.",  # CST: US Central or China Standard Time
        "The call is at 09:00 IST on 3 Sep 2026.",  # IST: India, Ireland or Israel
        "Claims close at 17:00 Gotham time on 3 Sep 2026.",  # a place not on the list
    ],
)
def test_ambiguous_or_unknown_zones_are_not_read(text):
    assert lines(text) == []


def test_amounts_are_not_read_as_times():
    assert lines("On 3 Sep 2026: 36 lunch boxes at $12.41 each, delivery 10.30 EUR.") == []


def test_no_weekdays_units_or_clock_spans():
    text = "Rota for Mon 21 Sep 2026: 21:00 to 07:30. Net 5,280 lb, limit 2,500 kg."
    assert lines(text) == []


def test_a_citys_time_is_read_with_its_daylight_saving_rule_on_the_stated_date():
    text = "Year 2026.\n- 3 Sep, 13:00 Chicago time: took dose.\n- 4 Sep, 01:30 by my watch (Madrid time): next dose."
    assert lines(text) == [
        "13:00 Chicago time, 3 Sep 2026 = 18:00 UTC, 3 Sep 2026",  # CDT, UTC-5
        "01:30 by my watch (Madrid time), 4 Sep 2026 = 23:30 UTC, 3 Sep 2026",  # CEST, UTC+2
        "13:00 Chicago time, 3 Sep 2026 to 01:30 by my watch (Madrid time), 4 Sep 2026: 5 h 30 min",
    ]
    assert lines("Year 2026.\n- 3 Dec, 13:00 Chicago time: took dose.") == [
        "13:00 Chicago time, 3 Dec 2026 = 19:00 UTC, 3 Dec 2026"  # CST, UTC-6
    ]


def test_a_citys_time_needs_its_date_and_is_not_read_in_a_clock_change_hour():
    assert lines("The call is at 13:00 Chicago time.") == []
    assert lines("On 25 Oct 2026, 02:30 Berlin time the job ran.") == []  # 02:30 happens twice that night


def test_no_date_or_time_means_no_block():
    assert lines("Which of these is a mammal? A whale or a trout.") == []
    out = annotate("Which of these is a mammal? A whale or a trout.", version="time-v1")
    assert isinstance(out, Abstain) and out.reason == "no-time-facts"


def test_annotate_writes_the_time_block_and_both_blocks_together():
    text = "Payable 30 days after 3 Sep 2026."
    out = annotate(text, version="time-v1")
    assert isinstance(out, Annotation)
    assert out.text == "Calculator, times (exact):\n- 3 Sep 2026 + 30 days: 3 Oct 2026"
    both = annotate(text, version="calc-v1+time-v1")
    assert isinstance(both, Annotation) and both.text == out.text  # no billing case: the time block alone


def test_a_zone_joined_to_another_is_not_read():
    assert lines("The page fired at 19:45 BST/IST on 3 Sep 2026.") == []


def test_a_table_rows_zone_cell_beside_its_time_cell():
    text = (
        "| Seq. | Event | Date | Local Time | Time Zone |\n|---|---|---|---|---|\n"
        "| 1 | Picked up | 14 July 2026 | 08:00 | JST |\n| 2 | Departed hub | 14 July 2026 | 18:30 | UTC-3 |\n"
        "| 3 | Scanned | 15 July 2026 | 19:45 | BST/IST |"
    )
    out = lines(text)
    assert "08:00 JST, 14 Jul 2026 = 23:00 UTC, 13 Jul 2026" in out
    assert "18:30 UTC-3, 14 Jul 2026 = 21:30 UTC, 14 Jul 2026" in out
    assert "08:00 JST, 14 Jul 2026 to 18:30 UTC-3, 14 Jul 2026: 22 h 30 min" in out
    assert not any("19:45" in x for x in out)


@pytest.mark.parametrize(
    "text",
    [
        "| Start date | 8 December 2026 |\n| Review due | 1 week after the Start date |",
        "Start date: 8 December 2026. The review is due 1 week after the Start date.",
        "The review is due 1 week after the start date of 8 December 2026.",
        'The "Review" date is 1 week after the "Start date". The recorded Start date is 2026-12-08.',
        "The review is due 1 week after the employee's Start date. The Start date is 8 December 2026.",
    ],
)
def test_a_base_given_by_its_label_elsewhere(text):
    assert "8 Dec 2026 + 1 week: 15 Dec 2026" in lines(text)


def test_a_label_given_two_dates_is_not_read():
    text = "Start date: 8 December 2026. Start date: 9 December 2026. Due 1 week after the Start date."
    assert lines(text) == []


def test_a_stamp_drops_a_closing_parenthesis_it_does_not_open():
    out = lines("(The call ended at 05:45 UTC+3) on 27 May 2026.")
    assert out == ["05:45 UTC+3, 27 May 2026 = 02:45 UTC, 27 May 2026"]
