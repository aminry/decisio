# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Writes the walk-through's synthetic ticket-routing data: train.csv (10 tickets per queue), heldout.csv (4 per queue).

Every ticket is generated from the phrasings below, which were written for this example; none comes from a benchmark.
The held-out tickets use phrasings the training tickets never use, so a correction has to generalise to new wording.
The output is deterministic (seed 0); `--check` verifies the committed files instead of writing them.

    python examples/tasks/make_tickets.py [--check]
"""

import argparse
import csv
import io
import random
from pathlib import Path

HERE = Path(__file__).parent

# per queue: six phrasings for training, then two held out
PHRASINGS = {
    "billing": [
        "I was charged {amount} this month but my plan is supposed to cost less.",
        "Can you send me a corrected invoice for {month}? The VAT number is missing.",
        "Why does my invoice show two seats when we only have one user?",
        "The annual plan price changed without notice. What am I paying for now?",
        "Please explain the extra line item on our {month} invoice.",
        "Our finance team needs the invoices for the last three months in one PDF.",
        "There is a charge of {amount} I don't recognise on the statement.",
        "How is the price calculated when we add users halfway through the month?",
    ],
    "refund": [
        "I cancelled within the trial period and still got charged. I want my {amount} back.",
        "We paid twice for the same order. Please return one of the payments.",
        "The service was down for two days, so I'd like a refund for this month.",
        "I bought the wrong plan by mistake. Can I get the money back?",
        "Please refund the {amount} for the add-on we never used.",
        "Requesting a full refund, the product did not do what the sales page said.",
        "Can the unused months of my yearly subscription be paid back to my card?",
        "I'd like my money returned for the order I cancelled on {day}.",
    ],
    "payment_failed": [
        "My card keeps getting declined when I try to renew.",
        "The payment page says transaction failed but my bank shows nothing.",
        "Our bank transfer bounced and now the account says overdue.",
        "I updated my card but the renewal still failed on {day}.",
        "Why was my payment rejected? The card works everywhere else.",
        "Checkout errors out at the payment step with code 402.",
        "Autopay did not go through and we got a warning email.",
        "The SEPA debit was returned unpaid, what do I need to do?",
    ],
    "login": [
        "I can't sign in, it says my session expired right after I log in.",
        "The two-factor code never arrives on my phone.",
        "My account got locked after a few attempts and I need access today.",
        "Single sign-on sends me back to the login page in a loop.",
        "Login works in one browser but not in {browser}.",
        "I get 'invalid credentials' even though I'm sure the email is right.",
        "The authenticator app codes are rejected since I changed phones.",
        "Our whole team is stuck on the sign-in screen since this morning.",
    ],
    "password_reset": [
        "I forgot my password and the reset email never shows up.",
        "The password reset link says it has expired.",
        "How do I change my password? I can't find the option.",
        "Reset link opens a blank page in {browser}.",
        "I need to reset the password for a colleague who left.",
        "I reset my password twice and the new one still doesn't work.",
        "Where can I set a new password if I only remember my username?",
        "The 'forgot password' button does nothing when I click it.",
    ],
    "account_closure": [
        "Please close my account and delete everything you store about me.",
        "We are moving to another tool. How do we cancel and remove the workspace?",
        "I want to delete my profile permanently.",
        "Cancel our subscription at the end of this term and shut the account down.",
        "How do I remove my account? I no longer use the service.",
        "Our company is closing, please terminate the account for {company}.",
        "I'd like you to erase my account under data protection rules.",
        "Stop the plan and deactivate all our users, we're done.",
    ],
    "bug": [
        "The report page crashes with an error when I pick a date range.",
        "Saving a form deletes the text I typed in the notes field.",
        "Numbers in the dashboard are wrong: totals don't match the rows.",
        "Clicking export shows 'something went wrong' every time.",
        "Since the last update the calendar shows events on the wrong day.",
        "The mobile app closes as soon as I open a project.",
        "Uploading a CSV gives error 500 halfway through.",
        "Buttons overlap and I can't click submit on the settings page.",
    ],
    "feature_request": [
        "It would be great if we could schedule reports to be emailed weekly.",
        "Could you add a dark mode?",
        "We need custom fields on contacts, is that planned?",
        "Please let us set different permissions per project.",
        "Any chance of a Spanish version of the interface?",
        "I'd love a keyboard shortcut to create a new task.",
        "Suggestion: allow bulk editing of many records at once.",
        "Will you support recurring invoices in the future?",
    ],
    "performance": [
        "Everything is very slow today, pages take {seconds} seconds to load.",
        "Search times out when we filter by customer.",
        "The dashboard lags badly with more than a hundred projects.",
        "Uploads are crawling, a small file takes minutes.",
        "The app freezes for a few seconds every time I switch tabs.",
        "Reports that used to take seconds now take {seconds} seconds.",
        "Pages keep spinning and sometimes give a gateway timeout.",
        "Typing in the editor has a noticeable delay since {day}.",
    ],
    "data_export": [
        "How can I download all my data as a spreadsheet?",
        "I need a full export of our records before the audit.",
        "Is there a way to export contacts with their history?",
        "The export only gives me the first 1,000 rows, I need everything.",
        "Please send me a copy of all the data you hold for my account.",
        "Can I get our project files exported to CSV?",
        "Where do I find the backup download for the workspace?",
        "We want to move our data out, what formats can you export?",
    ],
    "integration": [
        "Our webhook stopped receiving events on {day}.",
        "Where do I generate a new API key?",
        "The connection to our accounting tool keeps disconnecting.",
        "API calls return 401 even with a fresh token.",
        "How do I connect the calendar sync with our team calendar?",
        "The workflow automation fails at the authorisation step.",
        "Rate limit errors on the API since this morning, what are the limits?",
        "Our CRM sync duplicates every contact it imports.",
    ],
    "device_delivery": [
        "The card reader I ordered on {day} still hasn't arrived.",
        "Tracking says delivered but there's no package here.",
        "The scanner arrived damaged, the box was crushed.",
        "Can you change the delivery address for my device order?",
        "I received the wrong model of the terminal.",
        "When will the hardware ship? The order shows processing for a week.",
        "The courier left a note but I can't find where to collect the device.",
        "Only one of the two printers we ordered was in the parcel.",
    ],
}
SLOTS = {
    "amount": ["$49", "$120", "EUR 30", "$15.99", "GBP 80"],
    "month": ["March", "July", "last month", "September"],
    "day": ["Monday", "the 3rd", "Friday", "last week"],
    "browser": ["Firefox", "Safari", "Edge", "Chrome"],
    "company": ["Acme Ltd", "Northwind", "Blue Harbor"],
    "seconds": ["20", "45", "30"],
}
OPENERS = ["", "Hello, ", "Hi team, ", "Good morning. ", "Hey, "]
CLOSERS = ["", " Thanks.", " Please help.", " Thank you in advance.", " This is urgent for us."]


def ticket(rng, phrasing):
    text = phrasing.format(**{k: rng.choice(v) for k, v in SLOTS.items()})
    opener = rng.choice(OPENERS)
    if opener.endswith(", ") and not text.startswith(("I ", "I'", "API", "SEPA")):
        text = text[0].lower() + text[1:]  # "Hi team, the report page ..."
    return (opener + text + rng.choice(CLOSERS)).strip()


def build(per_train=10, per_heldout=4, seed=0):
    rng = random.Random(seed)
    train, heldout = [], []
    for label, ph in PHRASINGS.items():
        for rows, pool, n in ((train, ph[:6], per_train), (heldout, ph[6:], per_heldout)):
            seen = set()
            while len(seen) < n:
                t = ticket(rng, pool[len(seen) % len(pool)])
                if t not in seen:
                    seen.add(t)
                    rows.append((t, label))
    rng.shuffle(train)
    rng.shuffle(heldout)
    return train, heldout


def as_csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["text", "label"])
    w.writerows(rows)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the committed files instead of writing them")
    a = ap.parse_args()
    train, heldout = build()
    for name, rows in (("train.csv", train), ("heldout.csv", heldout)):
        text = as_csv(rows)
        if a.check:
            if (HERE / name).read_text() != text:
                raise SystemExit(f"{name} differs from what make_tickets.py writes")
        else:
            (HERE / name).write_text(text)
    print(
        f"train {len(train)} tickets, held out {len(heldout)}, {len(PHRASINGS)} queues"
        + (" (checked)" if a.check else "")
    )


if __name__ == "__main__":
    main()
