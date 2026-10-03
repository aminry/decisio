# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Writes the triage demo's synthetic support tickets: data/train.json, data/heldout.json, data/stream.json.

Every ticket is assembled from the hand-written phrasings and slot lists below (no model writes them): a core request
in one of the 20 intents of taxonomy.json, filled with product details, and optionally a greeting, a sentence about
the customer's setup, a tone (polite, frustrated, urgent, terse), a short aside on a second topic that is clearly not
the point of the ticket, a sign-off and typos. A ticket has one to four sentences.

The training tickets use one set of core phrasings per intent and the held-out and stream tickets another, so a
correction learned from the training tickets has to carry over to new wording. No ticket text appears twice in or
across the splits. The output is a pure function of the seed; `--check` verifies the committed files instead of
writing them.

    python examples/demos/triage/make_tickets.py [--seed 20261003] [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
SEED = 20261003
COUNTS = {"train": 10, "heldout": 20, "stream": 4}  # tickets per intent in each split
POOL = {"train": "train", "heldout": "eval", "stream": "eval"}  # which core phrasings each split draws from
GENERATOR = "examples/demos/triage/make_tickets.py"

# Per intent: core phrasings for the training split, then core phrasings for the held-out and stream splits.
CORES = {
    "billing": {
        "train": [
            "I was charged {amount} on {day} and I don't understand what it is for.",
            "Can you send me a receipt for my last payment? I need it for my expenses.",
            "Why does my {month} invoice show {amount} when the {plan} plan is advertised as cheaper?",
            "My bank statement shows two charges from Notewell this month. Is one of them a mistake?",
            "How much will I pay when my yearly {plan} plan renews? The price on the website changed.",
            "I need my company name and VAT number on the invoices. Where can I add them?",
        ],
        "eval": [
            "There's a {amount} charge from Notewell on my card and I'm not sure which account it belongs to.",
            "Could you explain the line items on my latest invoice? One says 'proration adjustment'.",
            "Where do I find older invoices? The billing page only shows the last one.",
            "Does the {amount} include tax? My receipt doesn't say.",
            "I got an email saying my price goes up next month. What will I be paying exactly?",
            "The renewal was charged in dollars but I live in {country}. Why not in my own currency?",
            "My accountant needs every receipt from last year, can you send them all in one go?",
            "I see a pending charge of {amount} that I didn't expect. What is it?",
        ],
    },
    "refund": {
        "train": [
            "I cancelled on {day} but was still charged {amount}. Please give me my money back.",
            "I upgraded to {plan} by accident yesterday. Can you refund it?",
            "My yearly plan renewed without a reminder email and I don't want it. I'd like a refund.",
            "I was charged twice for the same month. Please return the duplicate payment.",
            "The app hasn't worked for me all week, so I'd like this month refunded.",
            "My {person} bought Plus on my card without asking. Can that be reversed?",
        ],
        "eval": [
            "Please refund the {amount} renewal, I stopped using Notewell months ago.",
            "I paid for the Family plan but nobody in my family ever joined. Can I get that money back?",
            "You charged me {amount} after the free trial even though I cancelled during it. I want it back.",
            "I'm asking for a refund of this year's subscription, I forgot it was still running.",
            "Charged for {plan} again after I downgraded. Please send the money back to my {payment}.",
            "Can I get a partial refund for the months I have left on the annual plan?",
            "I want my money back. Handwriting was the only reason I subscribed and it doesn't work on my {device}.",
            "Your invoice shows I paid {amount} twice on {day}. Refund one of them please.",
        ],
    },
    "payment_failed": {
        "train": [
            "My card keeps getting declined when I try to renew.",
            "I tried to upgrade to {plan} but the payment page says 'transaction failed'.",
            "PayPal says the payment went through but Notewell says my renewal failed.",
            "I got an email that my payment didn't go through and my account will be downgraded.",
            "I updated my card details but the charge still fails.",
            "Checkout gives me an error at the payment step every time.",
        ],
        "eval": [
            "My {payment} payment was rejected even though there's plenty of money in the account.",
            "Why can't I pay for {plan}? It spins and then says 'payment could not be processed'.",
            "The renewal failed and now my notes are read-only. My card is valid until 2029.",
            "Google Play shows my subscription as 'on hold' because the payment failed.",
            "I keep getting 'card not supported' when I enter my {payment} card.",
            "My bank says they never saw the charge, but the app says my payment failed.",
            "I can't complete the purchase, it says my billing address doesn't match.",
            "Every attempt to pay ends with error 402 and nothing is taken from my card.",
        ],
    },
    "plan_change": {
        "train": [
            "I'd like to move from monthly to yearly billing on {plan}.",
            "How do I upgrade from Plus to Family so my {person} can join?",
            "Please downgrade me to the Free plan at the end of this period.",
            "If I switch to Family now, what happens to the rest of my Plus month?",
            "I want to change my plan to Plus. Where is the option in the iPhone app?",
            "Can I go back to monthly instead of yearly? I don't use it enough.",
        ],
        "eval": [
            "We don't need Family anymore, it's just me now. Can I switch to Plus?",
            "What's the difference between Plus and Family, and how do I change between them?",
            "I want to upgrade to {plan} but keep my current billing date.",
            "Please switch my subscription from yearly to monthly when it renews.",
            "Can I upgrade for just one month to get more storage, then go back to Free?",
            "I'm on Free and hit the device limit. How do I move to Plus?",
            "I'm moving from Family to Plus because my kids have their own accounts now.",
            "Can you change my plan from Plus monthly to Plus yearly today?",
        ],
    },
    "promo_code": {
        "train": [
            "The code {code} says it's invalid at checkout.",
            "I'm a student, where do I get the student discount?",
            "I used {code} but was charged the full {amount}.",
            "Your email promised 30% off the yearly plan but I don't see the discount anywhere.",
            "Does the Black Friday offer work for people who already subscribe?",
            "I entered a coupon on my phone and it didn't apply.",
        ],
        "eval": [
            "The discount from the newsletter disappeared when I clicked pay.",
            "Can I still use {code}? It says expired but the email said it was good until the end of the month.",
            "I verified my student email but the price didn't change.",
            "My friend referred me but I never got the free month you advertised.",
            "I was supposed to get the first year at half price, but my invoice shows {amount}.",
            "Where do I type in a promo code on the Android app? I can't find the field.",
            "Is there a discount for teachers or non-profits?",
            "The {code} code worked last year, why not now?",
        ],
    },
    "cancel_subscription": {
        "train": [
            "Please cancel my subscription. I don't want to be charged again.",
            "How do I turn off auto-renew? I want to stay on Free.",
            "I want to stop paying for {plan}, but keep my notes.",
            "Cancel my Notewell membership at the end of this month please.",
            "I subscribed through the App Store, how do I cancel?",
            "Stop my plan from renewing in {month}.",
        ],
        "eval": [
            "I don't need Plus anymore. Please make sure it doesn't renew.",
            "Where is the cancel button? I've looked through every setting.",
            "Please end my Family subscription, we're moving everyone to Free.",
            "I tried to unsubscribe but the page just reloads.",
            "I want to cancel before the trial ends so I don't get billed.",
            "Can you cancel the subscription for me? I'm away and only have my phone.",
            "Please stop charging me, I want to cancel. The account itself can stay.",
            "My Plus renews tomorrow and I don't want it to. How do I cancel tonight?",
        ],
    },
    "delete_account": {
        "train": [
            "Please delete my account and all of my data.",
            "I want my Notewell account removed permanently, including notes and attachments.",
            "Under GDPR I am asking you to erase all personal data you hold about me.",
            "How do I close my account for good? I won't be coming back.",
            "My late father had a Notewell account. How can we have it deleted?",
            "Delete my profile. I made it by mistake with the wrong email.",
        ],
        "eval": [
            "Please wipe my account and everything in it, I'm done with this app.",
            "I want to remove my account completely, not just stop paying.",
            "How long after I delete my account is my data actually gone from your servers?",
            "Close and erase my account, I've already cancelled the subscription.",
            "I made two accounts by mistake. Please delete the one with {email}.",
            "Where's the option to delete my account in the app? I only see sign out.",
            "I'd like my account and all backups of it permanently destroyed.",
            "Please remove my account. I don't want any of my notes kept.",
        ],
    },
    "login": {
        "train": [
            "I can't sign in. It says my email or password is wrong but I'm sure they're right.",
            "Sign in with Apple takes me back to the login screen every time.",
            "Every time I log in on the web app it logs me out again after a few seconds.",
            "I can log in on my phone but not on my {device}.",
            "The login page just spins forever in {browser}.",
            "It says no account exists with my email, but I've been paying for two years.",
        ],
        "eval": [
            "Can't get into my account since this morning, the sign in button does nothing.",
            "Google sign-in says 'something went wrong' and won't let me in.",
            "I'm stuck on the login screen on my {device}, it keeps saying session expired.",
            "I think I made my account with a different email and now I can't sign in.",
            "I was logged out after the update and now my login doesn't work.",
            "The desktop app won't accept my password but the website does.",
            "I can't log in on {browser}, it worked fine yesterday.",
            "Signing in loops back to the start page on the Windows app.",
        ],
    },
    "two_factor": {
        "train": [
            "I'm not getting the verification code by text.",
            "I got a new phone and my authenticator app no longer has the Notewell code.",
            "Where do I find my backup codes? I lost access to my phone.",
            "The six-digit code is always rejected as invalid.",
            "How do I turn on two-step verification?",
            "I want to switch my 2FA from SMS to an authenticator app.",
        ],
        "eval": [
            "The code from Google Authenticator never works, it says expired right away.",
            "I changed my phone number and now I can't receive the login code.",
            "My security key isn't recognised when I try to sign in.",
            "I'm asked for a verification code every time I open the app. Can it remember my device?",
            "The SMS code arrives ten minutes later, after it has already expired.",
            "I lost my phone, so I can't get past the code screen. How do I turn off two-step verification?",
            "Can I have the two-factor codes sent to my email instead of my phone?",
            "I'm travelling abroad and the text messages with codes don't come through.",
        ],
    },
    "account_locked": {
        "train": [
            "My account says it's been locked after too many attempts.",
            "I got an email saying my account was suspended. Why?",
            "It says 'account disabled, contact support'. What did I do?",
            "I'm locked out and the message says try again in 24 hours. I need my notes now.",
            "My account was frozen for 'unusual activity'. Can you unlock it?",
            "Notewell flagged my account and I can't open anything. Please restore access.",
        ],
        "eval": [
            "The app says my account is temporarily blocked. How long does that last?",
            "After typing the wrong password a few times I'm now locked out for good?",
            "My account has been deactivated without any warning.",
            "I get 'this account is locked' even with the right password.",
            "Please unlock my account, I need my work notes for a meeting at {time}.",
            "Why was my account suspended? I only use it for school.",
            "There's an 'account locked' message on every device. I didn't change anything.",
            "It says my account is under review and restricted. Please lift it.",
        ],
    },
    "password_reset": {
        "train": [
            "I forgot my password and the reset email never arrives.",
            "The password reset link says it has expired.",
            "How do I change my password? I can't find it in settings.",
            "I reset my password but the new one isn't accepted.",
            "The 'forgot password' button does nothing when I tap it.",
            "Reset emails go to my old work address, which I can't access anymore.",
        ],
        "eval": [
            "I'm locked out because I can't remember my password. Can you send me a new reset link?",
            "Clicking the link in the reset email opens a blank page.",
            "I need to set a new password but the email never comes, not even in spam.",
            "Every reset link I get says 'invalid token'.",
            "I want to update my password to something stronger, where is that?",
            "I forgot my password and the reset page says my email isn't recognised.",
            "The reset form says my new password doesn't meet the rules but doesn't say which.",
            "The reset link worked but now the app asks for my old password again.",
        ],
    },
    "security_concern": {
        "train": [
            "I got an email that someone signed in from another country. That wasn't me.",
            "Somebody changed my account email and I can't get in anymore.",
            "There are notes in my account I didn't write. Has it been hacked?",
            "I received an email asking for my Notewell password. Is it real?",
            "Someone else is using a seat on my Family plan and I didn't invite them.",
            "I think my password was in a data leak. What should I do?",
        ],
        "eval": [
            "I just got a login alert from a device I don't own.",
            "My notes were deleted overnight and I didn't do it, someone has my account.",
            "Is this text message saying my Notewell account is suspended a scam?",
            "There's a session from {country} in my device list that isn't mine.",
            "Someone changed my password and recovery phone, I believe my account was taken over.",
            "I clicked a link in a suspicious email that looked like it came from you. Am I at risk?",
            "My shared notebooks have a stranger as a collaborator, I never added them.",
            "An unknown device signed in to my account last night, please help me secure it.",
        ],
    },
    "bug": {
        "train": [
            "Search doesn't find notes that I know contain the word.",
            "Checkboxes in my lists uncheck themselves after I close the note.",
            "The web clipper saves an empty note instead of the page.",
            "Reminders go off an hour late since the clocks changed.",
            "When I paste a table it loses all its formatting.",
            "Images in my notes show as broken grey boxes in {app}.",
        ],
        "eval": [
            "Tags I add disappear when I reopen the note.",
            "The editor jumps to the top every time I type a bullet point.",
            "Sorting by date modified puts old notes first. That looks wrong.",
            "Handwriting on the iPad turns into random letters when I convert it to text.",
            "The word count at the bottom is clearly wrong, it says 0 words.",
            "Templates put the wrong date in the title.",
            "In dark mode some of my text is black on black and unreadable.",
            "Attachments over 10 MB show 'upload error' even though Plus allows 200 MB.",
        ],
    },
    "crash": {
        "train": [
            "The app crashes every time I open it on my {device}.",
            "Notewell closes itself when I try to attach a photo.",
            "Since version {version} the Mac app won't start at all.",
            "The Android app freezes and then shuts down when I scroll a long note.",
            "The Windows app crashes as soon as I open the {notebook} notebook.",
            "Every time I tap 'share' the app quits.",
        ],
        "eval": [
            "The app force closes on launch after today's update.",
            "The iPad app goes to a white screen and then closes when I open a PDF.",
            "Notewell keeps crashing on my {device} when I record audio.",
            "The desktop app hangs completely and I have to force quit it.",
            "It quits every time I try to print a note.",
            "Opening the {notebook} notebook freezes the whole app and nothing responds.",
            "The app crashed in the middle of typing and I lost what I wrote.",
            "I can't open Notewell at all, it shows the logo and then disappears.",
        ],
    },
    "performance": {
        "train": [
            "The app has become very slow, it takes ages to open a note.",
            "Typing lags behind by a second or two in long notes.",
            "Notewell drains my phone battery even when I'm not using it.",
            "Search takes more than 30 seconds with about {count} notes.",
            "The web app is sluggish in {browser} and scrolling stutters.",
            "The desktop app uses 4 GB of memory and my laptop fans go crazy.",
        ],
        "eval": [
            "Everything is slow since the update, switching notebooks takes 10 seconds.",
            "Loading my notes on the {device} takes forever on wifi that's fine for everything else.",
            "The editor stalls for a moment every few words I type, it's very laggy.",
            "Startup now takes almost a minute on my {device}.",
            "Why is the app using so much CPU in the background?",
            "Opening a note with a few images takes a really long time.",
            "The website is painfully slow during the day and fine at night.",
            "My phone gets hot and slow whenever Notewell is open.",
        ],
    },
    "sync": {
        "train": [
            "Notes I write on my phone don't show up on my laptop.",
            "I edited a note on two devices and now there are two conflicting copies.",
            "Sync has been stuck on 'syncing' for two days.",
            "A notebook I deleted on the web keeps coming back on my iPad.",
            "Changes I made this morning are missing on my {device}.",
            "My {notebook} notebook is empty on the desktop but full on my phone.",
        ],
        "eval": [
            "Last sync says three days ago and it won't update.",
            "The note on my phone is an old version, my edits from the computer aren't there.",
            "I see 'sync error 1031' in the Mac app.",
            "I added notes offline on the plane and they never uploaded.",
            "Some notes show on the web app but not on my Android phone.",
            "Every time it syncs, a few notes get duplicated.",
            "My laptop and phone show different versions of the same checklist.",
            "Nothing I write on the {device} reaches my other devices anymore.",
        ],
    },
    "data_export": {
        "train": [
            "How can I export all my notes at once?",
            "I need a backup of everything, including attachments.",
            "Can I download my notes as PDF or Markdown?",
            "Before I leave I want a copy of all my notebooks. How?",
            "Is there a way to export one notebook to a file?",
            "I'd like a copy of all the data you hold about me.",
        ],
        "eval": [
            "What's the best way to get my {count} notes out of Notewell and into another app?",
            "Can I schedule automatic backups to my computer?",
            "I want to download every attachment I've ever uploaded.",
            "Does the export include tags and creation dates?",
            "I need my notes as plain text files for my archive.",
            "How do I save the whole {notebook} notebook to my hard drive?",
            "Can you send me an archive of my account? The export button times out with so many notes.",
            "Please tell me how to export my notes before my subscription ends.",
        ],
    },
    "feature_request": {
        "train": [
            "It would be great to have a dark mode on the web app.",
            "Please add a way to lock individual notes with a PIN.",
            "Could you support Markdown tables?",
            "I'd love to be able to set recurring reminders.",
            "Any plans for an Apple Watch app?",
            "Please let us pick a colour for each notebook.",
        ],
        "eval": [
            "Could you add a way to schedule notes to be sent by email?",
            "Would it be possible to have a calendar view of notes?",
            "I wish I could undo more than one step in the editor.",
            "Please make nested notebooks possible, more than one level deep.",
            "A word count goal for each note would really help my writing.",
            "Suggestion: let me pin notes to the top of a notebook.",
            "Is there any chance of a Linux desktop app?",
            "Could you add read-only links that don't need an account to view?",
        ],
    },
    "integration": {
        "train": [
            "How do I connect Notewell to {tool}?",
            "My Google Calendar events stopped appearing in Notewell.",
            "The Slack integration hasn't posted anything since yesterday.",
            "Where do I get an API key for my own scripts?",
            "The Zapier trigger 'new note' doesn't fire anymore.",
            "Saving emails to Notewell from Outlook stopped working.",
        ],
        "eval": [
            "I connected {tool} but my notes don't appear there.",
            "Our webhook gets a 401 error from your API since this week.",
            "Reminders from Notewell aren't showing up in my Outlook calendar anymore.",
            "How do I disconnect Google Drive from my account?",
            "The Dropbox attachment link opens a permissions error.",
            "How do I set up the Microsoft Teams integration?",
            "The API returns 'rate limit exceeded' after just a few calls.",
            "Notes I send from {tool} land in the wrong notebook.",
        ],
    },
    "sharing": {
        "train": [
            "How do I share a notebook with my {person}?",
            "My coworker says she can't edit the note I shared with her.",
            "How do I add my {person} to our Family plan?",
            "I want to remove someone's access to a shared notebook.",
            "The invite link I sent says it has expired.",
            "Can people I share with see my other notebooks?",
        ],
        "eval": [
            "My {person} accepted the Family invite but still has the Free plan.",
            "How do I give someone view-only access instead of edit?",
            "The shared notebook disappeared from my {person}'s account.",
            "I own a shared notebook but it says I don't have permission.",
            "How do I leave a shared notebook someone added me to?",
            "The Family plan says all seats are taken, but there are only three of us.",
            "Can I transfer ownership of a shared notebook to my colleague?",
            "Invites to my notebook never reach the other person's inbox.",
        ],
    },
}

# Short asides on a second topic, worded so that the ticket's main request stays clear.
ASIDES = {
    "feature_request": [
        "Also, a dark mode for the web app would be lovely.",
        "Unrelated, but a Linux app would be nice someday.",
    ],
    "billing": [
        "Also, could you resend my last receipt when you get a chance?",
        "Side question, no rush: does my plan price include VAT?",
    ],
    "performance": [
        "The app has also been a bit slow lately, but that's minor.",
        "PS the web app feels sluggish too, not a big deal.",
    ],
    "sharing": ["Also, no rush, but how do I share a notebook with my {person}?"],
    "data_export": ["Also, just curious, is there a way to export notes as Markdown?"],
    "integration": ["By the way, is there a Slack integration?"],
    "two_factor": ["Separately, how do I turn on two-step verification later?"],
    "promo_code": ["Also, do you still have a student discount?"],
    "sync": ["Separately, one note seems to sync slowly, but that's not why I'm writing."],
}

CONTEXT = [
    "I'm on the {plan} plan.",
    "I mostly use {app}.",
    "I use it on my {device}.",
    "I'm on version {version}.",
    "I've been a customer since {year}.",
    "I have about {count} notes.",
    "My account email is {email}.",
    "We're on the Family plan.",
]

GREETINGS = ["Hi!", "Hello!", "Hi there!", "Hey!", "Good morning!", "Hello Notewell team!", "Hi team!", "Hello."]
CLOSERS = ["Thanks!", "Thank you.", "Thanks, {name}", "Cheers, {name}", "Regards, {name}", "- {name}"]
TONES = {
    "neutral": [],
    "polite": ["Any help would be great.", "Sorry if this has been asked before.", "I appreciate any help."],
    "frustrated": [
        "This is really frustrating.",
        "I've already written about this twice.",
        "Honestly not happy about this.",
        "Please sort this out, it's been days.",
    ],
    "urgent": ["I need this fixed today.", "Urgent please.", "I have a deadline tomorrow."],
    "terse": [],
}
TONE_WEIGHTS = {"neutral": 0.35, "polite": 0.2, "frustrated": 0.2, "urgent": 0.1, "terse": 0.15}

SLOTS = {
    "amount": ["$4.99", "$49.99", "$8.99", "$89.99", "€5.49", "£4.49", "$59.88", "€54.99"],
    "day": ["Monday", "Tuesday", "Friday", "the 3rd", "the 15th", "the 28th", "the 1st"],
    "month": ["January", "March", "April", "June", "September", "October", "December"],
    "plan": ["Plus", "Family"],
    "payment": ["Visa", "Mastercard", "Amex", "debit card", "PayPal"],
    "code": ["STUDENT50", "WELCOME20", "SPRING25", "BF2026", "NOTES30", "FRIEND1M"],
    "person": ["wife", "husband", "son", "daughter", "sister", "colleague", "partner"],
    "device": ["iPhone", "iPad", "Android phone", "Pixel 8", "Galaxy S23", "MacBook", "Windows laptop", "Chromebook"],
    "browser": ["Chrome", "Safari", "Firefox", "Edge"],
    "app": ["the Mac app", "the Windows app", "the iPhone app", "the Android app", "the web app", "the iPad app"],
    "version": ["8.4.1", "8.5.0", "8.5.2", "9.0.1", "9.1.0"],
    "notebook": ["Recipes", "Work", "Thesis", "Q3 planning", "Journal", "Travel", "Meeting notes"],
    "count": ["300", "1,200", "2,000", "5,000", "12,000"],
    "tool": ["Google Calendar", "Outlook", "Slack", "Zapier", "Google Drive", "Dropbox"],
    "country": ["Germany", "Brazil", "India", "Canada", "Japan", "Nigeria", "Spain"],
    "email": ["m.rossi@example.com", "jt1987@example.org", "priya.n@example.net", "chen.w@example.com"],
    "time": ["9am", "10:30", "noon", "3pm"],
    "name": ["Maria", "Tom", "Priya", "Ahmed", "Chen", "Sofia", "Jake", "Amara", "Lukas", "Yuki"],
    "year": ["2019", "2021", "2022", "2024"],
}


def load_taxonomy(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or HERE / "taxonomy.json").read_text())


def rng_for(seed: int, *parts: str) -> random.Random:
    """A generator of its own for each split and intent, from a hash (stable across Python versions and processes)."""
    digest = hashlib.sha256(":".join([str(seed), *parts]).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def draw_slots(rng: random.Random) -> dict:
    """One value per slot for a ticket, so its parts agree (the plan or the device named twice is the same one). Every
    slot is drawn every time, so the random stream does not depend on which slots a phrasing uses."""
    return {k: rng.choice(v) for k, v in SLOTS.items()}


def n_sentences(text: str) -> int:
    return max(1, len(re.findall(r"[.!?](?:\s|$)", text.strip())))


def add_typos(text: str, rng: random.Random, n: int) -> str:
    """`n` keyboard slips in words of four letters or more: two letters swapped, one dropped, or one doubled."""
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(re.sub(r"[^A-Za-z]", "", w)) >= 4 and w.isascii()]
    for i in rng.sample(candidates, min(n, len(candidates))):
        w = words[i]
        letters = [j for j, c in enumerate(w) if c.isalpha()]
        j = rng.choice(letters[1:-1] or letters)
        op = rng.choice(("swap", "drop", "double"))
        if op == "swap" and j + 1 < len(w) and w[j + 1].isalpha():
            w = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
        elif op == "drop":
            w = w[:j] + w[j + 1 :]
        else:
            w = w[:j] + w[j] + w[j:]
        words[i] = w
    return " ".join(words)


def casual(text: str) -> str:
    """Apostrophes dropped and a lowercase 'i', as people type on a phone."""
    text = text.replace("'", "").replace("’", "")
    return re.sub(r"\bI\b", "i", text)


def ticket(intent: str, core: str, rng: random.Random, aside_intents: list[str]) -> dict:
    """One ticket: the core request with the optional parts around it, at most four sentences. Typos fall in what the
    customer wrote, never in the greeting or the name they sign with."""
    tone = rng.choices(list(TONE_WEIGHTS), weights=list(TONE_WEIGHTS.values()))[0]
    if intent == "feature_request" and tone in ("urgent", "frustrated"):
        tone = "polite"  # nobody needs a suggestion "fixed today"
    slots = draw_slots(rng)
    body = core.format_map(slots)
    budget = 4 - n_sentences(body)
    context = aside = tone_line = closer = None
    secondary = None
    optional = ["context", "aside", "tone", "closer"]
    rng.shuffle(optional)
    for part in optional:
        if budget <= 0:
            break
        if part == "context" and rng.random() < 0.4:
            # a context that repeats a detail the core already names reads oddly ("my email is X ... delete X")
            options = [c for c in CONTEXT if not set(re.findall(r"{(\w+)}", c)) & set(re.findall(r"{(\w+)}", core))]
            context = rng.choice(options).format_map(slots)
        elif part == "aside" and rng.random() < 0.22:
            secondary = rng.choice([k for k in aside_intents if k != intent])
            aside = rng.choice(ASIDES[secondary]).format_map(slots)
        elif part == "tone" and TONES[tone]:
            tone_line = rng.choice(TONES[tone])
        elif part == "closer" and tone != "terse" and rng.random() < 0.45:
            closer = rng.choice(CLOSERS).format_map(slots)
        else:
            continue
        budget -= 1  # every optional part is one sentence (test_demo_triage checks the total)
    text = " ".join(p for p in (context, body, aside, tone_line) if p)
    typos = 0
    if rng.random() < 0.3:
        typos = rng.choice((1, 1, 2))
        text = add_typos(text, rng, typos)
    if rng.random() < 0.1:
        text = casual(text)
    if tone == "terse":
        text = text[0].lower() + text[1:]
        if rng.random() < 0.5:
            text = text.rstrip(".")
    else:
        if rng.random() < 0.5:
            text = rng.choice(GREETINGS) + " " + text
        if closer:
            text += " " + closer
    return {"text": text, "intent": intent, "secondary": secondary, "tone": tone, "typos": typos}


def normalised(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def generate(seed: int = SEED, taxonomy: dict | None = None, counts: dict | None = None) -> dict[str, list[dict]]:
    """{split: [ticket]}: every split shuffled, ids in order, no ticket text twice anywhere (after normalising case
    and punctuation). Splits are drawn in a fixed order (train, heldout, stream) so the result is a function of the
    seed alone."""
    taxonomy = taxonomy or load_taxonomy()
    counts = counts or COUNTS
    keys = [i["key"] for i in taxonomy["intents"]]
    if set(keys) != set(CORES):
        raise ValueError("taxonomy.json and the phrasings name different intents")
    aside_intents = sorted(ASIDES)
    seen: set[str] = set()
    out = {}
    for split, per_intent in counts.items():
        pool = POOL[split]
        rows = []
        for intent in keys:
            rng = rng_for(seed, split, intent)
            cores = list(enumerate(CORES[intent][pool]))
            rng.shuffle(cores)
            made, attempts = 0, 0
            while made < per_intent:
                attempts += 1
                if attempts > 100 * per_intent:
                    raise RuntimeError(f"cannot write {per_intent} distinct {split} tickets for {intent}")
                idx, core = cores[(attempts - 1) % len(cores)]
                t = ticket(intent, core, rng, aside_intents)
                key = normalised(t["text"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append({**t, "core": f"{intent}/{pool}{idx + 1:02d}"})
                made += 1
        rng_for(seed, split, "order").shuffle(rows)
        out[split] = [{"id": f"{split}-{n + 1:04d}", **r} for n, r in enumerate(rows)]
    return out


def split_file(split: str, tickets: list[dict], seed: int) -> dict:
    per = {}
    for t in tickets:
        per[t["intent"]] = per.get(t["intent"], 0) + 1
    return {
        "split": split,
        "seed": seed,
        "generator": GENERATOR,
        "synthetic": "templated tickets written for this demo; no real customer wrote them",
        "phrasings": POOL[split],
        "n": len(tickets),
        "per_intent": per,
        "tickets": tickets,
    }


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def load_split(split: str, data: str | Path | None = None) -> list[dict]:
    return json.loads((Path(data or DATA) / f"{split}.json").read_text())["tickets"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--check", action="store_true", help="compare with the files in --out instead of writing them")
    a = ap.parse_args()
    splits = generate(a.seed)
    out = Path(a.out)
    stale = []
    for split, tickets in splits.items():
        path = out / f"{split}.json"
        text = dumps(split_file(split, tickets, a.seed))
        if a.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path.name)
        else:
            out.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"{path}: {len(tickets)} tickets")
    if stale:
        sys.exit(f"not what seed {a.seed} writes: {', '.join(stale)}")


if __name__ == "__main__":
    main()
