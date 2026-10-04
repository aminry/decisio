<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Support-ticket triage: teaching the server one recurring question

A live feed of support tickets for Notewell, a fictional note-taking app sold as a subscription.
Every ticket is one `/v1/systemone` call with the same question, "Which support queue should handle this ticket?", over 20 queues, each with a one-line description (`taxonomy.json`).
The clip shows the same tickets twice, side by side: on the left as the frozen model reads the question plainly, on the right after the question was registered with 200 labelled tickets through `POST /v1/tasks` (task registration, `docs/tasks.md`).
The header carries the change in accuracy measured on 400 held-out tickets that are neither in the clip nor among the registered examples.

The queues are confusable on purpose, as real ones are: billing, refund, payment failed and promo code; cancelling the subscription and deleting the account; login, two-step verification, a locked account, a password reset and a hacked account; a bug, a crash, slowness and sync.
A generalist reading the queue names gets a fair share of these wrong; the registered examples show the server what this desk means by each queue.

## The tickets are synthetic

No customer wrote these tickets and no model wrote them either.
`make_tickets.py` assembles each one from hand-written phrasings and slot lists in the script: a core request in one intent, with product details (plan, device, app version, amounts), and optionally a greeting, a sentence about the customer's setup, a tone (polite, frustrated, urgent, terse), a short aside on a second topic that is clearly not the point, a sign-off and typos.
A ticket has one to four sentences.
The output is a pure function of the seed (20261003); `python make_tickets.py --check` verifies the committed files.

| split | tickets | per intent | what it is for |
| --- | --- | --- | --- |
| `data/train.json` | 200 | 10 | the labelled examples registered with `POST /v1/tasks` |
| `data/heldout.json` | 400 | 20 | accuracy before and after registration |
| `data/stream.json` | 80 | 4 | the tickets shown in the clip |

The training tickets are written from one set of core phrasings per intent and the held-out and stream tickets from another, so what registration learns has to carry over to new wording.
No ticket text appears twice in or across the splits.
Each ticket records its intent, the phrasing it came from (`core`), its tone, the intent of its aside (`secondary`, if any) and how many typos it has.

## Run it

Start a System One server (a decisio server on a GPU, or the CPU stand-in for a check that the steps work), then, from `examples/demos/tools`:

```sh
python measure_triage.py --url http://127.0.0.1:8000 --label Decisio --player player.json --pace-ms 1500
python render_triage.py --compare runs/<run>/plain.jsonl.gz runs/<run>/taught.jsonl.gz \
    --summary runs/<run>/summary.json --out runs/<run>/media --stills 6
```

`measure_triage.py` routes every held-out and stream ticket (the plain arm), registers the training tickets as the task `notewell-triage` and times it, routes the same tickets again (the taught arm), and deletes the task, so the server is left as it was found.
It refuses to run if a task of that id is already registered or if a task already applies to the question.
The held-out tickets run unpaced; stream tickets keep at least `--pace-ms` between their starts, so a viewer can read each one.
`--limit-heldout N` routes N held-out tickets (N/20 per intent), for a quick check on a slow server.

The run record (`runs/<date>_demos-triage`) has `summary.md` and `summary.json` (held-out accuracy before and after with 95% bootstrap intervals and the paired difference, macro-F1, per-intent accuracy, the most frequent confusions, the registration time and what the server kept, decisions per second and latency), `heldout_rows.json`, `registration.json`, `decisions.jsonl.gz` (every request and full answer), the two stream trajectories and `manifest.json` with `files.json`.

`render_triage.py` draws only from the trajectories: one feed (`render_triage.py runs/<run>/taught.jsonl.gz`) or both arms side by side, at the recorded speed, with the model, card and median latency in the caption strip.

A number from the CPU stand-in or from `tools/mock_systemone.py` says that the steps work, not how Decisio routes tickets; the mock accepts a registration and fits nothing.
