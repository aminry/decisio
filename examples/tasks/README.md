<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Walk-through: teach the server one recurring question

A support desk routes every ticket to one of 12 queues.
This walk-through registers that question from labelled tickets, asks it, measures what registration changed on held-out tickets, and shows how to list, export, reload and delete the task.
The guide that explains each step is `docs/tasks.md`.

It runs on any machine against the CPU stand-in (`--backend hf`), a small Qwen model on the CPU, so nothing here needs a GPU.
The stand-in is not the measured system: its numbers below say only that the steps work.
The numbers in `docs/tasks.md` come from the served model on a GPU (`EVAL_CARD.md`, `runs/`).

## Files

| File | What it is |
| --- | --- |
| `question.json` | The recurring question: its instructions and its 12 queues (keys and descriptions) |
| `train.csv` | 120 labelled tickets, 10 per queue (`text,label`) |
| `heldout.csv` | 48 other tickets, 4 per queue, in phrasings the training tickets never use |
| `examples.json` | `train.csv` as the body of `POST /v1/tasks`, for curl |
| `make_tickets.py` | Writes the three data files; every ticket is synthetic, written for this example |
| `register.py` | Builds the registration from a CSV or JSONL of text and label, posts it, prints the fit |
| `ask.py` | Asks the question about tickets and shows the header and the distribution |
| `evaluate.py` | Paired accuracy on the held-out tickets before and after registration, with a bootstrap interval |

## Run it

From the repository root, with the package installed (`uv sync --extra dev --frozen`), start the stand-in in one terminal:

```bash
uv run python -m decisio.serve.vllm_engine --backend hf --model Qwen/Qwen3-0.6B-Base
```

It is ready when `curl -s http://127.0.0.1:8000/health` answers (the first start downloads the 0.6B model).
On a GPU the same steps run against the served model: `uv run python -m decisio.serve.vllm_engine --model Qwen/Qwen3.6-35B-A3B-FP8`.

In another terminal, register the question from the training tickets:

```bash
uv run python examples/tasks/register.py --id ticket-routing --data examples/tasks/train.csv
```

```text
120 examples, 12 labels, 10 to 10 per label
task 'ticket-routing': 120 examples, 12 options, at least 10 per option, registered in 62 s
  calibration: applied (cross-validated gain); cross-validated log loss 2.37021 -> 2.2986, accuracy 0.1667 -> 0.2083
  head: applied (cross-validated gain); penalty 0.0001; cross-validated log loss by penalty {"none": 2.37021, ..., "0.0001": 0.22708}
```

(Stand-in output, abridged.)
The same registration with curl, from the prepared body:

```bash
curl -s http://127.0.0.1:8000/v1/tasks -H 'Content-Type: application/json' -d @examples/tasks/examples.json
```

Ask it about new tickets:

```bash
uv run python examples/tasks/ask.py "My card keeps getting declined when I renew."
```

```text
My card keeps getting declined when I renew.
  choice payment_failed; x-decisio-tasks: ticket-routing; route text, 224.7 ms
  payment_failed 0.998, account_closure 0.001, billing 0.000
```

`x-decisio-tasks: ticket-routing` says the question matched the registered task, whose kept corrections answered it; `none` means it matched no task and was served exactly as without registration.

Measure the change on the held-out tickets (this removes the task, asks every held-out ticket, registers again, asks again):

```bash
uv run python examples/tasks/evaluate.py --id ticket-routing --train examples/tasks/train.csv --heldout examples/tasks/heldout.csv
```

```text
held-out tickets: 48; matched task: ticket-routing
accuracy before 0.083, after 0.833; difference +75.0 points, 95% interval [+60.4, +87.5]
changed answers: 45 of 48 (37 fixed, 1 broken)
```

The stand-in's 0.6B model barely reads this question without help, so the change is large; on the served model the plain readout starts far higher (`docs/tasks.md`, "What you get").

## List, export, reload, delete

A registration lives in the server's memory; a restart forgets it unless it is exported and loaded back.

```bash
curl -s http://127.0.0.1:8000/v1/tasks                        # what is registered, with each fit's summary
curl -s 'http://127.0.0.1:8000/v1/tasks?full=1' > tasks.json  # the export, head parameters included
curl -s -X DELETE http://127.0.0.1:8000/v1/tasks/ticket-routing
curl -s http://127.0.0.1:8000/v1/tasks/import -H 'Content-Type: application/json' -d @tasks.json
```

Or load the export when the server starts:

```bash
uv run python -m decisio.serve.vllm_engine --backend hf --model Qwen/Qwen3-0.6B-Base --tasks-file tasks.json
```

A task is valid only for the model and rendering it was fitted under: an export from the stand-in does not load into the served model, and the server says so.

## Your own question

Put your labelled texts in a CSV with `text` and `label` columns (or a JSONL of `{"text", "label"}`), write your question to a JSON file in the shape of `question.json`, and pass it with `--question` to each script.
Every label must be one of the question's option keys, and every example must be the same question with the same options in the same order.

`tests/unit/test_tasks_walkthrough.py` runs these steps against the stand-in: a smoke on a 3-queue version of the question in the fast tier, and the full 12 queues with the `--tasks-file` restart in the slow tier (`pytest -m slow`).
