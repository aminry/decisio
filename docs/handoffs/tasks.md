<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Handoff: per-task corrections on `/v1/systemone`

How a deployment teaches the server about its own tasks, from labelled examples, and what the server does with them.
This is the reference; `docs/tasks.md` is the guide for a first-time user, and `examples/tasks/` a runnable walk-through.
Code: `decisio.serve.tasks`, `decisio.serve.abstention`, `decisio.serve.temperature`, `decisio.serve.hidden_engine`, the routes in `decisio.serve.systemone.add_routes`.

## What a task is

A task is one question's option list: the question type and its options (keys and descriptions) in the served order, plus the instructions for yes/no and score questions, whose options alone do not identify a task.
Any change to the list is a new task and is registered again.
A task is valid only for the model and the rendering it was fitted under (the store's fingerprint: served name, model directory, padding, `--hide-index-keys`, `--desnake-labels`); tasks fitted under another fingerprint are not applied.

## Registering

`POST /v1/tasks` with `{"id": ..., "examples": [{"request": <a /v1/systemone request with one question>, "answer": <key>}, ...]}`.
The server scores every example exactly as it would serve it and fits, with the reference implementations unchanged:

- **Calibration** (`decisio.readout.calibration`): one bias per option, kept only if 5-fold cross-validation on the examples shows a clear gain; otherwise the task stores it as off and serves exactly as before.
  At least 10 examples; about 20 recommended.
- **The intent head** (`decisio.readout.intent_head`): a linear correction from the hidden state at the answer position, for questions of at least 10 options with at least 5 examples of every option; the L2 penalty is cross-validated on the examples, and "no head" is one of the candidates.
  Where a head applies, calibration is not stacked on it.
  The head is bound to its option order and is refused with two-order averaging.

`GET /v1/tasks` lists the tasks (`?full=1` with the heads' arrays, the form `--tasks-file` and `POST /v1/tasks/import` load); `DELETE /v1/tasks/{id}` removes one.

## The head's two modes

The head needs the final-norm hidden state at the answer position, which vLLM's generate path does not return.
Two modes read it; the server's flags choose (`decisio.serve.vllm_engine.resolve_head_mode`).

| | Single engine (the default) | Second engine (`--head-engine`) |
| --- | --- | --- |
| How | The text engine runs decisio's hidden-readout class, which writes the hidden state into reserved logit columns; the server reads them in three requests, one at a time (docs/design/hidden-state-readout.md) | A second copy of the model in vLLM's pooling mode returns the hidden state for the same token rows |
| Server time per head question | about 127 ms, against 46 ms for the same question without a task (runs/2026-09-30_plugin-verification) | about 82 ms, the same as a plain question on that run (2026-09-29, another card and code state; not like for like) |
| Weights on the card | one copy; the text engine at 0.90 of the card's memory | two copies (33 GiB each); the two engines at 0.47 each |
| Image engine on the same card | yes: the two engines at 0.47 and 0.49 (measured) | no: refused at start-up |
| Other questions | unchanged, bit for bit (1,400 of 1,400 suite answers) | unchanged (the text engine is not touched) |
| Label arithmetic | the engine's own: logits rounded to bf16 | float64 from the pooled state |

Choose the second engine for a deployment with heavy intent traffic on a dedicated card, where 45 ms per head question matters more than the second weight copy and the image route.
Everything else should use the default.

Both modes give the same declared choice on every evaluation item of the stored intent readouts (BANKING77, 150 items; CLINC150, 100 items; `tests/data/intent_readouts`), each mode fitting its own head from its own reading of the same hidden states, with probabilities within 1e-2 (`tests/unit/test_head_modes.py`, M3).
Tasks carry no mode, so a task registered in one mode loads in the other.
The 82 ms was measured with the text-only view as the model directory (`--model-class view --head-engine`).
The second-engine mode's default pairing, with the registered text-only class (`--head-engine` alone), is unmeasured for latency; it is to be timed in the next session on a card.

## The global temperature

`/v1/systemone` serves a question's plain readout as `softmax(log p / T)` with T = 1.506 (`--temperature`; 1 switches it off, bit for bit), and choice questions at their own T = 1.370 (`--temperature-choice`; `--temperature-noul` and `--temperature-score` exist and are unset, so those types take the global T).
Both were fitted on the suite's plain readouts under the served prompt of 2026-10-03; under the earlier prompt the one T was 1.307.
It never changes the chosen option.
A registered task's calibration or head replaces it; the image route is not tempered; `/v1/answer` is the raw readout.
How T was fitted, and where it helps and hurts, is in EVAL_CARD.md: on the Decision Index it lowered ECE on MMLU-Pro, BANKING77 and GPQA Diamond and raised it on CLINC150+OOS (0.031 to 0.170, 151 options).

## Re-registration after the 2026-10-03 prompt change

The served prompt changed on 2026-10-03 (`--prompt-tail cygnet`; yes/no questions as a two-option letter choice with the sides named, `--noul-rendering letters-keys`).
The prompt format enters the task fingerprint and the yes/no rendering enters a yes/no question's task key, so tasks fitted under the earlier prompt load but are not applied, with a start-up warning.
Register them again from the same examples, or serve them unchanged with `--prompt-tail decisio --noul-rendering words`.
`--mode packed` reads the earlier prompt only and now needs `--prompt-tail decisio`.

## Abstention

`POST /v1/abstention/tasks` registers a threshold on an abstain option's probability (`decisio.serve.abstention`): the customer's own option (for example "out of scope") or one the server appends (`--abstain-option`, answered through imajev's `unknown_probability` and `abstained` fields).
The threshold is fitted from labelled examples that include unanswerable ones, with leave-one-out cross-validation and an acceptance rule that keeps the plain behaviour when the examples do not justify it; it decides on the distribution after the temperature or the task's correction.

## Exact ties

The declared choice is the most probable option; among options with exactly equal probabilities, the key that sorts first in Unicode code-point order (`systemone.top_index`).
JSON objects are unordered and some clients serialise with sorted keys, so the choice must not depend on the order of the keys on the wire (`tests/unit/test_ties.py`).

## Checking what the server did

With `--debug-readout`, a request may send `x-decisio-debug: readout` (the readout before any task and the path taken) or `hidden` (also the hidden state); registration returns the examples' readouts the same way.
The response headers `x-decisio-route` and `x-decisio-tasks` name the route and the tasks applied, and the body's `decisio_debug` field carries the debug readout.
Requests sent with the earlier `x-rlcd-debug` and `x-rlcd-route` headers are still honoured, and task records in the earlier `rlcd-*/1` formats, or fitted under the earlier default served name `rlcd-qwen3.6-35b-a3b-letters`, still load and apply (`decisio.names`).
