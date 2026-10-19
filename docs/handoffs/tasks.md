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
- **The intent head** (`decisio.readout.intent_head`): a linear correction from the hidden state at the answer position, for questions of at least 10 options with at least 5 examples of every option; the L2 penalty is cross-validated on the examples, and "no head" is one of the candidates. A kept head carries a temperature T (`head.temperature`, with `head.temperature_cv`: out-of-fold log loss and ECE without and with it) and is served as `softmax(logits / T)`; T is fitted on the head's out-of-fold logits (five folds, each refitted at the chosen penalty), searched on log T within 0.2 to 20, and never on held-out examples. A record without a temperature serves as T = 1, so tasks stored before this load and answer as they did; the fit digest a certificate names includes T, so a certificate made on a head without one is not carried over to a head with one.
  Where a head applies, calibration is not stacked on it.
  The head is bound to its option order and is refused with two-order averaging.

`GET /v1/tasks` lists the tasks (`?full=1` with the heads' arrays, the form `--tasks-file` and `POST /v1/tasks/import` load); `DELETE /v1/tasks/{id}` removes one.

## Certification and the monitor

A task registered with a `certify` block also stores a **certificate** (`decisio-certificate/1`, `decisio.readout.certify`) and the **baseline of an answer-mix monitor** (`monitor`, `decisio.serve.monitor`); both are in the task's export and load back unchanged.

- **The held-out split.** The certificate is made on examples the fit has not seen: a separate `certify.examples` list, or `holdout` (default 0.5) of `examples`, chosen by a rule that does not depend on their order (examples with the same input stay together; groups are ordered by the sha256 of the task id and the input; the first groups certify). The calibration and the head are fitted on the rest only. The stored `split` names the method, the sizes and the 16-hex-digit digests of the fit examples' inputs; re-certification refuses any example sharing an input with them.
- **The procedure** is Learn-then-Test with a fixed-sequence arm and a Bonferroni arm, each given half of `delta`, and Clopper-Pearson upper bounds (the shad0w package's `auto` controller, written out in `decisio.readout.certify`). Confidence is the top probability of the answer as served (after the task's correction and `--noul-commit`). Score questions are not certified. The fewest held-out examples that can certify anything is `certify.minimum_examples(alpha, delta)`: 59 at alpha 0.05 and delta 0.1.
- **The serving-stack fingerprint** (`decisio.serve.stack`): the decisio version, base, repository and revision, the engine and its version, the card or device, quantisation, the attention backend, cache units, a hash of the token rows built for a fixed probe question (tokenizer, chat template, system turn and rendering), the temperatures, the serving flags (orders, multi-question, pad policy, rendering, abstention, `--noul-commit`, head engine) and a digest of the task's own fit. Its sha256 is `certificate.stack_sha256`; a server whose fingerprint differs certifies nothing for the task until `POST /v1/tasks/{id}/certify`.
- **The answer-mix monitor.** Per certified task the server keeps the index of the option each of the last 500 answers chose (never the question, the state or the option names) and compares the window's mix with the certification examples' every 100 answers once full, by total variation distance, against a line fixed at certification (the 99.9th percentile of the same distance between two independent resamples of the examples, 2,000 draws). The alert (`status.state: traffic_changed`, and `decisio_task_state` in `/metrics`) says the traffic changed; it does not suspend the certificate or change an answer. The counters are in memory and start again with the server.

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
Both were fitted on the suite's plain readouts under the served prompt of 2026-10-03; under the compact layout the one T was 1.307.
It never changes the chosen option.
A registered task's calibration or head replaces it; the image route is not tempered; `/v1/answer` is the raw readout.
How T was fitted, and where it helps and hurts, is in EVAL_CARD.md: on the Decision Index it lowered ECE on MMLU-Pro, BANKING77 and GPQA Diamond and raised it on CLINC150+OOS (0.031 to 0.170, 151 options).

## Re-registration after the 2026-10-03 prompt change

The served prompt changed on 2026-10-03 to the spaced layout (`--prompt-tail spaced`: blank lines around the lettered options and one closing line asking for the chosen option's letter alone; yes/no questions as a two-option letter choice with the sides named, `--noul-rendering letters-keys`).
The layout enters the task fingerprint and the yes/no rendering enters a yes/no question's task key, so tasks fitted under the earlier compact layout load but are not applied, with a start-up warning.
Register them again from the same examples, or serve them unchanged with `--prompt-tail compact --noul-rendering words`.
`--mode packed` reads the compact layout only and needs `--prompt-tail compact`.

## Committed yes/no answers

`--noul-commit` (off by default) is an output transform on yes/no answers (`systemone.commit_noul`): P(yes) strictly between 0.20 and 0.80 is reported as 0.80 if it is above 0.5 and as 0.20 otherwise, so an answer at exactly 0.5 stays no, as the served tie-break has it; outside the band nothing changes.
It runs last, after the temperature or a task's correction; answers decided by an abstention task keep their own probabilities; choice and score questions are untouched.
The served answer never changes, its calibration does.
Measured on 1,474 yes/no items at the served default (`runs/2026-10-03_noul-commit/`):

| | Accuracy | Committed accuracy | Inside 0.20 to 0.80 | Log loss | ECE, tie-robust |
| --- | ---: | ---: | ---: | ---: | ---: |
| served | 83.9% | 67.2% | 25.7% | 0.3977 | 0.044 |
| `--noul-commit` | 83.9% | 83.9% | 0.0% | 0.4145 | 0.078 |

Under the compact layout (yes/no in words, T 1.307) the same items gave 81.4% accuracy with 34.3% inside the band, and the transform moved log loss from 0.4124 to 0.4379 and ECE from 0.033 to 0.072.

Committed accuracy counts an answer right only if it is at or beyond the band's edge on the right side.
On JevBench v1.5, which scores a yes/no answer inside the band as no answer, the yes/no competence rose from 13.66 to 79.57 and I_open A from 49.39 to 71.36 (under the compact layout: -10.9 to 59.15, and 42.69 to 66.04).
Use it only where a scorer or a downstream rule needs a committed probability; anything that reads the probability as a probability is better served without it.

## Abstention

`POST /v1/abstention/tasks` registers a threshold on an abstain option's probability (`decisio.serve.abstention`): the customer's own option (for example "out of scope") or one the server appends (`--abstain-option`, answered through imajev's `unknown_probability` and `abstained` fields).
A task matches by `match`: `option_set` (the question's exact option list), `option_keys` (the question's set of keys alone, for items that share their labels and not their texts: the keys A to D and `idk` over item-specific answers; the whole key set, not the abstain key alone, because the abstain option's probability moves with the number of candidates) or `imajev_extension`. Examples that do not share the match identity are refused with 422.
The threshold is fitted from labelled examples that include unanswerable ones, with leave-one-out cross-validation and an acceptance rule that keeps the plain behaviour when the examples do not justify it; it decides on the distribution after the temperature or the task's correction.

## Exact ties

The declared choice is the most probable option; among options with exactly equal probabilities, the key that sorts first in Unicode code-point order (`systemone.top_index`).
JSON objects are unordered and some clients serialise with sorted keys, so the choice must not depend on the order of the keys on the wire (`tests/unit/test_ties.py`).

## Checking what the server did

With `--debug-readout`, a request may send `x-decisio-debug: readout` (the readout before any task and the path taken) or `hidden` (also the hidden state); registration returns the examples' readouts the same way.
The response headers `x-decisio-route` and `x-decisio-tasks` name the route and the tasks applied, and the body's `decisio_debug` field carries the debug readout.
Task records in the earlier `rlcd-*/1` formats, or fitted under the earlier default served name `rlcd-qwen3.6-35b-a3b-letters`, still load and apply (`decisio.names`).
