<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Decisio

**A decision server: typed questions about a piece of text, a probability for every option, one forward pass per question.**

Decisio serves a frozen open model, Qwen3.6-35B-A3B, on vLLM and reads its answer to a closed question from the option letters' logits.
No text is generated.
A request carries a state (a ticket, a document, a record) and one or more questions of three types: yes/no, a choice among named options, or a score on an ordered scale.
The answer to each question is a distribution over its options, and the request format is TypeSafe's System One wire format, so existing clients work unchanged.

What ships here is code, recipes and evaluation records.
No trained weights: every measured answer comes from the official checkpoint.
`EVAL_CARD.md` states every number, what it was measured on, and what was fitted on what.

## What it does

| | |
| --- | --- |
| Question types | yes/no (`noul`), choice among up to 255 options with descriptions, score on an ordered scale |
| Many questions per request | the state is prefilled once and shared through vLLM's prefix cache |
| Probabilities | a distribution per question, with one fitted global temperature; per-task calibration from labelled examples |
| Task registration | `POST /v1/tasks`: from labelled examples of one recurring question the server fits per-task calibration and, for option lists of 10 or more, a linear head on the model's hidden state |
| Abstention | an opt-in per-task threshold on a declared "can't tell" option (`POST /v1/abstention/tasks`) |
| Image input | photos in the state, served by a second engine on the same card (`--image-model`) |
| Rendering rules | enumerated option keys are hidden, snake_case labels are shown as words, ties resolve by key so the answer never depends on the order keys arrive in |
| Determinism | one request at a time, on the same card: bit-identical answers when the server's request history is the same, and repeats agreed to 1.2e-7 in probability; identical requests sent inside one batch are not the same forward pass and differed by up to 0.59 in a label probability (`EVAL_CARD.md` section 4); the serving gates in `tests/gpu` check the one-at-a-time case |
| Context | up to 32,768 tokens of state |

Latency and cost on one RTX PRO 6000, from `EVAL_CARD.md`: about 28 ms server time for one question, 2.7 to 5.4 ms per question when many questions share a state, and about $0.001 to $0.012 per 1,000 decisions at $1.50 per card-hour.

## Quickstart

Requirements: Linux, one NVIDIA card with 96 GB (measured on an RTX PRO 6000 Blackwell) and a driver that supports CUDA 13.0 (the runtime `uv.lock` pins), Python 3.12, [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/aminry/decisio
cd decisio
uv sync --extra serve --frozen
uv run python -m decisio.serve.vllm_engine --model Qwen/Qwen3.6-35B-A3B-FP8 --model-class hidden-readout
```

The first start downloads the checkpoint (about 36 GB) and warms the engine.
The decisio plugin registers its model classes with vLLM through an entry point, so vLLM 0.30.0 loads them without any patch.
`patches/` holds an optional latency patch series, off by default; see `patches/README.md`.

Ask a question:

```
curl http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Hi, since this morning none of our 40 staff can log in to the dashboard. We get \"session expired\" right after entering the password. Payroll is due today.",
  "questions": {
    "urgent":   {"type": "noul",   "instructions": "Does this ticket need a response within the hour?"},
    "category": {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Invoices, payments, refunds", "access": "Login, passwords, permissions", "bug": "The product behaves wrongly", "other": null}},
    "impact":   {"type": "score",  "instructions": "Rate the business impact using only the reported facts.",
                 "criteria": ["No function impaired", "One user impaired, with a workaround", "Many users blocked from a core function", "Data loss or legal exposure"]}
  }
}'
```

Response, abridged:

```
{"answers": {
  "urgent":   {"type": "noul",   "noul": 0.96},
  "category": {"type": "choice", "choice": "access", "probabilities": {"billing": 0.00, "access": 0.99, "bug": 0.00, "other": 0.00}},
  "impact":   {"type": "score",  "score": 2.0, "probabilities": {"0": 0.00, "1": 0.00, "2": 0.99, "3": 0.00}}}}
```

Register a recurring question from labelled examples, so the server fits a per-task correction:

```
curl http://127.0.0.1:8000/v1/tasks -H 'Content-Type: application/json' -d '{
  "id": "ticket-routing",
  "examples": [{"request": {"state": "...", "questions": {"category": {...}}}, "answer": "access"}, ...]
}'
```

`docs/handoffs/tasks.md` is the full specification of task registration, the head, and the tie-break rule.

## Without a GPU

The unit tests and the CPU stand-in engine run on any machine:

```
uv sync --extra dev --frozen
uv run pytest
```

`--backend hf` serves a small Hugging Face model on the CPU for development of the routes; it is not the measured system.

A laptop version of the same model, packaged for [Ollama](https://ollama.com/aminroudaki/decisio)'s decision route, exists as `ollama pull aminroudaki/decisio`.
Ollama builds its own prompt and applies no calibration, so its numbers are the ones on that page, not the ones here.

## How it performs

All numbers are on the served default described in `EVAL_CARD.md`, measured privately with the public harnesses; nothing here is a leaderboard score.

| Measure | Result |
| --- | --- |
| JevBench, 231 published items, accuracy by tier | easy 1.000, standard 0.972, hard 0.694 |
| Decision Index 0.2.1, four benchmarks | BANKING77 macro-F1 0.731, CLINC150+OOS 0.814, GPQA Diamond 0.454, MMLU-Pro 0.609 |
| Intent heads from 10 labelled examples per intent | BANKING77 0.847, CLINC150 0.893 |
| Image input, ImajevBench v2.0-lite | 0.791 on the 230 answerable items |
| Latency, one question | about 28 ms server time |

Where it stands: on the public harnesses this frozen model is the most accurate open one-pass system we know of on the hard tier, and it is behind TypeSafe's Jev on hard knowledge questions by several points and on intent taxonomies without labelled examples.
With 10 labelled examples per intent, registered heads reach accuracy 0.847 on BANKING77 (150 held-out items) and 0.893 on CLINC150 (100 held-out items), means of three draws.
They use labelled examples, so those figures are not comparable with zero-shot systems.
Calibration trails too: hard-tier ECE on JevBench is 0.059, above the 0.05 we aimed for.
`EVAL_CARD.md` has the full tables, the calibration figures, and the three disclosures about what was fitted on what.

## Repository layout

```
src/decisio/serve/      the server: /v1/systemone, tasks, abstention, temperature, image route, conformance
src/decisio/readout/    the letters readout, calibration and the intent head (reference implementations)
src/decisio/vllm_plugin/  the vLLM entry point and the two model classes
src/decisio/bench/      benchmark scoring (JevBench v1.5 open-set reading, Decision Index reports)
benchmarks/             scripts that run the public harnesses against a server
patches/                optional vLLM patch series, off by default
tests/unit/  tests/gpu/ CPU tests run on every pull request; GPU tests run nightly (pytest -m gpu)
runs/                   evaluation records: a manifest, per-item results and hashes per run
docs/                   design notes and the task-registration specification
EVAL_CARD.md            what was measured, on what, and what was fitted on what
```

## Contributing

See `CONTRIBUTING.md`.
Pull requests need a DCO sign-off (`git commit -s`), pass the CPU tests and lint, and are reviewed by a maintainer before merge.
Security issues go through GitHub's private vulnerability reporting, as described in `SECURITY.md`.

## Licence

Apache-2.0 (`LICENSE`, `NOTICE`).
The model weights are Alibaba's Qwen3.6-35B-A3B under Apache-2.0 and are downloaded, not redistributed.
`THIRD-PARTY.md` lists everything else this project builds on.
