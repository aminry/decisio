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
| Many questions per request | the state is prefilled once and shared through vLLM's prefix cache, then each question is scored in its own engine call, so its answer equals the same question sent alone; `--multi-question warm` scores them in one batch for bulk scoring |
| Probabilities | a distribution per question, with one fitted global temperature; per-task calibration from labelled examples |
| Task registration | `POST /v1/tasks`: from labelled examples of one recurring question the server fits per-task calibration and, for option lists of 10 or more, a linear head on the model's hidden state; guide in `docs/tasks.md` |
| Abstention | an opt-in per-task threshold on a declared "can't tell" option (`POST /v1/abstention/tasks`) |
| Image input | photos in the state, served by a second engine on the same card (`--image-model`) |
| Rendering rules | an option with a description is shown as its description alone, without its key (`--no-describe-options` shows `key: description`): on 272 steps of a browser-agent demo it cut wrong "done" picks from 63 to 12 of 132, with accuracy within noise on JevBench and the Decision Index; enumerated option keys are hidden, snake_case labels are shown as words, ties resolve by key so the answer never depends on the order keys arrive in |
| Determinism | a question returns the same probabilities every time on a running server, alone or among other questions: each question is scored in its own engine call (272 of 272 four-question requests equal to the question sent alone, 5 repeats each; 1,400 of 1,400 suite items bit-identical); across restarts in 4 of 5 starts measured (the fifth moved 2 items by up to 0.0012, no choice changed). With `--multi-question warm` the questions are scored in one batch, faster, but on this stack a batched answer depends on the batch: on the same 272 requests it moved between repeats on 31 and differed from the question sent alone on 50, by up to 0.27 (reported upstream, vllm-project/vllm#59764). `EVAL_CARD.md` section 4 and `runs/2026-10-02_multi-question-and-rendering/` have the measurements |
| Context | up to 32,768 tokens of state |

Latency on one RTX PRO 6000, server time, from `EVAL_CARD.md` (medians; a state the server has never seen, unless marked):

| State | 1 question | 4 questions, served default | 4 questions, `--multi-question warm` (batched) |
| --- | ---: | ---: | ---: |
| 300 tokens | 48.5 ms | 110.3 ms | 62.0 ms |
| 1,000 tokens | 51.3 ms | 112.3 ms | 78.4 ms |
| 3,000 tokens | 86.3 ms | 152.4 ms | 116.8 ms |
| from the prefix cache | 27.9 ms | | |

Each further question costs about 17 ms in the served default and 2.7 to 5.4 ms batched when many share a state; at $1.50 per card-hour that is about $0.012 per 1,000 single-question decisions, about $0.007 per 1,000 further questions, and $0.0012 to $0.0022 batched.

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

Response, abridged (the values the first GPU start returned, rounded, under the defaults of 2026-10-01: the questions batched and options shown as `key: description`; the current defaults return each question's single-question answer under the description rendering, to be re-measured):

```
{"answers": {
  "urgent":   {"type": "noul",   "noul": 0.90},
  "category": {"type": "choice", "choice": "access", "probabilities": {"billing": 0.03, "access": 0.79, "bug": 0.16, "other": 0.02}},
  "impact":   {"type": "score",  "score": 2.0, "probabilities": {"0": 0.00, "1": 0.01, "2": 0.96, "3": 0.02}}}}
```

Repeat the request and you get the same probabilities: each question is scored in its own engine call after the state is prefilled once, so its answer equals the question sent on its own. With `--multi-question warm` the three questions are scored in one batch, faster, and their probabilities vary with that batch: by about 0.12 on this example (the probability of `access` ranged from 0.79 to 0.91, `runs/2026-10-01_docker-first-gpu-start/repeat_variability/`), and when two options are close the choice can change too.

## Teach it your question in ten examples

The model is frozen, but the server can learn one recurring question from your own labelled examples: it fits a per-task calibration and, for 10 or more options, a small head on the model's hidden state, each kept only if cross-validation on your examples shows a gain.
With 10 labelled examples per intent, accuracy on held-out test items rose from 0.740 to 0.847 on BANKING77 (77 intents) and from 0.820 to 0.893 on CLINC150 (150 intents), against the same model unregistered.

```
curl http://127.0.0.1:8000/v1/tasks -H 'Content-Type: application/json' -d @examples/tasks/examples.json
```

`docs/tasks.md` walks through it and states what each number was measured on; `examples/tasks/` runs it on a laptop against the CPU stand-in.

## Run with Docker

```
uv build --wheel                   # the image installs this wheel
docker compose up --build          # needs the NVIDIA Container Toolkit and a 96 GB card
curl http://127.0.0.1:8000/health  # answers once the first start has fetched the checkpoint
```
The first start downloads the checkpoint (about 36 GB) into the `decisio-data` volume; it is never part of the image.
The container runs as a non-root user, compose publishes the port on 127.0.0.1 only, and `Dockerfile` and `compose.yaml` explain the rest.
Release images go to `ghcr.io/aminry/decisio`, with their digest in the release notes.
First GPU start (2026-10-01, one RTX PRO 6000 Blackwell, image built on the machine from the `Dockerfile`): healthy in 651 s including the checkpoint download and in 206 s from the cached volume; the example request below and the conformance gates C2 to C4 pass against the container (`runs/2026-10-01_docker-first-gpu-start`); the image published with v0.1.0 repeated the start from its digest (healthy in 223 s with the checkpoint cached, example and conformance pass, `pushed_image_0.1.0` in that run).

## Without a GPU

The unit tests and the CPU stand-in engine run on any machine:

```
uv sync --extra dev --frozen
uv run pytest
```

`--backend hf` serves a small Hugging Face model on the CPU for development of the routes; it is not the measured system.

On a Mac with Apple silicon, `--backend mlx` serves the text route with every feature (tasks, the intent head, the temperature, abstention) from an MLX conversion of the same checkpoint; the image route is not served by it:

```
uv sync --extra mlx
uv run python -m decisio.serve.vllm_engine --backend mlx --model mlx-community/Qwen3.6-35B-A3B-6bit
```

The 6-bit conversion is the Mac default and the 4-bit one the option for 32 GB machines (docs/design/mlx-backend.md).
Its prompts are the served ones byte for byte, and at 6 bits it passes the served default's gates against the FP8 records (suite accuracy +0.4 [-0.5, +1.2] points, the intent heads reproduced, 30.7 GB at 32k tokens); a decision takes about 0.5 s on an M5 Pro, about 0.1 s per question when 100 share a state (docs/design/mlx-backend.md, `runs/2026-10-02_mlx-backend`).

A laptop version of the same model, packaged for [Ollama](https://ollama.com/aminroudaki/decisio)'s decision route, exists as `ollama pull aminroudaki/decisio`.
Ollama builds its own prompt and applies no calibration, so its numbers are the ones on that page, not the ones here.

## How it performs

All numbers are on the served default described in `EVAL_CARD.md`, measured privately with the public harnesses; nothing here is a leaderboard score.

| Measure | Result |
| --- | --- |
| JevBench, 231 published items, accuracy by tier | easy 1.000, standard 0.972, hard 0.685 |
| Decision Index 0.2.1, four benchmarks | BANKING77 macro-F1 0.731, CLINC150+OOS 0.814, GPQA Diamond 0.490, MMLU-Pro 0.609 |
| Intent heads from 10 labelled examples per intent | BANKING77 0.847, CLINC150 0.893 |
| Image input, ImajevBench v2.0-lite | 0.791 on the 230 answerable items |
| Latency, one question | 48.5 ms server time on a new 300-token state, 27.9 ms on a state from the cache |

Where it stands: on the public harnesses this frozen model is behind TypeSafe's Jev on hard knowledge questions by several points and on intent taxonomies without labelled examples.
With 10 labelled examples per intent, registered heads reach accuracy 0.847 on BANKING77 (150 held-out items) and 0.893 on CLINC150 (100 held-out items), means of three draws.
They use labelled examples, so those figures are not comparable with zero-shot systems.
Calibration trails too: hard-tier ECE on JevBench is 0.069, above the 0.05 we aimed for.
`EVAL_CARD.md` has the full tables, the calibration figures, and the three disclosures about what was fitted on what.

## Repository layout

```
src/decisio/serve/      the server: /v1/systemone, tasks, abstention, temperature, image route, conformance
src/decisio/readout/    the letters readout, calibration and the intent head (reference implementations)
src/decisio/vllm_plugin/  the vLLM entry point and the two model classes
src/decisio/bench/      benchmark scoring (JevBench v1.5 open-set reading, Decision Index reports)
benchmarks/             scripts that run the public harnesses against a server
patches/                optional vLLM patch series, off by default
Dockerfile  compose.yaml  docker/   the server image (vLLM 0.30.0 release image plus the wheel)
tests/unit/  tests/gpu/ CPU tests run on every pull request; GPU tests run nightly (pytest -m gpu)
runs/                   evaluation records: a manifest, per-item results and hashes per run
docs/                   the task-registration guide and specification, design notes
examples/tasks/         a runnable task-registration walk-through (CPU stand-in)
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
