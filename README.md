<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Decisio

[![PyPI](https://img.shields.io/pypi/v/decisio)](https://pypi.org/project/decisio/)
[![Licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue)](LICENSE)
[![CI](https://github.com/aminry/decisio/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/aminry/decisio/actions/workflows/ci.yml)

Decisio is a decision server: send it a piece of text and typed questions about it, and it returns a probability for every option of every question.
It serves a frozen open model on vLLM and reads each answer from the option letters' logits, one forward pass per question, with no text generated.
It speaks TypeSafe's System One wire format, so existing clients work unchanged, and it ships code, recipes and evaluation records, not trained weights.

## Contents

- [What it does](#what-it-does)
- [Watch it play](#watch-it-play)
- [Install and run](#install-and-run)
- [Ask a question](#ask-a-question)
- [Choosing a base](#choosing-a-base)
- [Benchmarks](#benchmarks)
- [Teach it your question](#teach-it-your-question)
- [Options](#options)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Contributing](#contributing)
- [Licence](#licence)

## What it does

| | |
| --- | --- |
| Question types | yes/no (`noul`), a choice among up to 255 named options, a score on an ordered scale |
| Many questions per request | the state is prefilled once and shared; each question is scored on its own, so its answer equals the question sent alone |
| Probabilities | a distribution per question, tempered by fitted temperatures, with per-task calibration from labelled examples |
| Task registration | `POST /v1/tasks` learns one recurring question from labelled examples: a calibration and, for long option lists, an intent head |
| Abstention | an opt-in per-task threshold on a declared "can't tell" option (`POST /v1/abstention/tasks`) |
| Image input | photos in the state, served by a second engine on the same card (`--image-model`) |

## Watch it play

Four demos call a decisio server for every move (`examples/demos/`), and each clip is rendered from a recorded run at the speed it happened.

| | |
| --- | --- |
| ![Pong](docs/demos/media/pong_decisio_cygnet.gif) | ![Driving](docs/demos/media/driving_decisio_s1-1.gif) |
| Pong ([MP4](docs/demos/media/pong_decisio_cygnet.mp4)) | A driving simulator, in real time ([MP4](docs/demos/media/driving_decisio_s1-1.mp4)) |
| ![Browser agent](docs/demos/media/browser_decisio_travel.gif) | ![Triage](docs/demos/media/triage_plain_vs_taught.gif) |
| A browser agent's travel task ([MP4](docs/demos/media/browser_decisio_travel.mp4)) | Support-ticket triage before and after registering labelled tickets ([MP4](docs/demos/media/triage_plain_vs_taught.mp4)) |

[`docs/demos/README.md`](docs/demos/README.md) has every player, every clip and the result tables.

## Install and run

| Machine | Path | Memory needed |
| --- | --- | --- |
| Linux with one NVIDIA card | [GPU server](#gpu-server) | a 96 GB card, as measured; each base's weights are in [Choosing a base](#choosing-a-base) |
| Linux with one NVIDIA card and Docker | [Docker](#docker) | as the GPU server |
| Mac with Apple silicon | [Mac with MLX](#mac-with-mlx) | 30.7 GB at 32,761 tokens of state at 6 bits, 22.0 GB at 4 bits |
| Mac, Linux or Windows with Ollama | [Ollama](#ollama) | the size of the tag, on the [model page](https://ollama.com/aminroudaki/decisio) |
| Any machine, for development | [CPU stand-in](#cpu-stand-in-for-development) | a small Hugging Face model on the CPU |

[`docs/running.md`](docs/running.md) has the details of every path.

### GPU server

Linux, one NVIDIA card, a driver that supports CUDA 13.0, Python 3.12 and [uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/aminry/decisio
cd decisio
uv sync --extra serve --frozen
uv run python -m decisio.serve.vllm_engine --base qwen3.6-35b-a3b
```

The first start downloads the checkpoint and warms the engine; the server then listens on `http://127.0.0.1:8000`.

### Docker

```
uv build --wheel                   # the image installs this wheel
docker compose up --build          # needs the NVIDIA Container Toolkit
curl http://127.0.0.1:8000/health  # answers once the first start has fetched the checkpoint
```

Release images are published to `ghcr.io/aminry/decisio`, with their digest in the release notes.

### Mac with MLX

```
uv sync --extra mlx
uv run python -m decisio.serve.vllm_engine --backend mlx --model mlx-community/Qwen3.6-35B-A3B-6bit
```

It serves the text route with every feature (tasks, the intent head, the temperatures, abstention), not the image route.

### Ollama

```
ollama pull aminroudaki/decisio
```

Requires Ollama 0.35.1 or later.
Ollama builds its own prompt and applies no calibration, so its numbers are the ones on its model page, not the ones here.

### CPU stand-in for development

```
uv sync --extra dev --frozen
uv run pytest
uv run python -m decisio.serve.vllm_engine --backend hf --model Qwen/Qwen3-0.6B-Base
```

The stand-in serves the routes from a small model on the CPU for development; it is not the measured system.

## Ask a question

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

Response, abridged and rounded, as recorded under the served defaults of 2026-10-02 (the same in 30 of 30 repeats on one server, `runs/2026-10-02_readme-example/`):

```
{"answers": {
  "urgent":   {"type": "noul",   "noul": 0.90},
  "category": {"type": "choice", "choice": "access", "probabilities": {"billing": 0.01, "access": 0.97, "bug": 0.01, "other": 0.00}},
  "impact":   {"type": "score",  "score": 2.0, "probabilities": {"0": 0.00, "1": 0.02, "2": 0.94, "3": 0.04}}}}
```

The routes ([`docs/api.md`](docs/api.md) has every field):
- `POST /v1/systemone`: ask questions about a state, as above.
- `POST /v1/tasks`: register a recurring question from labelled examples; `GET /v1/tasks` lists them.
- `POST /v1/tasks/import`: load tasks exported with `GET /v1/tasks?full=1`, without fitting again.
- `POST /v1/abstention/tasks`: register an abstention threshold for a task; `GET` lists them.
- `GET /health`: the engine, the base, its checkpoint and revision, the temperatures and the prompt being served.

## Choosing a base

Two base models are served behind the same routes, wire format and features, one per server, chosen with `--base`:

```
uv run python -m decisio.serve.vllm_engine --base qwen3.6-35b-a3b   # the default
uv run python -m decisio.serve.vllm_engine --base gemma-4-12b
```

`--base` brings its checkpoint at a pinned revision and every setting below; a flag given explicitly overrides the base's value.
Each base's settings were measured as one configuration.

| Setting | `qwen3.6-35b-a3b` (default) | `gemma-4-12b` |
| --- | --- | --- |
| Checkpoint | `Qwen/Qwen3.6-35B-A3B-FP8` at `95a723d0`, 33.3 GiB in memory | `google/gemma-4-12B-it` at `707f0a3b`, bf16, 22.8 GiB in memory |
| Temperatures | 1.370 for choice questions, 1.506 for yes/no and score | 3.592 for every question type |
| Prompt | no system turn, read after an "Answer:" prefill, one token per option letter | a system turn, read at the chat template's own answer position, every single-token form of each letter summed |
| Yes/no | a two-option letter choice with its sides named | a two-option letter choice, each side shown as its description |
| Padding | the state padded to the 1,056-token block | none |
| Several questions in one request | the same probabilities as each question sent alone, bit for bit | the same choice as each question sent alone, probabilities within 0.035 |

In plain words:
- Choose Gemma 4 12B for committed yes/no answers, scores and intent routing on taxonomies like CLINC150.
- Choose Qwen3.6-35B-A3B for knowledge questions, long states seen for the first time, and answers that repeat bit for bit.
- On JevBench's published items they are level.
- Gemma 4 12B does not fit a 32 GB card on vLLM; its path to smaller machines is an MLX build, in preparation.

What drives the choice, measured on one RTX PRO 6000 Blackwell in one session, each base with its own defaults, paired over the same items (95% bootstrap intervals; `runs/2026-10-04_gemma-base/`):

| Measure | Gemma 4 12B | Qwen3.6-35B-A3B | Gemma minus Qwen |
| --- | ---: | ---: | --- |
| JevBench v1.5 open-set reading, I_open (equal types) | 64.4 | 49.4 | +15.0 [+7.0, +23.6] |
| JevBench v1.5, yes/no / score | 49.0 / 63.7 | 13.7 / 57.2 | |
| Yes/no answers between 0.20 and 0.80 (74 items) | 15% | 39% | |
| JevBench, published items, correct | | | 0.0 points [-4.3, +4.3] |
| Decision Index accuracy, CLINC150+OOS | | | +4.5 [+3.6, +5.5] |
| Decision Index accuracy, BANKING77 | | | -1.4 [-2.7, -0.1] |
| Decision Index accuracy, GPQA Diamond | | | -13.6 [-21.7, -5.6] |
| Decision Index accuracy, MMLU-Pro | | | -6.4 [-7.2, -5.5] |
| 1,400-item suite, accuracy | 0.735 | 0.770 | -3.5 [-5.5, -1.6] |
| One question on a new 3,000-token state, server time | 293.5 ms | 91.1 ms | |

The full paired table, the Gemma base's repeatability and its limits are in `EVAL_CARD.md` section 6.

## Benchmarks

Both bases with their own defaults, on one RTX PRO 6000 Blackwell in one session (`runs/2026-10-04_gemma-base/`), except image input (`runs/2026-09-27_image-input/`).
Measured privately with the public harnesses (the Decision Index kit 0.2.1); none is a board score.

| Measure | Qwen3.6-35B-A3B (default) | Gemma 4 12B |
| --- | ---: | ---: |
| JevBench, 231 published items, accuracy: easy / standard / hard | 1.000 / 0.972 / 0.739 | 1.000 / 0.972 / 0.739 |
| Decision Index BANKING77, macro-F1 | 0.746 | 0.729 |
| Decision Index CLINC150+OOS, macro-F1 | 0.822 | 0.871 |
| Decision Index GPQA Diamond (196 scored), accuracy | 0.510 | 0.378 |
| Decision Index MMLU-Pro, accuracy | 0.613 | 0.549 |
| Intent heads from 10 labelled examples per intent, BANKING77 / CLINC150 (six draws) | 0.840 / 0.912 | 0.832 / 0.908 |
| Image input, ImajevBench v2.0-lite, the 230 answerable items | 0.791 | not measured |
| One question on a new 300-token state, server time | 48.4 ms | 39.1 ms |
| One question on a 1,000-token state from the prefix cache, server time | 20.9 ms | 26.7 ms |

Where it stands: on the public harnesses the Qwen base is behind TypeSafe's Jev on hard knowledge questions by several points and on intent taxonomies without labelled examples.
The intent heads use labelled examples, so their figures are not comparable with zero-shot systems.
Calibration on JevBench, as ECE on the standard and hard tiers: 0.121 and 0.043 on the Qwen base, 0.033 and 0.085 on the Gemma base.
`EVAL_CARD.md` has the full tables, the calibration figures and the disclosures of what was fitted on what (sections 4 and 6.4).

## Teach it your question

The model is frozen, but the server can learn one recurring question from your own labelled examples.
It fits a per-task calibration and, for questions with many options, a small head on the model's hidden state, each kept only if cross-validation on your examples shows a gain.
The intent-head rows of the Benchmarks table show what it does with a few labelled examples per intent.
Tasks registered under the earlier compact layout are not applied by the current default and need registering again.

```
curl http://127.0.0.1:8000/v1/tasks -H 'Content-Type: application/json' -d @examples/tasks/examples.json
```

[`docs/tasks.md`](docs/tasks.md) walks through it and states what each number was measured on; `examples/tasks/` runs it on a laptop against the CPU stand-in.

## Options

The served defaults need no flags; these change behaviour ([`docs/cli.md`](docs/cli.md) has every flag and its default per base):
- `--pad-policy row`: pads a single-question request so its whole row ends on the block boundary, faster on a new state at a small cost in repeatability.
- `--multi-question warm`: scores a request's questions in one batch for bulk scoring, faster, but each answer then depends on the batch.
- `--noul-commit`: reports a yes/no probability inside JevBench v1.5's no-answer band at the band's edge; the answer never changes, its calibration does.
- `--prompt-tail compact`: the earlier layout, for tasks registered under it.
- `--image-model`: a second engine on the same card for requests that carry images.
- `--backend`: `vllm` (the default), `mlx` for Apple silicon, `hf` for the CPU stand-in.

## How it works

Each question becomes one prompt, the state followed by the question and its lettered options, and its answer is the model's distribution over the option letters at one position, so every question costs one forward pass and no generated text.
The state is prefilled once per request and every question reads it from vLLM's prefix cache.
decisio's vLLM plugin registers its model classes through an entry point, so vLLM loads them without a patch, and the served class also returns the hidden state at the answer position for the intent head.
The design notes: [`docs/design/vllm-plugin.md`](docs/design/vllm-plugin.md), [`docs/design/hidden-state-readout.md`](docs/design/hidden-state-readout.md) and [`docs/design/mlx-backend.md`](docs/design/mlx-backend.md).

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
docs/                   running, the API, the command line, task registration, design notes
docs/demos/             the demos judged decision by decision, and their clips
examples/tasks/         a runnable task-registration walk-through (CPU stand-in)
examples/demos/         the demos' clients and the tools that render their clips
EVAL_CARD.md            what was measured, on what, and what was fitted on what
```

## Contributing

See `CONTRIBUTING.md`.
Pull requests need a DCO sign-off (`git commit -s`), pass the CPU tests and lint, and are reviewed by a maintainer before merge.
Security issues go through GitHub's private vulnerability reporting, as described in `SECURITY.md`.

## Licence

Apache-2.0 (`LICENSE`, `NOTICE`).
The model weights are Alibaba's Qwen3.6-35B-A3B under Apache-2.0 and, for the second base, Google's Gemma 4 12B under Apache-2.0 with Google's Gemma Prohibited Use Policy; both are downloaded, not redistributed.
`THIRD-PARTY.md` lists everything else this project builds on.
