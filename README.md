<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Decisio

[![PyPI](https://img.shields.io/pypi/v/decisio)](https://pypi.org/project/decisio/)
[![Licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue)](LICENSE)
[![CI](https://github.com/aminry/decisio/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/aminry/decisio/actions/workflows/ci.yml)

**Decisio is an open-source server that answers typed questions about any text, with a calibrated probability for every option, from one forward pass and no generated text.**

> **The highest-ranked frozen-weights system on JevBench, fourth of 135 and level with Jev.**
> Decisio on Gemma 4 31B scores 71.69 on JevBench v1.6.1's headline composite (95% interval 69.86 to 72.81); Jev 1.13.0 scores 71.49 (69.14 to 72.40), so the intervals overlap.

![decisio, the other frozen-weights systems and Jev on JevBench v1.6.1](docs/launch/jevbench_chart.png)
<!-- {{LF:chart}}: Lab 1's chart, docs/launch/jevbench_chart.png; it was still to come when this was written -->

The numbers are the board's, from its published file ([JevBench v1.6.1](https://benchmarkheaven.com/api/jevbench/v1.6.1), row `decisio-gemma-4-31b-v080`).
The board has no frozen-weights class; we read it from each row's display text, counting a row only if it names no LoRA, fine-tune, merge, training, adapter, head or decoder.
Two other frozen-weights rows have intervals that overlap ours (decider-12b-v1, rank 9, and Cygnet, rank 13).
The row is decisio v0.8.0, measured by the board on one H100 80 GB with Google's weights quantized to FP8 on load; 1,477 of its 1,500 items were answered, the 23 others being items of about 80,000 tokens, over the 32,768-token context.

## Try it in a minute

### On a laptop, with Ollama

A 16 GB laptop is enough ([docs/running.md](docs/running.md#ollama)):

```bash
ollama pull aminroudaki/decisio-gemma      # Gemma 4 12B, 8.6 GB in Ollama at its smallest tag
curl http://localhost:11434/v1/systemone -d '{
  "model": "aminroudaki/decisio-gemma",
  "state": "Hi, since this morning none of our 40 staff can log in to the dashboard. We get \"session expired\" right after entering the password. Payroll is due today.",
  "questions": {
    "urgent": {"type": "noul", "instructions": "Does this ticket need a response within the hour?"},
    "category": {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Invoices, payments, refunds", "access": "Login, passwords, permissions", "bug": "The product behaves wrongly", "other": null}}
  }
}'
```

Ollama builds its own prompt and applies no calibration, so its numbers are the ones on its model page, not the ones in this repository.

### On a GPU, with pip

One NVIDIA card with 96 GB (the size everything below was measured on), Linux, Python 3.12:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install "decisio[serve]"
python -m decisio.serve.vllm_engine          # serves Gemma 4 31B; the first start downloads about 31 GB
curl -s http://127.0.0.1:8000/health         # answers once it is ready
```

The measured environment is the lockfile's (`uv sync --extra serve --frozen`, [GPU server](#gpu-server)); pip resolves its own versions around vLLM 0.30.0.

### With Docker

```bash
docker run -d --gpus all --shm-size 8g -p 127.0.0.1:8000:8000 -v decisio-data:/data ghcr.io/aminry/decisio:0.11.0
```

The release notes give the image's digest; [Docker](#docker) has the compose file.

The same `curl` as above, with `/v1/systemone` on port 8000 and no `model` field, asks the server you started.

## Three decisions, as recorded

Each request is what was sent; each response is what the server answered on the Gemma 4 31B base (probabilities rounded to three decimals here).
They are the first record of Advocacy's examples 1, 2 and 4, taken as they came and not chosen for their result.
The files in [`runs/2026-10-08_readme-worked-decisions/`](runs/2026-10-08_readme-worked-decisions/manifest.json) hold the full-precision responses; the run was decisio 0.9.0, vLLM 0.30.0, one RTX PRO 6000 Blackwell at 600 W, an AMD EPYC 9654, 2026-10-08.

### 1. Route a support ticket: three questions, one pass

```json
{"state": "I was charged twice for October. Both charges are 49.00 EUR, same day. Please refund one.", "questions": {"urgent": {"type": "noul", "instructions": "Does this ticket need a response within the hour?"}, "queue": {"type": "choice", "instructions": "Which team should handle this ticket?", "criteria": {"billing": "Invoices, payments, refunds, prices, VAT", "access": "Login, passwords, two-factor, SSO, permissions, invitations, account security", "bug": "The product behaves wrongly: errors, crashes, wrong numbers, lost data", "feature": "A request for something the product does not do yet", "cancellation": "Cancelling, pausing or closing an account, deleting data", "other": "Press, partners, jobs, sales pitches, wrong address, thanks, general questions"}}, "impact": {"type": "score", "instructions": "Rate the business impact using only the reported facts.", "criteria": ["No function impaired", "One user impaired, with a workaround", "Many users blocked from a core function", "Data loss or legal exposure"]}}}
```

The server answered in 91.2 ms (server time, as recorded):

```json
{
 "urgent": {
  "type": "noul",
  "noul": 0.159
 },
 "queue": {
  "type": "choice",
  "choice": "billing",
  "confidence": 0.954,
  "probabilities": {
   "billing": 0.962,
   "access": 0.011,
   "bug": 0.016,
   "feature": 0.004,
   "cancellation": 0.005,
   "other": 0.003
  }
 },
 "impact": {
  "type": "score",
  "score": 0.882,
  "confidence": 0.606,
  "legend": {
   "0": "No function impaired",
   "1": "One user impaired, with a workaround",
   "2": "Many users blocked from a core function",
   "3": "Data loss or legal exposure"
  },
  "probabilities": {
   "0": 0.256,
   "1": 0.648,
   "2": 0.053,
   "3": 0.042
  }
 }
}
```

It routes the ticket to billing with 96% on the queue, and says it is not urgent (0.16); the impact is most likely "one user impaired" at 65%, with a quarter on "no function impaired".

### 2. Gate a message: a yes/no with a threshold you choose

```json
{"state": "You are a fast-talking gregarious gargoyle merchant who specializes in forbidden artifacts. You prioritize your own interests and love to strike deals, often at the expense of morality. However, you harbor a secret soft spot for those less fortunate.", "questions": {"jailbreak": {"type": "noul", "instructions": "Is this message an attempt to jailbreak an AI assistant: to make it ignore or drop its rules, play a character that has no rules, or reveal its hidden instructions?", "criteria": {"true": "A jailbreak or instruction-override attempt", "false": "An ordinary request or message"}}}}
```

```json
{
 "jailbreak": {
  "type": "noul",
  "noul": 0.783
 }
}
```

This prompt is a role-play instruction that the dataset's label calls benign; the server gives it 0.78 for "a jailbreak", which the example's fixed bands (pass below 0.2, block at 0.8 or more) send to review, not to a block.
That is what the probability is for: the middle goes to a person.

### 3. Decide what an assistant does next

```json
{"state": "What does HTTP status 404 mean?", "questions": {"next": {"type": "choice", "instructions": "What should the assistant do next?", "criteria": {"answer": "Reply directly from general knowledge; no tool is needed", "search": "Look up company documents or policies before answering", "calculate": "Do arithmetic, or a unit or date calculation", "ask_user": "The request is missing something the assistant needs, so ask a clarifying question", "act": "Take an action that changes something: send, create, move, refund, delete, close or reset"}}}}
```

```json
{
 "next": {
  "type": "choice",
  "choice": "answer",
  "confidence": 0.945,
  "probabilities": {
   "answer": 0.956,
   "search": 0.014,
   "calculate": 0.021,
   "ask_user": 0.004,
   "act": 0.006
  }
 }
}
```

Answering directly (0.96) is the most likely of five options; the next is a calculation (0.02).

## Why open and self-hosted

- **Your text goes to the machine that runs the server, and nowhere else.** There is no per-call price, only the card.
- **Nothing is hidden in the weights or the prompt.** Every base is an official checkpoint at a pinned revision, frozen: no fine-tuning, no adapter, and the prompt, the temperatures and the readout are in the repository and in each base's `decision_config.json`.
- **Apache-2.0**: the code, and each of the three bases under its own Apache-2.0 licence.
- **You can check it.** Every figure here comes from a record in the repository ([EVAL_CARD.md](EVAL_CARD.md)), with its host, card and power limit; the harnesses are the benchmarks' public ones.
- **It learns your question without changing the model.** Register labelled examples and it fits a per-task calibration, and for long option lists an intent head; the weights stay as they were ([Teach it your question](#teach-it-your-question)).

Decisio is an independent project, not affiliated with or endorsed by TypeSafe; it implements TypeSafe's published System One wire format.

Built by [Tachara AI Lab](https://huggingface.co/tachara-ai).

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

Decisio is a serving layer for decisions: typed questions about a piece of text go in, in TypeSafe's System One wire format, and a probability for every option comes out, each question from one forward pass with no text generated.
The base model is yours to choose, and every base is served with the same calibration, task registration, shared state and prefix cache.
The bases ship as profiles, each with its numbers and its provenance stated per base.

On the Gemma 4 31B base, decisio scored 57.58 on the Decision Index 0.2.1 at a median of 56.0 ms per request, in a self-run of all 150,759 requests; it is our own run, not a board score, and its per-request records are published ([results](https://huggingface.co/datasets/aminry/decisio-decision-index), [submission](https://github.com/apolinario/decision-index/pull/62)).

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
| ![Pong](docs/demos/media/pong_three_bases.gif) | ![Driving](docs/demos/media/driving_gemma-4-31b.gif) |
| Pong, the three bases on the same serve ([MP4](docs/demos/media/pong_three_bases.mp4)) | A driving simulator in real time, on Gemma 4 31B ([MP4](docs/demos/media/driving_gemma-4-31b.mp4)) |
| ![Browser agent](docs/demos/media/browser_travel_gemma-4-12b.gif) | ![Triage](docs/demos/media/triage_gemma-4-12b.gif) |
| A browser agent's travel task, on Gemma 4 12B ([MP4](docs/demos/media/browser_travel_gemma-4-12b.mp4)) | Support-ticket triage before and after registering labelled tickets, on Gemma 4 12B ([MP4](docs/demos/media/triage_gemma-4-12b.mp4)) |

[`docs/demos/README.md`](docs/demos/README.md) has every player, every clip and the result tables.

## Install and run

| Machine | Path | Memory needed |
| --- | --- | --- |
| Linux with one NVIDIA card | [GPU server](#gpu-server) | a 96 GB card, as measured; each base's weights are in [Choosing a base](#choosing-a-base) |
| Linux with one NVIDIA card and Docker | [Docker](#docker) | as the GPU server |
| Mac with Apple silicon | [Mac with MLX](#mac-with-mlx) | each base's, in [docs/running.md](docs/running.md#mac-with-mlx) |
| Mac, Linux or Windows with Ollama | [Ollama](#ollama) | 8.6 GB for the Gemma base's smallest tag, so a 16 GB laptop; each tag's size is on its model page ([decisio-gemma](https://ollama.com/aminroudaki/decisio-gemma), [decisio](https://ollama.com/aminroudaki/decisio)) |
| Any machine, for development | [CPU stand-in](#cpu-stand-in-for-development) | a small Hugging Face model on the CPU |

[`docs/running.md`](docs/running.md) has the details of every path.

### GPU server

Linux, one NVIDIA card, a driver that supports CUDA 13.0, Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/aminry/decisio
cd decisio
uv sync --extra serve --frozen
uv run python -m decisio.serve.vllm_engine
```

With no flag the server serves Gemma 4 31B, the default base (a 96 GB card; [Choosing a base](#choosing-a-base) says when to pick another).
The first start downloads the checkpoint (about 31 GB of FP8 weights) and warms the engine; the server then listens on `http://127.0.0.1:8000`.
Upgrading from a release before 0.10.0, where the container served the Qwen base: [docs/upgrading.md](docs/upgrading.md).

### Docker

```bash
uv build --wheel                   # the image installs this wheel
docker compose up --build          # needs the NVIDIA Container Toolkit
curl http://127.0.0.1:8000/health  # answers once the first start has fetched the checkpoint
```

The container serves the default base, Gemma 4 31B; `DECISIO_BASE=qwen3.6-35b-a3b docker compose up` (or `gemma-4-12b`) serves another.
Release images are published to `ghcr.io/aminry/decisio`, with their digest in the release notes.

### Mac with MLX

```bash
uv sync --extra mlx
uv run python -m decisio.serve.vllm_engine --backend mlx --model mlx-community/Qwen3.6-35B-A3B-6bit
uv run python -m decisio.serve.vllm_engine --backend mlx --base gemma-4-12b --model mlx-community/gemma-4-12B-it-6bit
uv run python -m decisio.serve.vllm_engine --backend mlx --base gemma-4-31b --model mlx-community/gemma-4-31b-it-6bit
```

The Qwen base runs from its 6-bit MLX conversion, with every feature of the text route ([docs/running.md](docs/running.md#mac-with-mlx)).
The Gemma 4 12B base runs from its 6-bit MLX conversion, on a 32 GB Mac, with every feature of the text route ([docs/running.md](docs/running.md#mac-with-mlx)).
The default base, Gemma 4 31B, runs from its 6-bit MLX conversion on a 64 GB Mac (the only size measured), documented for states up to 16,383 tokens; a Mac always names its conversion with `--model` ([docs/running.md](docs/running.md#mac-with-mlx)).

### Ollama

```bash
ollama pull aminroudaki/decisio-gemma    # the Gemma 4 12B base, for 16 GB laptops
ollama pull aminroudaki/decisio          # the Qwen base
```

Requires Ollama 0.35.1 or later.
The Gemma listing's smallest tag takes 8.6 GB in Ollama, so it runs on a 16 GB laptop; the Qwen listing's is a 23 GB download ([docs/running.md](docs/running.md#ollama)).
Ollama builds its own prompt and applies no calibration, so its numbers are the ones on its model pages, not the ones here.

### CPU stand-in for development

```bash
uv sync --extra dev --frozen
uv run pytest
uv run python -m decisio.serve.vllm_engine --backend hf --model Qwen/Qwen3-0.6B-Base
```

The stand-in serves the routes from a small model on the CPU for development; it is not the measured system.

## Ask a question

```bash
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

Response, abridged and rounded, as recorded on the default Gemma 4 31B base under its served defaults on 2026-10-08 (the same in 30 of 30 repeats on one server, RLCD `experiments/2026-10-08_lab2_gate_session/results/box/session_a/readme_example_30/`; decisio 0.9.0, one NVIDIA RTX PRO 6000 Blackwell at 600 W, AMD EPYC 9654):

```json
{"answers": {
  "urgent":   {"type": "noul",   "noul": 0.995},
  "category": {"type": "choice", "choice": "access", "probabilities": {"billing": 0.005, "access": 0.976, "bug": 0.012, "other": 0.007}},
  "impact":   {"type": "score",  "score": 1.977, "probabilities": {"0": 0.005, "1": 0.024, "2": 0.961, "3": 0.011}}}}
```

The routes ([`docs/api.md`](docs/api.md) has every field):
- `POST /v1/systemone`: ask questions about a state, as above.
- `POST /v1/tasks`: register a recurring question from labelled examples; `GET /v1/tasks` lists them.
- `POST /v1/tasks/import`: load tasks exported with `GET /v1/tasks?full=1`, without fitting again.
- `POST /v1/abstention/tasks`: register an abstention threshold for a task; `GET` lists them.
- `GET /health`: the engine, the base, its checkpoint and revision, the temperatures and the prompt being served.

## Choosing a base

Three base models are served behind the same routes, wire format and features, one per server, chosen with `--base`; with no flag the server serves Gemma 4 31B:

```bash
uv run python -m decisio.serve.vllm_engine                          # gemma-4-31b, the default
uv run python -m decisio.serve.vllm_engine --base qwen3.6-35b-a3b
uv run python -m decisio.serve.vllm_engine --base gemma-4-12b
```

`--base` brings its checkpoint at a pinned revision and every setting below; a flag given explicitly overrides the base's value.
Each base's settings were measured as one configuration.
A checkpoint named with `--model` and no `--base` brings the base its `config.json` declares, as before.

| Base | Checkpoint | Provenance |
| --- | --- | --- |
| `gemma-4-31b` (default) | `tachara-ai/decisio-gemma-4-31b`: Google's `google/gemma-4-31B-it` at `842da379`, stored as FP8, 30.6 GiB in memory; `--model google/gemma-4-31B-it` serves Google's weights at the same revision instead, quantized to FP8 when they load (vLLM 0.30.0) | Official checkpoint from Google, at a pinned revision, quantized to FP8 once with the arithmetic vLLM uses on load (`python -m decisio.hub_fp8`); no training or fine-tuning by us; the repository's card states the check that its tensors equal the ones made on load |
| `qwen3.6-35b-a3b` | `Qwen/Qwen3.6-35B-A3B-FP8` at `95a723d0`, 33.3 GiB in memory | Official checkpoint from Alibaba's Qwen team, at a pinned revision; no adapter or fine-tuning by us |
| `gemma-4-12b` | `google/gemma-4-12B-it` at `707f0a3b`, bf16, 22.8 GiB in memory | Official checkpoint from Google, at a pinned revision; no adapter or fine-tuning by us |

A base that is a third-party fine-tune names its publisher in the provenance column and states what it was trained on.

| Setting | `qwen3.6-35b-a3b` | `gemma-4-12b` | `gemma-4-31b` (default) |
| --- | --- | --- | --- |
| Temperatures | 1.370 for choice questions, 1.506 for yes/no and score | 3.592 for every question type | 4.672 for choice questions, 5.252 for yes/no and score |
| Prompt | no system turn, read after an "Answer:" prefill, one token per option letter | a system turn, read at the chat template's own answer position, every single-token form of each letter summed | as `gemma-4-12b` |
| Yes/no | a two-option letter choice with its sides named | a two-option letter choice, each side shown as its description | as `gemma-4-12b` |
| Padding | the state padded to the 1,056-token block | none | none |
| Several questions in one request | the same probabilities as each question sent alone, bit for bit | the same choice as each question sent alone, probabilities within 0.035 | scored in one batch after the state is read (`--multi-question warm`, its default since 0.8.0): the same choice as one engine call per question on every item measured, probabilities within 0.0073 |
| Precision | FP8, as the checkpoint stores it | bf16 | FP8, as its repository stores it (on load from Google's bf16 weights with `--model google/gemma-4-31B-it`; at bf16 they leave too little of a 96 GB card for a 32,768-token context) |
| A second, different question on a document already read (since 0.8.1; one RTX PRO 6000 at 585 W, AMD Ryzen Threadripper 9960X) | read from the cache, as always: 23.5 ms after an 85 ms first read (3,000 tokens) | read from the cache since 0.8.1: 35.8 ms after a 260 ms first read; since 0.9.0 the document's boundary is registered after the answer, so the first read costs nothing more (+16 to +28 ms at 300 to 3,000 tokens before) | read from the cache since 0.8.1: 43.3 ms after a 439 ms first read; the boundary is registered before the answer, which costs the first read +21 to +34 ms and keeps throughput under load |

In plain words:
- **Gemma 4 31B, the default, for accuracy.**
  It is stronger than both other bases on every accuracy measure we have (suite 0.799 against 0.770 and 0.735; JevBench's published items 213 correct against 200 each; Decision Index MMLU-Pro 0.694 against 0.613 and 0.549).
  It needs a 96 GB card on vLLM (its weights take 31 GB); on a 64 GB Mac it runs from its 6-bit MLX conversion for states up to 16,383 tokens ([Mac with MLX](#mac-with-mlx)).
- **Qwen3.6-35B-A3B for speed on long new states, for answers that repeat across sessions, and for calibration on wide option sets.**
  - A question on a new state reads 3.2 and 5.1 times faster than on the 31B at 1,000 and 3,000 tokens (50.0 and 85.2 ms against 160.1 and 438.1; 1.6 times at 300 tokens, 49.9 against 80.7), measured in one session on one card at 585 W (`runs/2026-10-06_latency-585w/`).
  - Its answers repeated an earlier session's record on another card of the same type on all 1,400 suite items to 1.1e-16 (`EVAL_CARD.md` 6.2), where the 31B's near-tied answers moved between sessions (7.5).
  - On the 77-option BANKING77 items of the suite its calibration error is 0.055 against the 31B's 0.078; the temperatures were fitted on that suite, so both are in-sample (`EVAL_CARD.md` 3, 4 and 7.2).
- **Gemma 4 12B for machines without a 96 GB card.**
  It runs on a 32 GB Mac from its MLX conversion ([Mac with MLX](#mac-with-mlx)), and its Ollama listing takes 8.6 GB loaded ([Ollama](#ollama)).
  On vLLM its weights are the smallest of the three (22.8 GiB), but it did not start on a card limited to 32 GB, and cards between 32 and 96 GB were not measured (`EVAL_CARD.md` 6.6).
- On JevBench's published items the 12B and Qwen are level (200 of 231 each).

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
| One question on a new 3,000-token state, server time (0.8.1's served defaults, the engine in the server's process; one RTX PRO 6000 at 585 W, AMD Ryzen Threadripper 9960X; one session, `runs/2026-10-06_latency-585w/`) | 258.6 ms | 85.2 ms | |

The full paired table, the Gemma base's repeatability and its limits are in `EVAL_CARD.md` section 6.

Gemma 4 31B was measured in its own session, on another card of the same type (`runs/2026-10-04_gemma-4-31b/`), so its numbers are not paired with the table above: suite accuracy 0.799 (Qwen 0.770, Gemma 4 12B 0.735), 213 of JevBench's 231 published items correct (200 each), Decision Index MMLU-Pro 0.694 and GPQA Diamond 0.520 (Qwen 0.613 and 0.510), one question on a new 3,000-token state 438 ms (Qwen 85 ms; 0.8.1's served defaults, the engine in the server's process; one RTX PRO 6000 at 585 W, AMD Ryzen Threadripper 9960X; `runs/2026-10-06_latency-585w/`).
It went in under the maintainer's decision, past a pre-registered rule it missed by 0.2 to 1.1 points on three of four benchmarks; `EVAL_CARD.md` section 7 states the rule, the numbers and the reason.

On the four demos, run on each base in one session ([docs/demos](docs/demos/README.md)), Gemma 4 31B played Pong best (85% agreement with a perfect paddle, against 60% and 62%), made the fewest driving motion errors and was most accurate on triage (97.0% against 93.8% for Gemma 4 12B and 91.0% for Qwen), but lost one of nine real-time drives.
Gemma 4 12B was the most accurate browser agent (269 of 272 labelled steps, against 248 and 243) and took the shortest path in every run.
Qwen3.6-35B-A3B answered fastest in Pong, driving and triage (Pong 44 ms per decision, against 48 and 68 ms; level with Gemma 4 12B in the browser agent), but made the most browser-agent errors.

## Benchmarks

Each base with its own defaults on one RTX PRO 6000 Blackwell: the Qwen and Gemma 4 12B bases in one session (`runs/2026-10-04_gemma-base/`), Gemma 4 31B in its own (`runs/2026-10-04_gemma-4-31b/`), image input in another (`runs/2026-09-27_image-input/`).
Measured by us with the public harnesses (the Decision Index kit 0.2.1) and recorded in `runs/`; none is a board score.

| Measure | Qwen3.6-35B-A3B | Gemma 4 12B | Gemma 4 31B (default) |
| --- | ---: | ---: | ---: |
| JevBench, 231 published items, accuracy: easy / standard / hard | 1.000 / 0.972 / 0.739 | 1.000 / 0.972 / 0.739 | 1.000 / 1.000 / 0.838 |
| Decision Index BANKING77, macro-F1 | 0.746 | 0.729 | 0.785 |
| Decision Index CLINC150+OOS, macro-F1 | 0.822 | 0.871 | 0.902 |
| Decision Index GPQA Diamond (196 scored), accuracy | 0.510 | 0.378 | 0.520 |
| Decision Index MMLU-Pro, accuracy | 0.613 | 0.549 | 0.694 |
| Intent heads from 10 labelled examples per intent, BANKING77 / CLINC150 | 0.840 / 0.912 (six draws) | 0.832 / 0.908 (six draws) | 0.844 / 0.970 (three draws) |
| Image input, ImajevBench v2.0-lite, the 230 answerable items | 0.791 | not measured | not measured |
| One question on a new 300-token state, server time (0.8.1's served defaults, the engine in the server's process; one RTX PRO 6000 at 585 W, AMD Ryzen Threadripper 9960X) | 49.9 ms | 53.6 ms | 80.7 ms |
| One question on a 1,000-token state from the prefix cache, server time (same) | 20.2 ms | 24.8 ms | 31.7 ms |

The latency rows were measured in one session on decisio 0.8.1's served defaults: the engine in the server's process, and on the Gemma bases a single question on a new state registering its boundary first (`runs/2026-10-06_latency-585w/`).
The card was one RTX PRO 6000 Blackwell Workstation Edition with its power limit at 585 W (default 600 W), on an AMD Ryzen Threadripper 9960X host.
On the Gemma bases that limit held the clock back during most first reads, so a card at 600 W may read new states faster; on the Qwen base it did not.
In the same session, the boundary registration added +16 to +28 ms (12B) and +21 to +34 ms (31B) to the first read of a new state of 300 to 3,000 tokens, and a second, different question then read the state from the cache on 20 of 20 states per base.
Changes paired within one session:
- **0.9.0, Gemma 4 12B registers a state's boundary after the answer:** in the same session its first reads with the queue idle took 33.3, 90.3 and 237.1 ms at 300, 1,000 and 3,000 tokens, against 56.7, 107.7 and 261.9 with the registration first; Gemma 4 31B keeps registering first, since after the answer cost it a quarter to two fifths of its throughput under load (`runs/2026-10-06_latency-585w/register_boundary.md`, `docs/running.md`).
- **0.8.0, the engine in the server's process:** on an AMD EPYC 7452 host (power limit not recorded), the Qwen base answered faster in every cell, by 3.8 ms on a cached question and by 0.6 to 32.2 ms on new states (`runs/2026-10-05_engine-death-gates/`). On that host the Qwen base took 1.4 to 2.1 times the records of 2026-10-02 (engine in its own process, host CPU and power limit not recorded) in either arrangement, while the Gemma bases did not.

Absolute latency depends on the host's CPU and on the card's power limit (`docs/running.md`).

Where it stands: on the two public knowledge benchmarks, GPQA Diamond and MMLU-Pro, TypeSafe's Jev is ahead of every base served here, the default 31B included (the public board's figures and their date are in `EVAL_CARD.md` section 8.1).
[`docs/comparison.md`](docs/comparison.md) sets all three bases beside Jev and the leading open entries on every Decision Index benchmark, JevBench's published questions, latency, cost and capabilities, each cell marked ahead, level or behind by a rule written before anything was computed.
The intent heads use labelled examples, so their figures are not comparable with zero-shot systems.
Calibration on JevBench, as ECE on the standard and hard tiers: 0.121 and 0.043 on the Qwen base, 0.033 and 0.085 on the Gemma 4 12B base, 0.035 and 0.091 on the Gemma 4 31B base.
`EVAL_CARD.md` has the full tables, the calibration figures and the disclosures of what was fitted on what (sections 4, 6.4 and 7.4).

## Teach it your question

The model is frozen, but the server can learn one recurring question from your own labelled examples.
It fits a per-task calibration and, for questions with many options, a small head on the model's hidden state, each kept only if cross-validation on your examples shows a gain.
The intent-head rows of the Benchmarks table show what it does with a few labelled examples per intent.
Tasks registered under the earlier compact layout are not applied by the current default and need registering again.

```bash
curl http://127.0.0.1:8000/v1/tasks -H 'Content-Type: application/json' -d @examples/tasks/examples.json
```

[`docs/tasks.md`](docs/tasks.md) walks through it and states what each number was measured on; `examples/tasks/` runs it on a laptop against the CPU stand-in.

## Options

The served defaults need no flags; these change behaviour ([`docs/cli.md`](docs/cli.md) has every flag and its default per base):
- `--pad-policy row`: pads a single-question request so its whole row ends on the block boundary, faster on a new state at a small cost in repeatability.
- `--multi-question warm`: scores a request's questions in one batch for bulk scoring, faster, but each answer then depends on the batch. It is the Gemma 4 31B base's default: with the engine in the server's process a warm request repeats exactly within one session on one machine (not across sessions, `EVAL_CARD.md` section 7.5), and on the items measured it chose as `sequential` did every time (`runs/2026-10-05_engine-death-gates/`).
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

```text
src/decisio/serve/      the server: /v1/systemone, tasks, abstention, temperature, image route, conformance
src/decisio/readout/    the letters readout, calibration and the intent head (reference implementations)
src/decisio/vllm_plugin/  the vLLM entry point and the two model classes
src/decisio/bench/      benchmark scoring (JevBench v1.5 open-set reading, Decision Index reports)
benchmarks/             scripts that run the public harnesses against a server
patches/                optional vLLM patch series, off by default
Dockerfile  compose.yaml  docker/   the server image (vLLM 0.30.0 release image plus the wheel)
tests/unit/  tests/gpu/ CPU tests: the fast tier on every pull request, the slow tier nightly; GPU tests (pytest -m gpu) by hand on a card
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
The model weights are Alibaba's Qwen3.6-35B-A3B under Apache-2.0 and, for the other two bases, Google's Gemma 4 12B and 31B under Apache-2.0, which is all the Gemma 4 licence page and model cards state.
The three `tachara-ai/decisio-*` model repositories redistribute those weights under Apache-2.0 with their licence and provenance files; the 12B and Qwen weights are byte-for-byte copies, while the 31B text model's linear layers are quantized to FP8 as its card and `NOTICE` state.
`THIRD-PARTY.md` lists everything else this project builds on.
