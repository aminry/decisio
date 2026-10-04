<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Decisio

**A decision server: typed questions about a piece of text, a probability for every option, one forward pass per question.**

Decisio serves a frozen open model on vLLM, Qwen3.6-35B-A3B by default or Gemma 4 12B (see "Choosing a base"), and reads its answer to a closed question from the option letters' logits.
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
| Probabilities | a distribution per question, with fitted temperatures (one for choice questions, one for yes/no and score questions); per-task calibration from labelled examples |
| Task registration | `POST /v1/tasks`: from labelled examples of one recurring question the server fits per-task calibration and, for option lists of 10 or more, a linear head on the model's hidden state; guide in `docs/tasks.md` |
| Abstention | an opt-in per-task threshold on a declared "can't tell" option (`POST /v1/abstention/tasks`) |
| Image input | photos in the state, served by a second engine on the same card (`--image-model`) |
| Rendering rules | the lettered options sit between blank lines and the question closes with one line asking for the chosen option's letter alone, and a yes/no question is asked as a two-option letter choice with its sides named (`--prompt-tail spaced`, `--noul-rendering letters-keys`, the served default since 2026-10-03); an option with a description is shown as its description alone, without its key (`--no-describe-options` shows `key: description`): on 272 steps of a browser-agent demo it cut wrong "done" picks from 63 to 12 of 132, with accuracy within noise on JevBench and the Decision Index; enumerated option keys are hidden, snake_case labels are shown as words, ties resolve by key so the answer never depends on the order keys arrive in |
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
These latencies were measured under the compact layout; on one card in one session the current one measured within 3 ms of it (+0.7 ms on a cached state, 0.0 to +0.5 ms for a single question on a new state, +1.8 to +2.9 ms for four), `runs/2026-10-03_qiv-default/`.

`--pad-policy row` (opt-in) cuts a fresh single question by about 10 to 12 ms: the whole row, state and question, is padded to end on the block boundary, so vLLM prefills it in one engine step instead of two.
Measured against the served default on one card in one session: 48.8 to 38.1 ms at 300 tokens, 52.1 to 40.9 at 1,000, 91.0 to 79.2 at 3,000; multi-question requests are unchanged.
Accuracy and calibration stayed within noise on the suite, JevBench, the four Decision Index benchmarks and the intent heads.
Its cost is repeatability: a repeated identical request can move by up to 0.015 on long prompts (2 of 1,400 suite items in the gate, no answer changed), where the default padding returns the same probabilities every time, so it stays off by default (`runs/2026-10-02_pad-policy-row/`).

## Watch it play

Four demos call a decisio server for every move (`examples/demos/`): Pong, a driving simulator, a browser agent and a support-ticket triage feed.
Each clip is rendered from a recorded run at the speed it happened, with the model, the card, the median latency and `--pad-policy row` in its caption.

| | |
| --- | --- |
| ![Pong, Decisio and Cygnet](docs/demos/media/pong_decisio_cygnet.gif) | ![Driving, Decisio](docs/demos/media/driving_decisio_s1-1.gif) |
| Pong, Decisio and Cygnet on the same serve ([MP4](docs/demos/media/pong_decisio_cygnet.mp4)) | Driving in real time ([MP4](docs/demos/media/driving_decisio_s1-1.mp4); [beside Cygnet](docs/demos/media/driving_decisio_cygnet_s1-1.mp4)) |
| ![Browser agent, Decisio](docs/demos/media/browser_decisio_travel.gif) | ![Triage, before and after registration](docs/demos/media/triage_plain_vs_taught.gif) |
| A browser agent's travel task ([MP4](docs/demos/media/browser_decisio_travel.mp4); [beside Cygnet](docs/demos/media/browser_decisio_cygnet_travel.mp4); [the reading room](docs/demos/media/browser_decisio_reading.mp4)) | Triage before and after registering 200 labelled tickets ([MP4](docs/demos/media/triage_plain_vs_taught.mp4)) |

`docs/demos/README.md` judges every decision against an oracle, classes the errors, and reports the renderings tested on held-out data and what teaching did.

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

Response, abridged (rounded; the same in 30 of 30 repeats on one server, `runs/2026-10-02_readme-example/`):

```
{"answers": {
  "urgent":   {"type": "noul",   "noul": 0.90},
  "category": {"type": "choice", "choice": "access", "probabilities": {"billing": 0.01, "access": 0.97, "bug": 0.01, "other": 0.00}},
  "impact":   {"type": "score",  "score": 2.0, "probabilities": {"0": 0.00, "1": 0.02, "2": 0.94, "3": 0.04}}}}
```

Repeat the request and you get the same probabilities: each question is scored in its own engine call after the state is prefilled once, so its answer equals the question sent on its own. With `--multi-question warm` the three questions are scored in one batch, faster, and their probabilities vary with that batch: by about 0.12 on this example (the probability of `access` ranged from 0.79 to 0.91, `runs/2026-10-01_docker-first-gpu-start/repeat_variability/`), and when two options are close the choice can change too.

## Choosing a base

Two base models are served behind the same routes, wire format and features, one per server, chosen with `--base`:

```
uv run python -m decisio.serve.vllm_engine --base qwen3.6-35b-a3b   # the default: Qwen/Qwen3.6-35B-A3B-FP8
uv run python -m decisio.serve.vllm_engine --base gemma-4-12b       # google/gemma-4-12B-it at a pinned revision
```

`--base` brings its checkpoint and every setting below; a flag given explicitly overrides the base's value.
Without `--base`, the base is detected from `--model`'s `config.json`, so a local copy of either checkpoint brings its own settings.
Each base's settings were measured as one configuration, and the numbers below hold for them as listed.

| Setting | `qwen3.6-35b-a3b` (default) | `gemma-4-12b` |
| --- | --- | --- |
| Checkpoint | `Qwen/Qwen3.6-35B-A3B-FP8`, 33.3 GiB in memory | `google/gemma-4-12B-it` at revision `707f0a3b`, bf16, 22.8 GiB in memory |
| Temperatures | 1.370 for choice questions, 1.506 for yes/no and score | 3.592 for every question type |
| Prompt | the spaced layout, no system turn, read after an "Answer:" prefill, one token per option letter | a system turn, the spaced layout, read at the chat template's own answer position, every single-token form of each letter summed |
| Yes/no | a two-option letter choice with its sides named (`--noul-rendering letters-keys`) | a two-option letter choice, each side shown as its description (`--noul-rendering letters`) |
| Padding | the state padded to the 1,056-token block | none |
| Several questions in one request | each scored in its own engine call: the same probabilities as the question sent alone, bit for bit | each scored in its own engine call: the same choice as the question sent alone, probabilities within 0.035 |

Measured on one RTX PRO 6000 Blackwell in one session (2026-10-04), each base with its own defaults, paired over the same items (Gemma minus Qwen, 95% bootstrap intervals; `runs/2026-10-04_gemma-base/`):

| Measure | Gemma 4 12B | Qwen3.6-35B-A3B | Gemma minus Qwen |
| --- | ---: | ---: | --- |
| JevBench, 231 published items, correct | 200 | 200 | 0.0 points [-4.3, +4.3] |
| JevBench v1.5 open-set reading, I_open (equal types) | 64.4 | 49.4 | +15.0 [+7.0, +23.6] |
| v1.5 by type: choice / yes/no / score | 80.6 / 49.0 / 63.7 | 77.3 / 13.7 / 57.2 | |
| Yes/no answers between 0.20 and 0.80 (74 items) | 15% | 39% | |
| 1,400-item suite, accuracy | 0.735 | 0.770 | -3.5 points [-5.5, -1.6] |
| 1,400-item suite, ECE | 0.029 | 0.033 | |
| Decision Index BANKING77, accuracy (3,080) | 0.741 | 0.755 | -1.4 [-2.7, -0.1] |
| Decision Index CLINC150+OOS, accuracy (5,500) | 0.872 | 0.827 | +4.5 [+3.6, +5.5] |
| Decision Index GPQA Diamond, accuracy (198) | 0.374 | 0.510 | -13.6 [-21.7, -5.6] |
| Decision Index MMLU-Pro, accuracy (12,032) | 0.549 | 0.613 | -6.4 [-7.2, -5.5] |
| Intent heads from 10 examples per intent, BANKING77 / CLINC150 (six draws) | 0.832 / 0.908 | 0.840 / 0.912 | |
| One question, state from the cache (server time) | 26.7 ms | 20.9 ms | |
| One question, new state of 300 / 1,000 / 3,000 tokens | 39 / 102 / 293 ms | 48 / 52 / 91 ms | |
| Four questions, new state of 300 / 1,000 / 3,000 tokens | 131 / 196 / 392 ms | 110 / 114 / 157 ms | |

In plain words:
- Choose Gemma 4 12B for committed yes/no answers, scores and intent routing: its v1.5 reading is 49.0 against 13.7 on yes/no, with 15% of its yes/no answers between 0.20 and 0.80 against 39%, and 63.7 against 57.2 on scores; it routes CLINC150+OOS 4.5 points better (BANKING77 1.4 points lower).
- Choose Qwen3.6-35B-A3B for knowledge questions and long states seen for the first time: GPQA Diamond is 13.6 points higher, MMLU-Pro 6.4 and the suite 3.5, and one question on a new 3,000-token state takes 91 ms against 293.
- On JevBench's 231 published items they are level, 200 correct each.

Repeatability differs.
On a running server, Qwen returns the same probabilities bit for bit every time.
Gemma's logits come out in bf16 after its soft cap, in steps of 0.0625 to 0.125 near the top, and a state read for the first time and the same state read from the prefix cache can land a step apart: within one session its answers moved by up to 0.035 (49 of 231 JevBench items, no choice changed), and between two sessions on two cards by up to 0.128, where two near-tied choices changed (`EVAL_CARD.md` section 6).

Gemma 4 12B does not fit a 32 GB card on vLLM's defaults.
Simulated on the 96 GB card with the engine's share cut to 28.8 GB (vLLM's 0.90 of 32 GB), it did not start at 32,768 or at 16,384 tokens of context: its 22.8 GiB of weights left no room for the cache.
Its path to 32 GB machines is an MLX build, in preparation.

`GET /health` says which base a server runs, in its `profile` block: the base, the checkpoint and its revision, the temperature each question type is served at, and the prompt (layout, answer position, label forms, system turn, yes/no and option rendering, multi-question scoring, padding).
`cache_hit_unit` is the step prefix-cache hits come in (64 tokens on Gemma, whose cache has groups of 16- and 64-token blocks; 1,056 on Qwen) and `hash_unit` the step prefixes are hashed at (16 and 1,056).

## Teach it your question in ten examples

The model is frozen, but the server can learn one recurring question from your own labelled examples: it fits a per-task calibration and, for 10 or more options, a small head on the model's hidden state, each kept only if cross-validation on your examples shows a gain.
With 10 labelled examples per intent, accuracy on held-out test items rose from 0.747 to 0.840 on BANKING77 (77 intents) and from 0.820 to 0.912 on CLINC150 (150 intents), against the same model unregistered (means of six draws).
Tasks registered before 2026-10-03, under the earlier compact layout, are not applied by the current default and need registering again (`docs/tasks.md`).

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
It does not pad a state (`--pad-policy none`, the MLX default; its cache needs no padding), so its prompts are the served ones without the padding, and a question asked alone and inside a request is one prompt.
At 6 bits it passes the served default's gates against the FP8 records (suite accuracy -0.1 [-1.4, +1.1] points, the intent heads reproduced, 30.7 GB at 32k tokens).
A single question takes about 0.27 s on an M5 Pro, about 0.11 s when its state was seen before (a cross-request prefix cache, exact by construction), and about 0.13 s per question when 100 share a state (docs/design/mlx-backend.md, `runs/2026-10-03_mlx-regate`).

A laptop version of the same model, packaged for [Ollama](https://ollama.com/aminroudaki/decisio)'s decision route, exists as `ollama pull aminroudaki/decisio`.
Ollama builds its own prompt and applies no calibration, so its numbers are the ones on that page, not the ones here.

## How it performs

All numbers are on the served default, the Qwen base, described in `EVAL_CARD.md`, measured privately with the public harnesses; nothing here is a leaderboard score.
The Gemma base's numbers are in "Choosing a base" and `EVAL_CARD.md` section 6.

| Measure | Result |
| --- | --- |
| JevBench, 231 published items, accuracy by tier | easy 1.000, standard 0.972, hard 0.739 |
| Decision Index 0.2.1, four benchmarks | BANKING77 macro-F1 0.746, CLINC150+OOS 0.822, GPQA Diamond 0.510, MMLU-Pro 0.613 |
| Intent heads from 10 labelled examples per intent | BANKING77 0.840, CLINC150 0.912 |
| Image input, ImajevBench v2.0-lite | 0.791 on the 230 answerable items (the image route keeps the compact layout) |
| Latency, one question | 48.5 ms server time on a new 300-token state, 27.9 ms on a state from the cache |

Where it stands: on the public harnesses this frozen model is behind TypeSafe's Jev on hard knowledge questions by several points and on intent taxonomies without labelled examples.
With 10 labelled examples per intent, registered heads reach accuracy 0.840 on BANKING77 (150 held-out items) and 0.912 on CLINC150 (100 held-out items), means of six draws.
They use labelled examples, so those figures are not comparable with zero-shot systems.
Calibration on JevBench's hard tier is an ECE of 0.043 (0.069 under the compact layout), with standard-tier ECE still 0.121.
`EVAL_CARD.md` has the full tables, the calibration figures, and the three disclosures about what was fitted on what.

`--noul-commit` (opt-in) is for scorers that treat a yes/no probability between 0.20 and 0.80 as no answer, as JevBench v1.5 does: such an answer is reported at the band's edge on its own side, 0.80 above 0.5 and 0.20 at or below it.
The answer never changes; its probability does, and calibration pays for it.
On 1,474 yes/no items at the served default (the suite's 600, PAWS, Civil Comments, Aegis 2.0 and JevBench's 74), 25.7% fell inside the band; accuracy stayed 83.9%, log loss rose from 0.398 to 0.415 and tie-robust ECE from 0.044 to 0.078, and JevBench v1.5's yes/no competence rose from 13.7 to 79.6 (`runs/2026-10-03_noul-commit/`, `docs/handoffs/tasks.md`).

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
The model weights are Alibaba's Qwen3.6-35B-A3B under Apache-2.0 and, for the second base, Google's Gemma 4 12B under Apache-2.0 with Google's Gemma Prohibited Use Policy; both are downloaded, not redistributed.
`THIRD-PARTY.md` lists everything else this project builds on.
