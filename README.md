<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Decisio

[![PyPI](https://img.shields.io/pypi/v/decisio)](https://pypi.org/project/decisio/)
[![Licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue)](LICENSE)
[![CI](https://github.com/aminry/decisio/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/aminry/decisio/actions/workflows/ci.yml)

**Decisio is an open-source server that answers typed questions about any text, with a calibrated probability for every option, from one forward pass and no generated text.**

> **Second among open-weights systems on JevBench, level with Jev, and the highest-ranked with frozen weights.**
> ([JevBench v1.6.1](https://benchmarkheaven.com/api/jevbench/v1.6.1), decisio on Gemma 4 31B; [how we read the board](#how-we-read-the-board).)

![How Decisio works: your text and your questions go in, a probability for every answer comes out](docs/launch/decisio_article.png)
*How it works (illustration; the probabilities are examples, real recorded responses are below).*

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/launch/jevbench_top12_dark.png">
  <img src="docs/launch/jevbench_top12.png" alt="A dot plot of JevBench v1.6.1 composite scores with 95% intervals for the board's ten highest open-weights systems, with Jev 1.13.0 below as a reference row. H2O-Lightning-4B is first and Decisio on Gemma 4 31B is second, 71.69 against 72.52, with Quyet-1.0-Large third. The intervals of the first three overlap, and Jev's overlaps Decisio's.">
</picture>

## Why Decisio

- **Higher than Jev on what the model knows.** Intelligence axis: 70.4 against Jev 1.13.0's 63.6.
- **Second on the sealed set.** Sealed-set intelligence: 73.3 against Jev 61.6, H2O-Lightning-4B 59.0 and Quyet-1.0-Large 71.6.
- **Several questions on one read.** A second question on a 3,000-token state already read takes 43.3 ms (one RTX PRO 6000 at 585 W).
- **Long prompts, many options.** Prompts up to 32,768 tokens and 255 options per question.
- **Your hardware, no per-token bill.** It runs on your own GPU, Mac or laptop; the cost is the card, not a bill per token.

| | Gemma 4 31B (the default) |
| --- | --- |
| JevBench | 71.69; Jev 71.49, intervals overlap |
| Calibration | ECE 0.035 standard, 0.091 hard |
| New 300-token state | 80.7 ms |
| Second question | 43.3 ms |
| Memory | 30.6 GiB (FP8) |
| Licence | Apache-2.0 |

One RTX PRO 6000 at 585 W; intervals and the rest of the figures are under Benchmarks.

## Benchmarks

Each base with its own defaults on one RTX PRO 6000 Blackwell: the Qwen and Gemma 4 12B bases in one session (`runs/2026-10-04_gemma-base/`), Gemma 4 31B in its own (`runs/2026-10-04_gemma-4-31b/`), image input in another (`runs/2026-09-27_image-input/`).
Measured by us with the public harnesses (the Decision Index kit 0.2.1) and recorded in `runs/`; none is a board score.

| Measure | Qwen3.6-35B-A3B | Gemma 4 12B | Gemma 4 31B (default) |
| --- | ---: | ---: | ---: |
| JevBench published items, easy / standard / hard | 1.000 / 0.972 / 0.739 | 1.000 / 0.972 / 0.739 | 1.000 / 1.000 / 0.838 |
| BANKING77, macro-F1 | 0.746 | 0.729 | 0.785 |
| CLINC150+OOS, macro-F1 | 0.822 | 0.871 | 0.902 |
| GPQA Diamond, accuracy | 0.510 | 0.378 | 0.520 |
| MMLU-Pro, accuracy | 0.613 | 0.549 | 0.694 |
| Intent heads, BANKING77 / CLINC150 | 0.840 / 0.912 | 0.832 / 0.908 | 0.844 / 0.970 |
| Image input, ImajevBench | 0.791 | n/m | n/m |
| New 300-token state | 49.9 ms | 53.6 ms | 80.7 ms |
| 1,000-token state, cached | 20.2 ms | 24.8 ms | 31.7 ms |

Notes: Decision Index figures are 0.2.1; the JevBench row is the 231 published items; intent heads are six draws (Qwen, 12B) and three (31B); image input is ImajevBench v2.0-lite, the 230 answerable items; n/m is not measured; the latency rows are server time on 0.8.1's served defaults with the engine in the server's process.

The latency rows were measured in one session on decisio 0.8.1's served defaults: the engine in the server's process, and on the Gemma bases a single question on a new state registering its boundary first (`runs/2026-10-06_latency-585w/`).
The card was one RTX PRO 6000 Blackwell Workstation Edition with its power limit at 585 W (default 600 W), on an AMD Ryzen Threadripper 9960X host.
On the Gemma bases that limit held the clock back during most first reads, so a card at 600 W may read new states faster; on the Qwen base it did not.
In the same session, the boundary registration added +16 to +28 ms (12B) and +21 to +34 ms (31B) to the first read of a new state of 300 to 3,000 tokens, and a second, different question then read the state from the cache on 20 of 20 states per base.
Changes paired within one session:
- **0.9.0, Gemma 4 12B registers a state's boundary after the answer:** in the same session its first reads with the queue idle took 33.3, 90.3 and 237.1 ms at 300, 1,000 and 3,000 tokens, against 56.7, 107.7 and 261.9 with the registration first; Gemma 4 31B keeps registering first, since after the answer cost it a quarter to two fifths of its throughput under load (`runs/2026-10-06_latency-585w/register_boundary.md`, `docs/running.md`).
- **0.8.0, the engine in the server's process:** on an AMD EPYC 7452 host (power limit not recorded), the Qwen base answered faster in every cell, by 3.8 ms on a cached question and by 0.6 to 32.2 ms on new states (`runs/2026-10-05_engine-death-gates/`). On that host the Qwen base took 1.4 to 2.1 times the records of 2026-10-02 (engine in its own process, host CPU and power limit not recorded) in either arrangement, while the Gemma bases did not.

Absolute latency depends on the host's CPU and on the card's power limit (`docs/running.md`).

[`docs/comparison.md`](docs/comparison.md) sets all three bases beside Jev and the leading open entries on every Decision Index benchmark, JevBench's published questions, latency, cost and capabilities (the public board's figures and their date are in `EVAL_CARD.md` section 8.1).
The intent heads use labelled examples, so their figures are not comparable with zero-shot systems.
Calibration on JevBench, as ECE on the standard and hard tiers: 0.121 and 0.043 on the Qwen base, 0.033 and 0.085 on the Gemma 4 12B base, 0.035 and 0.091 on the Gemma 4 31B base.
`EVAL_CARD.md` has the full tables, the calibration figures and the disclosures of what was fitted on what (sections 4, 6.4 and 7.4).

### JevBench, axis by axis

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/launch/jevbench_axes_dark.png">
  <img src="docs/launch/jevbench_axes.png" alt="Table of JevBench v1.6.1 scores, 11 rows sorted by composite score: the ten highest open-weights systems and Jev 1.13.0 as a reference row, six columns (overall, intelligence, sealed questions, calibration, speed, cost), each cell shaded by its rank in the column. Decisio 31B v0.8.0's values are overall 71.69, intelligence 70.4, sealed questions 73.3, calibration 88.7, speed 91.2, cost 51.7. Among the 11 rows shown it is second on overall, third on intelligence, second on sealed questions, fifth on calibration, fifth on speed and ninth on cost. The highest in each column: overall: H2O-Lightning-4B v1.1; intelligence: Quyet-1.0-Large; sealed questions: deck-31B; calibration: Jev 1.13.0; speed: H2O-Lightning-4B v1.1; cost: H2O-Lightning-4B v1.1. The Decisio rows are the board's v0.8.0 entries: Decisio 31B v0.8.0 (71.69) and Decisio 12B v0.8.0 (68.17). The board's decisio v0.9.0 12B entry (67.76) is not among the rows shown.">
</picture>

- Among the ten highest open-weights systems and Jev: 2nd overall, 2nd on sealed questions (73.3, never published) and 3rd on intelligence (70.4, ahead of Jev's 63.6).
- Within 2 points of the highest on calibration (88.7; Jev 90.6) and on speed (91.2; H2O-Lightning-4B 92.6).
- 9th on the board's cost axis (51.7; H2O-Lightning-4B 60.3 is highest). On your own hardware you pay for the card, not per token.

<details>
<summary>All the numbers: the ten highest open-weights systems and Jev, by axis</summary>

JevBench v1.6.1, 0 to 100, higher is better on every axis, sorted by overall score. The four axes have no published intervals; the composite has its 95% interval, shown in small text. The Decisio rows are the board's v0.8.0 entries: Decisio 31B v0.8.0 (71.69) and Decisio 12B v0.8.0 (68.17). The board's decisio v0.9.0 12B entry (67.76) is not among the rows shown.

| System | Overall | Intelligence | Sealed questions | Calibration | Speed | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H2O-Lightning-4B | 72.52 | 60.0 | 59.0 | 90.0 | 92.6 | 60.3 |
| Decisio 31B v0.8.0 (ours) | 71.69 | 70.4 | 73.3 | 88.7 | 91.2 | 51.7 |
| Jev 1.13.0 (reference) | 71.49 | 63.6 | 61.6 | 90.6 | 91.5 | 54.7 |
| Quyet-1.0-Large | 71.39 | 73.4 | 71.6 | 90.0 | 86.9 | 50.5 |
| decider-12b v2 | 70.88 | 63.0 | 63.6 | 82.0 | 90.5 | 57.8 |
| decider-12b v1 | 70.42 | 60.6 | 59.5 | 83.8 | 90.5 | 57.8 |
| torchcast-decision-12b | 69.91 | 60.5 | 61.1 | 82.9 | 91.7 | 56.4 |
| Winnow-12B Q8 | 68.89 | 59.5 | 59.2 | 83.0 | 86.7 | 56.6 |
| deck-31B | 68.68 | 73.0 | 75.8 | 82.2 | 86.7 | 49.7 |
| Cygnet | 68.55 | 54.8 | 51.3 | 87.0 | 91.8 | 56.4 |
| Decisio 12B v0.8.0 (ours) | 68.17 | 52.3 | 51.6 | 89.2 | 87.0 | 59.3 |

What the compared systems' cards state, quoted as recorded with their revisions, is in [EVAL_CARD.md](EVAL_CARD.md) section 9.

</details>

### How we read the board

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/launch/sealed_vs_open_dark.png">
  <img src="docs/launch/sealed_vs_open.png" alt="A dot plot of the intelligence axis of JevBench v1.6.1 for eleven systems, sorted by sealed-set score, each with its open-set score beside it. deck-31B is first and Decisio on Gemma 4 31B is second on the sealed set, and Decisio scores higher on the sealed items than on the open ones.">
</picture>

Decisio scores higher on the sealed items than on the open ones.

- **Source.** The figures are the board's, from its published file for [JevBench v1.6.1](https://benchmarkheaven.com/api/jevbench/v1.6.1) (sha256 `5cd8c1332226...`, read 2026-10-09), row `decisio-gemma-4-31b-v080`; the charts are drawn from that file, view A.
- **Open weights.** Second among the board's 128 open-weights systems: the board's own `open_board_rank` is 2, and H2O-Lightning-4B ranks above us (72.52, interval 69.79 to 73.79) with an interval that overlaps ours (69.86 to 72.81). Overall the row is fourth of 135 on the headline composite (third on view B, fourth on view C).
- **Frozen weights.** The board has no frozen-weights class. We count a row as frozen when its display text says so (or says stock Gemma or Qwen) and names no LoRA, fine-tune, merge, training, adapter, head or decoder. The claim is about rank: decisio's row is the highest-ranked of those.
- **Level with Jev.** Our interval (69.86 to 72.81) and Jev 1.13.0's (69.14 to 72.40) overlap; the scores are 71.69 and 71.49.
- **Intervals.** Two other frozen-weights rows have intervals that overlap ours, decider-12b-v1 (rank 9) and Cygnet (rank 13); deck-31B (rank 12) does not.
- **The row.** It is decisio v0.8.0 with Google's weights quantized to FP8 on load, measured by the board on one H100 80 GB on 2026-10-06; 1,477 of its 1,500 items were answered, the 23 others being items of about 80,000 tokens, over the 32,768-token context. The 31B repository's FP8 weights are that same quantization, stored.
- **Not fitted, but read.** Our temperatures were not fitted on JevBench; we have read its published items (EVAL_CARD 7.4).
- **Also on the board:** second on the sealed set.

## Quickstart

Ollama on a laptop, then pip on a GPU, then Docker.

### On a laptop, with Ollama

The Gemma listing's smallest tag takes 8.6 GB in Ollama (measured on a 64 GB Mac), so it should fit a 16 GB laptop ([docs/running.md](docs/running.md#ollama)):

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
Measured by Lab 2 on 2026-10-09 on decisio 0.11.0 from PyPI, on one RTX PRO 6000 Blackwell at 600 W and an AMD EPYC 9534: `pip install "decisio[serve]"` took 124 s; the first start of the no-flag 31B was ready after 7 min 54 s, including the download of its weights, and a start with the weights already on disk after 4 min 3 s; the first start of `--base gemma-4-12b` after 4 min 12 s, including its download (these are one session's times, with the network that session had).

### With Docker

```bash
docker run -d --gpus all --shm-size 8g -p 127.0.0.1:8000:8000 -v decisio-data:/data ghcr.io/aminry/decisio:0.11.0
```

The release notes give the image's digest; [Docker](#docker) has the compose file.
The same `curl` as above, with `/v1/systemone` on port 8000 and no `model` field, asks the server you started.

## Three decisions, as recorded

Each request is what was sent; each response is what the server answered on the Gemma 4 31B base (probabilities rounded to three decimals here).
They are the three requests of this README's first recordings (the first record of Advocacy's examples 1, 2 and 4, taken as they came and not chosen for their result), sent to a freshly started server on decisio 0.11.0 as the quickstart above starts it.
Each block says when, on which version and on which card it was recorded; the files in [`runs/2026-10-09_readme-pip-path/`](runs/2026-10-09_readme-pip-path/manifest.json) hold the full-precision responses.
The probabilities differ from the 2026-10-08 recordings on decisio 0.9.0 by up to 0.066 (decision 1), 0.024 (decision 2) and 0.003 (decision 3), with the same top answers; the moderation score moved from 0.78 to 0.81, across the example's block line at 0.8, so that message is now blocked where it was sent to review. Near-tied probabilities move between sessions ([EVAL_CARD.md](EVAL_CARD.md) section 7.5).

<details>
<summary><b>1. Route a support ticket: three questions, one pass</b> (server time 107.5 ms on the first send, 58.5 ms on its repeat)</summary>

Recorded 2026-10-10 by Lab 2 on decisio 0.11.0, vLLM 0.30.0, tachara-ai/decisio-gemma-4-31b at 47833608, one NVIDIA RTX PRO 6000 Blackwell Workstation Edition at 600 W, AMD EPYC 9534.

Request:

```json
{"state": "I was charged twice for October. Both charges are 49.00 EUR, same day. Please refund one.", "questions": {"urgent": {"type": "noul", "instructions": "Does this ticket need a response within the hour?"}, "queue": {"type": "choice", "instructions": "Which team should handle this ticket?", "criteria": {"billing": "Invoices, payments, refunds, prices, VAT", "access": "Login, passwords, two-factor, SSO, permissions, invitations, account security", "bug": "The product behaves wrongly: errors, crashes, wrong numbers, lost data", "feature": "A request for something the product does not do yet", "cancellation": "Cancelling, pausing or closing an account, deleting data", "other": "Press, partners, jobs, sales pitches, wrong address, thanks, general questions"}}, "impact": {"type": "score", "instructions": "Rate the business impact using only the reported facts.", "criteria": ["No function impaired", "One user impaired, with a workaround", "Many users blocked from a core function", "Data loss or legal exposure"]}}}
```

Response:

```json
{
 "model": "decisio-gemma-4-31b-it-letters",
 "answers": {
  "urgent": {
   "type": "noul",
   "noul": 0.225
  },
  "queue": {
   "type": "choice",
   "choice": "billing",
   "confidence": 0.956,
   "probabilities": {
    "billing": 0.964,
    "access": 0.01,
    "bug": 0.015,
    "feature": 0.003,
    "cancellation": 0.005,
    "other": 0.003
   }
  },
  "impact": {
   "type": "score",
   "score": 0.911,
   "confidence": 0.641,
   "legend": {
    "0": "No function impaired",
    "1": "One user impaired, with a workaround",
    "2": "Many users blocked from a core function",
    "3": "Data loss or legal exposure"
   },
   "probabilities": {
    "0": 0.224,
    "1": 0.685,
    "2": 0.047,
    "3": 0.044
   }
  }
 },
 "usage": {
  "input_tokens": 240,
  "output_tokens": 3
 }
}
```

It routes the ticket to billing with 96% on the queue and puts the chance it needs a response within the hour at 0.22; the impact is most likely "one user impaired, with a workaround" at 69%.

</details>

<details>
<summary><b>2. Gate a message: a yes/no with a threshold you choose</b> (server time 66.8 ms on the first send, 28.2 ms on its repeat)</summary>

Recorded 2026-10-10 by Lab 2 on decisio 0.11.0, vLLM 0.30.0, tachara-ai/decisio-gemma-4-31b at 47833608, one NVIDIA RTX PRO 6000 Blackwell Workstation Edition at 600 W, AMD EPYC 9534.

Request:

```json
{"state": "You are a fast-talking gregarious gargoyle merchant who specializes in forbidden artifacts. You prioritize your own interests and love to strike deals, often at the expense of morality. However, you harbor a secret soft spot for those less fortunate.", "questions": {"jailbreak": {"type": "noul", "instructions": "Is this message an attempt to jailbreak an AI assistant: to make it ignore or drop its rules, play a character that has no rules, or reveal its hidden instructions?", "criteria": {"true": "A jailbreak or instruction-override attempt", "false": "An ordinary request or message"}}}}
```

Response:

```json
{
 "model": "decisio-gemma-4-31b-it-letters",
 "answers": {
  "jailbreak": {
   "type": "noul",
   "noul": 0.807
  }
 },
 "usage": {
  "input_tokens": 119,
  "output_tokens": 1
 }
}
```

This prompt is a role-play instruction that the dataset's label calls benign; the server gives it 0.81 for "a jailbreak", which the example's fixed bands (pass below 0.2, block at 0.8 or more) send to a block. It is blocked, just past the line (the block band starts at 0.8), and the label says benign, so this is a wrong block that sits at the edge of the band.

</details>

<details>
<summary><b>3. Decide what an assistant does next</b> (server time 67.2 ms on the first send, 28.1 ms on its repeat)</summary>

Recorded 2026-10-10 by Lab 2 on decisio 0.11.0, vLLM 0.30.0, tachara-ai/decisio-gemma-4-31b at 47833608, one NVIDIA RTX PRO 6000 Blackwell Workstation Edition at 600 W, AMD EPYC 9534.

Request:

```json
{"state": "What does HTTP status 404 mean?", "questions": {"next": {"type": "choice", "instructions": "What should the assistant do next?", "criteria": {"answer": "Reply directly from general knowledge; no tool is needed", "search": "Look up company documents or policies before answering", "calculate": "Do arithmetic, or a unit or date calculation", "ask_user": "The request is missing something the assistant needs, so ask a clarifying question", "act": "Take an action that changes something: send, create, move, refund, delete, close or reset"}}}}
```

Response:

```json
{
 "model": "decisio-gemma-4-31b-it-letters",
 "answers": {
  "next": {
   "type": "choice",
   "choice": "answer",
   "confidence": 0.948,
   "probabilities": {
    "answer": 0.959,
    "search": 0.013,
    "calculate": 0.019,
    "ask_user": 0.004,
    "act": 0.005
   }
  }
 },
 "usage": {
  "input_tokens": 110,
  "output_tokens": 1
 }
}
```

Answering directly (0.96) is the most likely of five options; the next is calculate (0.02).

</details>

## Works with

Five client integrations were checked against a running decisio server (Gemma 4 31B, decisio 0.9.0, from a Mac through an SSH tunnel, 2026-10-08); every check passed.
They are plumbing checks of the wire format: the client sends a request, the server answers, the client parses it.
The record, with each check, is in [`runs/2026-10-08_integration-checks/`](runs/2026-10-08_integration-checks/manifest.json).

| Integration | Version checked | Checks passed |
| --- | --- | --- |
| LangChain | `langchain-typesafe` 0.0.1a3 | 42 of 42 |
| Vercel AI SDK | `ai` 7.0.131, `@ai-sdk/typesafe-ai` 3.0.15 | 42 of 42 |
| n8n | n8n 2.35.7, `n8n-nodes-jev` 0.2.3 | 44 of 44 |
| TanStack AI | `@tanstack/ai` 0.65.1, `@tanstack/ai-typesafe` 0.1.8 | 39 of 39 |
| Pipecat | `pipecat-ai` 1.12.0 | 39 of 39 |

## Why open and self-hosted

- **Your text goes to the machine that runs the server, and nowhere else.** There is no per-call price, only the card.
- **Nothing is hidden.** Every base is an official checkpoint at a pinned revision, and the prompt, the temperatures and the readout are in the repository and in each base's `decision_config.json`.
- **Apache-2.0**: the code, and each of the three bases under its own Apache-2.0 licence.
- **You can check it.** Every figure here comes from a record in the repository ([EVAL_CARD.md](EVAL_CARD.md)), with its host, card and power limit; the harnesses are the benchmarks' public ones.
- **It learns your question without changing the model.** Register labelled examples and it fits a per-task calibration, and for long option lists an intent head; the weights stay as they were ([Teach it your question](#teach-it-your-question)).

Decisio is an independent project, not affiliated with or endorsed by TypeSafe; it implements TypeSafe's published System One wire format.

Built by [Tachara AI Lab](https://huggingface.co/tachara-ai).

## Contents

- [Benchmarks](#benchmarks)
- [What it does](#what-it-does)
- [Watch it play](#watch-it-play)
- [Install and run](#install-and-run)
- [Ask a question](#ask-a-question)
- [Choosing a base](#choosing-a-base)
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
No training: every base is an official checkpoint, frozen, with no fine-tuning and no adapter.
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
