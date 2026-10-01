<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Evaluation card

What the served default is, what it can do, how it measures, and what was fitted on what.
Every number is on the current served default and names the record it comes from; each run under `runs/` has a `manifest.json` with its configuration and harness versions and a `files.json` with every file's sha256.

## 1. The system measured

| Part | The served default |
| --- | --- |
| Checkpoint | `Qwen/Qwen3.6-35B-A3B-FP8`, the official weights, untrained by us: no adapter, no fine-tuning |
| Engine | vLLM 0.30.0 with decisio's plugin: the official checkpoint under the hidden-readout class (the text model without its vision tower, plus the hidden state at the answer position); DeepGEMM off; front padding to the 1,056-token block; CUDA graphs up to 4,096 tokens; float32 recurrent state; the suffix-staging patch series optional (latency only) |
| Readout | Letters: each question is one prompt, and its distribution is the softmax over the option letters' logits at the last position |
| Rendering rules | Enumerated keys (`option_0`, `option_1`, ...) are hidden and only descriptions shown; bare snake_case labels are shown as words; the answer keys are the request's own |
| Temperature | T = 1.307 on the text route's plain readout (`softmax(log p / T)`); a registered task's correction replaces it; never changes the chosen option |
| Tie-break | Among exactly tied options, the key that sorts first in Unicode code-point order |
| Intent head | Single engine: the hidden state is read from the serving engine (three extra requests per head question); `--head-engine` reads it from a second engine instead |
| Image route | `--image-model`: the official checkpoint under the multimodal class as a second engine (0.49 of the card beside the text engine at 0.47), for requests that carry images; not tempered |
| Hardware | One NVIDIA RTX PRO 6000 Blackwell (96 GB) |
| Wire format | TypeSafe's System One (`POST /v1/systemone`), checked before every measurement by the serving gates C1 and the conformance gates C2-C4 |

## 2. Capabilities

- Question types: yes/no (`noul`), choice, and score (ordered levels, with an expected level and a confidence).
- Up to 255 options per question, each a key with an optional description.
- Any number of questions per request about one state (a string, an object or an array); the state is prefilled once and shared through the prefix cache.
- Task registration from labelled examples (`POST /v1/tasks`): per-task calibration from 10 examples (about 20 recommended), and an intent head on questions of 10 or more options with 5 examples per option; each is kept only when cross-validation on the examples shows a gain.
- An opt-in abstention threshold per task (`POST /v1/abstention/tasks`), on a declared or appended "can't tell" option.
- Image input: up to 2 images per request (PNG, JPEG or WebP, up to 20 MB each), as data URLs, multipart, or inside the state.
- Two-order averaging, opt-in (`--orders 2` or a request's `orders`), with the two orders' disagreement logged per question.
- Context up to 32,768 tokens per prompt.

## 3. Numbers

| Measure | Result | Record |
| --- | --- | --- |
| JevBench, 231 published items, harness accuracy | easy 1.000 (48), standard 0.972 (72), hard 0.694 (111); every answer schema-valid | `runs/2026-09-30_served-default/jevbench/default/*/summary.json` |
| JevBench, ECE / Brier per file | easy 0.023 / 0.002, standard 0.148 / 0.105, hard 0.059 / 0.393 | same |
| JevBench v1.5 open-set reading | choice 79.5 (easy 100, standard 100, hard 64.1); yes/no -10.9, 54% of yes/no answers between 0.20 and 0.80; score 60.6; I_open 43.1 (equal types), 52.2 (50/25/25) | `runs/2026-09-30_served-default/jevbench/default/v15.json` |
| JevBench latency, one request at a time | p50 50 to 61 ms, p95 52 to 120 ms by file | `.../summary.json` |
| Decision Index 0.2.1, BANKING77 (3,080) | macro-F1 0.731, accuracy 0.740, ECE 0.042 | `runs/2026-09-30_served-default/decision_index/default/di_report.json`, `di_cal.json` |
| Decision Index 0.2.1, CLINC150+OOS (5,500) | macro-F1 0.814, accuracy 0.824, ECE 0.170 | same |
| Decision Index 0.2.1, GPQA Diamond (196 scored) | accuracy 0.454, ECE 0.144 | same |
| Decision Index 0.2.1, MMLU-Pro (12,032) | accuracy 0.609, ECE 0.009 | same |
| 1,400-item suite, accuracy / ECE | BoolQ 0.860 / 0.052, BANKING77 0.740 / 0.092, ToxicChat 0.940 / 0.032, MMLU 0.880 / 0.069, SciFact 0.740 / 0.125, SciFact clarified 0.820 / 0.081, MMLU-Pro 0.600 / 0.083 (150) and 0.657 / 0.050 (350); pooled 0.762 / 0.020 | `runs/2026-09-30_plugin-verification/conformance_H.json.gz` (the C4 rows) |
| Intent heads from 10 labelled examples per intent | BANKING77 0.847 (150 test items), CLINC150 0.893 (100 test items), means of three draws | `runs/2026-09-30_plugin-verification/intent_heads/` |
| Image input, ImajevBench v2.0-lite (254 labelled items) | 0.717 on all items, 0.791 on the 230 answerable; with "can't tell" offered 0.756 and 0.813 (5 of 24 unanswerable right, 2 false abstentions); p50 166 ms | `runs/2026-09-27_image-input/accuracy.json` |
| Latency, one question per request | 27.9 ms server time; 46.5 ms for an intent question, 126.8 ms with its head applied | `runs/2026-09-30_plugin-verification/manifest.json` (`latency.json.gz`) |
| Latency, many questions per request (8,000-token state) | 5.4 ms per question at 100 questions per request (5.4 ms with 20 such requests at once); 2.7 ms at 1,000 | `runs/2026-09-30_plugin-verification/bench_patched_on.json` |
| Cost per 1,000 decisions, at $1.50 per card-hour | $0.012 one question per request, one at a time; $0.0022 at 100 questions per request under load; $0.0012 at 1,000 | same, and the single-request latency above |

ECE is over 10 equal-mass bins of the top probability; JevBench's ECE is its harness's own (10 equal-width bins).
The JevBench and Decision Index numbers were measured with the text-only view of the checkpoint; the hidden-readout class answers the 1,400-item suite bit-identically to it (`runs/2026-09-30_plugin-verification/suite_compare_H.txt`).
With `--temperature 1` every choice and every accuracy is the same; calibration changes, and with it the v1.5 yes/no and score values, which depend on the probabilities (`runs/2026-09-30_served-default`, arm `temperature_1`); T = 1 is a bit-exact no-op, so that arm's JevBench responses are byte for byte those of the untempered run recorded in `runs/2026-09-27_boards-baseline`.
None of these is a board number: JevBench's official score needs its sealed set, and a Decision Index value needs all 38 of its benchmarks.
Nothing was submitted, and no output of Jev (TypeSafe's hosted model behind the System One API, which JevBench is named after) is in this repository.
The records keep the wire names they were written with (`x-rlcd-*`, `rlcd-*/1`, the served name `rlcd-qwen3.6-35b-a3b-letters`); decisio reads both spellings (`decisio.names`).

## 4. What was fitted on what

| Component | Fitted or chosen on |
| --- | --- |
| Model weights | Nothing by us |
| Temperature T = 1.307 | Minimum log loss on the served readouts of the private 1,400-item suite above (eight tasks) |
| Rendering rules | Selected, with no parameter fitted, on the Decision Index's BANKING77 and CLINC150+OOS rows; checked to change nothing on the suite |
| Tie-break | A rule; nothing fitted |
| Intent heads and calibration priors | Per task, on 10 labelled training examples per intent (BANKING77, CLINC150), three draws |
| JevBench v1.5 thresholds and weights | JevBench's published method, as written |

- **The temperature's fit set is the suite in section 3,** so the suite's ECE figures are in-sample; 500 of its MMLU-Pro items and its 150 BANKING77 items are also in the Decision Index's pools (4.2% and 4.9%), in our wording rather than the board's, so T can move ECE on those two benchmarks, never a choice.
- **The rendering rules were selected on the Decision Index's intent rows,** so the BANKING77 and CLINC150+OOS scores measure a configuration chosen on them.
- **The intent heads' test items are also in the Decision Index's pools;** no head was registered during any Decision Index run.

Batch-forward finding: on this stack, questions scored in one batch are not the same forward pass as each scored alone. Identical requests sent in one batch differed by up to 0.59 in a label probability (median 0.029) on 250 intent items, while the same requests sent one at a time agreed to 1.2e-7 (`runs/2026-09-30_plugin-verification/diag_same_row.json`, `diag_same_row_sequential.json`).
The questions of one multi-question request are scored in one batch, so its probabilities vary between repeats on one server within the measured spread, with the chosen option unchanged: about 0.12 on the README's three-question example over repeats, while each of its questions sent alone was identical 30 of 30 times (`runs/2026-10-01_docker-first-gpu-start/repeat_variability/`).
The engine's logits are bf16, and label probabilities recomputed from the hidden state match the engine's within 3e-8 only when rounded to bf16 (`diag_same_forward.json`).
The single-engine head therefore sends its three requests one at a time and recomputes in bf16, and its same-forward gate passes at 3.1e-8 (`same_forward.json`).
Every harness in `runs/` sends one single-question request at a time, the serving gate G2 bounds the effect of 16 extra questions in a request at 6.46e-2 with no change of choice (`gates.log`), and every bit-identity claim compares single-question requests under like request histories.

## 5. Limits

- One card per run and one run per configuration.
- JevBench's v1.5 reading covers 231 of the board's 904 open items, with no judge tier and no sealed half; its published items have been public since v1.2 and may be in any model's pretraining data.
- The Decision Index rows were rebuilt with the kit for four benchmarks; their counts and the rows' sha256 match (`decisio.bench.di_rows`), but a partial rebuild cannot be checked against the whole suite's hash.
- The ImajevBench record was made without the rendering rules and the key-order tie-break; 2 of its 254 items ended in exact ties that the harness rejected and are counted wrong.
- The second-engine head mode with the registered text-only class is unmeasured for latency.
- Latency is server-side on the card's localhost; cost is the card-hour price divided by measured throughput, with nothing else counted.
- The per-item Decision Index records keep each request's id, the payload's sha256 and our response, not the item text (GPQA's authors ask that its items not be published in plain text).
