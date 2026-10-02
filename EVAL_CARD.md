<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Evaluation card

What the served default is, what it can do, how it measures, and what was fitted on what.
Every number is on the current served default and names the record it comes from; each run under `runs/` has a `manifest.json` with its configuration and harness versions and a `files.json` with every file's sha256.

## 1. The system measured

| Part | The served default |
| --- | --- |
| Checkpoint | `Qwen/Qwen3.6-35B-A3B-FP8`, the official weights, untrained by us: no adapter, no fine-tuning |
| Engine | vLLM 0.30.0 with decisio's plugin: the official checkpoint under the hidden-readout class (the text model without its vision tower, plus the hidden state at the answer position); DeepGEMM off; front padding to the 1,056-token block; the questions of a request scored one engine call each after the state is prefilled once (`--multi-question sequential`); CUDA graphs up to 4,096 tokens; float32 recurrent state; the suffix-staging patch series optional (latency only) |
| Readout | Letters: each question is one prompt, and its distribution is the softmax over the option letters' logits at the last position |
| Rendering rules | An option with a description is shown as its description alone, without its key (`--describe-options`; `--no-describe-options` shows `key: description`); enumerated keys (`option_0`, `option_1`, ...) are hidden either way; bare snake_case labels are shown as words; the answer keys are the request's own |
| Temperature | T = 1.307 on the text route's plain readout (`softmax(log p / T)`); a registered task's correction replaces it; never changes the chosen option |
| Tie-break | Among exactly tied options, the key that sorts first in Unicode code-point order |
| Intent head | Single engine: the hidden state is read from the serving engine (three extra requests per head question); `--head-engine` reads it from a second engine instead |
| Image route | `--image-model`: the official checkpoint under the multimodal class as a second engine (0.49 of the card beside the text engine at 0.47), for requests that carry images; not tempered |
| Hardware | One NVIDIA RTX PRO 6000 Blackwell (96 GB) |
| Wire format | TypeSafe's System One (`POST /v1/systemone`), checked before every measurement by the serving gates C1 and the conformance gates C2-C4 |

## 2. Capabilities

- Question types: yes/no (`noul`), choice, and score (ordered levels, with an expected level and a confidence).
- Up to 255 options per question, each a key with an optional description.
- Any number of questions per request about one state (a string, an object or an array); the state is prefilled once and shared through the prefix cache, and each question is then scored in its own engine call, so its answer equals the same question sent alone. `--multi-question warm` scores them in one batch instead, for bulk scoring, where each answer depends on the batch (section 4).
- Task registration from labelled examples (`POST /v1/tasks`): per-task calibration from 10 examples (about 20 recommended), and an intent head on questions of 10 or more options with 5 examples per option; each is kept only when cross-validation on the examples shows a gain.
- An opt-in abstention threshold per task (`POST /v1/abstention/tasks`), on a declared or appended "can't tell" option.
- Image input: up to 2 images per request (PNG, JPEG or WebP, up to 20 MB each), as data URLs, multipart, or inside the state.
- Two-order averaging, opt-in (`--orders 2` or a request's `orders`), with the two orders' disagreement logged per question.
- Context up to 32,768 tokens per prompt.

## 3. Numbers

| Measure | Result | Record |
| --- | --- | --- |
| JevBench, 231 published items, harness accuracy | easy 1.000 (48), standard 0.972 (72), hard 0.685 (111); every answer schema-valid | `runs/2026-10-02_multi-question-and-rendering/jevbench_describe/*/summary.json` |
| JevBench, ECE / Brier per file | easy 0.022 / 0.002, standard 0.157 / 0.112, hard 0.069 / 0.401 | same |
| JevBench v1.5 open-set reading | choice 78.3 (easy 100, standard 100, hard 62.1); yes/no -10.9, 54% of yes/no answers between 0.20 and 0.80; score 60.6; I_open 42.7 (equal types), 51.6 (50/25/25) | `runs/2026-10-02_multi-question-and-rendering/jevbench_describe/v15.json` |
| JevBench latency, one request at a time | p50 47 to 60 ms, p95 49 to 106 ms by file | `.../summary.json` |
| Decision Index 0.2.1, BANKING77 (3,080) | macro-F1 0.731, accuracy 0.740, ECE 0.042 | `runs/2026-09-30_served-default/decision_index/default/di_report.json`, `di_cal.json` (the description rule changes none of its rows) |
| Decision Index 0.2.1, CLINC150+OOS (5,500) | macro-F1 0.814, accuracy 0.824, ECE 0.170 | same |
| Decision Index 0.2.1, GPQA Diamond (196 scored) | accuracy 0.490, ECE 0.116 | `runs/2026-10-02_multi-question-and-rendering/di_describe/di_report.json`, `di_cal.json` |
| Decision Index 0.2.1, MMLU-Pro (12,032) | accuracy 0.609, ECE 0.009 | same |
| 1,400-item suite, accuracy / ECE (neither default changes a row) | BoolQ 0.860 / 0.052, BANKING77 0.740 / 0.092, ToxicChat 0.940 / 0.032, MMLU 0.880 / 0.069, SciFact 0.740 / 0.125, SciFact clarified 0.820 / 0.081, MMLU-Pro 0.600 / 0.083 (150) and 0.657 / 0.050 (350); pooled 0.762 / 0.020 | `runs/2026-09-30_plugin-verification/conformance_H.json.gz` (the C4 rows) |
| Intent heads from 10 labelled examples per intent | BANKING77 0.847 (150 test items), CLINC150 0.893 (100 test items), means of three draws | `runs/2026-09-30_plugin-verification/intent_heads/` |
| Image input, ImajevBench v2.0-lite (254 labelled items) | 0.717 on all items, 0.791 on the 230 answerable; with "can't tell" offered 0.756 and 0.813 (5 of 24 unanswerable right, 2 false abstentions); p50 166 ms | `runs/2026-09-27_image-input/accuracy.json` |
| Latency, one question per request, state from the cache | 27.9 ms server time; 46.5 ms for an intent question, 126.8 ms with its head applied | `runs/2026-09-30_plugin-verification/manifest.json` (`latency.json.gz`) |
| Latency, one question, a state the server has never seen | 48.5 ms server time at 300 tokens, 51.3 at 1,000, 86.3 at 3,000 (median of 20); the padding and a second engine step at the block boundary account for most of it above the forward's own 26 to 67 ms | `runs/2026-10-02_multi-question-and-rendering/fresh_default.json`, `fresh_floor_cache_*.json` |
| Latency, four questions, a state never seen | 110.3 ms at 300 tokens, 112.3 at 1,000, 152.4 at 3,000; `--multi-question warm` 62.0, 78.4, 116.8; `--multi-question batch` 56.6, 58.5, 94.8 | `runs/2026-10-02_multi-question-and-rendering/fresh_sequential.json`, `fresh_default.json` (measured when warm was the default), `fresh_batch.json` |
| Latency, many questions per request (8,000-token state), `--multi-question warm` (batched) | 5.4 ms per question at 100 questions per request (5.4 ms with 20 such requests at once); 2.7 ms at 1,000 | `runs/2026-09-30_plugin-verification/bench_patched_on.json` |
| Cost per 1,000 decisions, at $1.50 per card-hour | $0.012 one question per request, one at a time; about $0.007 per further question in a request (about 17 ms each, the served default); with `--multi-question warm` $0.0022 at 100 questions per request under load and $0.0012 at 1,000 | same, and the latencies above |

ECE is over 10 equal-mass bins of the top probability; JevBench's ECE is its harness's own (10 equal-width bins).
The 2026-09-30 JevBench and Decision Index numbers were measured with the text-only view of the checkpoint, the 2026-10-02 ones with the served hidden-readout class; the two answer the 1,400-item suite bit-identically (`runs/2026-09-30_plugin-verification/suite_compare_H.txt`), and the served default of 2026-10-02 reproduced the 2026-09-30 JevBench answers exactly before the description rule was switched on (`runs/2026-10-02_multi-question-and-rendering/jevbench_default/`).
With `--temperature 1` every choice and every accuracy is the same; calibration changes, and with it the v1.5 yes/no and score values, which depend on the probabilities (`runs/2026-09-30_served-default`, arm `temperature_1`); T = 1 is a bit-exact no-op, so that arm's JevBench responses are byte for byte those of the untempered run recorded in `runs/2026-09-27_boards-baseline`.
None of these is a board number: JevBench's official score needs its sealed set, and a Decision Index value needs all 38 of its benchmarks.
Nothing was submitted, and no output of Jev (TypeSafe's hosted model behind the System One API, which JevBench is named after) is in this repository.
The records keep the wire names they were written with (`x-rlcd-*`, `rlcd-*/1`, the served name `rlcd-qwen3.6-35b-a3b-letters`); decisio reads both spellings (`decisio.names`).

## 4. What was fitted on what

| Component | Fitted or chosen on |
| --- | --- |
| Model weights | Nothing by us |
| Temperature T = 1.307 | Minimum log loss on the served readouts of the private 1,400-item suite above (eight tasks) |
| Rendering rules | Index keys and de-snaking: selected, with no parameter fitted, on the Decision Index's BANKING77 and CLINC150+OOS rows; the description rule: chosen on 272 steps of a browser-agent demo (wrong "done" on CLICK steps 63 to 12 of 132) and gated against the earlier rendering on JevBench 231 (−0.43 points [−2.60, +1.30]) and the Decision Index's MMLU-Pro (−0.07 [−0.52, +0.37]) and GPQA Diamond (+3.54 [−1.01, +8.08]); each checked to change nothing on the suite |
| Tie-break | A rule; nothing fitted |
| Intent heads and calibration priors | Per task, on 10 labelled training examples per intent (BANKING77, CLINC150), three draws |
| JevBench v1.5 thresholds and weights | JevBench's published method, as written |

- **The temperature's fit set is the suite in section 3,** so the suite's ECE figures are in-sample; 500 of its MMLU-Pro items and its 150 BANKING77 items are also in the Decision Index's pools (4.2% and 4.9%), in our wording rather than the board's, so T can move ECE on those two benchmarks, never a choice.
- **The rendering rules were selected on the Decision Index's intent rows,** so the BANKING77 and CLINC150+OOS scores measure a configuration chosen on them.
- **The intent heads' test items are also in the Decision Index's pools;** no head was registered during any Decision Index run.

Batch-forward finding: on this stack, questions scored in one batch are not the same forward pass as each scored alone. Identical requests sent in one batch differed by up to 0.59 in a label probability (median 0.029) on 250 intent items, while the same requests sent one at a time agreed to 1.2e-7 (`runs/2026-09-30_plugin-verification/diag_same_row.json`, `diag_same_row_sequential.json`).
So the served default scores each question of a request in its own engine call after the shared prefill (`--multi-question sequential`).
With the questions batched (`--multi-question warm`, the default until 2026-10-02), their probabilities vary between repeats on one server: about 0.12 on the README's three-question example, while each of its questions sent alone was identical 30 of 30 times (`runs/2026-10-01_docker-first-gpu-start/repeat_variability/`).
The choice can change when options are close: on 272 four-question requests from a browser-agent demo, the batched answer moved between 5 repeats on 31 requests and differed from the same question sent alone on 50, by up to 0.27 (`runs/2026-10-02_multi-question-and-rendering/`).
The dependence is in vLLM 0.30.0's forward for this model whenever more than one sequence shares an engine step. It appears in the bf16 checkpoint as in FP8, and with prefix caching off, the GDN prefill on Triton, CUDA graphs off and cuBLAS's deterministic workspace. It disappears with one sequence per step, and it reproduces on stock vLLM with no decisio code: alone against a batch of two, up to 0.198 in log-probability (`batch_diagnosis/` in the same record); the same batch of two, repeated, differs by up to 0.168, while a lone prompt repeats exactly.
This model's answers move at that scale with any change of rounding, not only the batch: the same lone row prefilled in one step and split at the 1,056-token block boundary differs by up to 0.44 in a probability (median 0.068), so the bit-identity claims here hold for a fixed configuration and request form.
Scored one engine call each, every answer equals the question sent alone: 272 of 272 requests, 5 repeats each, max |dp| 0. The cost is about 17 ms per question instead of 5 to 9 batched (four questions on a fresh 300-token state: 110 ms against 62). Single-question requests are the same in every mode, bit for bit on the 1,400-item suite. The issue is reported upstream (vllm-project/vllm#59764).
Across server restarts, single-question answers matched a record from two days earlier on all 1,400 suite items in 4 of 5 starts; the fifth moved 2 items by up to 0.0012, with no choice changed.
The engine's logits are bf16, and label probabilities recomputed from the hidden state match the engine's within 3e-8 only when rounded to bf16 (`diag_same_forward.json`).
The single-engine head therefore sends its three requests one at a time and recomputes in bf16, and its same-forward gate passes at 3.1e-8 (`same_forward.json`).
Every harness in `runs/` sends one single-question request at a time, the serving gate G2 bounded the effect of 16 extra questions in a batched request at 6.46e-2 with no change of choice (`gates.log`; with the questions scored one call each it is 0 by construction), and every bit-identity claim compares single-question requests under like request histories.

## 5. Limits

- One card per run and one run per configuration.
- JevBench's v1.5 reading covers 231 of the board's 904 open items, with no judge tier and no sealed half; its published items have been public since v1.2 and may be in any model's pretraining data.
- The Decision Index rows were rebuilt with the kit for four benchmarks; their counts and the rows' sha256 match (`decisio.bench.di_rows`), but a partial rebuild cannot be checked against the whole suite's hash.
- The ImajevBench record was made without the rendering rules and the key-order tie-break; 2 of its 254 items ended in exact ties that the harness rejected and are counted wrong.
- The second-engine head mode with the registered text-only class is unmeasured for latency.
- Latency is server-side on the card's localhost; cost is the card-hour price divided by measured throughput, with nothing else counted.
- The per-item Decision Index records keep each request's id, the payload's sha256 and our response, not the item text (GPQA's authors ask that its items not be published in plain text).
