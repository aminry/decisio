<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Evaluation card

What the served default is, what it can do, how it measures, and what was fitted on what.
Sections 1 to 5 are the served default, the Qwen base; section 6 is the second base, Gemma 4 12B (`--base gemma-4-12b`); section 7 is the third, Gemma 4 31B (`--base gemma-4-31b`); section 8 is the Decision Index's whole suite on all three, self-run and submitted.
Every number is on the current served default and names the record it comes from; each run under `runs/` has a `manifest.json` with its configuration and harness versions and a `files.json` with every file's sha256.

## 1. The system measured

| Part | The served default |
| --- | --- |
| Checkpoint | `Qwen/Qwen3.6-35B-A3B-FP8`, the official weights, untrained by us: no adapter, no fine-tuning |
| Provenance | Official checkpoint from Alibaba's Qwen team, at a pinned revision; no adapter or fine-tuning by us (revision `95a723d08a9490559dae23d0cff1d9466213d989`) |
| Engine | vLLM 0.30.0 with decisio's plugin: the official checkpoint under the hidden-readout class (the text model without its vision tower, plus the hidden state at the answer position); DeepGEMM off; front padding to the 1,056-token block; the questions of a request scored one engine call each after the state is prefilled once (`--multi-question sequential`); CUDA graphs up to 4,096 tokens; float32 recurrent state; the suffix-staging patch series optional (latency only) |
| Readout | Letters: each question is one prompt, and its distribution is the softmax over the option letters' logits at the last position |
| Prompt | Since 2026-10-03, the spaced layout (`--prompt-tail spaced`): no system prompt; the question's instructions, a blank line, the lettered options, a blank line, and one closing line asking for the chosen option's letter alone (`decisio.readout.spaced`); a yes/no question asked as a two-option letter choice, the false side first, the sides named ("No: ...", "Yes: ...", `--noul-rendering letters-keys`). The image route keeps the compact layout (no blank lines, "Answer with the letter only.", yes/no in words) |
| Rendering rules | An option with a description is shown as its description alone, without its key (`--describe-options`; `--no-describe-options` shows `key: description`); enumerated keys (`option_0`, `option_1`, ...) are hidden either way; bare snake_case labels are shown as words; the answer keys are the request's own |
| Temperature | On the text route's plain readout (`softmax(log p / T)`): T = 1.370 for choice questions, T = 1.506 for yes/no and score questions (1.307 for all under the compact layout); a registered task's correction replaces it; never changes the chosen option |
| Tie-break | Among exactly tied options, the key that sorts first in Unicode code-point order |
| Intent head | Single engine: the hidden state is read from the serving engine (three extra requests per head question); `--head-engine` reads it from a second engine instead |
| Image route | `--image-model`: the official checkpoint under the multimodal class as a second engine (0.49 of the card beside the text engine at 0.47), for requests that carry images; not tempered |
| Hardware | One NVIDIA RTX PRO 6000 Blackwell (96 GB) |
| Wire format | TypeSafe's System One (`POST /v1/systemone`), checked before every measurement by the serving gates C1 and the conformance gates C2-C4 |

## 2. Capabilities

- Question types: yes/no (`noul`), choice, and score (ordered levels, with an expected level and a confidence).
- Up to 255 options per question, each a key with an optional description.
- Any number of questions per request about one state (a string, an object or an array); the state is prefilled once and shared through the prefix cache, and each question is then scored in its own engine call, so its answer equals the same question sent alone.
  `--multi-question warm` scores them in one batch instead, for bulk scoring, where each answer depends on the batch (section 4).
- Task registration from labelled examples (`POST /v1/tasks`): per-task calibration from 10 examples (about 20 recommended), and an intent head on questions of 10 or more options with 5 examples per option; each is kept only when cross-validation on the examples shows a gain.
- An opt-in abstention threshold per task (`POST /v1/abstention/tasks`), on a declared or appended "can't tell" option.
- Image input: up to 2 images per request (PNG, JPEG or WebP, up to 20 MB each), as data URLs, multipart, or inside the state.
- Two-order averaging, opt-in (`--orders 2` or a request's `orders`), with the two orders' disagreement logged per question.
- Context up to 32,768 tokens per prompt.

## 3. Numbers

| Measure | Result | Record |
| --- | --- | --- |
| JevBench, 231 published items, harness accuracy | easy 1.000 (48), standard 0.972 (72), hard 0.739 (111); every answer schema-valid (compact layout: hard 0.685) | `runs/2026-10-03_qiv-default/` `jevbench_qiv/*/summary.json` |
| JevBench, ECE / Brier per file | easy 0.014 / 0.001, standard 0.121 / 0.075, hard 0.043 / 0.367 (compact layout: 0.022 / 0.002, 0.157 / 0.112, 0.069 / 0.401) | `runs/2026-10-03_qiv-default/` `served_qiv.json` (the harness's own metric on the answers re-tempered to the served temperatures) |
| JevBench v1.5 open-set reading | choice 77.3; yes/no 13.7, 39% of yes/no answers between 0.20 and 0.80; score 57.2; I_open 49.4 (equal types) (compact layout: 78.3, -10.9 with 54% between, 60.6, 42.7) | `runs/2026-10-03_qiv-default/` `served_qiv.json` |
| JevBench latency, one request at a time | p50 47 to 60 ms, p95 49 to 106 ms by file | `.../summary.json` |
| Decision Index 0.2.1, BANKING77 (3,080) | macro-F1 0.746, accuracy 0.755, ECE 0.008 (compact layout: 0.731, 0.740, 0.042) | `runs/2026-10-03_qiv-default/` `di_qiv/di_report.json`, `served_qiv.json` |
| Decision Index 0.2.1, CLINC150+OOS (5,500) | macro-F1 0.822, accuracy 0.827, ECE 0.099 (compact layout: 0.814, 0.824, 0.170) | same |
| Decision Index 0.2.1, GPQA Diamond (196 scored) | accuracy 0.510, ECE 0.115 (compact layout: 0.490, 0.116) | same |
| Decision Index 0.2.1, MMLU-Pro (12,032) | accuracy 0.613, ECE 0.013 (compact layout: 0.609, 0.009) | same |
| 1,400-item suite, accuracy / ECE | BoolQ 0.860 / 0.092, BANKING77 0.747 / 0.055, ToxicChat 0.960 / 0.029, MMLU 0.873 / 0.057, SciFact 0.793 / 0.105, SciFact clarified 0.853 / 0.070, MMLU-Pro 0.613 / 0.061 (150) and 0.637 / 0.022 (350); pooled 0.770 / 0.033 (compact layout: pooled 0.762 / 0.020) | `runs/2026-10-03_qiv-default/` `served_qiv.json`, `capture_qiv.json.gz` |
| Intent heads from 10 labelled examples per intent | BANKING77 0.840 (150 test items), CLINC150 0.912 (100 test items), means of six draws (compact layout, same six draws: 0.849, 0.900) | `runs/2026-10-03_qiv-default/` `heads6.json`, `intents_*.json.gz` |
| Image input, ImajevBench v2.0-lite (254 labelled items) | 0.717 on all items, 0.791 on the 230 answerable; with "can't tell" offered 0.756 and 0.813 (5 of 24 unanswerable right, 2 false abstentions); p50 166 ms | `runs/2026-09-27_image-input/accuracy.json` |
| Latency, one question per request, state from the cache | 27.9 ms server time; 46.5 ms for an intent question, 126.8 ms with its head applied | `runs/2026-09-30_plugin-verification/manifest.json` (`latency.json.gz`) |
| Latency, one question, a state the server has never seen | 48.5 ms server time at 300 tokens, 51.3 at 1,000, 86.3 at 3,000 (median of 20); the padding and a second engine step at the block boundary account for most of it above the forward's own 26 to 67 ms | `runs/2026-10-02_multi-question-and-rendering/fresh_default.json`, `fresh_floor_cache_*.json` |
| Latency, four questions, a state never seen | 110.3 ms at 300 tokens, 112.3 at 1,000, 152.4 at 3,000; `--multi-question warm` 62.0, 78.4, 116.8; `--multi-question batch` 56.6, 58.5, 94.8 | `runs/2026-10-02_multi-question-and-rendering/fresh_sequential.json`, `fresh_default.json` (measured when warm was the default), `fresh_batch.json` |
| Latency, many questions per request (8,000-token state), `--multi-question warm` (batched) | 5.4 ms per question at 100 questions per request (5.4 ms with 20 such requests at once); 2.7 ms at 1,000 | `runs/2026-09-30_plugin-verification/bench_patched_on.json` |
| Cost per 1,000 decisions, at $1.50 per card-hour | $0.012 one question per request, one at a time; about $0.007 per further question in a request (about 17 ms each, the served default); with `--multi-question warm` $0.0022 at 100 questions per request under load and $0.0012 at 1,000 | same, and the latencies above |

ECE is over 10 equal-mass bins of the top probability; JevBench's ECE is its harness's own (10 equal-width bins).
The prompt changed on 2026-10-03; the rows above give both, and the latency rows were measured under the compact layout: on one card in one session the spaced layout measured within 3 ms of it (+0.7 ms on a cached state, 0.0 to +0.5 ms for one question on a new state, +1.8 to +2.9 ms for four; `runs/2026-10-03_qiv-default/` `latency_*.json`).
The image route keeps the compact layout, so its row is unchanged.
The 2026-09-30 JevBench and Decision Index numbers were measured with the text-only view of the checkpoint, the 2026-10-02 ones with the served hidden-readout class; the two answer the 1,400-item suite bit-identically (`runs/2026-09-30_plugin-verification/suite_compare_H.txt`), and the served default of 2026-10-02 reproduced the 2026-09-30 JevBench answers exactly before the description rule was switched on (`runs/2026-10-02_multi-question-and-rendering/jevbench_default/`).
With `--temperature 1` every choice and every accuracy is the same; calibration changes, and with it the v1.5 yes/no and score values, which depend on the probabilities (`runs/2026-09-30_served-default`, arm `temperature_1`); T = 1 is a bit-exact no-op, so that arm's JevBench responses are byte for byte those of the untempered run recorded in `runs/2026-09-27_boards-baseline`.
None of these is a board number: JevBench's official score needs its sealed set, and a Decision Index value needs all 38 of its benchmarks, which section 8 runs.
Nothing was submitted to JevBench's board; the Decision Index runs of section 8 were self-run and submitted for review.
No output of Jev (TypeSafe's hosted model behind the System One API, which JevBench is named after) is in this repository; section 8 quotes two of its values from the Decision Index's public board.
The records keep the wire names they were written with (`x-rlcd-*`, `rlcd-*/1`, the served name `rlcd-qwen3.6-35b-a3b-letters`); decisio reads both spellings (`decisio.names`).

## 4. What was fitted on what

| Component | Fitted or chosen on |
| --- | --- |
| Model weights | Nothing by us |
| Temperatures T = 1.506 and choice T = 1.370 | Minimum log loss on the served prompt's plain readouts of the private 1,400-item suite above (eight tasks; the choice T on its 800 choice items), each checked by 5-fold cross-validation; never on JevBench or the Decision Index |
| The prompt (the spaced layout, yes/no as named letters) | Chosen after measuring it against the compact layout on the suite, JevBench 231, the four Decision Index benchmarks, conformance, latency and the intent heads, under a pre-registration amended once to retest the heads with six draws |
| Rendering rules | Index keys and de-snaking: selected, with no parameter fitted, on the Decision Index's BANKING77 and CLINC150+OOS rows; the description rule: chosen on 272 steps of a browser-agent demo (wrong "done" on CLICK steps 63 to 12 of 132) and gated against the earlier rendering on JevBench 231 (−0.43 points [−2.60, +1.30]) and the Decision Index's MMLU-Pro (−0.07 [−0.52, +0.37]) and GPQA Diamond (+3.54 [−1.01, +8.08]); each checked to change nothing on the suite |
| Tie-break | A rule; nothing fitted |
| Intent heads and calibration priors | Per task, on 10 labelled training examples per intent (BANKING77, CLINC150), three draws |
| JevBench v1.5 thresholds and weights | JevBench's published method, as written |

- **The temperature's fit set is the suite in section 3,** so the suite's ECE figures are in-sample; 500 of its MMLU-Pro items and its 150 BANKING77 items are also in the Decision Index's pools (4.2% and 4.9%), in our wording rather than the board's, so T can move ECE on those two benchmarks, never a choice.
- **The rendering rules were selected on the Decision Index's intent rows,** so the BANKING77 and CLINC150+OOS scores measure a configuration chosen on them.
- **The prompt was chosen with JevBench and the Decision Index among its gates,** so those numbers measure a configuration selected partly on them.
- **The intent heads' test items are also in the Decision Index's pools;** no head was registered during any Decision Index run.

Batch-forward finding: on this stack, questions scored in one batch are not the same forward pass as each scored alone.
Identical requests sent in one batch differed by up to 0.59 in a label probability (median 0.029) on 250 intent items, while the same requests sent one at a time agreed to 1.2e-7 (`runs/2026-09-30_plugin-verification/diag_same_row.json`, `diag_same_row_sequential.json`).
So the served default scores each question of a request in its own engine call after the shared prefill (`--multi-question sequential`).
With the questions batched (`--multi-question warm`, the default until 2026-10-02) and vLLM's engine in a process of its own (vLLM's arrangement, `--engine-process separate`, the default), their probabilities vary between repeats on one server: about 0.12 on the README's three-question example, while each of its questions sent alone was identical 30 of 30 times (`runs/2026-10-01_docker-first-gpu-start/repeat_variability/`); on 272 four-question requests from a browser-agent demo, 22 repeated exactly over 5 repeats and a choice changed on 52 (`runs/2026-10-05_engine-process/`; 31 requests moved on its operation question in `runs/2026-10-02_multi-question-and-rendering/`).
That variation comes from the separate engine process: it receives a batch's requests one at a time and starts an engine step with whatever has arrived, so a request's questions do not always share a step (vllm-project/vllm#59764).
With the engine in the server's process (`--engine-process in`), every question is added before the first step, and an identical batched request repeats exactly: 272 of 272 four-question requests, 5 repeats each, also with four concurrent clients sending different requests (`runs/2026-10-05_engine-process/`).
A batched answer is still not the same forward pass as the question sent alone: on those requests all 272 differ from each question sent alone on three of the four questions, by up to 0.39, and 11 to 25 choices change per question; scored against the demo's acceptable actions, accuracy batched against alone is +1.5 points [-1.1, +4.0] on the operation and -3.5 [-7.5, +0.0] on the click target.
The dependence is in vLLM 0.30.0's forward for this model whenever more than one sequence shares an engine step.
It appears in the bf16 checkpoint as in FP8, and with prefix caching off, the GDN prefill on Triton, CUDA graphs off and cuBLAS's deterministic workspace.
It disappears with one sequence per step, and it reproduces on stock vLLM with no decisio code: alone against a batch of two, up to 0.198 in log-probability (`batch_diagnosis/` in the same record); the same batch of two, repeated, differs by up to 0.168 with the engine in its own process and not at all with it in the server's process (0.145 and 0.0, the FP8 and bf16 checkpoints alike, `runs/2026-10-05_engine-process/`), while a lone prompt repeats exactly.
This model's answers move at that scale with any change of rounding, not only the batch: the same lone row prefilled in one step and split at the 1,056-token block boundary differs by up to 0.44 in a probability (median 0.068), so the bit-identity claims here hold for a fixed configuration and request form.
Scored one engine call each, every answer equals the question sent alone: 272 of 272 requests, 5 repeats each, max |dp| 0.
The cost is about 17 ms per question instead of 5 to 9 batched (four questions on a fresh 300-token state: 110 ms against 62).
Single-question requests are the same in every mode, bit for bit on the 1,400-item suite.
The issue is reported upstream (vllm-project/vllm#59764).
Across server restarts, single-question answers matched a record from two days earlier on all 1,400 suite items in 4 of 5 starts; the fifth moved 2 items by up to 0.0012, with no choice changed.
The engine's logits are bf16, and label probabilities recomputed from the hidden state match the engine's within 3e-8 only when rounded to bf16 (`diag_same_forward.json`).
The single-engine head therefore sends its three requests one at a time and recomputes in bf16, and its same-forward gate passes at 3.1e-8 (`same_forward.json`).
Every harness in `runs/` sends one single-question request at a time, the serving gate G2 bounded the effect of 16 extra questions in a batched request at 6.46e-2 with no change of choice (`gates.log`; with the questions scored one call each it is 0 by construction), and every bit-identity claim compares single-question requests under like request histories.

## 5. Limits

- One card per run and one run per configuration.
- JevBench's v1.5 reading covers 231 of the board's 904 open items, with no judge tier and no sealed half; its published items have been public since v1.2 and may be in any model's pretraining data.
- The Decision Index rows of sections 3, 6 and 7 were rebuilt with the kit for four benchmarks; their counts and the rows' sha256 match (`decisio.bench.di_rows`), but a partial rebuild cannot be checked against the whole suite's hash.
  Section 8's runs used the whole suite, rebuilt with the kit and matched against the lab's hashes.
- The ImajevBench record was made without the rendering rules and the key-order tie-break; 2 of its 254 items ended in exact ties that the harness rejected and are counted wrong.
- The second-engine head mode with the registered text-only class is unmeasured for latency.
- Latency is server-side on the card's localhost; cost is the card-hour price divided by measured throughput, with nothing else counted.
- The per-item Decision Index records keep each request's id, the payload's sha256 and our response, not the item text (GPQA's authors ask that its items not be published in plain text).

## 6. The Gemma base

`--base gemma-4-12b` serves Gemma 4 12B behind the same routes and wire format; the README's "Choosing a base" compares the two bases.
Every number in this section comes from one session on one card (2026-10-04, `runs/2026-10-04_gemma-base/`), where the Qwen base was measured beside it on the same code, card and items.

### 6.1 The system measured

| Part | The Gemma base |
| --- | --- |
| Checkpoint | `google/gemma-4-12B-it` at revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, bf16, the official weights, untrained by us |
| Provenance | Official checkpoint from Google, at a pinned revision; no adapter or fine-tuning by us |
| Engine | vLLM 0.30.0 with decisio's plugin: the checkpoint's encoder-free class `Gemma4UnifiedForConditionalGeneration`, served through its text path with every multimodal input off, under decisio's hidden-readout subclass (the hidden state at the answer position, written after the final-logit soft cap of 30); the TRITON_ATTN attention backend, which vLLM picks for the model's two head sizes; no padding; `--multi-question sequential` |
| Readout | Letters, as on the Qwen base, with every single-token form of each option letter (with a leading space, bare, and as a byte) summed into the letter's probability |
| Prompt | A system turn (`decisio.readout.system_prompt`), then the spaced layout, read at the chat template's own answer position; a yes/no question asked as a two-option letter choice, the false side first, each side shown as its description (`--noul-rendering letters`) |
| Temperature | T = 3.592 for every question type, on the plain readout; no separate choice temperature |
| Intent head | Single engine; the head takes its label log-probabilities from the engine's own readout of the question (one more request), so a head is fitted and served on exactly the plain readout's probabilities |
| Hardware | One NVIDIA RTX PRO 6000 Blackwell (96 GB); the weights take 22.8 GiB, leaving 57.4 GiB for the KV cache at vLLM's 0.90 share |
| `/health` | The `profile` block quotes the base, the checkpoint and revision, the temperature per question type and the prompt; `cache_hit_unit` 64 and `hash_unit` 16: vLLM keeps this model's cache in groups of 16- and 64-token blocks, hashes prefixes every 16 tokens, and a hit must hold in every group, so hits come in 64-token steps |

The capabilities of section 2 apply, with these differences: several questions in one request return the same choice as each question sent alone, with probabilities within 0.035 rather than bit for bit (6.3); the image route was not measured with this base.

### 6.2 Numbers

| Measure | Result | Record |
| --- | --- | --- |
| JevBench, 231 published items, harness accuracy | easy 1.000 (48), standard 0.972 (72), hard 0.739 (111): 200 correct, as the Qwen base in the same session (0.0 points [-4.3, +4.3], paired) | `jevbench_gemma/*/summary.json`, `report.json` |
| JevBench, ECE per file | easy 0.023, standard 0.033, hard 0.085 | `jevbench_gemma/*/summary.json` |
| JevBench v1.5 open-set reading | choice 80.6; yes/no 49.0, 15% of yes/no answers between 0.20 and 0.80; score 63.7; I_open 64.4 (equal types), +15.0 [+7.0, +23.6] against the Qwen base, paired | `jevbench_gemma/v15.json`, `report.json` |
| JevBench latency, one request at a time | p50 23.5 to 28.1 ms by file | `jevbench_gemma/*/summary.json` |
| Decision Index 0.2.1, BANKING77 (3,080) | macro-F1 0.729, accuracy 0.741, ECE 0.042 | `di_gemma/di_report.json`, `report.json` |
| Decision Index 0.2.1, CLINC150+OOS (5,500) | macro-F1 0.871, accuracy 0.872, ECE 0.079 | same |
| Decision Index 0.2.1, GPQA Diamond (196 scored) | accuracy 0.378, ECE 0.165 (over the 198 rows) | same |
| Decision Index 0.2.1, MMLU-Pro (12,032) | accuracy 0.549, ECE 0.034 | same |
| 1,400-item suite, accuracy / ECE | BoolQ 0.873 / 0.061, BANKING77 0.747 / 0.059, ToxicChat 0.947 / 0.023, MMLU 0.827 / 0.084, SciFact 0.833 / 0.099, SciFact clarified 0.880 / 0.071, MMLU-Pro 0.547 / 0.101 (150) and 0.517 / 0.070 (350); pooled 0.735 / 0.029 | `capture_gemma.json.gz`, `report.json` |
| Intent heads from 10 labelled examples per intent | BANKING77 0.832 (plain readout 0.747), CLINC150 0.908 (0.850), means of six draws; registration 195 s and 441 s | `intents_gemma_*.json.gz`, `report.json` |
| Latency, one question, state from the cache | 26.7 ms server time (median of 20) | `latency_gemma.json` |
| Latency, one question, a state never seen | 39.1 ms at 300 tokens, 102.4 at 1,000, 293.5 at 3,000 | same |
| Latency, four questions, a state never seen | 131.0 ms at 300 tokens, 195.8 at 1,000, 391.6 at 3,000 | same |

ECE is as in section 3; the suite's is the tie-robust version over 10 equal-mass bins.
The Qwen base measured in the same session reproduced its record in section 3: 200 correct on JevBench, I_open 49.4, suite 0.770 / 0.033, and its plain readouts chose the record's answer on 1,400 of 1,400 suite items (largest difference 1.1e-16).

### 6.3 Gates

Each claim above was gated before it was written, in a pre-registration fixed before the session:
- Conformance C2, C3 and C4 pass on the 1,400 items (C4 max |dp| 0).
- The intent head is exact: on the 250 intent items, read under like request histories, its label log-probabilities equal the log of the plain readout's probabilities bit for bit (250 of 250; `samelog_gemma.json`).
  The session's own check compared the exponential of those log-probabilities with the probabilities and read 5.55e-17, the rounding of that conversion (`sameforward_gemma.json`).
- Intent heads gain over the plain readout across six draws, +8.6 points on BANKING77 and +5.8 on CLINC150, and BANKING77's first draw, run twice, returned the same rows.
- Several questions in one request: on 272 four-question requests from a browser-agent demo, 5 repeats each, every request chose what each question chose alone; the largest probability difference was 0.035 (`travel_gemma.json.gz`).
- Prefix sharing: in four-question requests on states of 300, 1,000 and 3,000 tokens, questions 2 to 4 read at least the shared prefix minus 64 tokens from the cache (9 of 9; `cache_gemma.json`).
- The feature checks (question types and states, summed label forms, sequential scoring, two-order mode, task registration, abstention) pass (`features_gemma.json`).

### 6.4 What was fitted on what

| Component | Fitted or chosen on |
| --- | --- |
| Model weights | Nothing by us |
| Temperature T = 3.592 | Minimum log loss on this prompt's plain readouts of the private 1,400-item suite of section 3, with 5-fold cross-validation stratified by task, in an earlier session with token rows identical to these on JevBench's 231 items; recomputed on this session's readouts it is 3.608, and a separate choice temperature is not supported (out-of-fold log loss -0.0012 [-0.0051, +0.0029]; `temps_gemma_refit.json`) |
| The prompt (the system turn, the answer position, summed forms, yes/no as letters) | Chosen in an earlier session by ablation on the suite and JevBench 231, then tested in this session against the Qwen base's prompt on this model, its temperature fitted the same way (3.396): suite accuracy +0.14 points [-1.21, +1.43], JevBench hard tier +3.6 [-1.8, +9.0], suite ECE 0.041 against 0.029, JevBench choice ECE 0.074 against 0.064; level on accuracy, worse on calibration, so the system turn stays (`gemma_q.json`, `temps_gemma_q.json`) |
| Intent heads and calibration priors | Per task, on 10 labelled training examples per intent, six draws |

The disclosures of section 4 hold here too: the suite is the temperature's fit set, so its ECE figures are in-sample, and the prompt was chosen with JevBench among its measurements.

### 6.5 Reproducibility

Unlike the Qwen base, this base does not return the same probabilities bit for bit for a repeated question.
Its final logits are bf16 after the soft cap, so near the top they lie on a grid of 0.0625 to 0.125 (every one of 212 spacings between leading labels in this session's JevBench readouts is a multiple of 1/16), and a state read for the first time and the same state read from the prefix cache can land a step apart.
Within the session, the JevBench answers read on first contact and the same questions asked again later differed by up to 0.035 at the served temperature, on 49 of 231 items, and no choice changed; on the Qwen base the two were identical.
Between this session and an earlier one on another card of the same type, with the same prompts and temperature, the answers differed by up to 0.128, on 212 of 231 items; two near-tied choices changed, one of them from right to wrong (`repro.json`).
Several questions in one request stay within the same bound (0.035, 6.3).
With the questions batched (`--multi-question warm`) and the engine in the server's process, a repeated request moves only as a single question does, between a state's first read and later reads from the prefix cache: on 272 four-question travel requests the first answer differed from the next four, which agreed, on 204, by up to 0.046, with one choice changed (`runs/2026-10-05_engine-process/`).

### 6.6 Limits

- One card and one session; the Qwen base's numbers beside it are from the same session.
- The Gemma base does not fit a 32 GB card on vLLM's defaults.
  Simulated on the 96 GB card with the engine's share cut to 28.8 GB (vLLM's 0.90 of 32 GB), it did not start at 32,768 or at 16,384 tokens of context: its 22.8 GiB of weights left no room for the KV cache.
  Its path to 32 GB machines is an MLX build, in preparation.
- The image route and `--pad-policy row` were not measured on this base; `--multi-question warm` only on the 272 travel requests of section 6.5.
- The limits of section 5 on JevBench, the Decision Index rows and the board apply as written; nothing was submitted to JevBench's board, and the Decision Index's whole suite was self-run and submitted (section 8).

## 7. The Gemma 4 31B base

`--base gemma-4-31b` serves Gemma 4 31B behind the same routes and wire format.
Every number in this section comes from one session on one card (2026-10-04, `runs/2026-10-04_gemma-4-31b/`), a different session and card from sections 3 and 6, so its comparisons with the other bases are not paired.

### 7.1 The system measured

| Part | The Gemma 4 31B base |
| --- | --- |
| Checkpoint | `google/gemma-4-31B-it` at revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, the official weights, untrained by us |
| Precision | Quantized to FP8 by vLLM 0.30.0 when it loads (`quantization="fp8"`, part of the base's profile); a pinned FP8 checkpoint replaces this when one exists |
| Engine | vLLM 0.30.0 with decisio's plugin: vLLM's `Gemma4ForCausalLM`, the text class of the `Gemma4ForConditionalGeneration` checkpoint, which loads the language model and leaves the vision tower out (`DecisioGemma4TextOnly`), and the same class returning the hidden state after the final-logit soft cap of 30 (`DecisioGemma4HiddenReadout`); no padding; `--multi-question sequential` |
| Prompt and readout | As the Gemma 4 12B base (section 6.1): a system turn, the spaced layout read at the chat template's own answer position, every single-token form of each option letter summed, yes/no as a two-option letter choice with each side shown as its description |
| Temperature | T = 4.672 for choice questions, T = 5.252 for yes/no and score questions, on the plain readout |
| Intent head | Single engine; the head takes its label log-probabilities from the engine's own readout of the question, as on the 12B; its reserved ids start at 210,000 |
| Hardware | One NVIDIA RTX PRO 6000 Blackwell (96 GB); the FP8 weights take 30.61 GiB, leaving 48.6 GiB for the KV cache at vLLM's 0.90 share (57,882 tokens) |
| `/health` | The `profile` block, with `quantization_on_load` "fp8"; `cache_hit_unit` 32 and `hash_unit` 16 |

The capabilities of section 2 apply; the image route was not measured with this base, and it is not served on a Mac (`--backend mlx` refuses it).

### 7.2 Numbers

| Measure | Result | Record |
| --- | --- | --- |
| JevBench, 231 published items, accuracy | easy 1.000 (48), standard 1.000 (72), hard 0.838 (111): 213 correct | `capture_g31.json.gz`, `jevbench_g31/` |
| JevBench, ECE per file (10 equal-width bins, at the served temperatures) | easy 0.020, standard 0.035, hard 0.091 | `capture_g31.json.gz` |
| JevBench v1.5 open-set reading | choice 86.3; yes/no 85.6, 1.4% of yes/no answers between 0.20 and 0.80; score 71.9; I_open 81.3 (equal types) | same |
| Decision Index 0.2.1, BANKING77 (3,080) | macro-F1 0.785, accuracy 0.792 | `di_g31/di_report.json` |
| Decision Index 0.2.1, CLINC150+OOS (5,500) | macro-F1 0.902, accuracy 0.901 | same |
| Decision Index 0.2.1, GPQA Diamond (196 scored) | accuracy 0.520 | same |
| Decision Index 0.2.1, MMLU-Pro (12,032) | accuracy 0.694 | same |
| 1,400-item suite, accuracy / ECE | BoolQ 0.887 / 0.096, BANKING77 0.787 / 0.078, ToxicChat 0.973 / 0.036, MMLU 0.887 / 0.064, SciFact 0.827 / 0.140, SciFact clarified 0.840 / 0.134, MMLU-Pro 0.627 / 0.095 (150) and 0.700 / 0.058 (350); pooled 0.799 / 0.054 | `capture_g31.json.gz`, `temps_g31.json` |
| Intent heads from 10 labelled examples per intent | BANKING77 0.844 (plain readout 0.787), CLINC150 0.970 (0.940), means of three draws; registration 308 s and 699 s | `intents_g31.json.gz` |
| Latency, one question, state from the cache | 40.8 ms server time (median of 20) | `latency_g31.json` |
| Latency, one question, a state never seen | 61.8 ms at 300 tokens, 170.1 at 1,000, 476.1 at 3,000 | same |
| Latency, four questions, a state never seen | 182.2 ms at 300 tokens, 282.3 at 1,000, 614.5 at 3,000 | same |

ECE is as in section 3; the suite's is the tie-robust version over 10 equal-mass bins.

### 7.3 Gates, and the decision

Measured in the session: conformance C2, C3 and C4 pass on the 1,400 items; on the 250 intent items the head's label log-probabilities equal the log of the plain readout's probabilities bit for bit (250 of 250); each of JevBench's 231 items sent twice in a row moved by at most 5.6e-8, and no choice changed (`repeat_g31.json.gz`).

The session's pre-registered go rule compared the four Decision Index values with the board's "Decider chat · Gemma-4-31B" row (BANKING77 macro-F1 0.791, CLINC150+OOS macro-F1 0.912, GPQA Diamond 0.490, MMLU-Pro 0.696) and one question on a new 300-token state with 108 ms.
The 31B passed GPQA Diamond (0.520) and the latency (61.8 ms), and missed BANKING77 by 0.0065, CLINC150+OOS by 0.0105 and MMLU-Pro by 0.0021.
**It is a base by the maintainer's decision, overriding that rule.**
The reason: the thresholds were one board row taken as a proxy for the top of the open board; the three misses, 0.2 to 1.1 points, lie within each benchmark's sampling noise; and against the two served bases it is stronger on every accuracy measure:

| Measure | Gemma 4 31B | Qwen3.6-35B-A3B | Gemma 4 12B |
| --- | ---: | ---: | ---: |
| Suite accuracy | 0.799 | 0.770 | 0.735 |
| JevBench correct; hard tier | 213; 93 | 200; 82 | 200; 82 |
| Decision Index BANKING77 / CLINC150+OOS, macro-F1 | 0.785 / 0.902 | 0.746 / 0.822 | 0.729 / 0.871 |
| Decision Index GPQA Diamond / MMLU-Pro, accuracy | 0.520 / 0.694 | 0.510 / 0.613 | 0.378 / 0.549 |
| One question, a new 300 / 3,000-token state | 61.8 / 476.1 ms | 48.4 / 91.1 ms | 39.1 / 293.5 ms |

The Qwen and 12B columns are section 3's and section 6's records, from another session on another card of the same type; nothing in this table is paired.

### 7.4 What was fitted on what

| Component | Fitted or chosen on |
| --- | --- |
| Model weights | Nothing by us |
| Temperatures T = 5.252 and choice T = 4.672 | Minimum log loss on this base's plain readouts of the private 1,400-item suite of section 3 (the choice T on its 800 choice items, kept because out of fold it lowers their log loss: −0.0115 [−0.0211, −0.0015]), each with 5-fold cross-validation stratified by task, T searched in [0.2, 50]; never on JevBench or the Decision Index |
| The prompt | The 12B base's, unchanged (section 6.4) |
| FP8 on load | Chosen because at bf16 the checkpoint does not start at a 32,768-token context on one 96 GB card (7.5) |

The disclosures of section 4 hold here too, with one more: the go rule this base was measured against was set before the session, and the decision to serve it was taken after seeing its numbers (7.3).

### 7.5 Limits

- At bf16 the checkpoint does not serve a 32,768-token context on one 96 GB card: its weights (57.91 GiB) leave 20.24 GiB for the KV cache at vLLM's 0.90 share, while one 32,768-token request needs 27.51 GiB; vLLM estimates 24,096 tokens as the most that fits (`bf16_start.txt`), hence FP8 on load.
- Several questions in one request, scored one engine call each as on the other bases, equalled each question sent alone on all 272 travel requests; batched (`--multi-question warm`, engine in the server's process) they repeated exactly and matched each question alone except 6 of 1,088 answers, by at most 0.005, with no choice changed (`runs/2026-10-05_engine-process/`); the 1,400-item suite and JevBench were not measured that way.
- Slower than both served bases on states it has not seen before, the more so the longer the state (one question on a new 3,000-token state: 476 ms, against 91 ms on the Qwen base).
- One card and one session; the intent heads over three draws, the other bases' over six.
- Not served on a Mac; the image route was not measured.
- The limits of section 5 on JevBench, the Decision Index rows and the board apply as written; nothing was submitted to JevBench's board, and the Decision Index's whole suite was self-run and submitted (section 8).

## 8. The Decision Index's whole suite, self-run

The kit's whole 0.2.1 suite, 150,759 requests of which 150,317 are scored, was run once per base on one RTX PRO 6000 Blackwell Workstation Edition (`runs/2026-10-05_decision-index/`).
Each base was served by its tag with `--base` alone, and the kit's `http` engine at `87d4650` sent one request at a time.
These are self-run scores, submitted in [apolinario/decision-index#62](https://github.com/apolinario/decision-index/pull/62) on 2026-10-05 and pending the maintainers' validation; they are not board numbers until the maintainers add them.

| Base | Tag | Decision Index 0.2.1 | Raw index | Breadth | Scored requests answered | Request latency, median / p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B (FP8) | v0.4.0 | 48.06 | 60.45 | 46.55 | 150,317 of 150,317 | 70.5 / 261.2 ms |
| Gemma 4 12B (bf16) | v0.4.0 | 49.43 | 61.47 | 47.57 | 150,317 of 150,317 | 37.8 / 275.9 ms |
| Gemma 4 31B (FP8 on load) | v0.6.0 | 57.58 | 67.33 | 56.50 | 150,317 of 150,317 | 56.0 / 439.3 ms |

The suite was rebuilt with the kit and matches the lab's hashes; no request was unsupported or in error.
Request latency is the kit's HTTP wall time per request on this card, one at a time, not the maintainers' measurement.
The four benchmarks of sections 3, 6 and 7 reproduce in these runs: exactly on the Qwen and 31B bases, and within 0.001 on the 12B except GPQA Diamond (two of 196 items), as its bf16 logits allow.
The 31B run was resumed once after a container restart: the same server was started again from v0.6.0 with an identical `/health`, and the kit sent only the requests not yet answered.
The per-request results, without item text, are in [aminry/decisio-decision-index](https://huggingface.co/datasets/aminry/decisio-decision-index); the area and per-benchmark values are in each run's `scores.json`.
What was chosen with the Decision Index in view is stated in sections 4, 6.4 and 7.4.

### 8.1 Where Jev leads

The public board's Jev row, read from its data file `data/index-v0.2.1.json` at space commit `cdbd1ca` ([multimodalart/jev-decision-index](https://huggingface.co/spaces/multimodalart/jev-decision-index), generated 2026-09-28), beside the three bases, accuracy:

| Benchmark | Jev (board, jev-1.13.0) | Qwen3.6-35B-A3B | Gemma 4 12B | Gemma 4 31B |
| --- | ---: | ---: | ---: | ---: |
| GPQA Diamond | 0.786 | 0.510 | 0.388 | 0.520 |
| MMLU-Pro | 0.827 | 0.613 | 0.550 | 0.694 |

Jev's values are quoted from the board, not measured by us (`runs/2026-10-05_decision-index/board_jev.json`).
On these two knowledge benchmarks Jev is ahead of every base served here.
