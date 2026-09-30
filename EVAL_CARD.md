<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Evaluation card

What the records in `runs/` measure, on what, and what was fitted on what.
Every number below is from those records; each run's `manifest.json` has the configuration, the harness versions and the file list, and `files.json` has every file's sha256.

## The system measured

- The official `Qwen/Qwen3.6-35B-A3B-FP8` checkpoint, untrained by us: no adapter, no fine-tuning, in every measured answer.
- vLLM 0.30.0 on one NVIDIA RTX PRO 6000 Blackwell card (96 GB), `VLLM_USE_DEEP_GEMM=0`, with the suffix-staging patch series applied and switched on (`patches/`; it changes latency, not answers).
- The letters readout: each question is one prompt, and its answer is the softmax over the option letters' logits at the last position.
- Front padding to the 1,056-token block, CUDA graphs captured up to 4,096 tokens, float32 recurrent state, one request at a time.
- TypeSafe's System One wire format on `/v1/systemone`, checked before every measurement by the conformance gates C2-C4 (`decisio.serve.systemone_conformance`) and by the serving gates C1 (`tests/gpu/serving_gates.py`).

## The runs

| Run | What it measures | Served configuration |
| --- | --- | --- |
| `runs/2026-09-27_boards-baseline` | JevBench's 231 published items and four Decision Index 0.2.1 benchmarks (20,810 requests), one order and two-order averaging | the code of that date: no rendering rules, no temperature, the first maximum in wire order on ties |
| `runs/2026-09-30_step4` | The same two harnesses with the rendering rules, the tie-break and the global temperature, with the temperature on and off | the served default before the plugin (the text-only view) |
| `runs/2026-09-30_plugin-verification` | decisio's two model classes, the single-engine intent head and the patch series against stock vLLM | the text-only view, the text-only class and the hidden-readout class side by side |

None of these is a board number.
JevBench's official score needs its sealed item set, which only its operator runs, and the v1.5 reading here covers 231 of the board's 904 open items with no judge tier.
A Decision Index value needs all 38 of its benchmarks; four were run.
Nothing was submitted to either board, and no Jev API output is in this repository.

## Headline numbers

JevBench, the harness's accuracy per published file (`runs/2026-09-30_step4/jevbench/*/summary.json`): easy 1.000, original 0.972, hard 0.694, identical with and without the temperature and on 2026-09-27.
Under v1.5's open-set rules (`decisio.bench.jevbench_v15`, a private reading): choice competence 79.5, yes/no -10.9 with 54% of yes/no answers between 0.20 and 0.80 (counted wrong by the method), score 60.6; I_open 43.1 with equal type weights, 52.2 with 50/25/25.

Decision Index, the kit's own metric (`runs/2026-09-30_step4/decision_index/default/di_report.json`):

| Benchmark | Requests | Metric | 2026-09-27 | 2026-09-30 | ECE, T 1.307 | ECE, T 1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| BANKING77 | 3,080 | macro-F1 | 0.420 | 0.731 | 0.042 | 0.066 |
| CLINC150+OOS | 5,500 | macro-F1 | 0.561 | 0.814 | 0.170 | 0.031 |
| GPQA Diamond | 198 (196 scored) | accuracy | 0.454 | 0.454 | 0.144 | 0.164 |
| MMLU-Pro | 12,032 | accuracy | 0.609 | 0.609 | 0.009 | 0.072 |

The rise on the two intent sets between the runs is the rendering rules: in a separate run on the same rows (2026-09-29, not in `runs/`), the kit's macro-F1 went from 0.420 to 0.729 and from 0.561 to 0.812 with the rules switched on.
Those rules were selected on these very rows (below), so the 2026-09-30 intent scores are not held out.
The temperature changes no answer.
Median latency per request was 26 to 57 ms by benchmark on the card's localhost.

Plugin verification (`runs/2026-09-30_plugin-verification`): the text-only class and the hidden-readout class answer the 1,400-item suite bit-identically to the text-only view; an intent head registered from 10 labelled examples per intent scores BANKING77 0.847 and CLINC150 0.893 on 150 and 100 test items (means of three draws); a head question takes 127 ms of server time against 46 ms for the same question without a task.

## What was fitted on what

| Component | Fitted or chosen on | Used in |
| --- | --- | --- |
| Model weights | Nothing by us: the official checkpoint | every run |
| Global temperature T = 1.307 (`decisio.serve.temperature`) | Minimum log loss on the served readouts of a private 1,400-item suite of eight tasks (BoolQ, BANKING77, ToxicChat, MMLU, SciFact twice, MMLU-Pro 150 and 350), before step 4 | step 4's default arm |
| Rendering rules: index keys hidden, snake_case labels shown with spaces (`--hide-index-keys`, `--desnake-labels`) | Selected, with no parameter fitted, on the Decision Index's own BANKING77 and CLINC150+OOS rows: the 2026-09-27 error signature (off-by-one answers where the board's `option_N` keys sat beside our letters) led to hiding index keys, and de-snaking was added after the first reading on BANKING77 fell short of its gate; checked to change nothing on the 1,400-item suite | step 4, plugin verification |
| Tie-break: among exactly tied options, the key that sorts first | A rule, nothing fitted | step 4, plugin verification |
| Intent heads and per-task calibration | Per task, on 10 labelled examples per intent from the training splits of BANKING77 and CLINC150, registered through `POST /v1/tasks`; three draws | plugin verification only (evaluated on 150 and 100 test items) |
| The JevBench v1.5 thresholds (0.20 / 0.80) and weights | JevBench's published method, implemented as written | step 4's v15.json |

Nothing was fitted on any JevBench item, and no parameter was fitted on any Decision Index request.
Three points are stated so a reader can weigh them:

- **The temperature's fit set shares underlying questions with two Decision Index pools.** 500 of the suite's MMLU-Pro items and its 150 BANKING77 items are the same questions as items in the Decision Index's MMLU-Pro (12,032) and BANKING77 (3,080) pools, in our wording rather than the board's (4.2% and 4.9% of those pools; matched by question text in the 2026-09-27 run).
  T is one scalar and never changes which option is chosen, so accuracy and macro-F1 are unaffected; it can move ECE on those two sets.
- **The rendering rules were selected on the Decision Index's intent rows.** No number was tuned and the rules are general (an enumerated key is never shown beside our letters; a snake_case label is shown as words), but they were chosen and gated on BANKING77's and CLINC150+OOS's rows, so step 4's scores on those two benchmarks measure a configuration selected on them.
- The intent heads' test items (150 BANKING77, 100 CLINC150) are also in the Decision Index's pools; no head was registered during any Decision Index run in `runs/`.

## The batch-forward finding

On this stack (vLLM 0.30.0, the FP8 checkpoint, one RTX PRO 6000 Blackwell Max-Q), identical requests sent in one batch are not the same forward pass.

- Three identical prompts in one batch gave label probabilities that differed from each other by up to 0.59 (median 0.029) on 250 intent items (`runs/2026-09-30_plugin-verification/diag_same_row.json`).
- The same requests sent one at a time agreed with each other to 1.2e-7, and with the text path's answer to 6e-8 (`diag_same_row_sequential.json`).
- The engine computes logits in bf16: label logits recomputed from the recovered hidden state in float64 differ from the engine's by up to 0.019 in probability, and rounded to bf16 they agree within 3e-8 (`diag_same_forward.json`).

What this means for the records:

- The single-engine head reads the hidden state in three requests; they are sent one at a time and the label logits are recomputed in bf16, which is why its same-forward gate passes at 3.1e-8 (`same_forward.json`; the failed run before this fix is kept as `same_forward_run2.json`).
- A question's answer depends, within this noise, on what else is in its batch.
  Every harness in `runs/` sends one request at a time, but a request with several questions batches them, and a loaded server batches requests.
  The serving gate G2 bounds the effect of adding 16 unrelated questions to a request at 6.46e-2 in probability with no change of the chosen option on its items (`gates.log`), and that tolerance is twice the batch noise measured on this card class.
- Bit-identity claims in `runs/` (the suite under the three model classes, C4's paired probabilities) are made under the same request history on both sides.

## Limits

- One card per run and one run per configuration; no run was repeated for run-to-run variation beyond the three suite repetitions of the hidden-readout class.
- The Decision Index rows were rebuilt with the kit (`suite rebuild --only 4 5 21 25 57`); the counts match its manifest and MMLU-Pro is pinned, but a partial rebuild cannot be checked against the whole suite's hash.
- JevBench's published items have been public since v1.2 and may be in any model's pretraining data.
- Latency is server-side on the card's localhost, one request at a time; it is not a hosted-API latency.
- The per-item Decision Index records keep each request's id, the payload's sha256 and our response, not the item text (GPQA's authors ask that its items not be published in plain text); the kit rebuilds the text from its sources.
