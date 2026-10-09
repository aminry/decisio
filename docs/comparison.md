<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# decisio, Jev and the leading open entries

decisio's three bases set beside TypeSafe's Jev 1.13 and the three open-weights entries with the highest Decision Index on its public board, as of 2026-10-05.
Our values are self-run: the Decision Index runs are submitted in [apolinario/decision-index#62](https://github.com/apolinario/decision-index/pull/62) and pending the maintainers' validation, and the JevBench counts are this repository's records, not board entries.
Jev's and the other entries' values come only from the public boards, with each board's revision, the fields read and the date read in [`runs/2026-10-05_comparison/`](../runs/2026-10-05_comparison/).
No figure here comes from any call of ours to TypeSafe's API, and nothing here is a rank.

**How a cell is marked.** ▲ ahead of Jev, = level, ▼ behind, by one rule for every cell, written and committed before anything was computed ([`RULE.md`](../runs/2026-10-05_comparison/RULE.md)).
A difference counts only when it lies outside a 95% interval for the benchmark's size, h = 1.96 √(2m(1 − m)/n), with m the two values' mean and n the benchmark's items; otherwise the cell is level.
For the index and the areas, the same per-benchmark variance is carried through the board's formula.
The one lower-is-better metric, ForecastBench's Brier loss, is compared in its own direction.
Only our cells are marked; the other entries' columns are shown as the board has them.

## 1. Headline

| | decisio · Qwen3.6-35B-A3B | decisio · Gemma 4 12B | decisio · Gemma 4 31B | Jev 1.13 | Surogate Rune 26B-A4B v3 | Decider chat · Gemma-4-31B | pplx-decider-v1-27b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Decision Index 0.2.1** (0 to 100) | 48.06 ▼ | 49.43 ▼ | 57.58 = | 57.91 | 57.44 | 57.33 | 56.40 |
| Knowledge & Reasoning | 32.45 ▼ | 31.37 ▼ | 45.57 ▼ | 51.40 | 43.36 | 44.27 | 40.88 |
| Language Understanding | 51.39 ▼ | 51.66 ▼ | 61.02 = | 62.02 | 63.06 | 60.38 | 63.46 |
| Retrieval & Classification | 53.56 = | 58.00 ▲ | 62.75 ▲ | 55.42 | 63.52 | 63.08 | 54.86 |
| Tools & Automation | 68.43 ▼ | 71.62 ▼ | 75.04 = | 75.09 | 71.22 | 75.57 | 79.35 |
| Arts & Human Taste | 31.57 ▼ | 32.59 ▼ | 37.43 = | 37.66 | 41.90 | 38.34 | 39.39 |
| **JevBench, 231 published questions,** correct | 200 = | 200 = | 213 ▲ | 200 | not on JevBench's board | not on JevBench's board | not on JevBench's board |
| JevBench hard tier, correct | 82 of 111 | 82 of 111 | 93 of 111 | 0.741 of 220, not compared (a) | not on JevBench's board | not on JevBench's board | not on JevBench's board |
| Latency, median per request | 70.5 ms (b) | 37.8 ms (b) | 56.0 ms (b) | 524.1 ms, round trip over the internet (c) | 120.5 ms (d) | 108.5 ms (d) | 101.4 ms (d) |
| Price or self-hosting cost, one pass over the 38 index benchmarks | $6.98 est. (e) | $6.92 est. (e) | $10.28 est. (e) | $6.05 at its public tariff (f) | $16.53 est. (e) | $22.18 est. (e) | $24.94 est. (e) |
| Context length | 32,768 tokens per prompt, the state plus one question | 32,768 tokens per prompt | 32,768 tokens per prompt | 64k tokens per request; 32k tokens for the state plus the longest question (g) | not on the board | not on the board | not on the board |
| What it is | official Qwen weights, frozen, letters readout | official Gemma weights, frozen, letters readout | official Gemma weights, frozen, FP8 on load, letters readout | hosted model, TypeSafe's API | full fine-tune of `google/gemma-4-26B-A4B-it` | an inference technique on `google/gemma-4-31B-it` | full fine-tune of `Qwen/Qwen3.8-27B` |

(a) JevBench's board reports Jev's hard tier over 220 questions, 109 of them held out and never published, so it is not the same item set as the 111 published ones we can run.

(b) Our server time on one RTX PRO 6000 Blackwell: the Decision Index kit's HTTP wall time to a server on the same machine, one request at a time, over all 150,759 requests.

(c) The board's own measurement of TypeSafe's hosted API: an HTTPS round trip over the internet from its lab, one request at a time, on its latency sample.
It includes the network, so it is not the same measurement as (b) and (d), and it carries no mark.

(d) The board's on-card median on one RTX PRO 6000, one request at a time, on its latency sample.

(e) Self-hosted, estimated as mean request latency × 150,759 requests at $1.48 per card-hour, what the card for our runs cost on vast.ai; for the other entries, with the board's mean latency.
One request at a time throughout; a server answering requests concurrently would cost less per pass.

(f) The board's recorded API cost of Jev's run over the same benchmarks, at TypeSafe's tariff of $0.042 per million input tokens, output tokens free (`jev.results[].usd`).

(g) From TypeSafe's documentation (section 3).

## 2. Every index benchmark

One row per benchmark, grouped by area, with the coverage-adjusted value that feeds the index (`raw` on the board); ForecastBench shows its Brier loss, where lower is better.
n is the number of items behind the interval: the smallest of the benchmark's cases, our scored requests and Jev's cases.
Each interval is in `runs/2026-10-05_comparison/results.json`, and `tables.md` there is these tables as the script writes them.

### Knowledge & Reasoning

| Benchmark | n | Qwen3.6-35B-A3B | Gemma 4 12B | Gemma 4 31B | Jev 1.13 | Rune 26B-A4B v3 | Decider chat · Gemma-4-31B | pplx-decider-v1-27b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPQA Diamond, accuracy | 196 | 0.510 ▼ | 0.388 ▼ | 0.520 ▼ | 0.786 | 0.469 | 0.490 | 0.495 |
| GSM8K, accuracy | 1,319 | 0.544 ▼ | 0.599 ▼ | 0.805 = | 0.799 | 0.797 | 0.826 | 0.674 |
| ChessBench, accuracy | 5,000 | 0.162 = | 0.165 = | 0.220 ▲ | 0.172 | 0.187 | 0.223 | 0.172 |
| MuSR, accuracy | 752 | 0.581 ▼ | 0.581 ▼ | 0.640 = | 0.661 | 0.649 | 0.642 | 0.618 |
| SATA-Bench, case exact accuracy | 1,650 | 0.232 ▼ | 0.264 = | 0.323 ▲ | 0.264 | 0.348 | 0.293 | 0.299 |
| CRUXEval, accuracy | 570 | 0.625 ▼ | 0.654 ▼ | 0.832 ▲ | 0.730 | 0.744 | 0.793 | 0.749 |
| CLadder, accuracy | 5,000 | 0.622 ▼ | 0.634 ▼ | 0.764 ▲ | 0.726 | 0.727 | 0.746 | 0.745 |
| HLE, accuracy | 501 | 0.114 ▼ | 0.106 ▼ | 0.150 ▼ | 0.204 | 0.106 | 0.148 | 0.120 |
| MMLU-Pro, accuracy | 12,032 | 0.613 ▼ | 0.550 ▼ | 0.694 ▼ | 0.827 | 0.672 | 0.696 | 0.645 |
| BBH, accuracy | 5,507 | 0.694 ▼ | 0.691 ▼ | 0.762 ▼ | 0.929 | 0.811 | 0.762 | 0.781 |

### Language Understanding

| Benchmark | n | Qwen3.6-35B-A3B | Gemma 4 12B | Gemma 4 31B | Jev 1.13 | Rune 26B-A4B v3 | Decider chat · Gemma-4-31B | pplx-decider-v1-27b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ContractNLI, macro-F1 | 123 | 0.733 = | 0.743 = | 0.726 = | 0.717 | 0.770 | 0.736 | 0.781 |
| ANLI, macro-F1 | 3,200 | 0.686 ▼ | 0.702 ▼ | 0.737 = | 0.748 | 0.735 | 0.732 | 0.707 |
| WinoGrande, accuracy | 1,267 | 0.808 ▼ | 0.730 ▼ | 0.852 ▼ | 0.919 | 0.860 | 0.837 | 0.852 |
| HellaSwag, accuracy | 10,042 | 0.940 = | 0.867 ▼ | 0.923 ▼ | 0.945 | 0.924 | 0.920 | 0.940 |
| ACOS, per-review F1 | 400 | 0.174 ▼ | 0.155 ▼ | 0.215 ▼ | 0.295 | 0.268 | 0.188 | 0.204 |
| FinEntity, macro-F1 | 979 | 0.803 ▼ | 0.905 ▲ | 0.931 ▲ | 0.870 | 0.884 | 0.909 | 0.929 |
| iSarcasmEval, Sarcasm F1 · track A, English | 1,400 | 0.474 = | 0.427 ▼ | 0.506 = | 0.505 | 0.604 | 0.511 | 0.603 |
| VAST, macro-F1 | 3,006 | 0.490 ▼ | 0.591 ▼ | 0.687 ▲ | 0.646 | 0.766 | 0.732 | 0.708 |
| NLI4CT, macro-F1 | 5,500 | 0.772 ▼ | 0.813 ▼ | 0.835 = | 0.841 | 0.804 | 0.829 | 0.848 |
| RAGTruth, F1 on hallucinated class | 2,700 | 0.702 ▼ | 0.696 ▼ | 0.783 = | 0.765 | 0.768 | 0.769 | 0.803 |

### Retrieval & Classification

| Benchmark | n | Qwen3.6-35B-A3B | Gemma 4 12B | Gemma 4 31B | Jev 1.13 | Rune 26B-A4B v3 | Decider chat · Gemma-4-31B | pplx-decider-v1-27b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BANKING77, macro-F1 | 3,080 | 0.746 ▼ | 0.729 ▼ | 0.784 = | 0.797 | 0.838 | 0.791 | 0.790 |
| CLINC150+OOS, macro-F1 | 5,500 | 0.822 ▼ | 0.872 ▼ | 0.901 = | 0.893 | 0.874 | 0.912 | 0.878 |
| BRIGHT, nDCG@10 | 220 | 0.444 = | 0.471 = | 0.439 = | 0.475 | 0.463 | 0.436 | 0.487 |
| Amazon ESCI, macro-F1 | 5,000 | 0.450 ▼ | 0.535 = | 0.540 = | 0.552 | 0.553 | 0.531 | 0.551 |
| PhishNChips phishing decisions, accuracy | 2,000 | 0.752 ▲ | 0.813 ▲ | 0.875 ▲ | 0.625 | 0.809 | 0.875 | 0.600 |
| HoVer, accuracy | 4,000 | 0.700 ▼ | 0.693 ▼ | 0.756 ▲ | 0.729 | 0.807 | 0.764 | 0.742 |

### Tools & Automation

| Benchmark | n | Qwen3.6-35B-A3B | Gemma 4 12B | Gemma 4 31B | Jev 1.13 | Rune 26B-A4B v3 | Decider chat · Gemma-4-31B | pplx-decider-v1-27b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BFCL, case exact accuracy | 1,694 | 0.960 = | 0.962 = | 0.979 ▲ | 0.958 | 0.948 | 0.980 | 0.976 |
| ToolRet, nDCG@10 | 685 | 0.625 = | 0.622 = | 0.613 = | 0.653 | 0.644 | 0.636 | 0.674 |
| API-Bank, accuracy | 508 | 0.817 ▼ | 0.870 = | 0.854 = | 0.882 | 0.833 | 0.850 | 0.841 |
| Home appliance simulator, case exact accuracy | 88 | 0.398 = | 0.443 = | 0.614 = | 0.523 | 0.466 | 0.625 | 0.739 |
| When2Call, accuracy | 3,652 | 0.714 ▼ | 0.761 ▼ | 0.772 ▼ | 0.810 | 0.760 | 0.769 | 0.817 |

### Arts & Human Taste

| Benchmark | n | Qwen3.6-35B-A3B | Gemma 4 12B | Gemma 4 31B | Jev 1.13 | Rune 26B-A4B v3 | Decider chat · Gemma-4-31B | pplx-decider-v1-27b |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BPoMP, accuracy | 811 | 0.904 = | 0.910 = | 0.894 = | 0.909 | 0.900 | 0.906 | 0.939 |
| Humicroedit, accuracy | 2,628 | 0.608 = | 0.617 = | 0.625 = | 0.619 | 0.622 | 0.638 | 0.624 |
| POP909-CL, accuracy | 2,000 | 0.142 ▼ | 0.060 ▼ | 0.250 ▲ | 0.166 | 0.661 | 0.279 | 0.378 |
| cfcolor, accuracy | 5,000 | 0.600 ▼ | 0.624 ▼ | 0.639 = | 0.644 | 0.626 | 0.626 | 0.644 |
| ForecastBench, Brier loss (lower is better) | 10,139 | 0.228 ▼ | 0.202 ▼ | 0.207 ▼ | 0.174 | 0.203 | 0.202 | 0.195 |
| Habermas Machine, accuracy | 1,676 | 0.447 = | 0.465 = | 0.484 = | 0.459 | 0.422 | 0.479 | 0.418 |
| New Yorker, accuracy | 528 | 0.688 = | 0.633 ▼ | 0.739 = | 0.701 | 0.741 | 0.739 | 0.703 |

### Hosted chat models on JevBench's 231 published questions

The JevBench operator's published counts for the general chat models it measured through their providers' APIs, by the board's model IDs (revision v1.4.2.2, `public_accuracy` × 231); they are not decision models, have no Decision Index value, and carry no marks.

| Model, as the board names it | Board ID | Correct of 231 |
| --- | --- | ---: |
| GPT-6 Luna, low reasoning effort | `gpt-6-luna-low` | 229 |
| GPT-6 Luna, default medium reasoning effort | `gpt-6-luna` | 230 |
| GPT-5.6 Luna, low reasoning effort | `gpt-5.6-luna` | 225 |
| Gemini 3.1 Flash-Lite | `gemini-3.1-flash-lite` | 201 |
| DeepSeek V4.1 Flash, thinking by default | `deepseek-flash` | 226 |
| Qwen3.8 27B, on Chutes | `qwen3.8-27b` | 166 |

## 3. Capabilities

Each statement about Jev is what TypeSafe's public documentation states, quoted in [`snapshots/typesafe_docs.json`](../runs/2026-10-05_comparison/snapshots/typesafe_docs.json) with each page's hash, read 2026-10-05 at 21:44 UTC.
Where a row says the documentation does not state something, that is about the pages read, not a claim about the service.

| | decisio (all three bases unless noted) | Jev 1.13, as TypeSafe's documentation states it |
| --- | --- | --- |
| Question types | yes/no (`noul`), choice and score, on the same `/v1/systemone` wire format | "The three TypeSafe question types (Choice, Score, Noul)" ([primitives](https://docs.typesafe.ai/primitives)) |
| Options per question | up to 255 named options per choice | "a maximum of 255 options per Choice"; a Score "accepts up to 10" levels ([API](https://docs.typesafe.ai/api)) |
| Several questions per request | the state is read once and shared by every question in the request | "System One models evaluate every question in a request in parallel." ([primitives](https://docs.typesafe.ai/primitives)) |
| Context | 32,768 tokens per prompt: the state plus one question | "64k tokens per request; 32k tokens for `state` plus the longest question" ([models](https://docs.typesafe.ai/models)) |
| Self-hosting | yes: one 96 GB card on vLLM, the Docker image, or a Mac (below) | served by "the same endpoint, `POST /v1/systemone`"; the pages read describe the hosted API and no self-hosted option ([models](https://docs.typesafe.ai/models)) |
| Open weights | the bases' official weights, downloaded at pinned revisions, never redistributed | "the same weights serve every account"; the pages read name no downloadable weights ([models](https://docs.typesafe.ai/models)) |
| Teaching from labelled examples | `POST /v1/tasks` fits a per-task calibration and, for long option lists, an intent head on your labelled examples; the model stays frozen | "Jev is not fine-tuned or LoRA-adapted with customer data", and "You shape its answers to your domain through the request rather than through per-account weights" ([models](https://docs.typesafe.ai/models)) |
| Image input | photos in the state, through a second engine on the same card (`--image-model`); measured on the Qwen base only | "Text only. String, JSON object, or array of text values. No image, audio, or video input." ([models](https://docs.typesafe.ai/models)) |
| Repeatability | checkpoints at pinned revisions; Qwen: a repeated request returns the same probabilities bit for bit; Gemma 4 12B: the same choice, probabilities within 0.035; Gemma 4 31B: a repeated request returns the same answers with the engine in the server's process, within one session on one machine; across sessions near-tied answers can move (`EVAL_CARD.md` sections 5 to 7) | "An alias moves when a new release ships, so the answers behind it can change without a change on your side." The pages read do not state whether a repeated request returns the same probabilities ([models](https://docs.typesafe.ai/models)) |
| Calibration | fitted temperatures per base; JevBench ECE on the standard and hard tiers: Qwen 0.121 and 0.043, Gemma 4 12B 0.033 and 0.085, Gemma 4 31B 0.035 and 0.091 | "System One models are trained for calibrated decisions: their probabilities are optimized against outcomes to reflect uncertainty." ([System One](https://docs.typesafe.ai/concepts/system-one)) |
| Where data goes | to the machine that runs the server, and nowhere else | "Jev is not trained on customer requests or responses."; zero data retention is offered "for enterprise customers" ([models](https://docs.typesafe.ai/models), [legal](https://docs.typesafe.ai/legal)) |
| Mac and Ollama | Qwen, Gemma 4 12B and Gemma 4 31B (up to 16,383 tokens of state) on Apple silicon with MLX; the Qwen base on Ollama (`aminroudaki/decisio`), with Ollama's own prompt and no calibration | the pages read describe the hosted API only |
| Licence | Apache-2.0; the weights are Qwen's under Apache-2.0 and Google's Gemma 4 under Apache-2.0 | a hosted service under TypeSafe's customer agreements, which "govern your use of TypeSafe" ([legal](https://docs.typesafe.ai/legal)) |
| Price | no per-call price; the card's cost (section 1, e) | "$42 / $0.042" per billion / million tokens, "Charged per input token. Output tokens are free." ([models](https://docs.typesafe.ai/models)) |

## 4. Summary

- **Qwen3.6-35B-A3B:** on the 38 index benchmarks it is ahead of Jev on 1, level on 12 and behind on 25, and its index, 48.06, is behind Jev's 57.91.
- **Gemma 4 12B:** ahead on 2, level on 12 and behind on 24; its index, 49.43, is behind.
- **Gemma 4 31B:** ahead on 10, level on 19 and behind on 9; its index, 57.58, is level with Jev's 57.91 (the interval is ±1.03).

Jev leads all three bases on the knowledge-heavy benchmarks, GPQA Diamond, MMLU-Pro, BBH and HLE, and on WinoGrande, ACOS, When2Call and ForecastBench's Brier loss.
On the Qwen and Gemma 4 12B bases it also leads most of language and tools, and their indices are 9.9 and 8.5 points below its own.
The Gemma 4 31B base is level with Jev on the index and on three of the five areas, ahead on Retrieval & Classification, and ahead on ten benchmarks, among them CRUXEval, CLadder, VAST, FinEntity, HoVer, BFCL and POP909-CL.
All three bases are ahead on PhishNChips, and on JevBench's 231 published questions the 31B answers 213 against Jev's 200, with the other two level at 200.
Beyond the scores, the difference is in how each runs: decisio runs on your own card or Mac with the data staying there, takes images and learns a recurring question from labelled examples, while Jev is a hosted API with a longer context per request and, at its tariff, a lower cost for one pass over the suite than our one-request-at-a-time estimates.
