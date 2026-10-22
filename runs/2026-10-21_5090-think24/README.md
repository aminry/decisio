<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Think mode on decisio 0.24.0, the default server, one RTX 5090

What this shows: after the fix of 0.24.0, a default server (decisio's hidden-readout class) writes real reasoning when it thinks, and stops on its own on most questions. 0.21.0 to 0.23.x did not (every thinking question ran to the budget on reserved token ids).

One RTX 5090 (32 GB), vLLM 0.31.0, decisio 0.24.0, the 12B quantized to FP8 on load, `--think --think-factor 1.5`, budget 512 tokens, the engine in the server's process, 20 questions sent one at a time, 2026-10-21.

| | |
| --- | --- |
| Questions that thought | 20 of 20, real reasoning (no request failed, none `thinking: unavailable`) |
| Stopped on their own at the end of the turn | 15 (median 402 tokens over all 20; 6,348 written) |
| Ran to the 512-token budget | 5, all hard items |
| Ids in the reserved readout range | 10 of 6,348 (0.16%), 6 distinct ids; the largest share in one question is 1.2% |

The readout range (Gemma 4: ids from 170,000) holds ordinary vocabulary tokens as well as the columns decisio keeps its hidden state in; with the columns switched off for the writing call, a word of that range appears now and then in normal text, as here. Each of the 10 was decoded with its context: ordinary English words in coherent sentences, none a leak. The bug it replaces wrote 100% reserved ids (20 of 20 questions, 10,240 of 10,240 tokens, never stopping) on 0.23.0.

`think24.json` holds the sampling parameters as passed, the summary and one row per question (tokens, finish reason, whether it stopped at the end of the turn, reserved-range ids, the second read's top probability). It holds no question or reasoning text.
Only 20 questions were run on one card and one base: it shows the mechanism works, not how much thinking helps (that is the evaluation's, not this record's).
