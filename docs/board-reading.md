<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# How we read the JevBench board

The README's headline is second among open-weights systems on JevBench, level with Jev, and the highest-ranked with frozen weights.
It rests on the board's own ranks and on interval overlap, not on separation.
This page says how each part is read.

- **Source.** The figures are the board's, from its published file for [JevBench v1.6.1](https://benchmarkheaven.com/api/jevbench/v1.6.1) (sha256 `5cd8c1332226...`, read 2026-10-09), row `decisio-gemma-4-31b-v080`; the charts are drawn from that file, view A.
- **Open weights.** Second among the board's 128 open-weights systems: the board's own `open_board_rank` is 2, and H2O-Lightning-4B ranks above us (72.52, interval 69.79 to 73.79) with an interval that overlaps ours (69.86 to 72.81). Overall the row is fourth of 135 on the headline composite (third on view B, fourth on view C).
- **Frozen weights.** The board has no frozen-weights class. We count a row as frozen when its display text says so (or says stock Gemma or Qwen) and names no LoRA, fine-tune, merge, training, adapter, head or decoder. The claim is about rank: decisio's row is the highest-ranked of those.
- **Level with Jev.** Our interval (69.86 to 72.81) and Jev 1.13.0's (69.14 to 72.40) overlap; the scores are 71.69 and 71.49.
- **Intervals.** Two other frozen-weights rows have intervals that overlap ours, decider-12b-v1 (rank 9) and Cygnet (rank 13); deck-31B (rank 12) does not.
- **The row.** It is decisio v0.8.0 with Google's weights quantized to FP8 on load, measured by the board on one H100 80 GB on 2026-10-06; 1,477 of its 1,500 items were answered, the 23 others being items of about 80,000 tokens, over the 32,768-token context. The 31B repository's FP8 weights are that same quantization, stored.
- **Not fitted, but read.** Our temperatures were not fitted on JevBench; we have read its published items (EVAL_CARD 7.4).
- **Also on the board:** second on the sealed set.

The figures are the board's, from its published file for [JevBench v1.6.1](https://benchmarkheaven.com/api/jevbench/v1.6.1).
