# Gemma 4 31B on MLX at 6 bits: the gates run so far (2026-10-07)

**Status: partial.** The suite, the JevBench capture, conformance and memory reading a are finished and reported here; the intent heads (three draws per set), memory reading b (under working-day load) and the prefix-cache entry sizes were still running when this was written, and are marked pending. The record is completed before the pull request is marked ready.

`--backend mlx --base gemma-4-31b` serving `mlx-community/gemma-4-31b-it-6bit@7d13b58`, prompts built with `google/gemma-4-31B-it@842da37`, on decisio branch `feat/mlx-gemma-4-31b` (`8512f27` before its rebase onto main), an Apple M5 Pro (64 GB), macOS 26.6.2, mlx 0.32.3, mlx-lm 0.32.0.
Reference: the base's vLLM FP8 record, `runs/2026-10-04_gemma-4-31b` (suite and JevBench readouts at the served temperatures 5.252 and 4.672, and three intent-head draws per set).
Pre-registered before any gate measurement (the RLCD repository, `experiments/2026-10-06_ls_gemma31b_mlx/PREREG_6bit.md`, with four dated amendments, all committed before the gate run started; the amendments concern the memory rule and are disclosed there as prompted by earlier trial runs).

## Gates

| Gate | MLX 6-bit | FP8 | Verdict |
| --- | --- | --- | --- |
| S1: suite accuracy (1,400 items) | 0.8064 | 0.7993 | pass: +0.7 [0.0, +1.5] points |
| S2: top-answer flips | 42 of 1,400; largest probability difference 0.572 | | counted |
| S3: pooled ECE (allowed: within 0.01) | 0.0535 | 0.0540 | pass |
| F3: JevBench 231, ECE at the served temperatures (reported; decides only if S3 fails) | 0.0442 | 0.0349 | +0.0093; accuracy 0.918 vs 0.922 |
| C2 / C3 / C4 conformance | pass / pass (0 of 1,400 differ) / pass (largest and mean probability difference 0.0, 0 flips) | | pass |
| H: intent heads, three draws, BANKING77 / CLINC150 | pending | 0.844 / 0.970 | pending |

The lower end of the S1 interval is exactly 0.0: MLX is not measurably below FP8, and its point estimate is above it.

## Memory, reading a (the quiet window)

Taken right after conformance, at night, with nothing large resident (the eight largest resident processes at its start and end are in `6bit/memory_load_a.txt`, by PID).
GB are 1e9 bytes. The GPU memory limit is 55.66 GB; the weights take 24.9 GB.

A length fits if its cells completed, every peak was at most 90% of the limit (50.1 GB), and macOS memory pressure was normal in every sample (one per 5 s) from before the model loaded through the length's last cell. Swap growth is recorded, not gated.

| State tokens | Peak | Limit free | Samples not normal, in the length's cells | Not normal since the start | Swap growth, 1 and 10 questions | Verdict |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 8,056 | 29.62 GB | 46.8% | 0 of 13 | 0 of 16 | 0, -8 MB | fits |
| 16,383 | 31.08 GB | 44.2% | 0 of 25 | 0 of 40 | -8, -48 MB | fits |
| 32,687 | 33.72 GB | 39.4% | 1 of 54 (level 2, warning, at +326 s in the one-question cell) | 1 of 94 | +78, -8 MB | runs, with memory pressure |

**Documented for states up to 16,383 tokens.** At 32,687 tokens it runs, with one warning-level pressure sample in 54, swap flat (4.9 to 5.0 GB over the whole reading) and 39% of the GPU memory limit free. The rule was applied as written; it was not amended after this reading.

## Pending

- Memory reading b, under working-day load, recorded beside reading a with its resident-process list and not gated.
- The intent heads, three draws per set, against the FP8 record (the evaluation items are the same: gold labels checked identical to the 12B MLX run's).
- Prefix-cache entry sizes at 1,000, 3,000, 8,000 and 32,000 tokens, and what the default budget keeps.
- Server times (the quiet-window latency drivers; not part of this run).

## Files

`6bit/`: the suite answers (`suite.json.gz`), the JevBench capture, conformance, the scored gates (`report.json`, `compare.log`), memory reading a (`memory_a.json`, its pressure samples, resident-process list and per-length verdict), the server's `/health` (A: before reading a; A2: after the restart) and log, the host record.
