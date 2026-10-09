# Gemma 4 31B on MLX at 6 bits: the gates (2026-10-07 to 2026-10-08)

**Status: complete.** Every pre-registered stage ran and is reported: the suite, the JevBench capture, conformance, the intent heads (three draws per set), two memory readings and the prefix-cache entry sizes. Server times were not part of this run.

`--backend mlx --base gemma-4-31b` serving `mlx-community/gemma-4-31b-it-6bit@7d13b58`, prompts built with `google/gemma-4-31B-it@842da37`, on an Apple M5 Pro (64 GB), macOS 26.6.2, mlx 0.32.3, mlx-lm 0.32.0.
Reference: the base's vLLM FP8 record, `runs/2026-10-04_gemma-4-31b` (suite and JevBench readouts at the served temperatures 5.252 and 4.672, and three intent-head draws per set).
Pre-registered before any gate measurement (the RLCD repository, `experiments/2026-10-06_ls_gemma31b_mlx/PREREG_6bit.md`), with four dated amendments, all committed before the gate run started; the amendments concern the memory rule and are disclosed there as prompted by earlier trial runs.

**Code.** The suite, the capture, conformance and memory reading a ran on branch `feat/mlx-gemma-4-31b` at `8512f27` (21:31 UTC, 2026-10-07). The branch was then rebased onto main (which added the per-base cache budget) and gained the FP8-repository guard, so the heads, memory reading b and the cache entries ran at `54b6754` (from 01:36 UTC, 2026-10-08); the finished stages were re-scored from their kept outputs with identical numbers. Between the two commits the MLX engine file differs only in a docstring and a comment, and the 31B's profile gains a vLLM-only setting (`register_boundary`) and the per-base cache-budget field, which stays 2,048 MiB for the 31B; its prompt and temperature settings are unchanged. The one head draw run on both commits, BANKING77 draw 0, gave the same accuracies on both (0.847 with the head, 0.800 plain).

**Deviation 1.** The first heads attempt (23:22 UTC) failed after BANKING77 draw 0, because a registration outlasted the shared harness's fixed 3,600 s request timeout. The heads were rerun from draw 0 with a copy of the driver whose timeout is set from the environment (`code/t7_31b_intents.py`, 6 hours); no rule or gate changed. The failed attempt's log is `6bit/intents_attempt1_timed_out.log`; its draw 0 (0.847, plain 0.800) was reproduced exactly by the rerun.

## Gates

| Gate | MLX 6-bit | FP8 | Verdict |
| --- | --- | --- | --- |
| S1: suite accuracy (1,400 items) | 0.8064 | 0.7993 | pass: +0.7 [0.0, +1.5] points |
| S2: top-answer flips | 42 of 1,400; largest probability difference 0.572 | | counted |
| S3: pooled ECE (allowed: within 0.01) | 0.0535 | 0.0540 | pass |
| F3: JevBench 231, ECE at the served temperatures (reported; decides only if S3 fails) | 0.0442 | 0.0349 | +0.0093; accuracy 0.918 vs 0.922 |
| C2 / C3 / C4 conformance | pass / pass (0 of 1,400 differ) / pass (largest and mean probability difference 0.0, 0 flips) | | pass |
| H: BANKING77 intent head, three draws (of 150) | 127 / 130 / 125, mean 0.849 | 125 / 130 / 125, mean 0.844 | pass: +0.4 [-2.2, +3.3] points |
| H: CLINC150 intent head, three draws (of 100) | 97 / 97 / 97, mean 0.970 | 97 / 97 / 97, mean 0.970 | pass: 0.0 [0.0, 0.0] points |

The lower end of the S1 interval is exactly 0.0: MLX is not measurably below FP8, and its point estimate is above it.
Heads: every draw's head was applied. Plain readouts: BANKING77 0.800 against the record's 0.787, CLINC150 0.930 against 0.940. On CLINC150 every one of the 300 head answers equals the record's; on BANKING77 the top answers agree on 137, 143 and 143 of 150 per draw.
Registration took 3,657 / 3,812 / 3,759 s per BANKING77 draw and 9,718 / 9,466 / 9,401 s per CLINC150 draw.

## Memory

GB are 1e9 bytes. The GPU memory limit is 55.66 GB; the weights take 24.9 GB. A length fits if its cells completed, every peak was at most 90% of the limit (50.1 GB), and macOS memory pressure was normal in every sample (one per 5 s) from before the model loaded through the length's last cell. Swap growth is recorded, not gated.
Two readings, both through the same code:
- **Reading a (the quiet window; governs):** right after conformance, at night, 23:14 to 23:22 UTC, with nothing large resident.
- **Reading b (the working-day reading; recorded, not gated):** the pre-registered position, after the heads, 13:24 to 13:32 UTC (06:24 PDT), labelled "under working-day load" by Amendment 3. The eight largest resident processes at its start were desktop applications, the largest 0.5 GB (`6bit/memory_load_b.txt`), so the label is the registered one and not a measurement of heavy use.

| State tokens | Peak | Limit free | Reading a: samples not normal in the cells; swap growth (1, 10 questions) | Reading b: samples not normal in the cells; swap growth | Verdict |
| ---: | ---: | ---: | --- | --- | --- |
| 8,056 | 29.62 GB | 46.8% | 0 of 13; 0, -8 MB | 0 of 12; 0, 0 MB | fits |
| 16,383 | 31.08 GB | 44.2% | 0 of 25; -8, -48 MB | 0 of 25; 0, 0 MB | fits |
| 32,687 | 33.72 GB | 39.4% | 1 of 54 (warning level); +78, -8 MB | 9 of 61 (warning level, none critical); +3,688, +1,905 MB | runs, with memory pressure |

**Documented for states up to 16,383 tokens.** At 32,687 tokens it runs, with memory pressure: one warning sample in 54 in the quiet reading (swap flat, 4.9 to 5.0 GB over the whole reading), nine in 61 in the working-day reading (swap up 3.7 and 1.9 GB). The rule was applied as written to reading a and not amended after either reading.

## Prefix-cache entries

One state per length on an empty server with a 65,536 MiB budget (nothing evicted); `prefix_cache.mb` after one request is the entry's size (MiB, 2^20 bytes).

| State tokens | Entry | First request (server time, state not cached) |
| ---: | ---: | ---: |
| 1,000 | 935.2 MiB | 3.2 s |
| 3,016 | 1,869.4 MiB | 9.3 s |
| 8,014 | 2,954.1 MiB | 26.7 s |
| 32,017 | 4,386.4 MiB | 126.5 s |

At the default budget the 31B inherits (2,048 MiB; no per-base value is set for it), 20 distinct 1,000-token states asked in a row left 2 in the cache (1,870.3 MiB). A 3,000-token entry fits once; 8,000- and 32,000-token entries are larger than the whole budget and are never stored, so a repeat question on such a state reads it again. In the memory readings the same applies: the ten-question cell at each length re-read its state (equal prefix times in `memory_a.json`, `memory_b.json`), so every peak above was measured with an empty cache.
A cache hit's time for the 31B was not measured.
The entries are large because the 50 sliding-window layers keep their last prefill chunk whole (inferred from the config, fitted to these four entries to within 70 to 80 MiB): an entry grows with the length of the state's last 2,048-token chunk, so entries at one length can differ by up to about 1.6 GiB, and a 32,768-token state, a whole number of chunks, is estimated near 5.0 GiB.

## Not measured

- Server times (the quiet-window latency drivers).
- A cache hit's time, and memory pressure with the cache full (the budget proposal in the RLCD folder says why that matters).
- The 4-bit and 8-bit conversions, and any Mac smaller than 64 GB.

## Files

`6bit/`: the suite answers (`suite.json.gz`), the JevBench capture, conformance, the intent heads with every answer (`intents.json.gz`), the scored gates (`report.json`, `compare.log`), both memory readings (`memory_a.json`, `memory_b.json`, each with its pressure samples, resident-process list and per-length verdict), the cache entries (`cache_e_*.json`, `cache_f_1000.json`), each server's `/health` and log, the host record and the failed first heads attempt's log.
