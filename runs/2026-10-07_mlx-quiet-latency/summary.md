# MLX server time on a quiet Mac, both bases (2026-10-07)

Every MLX latency figure since 2026-10-03 had been measured beside other processes.
This record measures them again with no other job running on the Mac (its owner away, the display asleep), with the same drivers and run counts as `runs/2026-10-03_mlx-regate` and `runs/2026-10-05_mlx-gemma`, on decisio `83e461d`, an Apple M5 Pro (64 GB), on AC power.
Pre-registered before any measurement (the RLCD repository, `experiments/2026-10-06_ls_mlx_quiet_latency/PREREG.md`).

- **Quiet rule:** each base started after five minutes with no process above 25% CPU outside the desktop set (WindowServer and two desktop apps) and the desktop set under 100% (`<base>/wait_quiet.log`).
- **During the passes:** `latency.py`'s own idle rule (no other process above 25%) did not hold in any pass; macOS's indexing and analysis services (`mds_stores`, `corespotlightd`, `duetexpertd`, `searchpartyd`) peaked at 35% to 150%. Its three passes agree within 3%; the third is reported, as before.
- **Models:** `mlx-community/Qwen3.6-35B-A3B-6bit` at `cb7e092` (tokenizer `Qwen/Qwen3.6-35B-A3B` at `995ad96`), `mlx-community/gemma-4-12B-it-6bit` at `2fe53ae`.

## Server time, median

| Cell | Qwen | Gemma |
| --- | ---: | ---: |
| One question on a new state (`latency.py`, prefix cache off) | 258 ms | 392 ms |
| One question on a state seen before (prefix cache hit, 30 states) | 108 ms | 156 ms |
| Per question, 10 sharing a state | 123 ms | 179 ms |
| Per question, 100 sharing a state | 123 ms | 166 ms |
| Per question, 100 on an 8,000-token state | 176 ms | 289 ms |
| One question on a new state of 300 / 1,000 / 3,000 tokens (`t7.py latency`) | 297 / 515 / 1,214 ms | 554 / 1,312 / 3,630 ms |
| Four questions on a new state of 300 / 1,000 / 3,000 tokens | 627 / 853 / 1,545 ms | 1,029 / 1,783 / 4,206 ms |
| One question on a cached 1,000-token state | 133 ms | 165 ms |

Every prefix-cache hit's answers equal its miss's (30 of 30 on each base).

## Against the earlier records

- Qwen is faster in every cell: one question on a new state 266 to 258 ms, on a state seen before 111 to 108 ms, per question at 100 130 to 123 ms, and one question on a new 1,000-token state 704 to 515 ms (that cell had run on the gate server under heavy load).
- Gemma is level with its record, every cell within about 2% (one question on a new 1,000-token state 1,299 to 1,312 ms).

## Files

`<base>/`: `latency.json` and `.log` (three passes), `prefix_cache_latency.json`, `latency_t7.json`, the quiet-wait log, the host record at the start and end (`host.txt`, `host_end.txt`, with the decisio commit), and each server's `/health` (L: every flag the default; C: `--prefix-cache-mb 0`; the Gemma servers also `--debug-readout`, as its record).
