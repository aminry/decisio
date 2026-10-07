# The boundary registration's three orders, and the GPU tier on #99 (session of 2026-10-06 to 07)

The same session as `summary.md` (one RTX PRO 6000 Blackwell Workstation Edition at an enforced 585 W, an AMD Ryzen Threadripper 9960X host, vLLM 0.30.0), on decisio #99 at `c0d251f`, which added `--register-boundary` (`off`, `before`, `after`).
Server time is the `x-decisio-server-ms` the HTTP route reports, median of 20 unless stated.
The files are checked against RLCD `experiments/2026-10-06_lab2_latency_600w/results/checksums_box.sha256`.

## The three orders

One engine per Gemma base, the order switched between passes with the queue drained and the cache emptied (`pr_g12.json`, `pr_g31.json`).

First reads with the queue idle, median of 20 paired states (ms):

| Base | State | `off` | `before` | `after` |
| --- | ---: | ---: | ---: | ---: |
| 12B | 300 | 34.4 | 56.7 | 33.3 |
| | 1,000 | 88.3 | 107.7 | 90.3 |
| | 3,000 | 237.7 | 261.9 | 237.1 |
| 31B | 300 | 52.0 | 78.6 | 52.0 |
| | 1,000 | 145.8 | 157.2 | 149.2 |
| | 3,000 | 407.4 | 435.4 | 415.5 |

- Back to back (each request sent as the previous answer arrived), `before` minus `after`, median of pairs: 12B +4.9, +22.1 and -36.5 ms; 31B +5.1, -80.9 and -212.4 ms at 300, 1,000 and 3,000 tokens. Under `after` a request waits behind the previous state's warm-up.
- A different follow-up question on 20 new 3,000-token states read the state from the cache on 20 of 20 at 0, 50, 200 and 1,000 ms after the answer, in every order. Under `after` it waited for the warm-up: 267.1, 221.7, 69.8 and 49.7 ms on the 12B, 440.6, 392.1, 243.8 and 46.0 on the 31B; under `before` 34.4 to 45.7 and 38.9 to 44.8.
- Under load (clients sending a first and a different question per new 1,000-token state), `after` against `before`: the 12B 0.99 and 0.98 of the throughput at 32 and 64 clients, p95 1.02 and 1.01 times; the 31B 0.61 and 0.74, p95 1.69 and 1.62 times. No errors; every registrar ended with 488 deferred and 488 registered, none dropped or failed.
- A first read against its repeat under `after`: up to 0.031 on the 12B, no choice changed; 0.0 on the 31B.
- Decision (Amin, 2026-10-07): `after` is the 12B's default and `before` the 31B's, with `--register-boundary` on both.

## The GPU tier (#99 at `c0d251f`)

- `test_warm_in_process.py` and `test_model_classes.py` passed; `test_suffix_staging.py` skipped (vLLM without the optional series).
- `test_second_question_cached.py`: 12B `after` and `before` and 31B `before` passed; Qwen `after` and 31B `after` failed (`gpu_tier/second_question_cached.txt`, run directly in `gpu_tier/diag/`):
  - Qwen `after`: part B counted 3,000 tokens per state, but each request occupies four 1,056-token blocks (the state padded to 3,168, and the question), so the "0.9 times the pool" fill overfilled it and the 8 earliest states were evicted; a fault of the test, fixed since by counting the blocks a request occupies.
  - 31B `after`: parts A and C passed (10 of 10 follow-ups from the cache); part B kept none of the 8 earliest states at 1.3 times the reported pool, where `before` kept all 8, which the load arm agrees with (82 of 96 follow-ups found their state at 32 clients, against 96 of 96); then the process aborted at exit (SIGABRT, "terminate called without an active exception") with the registrar's thread inside the engine. Fixed since: at exit the registrar waits for the warm-up in progress and starts no other.
- `test_serving_gates.py`: G3 passed; the separate part's G1 and G2 failed in the tier and passed when run directly (`gpu_tier/diag/gates_separate.stdout`); not reproduced, and the tier's output had kept only start-up lines. The harness now reports each child's verdict lines and the tails of stdout and stderr apart.
