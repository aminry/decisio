# Latency on 0.8.1 at 585 W: the tables (session of 2026-10-06)

- **Card:** one RTX PRO 6000 Blackwell Workstation Edition, **power limit 585 W** while a process holds it (default and maximum 600 W; it reads 600 W with no process), driver 580.126.09 (`first_contact.json`, `gpu_power_clocks_first.txt`).
- **Host:** an **AMD Ryzen Threadripper 9960X**, 24 cores, 48 threads (`host.txt`; the container's name removed).
- **Software:** vLLM 0.30.0; decisio at `83e461d`, whose `src/` equals v0.8.1's; every checkpoint at its pinned revision.
- **Served defaults:** the engine in the server's process; Qwen and Gemma 4 12B `--multi-question sequential`, Gemma 4 31B `warm`; on the Gemma bases a single question on a new state registers its boundary before the question.
- **Method:** decisio's own server for `--base <base> --model <checkpoint>`, its uvicorn on 127.0.0.1:18000 in the measuring process, asked over HTTP by one client, one request at a time.
  Server time is the route's `x-decisio-server-ms`; client time, the HTTP round trip, is 0.7 to 2.3 ms more (both in `measure_*.json`).
- **First contact** (CPU, disk, power limit and clocks) and a CPU check before any cell: a cached Qwen question took 20.3 ms, against a threshold of 30 ms.
- The pre-registration, its amendments (the 570 W rule among them) and the session's record are in RLCD `experiments/2026-10-06_lab2_latency_600w/`.

## The cells (server time, median of 20)

| Cell | Qwen3.6-35B-A3B | Gemma 4 12B | Gemma 4 31B |
| --- | ---: | ---: | ---: |
| 1 question, cached 1,000-token state | 20.2 ms | 24.8 ms | 31.7 ms |
| 1 question, new 300-token state | 49.9 | 53.6 | 80.7 |
| 1 question, new 1,000-token state | 50.0 | 109.3 | 160.1 |
| 1 question, new 3,000-token state | 85.2 | 258.6 | 438.1 |
| 4 questions, new 300-token state | 110.3 | 123.3 | 104.8 |
| 4 questions, new 1,000-token state | 109.5 | 177.4 | 184.9 |
| 4 questions, new 3,000-token state | 148.3 | 336.8 | 466.9 |

Four questions in the other modes, on new states of 300 / 1,000 / 3,000 tokens:
- Qwen `warm` 59.8 / 60.8 / 101.9 ms, `batch` 52.6 / 53.7 / 89.5 ms;
- Gemma 4 31B `sequential` 170.1 / 252.9 / 537.6 ms, `batch` 84.5 / 161.3 / 825.5 ms.

## The second question

20 new 3,000-token states per base: a first question, then a different one in its own request.

| Base | First question | Second, different question | Read the state from the cache |
| --- | ---: | ---: | --- |
| Qwen3.6-35B-A3B | 85.1 ms | 23.5 ms | 20 of 20, with no warm-up (its padding keeps the boundary) |
| Gemma 4 12B | 260.0 ms | 35.8 ms | 20 of 20 |
| Gemma 4 31B | 439.0 ms | 43.3 ms | 20 of 20 |

## The boundary registration's cost on a new state

20 states per size, each asked one question with the registration off and then on, the prefix cache emptied before each.

| Base | State | Off | On | Extra (median of pairs) | Largest \|dp\| | Choice changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 12B | 300 tokens | 41.0 ms | 57.5 ms | +16.1 ms | 0.034 | 0 |
| | 1,000 | 88.8 | 111.1 | +24.7 | 0.113 | 1 |
| | 3,000 | 240.0 | 269.8 | +28.1 | 0.064 | 0 |
| Gemma 4 31B | 300 | 60.2 | 81.4 | +20.6 | 0.0 | 0 |
| | 1,000 | 152.7 | 186.6 | +34.0 | 0.0 | 0 |
| | 3,000 | 415.4 | 450.9 | +33.6 | 0.0 | 0 |

Both arms pay for the 64-token system prompt here, since the cache is emptied first; their difference is the registration's cost, and their absolute values are not the cells.
On the 12B a registered read and a fresh one differed by up to 0.113, and one choice of the 60 changed (EVAL_CARD 6.5); on the 31B they were identical.

## Power and clocks

`gpu_samples.csv` holds the card's draw, limit, clocks, temperature, utilisation and clock-event reasons once a second for the whole session (other parts of the session ran on the same card after these cells).
`power_by_stage.json` splits it by the runner's steps; for the cells:

| Step | Busy samples | Power-capped | Draw p50 / max (W) | SM clock p50 (MHz) | Temperature max (°C) |
| --- | ---: | ---: | --- | ---: | ---: |
| Qwen | 141 | 3 (2%) | 269 / 365 | 2,797 | 60 |
| Gemma 4 12B | 110 | 90 (82%) | 529 / 591 | 2,580 | 81 |
| Gemma 4 31B | 219 | 192 (88%) | 502 / 591 | 2,467 | 86 |

A busy sample has the card 50% or more utilised; a power-capped one has the software power cap among its clock-event reasons.
On the Gemma bases the 585 W limit held the clock back on most busy samples even one request at a time, so a card at 600 W may read new states faster; on the Qwen base it hardly did.
The CPU matters for the Qwen base instead: a cached Qwen question took 39.4 ms in-process on an AMD EPYC 7452 host, 18.5 ms on an AMD Ryzen 9 9950X and 20.2 ms here, the same card model and vLLM (`runs/2026-10-05_engine-death-gates/`, `runs/2026-10-06_latency-0.8.1/`).

## Against the earlier records (not paired)

- Track 3's session on the Ryzen 9 9950X at 400 W (`runs/2026-10-06_latency-0.8.1/`): Qwen within about 2 ms in every cell; the Gemma first reads at 3,000 tokens 258.6 against 347.4 ms (12B) and 438.1 against 614.7 ms (31B).
- The records these cells replace in README and EVAL_CARD were measured with the engine in its own process, on hosts whose CPU and power limit were not recorded (`runs/2026-09-30_plugin-verification/`, `runs/2026-10-02_multi-question-and-rendering/`, `runs/2026-10-04_gemma-base/`, `runs/2026-10-04_gemma-4-31b/`).
