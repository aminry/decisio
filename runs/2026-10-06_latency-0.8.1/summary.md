# Latency on 0.8.1, the boundary registration's open questions, and the GPU tier (session of 2026-10-06)

- **Card:** one RTX PRO 6000 Blackwell Workstation Edition, **power-limited to 400 W** (default and maximum 600 W; `gpu_power_clocks.txt`).
- **Host:** an **AMD Ryzen 9 9950X** (`host.txt`).
- **Software:** vLLM 0.30.0, decisio v0.8.1 as released, every checkpoint at its pinned revision.
- **Served defaults:** the engine in the server's process; Gemma 4 31B scoring several questions warm; the Gemma bases registering a single question's state boundary.
- **Method:** each base measured on the server decisio builds for `--base <base> --model <checkpoint>`, with uvicorn left out. Server time is the `x-decisio-server-ms` the HTTP route reports.
- The pre-registration and the session's record are in RLCD `experiments/2026-10-06_t3_latency_and_gpu_tier/`.

## The second question (since 0.8.1)

20 new 3,000-token states per base: a first question, then a different one in its own request.

| Base | First question | Second, different question | Read the state from the cache |
| --- | ---: | ---: | --- |
| Qwen3.6-35B-A3B | 85.2 ms | 22.5 ms | 20 of 20, with no warm-up (its padding keeps the boundary) |
| Gemma 4 12B | 347.8 ms | 29.2 ms | 20 of 20 |
| Gemma 4 31B | 615.3 ms | 37.1 ms | 20 of 20 |

## The registration's cost on a new state

20 states per size, each asked with the registration off and then on, the cache emptied before each.

| Base | State | Off | On | Extra (median of pairs) | Largest \|dp\| | Choice changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 12B | 300 tokens | 59.6 ms | 70.4 ms | +11.0 ms | 0.0658 | 0 |
| | 1,000 | 119.9 | 157.3 | +37.3 | 0.0658 | 0 |
| | 3,000 | 334.9 | 393.8 | +59.1 | 0.0276 | 0 |
| Gemma 4 31B | 300 | 91.8 | 115.1 | +23.1 | 0.0 | 0 |
| | 1,000 | 214.2 | 256.9 | +42.8 | 0.0 | 0 |
| | 3,000 | 595.1 | 667.6 | +72.4 | 0.0 | 0 |

Both arms pay for the 64-token system prompt, since the cache is emptied first. Their difference is the registration's cost; their absolute values are not the latency cells.

## The latency cells, as recorded (not the published tables)

Served defaults, in-process, server time, median of 20 (`measure_*.json`):

| Cell | Qwen | Gemma 4 12B | Gemma 4 31B (warm) |
| --- | ---: | ---: | ---: |
| 1 question, cached 1,000-token state | 18.5 ms | 23.4 ms | 30.2 ms |
| 1 question, new 300-token state | 50.7 | 68.3 | 110.3 |
| 4 questions, new 300-token state | 101.3 | 137.5 | 148.9 |
| 1 question, new 1,000-token state | 47.9 | 138.6 | 226.2 |
| 4 questions, new 1,000-token state | 98.8 | 211.7 | 263.8 |
| 1 question, new 3,000-token state | 85.4 | 347.4 | 614.7 |
| 4 questions, new 3,000-token state | 143.3 | 420.4 | 656.7 |

**Why they do not replace README's and EVAL_CARD's tables.** The Gemma bases' first reads here are slower than their 2026-10-04 records, even where the registration does not apply.
- The 12B's four-question cells: 137.5 / 211.7 / 420.4 against 131.0 / 195.8 / 391.6 ms.
- Its single questions with the registration off: 59.6 / 119.9 / 334.9 against 39.1 / 102.4 / 293.5 ms.

First reads are bound by the card, and this card ran at 400 W of 600. The records' power limits and hosts were not captured, so the gap is not fully accounted for.
The absolute tables keep their numbers, labelled with what is known; only the paired changes above are quoted from this session.

**The host's CPU.** A cached Qwen question in-process took 18.5 ms here and 39.4 ms on an AMD EPYC 7452 host (`runs/2026-10-05_engine-death-gates/`, same card model and vLLM). The 12B took 23.4 against 28.0 ms.

## The GPU tier on 0.8.1

| File | Result |
| --- | --- |
| `test_second_question_cached.py` | first run: 3 failed (the test's pool, below); rerun with the pool fixed: Gemma 4 12B and 31B pass parts A and B; Qwen passes part A and fails part B (below) |
| `test_warm_in_process.py` | 1 passed |
| `test_model_classes.py` | 1 passed |
| `test_serving_gates.py` | 2 passed (G4 skipped: no adapter) |
| `test_suffix_staging.py` | 1 skipped (vLLM without the optional series) |

**Two faults in `test_second_question_cached.py`, fixed in a test pull request:**
1. **The pool:** it read the pool as `num_gpu_blocks × block_size` (944,704 tokens on the 12B) instead of vLLM's reported 179,859, so part B first filled 6.8 times the pool. The reruns (`gpu_tier/second_question_cached_rerun_*.txt`) used `cache_config.kv_cache_size_tokens`.
2. **The order:** it asked the different question before the repeat, so a miss reloaded the state and the repeat then hit.

On Qwen at 1.3 times its 2.0-million-token pool, the 8 earliest states were evicted (the different question hit on 0 of 8; the repeat's 8 of 8 is the ordering artefact). Qwen registers nothing, and its reported pool is its capacity, so this is LRU eviction under overload, not a defect.
The Gemma bases' 8 earliest states hit on 8 of 8 for both questions.

Part A on each base: a first question, its repeat (answers equal), and a different question that read the state's whole hit units from the cache.
