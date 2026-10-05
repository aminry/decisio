# The engine-death gates, and the defaults they cleared (session of 2026-10-05)

One RTX PRO 6000 Blackwell Workstation Edition on a host with an AMD EPYC 7452 (Zen 2, up to 3.35 GHz; `host_lscpu.txt`), vLLM 0.30.0.
The servers ran decisio v0.7.2 plus a session-only patch that injects the faults below, never merged.
Every checkpoint was at its pinned revision.
"In-process" is the engine in the server's process (`--engine-process in`); "separate" is vLLM's own arrangement, the engine in a process of its own.
Verdicts follow the session's pre-registration with its two dated amendments (RLCD `experiments/2026-10-05_t3_engine_death_gates/`); `gates.json` holds them all.
The browser-agent demo's travel requests are published as aggregates only.

## What this cleared

- **`--engine-process` resolves to `in` for a single-engine server,** and to `separate` with a second engine (`--image-model`, `--head-engine`, `--one-engine`).
  Gate C: an engine fault ends the server in both arrangements.
  Gate B, carried over: under load the in-process engine served as fast as the separate one, with every answer unchanged.
  Two vLLM engines in one process failed every request in the earlier session (gate A), so `in` stays refused with a second engine.
- **The gemma-4-31b profile's `--multi-question` default is `warm`** when its engine runs in the server's process (gate D).

## Gate C: an engine fault, on the real engine (Qwen base)

| Arm | What happened |
| --- | --- |
| P, separate: a bug on decisio's side, the engine healthy | 10 requests answered 500 in 23 to 31 ms; the engine answered each probe; no exit; 200 of 200 conformance answers identical after |
| P, in-process | 10 requests answered 500 in 26 to 34 ms; no exit; 200 of 200 identical after |
| F1, separate: an exception inside the model | 503 in 38 ms (vLLM's `EngineDeadError`), the next two 503 in 3 ms; exit code 70 at 3.3 s; a healthy server again in 304 s; 200 of 200 identical |
| F1, in-process | the probe did not answer within 5 s (in-process vLLM hangs after the exception); 503 at 5.0 s, the next two in 6 to 7 ms; exit code 70 at 8.2 s; healthy again in 284 s; 200 of 200 identical |
| F2, separate: the engine-core process SIGKILLed | `/health` 503 at 2.0 s with no request; exit code 70 at 3.5 s; healthy again in 304 s; 200 of 200 identical |

The restarts come from compose's restart policy, emulated (Docker does not run inside the rented container).

**The probe's cost:** 18 to 21 ms separate (median 18), 19 to 23 ms in-process (median 20).
It is one forward pass over a one-token prompt, sent only after an unexpected exception.

## Gate B, carried over (decisio `993db22`, the earlier session's card)

| Clients | Throughput, separate / in-process | p95, separate / in-process | Errors | Answers identical to one client |
| ---: | --- | --- | ---: | --- |
| 32 | 15.8 / 16.0 requests per second | 2,344 / 2,311 ms | 0 | 1,672 of 1,672 |
| 64 | 15.8 / 15.9 requests per second | 4,427 / 4,441 ms | 0 | 1,672 of 1,672 |

## Gate D: gemma-4-31b, in-process, warm against sequential

| Set | Items | Choice changes | Largest \|dp\| | Median \|dp\| | Accuracy, warm / sequential |
| --- | ---: | ---: | ---: | ---: | --- |
| the suite, four-question form | 1,400 | 0 | 0.0 | 0.0 | 0.804 / 0.804 |
| JevBench's published items, four-question form | 231 | 0 | 0.0 | 0.0 | 0.913 / 0.913 |
| travel, four-question requests | 272 | 0 | 0.0073 | 0.0 | |

Warm repeated exactly on all 272 travel requests, 5 repeats each.
The four-question form is the item's own question first, then three fixed filler questions about the same state; the comparison is on the item's own question (on travel, on each of the four).

## Latency on this host: in-process against separate (Qwen base)

Each figure is the server time, median of 20; both arrangements ran on this host and card in one session.

| Cell | separate | in-process | in-process minus separate |
| --- | ---: | ---: | ---: |
| one question, cached 1,000-token state | 43.2 ms | 39.4 ms | -3.8 ms |
| one question, new 300-token state | 69.3 | 68.7 | -0.6 |
| four questions, new 300-token state | 197.9 | 179.4 | -18.5 |
| one question, new 1,000-token state | 77.6 | 71.2 | -6.4 |
| four questions, new 1,000-token state | 197.8 | 185.2 | -12.6 |
| one question, new 3,000-token state | 111.2 | 94.3 | -16.9 |
| four questions, new 3,000-token state | 247.2 | 215.0 | -32.2 |

**This host is slow for the Qwen base.** Its separate column is 1.4 to 2.1 times the README's record on the same card type (a cached question 43.2 ms against 20.9 in `runs/2026-10-04_gemma-base/`).
decisio v0.7.1 measured the same here (`lat_qwen_separate_v071.json`: 44.4 ms for the cached question; no cell lower by more than the two runs' spread), so the extra time is the host's, not 0.7.2's.
The Gemma bases were not slower here: a cached question took 28.0 ms on the 12B against 26.7 on record, and a new 300-token state 55.5 ms on the 31B against 61.8 (`lat_g12_in.json`, `lat_g31_warm_in.json`, both in-process).
The host's CPU is the likely cause, since the card, vLLM and the settings match the record's, but the record host's CPU was not captured.
So the absolute tables in README and EVAL_CARD keep their earlier numbers, labelled with their arrangement, and only the paired difference above comes from this host.

## Files

| File | What it holds |
| --- | --- |
| `gates.json` | every verdict, with gate B's record from the earlier session |
| `fault_*.json.gz` | each gate C arm: the timeline (requests, `/health` each second, exits, the restart), the conformance answers before and after |
| `d_{warm,sequential}_{suite4,jev4}.json.gz` | gate D's answers per item |
| `lat_*.json` | the latency cells: Qwen separate (v0.7.2 and v0.7.1) and in-process, gemma-4-31b in-process with warm, gemma-4-12b in-process |
| `health/` | each server's `/health` |
| `host_lscpu.txt` | the host's CPU |
