<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Running decisio

Every way to run the server, in the order of the README's chooser ("Install and run").
The README has the short version of each; this page has what it leaves out.
Every number names the record it comes from.

## GPU server

If a first start fails (a card too small, a driver, a busy port, a download), [`troubleshooting.md`](troubleshooting.md) has the messages and the fixes.

Requirements: Linux, one NVIDIA card (measured on an RTX PRO 6000 Blackwell with 96 GB), a driver that supports CUDA 13.0 (the runtime `uv.lock` pins), Python 3.12 and [uv](https://docs.astral.sh/uv/).

<!-- x-release-please-start-version -->
```bash
git clone --branch v0.24.0 https://github.com/aminry/decisio
cd decisio
uv sync --extra serve --frozen
uv run python -m decisio.serve.vllm_engine
```
<!-- x-release-please-end -->

- With no flag the server serves Gemma 4 31B (`--base gemma-4-31b`), the default base, from its FP8 repository `tachara-ai/decisio-gemma-4-31b`; it is measured in full on a 96 GB card, and it also started and answered on an 80 GB H100 ("The context length" below).
  `--model google/gemma-4-31B-it` serves Google's weights at the pinned revision instead, quantized to FP8 each time they load (`--revision` does the same, naming a revision of Google's checkpoint).
- A third party served the 31B on a smaller card than ours: JevBench's maintainer ran decisio 0.8.0 with `--base gemma-4-31b` (Google's weights, quantized to FP8 on load) on one H100 80 GB (Lium) and "no flag of theirs was changed to make it fit".
  That is the board's own account of its run of 2026-10-06 on the v1.6.0/v1.6.1 pool, in the row `decisio-gemma-4-31b-v080` (JevBench v1.6.1, `https://benchmarkheaven.com/api/jevbench/v1.6.1`); we did not run it. It agrees with our own 80 GB record ("The context length" below): the 31B is measured in full on a 96 GB card, and it also runs on an 80 GB H100.
  The row also notes that 23 of the pool's items, of about 80,000 tokens, are over the 32,768-token context.
- `--base qwen3.6-35b-a3b` serves `Qwen/Qwen3.6-35B-A3B-FP8` and `--base gemma-4-12b` serves `google/gemma-4-12B-it`, each at its pinned revision and with its own settings (the README's "Choosing a base"; `docs/cli.md` lists every setting per base).
  `--base gemma-4-12b-fp8` serves the same 12B checkpoint in FP8 ([The 12B in two precisions](#the-12b-in-two-precisions)).
- With a `--model` and no `--base`, the base is detected from the checkpoint's `config.json`, so a local copy of any base's checkpoint brings its own settings; a base's pinned revision applies whenever `--model` names the base's own Hugging Face repository and `--revision` is not given.
- The first start downloads the checkpoint into the Hugging Face cache and warms the engine: about 31 GB for the 31B's FP8 repository (62 GB for Google's bf16 weights) and about 36 GB for the Qwen base (`runs/2026-10-01_docker-first-gpu-start/`); `/health` answers once it is ready.
- Upgrade notes: `docs/upgrading.md`.
- The decisio plugin registers its model classes with vLLM through an entry point, so vLLM 0.31.0 loads them without any patch; the package must be installed (as `uv sync` does) for vLLM's engine processes to find it (`docs/design/vllm-plugin.md`).
- `patches/` holds an optional latency patch series, off by default and not needed for correct answers (`patches/README.md`).
- The server refuses to start when `VLLM_USE_DEEP_GEMM` is set to anything other than `0`, unless `--allow-deep-gemm` is given.
- It listens on `127.0.0.1:8000` (`--host`, `--port`); put a reverse proxy in front of it to expose it.
- `--image-model` adds a second engine on the same card for requests that carry images. The two engines default to the shares they were measured with, 0.47 for the text engine and 0.49 for the image engine (`EVAL_CARD.md` section 1); a pair that cannot fit, such as the text engine given at 0.90 beside the image engine's default, is refused at start, before either is loaded; the start prints the shares it uses (`ENGINE SHARES`). `--one-engine` serves both routes from the image engine alone, at `--gpu-memory-utilization`.
- vLLM's engine runs in the server's process by default; with a second engine (`--image-model`, `--head-engine`, `--one-engine`) it runs in a process of its own (`--engine-process`, `docs/cli.md`).
- A later, different question about a state read before, sent in its own request, reads the state from the prefix cache; on the Gemma bases that needs the state's boundary registered, and `--register-boundary` sets when (below).
- vLLM's `prefix_cache_retention_interval` would also keep the boundary, but it keeps every sliding-window block of each state, and with the cache filled to 1.3 times its pool the earliest states lost even their repeated question, where the default (0) kept them at twice the pool; it stays at the default. The Qwen base needs neither: its state is padded to end on its 1,056-token block, the latest checkpoint of a request with one question. `/health` reports `cache_hit_unit`, `hash_unit`, `registers_state_boundary` and `register_boundary`.

Per-request latency depends on the host's CPU and on the card's power limit, as well as the card.
- The CPU: with the engine in the server's process, a Qwen question on a cached state took 39.4 ms of server time on an AMD EPYC 7452 (Zen 2) host, 18.5 ms on an AMD Ryzen 9 9950X (Zen 5) and 20.2 ms on an AMD Ryzen Threadripper 9960X, with the same card model and vLLM (`runs/2026-10-05_engine-death-gates/`, `runs/2026-10-06_latency-0.8.1/`, `runs/2026-10-06_latency-585w/`). The Gemma bases differed far less between hosts (Gemma 4 12B: 28.0, 23.4 and 24.8 ms).
- The power limit: first reads on the Gemma bases are bound by the card. With the card's enforced limit at 585 W (default 600 W), the software power cap held the clock back on 72 to 92% of busy samples on the Gemma bases, single requests included, and on 0 to 2% on the Qwen base (`runs/2026-10-06_latency-585w/`); at 400 W their first reads were slower still (`runs/2026-10-06_latency-0.8.1/`). The latency figures in the README and `EVAL_CARD.md` are from the 585 W session.

Quote latency with the host's CPU model and the card's power limit beside it (`nvidia-smi -q -d POWER`).

### First-question latency by state length

One question about a state the server has never seen (**first**), and a different question about the same state asked next (**second**), by the state's length in tokens. Server time (`x-decisio-server-ms`), median of 10 new documents per length, on one RTX PRO 6000 Blackwell Workstation Edition at 600 W (driver 595.104.02) with an AMD EPYC 9534 host, decisio 0.11.1 on vLLM 0.30.0, each base at its repository's pin, the engine in the server's process, the Gemma bases at `--register-boundary before`; the SM clock ran at a median of 2,280 to 2,380 MHz on the 31B with the software power cap active on 66 to 83 percent of busy samples (`runs/2026-10-14_s11-session/L/` and `stage_power_clock.md`).

| State, tokens | 31B first, ms | 31B second | 12B first | 12B second | Qwen first | Qwen second |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 80.6 | 29.6 | 55.0 | 23.4 | 47.5 | 25.0 |
| 1,000 | 170.7 | 37.0 | 110.8 | 27.6 | 48.8 | 26.6 |
| 3,000 | 485.4 | 41.4 | 278.7 | 33.9 | 86.5 | 32.6 |
| 6,000 | 1,033.9 | 54.0 | 568.4 | 46.6 | 153.9 | 41.0 |
| 16,000 | 3,370.1 | 93.5 | 1,779.0 | 85.2 | 452.4 | 66.2 |
| 32,000 | 8,880.3 | 161.6 | 4,382.9 | 150.9 | not measured | not measured |

- The first question grows roughly linearly with the state: about 0.30 ms per token for the 31B and 0.14 for the 12B between 6,000 and 32,000 tokens (the Qwen base reads fastest, 453 ms at 16,000).
- The second question stays under 165 ms on every length and base.
- The Qwen base has no 32,000-token value: the run returned none (the probe crashed on the missing figure; the likely cause is that a 32,000-token state plus the 77-option question does not fit the 32,768-token context, which was not confirmed from the server log).
- This is one box and one session; the README's and `EVAL_CARD.md`'s tables are from the 585 W session of 2026-10-06.

### A later question about the same state (`--register-boundary`)

On the Gemma bases vLLM keeps only a finished request's latest sliding-window checkpoint, which lies inside its question.
So the server registers each state's boundary with the warm-up that multi-question requests already send: the state and one token.
Without it, a later, different question about the state reads the whole state again, which costs about a first read of it: on the 31B at 3,000 tokens, 439.0 ms for the first read against 43.3 ms for a different question read from the cache (one card at 585 W, an AMD Ryzen Threadripper 9960X; `runs/2026-10-06_latency-585w/`).
The one measurement of that at 31,000 tokens (decisio 0.8.0, on another host) is 9.2 s instead of 0.17 (531 ms instead of 47 at 3,000 tokens there).
The server remembers the last 4,096 states it registered and sends the warm-up again only when a request finds the boundary gone, so a repeated question costs no extra engine call.
Several questions in one request are not affected, and the Qwen base needs no registration (its padded state already ends on a single question's latest checkpoint).

`--register-boundary` sets when a single question on a new state registers it:
- `after`: the question is answered first, reading the state fresh, and the warm-up is sent once the response has been handed to the server (in library use, once `answer()` returns). The caller does not wait for it.
- `before` (the default of both Gemma bases): the warm-up goes ahead of the question, in the caller's request. On one card at 585 W with an AMD Ryzen Threadripper 9960X that added +16, +25 and +28 ms to a first read at 300, 1,000 and 3,000 tokens on Gemma 4 12B, and +21, +34 and +34 ms on Gemma 4 31B (`runs/2026-10-06_latency-585w/`, and `register_boundary.md` there for every figure below).
- `off`: never; a later, different question about a state reads it again. For traffic that asks one question per state, where a registration would only cost throughput.

Why the 12B's default is `before`, not `after`: our grid of 2026-10-09 (12B, `after`, AMD EPYC 9534, RTX PRO 6000 at 600 W, decisio 0.10.0; measured by the maintainers, the record is not published) found that one question on each new state, sent back to back, took 774 ms at 3,000 tokens instead of about 390 and 1.6 s instead of 0.8 at 6,000, because each request waited behind the previous state's registration (385 and 796 ms outside the engine's stages, one read of the state); `batch` mode, three-question requests and the 31B's `before` showed no gap. `before` reads the state once plus one token inside the request, so nothing is pending behind a response. A direct check then confirmed it (2026-10-09, one server on decisio 0.10.0, measured by the maintainers; the record is not published): one question per new state, ten states, server time of the first request and the median of the nine later ones. Back to back, `after` took 378 and 771 ms at 3,000 tokens and 812 and 1,613 ms at 6,000; `before` 413 and 417 ms and 834 and 834 ms; `off` 383 and 395 ms and 816 and 818 ms. With the client waiting for the registrar between states, `after` took 396 and 396 ms and 818 and 827 ms. So the doubling is only under `after` and only back to back; `before` costs 22 ms (3,000 tokens) and 17 ms (6,000) over `off`. `off` shows the cost of no registration and nothing else: it is the baseline the other two are read against, and it does not show how a later, different question fares, since without a registration that question reads the state again. The response's `timing.state_boundary.ran_before` was 0 on every request of every order: it counts only a registration the request sends itself, and no field records the wait for the registrar's background warm-up.

Why `after` is still there: on the same card, with the queue idle, `after` took the registration off the caller's first read on both bases (12B 33.3, 90.3 and 237.1 ms against 56.7, 107.7 and 261.9 at 300, 1,000 and 3,000 tokens; 31B 52.0, 149.2 and 415.5 against 78.6, 157.2 and 435.4).
Under load, with 32 and 64 clients sending first and follow-up questions about new 1,000-token states, the 12B kept 0.99 and 0.98 of the throughput it had with `before`, but the 31B kept only 0.61 and 0.74, with p95 1.69 and 1.62 times; so the 31B registers before the question.

What `after` guarantees, and its limits:
- A follow-up question sent after the first answer reads the state from the cache.
  A registration that is due goes before every later request's questions: a background thread sends it as soon as the engine is free, and a request that takes the engine first sends it ahead of its own questions.
- The bound on that follow-up: it may wait, at most for the engine call already in progress and then the warm-up, which reads the whole state again, since vLLM kept no checkpoint at the boundary.
  The warm-up costs about one first read of the state: 34, 88 and 238 ms at 300, 1,000 and 3,000 tokens on Gemma 4 12B, and 52, 146 and 407 ms on Gemma 4 31B (the reads without registration).
  Measured on 3,000-token states, a follow-up read the state from the cache on 20 of 20 at 0, 50, 200 and 1,000 ms after the answer, and took 267, 222, 70 and 50 ms on the 12B (441, 392, 244 and 46 ms on the 31B), against 34 to 46 ms with `before`.
- A second question sent before the first answer is out sends the warm-up itself, ahead of its question, as `before` does.
- A follow-up reads the state again only when its registration was dropped (more than 256 pending, the oldest dropped first), when its warm-up failed, or when the state was evicted from the cache since.
  `/health` reports `state_boundary`: the registrations pending, whether one is `running` (being sent: it is no longer pending then, so the registrar is idle when `pending` is 0 and `running` is false), and how many were deferred, registered, dropped and failed.
- The cost moves from latency to throughput: a new state is read twice on the card, once for the answer and once for the warm-up, where `before` read it once plus one token.
  A request about another state that arrives during a warm-up waits for it, so one client sending new states back to back pays the previous state's warm-up: on the 12B it saved 4.9 and 22.1 ms at 300 and 1,000 tokens and lost 36.5 ms at 3,000.
- The cache can hold fewer states under `after`: on the 31B, with states filling 1.3 times its reported pool, the 8 earliest were evicted under `after` and kept under `before`, and at 32 clients 82 of 96 follow-ups found their state against 96 of 96; the 12B kept the 8 earliest under both orders.
- When the process ends (a library caller returning, or the server stopping), the warm-up in progress is waited for, at most 60 s, and the pending ones are dropped: an interpreter that shut down under a warm-up aborted the process on a card.
- The first answer about a new state is a fresh read again.
  On Gemma 4 12B a fresh read and a cached read of the same question can differ, by up to 0.113, and one near-tied choice of 60 new states changed (`EVAL_CARD.md` 6.5), so with `after` a question asked again can differ from its first answer by that much; with `before` the two are equal.
  On Gemma 4 31B the two reads were identical.
- Each request's `timing.state_boundary` shows what it did: `registered` (warm-ups sent ahead of its question), `found` (states already registered), `deferred` (warm-ups sent after its response), and `ran_before` and `ran_before_ms` (other requests' due warm-ups it sent first, and their time).
  `ran_before` counts only a registration that request sends itself. The time a request waited for the engine, behind another request or behind a background registration, is `timing.waited_ms` (`wait=` in the `x-decisio-stages` header), and `timing.waited_behind_registration` (the header `x-decisio-wait-behind: registration`) says when a registration was running as it arrived.

## Model repositories

A decisio model repository on Hugging Face holds one base's weights, its licence and notice files, a card and `decision_config.json`: the base's profile as the decisio release it was made for serves it (`decisio.hub`).
```bash
uv run python -m decisio.serve.vllm_engine --base tachara-ai/decisio-gemma-4-31b            # its current commit, printed
uv run python -m decisio.serve.vllm_engine --base tachara-ai/decisio-gemma-4-31b@<commit>   # a pinned commit
```
- The repository's file names the base, and the repository's weights are the checkpoint, so `--model` is not given with it.
- A repository of a tuned model (an adapter merged into the base) has a file of schema `decisio.decision_config/2` that also carries the temperatures and the commit setting the model is served at ([Shipping a tuned model](#shipping-a-tuned-model)).
- The server compares the file's profile (prompt, readout, temperatures, serving defaults, the classes, the vLLM version the plugin supports) with the installed one, and refuses to start on a difference, naming the keys and the decisio release the file was made for.
  A profile change therefore reaches a repository only as a new revision of it, made for the release that changed it.
- A stored FP8 checkpoint is not quantized again on load (the Gemma 4 31B profile quantizes Google's bf16 weights to FP8 when it loads them; a repository holds the FP8 tensors).
- Tasks are named for the checkpoint a repository copies, not for the repository, so a task registered under `google/gemma-4-12B-it` serves from the copy, and the other way round.
- `/health` reports the repository, its revision, the decisio release its file was made for and what its weights are (`weights.modified` is false for a byte-for-byte copy).
- `python -m decisio.hub export --base <base>` prints the file for a base, and `python -m decisio.hub check <file>` compares a file with the installed profile; the files for this release are in `hub/`, and a test fails when a profile no longer exports them.
- With the repository's revision pinned in a release, `--base gemma-4-31b` with no checkpoint named is served from it, and so is the default: the server fetches `decision_config.json` at that revision, checks it against its own profile as above, and does not quantize the stored FP8 weights again.
  Naming a checkpoint (`--model google/gemma-4-31B-it`) or a `--revision` serves that checkpoint instead.
  Tasks are named for Google's checkpoint either way, so a task registered under one serves on the other.
- The Gemma 4 31B repository holds FP8 weights, made from Google's bf16 ones by `python -m decisio.hub_fp8 convert` exactly as the server quantizes them on load: one scale per fused layer, `float32(amax) / 448`, the weight times the fp32 reciprocal of the scale, round to nearest even.
  The conversion also runs vLLM's loader step (its requantization is the same code in 0.30.0 and 0.31.0; the identity gate has been run on a card with 0.30.0 only) (it requantizes a layer's shards with the largest of their scales, through float16) and reports how many weights loading would change; zero means the loaded tensors equal the ones made on load.
  `tests/gpu/fp8_fingerprints.py` reads the FP8 weights an engine holds, and `python -m decisio.hub_fp8 expected|compare` predict and compare them.
- The three repositories are public on the Hub, at revisions pinned in `decisio.hub.PINNED_REVISIONS`: `tachara-ai/decisio-gemma-4-31b`, `tachara-ai/decisio-gemma-4-12b` and `tachara-ai/decisio-qwen3.6-35b-a3b`.
  Each carries the licence, a card and `SHA256SUMS`; the 12B and Qwen files are byte-for-byte copies of Google's and Qwen's, and the 31B's text model is quantized to FP8 (its card says so).
- The 12B's FP8 conversion has a repository of its own, `tachara-ai/decisio-gemma-4-12b-fp8`, served with `--base tachara-ai/decisio-gemma-4-12b-fp8` at the revision this release pins ([The 12B in two precisions](#the-12b-in-two-precisions)).
  `--base gemma-4-12b-fp8` alone keeps quantizing Google's bf16 weights on load, to the same tensors (checked on a card: below).

## The 12B in two precisions

The Gemma 4 12B is served in bf16 (`--base gemma-4-12b`) or in FP8 (`--base gemma-4-12b-fp8`), from the same checkpoint.
bf16 is the default and the precision of every 12B record in this repository: the figures in the README, `EVAL_CARD.md` and `runs/` are bf16 unless they say otherwise.
FP8 is a second base beside it, for a smaller footprint: its linear layers are quantized to FP8 when they load, per tensor with per-token activations, as the 31B's are.

```bash
uv run python -m decisio.serve.vllm_engine --base gemma-4-12b-fp8                                   # Google's weights, quantized on load
uv run python -m decisio.serve.vllm_engine --base tachara-ai/decisio-gemma-4-12b-fp8                # the stored conversion, once pinned
uv run python -m decisio.hub_fp8 convert --source <bf16 12B dir> --out <converted dir>
uv run python -m decisio.serve.vllm_engine --base gemma-4-12b-fp8 --model <converted dir>           # a local conversion
```

- `--base gemma-4-12b-fp8` alone serves `google/gemma-4-12B-it` at its pinned revision and quantizes it on load.
- The stored conversion, made by `python -m decisio.hub_fp8 convert` as the 31B's is, holds the same tensors: on the real 12B the conversion predicts that loading it changes 0 weights in all 192 fused layers and that the tensors it loads to are the ones quantizing on load makes (`fp8_conversion.json` beside it), and on a card (an RTX 5090, vLLM 0.31.0) the engine's own FP8 tensors (`tests/gpu/fp8_fingerprints.py`), read from a server that quantized the bf16 12B on load and from one that loaded the conversion, differed in zero tensors from each other and from the prediction; a tuned 12B merged to FP8 (`hub_fp8 merge`) loaded to the tensors its merge predicted.
  It is 12.13 GiB on disk, against 22.3 GiB for the bf16 checkpoint, and it is loaded as stored, not quantized again.
  The repository `tachara-ai/decisio-gemma-4-12b-fp8` holds it, at the revision this release pins; `--model <converted dir>` with `--base gemma-4-12b-fp8` serves a local conversion.
- Without `--base`, a 12B checkpoint is served as `gemma-4-12b` (bf16) unless its `config.json` declares FP8 weights, which makes it `gemma-4-12b-fp8`, loaded as stored.
  `--base gemma-4-12b` with an FP8 checkpoint is refused, naming the FP8 base, and a repository whose file names the FP8 base over a checkpoint that declares no FP8 weights is refused too.
  The start prints a `BASE` line with the base, its precision and how it is loaded (for example `BASE gemma-4-12b-fp8 (fp8, quantized to FP8 on load) from google/gemma-4-12B-it`).

What changes with FP8:
- **The answers.** FP8 weights and activations are not the same function as bf16, so the FP8 12B's answers are not the bf16 12B's. Measured on an RTX 5090 (vLLM 0.31.0, both at 4,096 tokens and 0.95 of the card, 200 single-question requests, `runs/2026-10-21_5090-fp8-12b/`): the top answer was the bf16 12B's on 96% of them (8 changed), the largest probability difference 0.438 and the mean 0.031, where two fresh starts of either precision were bit for bit the same. Every other 12B figure in this repository is bf16.
- **The speed.** On the same card and length its server time for one question was 0.64 times the bf16 12B's (median 88.4 against 135.2 ms, paired on 40 texts).
- **The stack identity.** The FP8 12B has its own served name (`decisio-gemma-4-12b-it-fp8-letters`), task fingerprint and serving stack: a task registered on the bf16 12B does not apply to it, and a certificate made on bf16 is suspended on it (`serving_stack_changed`, [tasks.md](tasks.md#where-it-stops)). Register and certify again on the FP8 base.
- **The temperatures** are the bf16 12B's (3.592 for every question type) until they are refitted on FP8 reads.
- **The context length** follows the same rule as the bf16 12B, with more room for the KV cache: on a 32 GB RTX 5090 at the default share the FP8 12B serves its full 65,536 tokens, where the bf16 12B does not start at 32,768 and starts at 0.95 of the card only at 4,096 ([The context length](#the-context-length)).
- **MLX does not serve it:** FP8 is vLLM's format, and `--backend mlx` refuses the FP8 base; on a Mac, serve an MLX conversion of the bf16 12B ([Mac with MLX](#mac-with-mlx)).

## Shipping a tuned model

A model made by merging a LoRA adapter into the Gemma 4 31B (stored in FP8) or the Gemma 4 12B (stored in bf16 as `gemma-4-12b`, or in FP8 as `gemma-4-12b-fp8`) is shipped as a repository of its own, with a `decision_config.json` of schema `decisio.decision_config/2`.
A plain request to it, with no flag and no `adapter` field, is already fixed: the repository carries the temperatures, the choice bands and the yes/no commit setting that the adapter needs, and the server applies them.
This is how a harness that sends no adapter field and no flag reaches a tuned model: it is told `--base owner/name`.
`--base gemma-4-31b`, `--base gemma-4-12b` and `--base gemma-4-12b-fp8` are still the bases' own checkpoints.

**Making it**, on any machine that holds the bf16 base and the adapter's directory:

```bash
# 1. merge in fp32 and round once: the 31B to FP8, the 12B to bf16 or to FP8 (a card is not needed; --device cuda is faster)
uv run python -m decisio.hub_fp8 merge --source <bf16 base dir> --adapter <peft dir> --out <merged dir>      # 31B, 12B in FP8
uv run python -m decisio.hub_bf16 merge --source <bf16 base dir> --adapter <peft dir> --out <merged dir>     # 12B in bf16
# 2. fit the temperatures on the MERGED model's own reads of items that are not a benchmark's
uv run python -m decisio.serve.adapter_temperatures fit-split --h H.jsonl.gz --h-keys h.json --dev D.jsonl.gz --dev-keys d.json --variant dev --out temperatures.json
# 3. write the repository's file, stating the commit setting, and check it
uv run python -m decisio.hub tuned-config --base gemma-4-31b --temperatures temperatures.json --noul-commit \
    --noul-commit-basis basis.json --adapter-dir <peft dir> --adapter-repository <owner/adapter> --adapter-revision <commit> \
    --merge-report <merged dir>/MERGE_FP8.json --weights-dir <merged dir> --out <merged dir>/decision_config.json
#    for the 12B in bf16: --base gemma-4-12b and --merge-report <merged dir>/MERGE_BF16.json; in FP8: --base
#    gemma-4-12b-fp8 and MERGE_FP8.json; --think-factor F turns thinking on for the model and --over-limit truncate
#    answers states over its prompt ceiling from their start
uv run python -m decisio.hub check <merged dir>/decision_config.json --weights-dir <merged dir>
```

- **The 12B's merge** (`hub_bf16`) adds the same change in fp32 and rounds once to bf16, **stochastically**: each entry goes to one of its two bf16 neighbours with the probability that makes its expected value the fp32 value, with noise seeded per tensor (`sha256(seed | tensor name)`, so the result is repeatable and independent of the order; `--seed`, default 0, goes in the report and the repository's file). Rounding to nearest cannot carry a small adapter: the base is already on the bf16 grid, so it rounds the change away for most entries (it kept 0.88 of one real 12B adapter's change in weight space, and about half of a smaller one on the 31B); the stochastic merge kept 1.0000 of the 12B's (`runs/2026-10-20_tuned-12b-merge/`). Every other tensor is copied byte for byte, `--null` equals the source byte for byte, and the K=V rule below holds. The repository is bf16 (22 GB) and nothing is quantized on load.
- **The 12B's merge to FP8** is the 31B's (`hub_fp8 merge`, below) on the 12B's bf16 base, and its repository is a `gemma-4-12b-fp8` one: FP8 weights, not quantized on load, its own identity beside the bf16 tuned 12B's. Each 12B refuses the other's facts: `tuned-config --base gemma-4-12b` refuses an FP8 merge report or a weights directory whose `config.json` declares FP8, and `--base gemma-4-12b-fp8` a bf16 one (output bf16, stochastic rounding, a seed), each naming the base they belong to; `check --weights-dir` refuses a directory of the other precision. A tuned 12B's gates and temperatures read on bf16 do not carry over to its FP8 merge: they are read again on the FP8 model.
- **The 31B's merge** adds `scale * (B @ A)` to each weight in fp32 and quantizes once, with the adapter's change inside each fused group's amax; a merge through bf16 would round most of a small change away. Every pair of the adapter must land on a tensor of a quantized group. `--null` merges an adapter of scale 0, which must equal `hub_fp8 convert` byte for byte. The full-attention layers of the 31B and of the 12B ship no `v_proj` (`attention_k_eq_v`: V is K), so an adapter's `v_proj` pair there has no tensor to land on. The trainer's export writes such a pair as a copy of the layer's `k_proj` pair; the merge checks that it is bit for bit that copy (both halves), leaves it out, and counts it in the report and the repository's file (`dropped_k_eq_v_pairs`): merging the `k_proj` delta already gives V that delta, so nothing of the adapter is lost. A `v_proj` pair that differs from its layer's `k_proj` pair asks for a V that differs from K, which this layout cannot hold, and the merge refuses with the difference.
- **The file** keeps the base's profile exactly (a repository cannot change how the base is read) and adds `tuned` (the base, the adapter with its digests and revision, the merge report, every file by sha256) and `defaults` (`temperatures` per question type, `choice_bands`, `noul_commit` always stated true or false with its basis, `over_limit` and `think` when the model says them, and how the temperatures were fitted: the method only. `tuned-config` drops the fit tools' notes on data, counts and details, so neither the file nor anything generated from it says what a model was trained or calibrated on; the repository's card carries one generic data statement of its own).
- **Precedence at start:** a flag, then the repository's `defaults`, then the base's profile. `--temperature-choice X` is one number for every choice question, so it drops the repository's bands. `--noul-commit` and `--no-noul-commit` override the commit setting either way. The start prints `REPOSITORY DEFAULTS ...` with what was applied, and `/health` reports it under `systemone.repository_defaults` with the digest of `tuned` and `defaults`.
- **Fingerprint and tasks:** the digest and the bands are in the serving-stack fingerprint, so a certificate made on the tuned model is suspended by any change of either. Tasks are named for the repository and its defaults, so a task fitted on the base is not applied to the tuned model, nor the other way round.
- **An older decisio refuses the repository** by its schema (it reads `/1` only), instead of serving the model at the base's temperatures without saying.
- The commit setting is a stated choice of the model, shown in the repository's card and in `/health`: with it on, a yes/no probability between 0.20 and 0.80 is reported as 0.80 or 0.20 (the answer never changes; [cli.md](cli.md)).

## Thinking when unsure

Off by default: a request is one read, as fast and as repeatable as before.
`--think`, a request's `"think": true`, or a tuned repository's `defaults.think` turns on the procedure of Chon (github.com/xoai/chon, Apache-2.0), with the served model writing its own reasoning:

1. The question is read as always and given its type's temperature. It is **unsure** when the served top probability (yes/no: the larger of P(yes) and P(no)) is below the type's threshold times `--think-factor`: 0.75 for a choice, 0.92 for a yes/no, 0.70 for a score; and it has at most 16 options.
2. An unsure question is sent again with `Let me check the deciding facts first.` written inside the reply. **The same model that is served writes** up to `--think-budget` (512) tokens, greedily: the adapter the request names, or the weights as loaded (the merged weights of a tuned repository). Then `Final answer:` follows and the options are read from the next token. All the unsure questions of a request go in one call.
3. The two reads are mixed in log space (the mean of the two log-probabilities, normalised) and the mix goes through the temperature a single read would get (the type's, the band's for a choice), and then the commit rule where there is one. No temperature of its own, no new parameter.

No history decides anything: whether a question thinks depends on that question and the settings alone, so the same request thinks the same way whatever came before it. A lower factor thinks on fewer questions; the factor of a tuned model is chosen so that the share of an ordinary mix that thinks stays small (a thinking question costs seconds where a single read costs milliseconds, and the median request should stay a single read).
What stays at one read: a question a registered task answers (its calibration and certificate are fitted on one read; [tasks.md](tasks.md)), a question with more than 16 options, and every question of a request of two orders, an abstention option or the image route. The response says which questions thought and what it cost (`thinking`, `usage.thinking_tokens`, `usage.output_tokens` counting the reasoning tokens, `x-decisio-think-ms`; [api.md](api.md)) and `/health` counts them under `systemone.think`.
Thinking needs the vLLM engine; the MLX build cannot write reasoning, and `--think` there is refused at start. With decisio's hidden-readout class (the served default) it also needs the engine in the server's process, `--engine-process in`: the class keeps the hidden state in reserved logit columns, which the writing call switches off for its duration and which would otherwise be sampled; `--think` with another engine process is refused at start. The server's lock is held for the whole call, so another request waits behind a thinking one: use it where requests arrive one at a time, or size for that. The reasoning is greedy text from a forward pass that does not repeat bit for bit between fresh starts, so a thinking answer can move more between two starts than a single read does. A tuned repository turns it on with `"think": {"factor": 0.79, "budget": 512}` in its `defaults` (`python -m decisio.hub tuned-config --think-factor 0.79`); an older decisio refuses such a repository.

## States over the limit

A request is a state and its questions; each question is one row, the state followed by the question. The engine refuses a row longer than its prompt ceiling, which on the vLLM engine is the context length served (`/health` shows it as `max_model_len`; [The context length](#the-context-length)) less the one generated token. By default the server passes that refusal on as a 422 that says the rendered prompt is over the context.

`--over-limit truncate` answers instead. The server cuts the state once for the whole request, to what the longest question leaves, and answers every question from the same cut state (the questions share one state prefix, so the cache still works). The rules:

- **The state is cut, never a question.** The instruction, the options and the closing line are what the caller asks; the state is the evidence.
- **The start is kept.** A cut state is a prefix of the full one, so the same state at two ceilings nests, and the rule is one sentence: everything after a token is dropped. The cut backs up to whitespace within 64 tokens so it does not end mid-word; no marker is put in the state. A margin of 32 tokens (and the block padding) stays free for the second order's rows.
- **The response says what was cut.** `state_truncated` in the body and the header `x-decisio-state-truncated` carry the state's tokens kept and in all, the part and the ceiling ([api.md](api.md)). `/health` counts truncated requests under `systemone.over_limit` with the largest state seen.
- **Nothing cut is certified.** Every answer of a certified task in a truncated request is `certified: false` with the reason `state_truncated`, and the traffic monitor does not count it. A calibration or head the task has is still applied, as the best available correction, but the answer says it is not covered. Registration and certification refuse an over-limit example.
- **A question over the ceiling alone is still a 422**, and so is an image request: neither is cut.

Refuse stays the default. A refusal says what is wrong and costs the caller one retry; an answer from part of a document is made with a confidence that nothing was fitted on. Turn it on for one request with `"over_limit": "truncate"`, for a server with `--over-limit truncate`, or for a tuned model with `over_limit` in the repository's `defaults` (`python -m decisio.hub tuned-config --over-limit truncate`), shown at start and in `/health`. A flag beats the repository's default, and a request's field beats both. The setting is not part of the serving-stack fingerprint: a truncated answer is never certified, so the setting cannot change a certified one.

On the MLX build the limit is a memory budget, not a context length: give the ceiling to cut to with `--over-limit-tokens N`.

## Concurrency

By default (`--concurrency serial`) the server takes one request at a time through the engine: concurrent requests queue behind the engine's lock and are answered in turn.
So a request's rows are always scheduled alone, and its answer is a function of the request: the same request gives the same bytes whatever else the server is sent at the time.
That is what certificates promise ([tasks.md](tasks.md#certified-answers)) and what makes two runs of a benchmark comparable request by request.

`--concurrency batched` trades that for throughput under concurrent load (measured so far on the FP8 12B with new states of about 1,000 tokens at 8 and 16 clients, below), for questions no certified task answers.
One engine thread takes every request waiting at that moment, up to `--batch-max-requests` (16, counted in requests, not rows; the oldest waiting request is always taken), and scores all their rows in one engine call, so concurrent requests share forward passes.

What it does not guarantee, first:

- **An uncertified answer can depend on the traffic it arrived with.** A question scored beside other requests' rows can differ slightly from the same question scored alone, because the engine's kernels choose their tiling and reduction order by the shape of the batch.
  Measured across clients on one RTX 5090 with vLLM 0.31.0 (`runs/2026-10-21_5090-batched/`): against the same server's answers one request at a time, the FP8 12B under 4 and 16 clients answered 196 and 194 of 200 requests bit for bit alike, the largest difference 0.00038, no choice changed; the bf16 12B 19 and 31 of 200, the largest 0.073, 2 choices changed, where its own passes one at a time agree on 81 of 200 (its prefix-cache noise, [EVAL_CARD.md](../EVAL_CARD.md) section 6.5).
  Within one request, the questions scored together (`--multi-question warm`, the same mechanism within one request, not across clients; vLLM 0.30.0, one RTX PRO 6000, 272 four-question requests; `runs/2026-10-05_engine-process/`): against each question scored alone, the Gemma 4 31B changed no choice, its answers differing on 6 of the 272 select questions by at most 0.0046; the Gemma 4 12B changed 3 and 5 choices of 272 on two of the questions; the Qwen base changed 11 to 25 per question, every request differing, by up to 0.39 ([EVAL_CARD.md](../EVAL_CARD.md) section 4).
  Against the demo's acceptable actions no loss was measurable at 272 steps, except that the Qwen base's click target was -3.5 points with an interval ending at zero, [-7.5, +0.0].
- **Throughput, measured on one RTX 5090 (vLLM 0.31.0).** With every request about a new state of about 1,000 tokens, the FP8 12B batched did 8.9 and 9.2 requests a second at 8 and 16 clients against 7.2 serial, the 95th percentile within 4 percent of the median (`runs/2026-10-21_5090-tokenizer-fix/`). In 0.23.0 batched mode was slower than serial on new states (2.3 against 7.2 at 16 clients): when many requests were answered together their threads tokenized at once, and vLLM's thread-safe tokenizer made a deep copy of the whole tokenizer for each concurrent caller, seconds of Python holding the interpreter lock while the card waited (`runs/2026-10-21_5090-batched/`); the engine's tokenizer is now used one call at a time. Measure on your own traffic: the gain depends on the card, the base and how often states repeat. Serial is not affected.
- **A batched server's certificates are its own.** The mode is part of the serving-stack fingerprint when it is batched (a serial server's fingerprint is what it was before the flag existed): a certificate made on a serial server is `certified: false` with the reason `concurrency_changed` on a batching server, and the other way round, until the task is certified again where it is served.

What it guarantees:

- **Certified answers stay a function of the request.** A certified task's questions, and a task's registration and certification reads, take the serial path on a batched server too: one request at a time, never sharing an engine call with other traffic. So a certificate made on a batching server is made and served on the reads a serial server would make, and the drift above is confined to questions no certified task answers.
  The monitor counts only such reads: an answer of a certified task scored in a shared call (a request of two orders, which is not certified) is counted as `shared_skipped` and not observed against the certificate's baseline.
- **A lone request is scored as on a serial server.** Nothing waits on purpose: a request that arrives at an idle server is scored at once, alone, by the same engine call a serial server makes (the same rows, and under `--multi-question sequential` one call per question), so single-stream latency is unchanged, and its answer is the serial answer bit for bit (200 of 200 requests on the FP8 and the bf16 12B, RTX 5090, vLLM 0.31.0; `runs/2026-10-21_5090-batched/`). On the bf16 12B that is compared phase for phase, a batched server's first and second passes against a serial server's first and second: its consecutive passes alternate between two states whatever the mode, so switching the mode between consecutive passes is not a valid comparison there.
- **Each request gets its own answers and facts.** A request's probabilities, its `cached_tokens` and label forms are its own rows'; when it shared a call, the header `x-decisio-batch` says with how many requests and rows (`timing.batch` on `/v1/answer`, with its place in the call), `wait=` in `x-decisio-stages` is the time it waited for the call, and its `questions=` is the whole call's time. A request that fails for itself (a state over the context, a malformed question) fails alone: the call's other requests are scored again without it.
- **One adapter per call.** A call holds the requests of one adapter (or of the weights as loaded) only: the oldest waiting request's, oldest first.
- **The prefix cache keeps working.** Each request's boundary registration (`--register-boundary`) is deferred to its own response and due registrations still go before the next call's questions; within one batched call, two requests about one new state send one warm-up (a serial call sends each request's own, as before).
- **Thinking, the image route and the hidden-state readout are not starved.** A thinking request (`--think`), an image request and a request a task with an intent head answers take the engine alone; the engine's lock is first come, first served, so they take it after the call in progress, whatever the load.

`/health` reports the mode under `concurrency`, and in batched mode the calls made, the requests in them, the largest and how many requests each call held (`requests_per_call`).
Batched mode needs the vLLM engine or the CPU stand-in, in separate mode; MLX and `--mode packed` refuse it at start.

## Docker

```bash
uv build --wheel                   # the image installs this wheel
docker compose up --build          # needs the NVIDIA Container Toolkit and a GPU host
curl http://127.0.0.1:8000/health  # answers once the first start has fetched the checkpoint
```

- The image is vLLM's 0.31.0 release image, pinned by digest, plus the decisio wheel with its `serve` extra; the build argument `APPLY_PATCHES=1` adds the patch series (off by default).
- The first start downloads the checkpoint into the `decisio-data` volume, mounted at `/data`; it is never part of the image.
- The container runs as a non-root user, and compose publishes the port on 127.0.0.1 only.
- The entrypoint starts the server with `--base "${DECISIO_BASE:-gemma-4-31b}"` and `--model-class hidden-readout`, listening on `DECISIO_HOST` and `DECISIO_PORT`; arguments after the image name are passed to the server.
  `DECISIO_BASE` names another base (`qwen3.6-35b-a3b`, `gemma-4-12b`) or a decisio repository, and `DECISIO_MODEL` names a checkpoint instead: alone it brings the base its `config.json` declares, so a container configured with `DECISIO_MODEL=Qwen/Qwen3.6-35B-A3B-FP8` keeps serving the Qwen base.
- It checks for an NVIDIA GPU first and exits with an explanation when the container cannot see one.
- Release images go to `ghcr.io/aminry/decisio`, with their digest in the release notes; to run one instead of building, replace `build:` in `compose.yaml` with `image: ghcr.io/aminry/decisio@sha256:<digest>`.
- `Dockerfile` and `compose.yaml` explain the rest.

First GPU start (2026-10-01, one RTX PRO 6000 Blackwell, the image built on the machine from the `Dockerfile`): healthy in 651 s including the checkpoint download and in 206 s from the cached volume, and the README's example request and the conformance gates C2 to C4 pass against the container (`runs/2026-10-01_docker-first-gpu-start/`).
The image published with v0.1.0 repeated the start from its digest: healthy in 223 s with the checkpoint cached, example and conformance passing (`pushed_image_0.1.0/` in the same record).
Those starts were with the Qwen base, the container's default then.
The no-flag vLLM start selected Gemma 4 31B on 2026-10-08 on one RTX PRO 6000 Blackwell at 600 W and AMD EPYC 9654, using decisio 0.9.0 at the PR head; `/health` recorded `base: gemma-4-31b`, and the README request produced one distinct response in 30 repeats (measured by the maintainers on 2026-10-08; the record is not published; the first response is `runs/2026-10-08_readme-worked-decisions/ticket.json`).
The same start through a built 0.10.0 container remains to be recorded.

## Mac with MLX

```bash
uv sync --extra mlx
uv run python -m decisio.serve.vllm_engine --backend mlx --model mlx-community/Qwen3.6-35B-A3B-6bit
uv run python -m decisio.serve.vllm_engine --backend mlx --base gemma-4-12b --model mlx-community/gemma-4-12B-it-6bit
uv run python -m decisio.serve.vllm_engine --backend mlx --base gemma-4-31b --model mlx-community/gemma-4-31b-it-6bit
uv run python -m decisio.serve.vllm_engine --backend mlx --base gemma-4-31b --model tachara-ai/decisio-gemma-4-31b-mlx-4bit --revision v0.15.0   # 48 GB Macs
```

- `--backend mlx` serves the text route on Apple silicon from an MLX conversion of a base's checkpoint, with every feature of that route: the letters readout, the shared state prefix, the temperatures, task registration with calibration and the intent head, abstention, the rendering rules and the tie-break.
- The base comes from `--base`, or from the conversion's model type, and brings its own settings, as on vLLM; `--model` names the conversion.
- It refuses the image route, packed mode, LoRA adapters and the second-engine head.
- It refuses a checkpoint whose `config.json` declares FP8 quantization (the Qwen base's `Qwen/Qwen3.6-35B-A3B-FP8`, the `tachara-ai/decisio-*` repositories of the Qwen base and the 31B), by `--model` or by `--base <repository>`: those weights are vLLM's format, and an MLX conversion is what `--model` takes.

The Qwen base:
- The 6-bit conversion is the Mac default; the 8-bit one is not shipped.
- The 4-bit conversion is for Macs with more than 32 GB of memory, or 32 GB with the GPU memory limit raised (`sysctl iogpu.wired_limit_mb`): its weights take 19.5 GB and it peaks at 21.6, 22.0 and 22.7 GB at about 8k, 16k and 32k tokens of state (GB are 1e9 bytes; `runs/2026-10-06_mlx-qwen-4bit-regate/`). Two thirds of a 32 GB Mac's memory (34.4 GB) is 22.9 GB, taken here as a conservative stand-in for its default GPU limit, which was not measured on a 32 GB Mac: the peaks are under it, by 0.2 GB at 32k tokens, which leaves almost nothing for the system and other apps. A 32 GB Mac runs the Gemma base's 6-bit conversion below (7.8 GB under that line at 32k) or the [Ollama listings](#ollama).
- Peak memory at 32,761 tokens of state at 6 and 8 bits: 30.7 GB and 39.4 GB (`runs/2026-10-02_mlx-backend/`).
- It does not pad a state (`--pad-policy none`, the MLX default; its cache needs no padding), so its prompts are the served ones without the padding, and a question asked alone and inside a request is one prompt.
- A cross-request prefix cache (`--prefix-cache-mb`, in MiB: the base's default, 2,048 for the Qwen base and Gemma 4 31B and 7,400 for Gemma 4 12B; 0 turns it off) lets a request whose state was seen before continue from the kept cache, with the same answers bit for bit.
- The prompts are built with the official tokenizer by default (`--tokenizer`), so they are the vLLM path's byte for byte.
- At 6 bits, under the current served default, it passes the gates against the FP8 records except the pooled-ECE gate, which changes no setting (`runs/2026-10-03_mlx-regate/summary.md` explains why).
- At 4 bits, under the current served default, it passes every gate against the same records: suite accuracy and ECE, the intent heads over six draws and conformance (`runs/2026-10-06_mlx-qwen-4bit-regate/summary.md`).
- Server time on an Apple M5 Pro (64 GB) with no other job running, median: 258 ms for one question on a new state, 108 ms on a state seen before, 123 ms per question when 100 share a state; macOS's indexing services ran during the passes (`runs/2026-10-07_mlx-quiet-latency/`).

The Gemma base (`--base gemma-4-12b`):
- The 6-bit conversion is the default; there is no 4-bit option, since the 4-bit conversion fails the suite-accuracy gate (`runs/2026-10-05_mlx-gemma/`).
- The 6-bit conversion holds 9.7 GB of weights and peaks at 15.1 GB at 32,687 tokens of state, so it runs on a 32 GB Mac (`runs/2026-10-05_mlx-gemma/6bit/memory.json`).
- The prompts are built with the base's own tokenizer and chat template, `google/gemma-4-12B-it` at its pinned revision, not the conversion's, whose chat template is older.
- At 6 bits it passes every gate against the base's vLLM bf16 record: suite accuracy and ECE, the intent heads over six draws, conformance, and answers bit for bit on a fresh server (`runs/2026-10-05_mlx-gemma/summary.md`).
- It reads a new state more slowly than the Qwen base on the same Mac: one question on a new 1,000-token state takes 1,312 ms, against 515 ms for the Qwen base (server medians on an Apple M5 Pro with no other job running; `runs/2026-10-07_mlx-quiet-latency/`).

The Gemma 31B base (`--base gemma-4-31b`):
- The 6-bit conversion `mlx-community/gemma-4-31b-it-6bit` (26.1 GB to download, 24.9 GB of weights in memory) is the default, on a 64 GB Mac. The 4-bit conversion `tachara-ai/decisio-gemma-4-31b-mlx-4bit` (group size 32, 19.2 GB of weights) is for Macs with 48 GB of memory; its figures are below. `--revision v0.15.0` names the repository's tag of the build that was measured, so a later change to its main does not change what you serve (decisio 0.15.1 or later; before it, `--revision` was ignored for the download). The 8-bit one is not measured.
- Measured on one Apple M5 Pro with 64 GB; no smaller Mac was measured. GB are 1e9 bytes.
- It peaks at 29.6, 31.1 and 33.7 GB at about 8k, 16k and 32k tokens of state, against the Mac's GPU memory limit of 55.7 GB.
- **It is documented for states up to 16,383 tokens.** In the quiet reading (nothing else large running), macOS memory pressure stayed normal from the model's load through the 16k cells.
- **At 32,687 tokens it runs, with memory pressure:** in the quiet reading one of 54 samples in its cells was at the warning level, swap stayed flat (4.9 to 5.0 GB), and 39% of the GPU memory limit was still free. A second reading, taken after the head draws with the Mac's usual desktop applications open (the largest resident process was 0.5 GB), had 9 of 61 samples at the warning level, none critical, and swap up 3.7 and 1.9 GB; 8k and 16k were clean in both (`runs/2026-10-07_mlx-gemma-31b/`).
- It scores each question as its own continuation of the shared prefix, so `--multi-question` is `sequential` on MLX; the base's vLLM default, `warm`, is refused there.
- The prompts are built with the base's own tokenizer and chat template, `google/gemma-4-31B-it` at its pinned revision, not the conversion's.
- Against the base's vLLM FP8 record it passes the suite accuracy and ECE gates, conformance with answers equal to the served path's, and the intent heads over three draws per set (BANKING77 0.849 against 0.844, CLINC150 0.970 against 0.970; `runs/2026-10-07_mlx-gemma-31b/`). Server times are not measured.
- The 4-bit conversion peaks at 23.9, 25.3 and 28.0 GB at about 8k, 16k and 32k tokens of state, under a 27 GiB MLX memory limit on a 64 GB Mac, with memory pressure normal and swap flat (`runs/2026-10-16_mlx-gemma-31b-4bit/`).
  It is documented for states up to 16,383 tokens, as the 6-bit one is; at 32k it completes at 96.5% of that limit.
  A 48 GB Mac is the smallest it starts on by decisio's memory planner, which refuses 32 and 36 GB; no 48 GB Mac was measured.
- Against the base's vLLM FP8 record, the 4-bit conversion passes the suite accuracy and ECE gates, JevBench calibration, conformance and the intent heads over three draws per set (BANKING77 0.833 against 0.844, CLINC150 0.967 against 0.970).
  It answers a second fresh start bit for bit.
  It scores 1.5 points lower on MMLU-Pro (0.679 against 0.694; -2.0 to -1.1), mostly in engineering, math, chemistry and physics (-2.9 to -3.6 points), and 2.6 lower in history.
  The 6-bit conversion's MMLU-Pro was not measured.
- Server time for one question on a new state with the 4-bit conversion, on a busy Mac under the 27 GiB limit: 9.4 s at 3,000 tokens, 66 s at 16,000 and 152 s at 32,000; the prefix cache was off under that limit, so a later question took as long.
  With a 1,789 or 2,048 MiB cache, as the planner sets on a 48 GB or 64 GB Mac, a 3,000-token state would fit; that was not measured.
- A registration is slow: about an hour for a BANKING77 draw and about 2.6 hours for a CLINC150 draw.
- Its prefix-cache entries are large: 935 MiB at 1,000 tokens, 1,869 MiB at 3,000, 2,954 MiB at 8,000 and 4,386 MiB at 32,000. At the default budget (2,048 MiB, as for the Qwen base) the cache keeps two 1,000-token states or one 3,000-token state and never stores a state of 8,000 tokens or more, so a repeat question on a long state reads it again (27 s at 8,000 tokens, 126 s at 32,000). `--prefix-cache-mb` raises the budget; each stored state takes memory beside the peaks above, and the peaks were measured with the cache empty.

`docs/design/mlx-backend.md` has the design, the gates of each conversion and the padding decision.

### Memory on a Mac

MLX keeps the buffers a computation has freed, up to a cache limit, and allows allocation up to a memory limit.
Both default to about 80 percent of the Mac's memory (51.8 GiB on a 64 GB Mac).
A server that answers many different states therefore grows toward that size, not toward the size of its weights: one Gemma 4 12B server reached about 50 GB on a 64 GB Mac, and with a second server froze the machine twice (measured by the maintainers on 2026-10-14; the record is not published).

So the server sets both limits itself, before it loads the model, and prints the line `MLX MEMORY ...` at start-up with the figures.

| Setting | Default | Flag |
| --- | --- | --- |
| MLX cache limit (freed buffers kept for reuse) | 2 GiB | `--mlx-cache-limit-gb` |
| MLX memory limit | what the base needs, rounded up to whole GiB, never above two thirds of the Mac's memory; `0` leaves MLX's own | `--mlx-memory-limit-gb` |

What the base needs is `weights + working set + prefix cache + cache limit + 2 GiB of headroom`.
The weights are the size of the conversion's safetensors files.
The working set is the base's measured peak above its weights at a 32k-token state (`mlx_working_gb` in `decisio/families.py`: 4 GB for the Qwen base, 6 GB for Gemma 4 12B, 9 GB for Gemma 4 31B, from the peaks listed above).
Two thirds of the machine is the figure the 12B's prefix-cache budget was already sized against.
The defaults on a 64 GB Mac: 26 GiB for the 12B (with its 7,400 MiB prefix cache) and 38 GiB for the 31B (with 2,048 MiB).
On a 32 GB Mac the 12B gets 21.3 GiB and a prefix cache cut from 7,400 to 2,776 MiB, and says so in the start-up line; a base that does not fit is refused.

**The server refuses to start, with the numbers, when the base cannot fit:**

```text
decisio: the gemma-4-12b base needs about 16.6 GiB under MLX (9.1 GiB of weights, a working set of 5.6 GiB at a 32k-token state, and the MLX cache limit of 2.0 GiB), which is more than the memory limit of 14.0 GiB (--mlx-memory-limit-gb 14). Raise the limit, lower --mlx-cache-limit-gb, or start a smaller base
```

- `weights + working set + cache limit` over the limit (the default's ceiling or the flag's) is refused, as above.
- An explicit `--prefix-cache-mb` that does not fit under the limit is refused; a prefix cache left at the base's default is made smaller to fit.
- After the model is loaded, a process whose physical footprint is already over the limit is refused.

**`/health` reports the memory** under `memory` (bytes): the limits as set and where they came from, MLX's active, cache and peak memory, and the process's `physical_footprint_bytes` and `resident_bytes`.
The footprint is macOS's `phys_footprint`, what Activity Monitor calls memory and what the kernel's memory-pressure decisions follow; the check after loading uses it too.
Resident size leaves out part of MLX's Metal buffers: 10.7 GB resident against 14 GB of footprint for a 12B server.
Watch the footprint, not the resident size.

Run one model server on a Mac at a time.
Two servers each stay under their own limit and can still add up to more than the machine has.

## The context length

Every record in this repository was measured at **32,768 tokens**.
Without `--max-model-len`, the vLLM server on a Gemma base serves **the largest length that fits the card, up to the base's ceiling and not below 32,768**; the Qwen base stays at 32,768.

| Base | Without `--max-model-len` | Ceiling |
| --- | --- | ---: |
| `qwen3.6-35b-a3b` | 32,768 | 32,768 |
| `gemma-4-12b` | the largest that fits the card, 32,768 to 65,536 | 65,536 |
| `gemma-4-12b-fp8` | the largest that fits the card, 32,768 to 65,536 (65,536 on a 32 GB card: below) | 65,536 |
| `gemma-4-31b` | the largest that fits the card, 32,768 to 98,304 | 98,304 |

**How the length is chosen.** At start-up the server asks vLLM for the longest single request that fits the card (`max_model_len=-1`: vLLM profiles the card, takes the KV cache memory it reports and searches the longest length that fits), then serves the largest multiple of 8,192 not above that and not above the ceiling. It is one start of the engine, not a trial. If even 32,768 does not fit, the start fails with a message that names the length that fits, as a fixed 32,768 failed before.
`/health` reports the length in force as `max_model_len`, why as `max_model_len_source` (for example "auto: fits 81,920 of 98,304 on this card"), and the ceiling as `max_model_len_ceiling`; a prompt over it is answered 422 as before.

**What a card gives depends heavily on `--gpu-memory-utilization`.** The 31B (stored FP8) on one H100 PCIe 80 GB, vLLM 0.31.0, the engine's share of the card varied and no length flag (`runs/2026-10-15_s14-session/`):

| Engine share | KV cache vLLM reported | vLLM's fit (the longest single request) | Served |
| ---: | ---: | ---: | --- |
| 0.90 (the default) | 33.85 GiB | 105,600 | 98,304 (the ceiling) |
| 0.85 | 29.9 GiB | 53,696 | 49,152 |
| 0.70 | 18.54 GiB | 22,064 | does not start |
| 0.55 | 6.71 GiB | 7,968 | does not start |

The fit is not linear in the KV memory: 12% less KV cache (0.90 to 0.85) halved it, and the 31B's cache has several groups (vLLM's hybrid KV cache manager) that all have to fit, so the length cannot be estimated from one measurement or from a card's size alone. On the same 80 GB card the 12B served its ceiling of 65,536 at the default share (43.28 GiB of KV cache). Read `/health` for what a server chose, and give `--max-model-len` to fix a length. Where 32,768 does not fit the server does not start (as a fixed 32,768 did not): raise the share, start a smaller base, or give a `--max-model-len` below 32,768.

**The 12B on a 32 GB card, measured** on an RTX 5090 (32,607 MiB), vLLM 0.31.0, 2026-10-21 (`runs/2026-10-21_5090-fp8-12b/`):

| | bf16 12B | FP8 12B (the stored conversion) |
| --- | --- | --- |
| Weights in memory (vLLM's "model loading took") | 22.83 GiB | 12.74 GiB |
| KV cache at the default share (0.90) | 0.48 GiB: does not start at 32,768 (5.82 GiB needed) | 11.66 GiB: serves 65,536, the ceiling |
| At 0.95 of the card | 2.05 to 3.05 GiB: refused at 32,768, 16,384 and 8,192; starts at 4,096 | |

The bf16 12B's KV cache is dominated by its sliding-window layers, so a shorter context saves little: 16,384 tokens still needed 5.25 GiB against 5.82 at 32,768.
For larger cards the arithmetic (0.90 of the card, less the weights, less about 2.5 GiB for activations and graphs) runs high: it gave 13.8 GiB for the FP8 12B on 32 GB, where vLLM reported 11.66, and 46.9 GiB for the bf16 12B on an 80 GB H100, where vLLM reported 43.28, so read these as upper bounds:

| Card | bf16 12B, by the arithmetic | FP8 12B, by the arithmetic |
| --- | --- | --- |
| 48 GiB | 18.4 GiB | 28.2 GiB |
| 96 GiB | 61.6 GiB | 71.4 GiB |

The 12B's KV cache is small per token (about 16 KB a token for its 8 full-attention layers, beside about 0.34 GiB a sequence for its sliding-window layers), so beyond the ceiling the extra room buys concurrent sequences and prefix cache, not a longer request: both precisions reach the 12B's ceiling on a 96 GB card.

**Where it does not apply.** With `--max-model-len` the server serves exactly that length (above 32,768 only up to the ceiling; beyond it the server refuses to start, as before), and so with a `max_model_len` in `--engine`. The default stays 32,768 with `--image-model` (the image engine serves 32,768), a second engine (`--head-engine`), `--engine-process separate` (the fit is read from the engine's configuration in the server's own process), `--backend mlx` and `--backend hf`.
To keep one length on every card, give `--max-model-len`.

**What it does to certificates, and to short answers.** The context length served is part of the serving-stack fingerprint (the card, the engine, the KV cache dtype and the rest of [Where it stops](tasks.md#where-it-stops)), so a certificate made at one length is suspended at another, whether the length was given or followed the card; a card's fit is the same for the same flags, and the 8,192 step absorbs small differences in the memory vLLM reports. It is in the fingerprint because the length is not inert on every card. What was measured, on the stored FP8 31B with 200 single-question requests answered one at a time:
- On an H100 PCIe 80 GB, the default (98,304) was **exactly equal** to a fixed 32,768 start on all 200, with every tensor's bytes identical (`runs/2026-10-15_s14-session/C/`).
- On an RTX PRO 6000, 98,304 against two fresh starts at 32,768 that were exactly equal to each other: **0 of 200 equal**, median difference 4.3e-3, largest 0.389, **6 of 200 top answers changed**. A start at 32,768 on the same code and `--engine` route was exactly equal to those two, so the difference is the length (`runs/2026-10-15_identity-at-98304/`). That is the size of the start-to-start differences seen on that card class in other sessions, not a new kind of error; the likely cause is the compiled graph, which depends on the length, and that is a reading, not a test.
**The published figures hold at the auto length.** JevBench's 231 public items on the stored FP8 31B started at 98,304 tokens on that RTX PRO 6000: 213 correct (1.000, 1.000 and 0.838 on the easy, original and hard tiers, the published figures), ECE 0.035 on the standard tier and 0.092 on the hard tier, against 210 correct and ECE 0.039 and 0.092 at 32,768 on the same box; 3 of 231 top answers changed, all three toward the published figures, and the calibration is of the same order (`runs/2026-10-15_s13-session/jevbench_98304/`). Three items between two runs of one length is what restarts and boxes already produce, so this says the figures hold at the auto length, not that the longer length is better.

To get the same answers on every card, give `--max-model-len` the same value on all of them.

**What the records say about memory.** The measurement of 2026-10-15 on one 96 GB card started the 31B (FP8 weights) at a context of 98,304 tokens on one 96 GB card at the default 0.90 share with the model's own cache dtype: vLLM 0.31.0 reported 48.05 GiB of KV cache and room for 141,865 tokens, and the 12B at 131,072 reported 57.53 GiB and 612,140 tokens (`runs/2026-10-15_s12-session/X/` and `X5/`). vLLM's start-up check counts the cache differently across versions: on 0.30.0 the same share of the 31B was called 48.6 GiB and 57,882 tokens (EVAL_CARD section 7; the bf16 start needed 27.51 GiB for one 32,768-token request, `runs/2026-10-04_gemma-4-31b/bf16_start.txt`), so a 0.30.0 server may need `--kv-cache-dtype fp8`, a larger `--gpu-memory-utilization`, or both beyond about 57,000 tokens, and a 0.31.0 server on a smaller card is not measured. `--kv-cache-dtype fp8` changes the numerics: a certificate made under the model's own cache dtype is suspended under fp8 (the serving-stack fingerprint includes the KV cache dtype), and the answers of a registered task can move; register and certify again under the dtype you serve.

**What the records say about time.** The first question about a state the server has not seen grows faster than the state does, because attention does (the table of first-question latency above stops at 32,000 tokens: 3,370 ms at 16,000 and 8,880 ms at 32,000 on the 31B; 1,779 ms and 4,383 ms on the 12B). The faster attention backends are refused for Gemma 4 on vLLM 0.30.0 and 0.31.0, so a longer context is slower than a proportional guess.

**Where the ceilings come from.** The card session of 2026-10-15 (the long-context checks, pass rules written before the run; `runs/2026-10-15_s12-session/`): the 31B started at a context of 98,304 tokens and answered a check at a state of 85,000 tokens (23 of 30 items right against 25 of 30 for the same items short; first question 40.8 s, second 0.40 s), which is its ceiling. The 12B started at 131,072 and was right on 23 of 30 items at 60,000 tokens (26 short; first question 11.0 s) but on 19 of 30 at 120,000 (26 short; first question 23.2 s), which failed the pass rule (a drop of at most 3), so its ceiling is the lower 65,536 (started and not itself checked; the check was at 60,000). Each check is 30 items, a smoke test and not an accuracy measurement.

**What is not measured.** Anything between the checked states and the ceilings other than what the session ran, the first and later question at those lengths, the prefix cache and the state boundary at that size, the temperatures (fitted on states of a few thousand tokens), and the accuracy of answers at long context on a suite (no suite here has a state over 32,768 tokens). A length that the card allows is not a length that was checked: to serve only what was measured, give `--max-model-len 32768`.

## LoRA adapters

`--adapter name=path` loads a vLLM-format LoRA adapter, and a request that names it (`"adapter": "<name>"`) is answered by the base with that adapter.
The adapter is served unmerged, on the text route, in separate mode; the MLX backend refuses adapters.
The CPU stand-in (`--backend hf`) serves no adapters either: it loads none, so a request that names one is refused as an unknown adapter (`loaded: []`). Adapters need the vLLM backend on a GPU.
The request fields are in [api.md](api.md), the flags in [cli.md](cli.md).

### An adapter's own temperatures

The base's temperatures were fitted on the base's plain readouts, and an adapter changes the scale of the readout.
Served at the base's, a tuned 31B hedged on 70.3% of a benchmark's yes/no items, which scored it -45 (its top answer was right on 70 of 74 of them, as the stored base's was).
So an adapter carries its own temperatures, one per question type (`choice`, `noul` for yes/no, `score`), in a file `decisio_adapter.json` beside its weights, or in a file given with `--adapter-temperatures name=file` for an adapter whose folder is not yours to change:

```json
{"schema": "decisio.adapter/1",
 "temperatures": {"choice": 2.141, "noul": 3.075, "score": 4.185},
 "fit": {"objective": {"choice": "log_loss", "noul": "log_loss", "score": "log_loss"}, "n": {"choice": 173, "noul": 155, "score": 172}}}
```

- A request that names the adapter is served at them. A request that does not is served at the base's, as before.
- A type the file does not list keeps the base's temperature. An adapter with no file is served at the base's, and the server says so when it starts.
- The file is checked before the engine is built: the schema, the types, the bands, positive finite numbers, and that `--adapter-temperatures` names a loaded adapter.
- `/health` shows them in `adapter_temperatures`, and the debug readout of a request names the adapter and the temperature it was served at.

#### A choice temperature by the number of options

An adapter's over-confidence on choice questions can grow with the number of options, and the one `choice` temperature cannot follow it: an adapter fitted on questions with a few options is calibrated there and over-confident on questions with many.
The file can therefore carry a choice temperature per band of option count, beside the flat one:

```json
{"schema": "decisio.adapter/1",
 "temperatures": {"choice": 2.141, "noul": 3.075, "score": 4.185},
 "choice_bands": {"2-10": 1.9, "11-26": 2.6, "27-255": 3.1}}
```

- A choice question is served at the temperature of the band its number of options falls in. A count in no band is served at `temperatures.choice`, then at the base's. Yes/no and score questions are not banded (a score question with 12 levels is not a choice with 12 options).
- The keys are `lo-hi`, within 2 to 255 (the most options a choice can have) and not overlapping. A file may have bands and no flat `choice`. A file without `choice_bands` is served as before.
- The bands are in `/health`'s `adapter_temperatures` and in the serving-stack fingerprint, so changing one is a different stack.

`python -m decisio.serve.adapter_temperatures fit` writes the file from the adapter's readouts on labelled items:

```bash
python -m decisio.serve.adapter_temperatures fit --reads reads.jsonl --out decisio_adapter.json --data "what the items are"
```

One row per item, `{"type": "choice" | "noul" | "score", "probs": [...], "gold": <index>}`, with the raw readout (`p_plain` of a `--debug-readout` response) or probabilities and the temperature they were served at (`"read_at"`).
Every type is fitted by minimum log loss: the temperature at which the probabilities are calibrated on those items.
Choice is also fitted per band of option count (`--choice-bands`, default `2-10,11-26,27-255`; `none` writes the one temperature only), each on the choice rows with that many options.
A band with fewer than `--min-band-items` rows (default 20) is not fitted and is served at the temperature fitted on all the choice rows, which is always written beside the bands; when the rows fall in one band only, the file has no bands, since they would repeat that temperature.
Use items that are not a benchmark's; a temperature is a statement about the items it was fitted on, and an adapter fitted on hard public-source questions was underconfident on an easier set of synthetic ones (`runs/2026-10-17_adapter-temperatures/`).
A score item sent as a choice question is served at the choice temperature, so name the temperature a row was served at.

### Fitting them from a model's stored reads

`python -m decisio.serve.adapter_temperatures fit-split` is Lab 1's split fit as a tool: from the model's stored reads of two pools (an H pool of choice items and a dev pool with choice, yes/no and true score items, both with their gold answers in a keys file) it fits one kappa per type by minimum log loss and writes the temperatures (`T = served temperature / kappa`) as the file above.

```bash
python -m decisio.serve.adapter_temperatures fit-split --h H_reads.jsonl.gz --h-keys h_keys.json \
    --dev dev_reads.jsonl.gz --dev-keys dev_keys.json --variant half0 --out decisio_adapter.json
```

- Choice is also fitted per band of option count on the choice pool's items (`--choice-bands`, `--min-band-items`, as for `fit`), with each band's held-out figures in `fit.checks.choice.bands` and its swap-validated ECE in `fit.swap_validated_dev.choice.bands`. The H pool has 3 to 6 options only, so its file has no bands: to fit the 11-26 and 27-255 bands, use a `dev` variant on a natural pool with those option counts, one that is not the gate's set.
- The choice kappa is fitted on the H pool, the yes/no and score kappas on the dev pool. `--variant half0` fits on the items whose id has an even sha256 prefix and reports the other half as held-out (log loss and ECE before and after), `half1` is the swap, `full` uses all of both pools, `dev` fits the choice kappa on the dev pool too, and `devhalf0` and `devhalf1` do that on halves of the dev pool with the other half held out (a model whose H-fitted choice temperature does not transfer to public-source questions).
- The reads are rows of `id`, `probs` (as the server returned them, at the base's temperatures; `--served` and `--dev-served` say which if they differ from the defaults) and `p_plain` (the plain readout of a `--debug-readout` server; `--use p_plain` fits on it, with temperature 1 as the reference).
- **Two ECE definitions, both reported, named.** The held-out figures of a half variant (`fit.checks.<type>.heldout`) are per half, with ten equal-count bins and ties shuffled (the definition of the first evaluation's fits). The figure a ship gate uses is `fit.swap_validated_dev`: the kappa of each half applied to the other half's items, pooled over all of the dev pool's items of the type, with ten equal-width bins, next to the unscaled figure and the two halves' kappas with their ratio (over 1.25 the gates call the fit unstable). They differ on the same fit: BH0's choice figure is 0.061 pooled with equal-width bins and 0.095 and 0.127 on its two halves with equal-count bins.
- It never reads a benchmark. On the stored reads of the first MeTask evaluation it reproduces every kappa and held-out figure of Lab 1's fits (`runs/2026-10-18_adapter-fit/`).

### Yes/no stays a probability

An adapter's yes/no answer is the calibrated probability, and the server does not commit it.
`--noul-commit` (the answer as 0.9 or 0.1) remains the operator's choice and applies to an adapter's answers as to the base's.
On JevBench's 74 public yes/no items an honest adapter hedges (35.1% of its answers fall between 0.2 and 0.8, where the reading counts no answer), while a service that returns two constants never does; the top answers are the same.

### Score questions

A score answer is the expected position of the returned distribution, and JevBench scores that number.
A calibrated distribution over levels is wider than the position error asks for, so a score temperature fitted by log loss pulls the expected position to the middle level.
The fit tool's `--score-objective expected_position_error` fits the score temperature for the position error instead: the flattest temperature whose score CC is within one point of the best, which is sharper than the calibrated one, and the file says so in `fit.objective`.
On the 18 public score items, with MeTask's adapter: 63.12 CC at the log-loss temperature, 70.09 at the expected-position one, 75.60 at MeTask's own temperature, which they fitted on those 18 items, against 73.21 for their service and 70.91 for the stored base (`runs/2026-10-17_adapter-temperatures/`).
The gap to their service is that temperature, fitted on the items it is scored on, and not the readout.

### Identity and the fingerprint

The server hashes each adapter's weights file and configuration when it starts, and `/health` reports them in `adapter_identity`.
The serving-stack fingerprint of [a certificate](tasks.md#where-it-stops) holds each loaded adapter's name, weights hash and temperatures, so a different adapter, a retrained one or a changed temperature is a different stack.
A server with no adapter has none of these keys, and its certificates are unchanged.
A request that names an adapter is never certified, and a question that a registered task matches is refused with an adapter, as [api.md](api.md) says.

## Ollama

```bash
ollama pull aminroudaki/decisio-gemma    # the Gemma 4 12B base, for 16 GB laptops
ollama pull aminroudaki/decisio          # the Qwen base
```

Requires Ollama 0.35.1 or later.
The model pages, [decisio-gemma](https://ollama.com/aminroudaki/decisio-gemma) and [decisio](https://ollama.com/aminroudaki/decisio), package laptop versions of the two bases for Ollama's decision route and list their tags and their sizes.
- The Gemma listing is the path for 16 GB laptops: its `q4_k_m` tag (also `latest`) is a 7.7 GB download and takes 8.6 GB in Ollama at the listing's 8,192-token context; its `q8_0` tag takes 13.6 GB (Ollama 0.35.1 on an Apple M5 Pro, `runs/2026-10-07_ollama-gemma-memory/`).
- The Qwen listing's smallest tag is a 23 GB download, so it needs a machine with more memory.

Ollama builds its own prompt and applies no calibration, so its numbers are the ones on those pages, not the ones in this repository.

## CPU stand-in for development

```bash
uv sync --extra dev --frozen
uv run pytest
uv run python -m decisio.serve.vllm_engine --backend hf --model Qwen/Qwen3-0.6B-Base
```

- `uv run pytest` runs the unit tests on any machine; the GPU tests (`pytest -m gpu`) need a card, and run by hand until a nightly GPU runner is registered.
- Time to a first answer: measured on Apple silicon on 2026-10-14, the stand-in's process takes 4 to 8 seconds from start to `/health` answering with `Qwen/Qwen3-0.6B-Base` already in the Hugging Face cache, and a first question about 0.3 seconds on top. The download of that model (about 1.2 GB, once) and `uv sync` were not timed, and no time to a first answer is recorded for the Ollama path.
- `--backend hf` serves every route from a small Hugging Face model on the CPU, for developing and testing the routes; it is not the measured system and its answers are not for measurement.
- CI runs the stand-in on `Qwen/Qwen3-0.6B-Base` at a pinned revision (`STAND_IN_REVISION` in `.github/workflows/ci.yml`).
- `examples/tasks/` walks through task registration against it.

## Checking a server

`curl http://127.0.0.1:8000/health` returns what the server is serving: the engine, the base, the checkpoint and its revision, the temperature for each question type and the prompt.
`docs/api.md` describes every field.

## When the engine dies

An engine whose forward pass fails (a CUDA error, an out-of-memory error) or whose engine-core process exits does not serve again, so the server ends itself and leaves the restart to whatever runs it.

A death is confirmed before it is declared, so that a request which trips a bug cannot take a healthy server down:
- vLLM's own word is enough: its `EngineDeadError`, or its flag for an engine-core process that is gone.
- Any other unexpected error fails that request with 500, as it always has, and the server then sends the engine one probe: a forward pass over a one-token prompt, given 5 seconds, or three times the engine's slowest recent call when that is longer, so that a slow engine is not taken for a dead one (the server times one probe per engine at start-up, so its first request already allows for a slow engine).
- If the probe answers, the engine is alive and the server stays up; the server log says so (`the engine answered a probe in ... ms`).
- If the probe fails or does not answer in that time, the engine is dead. An in-process engine that has failed hangs rather than answering, so this is how its death shows.
- A probe still waiting when the engine is declared dead another way (vLLM's flag, another request) stops at once, and its request gets its 503.
- An error caused by the request (a malformed question, a state longer than the context: 422, `docs/api.md`) is answered with a 4xx and the engine is not probed.

From the moment the engine is dead:
- every request, including those already waiting for the engine, is answered at once with 503 and the reason, and the engine is not called again;
- `/health` answers 503 with `{"ok": false, "engine": "dead", "reason": "...", "exit_code": 70}`;
- after 2 seconds, once every request in flight has its answer (waiting at most 8 seconds more), the server stops its engine processes and exits with code 70; a request the dead engine never returns has its connection closed by the exit.

The in-process arrangement therefore exits about 7 seconds after the failing request (the probe's 5 seconds, then the grace period's 2).
In the separate arrangement (`--engine-process separate`, the default with a second engine) vLLM reports the death itself, so the server exits about 2 seconds after it.
The server also watches vLLM's flag between requests, so a death between requests turns `/health` to 503 within a second, before any request arrives.

Run the server under a restart policy, so that the exit brings up a new one:
- compose: `restart: unless-stopped`, as `compose.yaml` has it;
- systemd: `Restart=on-failure`;
- Kubernetes: the default `restartPolicy: Always`, with a liveness probe on `/health`.

[`operating.md`](operating.md) has a complete file for each of the three.

Docker restarts a container when its process exits, not when its `HEALTHCHECK` reports it unhealthy, so it is the exit that brings the server back; the image's check reports the dying container unhealthy in the meantime.
The new server loads the engine again (the start times above), and requests sent until it is healthy are refused.

In decisio 0.1.0 to 0.7.1 none of this happens: after the engine dies `/health` keeps answering 200 and the process keeps running, so no restart policy acts.
In the separate arrangement every request then fails with 500; with `--engine-process in` (0.7.0 and 0.7.1) every request after the failing one hangs.
