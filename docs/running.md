<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Running decisio

Every way to run the server, in the order of the README's chooser ("Install and run").
The README has the short version of each; this page has what it leaves out.
Every number names the record it comes from.

## GPU server

If a first start fails (a card too small, a driver, a busy port, a download), [`troubleshooting.md`](troubleshooting.md) has the messages and the fixes.

Requirements: Linux, one NVIDIA card (measured on an RTX PRO 6000 Blackwell with 96 GB), a driver that supports CUDA 13.0 (the runtime `uv.lock` pins), Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/aminry/decisio
cd decisio
uv sync --extra serve --frozen
uv run python -m decisio.serve.vllm_engine
```

- With no flag the server serves Gemma 4 31B (`--base gemma-4-31b`), the default base since 0.10.0, from its FP8 repository `tachara-ai/decisio-gemma-4-31b`; it needs the 96 GB card.
  `--model google/gemma-4-31B-it` serves Google's weights at the pinned revision instead, quantized to FP8 each time they load (`--revision` does the same, naming a revision of Google's checkpoint).
- A third party served the 31B on a smaller card than ours: JevBench's maintainer ran decisio 0.8.0 with `--base gemma-4-31b` (Google's weights, quantized to FP8 on load) on one H100 80 GB (Lium) and "no flag of theirs was changed to make it fit".
  That is the board's own account of its run of 2026-10-06 on the v1.6.0/v1.6.1 pool, in the row `decisio-gemma-4-31b-v080` (JevBench v1.6.1, `https://benchmarkheaven.com/api/jevbench/v1.6.1`); we did not run it, and nothing here changes what we state: the 31B is measured on, and needs, a 96 GB card.
  The row also notes that 23 of the pool's items, of about 80,000 tokens, are over the 32,768-token context.
- `--base qwen3.6-35b-a3b` serves `Qwen/Qwen3.6-35B-A3B-FP8` and `--base gemma-4-12b` serves `google/gemma-4-12B-it`, each at its pinned revision and with its own settings (the README's "Choosing a base"; `docs/cli.md` lists every setting per base).
- With a `--model` and no `--base`, the base is detected from the checkpoint's `config.json`, so a local copy of any base's checkpoint brings its own settings; a base's pinned revision applies whenever `--model` names the base's own Hugging Face repository and `--revision` is not given.
- The first start downloads the checkpoint into the Hugging Face cache and warms the engine: about 31 GB for the 31B's FP8 repository (62 GB for Google's bf16 weights) and about 36 GB for the Qwen base (`runs/2026-10-01_docker-first-gpu-start/`); `/health` answers once it is ready.
- Upgrading from a release before 0.10.0, where the container served the Qwen base: `docs/upgrading.md`.
- The decisio plugin registers its model classes with vLLM through an entry point, so vLLM 0.31.0 loads them without any patch; the package must be installed (as `uv sync` does) for vLLM's engine processes to find it (`docs/design/vllm-plugin.md`).
- `patches/` holds an optional latency patch series, off by default and not needed for correct answers (`patches/README.md`).
- The server refuses to start when `VLLM_USE_DEEP_GEMM` is set to anything other than `0`, unless `--allow-deep-gemm` is given.
- It listens on `127.0.0.1:8000` (`--host`, `--port`); put a reverse proxy in front of it to expose it.
- `--image-model` adds a second engine on the same card for requests that carry images; `EVAL_CARD.md` section 1 has the memory shares it was measured with.
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
Without it, a later, different question about the state read the whole state again, which costs about a first read of it: on the 31B at 3,000 tokens, 439.0 ms for the first read against 43.3 ms for a different question read from the cache (one card at 585 W, an AMD Ryzen Threadripper 9960X; `runs/2026-10-06_latency-585w/`).
The one measurement of the old behaviour at 31,000 tokens is from before 0.8.1, on another host: 9.2 s instead of 0.17 (531 ms instead of 47 at 3,000 tokens there).
The server remembers the last 4,096 states it registered and sends the warm-up again only when a request finds the boundary gone, so a repeated question costs no extra engine call.
Several questions in one request are not affected, and the Qwen base needs no registration (its padded state already ends on a single question's latest checkpoint).

`--register-boundary` sets when a single question on a new state registers it:
- `after`: the question is answered first, reading the state fresh, and the warm-up is sent once the response has been handed to the server (in library use, once `answer()` returns). The caller does not wait for it.
- `before` (the default of both Gemma bases since 0.11.1; the 12B's was `after` from 0.9.0 to 0.11.0): the warm-up goes ahead of the question, in the caller's request, as in 0.8.1. On one card at 585 W with an AMD Ryzen Threadripper 9960X that added +16, +25 and +28 ms to a first read at 300, 1,000 and 3,000 tokens on Gemma 4 12B, and +21, +34 and +34 ms on Gemma 4 31B (`runs/2026-10-06_latency-585w/`, and `register_boundary.md` there for every figure below).
- `off`: never; a later, different question about a state reads it again, as before 0.8.1. For traffic that asks one question per state, where a registration would only cost throughput.

Why the 12B's default moved from `after` to `before` in 0.11.1: our grid of 2026-10-09 (12B, `after`, AMD EPYC 9534, RTX PRO 6000 at 600 W, decisio 0.10.0; in our private records) found that one question on each new state, sent back to back, took 774 ms at 3,000 tokens instead of about 390 and 1.6 s instead of 0.8 at 6,000, because each request waited behind the previous state's registration (385 and 796 ms outside the engine's stages, one read of the state); `batch` mode, three-question requests and the 31B's `before` showed no gap. `before` reads the state once plus one token inside the request, so nothing is pending behind a response. A direct check then confirmed it (2026-10-09, one server on decisio 0.10.0, in our private records): one question per new state, ten states, server time of the first request and the median of the nine later ones. Back to back, `after` took 378 and 771 ms at 3,000 tokens and 812 and 1,613 ms at 6,000; `before` 413 and 417 ms and 834 and 834 ms; `off` 383 and 395 ms and 816 and 818 ms. With the client waiting for the registrar between states, `after` took 396 and 396 ms and 818 and 827 ms. So the doubling is only under `after` and only back to back; `before` costs 22 ms (3,000 tokens) and 17 ms (6,000) over `off`. `off` shows the cost of no registration and nothing else: it is the baseline the other two are read against, and it does not show how a later, different question fares, since without a registration that question reads the state again. The response's `timing.state_boundary.ran_before` was 0 on every request of every order: it counts only a registration the request sends itself, and no field records the wait for the registrar's background warm-up.

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
The no-flag vLLM start selected Gemma 4 31B on 2026-10-08 on one RTX PRO 6000 Blackwell at 600 W and AMD EPYC 9654, using decisio 0.9.0 at the PR head; `/health` recorded `base: gemma-4-31b`, and the README request produced one distinct response in 30 repeats (our private records of 2026-10-08).
The same start through a built 0.10.0 container remains to be recorded.

## Mac with MLX

```bash
uv sync --extra mlx
uv run python -m decisio.serve.vllm_engine --backend mlx --model mlx-community/Qwen3.6-35B-A3B-6bit
uv run python -m decisio.serve.vllm_engine --backend mlx --base gemma-4-12b --model mlx-community/gemma-4-12B-it-6bit
uv run python -m decisio.serve.vllm_engine --backend mlx --base gemma-4-31b --model mlx-community/gemma-4-31b-it-6bit
```

- `--backend mlx` serves the text route on Apple silicon from an MLX conversion of a base's checkpoint, with every feature of that route: the letters readout, the shared state prefix, the temperatures, task registration with calibration and the intent head, abstention, the rendering rules and the tie-break.
- The base comes from `--base`, or from the conversion's model type, and brings its own settings, as on vLLM; `--model` names the conversion.
- It refuses the image route, packed mode, LoRA adapters and the second-engine head.
- It refuses a checkpoint whose `config.json` declares FP8 quantization (the Qwen base's `Qwen/Qwen3.6-35B-A3B-FP8`, the `aminry/decisio-*` repositories of the Qwen base and the 31B), by `--model` or by `--base <repository>`: those weights are vLLM's format, and an MLX conversion is what `--model` takes.

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
- The 6-bit conversion `mlx-community/gemma-4-31b-it-6bit` (26.1 GB to download, 24.9 GB of weights in memory) is the one measured; the 4-bit and 8-bit conversions are not.
- Measured on one Apple M5 Pro with 64 GB; no smaller Mac was measured. GB are 1e9 bytes.
- It peaks at 29.6, 31.1 and 33.7 GB at about 8k, 16k and 32k tokens of state, against the Mac's GPU memory limit of 55.7 GB.
- **It is documented for states up to 16,383 tokens.** In the quiet reading (nothing else large running), macOS memory pressure stayed normal from the model's load through the 16k cells.
- **At 32,687 tokens it runs, with memory pressure:** in the quiet reading one of 54 samples in its cells was at the warning level, swap stayed flat (4.9 to 5.0 GB), and 39% of the GPU memory limit was still free. A second reading, taken after the head draws with the Mac's usual desktop applications open (the largest resident process was 0.5 GB), had 9 of 61 samples at the warning level, none critical, and swap up 3.7 and 1.9 GB; 8k and 16k were clean in both (`runs/2026-10-07_mlx-gemma-31b/`).
- It scores each question as its own continuation of the shared prefix, so `--multi-question` is `sequential` on MLX; the base's vLLM default, `warm`, is refused there.
- The prompts are built with the base's own tokenizer and chat template, `google/gemma-4-31B-it` at its pinned revision, not the conversion's.
- Against the base's vLLM FP8 record it passes the suite accuracy and ECE gates, conformance with answers equal to the served path's, and the intent heads over three draws per set (BANKING77 0.849 against 0.844, CLINC150 0.970 against 0.970; `runs/2026-10-07_mlx-gemma-31b/`). Server times are not measured.
- A registration is slow: about an hour for a BANKING77 draw and about 2.6 hours for a CLINC150 draw.
- Its prefix-cache entries are large: 935 MiB at 1,000 tokens, 1,869 MiB at 3,000, 2,954 MiB at 8,000 and 4,386 MiB at 32,000. At the default budget (2,048 MiB, as for the Qwen base) the cache keeps two 1,000-token states or one 3,000-token state and never stores a state of 8,000 tokens or more, so a repeat question on a long state reads it again (27 s at 8,000 tokens, 126 s at 32,000). `--prefix-cache-mb` raises the budget; each stored state takes memory beside the peaks above, and the peaks were measured with the cache empty.

`docs/design/mlx-backend.md` has the design, the gates of each conversion and the padding decision.

### Memory on a Mac

MLX keeps the buffers a computation has freed, up to a cache limit, and allows allocation up to a memory limit.
Both default to about 80 percent of the Mac's memory (51.8 GiB on a 64 GB Mac).
A server that answers many different states therefore grows toward that size, not toward the size of its weights: one Gemma 4 12B server reached about 50 GB on a 64 GB Mac, and with a second server froze the machine twice (our private records of 2026-10-14).

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

**Where it does not apply.** With `--max-model-len` the server serves exactly that length (above 32,768 only up to the ceiling; beyond it the server refuses to start, as before), and so with a `max_model_len` in `--engine`. The default stays 32,768 with `--image-model` (the image engine serves 32,768), a second engine (`--head-engine`), `--engine-process separate` (the fit is read from the engine's configuration in the server's own process), `--backend mlx` and `--backend hf`.
To keep one length on every card, give `--max-model-len`.

**What it does to certificates, and to short answers.** The context length served is part of the serving-stack fingerprint (the card, the engine, the KV cache dtype and the rest of [Where it stops](tasks.md#where-it-stops)), so a certificate made at one length is suspended at another, whether the length was given or followed the card; a card's fit is the same for the same flags, and the 8,192 step absorbs small differences in the memory vLLM reports. It is in the fingerprint because the length is not inert on every card. What was measured, on the stored FP8 31B with 200 single-question requests answered one at a time:
- On an H100 PCIe 80 GB, the default (98,304) was **exactly equal** to a fixed 32,768 start on all 200, with every tensor's bytes identical (`runs/2026-10-15_s14-session/C/`).
- On an RTX PRO 6000, 98,304 against two fresh starts at 32,768 that were exactly equal to each other: **0 of 200 equal**, median difference 4.3e-3, largest 0.389, **6 of 200 top answers changed**. A start at 32,768 on the same code and `--engine` route was exactly equal to those two, so the difference is the length (`runs/2026-10-15_identity-at-98304/`). That is the size of the start-to-start differences seen on that card class in other sessions, not a new kind of error; the likely cause is the compiled graph, which depends on the length, and that is a reading, not a test.
To get the same answers on every card, give `--max-model-len` the same value on all of them.

**What the records say about memory.** The card session of 2026-10-15 on one 96 GB card started the 31B (FP8 weights) at a context of 98,304 tokens on one 96 GB card at the default 0.90 share with the model's own cache dtype: vLLM 0.31.0 reported 48.05 GiB of KV cache and room for 141,865 tokens, and the 12B at 131,072 reported 57.53 GiB and 612,140 tokens (`runs/2026-10-15_s12-session/X/` and `X5/`). vLLM's start-up check counts the cache differently across versions: on 0.30.0 the same share of the 31B was called 48.6 GiB and 57,882 tokens (EVAL_CARD section 7; the bf16 start needed 27.51 GiB for one 32,768-token request, `runs/2026-10-04_gemma-4-31b/bf16_start.txt`), so a 0.30.0 server may need `--kv-cache-dtype fp8`, a larger `--gpu-memory-utilization`, or both beyond about 57,000 tokens, and a 0.31.0 server on a smaller card is not measured. `--kv-cache-dtype fp8` changes the numerics: a certificate made under the model's own cache dtype is suspended under fp8 (the serving-stack fingerprint includes the KV cache dtype), and the answers of a registered task can move; register and certify again under the dtype you serve.

**What the records say about time.** The first question about a state the server has not seen grows faster than the state does, because attention does (the table of first-question latency above stops at 32,000 tokens: 3,370 ms at 16,000 and 8,880 ms at 32,000 on the 31B; 1,779 ms and 4,383 ms on the 12B). The faster attention backends are refused for Gemma 4 on vLLM 0.30.0 and 0.31.0, so a longer context is slower than a proportional guess.

**Where the ceilings come from.** The card session of 2026-10-15 (the long-context checks, pass rules written before the run; `runs/2026-10-15_s12-session/`): the 31B started at a context of 98,304 tokens and answered a check at a state of 85,000 tokens (23 of 30 items right against 25 of 30 for the same items short; first question 40.8 s, second 0.40 s), which is its ceiling. The 12B started at 131,072 and was right on 23 of 30 items at 60,000 tokens (26 short; first question 11.0 s) but on 19 of 30 at 120,000 (26 short; first question 23.2 s), which failed the pass rule (a drop of at most 3), so its ceiling is the lower 65,536 (started and not itself checked; the check was at 60,000). Each check is 30 items, a smoke test and not an accuracy measurement.

**What is not measured.** Anything between the checked states and the ceilings other than what the session ran, the first and later question at those lengths, the prefix cache and the state boundary at that size, the temperatures (fitted on states of a few thousand tokens), and the accuracy of answers at long context on a suite (no suite here has a state over 32,768 tokens). A length that the card allows is not a length that was checked: to serve only what was measured, give `--max-model-len 32768`.

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
