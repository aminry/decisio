<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Running decisio

Every way to run the server, in the order of the README's chooser ("Install and run").
The README has the short version of each; this page has what it leaves out.
Every number names the record it comes from.

## GPU server

Requirements: Linux, one NVIDIA card (measured on an RTX PRO 6000 Blackwell with 96 GB), a driver that supports CUDA 13.0 (the runtime `uv.lock` pins), Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/aminry/decisio
cd decisio
uv sync --extra serve --frozen
uv run python -m decisio.serve.vllm_engine --base qwen3.6-35b-a3b
```

- `--base qwen3.6-35b-a3b` serves `Qwen/Qwen3.6-35B-A3B-FP8` and `--base gemma-4-12b` serves `google/gemma-4-12B-it`, each at its pinned revision and with its own settings (the README's "Choosing a base"; `docs/cli.md` lists every setting per base).
- Without `--base`, the base is detected from `--model`'s `config.json`, so a local copy of either checkpoint brings its own settings; a base's pinned revision applies whenever `--model` names the base's own Hugging Face repository and `--revision` is not given.
- The first start downloads the checkpoint (about 36 GB for the Qwen base, `runs/2026-10-01_docker-first-gpu-start/`) into the Hugging Face cache and warms the engine; `/health` answers once it is ready.
- The decisio plugin registers its model classes with vLLM through an entry point, so vLLM 0.30.0 loads them without any patch; the package must be installed (as `uv sync` does) for vLLM's engine processes to find it (`docs/design/vllm-plugin.md`).
- `patches/` holds an optional latency patch series, off by default and not needed for correct answers (`patches/README.md`).
- The server refuses to start when `VLLM_USE_DEEP_GEMM` is set to anything other than `0`, unless `--allow-deep-gemm` is given.
- It listens on `127.0.0.1:8000` (`--host`, `--port`); put a reverse proxy in front of it to expose it.
- `--image-model` adds a second engine on the same card for requests that carry images; `EVAL_CARD.md` section 1 has the memory shares it was measured with.
- vLLM's engine runs in the server's process by default; with a second engine (`--image-model`, `--head-engine`, `--one-engine`) it runs in a process of its own (`--engine-process`, `docs/cli.md`).
- A later, different question about a state read before, sent in its own request, reads the state from the prefix cache. On the Gemma bases vLLM keeps only a finished request's latest sliding-window checkpoint, which lies inside its question, so a single question first registers its state's boundary with the warm-up that multi-question requests already send (the state and one token). The server remembers which states it registered and sends the warm-up again only when a request finds the boundary gone, so a repeated question costs no extra engine call. 0.8.1 made that second question read the state from the cache: on one card at 400 W with an AMD Ryzen 9 9950X, 29.2 ms on Gemma 4 12B and 37.1 ms on Gemma 4 31B after first reads of 348 and 615 ms on a 3,000-token state, where before it read the whole state again (531 ms instead of 47 at 3,000 tokens and 9.2 s instead of 0.17 at 31,000 on the 31B, on another host). In exchange the registration adds to the first read of a new state: +11, +37 and +59 ms at 300, 1,000 and 3,000 tokens on the 12B, and +23, +43 and +72 ms on the 31B, on the same host (`runs/2026-10-06_latency-0.8.1/`). A repeated question and several questions in one request were not affected.
- vLLM's `prefix_cache_retention_interval` would also keep the boundary, but it keeps every sliding-window block of each state, and with the cache filled to 1.3 times its pool the earliest states lost even their repeated question, where the default (0) kept them at twice the pool; it stays at the default. The Qwen base needs neither: its state is padded to end on its 1,056-token block, the latest checkpoint of a request with one question. `/health` reports `cache_hit_unit`, `hash_unit` and `registers_state_boundary`.

Per-request latency depends on the host's CPU and on the card's power limit, as well as the card.
- The CPU: with the engine in the server's process, a Qwen question on a cached state took 39.4 ms of server time on an AMD EPYC 7452 (Zen 2) host and 18.5 ms on an AMD Ryzen 9 9950X (Zen 5) host, with the same card model and vLLM (`runs/2026-10-05_engine-death-gates/`, `runs/2026-10-06_latency-0.8.1/`). The Gemma bases differed far less between the two hosts (Gemma 4 12B: 28.0 against 23.4 ms).
- The power limit: first reads are bound by the card. On the Ryzen host the card was limited to 400 W of its 600 W, and its first reads on the Gemma bases were slower than earlier records whose power limit was not captured.

Quote latency with the host's CPU model and the card's power limit beside it (`nvidia-smi -q -d POWER`).

## Docker

```bash
uv build --wheel                   # the image installs this wheel
docker compose up --build          # needs the NVIDIA Container Toolkit and a GPU host
curl http://127.0.0.1:8000/health  # answers once the first start has fetched the checkpoint
```

- The image is vLLM's 0.30.0 release image, pinned by digest, plus the decisio wheel with its `serve` extra; the build argument `APPLY_PATCHES=1` adds the patch series (off by default).
- The first start downloads the checkpoint into the `decisio-data` volume, mounted at `/data`; it is never part of the image.
- The container runs as a non-root user, and compose publishes the port on 127.0.0.1 only.
- The entrypoint starts the server with `--model "${DECISIO_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"` and `--model-class hidden-readout`, listening on `DECISIO_HOST` and `DECISIO_PORT`; arguments after the image name are passed to the server.
- It checks for an NVIDIA GPU first and exits with an explanation when the container cannot see one.
- Release images go to `ghcr.io/aminry/decisio`, with their digest in the release notes; to run one instead of building, replace `build:` in `compose.yaml` with `image: ghcr.io/aminry/decisio@sha256:<digest>`.
- `Dockerfile` and `compose.yaml` explain the rest.

First GPU start (2026-10-01, one RTX PRO 6000 Blackwell, the image built on the machine from the `Dockerfile`): healthy in 651 s including the checkpoint download and in 206 s from the cached volume, and the README's example request and the conformance gates C2 to C4 pass against the container (`runs/2026-10-01_docker-first-gpu-start/`).
The image published with v0.1.0 repeated the start from its digest: healthy in 223 s with the checkpoint cached, example and conformance passing (`pushed_image_0.1.0/` in the same record).
The image has been started on a GPU with the Qwen base only.

## Mac with MLX

```bash
uv sync --extra mlx
uv run python -m decisio.serve.vllm_engine --backend mlx --model mlx-community/Qwen3.6-35B-A3B-6bit
uv run python -m decisio.serve.vllm_engine --backend mlx --base gemma-4-12b --model mlx-community/gemma-4-12B-it-6bit
```

- `--backend mlx` serves the text route on Apple silicon from an MLX conversion of either base's checkpoint, with every feature of that route: the letters readout, the shared state prefix, the temperatures, task registration with calibration and the intent head, abstention, the rendering rules and the tie-break.
- The base comes from `--base`, or from the conversion's model type, and brings its own settings, as on vLLM; `--model` names the conversion.
- It refuses the image route, packed mode, LoRA adapters and the second-engine head.

The Qwen base:
- The 6-bit conversion is the Mac default and the 4-bit one the option for 32 GB machines; the 8-bit one is not shipped.
- Peak memory at 32,761 tokens of state: 30.7 GB at 6 bits, 22.0 GB at 4 bits, 39.4 GB at 8 bits (`runs/2026-10-02_mlx-backend/`).
- It does not pad a state (`--pad-policy none`, the MLX default; its cache needs no padding), so its prompts are the served ones without the padding, and a question asked alone and inside a request is one prompt.
- A cross-request prefix cache (`--prefix-cache-mb`, 2048 by default; 0 turns it off) lets a request whose state was seen before continue from the kept cache, with the same answers bit for bit.
- The prompts are built with the official tokenizer by default (`--tokenizer`), so they are the vLLM path's byte for byte.
- At 6 bits, under the current served default, it passes the gates against the FP8 records except the pooled-ECE gate, which changes no setting (`runs/2026-10-03_mlx-regate/summary.md` explains why).
- Server time on an Apple M5 Pro (64 GB), median: 266 ms for one question on a new state, 111 ms on a state seen before, 130 ms per question when 100 share a state (`runs/2026-10-03_mlx-regate/`).

The Gemma base (`--base gemma-4-12b`):
- The 6-bit conversion is the default; there is no 4-bit option, since the 4-bit conversion fails the suite-accuracy gate (`runs/2026-10-05_mlx-gemma/`).
- The 6-bit conversion holds 9.7 GB of weights and peaks at 15.1 GB at 32,687 tokens of state, so it runs on a 32 GB Mac (`runs/2026-10-05_mlx-gemma/6bit/memory.json`).
- The prompts are built with the base's own tokenizer and chat template, `google/gemma-4-12B-it` at its pinned revision, not the conversion's, whose chat template is older.
- At 6 bits it passes every gate against the base's vLLM bf16 record: suite accuracy and ECE, the intent heads over six draws, conformance, and answers bit for bit on a fresh server (`runs/2026-10-05_mlx-gemma/summary.md`).
- It reads a new state more slowly than the Qwen base on the same Mac: one question on a new 1,000-token state takes 1,299 ms, against 704 ms for the Qwen base (server medians on an Apple M5 Pro under load; `runs/2026-10-05_mlx-gemma/`, `runs/2026-10-03_mlx-regate/`).

`docs/design/mlx-backend.md` has the design, the gates of each conversion and the padding decision.

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
- An error caused by the request (a malformed question, a state longer than the context) is answered with a 4xx and the engine is not probed.

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

Docker restarts a container when its process exits, not when its `HEALTHCHECK` reports it unhealthy, so it is the exit that brings the server back; the image's check reports the dying container unhealthy in the meantime.
The new server loads the engine again (the start times above), and requests sent until it is healthy are refused.

In decisio 0.1.0 to 0.7.1 none of this happens: after the engine dies `/health` keeps answering 200 and the process keeps running, so no restart policy acts.
In the separate arrangement every request then fails with 500; with `--engine-process in` (0.7.0 and 0.7.1) every request after the failing one hangs.
