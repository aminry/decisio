<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Running decisio

Every way to run the server, in the order of the README's chooser ("Install and run").
The README has the short version of each; this page has what it leaves out.
Every number names the record it comes from.

## GPU server

Requirements: Linux, one NVIDIA card (measured on an RTX PRO 6000 Blackwell with 96 GB), a driver that supports CUDA 13.0 (the runtime `uv.lock` pins), Python 3.12 and [uv](https://docs.astral.sh/uv/).

```
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

## Docker

```
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

```
uv sync --extra mlx
uv run python -m decisio.serve.vllm_engine --backend mlx --model mlx-community/Qwen3.6-35B-A3B-6bit
```

- `--backend mlx` serves the text route on Apple silicon from an MLX conversion of the Qwen base's checkpoint, with every feature of that route: the letters readout, the shared state prefix, the temperatures, task registration with calibration and the intent head, abstention, the rendering rules and the tie-break.
- It refuses the image route, packed mode, LoRA adapters, the second-engine head and the Gemma base.
- The 6-bit conversion is the Mac default and the 4-bit one the option for 32 GB machines; the 8-bit one is not shipped.
- Peak memory at 32,761 tokens of state: 30.7 GB at 6 bits, 22.0 GB at 4 bits, 39.4 GB at 8 bits (`runs/2026-10-02_mlx-backend/`).
- It does not pad a state (`--pad-policy none`, the MLX default; its cache needs no padding), so its prompts are the served ones without the padding, and a question asked alone and inside a request is one prompt.
- A cross-request prefix cache (`--prefix-cache-mb`, 2048 by default; 0 turns it off) lets a request whose state was seen before continue from the kept cache, with the same answers bit for bit.
- The prompts are built with the official tokenizer by default (`--tokenizer`), so they are the vLLM path's byte for byte.
- At 6 bits, under the current served default, it passes the gates against the FP8 records except the pooled-ECE gate, which changes no setting (`runs/2026-10-03_mlx-regate/summary.md` explains why).
- Server time on an Apple M5 Pro (64 GB), median: 266 ms for one question on a new state, 111 ms on a state seen before, 130 ms per question when 100 share a state (`runs/2026-10-03_mlx-regate/`).

`docs/design/mlx-backend.md` has the design, the gates of each conversion and the padding decision.

## Ollama

```
ollama pull aminroudaki/decisio
```

Requires Ollama 0.35.1 or later.
The [model page](https://ollama.com/aminroudaki/decisio) packages a laptop version of the Qwen base for Ollama's decision route and lists its tags and their sizes.
Ollama builds its own prompt and applies no calibration, so its numbers are the ones on that page, not the ones in this repository.

## CPU stand-in for development

```
uv sync --extra dev --frozen
uv run pytest
uv run python -m decisio.serve.vllm_engine --backend hf --model Qwen/Qwen3-0.6B-Base
```

- `uv run pytest` runs the unit tests on any machine; the GPU tests (`pytest -m gpu`) need a card and run nightly.
- `--backend hf` serves every route from a small Hugging Face model on the CPU, for developing and testing the routes; it is not the measured system and its answers are not for measurement.
- CI runs the stand-in on `Qwen/Qwen3-0.6B-Base` at a pinned revision (`STAND_IN_REVISION` in `.github/workflows/ci.yml`).
- `examples/tasks/` walks through task registration against it.

## Checking a server

`curl http://127.0.0.1:8000/health` returns what the server is serving: the engine, the base, the checkpoint and its revision, the temperature for each question type and the prompt.
`docs/api.md` describes every field.
