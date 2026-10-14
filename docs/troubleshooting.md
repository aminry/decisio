<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Troubleshooting a first start

The four things that most often go wrong on a first start, with the line the server or its libraries print and what to do.
The messages are quoted from the server's own code (`docker/entrypoint.sh`, `src/decisio/serve/`), from records in `runs/`, or from a run noted beside each; where a message could not be reproduced it says so.
Failures the rest of the documentation explains (a dead engine and exit code 70, a state longer than the context, `--engine` replacing the defaults, the 12B's `--register-boundary`) are in [`running.md`](running.md) and [`api.md`](api.md).

## The card is too small

**What it looks like.** vLLM stops the engine at start-up with a `ValueError` that says how much KV cache it needs and how much it has.
The record of Gemma 4 31B at bf16 on the 96 GB card (`runs/2026-10-04_gemma-4-31b/bf16_start.txt`; the 57.91 GiB of weights left 20.24 GiB for the cache):

```
ValueError: To serve at least one request with the model's max seq len (32768), (27.51 GiB KV cache is needed, which is larger than the available KV cache memory (20.24 GiB). Based on the available memory, the estimated maximum model length is 24096. Try increasing `gpu_memory_utilization` ... or decreasing `max_model_len` when initializing the engine.
```

The container prints its own warning first, before it starts the server, when the card has less than 90 GB: `decisio: warning: the record was measured on a 96 GB card; with less memory the engine may fail to start (see EVAL_CARD.md)`.

**What fits.**
- The default base, the 31B in FP8, is measured on a 96 GB card only: 30.61 GiB of weights and 48.6 GiB of KV cache at vLLM's 0.90 share (`EVAL_CARD.md` section 7).
- The Gemma 4 12B at bf16 takes 22.8 GiB of weights and, simulated on the 96 GB card with the engine's share cut to 28.8 GB (0.90 of 32 GB), did not start at 32,768 or at 16,384 tokens of context (`EVAL_CARD.md` section 6.6): it does not start on a 32 GB card on vLLM's defaults.
- No smaller card was run: nothing here says the 31B starts or does not start on a 48 GB card, or that a smaller context makes any of these fit.

**What to do.**
1. Pick a base for the card: `--base gemma-4-12b` or `--base qwen3.6-35b-a3b` (each base's weights are in [Choosing a base](../README.md#choosing-a-base)).
2. Or lower the context, as vLLM's message says: `--engine` takes `LLM(...)` keyword arguments as JSON and **replaces the server's defaults as a whole**, so repeat the default `compilation_config`: `--engine '{"compilation_config": {"max_cudagraph_capture_size": 4096}, "max_model_len": 16384}'`. That is vLLM's own suggestion; it was not run on a smaller card for this page, and a state over the lowered context is answered with 422.
3. Or leave the card for a Mac: [`running.md`](running.md#mac-with-mlx) (the 6-bit 12B needs 15.1 GB at the longest state) or Ollama (8.6 GB for the Gemma listing's smallest tag).

## The driver does not support CUDA 13.0

**What it looks like.**
- In the container, before anything starts, from `docker/entrypoint.sh`:

  ```
  decisio: no NVIDIA GPU is visible to this container (<the exception>).
    Start it with GPU access: `docker run --gpus all ...`, or the GPU reservation in compose.yaml
    (Docker with the NVIDIA Container Toolkit on a Linux host).
    The host driver must support CUDA 13.0. Nothing was started.
  ```

  and exit code 1. The exception in the parentheses is PyTorch's; a driver that is too old gives PyTorch's `The NVIDIA driver on your system is too old` text (not reproduced here, no older driver was at hand).
- On the pip path there is no such check: the first CUDA call fails with the same PyTorch error.

**What to do.** Update the NVIDIA driver to one that supports CUDA 13.0 (`nvidia-smi` prints the highest CUDA version the installed driver supports in its header); NVIDIA's compatibility table lists the minimum Linux driver (R580 or newer for CUDA 13.0). In a container, also check that the NVIDIA Container Toolkit is installed and that the container was given the GPU (`--gpus all`, or the `deploy.resources.reservations.devices` block of `compose.yaml`).

## Port 8000 is already in use

**What it looks like.** The server starts, then ends with exit code 1 and uvicorn's line, reproduced on the CPU stand-in on macOS (`--backend hf`, a second server on the port of a first):

```
ERROR:    [Errno 48] error while attempting to bind on address ('127.0.0.1', 18555): [errno 48] address already in use
```

On Linux the errno is 98 and the address is the one the server was asked to bind (`0.0.0.0` in the container).
The engine has finished loading by then, so the failure comes after the slow part of the start.

**What to do.** Start the server on another port, `--port 8001` (in the container `DECISIO_PORT=8001`, and the published port in `compose.yaml`), or stop what holds the port: `ss -ltnp | grep :8000` on Linux and `lsof -nP -iTCP:8000 -sTCP:LISTEN` on a Mac name the process.
Some GPU rental templates run a proxy on 8000: check before assuming the port is free.

## The checkpoint download fails or stops

**What it looks like.** The first start downloads the checkpoint from the Hugging Face Hub into its cache (about 31 GB for the 31B's FP8 repository, 62 GB for Google's bf16 weights).
With the Hub unreachable and the files not in the cache, transformers' loader says so, reproduced here with `HF_HUB_OFFLINE=1` and a model that was never downloaded:

```
OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
```

The vLLM path fetches through `huggingface_hub`, whose own connection and timeout errors name the file and the host; they were not reproduced here.

**What to do.**
- Start the same command again: files that finished downloading are in the cache and are not fetched again.
- Check the disk: the cache is `~/.cache/huggingface/hub`, or `$HF_HOME`, or `/data/hf` in the container (the named volume of `compose.yaml`), and needs room for the checkpoint plus the compile caches.
- A decisio repository is pinned to a revision (`PINNED_REVISIONS` in `decisio/hub.py`); a mirror or a proxy has to serve that revision, not the branch head.
- Set `HF_HUB_OFFLINE=1` only once the files are in the cache; with it set and a file missing, the start fails with the message above.
