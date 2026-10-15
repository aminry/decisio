<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Upgrading

What changes for a running deployment from one release to the next, newest first.
The changelog (`CHANGELOG.md`) lists every change; this page is for the ones that change what a deployment serves.

## 0.14.0: the default context follows the card

### What changed

- A vLLM server started without `--max-model-len` on a Gemma base now serves **the largest context that fits the card**, up to the base's ceiling (31B 98,304; 12B 65,536) and not below 32,768, instead of a fixed 32,768. The Qwen base stays at 32,768.
  The server asks vLLM for the longest single request the card's KV cache holds and serves the largest multiple of 8,192 of that; `/health` reports the length (`max_model_len`) and why (`max_model_len_source`, for example "auto: fits 81,920 of 98,304 on this card").
- Why: a fixed longer context can fail to start on a smaller card, and a failed start loses the whole run, while a fixed 32,768 leaves out the prompts of 50,000 to 98,000 tokens that a bigger card holds.
  The default now never fails where 32,768 would have started, and never goes above the ceiling.
  How long that is depends on the card and on `--gpu-memory-utilization`, strongly and not linearly: on an H100 PCIe 80 GB the 31B served 98,304 at the default share, 49,152 at 0.85, and did not start at 0.70 (`running.md` has the table); a card where 32,768 does not fit fails to start as it did before.
- Details: [`running.md`](running.md#the-context-length).

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| The 31B or the 12B on vLLM with no `--max-model-len` | A prompt between 32,768 tokens and the chosen length is answered instead of refused with 422; it takes longer the longer it is (the first question about a new 32,000-token state took 8.9 s on the 31B at 600 W) | Nothing, or give `--max-model-len 32768` to keep the old limit |
| The same with `--max-model-len N`, or a `max_model_len` in `--engine` | The same length as before | Nothing |
| The Qwen base, `--image-model`, `--head-engine`, `--engine-process separate`, `--backend mlx` or `hf` | 32,768, as before | Nothing |
| Certified answers | The context length is not part of the serving-stack fingerprint, so a certificate is not suspended by the new default or by moving to a card that fits a different length | Nothing |
| A client that relied on 422 for prompts over 32,768 tokens | Those prompts are answered where the card allows it | Check the length on your side, or pin `--max-model-len 32768` |

## 0.11.0: the model repositories moved to tachara-ai

### What changed

- The three model repositories are now under the Hugging Face organisation `tachara-ai` (Tachara AI Lab), with the same names: `tachara-ai/decisio-gemma-4-31b`, `tachara-ai/decisio-gemma-4-12b` and `tachara-ai/decisio-qwen3.6-35b-a3b`.
  The code, the weights and the settings are the same as in 0.10.0.
- Each repository has a new revision (a new card, and a `decision_config.json` that names 0.11.0), tagged `v0.11.0`; 0.11.0 pins these.
  The 0.10.0 revisions are still in each repository, tagged `v0.10.0`, and 0.10.0 still serves them.

### The old names keep working

- Hugging Face redirects `aminry/decisio-gemma-4-31b`, `aminry/decisio-gemma-4-12b` and `aminry/decisio-qwen3.6-35b-a3b` to the new names.
  A fresh install of 0.10.0 opens all three through the old names at its pinned revisions, with no token (checked on 2026-10-08, after the move).
- 0.11.0 treats an old name as the new one: `--base aminry/decisio-gemma-4-31b` serves `tachara-ai/decisio-gemma-4-31b` at 0.11.0's pin, and `/health` reports the new name.
  `--base aminry/decisio-gemma-4-31b@<commit>` pins that commit as before.

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| A command, script or image that names `aminry/decisio-...` | It works through the redirect, on 0.10.0 and on 0.11.0 | Change the name when convenient |
| A deployment with the models already in its Hugging Face cache | It downloads the new revision of a repository it serves by default (about 31 GB for the 31B) | Pin `@<commit>` to keep the 0.10.0 revision |
| `--base gemma-4-12b` or `--base qwen3.6-35b-a3b` | Nothing: those serve the source checkpoints, as before | Nothing |

## 0.10.0: the default base is Gemma 4 31B

### What changed

- A vLLM server started with neither `--base` nor `--model` serves `gemma-4-31b`.
  Before, it refused to start.
- The container, `docker compose` and the README's Docker path served the Qwen base before, because the image set `DECISIO_MODEL=Qwen/Qwen3.6-35B-A3B-FP8`.
  The image no longer sets it: the entrypoint starts the server with `--base "${DECISIO_BASE:-gemma-4-31b}"`.
- Gemma 4 31B is served from its FP8 repository, `aminry/decisio-gemma-4-31b` (moved to `tachara-ai/decisio-gemma-4-31b` in 0.11.0, above), so the first start downloads about 31 GB and the server does not quantize Google's weights at every start.
  `--model google/gemma-4-31B-it` serves Google's weights at the pinned revision instead, quantized to FP8 each time they load, as before.
- Nothing else changed for the Qwen and Gemma 4 12B bases: their checkpoints, settings, served names and answers are the same as in 0.9.0.
  They are still chosen with `--base qwen3.6-35b-a3b` and `--base gemma-4-12b`.

### Who is affected

| You run | What happens | What to do |
| --- | --- | --- |
| `docker compose up`, or the image, with no `DECISIO_MODEL` set | The container now serves Gemma 4 31B, which needs a 96 GB card | To keep the Qwen base, set `DECISIO_BASE=qwen3.6-35b-a3b` (in `compose.yaml`'s `environment`, or `-e` on `docker run`) |
| The image with `DECISIO_MODEL` set | Nothing: a checkpoint alone names its own base, so the container serves what it served | Nothing |
| A derived image that relied on `ENV DECISIO_MODEL` coming from this image | It no longer comes from it, so the default base is served | Set `DECISIO_MODEL` or `DECISIO_BASE` in your image |
| `uv run python -m decisio.serve.vllm_engine --base ...` or `--model ...` | Nothing | Nothing |
| A Mac with `--backend mlx`, or Ollama | Nothing: they always named a base or a conversion (`--backend mlx` needs `--model`), so the default base changes nothing for them | Nothing |
| A client that reads the served name from `GET /v1/models`, or matches `decisio-qwen3.6-35b-a3b-letters` | A server on the default base lists `decisio-gemma-4-31b-it-letters` | Read the name from `/v1/models`, or start the server with `--served-name` |
| Registered tasks (`POST /v1/tasks`, or a `--tasks-file`) fitted on the Qwen base | A server on another base does not apply them: it prints `WARNING: tasks fitted under another model or rendering are not applied` at start, and `POST /v1/tasks/import` refuses them | Register them again from their examples (`docs/tasks.md`), or keep serving the Qwen base |

### What differs in what you get from the default base

- The wire format and the routes are the same.
- The probabilities are another model's, calibrated with its own temperatures and prompt (`docs/cli.md`), so some choices differ from the Qwen base's.
  `EVAL_CARD.md` section 7 has the 31B's measurements and section 3 the Qwen base's; the README's "Choosing a base" says when each is the better choice.
- It reads a new state more slowly: 1.6, 3.2 and 5.1 times the Qwen base's time at 300, 1,000 and 3,000 tokens (80.7, 160.1 and 438.1 ms against 49.9, 50.0 and 85.2, one card at 585 W; `runs/2026-10-06_latency-585w/`).
- It needs a 96 GB card on vLLM; on a Mac it runs from its 6-bit MLX conversion for states up to 16,383 tokens (`docs/running.md`).
- Its probabilities repeat exactly within a session, not across sessions: near-tied answers moved between two sessions in our measurements (`EVAL_CARD.md` 7.5).

### Keeping the previous behaviour

```bash
uv run python -m decisio.serve.vllm_engine --base qwen3.6-35b-a3b    # the Qwen base, as the container served it
DECISIO_BASE=qwen3.6-35b-a3b docker compose up                       # the same in the container
```
