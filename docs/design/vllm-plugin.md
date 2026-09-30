<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Design: how decisio changes vLLM

decisio runs on stock vLLM 0.30.0.
It changes vLLM in exactly two ways, each the least invasive that works: a plugin that registers model classes, and one optional patch series.
Checked against vLLM v0.30.0's source (tag `v0.30.0`, commit `ced6857`) and verified on a card (`runs/2026-09-30_plugin-verification`).

## Summary

- **Correct answers need no patch.** Every answer comes from stock vLLM 0.30.0 behaviour; the one patch series is a latency optimisation that is inert unless an environment flag is set.
- **Our own FastAPI front process** serves the routes (`/v1/systemone`, `/v1/answer`, `/v1/tasks`, `/v1/abstention/tasks`, `/v1/models`, `/health`) and drives vLLM in process through its `LLM` class.
- **A `vllm.general_plugins` plugin** (`decisio.vllm_plugin`) registers two model classes.
- **Suffix staging** is a patch series applied at image build (`patches/`), and is to be proposed upstream.

## What vLLM gets from us

| Change | Where | Needed for correct answers? | Form |
| --- | --- | --- | --- |
| A text-only class that loads the official multimodal checkpoint without its vision tower | `decisio.vllm_plugin.models.DecisioQwen3_5MoeTextOnly` | Needed for the served latency (the multimodal class builds M-RoPE positions for every token) | Plugin |
| A text-only class that also returns the hidden state at the answer position | `decisio.vllm_plugin.models.DecisioQwen3_5MoeHiddenReadout` (docs/design/hidden-state-readout.md) | For the intent head only | Plugin |
| Staging only the uncached prompt suffix of prefix-cache hits in Model Runner V2 | `patches/vllm-0.30.0/suffix-staging` | No: 3 to 8% less time per question at 8,000-token states | Patch series, flag `VLLM_R9_SUFFIX_STAGING=1` |
| Engine settings: prefix caching, processed log-probabilities, CUDA graphs to 4,096 tokens, no multimodal inputs on the text engine, warm-ups, front padding to the block | `decisio.serve.vllm_engine` | Configuration of stock vLLM | Constructor arguments |
| `VLLM_USE_DEEP_GEMM=0` (vLLM's DeepGEMM FP8 path gave wrong results on a Blackwell card) | `vllm_engine.deep_gemm_guard` | Yes, on that card class | The launcher sets it before vLLM is imported and refuses to start otherwise (`--allow-deep-gemm` overrides) |

## The plugin

```toml
[project.entry-points."vllm.general_plugins"]
decisio = "decisio.vllm_plugin:register"
```

vLLM calls `register()` in every process it starts (front end, engine core, workers), possibly more than once, so it is re-entrant and imports nothing heavy.
It registers each class in vLLM's lazy `<module>:<class>` form, so the model module is imported only when vLLM builds one of these architectures.
It is written against one vLLM version: on any other version it registers nothing and logs one line, so stock vLLM behaves exactly as without it.
It never patches vLLM's own code.

vLLM also keys a per-architecture config hook by architecture name (`MODELS_CONFIG_MAP`): for the stock text class, the recurrent-state cache dtype from the checkpoint and the removal of the M-RoPE fields.
A class registered under a new name gets no hook, and then runs with another block size and other positions than the class it subclasses (found on the card: 544-token blocks instead of 1,056, and 50 of 1,400 suite answers equal).
So `register()` enters each class in that map with the stock class's hook; with it, the text-only class serves the 1,400-item suite bit-identically to the rewritten checkpoint.

The server selects a class per engine with `hf_overrides={"architectures": ["<name>"]}` (`decisio.vllm_plugin.engine_kwargs`, `--model-class`).
vLLM's engine processes find the plugin through the installed package's entry point, not through `PYTHONPATH`, so the package must be installed (`pip install .` or `uv sync`); the server refuses to start a decisio class otherwise.

`--model-class view` needs no plugin: `python -m decisio.serve.make_text_only` writes a directory of symlinks with the vision shards rewritten and `config.json` naming the stock text class, and vLLM loads it as it is.
It is the fallback, and the reference the classes were verified against.

## Why suffix staging is a patch and not a plugin

The change sits in the middle of four hot-path methods of Model Runner V2.
Shipped as a plugin, it would carry copies of those whole methods, pinned to one vLLM build: a patch with worse reviewability than a diff, and a risk of silent divergence the diff does not have.
As a patch series it is applied at image build or by `patches/apply.sh` to a pip-installed vLLM, with a dry run first, and `tests/unit/test_patches.py` checks that both forms apply to the tagged source.
With the flag off the patched build is stock behaviour.

## Why not vLLM's endpoint plugins

vLLM 0.30.0 has a `vllm.endpoint_plugins` group, loaded only in vLLM's own API server and only when named in `VLLM_PLUGINS`.
Our routes drive the engine in process (token prompts, prefix warm-ups, one engine lock, a second engine for images), and every gate in `runs/` was measured on that path; an endpoint plugin would mean porting them to the async `EngineClient` and measuring again, on a surface vLLM documents as liable to change between versions.
If a deployment needs our routes beside vLLM's OpenAI routes on one port, an endpoint plugin can be added later; nothing here prevents it.

## Version policy

- The `serve` extra pins `vllm==0.30.0`; the plugin registers nothing on any other version.
- Every run manifest records the vLLM version, whether the patch series was applied and its flag.
- A version bump is a deliberate change: re-apply or drop the series, run the GPU tier again (`tests/gpu`).

## Tests without a GPU

vLLM 0.30.0 has no macOS wheels, so the unit tier never imports the real package:

- the built wheel, installed alone into a fresh environment, exposes the entry point, and loading it without vLLM does nothing (`test_vllm_stub_suite.py`, V1);
- `register()` against a stand-in `vllm` (`tests/unit/stub_vllm.py`): the declared classes and their config hooks on 0.30.0, nothing on another version, once per process, lazily (`test_vllm_plugin.py`, V2);
- the patch series against the tagged source tree (`test_patches.py`, needs `bash scripts/fetch_vllm_source.sh`);
- the launcher's DeepGEMM guard and engine arguments (V3, V4);
- the CPU stand-in (`--backend hf`) serves every route, which shows the front process does not depend on the plugin.

The GPU tier (`tests/gpu`, `-m gpu`) runs the serving gates, the hidden-readout class's same-forward check and the patch series' own CUDA tests on a card.
