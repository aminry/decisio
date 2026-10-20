<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# The command line

```bash
uv run python -m decisio.serve.vllm_engine [flags]
```

[`reference/flags.md`](reference/flags.md) is the complete table of the flags, their defaults and values, generated from the server's argument parser at release time; this page says what each base takes.

`--base` chooses the base model and brings its checkpoint at a pinned revision and its value for every setting marked "the base's" below; a flag given explicitly overrides it.
With a `--model` and no `--base`, the base is detected from the checkpoint's `config.json`. A checkpoint whose `model_type` no base serves, or whose `config.json` cannot be read, is refused with the supported bases named, because a base's classes, token ranges and temperatures are fitted for its own model; `--base <key>` with the same `--model` is the explicit way to serve it as that base's profile, if it is one. The CPU stand-in (`--backend hf`) is the exception: it takes any small model and uses the Qwen profile.
With neither, the vLLM server serves `gemma-4-31b`, the default base (the CPU stand-in and the MLX backend always name a checkpoint, and refuse to start without one).
`--base` also takes a decisio model repository, `owner/name[@revision]` or a directory holding its `decision_config.json`: the repository's weights are the checkpoint, its file names the base, and the server refuses to start if the file differs from the installed profile (`docs/running.md`, "Model repositories").
The served defaults need no other flag: every default below is what was measured, base by base (`EVAL_CARD.md` sections 1, 6.1 and 7.1).
`gemma-4-31b` is served from its FP8 repository, `tachara-ai/decisio-gemma-4-31b`, and `--model google/gemma-4-31B-it` serves Google's weights instead, quantized to FP8 when they load (vLLM's `fp8_per_tensor` online quantization, which `quantization="fp8"` redirects to), a setting of its profile; `--engine '{"quantization": null}'` loads them at bf16, which does not leave room for a 32,768-token context on a 96 GB card (`EVAL_CARD.md` section 7).
`--help` prints the same list with each flag's description.

## Every flag

Where the columns differ, the value is the base's.

### Model and engine

| Flag | `qwen3.6-35b-a3b` | `gemma-4-12b` | `gemma-4-31b` | What it does |
| --- | --- | --- | --- | --- |
| `--base` | | | the default | the base, or a decisio model repository (`owner/name[@revision]`, or a directory); without it, detected from `--model`, or the default if no `--model` either |
| `--model` | `Qwen/Qwen3.6-35B-A3B-FP8` | `google/gemma-4-12B-it` | `tachara-ai/decisio-gemma-4-31b` (its FP8 repository; `google/gemma-4-31B-it` serves Google's weights instead) | the checkpoint, a local directory or a Hugging Face repository id |
| `--revision` | `95a723d08a9490559dae23d0cff1d9466213d989` | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` | `eaf1e4fa30a351ab87d07803ca60f1fa5606f882` (the repository's; `842da3794eaa0b77d5f08bae87a17459d91ff475` for `google/gemma-4-31B-it`) | the checkpoint's revision on the Hub; the pin applies whenever `--model` is the base's own repository |
| `--backend` | `vllm` | `vllm` | `vllm` | `vllm`; `mlx` for Apple silicon (the three bases, `docs/running.md`); `hf` for the CPU stand-in, not for measurement: it serves a base's own checkpoint (`--base` and `--model` that belong together), not another model under a base's profile (Gemma 4 E4B under `--base gemma-4-31b` carries the 31B's soft cap, prompt and temperatures and is not supported; a NaN it produces is answered 500 with the model and engine named, `docs/api.md`) |
| `--model-class` | `hidden-readout` | `hidden-readout` | `hidden-readout` | `hidden-readout`: decisio's text class that also returns the hidden state at the answer position, for the intent head; `text-only`: the same without it (the default with `--head-engine`); `view`: a directory built by `decisio.serve.make_text_only`, loaded as it is, needing no plugin |
| `--mode` | `separate` | `separate` | `separate` | `separate`: one prompt per question, the state shared through the prefix cache; `packed`: questions packed into one pooling request (the compact layout only) |
| `--pack` | 16 | 16 | 16 | questions per pack in packed mode |
| `--engine` | `{"compilation_config": {"max_cudagraph_capture_size": 4096}}` | the same | the same | extra `LLM(...)` keyword arguments, as JSON |
| `--adapter` | none | none | none | `name=path` of a vLLM-format LoRA adapter; repeatable. A request uses it only when its body names it (`"adapter": "<name>"`). An adapter whose tensors are named for the other layout than the served class (`base_model.model.model.language_model.layers.` from a multimodal checkpoint, against `base_model.model.model.layers.` on a text-only base, or the reverse) is served from a mapped copy, made before the engine is built, with the tensor bytes unchanged; `/health` reports the mapping in `adapter_names`. An adapter in neither layout is refused at once, before the weights load |
| `--adapter-temperatures` | none | none | none | `name=file` of a `decisio.adapter/1` file of that adapter's temperatures per question type (`choice`, `noul`, `score`) and, for choice, optionally per band of option count (`choice_bands`); repeatable. A request that names the adapter is served at them; a type the file does not list keeps the base's. It replaces the `decisio_adapter.json` beside the adapter's weights, which is read when the flag is not given. An unusable file, or a name `--adapter` did not load, is refused before the engine is built. `/health` reports them in `adapter_temperatures`, and the adapter's weights hash and temperatures are part of the serving-stack fingerprint ([running.md](running.md#lora-adapters)) |
| `--gpu-memory-utilization` | 0.9 (0.47 with `--image-model`) | 0.9 | 0.9 | the text engine's share of the card. With `--image-model` the two engines share the card, and the default is the measured pair's 0.47; a pair that cannot fit (over 1.0 together, or over 0.96 with a default in it) is refused at start, before either engine is loaded |
| `--max-model-len` | 32768 | the largest that fits the card, 32768 to 65536 | the largest that fits the card, 32768 to 98304 | the longest prompt in tokens. Without it the vLLM server on a Gemma base serves the largest multiple of 8192 that fits the card's KV cache, up to the base's ceiling and not below 32768; given, that length, refused above the ceiling. The Qwen base, `--image-model`, a second engine, `--engine-process separate` and the other backends stay at 32768 ([`running.md`](running.md#the-context-length)) |
| `--kv-cache-dtype` | `auto` | `auto` | `auto` | `auto` (the model's own) or `fp8`, which holds twice the tokens in the KV cache and changes the numerics: a certificate is made under one dtype and suspended under the other |
| `--allow-deep-gemm` | off | off | off | start even when `VLLM_USE_DEEP_GEMM` is set to something other than 0 |
| `--served-name` | `decisio-qwen3.6-35b-a3b-letters` | `decisio-gemma-4-12b-it-letters` | `decisio-gemma-4-31b-it-letters` | the name `GET /v1/models` lists |
| `--host`, `--port` | `127.0.0.1`, 8000 | the same | the same | the listen address; put a reverse proxy in front to expose it |

### The prompt

| Flag | `qwen3.6-35b-a3b` | `gemma-4-12b` | `gemma-4-31b` | What it does |
| --- | --- | --- | --- | --- |
| `--prompt-tail` | `spaced` | `spaced` | `spaced` | `spaced`: a blank line before and after the lettered options, then one line asking for the chosen option's letter alone; `compact`: the compact layout, for tasks registered under it |
| `--answer-slot` | `prefill` | `template` | `template` | where the label is read: after "Answer:" prefilled in the assistant turn, or at the chat template's own first assistant position |
| `--label-variants` | `single` | `summed` | `summed` | the tokens read per label: the one form the slot reads, or every single-token form summed |
| `--system-prompt` | off | on | on | a system turn before each question |
| `--noul-rendering` | `letters-keys` | `letters` | `letters` | how a yes/no question is asked: `letters-keys`, a two-option letter choice, the false side first, the sides named; `letters`, the same with each side shown as its description; `words`, read from the yes and no tokens |
| `--describe-options` | on | on | on | show an option that has a description as its description alone; `--no-describe-options` shows `key: description` |
| `--hide-index-keys` | on | on | on | never show enumerated keys (`option_0`, `option_1`, ...) |
| `--desnake-labels` | on | on | on | show bare snake_case labels as words |

### Temperatures and outputs

| Flag | `qwen3.6-35b-a3b` | `gemma-4-12b` | `gemma-4-31b` | What it does |
| --- | --- | --- | --- | --- |
| `--temperature` | 1.506 | 3.592 | 5.252 | the temperature on the text route's plain readout, `softmax(log p / T)`; 1 switches it off; never changes the most probable option |
| `--temperature-choice` | 1.370 | the global one | 4.672 | the temperature for choice questions |
| `--temperature-noul` | the global one | the global one | the global one | the temperature for yes/no questions |
| `--temperature-score` | the global one | the global one | the global one | the temperature for score questions |
| `--noul-commit`, `--no-noul-commit` | off | off | off | report an uncommitted yes/no probability at the band's edge (below); a tuned repository's `defaults` state their own setting, and either flag overrides it |
| `--think`, `--no-think` | off | off | off | think when unsure ([running.md](running.md#thinking-when-unsure)): a question below its type's threshold times `--think-factor` is read again after the served model has written up to `--think-budget` tokens; a request's `think` field decides for that request, a tuned repository's `defaults.think` turns it on, and `--no-think` turns that off |
| `--think-factor`, `--think-budget` | 1, 512 | 1, 512 | 1, 512 | the factor on the thresholds (choice 0.75, yes/no 0.92, score 0.70) and the most reasoning tokens per question; a tuned repository's defaults state their own |
| `--over-limit` | refuse | refuse | refuse | a state over the engine's prompt ceiling: `refuse` answers 422, `truncate` cuts the state to its start and answers (never certified; [running.md](running.md#states-over-the-limit)); a request's `over_limit` field decides for that request, and a tuned repository's `defaults` state their own setting |
| `--over-limit-tokens` | the context less 1 | the context less 1 | the context less 1 | the prompt ceiling a truncating server cuts to; needed on the MLX build, which has no context length to read |
| `--orders` | 1 | 1 | 1 | 2: two-order averaging, each question also read in a second option order |
| `--branch-log` | none | none | none | a JSONL file for the two orders' disagreement per question |

### Several questions and padding

| Flag | `qwen3.6-35b-a3b` | `gemma-4-12b` | `gemma-4-31b` | What it does |
| --- | --- | --- | --- | --- |
| `--multi-question` | `sequential` | `sequential` | `warm` with the engine in the server's process, else `sequential` | how a request's questions are scored (below) |
| `--register-boundary` | not used (no registration needed) | `before` | `before` | when a single question on a new state registers its boundary for the next question: `after`, once the response is out; `before`, ahead of the question; `off`, never (`docs/running.md`) |
| `--engine-process` | `in` (`separate` with a second engine) | `in` (`separate` with a second engine) | `in` | where vLLM's engine runs: `in`, the server's process; `separate`, a process of its own (vLLM's arrangement) (below) |
| `--pad-policy` | `always` | `always` | `always` | `always`: pad every state; `shared`: pad only requests with more than one question; `row`: pad a single-question request's whole row (below); `none`: never pad (the default with `--backend mlx`) |
| `--pad-to` | `block` | `none` | `none` | what a state is padded to: the KV cache block, a token count, or nothing |
| `--pad-where` | `front` | `front` | `front` | where the padding goes: `front`, before the chat template; `user`, at the start of the user turn; `between`, between the state and the question |

### Tasks and abstention

| Flag | `qwen3.6-35b-a3b` | `gemma-4-12b` | `gemma-4-31b` | What it does |
| --- | --- | --- | --- | --- |
| `--tasks` | on | on | on | apply registered per-task calibration and intent heads (`POST /v1/tasks`) |
| `--tasks-file` | none | none | none | a JSON list of tasks to load at start, as `GET /v1/tasks?full=1` returns them |
| `--head-engine` | off | off | off | read the intent head's hidden state from a second copy of the model in vLLM's pooling mode instead of from the serving engine: faster head questions, at a second weight copy; not with `--image-model` |
| `--head-gpu-memory-utilization` | 0.47 | 0.47 | 0.47 | the second engine's share of the card |
| `--abstention` | on | on | on | apply registered abstention thresholds (`POST /v1/abstention/tasks`) |
| `--abstention-tasks` | none | none | none | a JSON list of abstention tasks to load at start |
| `--abstain-option` | none | none | none | offer this extra option (for example "can't tell") on requests that use imajev's extension, reported as `unknown_probability` and `abstained`; untrained |

### Images

| Flag | `qwen3.6-35b-a3b` | `gemma-4-12b` | `gemma-4-31b` | What it does |
| --- | --- | --- | --- | --- |
| `--image-model` | none | none | none | the full multimodal checkpoint, as a second engine serving only requests that carry images; measured with the Qwen base only |
| `--max-images` | 128 | 128 | 128 | with `--image-model`: the most images one request may carry (128 images of 512 by 512 pixels fill the context, so for photos the context ends a request; lower it to save multimodal profiling memory at start-up, raise it for smaller images) |
| `--image-gpu-memory-utilization` | 0.49 | 0.49 | 0.49 | the image engine's share of the card, beside the text engine's 0.47; a pair with a default in it may total at most 0.96, a pair given in full at most 1.0 |
| `--one-engine` | off | off | off | load only the image engine and serve text requests on it too, for a card that cannot hold both; changes the text route's class |

### The Mac (`--backend mlx`)

| Flag | Default | What it does |
| --- | --- | --- |
| `--prefix-cache-mb` | the base's: 2048 (Qwen and `gemma-4-31b`), 7400 (`gemma-4-12b`) | the cross-request prefix cache's budget in MiB (2^20 bytes); 0 turns it off |
| `--mlx-memory-limit-gb` | what the base needs, at most two thirds of the Mac's memory | the MLX memory limit in GiB; 0 leaves MLX's own (about 80 percent of the memory); a base that does not fit is refused ([Memory on a Mac](running.md#memory-on-a-mac)) |
| `--mlx-cache-limit-gb` | 2 | the MLX cache limit in GiB: freed buffers kept for reuse |
| `--tokenizer` | `Qwen/Qwen3.6-35B-A3B-FP8` | the tokenizer the prompts are built with, so they are the vLLM path's byte for byte |

### Verification

| Flag | Default | What it does |
| --- | --- | --- |
| `--debug-readout` | off | honour the `x-decisio-debug` request header (the raw readout in the response); never in production |

## What the behaviour flags do

### Several questions in one request (`--multi-question`)

With `sequential`, the served default, a request with several questions first prefills the state once, then scores each question in its own engine call, reading the state from the prefix cache, so every answer equals the same question sent alone.
With `warm`, for bulk scoring, the questions are scored in one batch after the same prefill: faster (four questions on a new state take about half the time of `sequential` on the Qwen base and about two thirds on the Gemma bases), but a batched answer differs from the same question sent alone, and when two options are close the choice can change (`EVAL_CARD.md` section 4).
With the engine in its own process (`--engine-process separate`) a warm answer can also move between repeats, because that process does not always put a request's questions in one engine step (vllm-project/vllm#59764); with the engine in the server's process (`--engine-process in`) an identical four-question warm request repeated exactly on a server that had answered it before (272 of 272 requests, 5 repeats each, vLLM 0.30.0); a single-question request repeated on such a server can differ slightly (5 of 200 requests, up to 0.031, on vLLM 0.30.0 and 0.31.0 alike; `EVAL_CARD.md` sections 4 and 7.5).
`--engine-process` resolves to `in` for a single-engine server and to `separate` when a second engine is configured (`--image-model`, `--head-engine`, `--one-engine`), with which `in` is refused; an explicit choice is honoured, and the server prints the arrangement and why at start-up and reports it in `/health` as `engine_process`.
On the Gemma 4 31B base `warm` is the default when the engine runs in the server's process: against `sequential` it changed no choice on the suite, JevBench and the travel requests, by at most 0.0073 in a probability (`runs/2026-10-05_engine-death-gates/`); the other bases keep `sequential`.
With `batch`, every question goes in one engine call that prefills the state itself, with no warm-up.
Measurements: `EVAL_CARD.md` section 4, `runs/2026-10-02_multi-question-and-rendering/`, and the README's example request in `runs/2026-10-01_docker-first-gpu-start/repeat_variability/`.

### Row padding (`--pad-policy row`)

By default every state is padded to end on the KV cache block, so a single question on a state the server has not seen is prefilled in two engine steps, the state and then the question.
With `row`, a single-question request is padded so that its whole row, state and question, ends on the block boundary, and vLLM prefills it in one engine step; requests with several questions are padded as by default.
It is faster on a new state, and accuracy and calibration stayed within noise on the suite, JevBench, the four Decision Index benchmarks and the intent heads.
Its cost is repeatability: a repeated identical request can move slightly on long prompts, where the default padding returns the same probabilities every time, so it is off by default.
The demos ran with it (`docs/demos/README.md`).
Measurements: `runs/2026-10-02_pad-policy-row/`; not measured on the Gemma base.

### Determinism

**Qwen base.** On a running server a question returns the same probabilities every time, alone or among other questions: each question is scored in its own engine call.
Across server restarts, single-question answers have matched an earlier record on the whole suite in most starts measured, and moved slightly without changing a choice in the others.
**Gemma 4 31B (the default) and Gemma 4 12B.** Neither repeats bit for bit: their logits come out in bf16 after the soft cap, and a state read for the first time and read again from the prefix cache can land a step apart; several questions in one request stay within the bound in the README's settings table.
For the 31B, on one machine, two fresh starts gave the same 200 single-question answers on all 200 items, but asking the same 200 again on the server that had just answered them changed 5 (largest 0.031, no top answer changed); on other machines the same requests differed from an earlier machine's record on every item (median 4.3e-3, largest 0.379, 4 top answers changed). So the 31B repeats exactly on a freshly started server on one machine, and nothing is promised across machines (`EVAL_CARD.md` section 7.5).
The 12B returns the same choice with probabilities that can move slightly (`EVAL_CARD.md` section 6.5).
Any change of rounding moves this model's answers, the batch, the padding or where a long row is split, so every bit-identity claim holds for a fixed configuration and request form.
Measurements: `EVAL_CARD.md` sections 4 and 6.5.

### Committed yes/no answers (`--noul-commit`)

For scorers that treat a yes/no probability strictly between 0.20 and 0.80 as no answer, as JevBench v1.5 does, such an answer is reported at the band's edge on its own side: 0.80 above 0.5, and 0.20 at or below it.
The answer never changes; its probability does, and calibration pays for it: log loss and ECE rise, while JevBench v1.5's yes/no competence rises.
Measurements: `runs/2026-10-03_noul-commit/`.

### The compact layout (`--prompt-tail compact`)

The compact layout has no blank lines around the options and ends with "Answer with the letter only.".
Tasks registered under one layout are not applied under the other, so a task registered under the compact layout needs `--prompt-tail compact` or registering again.
Packed mode reads the compact layout only, and the image route keeps it.
