<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Third-party material

decisio is Apache-2.0.
It bundles no third-party code except two prompt texts from the Cygnet recipe (MIT, below) and the three demo clients under `examples/demos/` (their rows are below, each with its own licence file and `ATTRIBUTION.md`); it runs on, patches, or references the rest.

| What | How decisio uses it | Licence | Source |
| --- | --- | --- | --- |
| vLLM 0.30.0 | The serving engine (the `serve` extra); `patches/vllm-0.30.0` modifies it at image build; the plugin subclasses its `Qwen3_5MoeForCausalLM` | Apache-2.0 | github.com/vllm-project/vllm |
| Qwen3.6-35B-A3B (FP8) | The Qwen base's weights, redistributed byte for byte in `aminry/decisio-qwen3.6-35b-a3b` with the source licence | Apache-2.0 (model card) | huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8 |
| Gemma 4 12B (google/gemma-4-12B-it at revision 707f0a3b) | The second base's weights (`--base gemma-4-12b`), redistributed byte for byte in `aminry/decisio-gemma-4-12b` with `LICENSE` and `NOTICE`; the plugin subclasses vLLM's `Gemma4UnifiedForConditionalGeneration` | Apache-2.0 (model card and the Gemma 4 licence page, ai.google.dev/gemma/docs/gemma_4_license, which names no use policy) | huggingface.co/google/gemma-4-12B-it |
| Gemma 4 31B (google/gemma-4-31B-it at revision 842da379) | The default base's weights (`--base gemma-4-31b`), with the text model's linear layers quantized to FP8 and redistributed in `aminry/decisio-gemma-4-31b` with `LICENSE` and `NOTICE`; the vision tower and every other tensor are unchanged; the plugin subclasses vLLM's `Gemma4ForCausalLM`, its text class | Apache-2.0 (model card and the Gemma 4 licence page, ai.google.dev/gemma/docs/gemma_4_license, which names no use policy) | huggingface.co/google/gemma-4-31B-it |
| Cygnet recipe (`shim/cygnet_shim.py` at 3cf591c) | Its trailing instruction, verbatim, the spaced layout's closing line in `src/decisio/readout/spaced.py` (`--prompt-tail spaced`); its system prompt, verbatim, the Gemma base's system turn in `src/decisio/readout/system_prompt.py`; the prompt around them is decisio's own code | MIT (Copyright (c) 2026 Nood Co and contributors) | github.com/blockbrain-ai/cygnet-recipe |
| Qwen3-0.6B-Base, Qwen3.5-0.8B-Base | Small checkpoints the CPU tests download and run as stand-ins | Apache-2.0 (model cards) | huggingface.co/Qwen |
| TypeSafe System One wire format, typesafe-sdk 0.7.0 | The request and response schema `/v1/systemone` implements; the SDK client in the conformance check (`bench` extra) | MIT | pypi.org/project/typesafe-sdk |
| JevBench | The benchmark harness the benchmark stage drives (fetched at a pinned commit); its v1.5 open-set scoring method is implemented in `decisio.bench.jevbench_v15` | MIT | github.com/fstandhartinger/jevbench |
| Decision Index kit | The benchmark kit the benchmark stage drives (fetched at a pinned commit); `decisio.bench.di_report` calls its scoring functions | MIT | github.com/apolinario/decision-index |
| Reflex | The two-order averaging and the branch-disagreement statistic (`--orders 2`) follow its method | MIT | github.com/kshetrajna12/reflex |
| imajev | The image request extension (`images`, `unknown_probability`, `abstained`) and its image limits | Apache-2.0 | github.com/mohit67890/imajev |
| BANKING77 intent names | The option labels of the latency grid's 77-option cells (`src/decisio/bench/banking77_labels.json`) | CC-BY-4.0 | PolyAI, github.com/PolyAI-LDN/task-specific-datasets |
| BANKING77, CLINC150 hidden-state readouts | Test fixtures (`tests/data/intent_readouts`): the model's hidden states and label log-probabilities on examples of the two datasets, no text | CC-BY-4.0 (BANKING77), CC-BY-3.0 (CLINC150) | PolyAI; github.com/clinc/oos-eval |
| Fitted intent heads and calibration priors | Derived parameters kept as records (`runs/2026-09-30_plugin-verification/intent_heads/*_task.json.gz`, `latency_tasks.json.gz`; `runs/2026-09-29_tasks-endpoint/intent_heads/*_task.json.gz`, `latency_tasks.json.gz`, `di_registered_tasks.json.gz`): linear heads and per-option biases fitted on the served model's readouts of 10 labelled training examples per intent; no example text | CC-BY-4.0 (BANKING77), CC-BY-3.0 (CLINC150), as the data they were fitted on | PolyAI; github.com/clinc/oos-eval |
| ImajevBench v2.0-lite | The image route's benchmark in `runs/2026-09-27_image-input`; the records hold item ids, our distributions and scores, not the items | CC BY 4.0 (items), CC0 (images) | github.com/mohit67890/imajev-bench |
| MMLU-Pro, GPQA Diamond, BANKING77, CLINC150+OOS | The Decision Index benchmarks in `runs/`; the records hold item ids, our answers and scores, never item text | MIT, CC-BY-4.0, CC-BY-4.0, CC-BY-3.0 | TIGER-Lab/MMLU-Pro; idavidrein/gpqa; as above |
| jev_fsd | The driving demo, vendored and changed to call a System One server (`examples/demos/fsd`, commit bbc9012) | MIT | github.com/BrendanH18/jev_fsd |
| jev-ultrafast | The browser-agent demo, vendored and changed to call a System One server (`examples/demos/ultrafast`, commit 1231850) | MIT | github.com/browser-use/jev-ultrafast |
| jev-pong | The Pong lanes demo, vendored in reduced form and changed to call System One and chat servers (`examples/demos/pong`, commit d28d6ae) | Apache-2.0 | github.com/ably-labs/jev-pong |
| Three.js | Bundled by the driving demo (`examples/demos/fsd/static/vendor/three`) | MIT | threejs.org |
| OpenStreetMap map packs | The driving demo's streets (`examples/demos/fsd/data/maps`), derived from OpenStreetMap | ODbL 1.0 | openstreetmap.org/copyright |
