# Changelog

## [0.22.0](https://github.com/aminry/decisio/compare/v0.21.2...v0.22.0) (2026-10-20)


### Features

* **hub:** a tuned 12B repository - the adapter merged in fp32 and rounded once to bf16, stochastically ([#254](https://github.com/aminry/decisio/issues/254)) ([0d10978](https://github.com/aminry/decisio/commit/0d109789f62ae5a085b682afed46d6fb772ce2ba))

## [0.21.2](https://github.com/aminry/decisio/compare/v0.21.1...v0.21.2) (2026-10-20)


### Bug Fixes

* **ci:** scan the image for its SBOM with TMPDIR on the root volume, not the 8 GB /tmp tmpfs ([#258](https://github.com/aminry/decisio/issues/258)) ([2c30460](https://github.com/aminry/decisio/commit/2c30460bef49cc56ea3db9c79832798feed2cef8))

## [0.21.1](https://github.com/aminry/decisio/compare/v0.21.0...v0.21.1) (2026-10-20)


### Bug Fixes

* **ci:** free disk space before the image is scanned for its SBOM ([#252](https://github.com/aminry/decisio/issues/252)) ([e465c8a](https://github.com/aminry/decisio/commit/e465c8a846c799daaf0f84b31cf786e05848fbd1))

## [0.21.0](https://github.com/aminry/decisio/compare/v0.20.0...v0.21.0) (2026-10-20)


### Features

* **serve:** think when unsure - an unsure question is read again after the served model has written its reasoning (off by default; --think, a request field, or a tuned repository default) ([35a03a8](https://github.com/aminry/decisio/commit/35a03a8faedf36e70591e3b76be6ff4ea5464dfe))

## [0.20.0](https://github.com/aminry/decisio/compare/v0.19.0...v0.20.0) (2026-10-20)


### Features

* **hub:** decision_config /2 for a tuned model's repository (tuned and defaults sections, tuned-config and check commands) ([0f0c092](https://github.com/aminry/decisio/commit/0f0c092595cbea702e47fa9d46988d9bee29e983))
* **hub:** hub_fp8 merge - a LoRA adapter merged into the base in fp32 and quantized once to FP8 ([3ca0e5e](https://github.com/aminry/decisio/commit/3ca0e5e84a055168c960ce9c368236a946c969ea))
* **serve:** --over-limit truncate - answer a state over the prompt ceiling from its start (never certified, reported in the response) ([ad16bbd](https://github.com/aminry/decisio/commit/ad16bbd4b43fc79c32c7681c8a144a602e699d1c))
* **serve:** a tuned repository's defaults are served - temperatures, choice bands and the commit setting, with the flags above them ([62b3d13](https://github.com/aminry/decisio/commit/62b3d13d6ba10ea126dd3c8eef9a493cafd2273d))
* **serve:** a tuned repository's defaults are served ([#244](https://github.com/aminry/decisio/issues/244)) ([62b3d13](https://github.com/aminry/decisio/commit/62b3d13d6ba10ea126dd3c8eef9a493cafd2273d))

## [0.19.0](https://github.com/aminry/decisio/compare/v0.18.1...v0.19.0) (2026-10-19)


### ⚠ BREAKING CHANGES

* **serve:** the earlier spellings of the debug and route request headers are no longer read; send x-decisio-debug and x-decisio-route

### Features

* **serve:** the earlier spellings of the debug and route request headers are no longer read; send x-decisio-debug and x-decisio-route ([2b08ccd](https://github.com/aminry/decisio/commit/2b08ccdcbf917738f56ac4fd57555bc679d6e5ae))

## [0.18.1](https://github.com/aminry/decisio/compare/v0.18.0...v0.18.1) (2026-10-19)


### Bug Fixes

* **serve:** a re-certification derives min_examples from its new alpha and delta unless the original call asked for more than they need ([#233](https://github.com/aminry/decisio/issues/233)) ([cd3f3cd](https://github.com/aminry/decisio/commit/cd3f3cdb6203c99cc16d68af40ae9792ca8f0a66))

## [0.18.0](https://github.com/aminry/decisio/compare/v0.17.0...v0.18.0) (2026-10-19)


### Features

* **serve:** an adapter carries its own temperatures per question type, and its identity and temperatures are in the serving-stack fingerprint ([#207](https://github.com/aminry/decisio/issues/207)) ([72e8aff](https://github.com/aminry/decisio/commit/72e8aff27154c57338ab69e432ed6420a567e6dc))
* **serve:** reference docs generated from the code - an OpenAPI file of the served routes and models and a table of the server's flags, written at release time ([#224](https://github.com/aminry/decisio/issues/224)) ([35d4287](https://github.com/aminry/decisio/commit/35d42879babdf398a9ea2fbc538d93f6004d9115))
* **serve:** the image count of a request is limited by the context, not by a cap of 2 (--max-images, default 128) ([#222](https://github.com/aminry/decisio/issues/222)) ([24568fc](https://github.com/aminry/decisio/commit/24568fc8e4ba428a8a3cde563b55d5a862190771))


### Bug Fixes

* **serve:** --image-model's text and image engines default to shares that fit one card (0.47 and 0.49), and a pair that cannot fit is refused before either engine is built ([#227](https://github.com/aminry/decisio/issues/227)) ([4693205](https://github.com/aminry/decisio/commit/4693205bd6d0bfba2724c288989ff5c4e1396233))
* **serve:** every thread that calls the in-process engine runs on the stream the engine was warmed up on (Marlin W4A16 checkpoints) ([#215](https://github.com/aminry/decisio/issues/215)) ([8454d7d](https://github.com/aminry/decisio/commit/8454d7dc47ff0cd72fb5fe8769eeadfd393f84c6))

## [0.17.0](https://github.com/aminry/decisio/compare/v0.16.0...v0.17.0) (2026-10-18)


### Features

* **serve:** System One's assets - placeholders in the state and the questions are bound to images, in JSON and multipart ([#221](https://github.com/aminry/decisio/issues/221)) ([b6bf2fe](https://github.com/aminry/decisio/commit/b6bf2fe8526c57291a5af2e5631b3a352af5f65e))


### Bug Fixes

* **serve:** a choice with more than 255 options is a 422 like System One's, and conformance gate C5 compares the wire with System One's OpenAPI ([#219](https://github.com/aminry/decisio/issues/219)) ([d95c891](https://github.com/aminry/decisio/commit/d95c8915d3177c8198e8d1bbab0001a317fab3cf))

## [0.16.0](https://github.com/aminry/decisio/compare/v0.15.0...v0.16.0) (2026-10-18)


### Features

* **serve:** a certified task's questions are scored by the function that certified them, and registration records how far the served confidences are ([#211](https://github.com/aminry/decisio/issues/211)) ([a494297](https://github.com/aminry/decisio/commit/a4942971ae605fd55609b19e591b387bef13b24e))
* **serve:** a confidence signal in the traffic monitor, certification that stops while it alerts, and a warning for a task with no catch-all option ([#214](https://github.com/aminry/decisio/issues/214)) ([f7ba1ed](https://github.com/aminry/decisio/commit/f7ba1edd950964a80c85d3afc746b1182f20689c))


### Bug Fixes

* **serve:** refuse a checkpoint of a model type no base serves instead of giving it the Qwen profile ([#213](https://github.com/aminry/decisio/issues/213)) ([a1bc43c](https://github.com/aminry/decisio/commit/a1bc43cdc8c21deb6f63c6d1bc505cee52b125bc))
* **serve:** the MLX backend downloads --model at --revision, and the revision is in the fingerprint when given ([#210](https://github.com/aminry/decisio/issues/210)) ([3fdadd8](https://github.com/aminry/decisio/commit/3fdadd808b991bd9238bb5c761d59cb7405d78b2))

## [0.15.0](https://github.com/aminry/decisio/compare/v0.14.0...v0.15.0) (2026-10-17)


### Features

* **serve:** an adapter in the other tensor-name layout is mapped, not refused, and the check runs before the engine is built ([#198](https://github.com/aminry/decisio/issues/198)) ([530971b](https://github.com/aminry/decisio/commit/530971b41123dc376a3997e535129e25eea171ce))


### Bug Fixes

* **serve:** pin the 0.15.0 Hub revisions ([#204](https://github.com/aminry/decisio/issues/204)) ([02f6229](https://github.com/aminry/decisio/commit/02f6229c34ceb8979e2fb8d0942d2425eea9d6f7))

## [0.14.0](https://github.com/aminry/decisio/compare/v0.13.0...v0.14.0) (2026-10-16)


### ⚠ BREAKING CHANGES

* **serve:** without --max-model-len the Gemma bases serve the largest context that fits the card, up to the base's ceiling ([#196](https://github.com/aminry/decisio/issues/196))

### Features

* **serve:** without --max-model-len the Gemma bases serve the largest context that fits the card, up to the base's ceiling ([#196](https://github.com/aminry/decisio/issues/196)) ([39b5852](https://github.com/aminry/decisio/commit/39b5852b65140f31fd488ad550c57259cc9b2990))


### Bug Fixes

* **serve:** /v1/systemone passes a request's adapter to the engine ([#200](https://github.com/aminry/decisio/issues/200)) ([cf90db5](https://github.com/aminry/decisio/commit/cf90db5ff9839b20aa17c4f89909c2d8af982c1b))
* **serve:** pin the 0.14.0 Hub revisions ([#197](https://github.com/aminry/decisio/issues/197)) ([30f43d3](https://github.com/aminry/decisio/commit/30f43d3a0fc2fdf252d96f4505efc32d262ff709))

## [0.13.0](https://github.com/aminry/decisio/compare/v0.12.0...v0.13.0) (2026-10-15)


### Features

* **serve:** a longer context as an opt-in: --max-model-len up to a ceiling per base (12B 65,536, 31B 98,304), and --kv-cache-dtype fp8 ([#179](https://github.com/aminry/decisio/issues/179)) ([f10a364](https://github.com/aminry/decisio/commit/f10a36483b8b027ea4f0776bf3d9f268a1324405))


### Bug Fixes

* **serve:** pin the 0.13.0 Hub revisions; the 31B's weights.method names what was measured ([#191](https://github.com/aminry/decisio/issues/191)) ([26e4b7e](https://github.com/aminry/decisio/commit/26e4b7ee40f37bede8d91c7dcde56df07acc4314))

## [0.12.0](https://github.com/aminry/decisio/compare/v0.11.1...v0.12.0) (2026-10-15)


### Features

* **reference:** a plain-transformers reference for the model repositories ([#123](https://github.com/aminry/decisio/issues/123)) ([00bc619](https://github.com/aminry/decisio/commit/00bc61991e35384ea9894b8dfccd35f64a07d364))
* **serve:** an abstention task can match by option keys, for items that share their labels and not their texts ([#176](https://github.com/aminry/decisio/issues/176)) ([8bd9d20](https://github.com/aminry/decisio/commit/8bd9d20d872b1172a91be00ae4df7530f7681109))
* **serve:** certified answers and a monitor in /v1/tasks ([#155](https://github.com/aminry/decisio/issues/155)) ([a6dd227](https://github.com/aminry/decisio/commit/a6dd227c4292529855da1b3eb688bd96433cef10))
* **serve:** registration names what it read: inputs_sha256 on the calibration and head records ([#177](https://github.com/aminry/decisio/issues/177)) ([14de77b](https://github.com/aminry/decisio/commit/14de77be4b372498a4866527140299af6f525e53))
* **serve:** the intent head carries a temperature, fitted on its out-of-fold logits ([#171](https://github.com/aminry/decisio/issues/171)) ([0ee2dcc](https://github.com/aminry/decisio/commit/0ee2dcc7c28833860872aad772ac2059fc7d4cc3))
* **serve:** vLLM 0.31.0 ([#137](https://github.com/aminry/decisio/issues/137)) ([3f615a3](https://github.com/aminry/decisio/commit/3f615a30c552cdcdc63e7f4ca78175a36e4e8f7d))
* **serve:** vLLM 0.31.0 ([#144](https://github.com/aminry/decisio/issues/144)) ([8375011](https://github.com/aminry/decisio/commit/83750116c1bd66f29955d7f73184fa8457f44199))


### Bug Fixes

* **serve:** /health state_boundary reports running, so a registration being sent is not read as idle ([#147](https://github.com/aminry/decisio/issues/147)) ([4d3fa3b](https://github.com/aminry/decisio/commit/4d3fa3bd3d370600e61224bd962678c452554af1))
* **serve:** a NaN or infinity in an engine's probabilities is a clear 500 that names the model and engine, not a 400 ([#175](https://github.com/aminry/decisio/issues/175)) ([e2a9f09](https://github.com/aminry/decisio/commit/e2a9f09fb13230c99089f766284c341148777ee0))
* **serve:** engine_kwargs reads the base from the checkpoint when the caller gives no family ([#150](https://github.com/aminry/decisio/issues/150)) ([e1404ae](https://github.com/aminry/decisio/commit/e1404aea5573c5311df01c0cd571ba799e2c533e))
* **serve:** refuse a POST that is not JSON with 415, before any work, on every route ([#157](https://github.com/aminry/decisio/issues/157)) ([aabd8fd](https://github.com/aminry/decisio/commit/aabd8fdd1fbd47a82372c79c99990d94b022ac3d))
* the MLX backend sets its own memory and cache limits ([#158](https://github.com/aminry/decisio/issues/158)) ([d21a31c](https://github.com/aminry/decisio/commit/d21a31c989622c8271d98b30c11665688b822386))


### Reverts

* "feat(serve): vLLM 0.31.0 ([#137](https://github.com/aminry/decisio/issues/137))" ([#143](https://github.com/aminry/decisio/issues/143)) ([82f3853](https://github.com/aminry/decisio/commit/82f3853c6b7f859e7427df1c790876402fb29c16))

## [0.11.1](https://github.com/aminry/decisio/compare/v0.11.0...v0.11.1) (2026-10-10)


### Bug Fixes

* **serve:** a request records how long it waited for the engine, and whether a registration was running ([#135](https://github.com/aminry/decisio/issues/135)) ([76d484c](https://github.com/aminry/decisio/commit/76d484cfec1213065efdfdcd8fdcad171cc712d3))
* **serve:** gemma-4-12b registers a state's boundary before the question, as the 31B does ([#128](https://github.com/aminry/decisio/issues/128)) ([b7af9c6](https://github.com/aminry/decisio/commit/b7af9c684d5caead51d7efe093c4a0df0c5d57f4))

## [0.11.0](https://github.com/aminry/decisio/compare/v0.10.0...v0.11.0) (2026-10-09)


### Features

* **hub:** the model repositories are under tachara-ai (Tachara AI Lab), pinned at their 0.11.0 revisions ([#121](https://github.com/aminry/decisio/issues/121)) ([852cce4](https://github.com/aminry/decisio/commit/852cce401a78aeb547e4016c6129adb86c2c8792))
* **serve:** --backend mlx serves gemma-4-31b from an MLX conversion, documented for states up to 16,383 tokens ([#116](https://github.com/aminry/decisio/issues/116)) ([a801341](https://github.com/aminry/decisio/commit/a801341211649def233529359d21d78c43fbe25f))


### Bug Fixes

* **serve:** --backend mlx refuses any checkpoint whose config.json declares FP8 quantization ([#120](https://github.com/aminry/decisio/issues/120)) ([18229d2](https://github.com/aminry/decisio/commit/18229d25dddf8c19aca6c5c3263bd91fd21947a5))
* **serve:** a prompt over the context is a 422 with a clear detail, not a bare 500 ([#124](https://github.com/aminry/decisio/issues/124)) ([a51be6d](https://github.com/aminry/decisio/commit/a51be6d9e96d84f8cd4dd31b8e3b7fb3a5c01046))

## [0.10.0](https://github.com/aminry/decisio/compare/v0.9.0...v0.10.0) (2026-10-08)


### ⚠ BREAKING CHANGES

* **serve:** a vLLM server started with neither `--base` nor `--model`, and the container, now serve gemma-4-31b (a 96 GB card); set `--base qwen3.6-35b-a3b` or `DECISIO_BASE=qwen3.6-35b-a3b` to keep the previous Qwen behaviour (docs/upgrading.md).

### Features

* **hub:** the 31B's FP8 checkpoint is made on a CPU as vLLM quantizes it on load, and the engine's FP8 weights can be fingerprinted ([#113](https://github.com/aminry/decisio/issues/113)) ([94d675c](https://github.com/aminry/decisio/commit/94d675c779ad14da9770926a2fdec21559e1ba79))
* **serve:** --base takes a decisio model repository; decision_config.json is exported from each base's profile ([#110](https://github.com/aminry/decisio/issues/110)) ([9d352de](https://github.com/aminry/decisio/commit/9d352def278e79e99eacf4eb6fd1c51ddeb3cfba))
* **serve:** the default base is Gemma 4 31B, served from its FP8 repository ([#114](https://github.com/aminry/decisio/issues/114)) ([389fe13](https://github.com/aminry/decisio/commit/389fe13787ca0831a50b43cdf9e24396c44fab22))
* **serve:** the MLX prefix cache budget is the base's: 7,400 MiB for gemma-4-12b, 2,048 MiB for the Qwen base ([#112](https://github.com/aminry/decisio/issues/112)) ([71facc7](https://github.com/aminry/decisio/commit/71facc77a0e645b8023fd0a61554ea9175bbb741))

## [0.9.0](https://github.com/aminry/decisio/compare/v0.8.2...v0.9.0) (2026-10-07)


### Features

* **serve:** register a state's boundary after the response on the Gemma bases, so a first question does not wait for it ([#99](https://github.com/aminry/decisio/issues/99)) ([787df0e](https://github.com/aminry/decisio/commit/787df0e6d097e4ea3106f3e2b459661f4f8d81aa))

## [0.8.2](https://github.com/aminry/decisio/compare/v0.8.1...v0.8.2) (2026-10-07)


### Bug Fixes

* **serve:** --backend mlx refuses gemma-4-31b by the resolved base, so a 31B conversion without --base is refused too ([#100](https://github.com/aminry/decisio/issues/100)) ([f6cc1e8](https://github.com/aminry/decisio/commit/f6cc1e86d221ebc175c719de6d344084ceb9d5db))

## [0.8.1](https://github.com/aminry/decisio/compare/v0.8.0...v0.8.1) (2026-10-05)


### Bug Fixes

* **serve:** a single question registers its state's boundary on the Gemma bases, so the next question reads the state from the cache ([#93](https://github.com/aminry/decisio/issues/93)) ([573d7d6](https://github.com/aminry/decisio/commit/573d7d698c830890f8624b4b69261c31dda278c2))
* **serve:** close the engine-death races; loop those tests under load nightly with the slow tier on a hosted runner ([#91](https://github.com/aminry/decisio/issues/91)) ([3c51e79](https://github.com/aminry/decisio/commit/3c51e795edd650760ba5c49ee65e55b561dec4c7))

## [0.8.0](https://github.com/aminry/decisio/compare/v0.7.2...v0.8.0) (2026-10-05)


### Features

* **serve:** the engine runs in the server's process by default; gemma-4-31b scores several questions warm ([#86](https://github.com/aminry/decisio/issues/86)) ([efd2b83](https://github.com/aminry/decisio/commit/efd2b83943c5dffaf9b692e138f4bffaaa79d1c1))

## [0.7.2](https://github.com/aminry/decisio/compare/v0.7.1...v0.7.2) (2026-10-05)


### Bug Fixes

* **serve:** a dead engine turns /health to 503, is refused fast and ends the server with code 70 ([#78](https://github.com/aminry/decisio/issues/78)) ([8b367d9](https://github.com/aminry/decisio/commit/8b367d92c39244385069571f59f1d2dbb1f51cf9))

## [0.7.1](https://github.com/aminry/decisio/compare/v0.7.0...v0.7.1) (2026-10-05)


### Bug Fixes

* **serve:** --head-engine gives the second engine the server's prompt format and base ([#76](https://github.com/aminry/decisio/issues/76)) ([993db22](https://github.com/aminry/decisio/commit/993db2200153c1db91854f525b96182a0577fbbb))

## [0.7.0](https://github.com/aminry/decisio/compare/v0.6.0...v0.7.0) (2026-10-05)


### Features

* **serve:** --engine-process in|separate, where vLLM's engine runs (default separate) ([#74](https://github.com/aminry/decisio/issues/74)) ([bbff50f](https://github.com/aminry/decisio/commit/bbff50f7c686b7c55ea00817705c5229699b3d9f))

## [0.6.0](https://github.com/aminry/decisio/compare/v0.5.0...v0.6.0) (2026-10-05)


### Features

* **serve:** Gemma 4 31B as a third base (--base gemma-4-31b), FP8 on load ([#71](https://github.com/aminry/decisio/issues/71)) ([73f0eff](https://github.com/aminry/decisio/commit/73f0eff184d44529a47726a4fc2e98045dca849f))

## [0.5.0](https://github.com/aminry/decisio/compare/v0.4.1...v0.5.0) (2026-10-04)


### Features

* the Gemma base on a Mac with --backend mlx, 6-bit by default ([#69](https://github.com/aminry/decisio/issues/69)) ([587c8b2](https://github.com/aminry/decisio/commit/587c8b26a9eb71ddc0080686a512f4bd73738ed8))

## [0.4.1](https://github.com/aminry/decisio/compare/v0.4.0...v0.4.1) (2026-10-04)


### Bug Fixes

* **serve:** load the pinned revision in packed mode, the hf stand-in and the second-engine head ([#65](https://github.com/aminry/decisio/issues/65)) ([b0b4ea5](https://github.com/aminry/decisio/commit/b0b4ea58dd6cbf5275bdcc627ec0c3cb18d7a93e))
* **serve:** pin the Qwen base's checkpoint revision at 95a723d ([#63](https://github.com/aminry/decisio/issues/63)) ([9a36b96](https://github.com/aminry/decisio/commit/9a36b9681d1a1b7b3d83e90286ef9ffcb90fcc9b))

## [0.4.0](https://github.com/aminry/decisio/compare/v0.3.0...v0.4.0) (2026-10-04)


### Features

* **serve:** Gemma 4 12B as a second base (--base gemma-4-12b) ([#48](https://github.com/aminry/decisio/issues/48)) ([61d7519](https://github.com/aminry/decisio/commit/61d7519bfed0910747970c2ff862ca10074a454d))


### Bug Fixes

* **conformance:** C4 applies each question type's served temperature ([#60](https://github.com/aminry/decisio/issues/60)) ([4ab98af](https://github.com/aminry/decisio/commit/4ab98afe0de5ce6ef6203d47bf45d913d0a1abd4))

## [0.3.0](https://github.com/aminry/decisio/compare/v0.2.0...v0.3.0) (2026-10-04)


### ⚠ BREAKING CHANGES

* **serve:** tasks registered under the earlier compact layout are not applied by the new default (the layout enters the task fingerprint and the yes/no rendering a yes/no question's task key); register them again, or serve them with --prompt-tail compact --noul-rendering words. --mode packed reads only the compact layout and needs --prompt-tail compact.

### Features

* **demos:** four demos on a System One server, judged decision by decision, with clips rendered from trajectories ([#33](https://github.com/aminry/decisio/issues/33)) ([86abd52](https://github.com/aminry/decisio/commit/86abd5243fa595a89426e11aa63a60d47cbbc955))
* **serve:** --noul-commit, an opt-in transform reporting yes/no answers inside 0.20 to 0.80 at the band's edge ([#53](https://github.com/aminry/decisio/issues/53)) ([62dfe1f](https://github.com/aminry/decisio/commit/62dfe1fa9af8bb490a53c1a1cccee345adbc2020))
* **serve:** the spaced layout and yes/no as named letters are the served default, with T 1.506 and choice T 1.370 ([6c581d4](https://github.com/aminry/decisio/commit/6c581d46e5062d902763b6aa9adee0600cc022e1))

## [0.2.0](https://github.com/aminry/decisio/compare/v0.1.1...v0.2.0) (2026-10-03)


### ⚠ BREAKING CHANGES

* **serve:** no padding (--pad-policy none) is the MLX backend's default ([#46](https://github.com/aminry/decisio/issues/46))
* --multi-question sequential and --describe-options are the served defaults ([#38](https://github.com/aminry/decisio/issues/38))

### Features

* --describe-options shows an option's description alone when it has one (off by default) ([#36](https://github.com/aminry/decisio/issues/36)) ([f2d936e](https://github.com/aminry/decisio/commit/f2d936e5cbcdcd8f1880f1569121478e37b0c3ba))
* --multi-question sequential and --describe-options are the served defaults ([#38](https://github.com/aminry/decisio/issues/38)) ([06a1273](https://github.com/aminry/decisio/commit/06a1273fcb1668c856fd182008db26b81f90c1d4))
* --multi-question sequential makes each answer equal the question asked alone; stage timings and --pad-policy; README wording for multi-question requests ([#35](https://github.com/aminry/decisio/issues/35)) ([5461e0e](https://github.com/aminry/decisio/commit/5461e0e90e039a24c0426555d8801e003d81bc9c))
* **serve:** --noul-rendering words|letters|letters-keys (words by default) ([#44](https://github.com/aminry/decisio/issues/44)) ([597f383](https://github.com/aminry/decisio/commit/597f3838cdf0e4a7f89b87f2fd5a007f4e850ffe))
* **serve:** --noul-rendering words|letters|letters-keys, how a yes/no question is asked (words by default) ([597f383](https://github.com/aminry/decisio/commit/597f3838cdf0e4a7f89b87f2fd5a007f4e850ffe))
* **serve:** --pad-policy row, a single-question row padded to end on the block boundary (off by default) ([#45](https://github.com/aminry/decisio/issues/45)) ([8ccf143](https://github.com/aminry/decisio/commit/8ccf14367f13ff9ceb09726b2959072351873d03))
* **serve:** a cross-request prefix cache for the MLX engine (--prefix-cache-mb) ([#47](https://github.com/aminry/decisio/issues/47)) ([8e3f54f](https://github.com/aminry/decisio/commit/8e3f54f645faaf16329f7be2b317f1071cc9bf89))
* **serve:** an MLX engine for the text route on Apple silicon (--backend mlx) ([#39](https://github.com/aminry/decisio/issues/39)) ([f3c5026](https://github.com/aminry/decisio/commit/f3c5026dc03da3af71958d187a15e997a30d1d27))
* **serve:** no padding (--pad-policy none) is the MLX backend's default ([#46](https://github.com/aminry/decisio/issues/46)) ([2c1204e](https://github.com/aminry/decisio/commit/2c1204e4a6c0db22f1c95f20d0b5cfe1575b7a5b))


### Bug Fixes

* task registration on a certain readout, and the CPU stand-in's head engine (two HTTP 500s) ([#49](https://github.com/aminry/decisio/issues/49)) ([6e71298](https://github.com/aminry/decisio/commit/6e7129842ef9460ed35897d598e9b86affde6f2a))

## [0.1.1](https://github.com/aminry/decisio/compare/v0.1.0...v0.1.1) (2026-10-01)


### Bug Fixes

* **ci:** attest a CycloneDX SBOM of the image instead of SPDX ([#22](https://github.com/aminry/decisio/issues/22)) ([eef30b4](https://github.com/aminry/decisio/commit/eef30b4a8a99d82a3b1b14303ff1f9586a2b7a7e))

## [0.1.0](https://github.com/aminry/decisio/compare/v0.1.0...v0.1.0) (2026-10-01)


### Features

* a generator for the conformance items file from two public sets ([#18](https://github.com/aminry/decisio/issues/18)) ([77f1bbd](https://github.com/aminry/decisio/commit/77f1bbde3c23cf81d77bd38229df0a028a5efe71))


### Bug Fixes

* the head's output-layer rows load from a Hugging Face repo id as well as a local directory ([9d7d0ab](https://github.com/aminry/decisio/commit/9d7d0ab8133e682a336de8600a7588173c9a94dd))


### Documentation

* community files, CI and release tooling ([#2](https://github.com/aminry/decisio/issues/2)) ([4dc5388](https://github.com/aminry/decisio/commit/4dc5388eb5743f1ddc19845723d22ca0499e8fdb))
* launch checklist for the public flip ([#4](https://github.com/aminry/decisio/issues/4)) ([c79ec96](https://github.com/aminry/decisio/commit/c79ec9675bc2415a0ae5f045c2051a55022b2849))
* launch checklist gate for the image's first GPU start; README says it is pending ([dff92fb](https://github.com/aminry/decisio/commit/dff92fbe407336858ea26216a81c6ba0587aab96))
* launch checklist records the flip and corrects three steps ([#14](https://github.com/aminry/decisio/issues/14)) ([93a885d](https://github.com/aminry/decisio/commit/93a885dc82441aa77df39df304e0421d2e7ad7ac))
* launch checklist section for the maintainer's commit signing ([80e36ca](https://github.com/aminry/decisio/commit/80e36caa08c013d0a2fa11618baedc30695d5e35))
* README section Run with Docker; launch checklist covers the image ([2e93c65](https://github.com/aminry/decisio/commit/2e93c65f57cb4fc2f77b714666b593763b848616))
* record that the merge settings of checklist 2.1 are applied ([7b60810](https://github.com/aminry/decisio/commit/7b608109eed12618de516bfef6f56928c00fc4b5))
* record the history rebuild of main in the launch checklist ([b5688cc](https://github.com/aminry/decisio/commit/b5688cc3da89bc181855d1da2b6b10e5ece14ab8))
