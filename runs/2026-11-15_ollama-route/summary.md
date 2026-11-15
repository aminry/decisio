# The Gemma listings on Ollama's decision route (2026-11-15)

Ollama 0.40.4 and later answer `/v1/systemone` for a GGUF that carries `<arch>.decision.type` and a chat template named `systemone` (llama.cpp's own System One server).
`python -m decisio.hub_gguf` writes those keys with decisio's prompt as the template and decisio's temperatures, and the files of `aminroudaki/decisio-gemma` (the Gemma 4 12B base) and `aminroudaki/decisio-gemma-tuned` (the tuned 12B, calibrated flavor) were made with it (`release_files.json`).
Each tag also carries Gemma 4's vision projector.

## Versions

The base and the tuned model's `q4_k_m`, with the projector, were loaded on Ollama 0.40.4, 0.40.9 and 0.40.16, each from its release archive with its own model store, and answered 20 text and 10 image questions (`ollama_versions.json`).
Every version answered every question with 0.40.16's top answer; the probabilities differ from 0.40.16's by at most 0.0007 on text and, on images, 0.019 (base) and 0.048 (tuned).
An image is read near its own resolution: 83 to 487 tokens on these images, the same on all three versions.

## Answers against decisio's own servers (`comparison.json`)

The same questions went to the Ollama tags (0.40.16) and to decisio:
- text: 300 questions of a private decision suite with at most 26 options, 30 of 77 options, and 40 requests of three questions, against decisio's MLX server on the same Mac (the base's 4-bit and 8-bit conversions, the tuned model's 6-bit and 8-bit);
- images: 257 questions about synthetic scenes and real photos, one image each, against decisio's answers from the bf16 models on vLLM (the base at an image budget of 560, the tuned model at 280), scored with the same scorer.

| Pair | Text: same answer | Text: median abs. difference | Images: same answer |
| --- | ---: | ---: | ---: |
| base `q8_0` vs decisio MLX 8-bit / bf16 | 98.3% | 0.0019 | 97.7% |
| base `q4_k_m` vs decisio MLX 4-bit / bf16 | 87.0% | 0.0160 | 97.3% |
| decisio MLX 4-bit vs MLX 8-bit (base) | 86.3% | 0.0158 | |
| tuned `q8_0` vs decisio MLX 8-bit / bf16 | 99.0% | 0.0025 | 97.3% |
| tuned `q4_k_m` vs decisio MLX 8-bit / bf16 | 94.7% | 0.0091 | 97.3% |

- Questions of 77 options: the same answer as decisio for 30 of 30 at `q8_0` (base and tuned).
- Requests of three questions (each question's prompt lists only its own question): 87.5% (base `q8_0`) and 95.0% (tuned `q8_0`).
- No accuracy difference between a tag and decisio is outside its 95% interval (paired bootstrap, 10,000 draws).
  Log loss differs in six places, all small (`comparison.json`, `paired_a_minus_b`).

## Memory (`memory.json`)

Each tag was loaded alone on Ollama 0.40.16 at its 8,192-token context and answered a text, an image (1,600 by 1,200) and a 6,600-token text, while the resident set of the server and its runner was sampled every 0.2 s.

| Tag | Resident, loaded | Resident, peak |
| --- | ---: | ---: |
| base `q4_k_m` | 11.12 GB | 11.25 GB |
| base `q8_0` | 15.93 GB | 16.06 GB |
| tuned `q4_k_m` | 10.83 GB | 10.96 GB |
| tuned `q8_0` | 15.93 GB | 16.06 GB |

The resident set counts the weights, which llama.cpp maps from the file, and about 3.5 GB of buffers.
Ollama's `/api/ps` reports 4.24 GB for every tag, and the process's physical footprint 3.5 GB: neither counts the mapped weights.

## Files

`comparison.json`, `ollama_versions.json`, `memory.json`, `release_files.json` (the sha256 of every file the listings carry), `host.txt`.
