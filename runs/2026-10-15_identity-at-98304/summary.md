# Does a context of 98,304 tokens change short answers? The stored FP8 31B on one RTX PRO 6000, 2026-10-15

The stored FP8 31B (`tachara-ai/decisio-gemma-4-31b`, vLLM 0.31.0) on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (600 W), the engine in the server's process, 200 single-question requests of the suite (public benchmark items; ids and probabilities, no item text) answered one at a time.
The 98,304-token server was started through `--engine` with `max_model_len` 98,304 and the default compilation configuration repeated (the route of the opt-in length, decisio's #179); the fixed starts at 32,768 ran decisio 0.12.0. `manifest.json` lists every file with its sha256.

| Pair | Exactly equal | Median difference | 90th percentile | Largest | Top answers changed | Files |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| two fresh starts at 32,768 (0.12.0), against each other | 200 of 200 | 0 | 0 | 0 | 0 | `clean_read/ident_read.json` |
| a start at 32,768 on the same code and `--engine` route as the 98,304 run, against those two | 200 of 200 | 0 | 0 | 0 | 0 | `control_32768_same_code/` |
| **98,304 against the two starts at 32,768** | **0 of 200** | 4.3e-3 | 5.2e-2 | 0.389 | **6** | `clean_read/` |

- The six top answers that moved: `boolq:124`, `scifact:85` (the largest difference, with a top-two margin of 0.707), `mmlu_pro_text:39`, `:54` (a tie), `:71` and `:122`. 150 of the 200 items differ by more than 1e-3 and 63 by more than 1e-2.
- The control separates the length from the rest: the same code, the same `--engine` route and the same compilation configuration at 32,768 gives exactly the answers of the 0.12.0 starts. What differs in the third row is the length.
- On the H100 PCIe 80 GB of the other session (`runs/2026-10-15_s14-session/C/`) the default length (98,304) was 200 of 200 exactly equal to a fixed 32,768 start, with every tensor's bytes identical. So whether the length moves short answers depends on the card.
- **Likely cause (a reading, not a test):** the length is part of vLLM's configuration and so of the compiled graph. On this card class a different compiled graph moved answers by the same amount in another comparison (the stored FP8 against Google's bf16 quantized on load, identical tensors, `runs/2026-10-15_s12-session/F/`: median 3.6e-3, 4 of 200 top answers). Nothing here isolates the compiled graph from the kernel it selects.
- An earlier reading on the same card (`first_read/`) could not separate the length: that session's two fresh starts at 32,768 were not equal to each other (median 5.0e-3, largest 0.405), and the 98,304 run was as far from each of them as they were from each other.

The context length served is part of the serving-stack fingerprint (`docs/tasks.md`), so a certificate made at one length is suspended at another.
