# The Qwen base's 4-bit MLX conversion re-gated under the current served default (2026-10-06)

The served default changed in #54, and only the 6-bit conversion had been gated again (`runs/2026-10-03_mlx-regate`).
This record gates the 4-bit conversion, `mlx-community/Qwen3.6-35B-A3B-4bit` at `38740b8`, with the same gates against the same FP8 record (`runs/2026-10-03_qiv-default`), on decisio `83e461d`, an Apple M5 Pro (64 GB).
Pre-registered before any measurement (the RLCD repository, `experiments/2026-10-06_ls_qwen_4bit_regate/PREREG.md`).

## Gates

| Gate | MLX 4-bit | FP8 | Verdict |
| --- | --- | --- | --- |
| S1: suite accuracy (1,400 items) | 0.7643 | 0.7700 | pass: -0.6 [-1.9, +0.8] points |
| S2: top-answer flips | 135 of 1,400 | | counted |
| S3: pooled ECE (allowed: within 0.01) | 0.0297 | 0.0327 | pass |
| JevBench 231, ECE at the served temperatures (reported) | 0.0436 | 0.0422 | +0.0015; accuracy 0.866 vs 0.870 |
| H: BANKING77 intent head, six draws (of 150) | 124 / 125 / 124 / 125 / 127 / 126, mean 0.834 | 126 / 126 / 122 / 127 / 126 / 129, mean 0.840 | pass: -0.6 [-2.9, +1.9] points; draw 0 repeated bit for bit |
| H: CLINC150 intent head, six draws (of 100) | 92 / 92 / 93 / 93 / 94 / 93, mean 0.928 | 88 / 92 / 91 / 91 / 91 / 94, mean 0.912 | pass: +1.7 [-0.7, +4.2] points |
| C2 / C3 / C4 conformance | pass / pass (0 of 1,400 differ) / pass (largest probability difference 0.0) | | pass |

## Memory (in process, `4bit/memory.json`)

| State tokens | Peak, 1 question | Peak, 10 questions |
| ---: | ---: | ---: |
| 8,473 | 21.58 GB | 20.05 GB |
| 16,921 | 21.98 GB | 20.65 GB |
| 32,761 | 22.70 GB | 21.73 GB |

The weights take 19.5 GB; no cell grew swap.
GB are 1e9 bytes: the memory driver divides MLX's bytes by 1e9.
Two thirds of a 32 GB Mac's memory (32 GiB, 34.36e9 bytes) is 22.9 GB, a conservative stand-in for its default GPU memory limit, which was not measured on a 32 GB Mac.
Every peak is under that line: with one question by 1.33 GB at 8k, 0.93 GB at 16k and 0.21 GB (0.9% of the line) at 32k tokens, and with ten questions by 2.85, 2.26 and 1.17 GB.
At 32k a 32 GB Mac would have about 0.2 GB left under that line for the system and other apps, so the conversion is documented for Macs with more than 32 GB, or 32 GB with the GPU memory limit raised.

## Files

`4bit/`: the suite answers (`suite.json.gz`), the JevBench capture, conformance, the intent heads (`intents.json.gz`, draws 0 to 2; `intents6.json.gz`, BANKING77 draw 0 again and draws 3 to 5), the scored gates (`report.json`), memory, the server's `/health` (every flag the default, `--debug-readout`) and log, the host record.
