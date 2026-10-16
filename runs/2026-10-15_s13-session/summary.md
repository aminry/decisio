# The optional parts of the card session of 2026-10-15 (S13): the stored and the on-load FP8 31B, and JevBench at 98,304 tokens

Run on one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (600 W), vLLM 0.31.0, the stored FP8 31B (`tachara-ai/decisio-gemma-4-31b`) and Google's bf16 31B quantized to FP8 on load (the on-load arm).
The 200 requests are public suite items (ids and probabilities); the JevBench files are the public items' summaries and per-item answers without the raw harness files.
`manifest.json` lists every file with its sha256.

| Part | What was run | Result | Files |
| --- | --- | --- | --- |
| F2-a | two fresh starts of the stored arm and two of the on-load arm, the 200 requests one at a time | stored pair **200 of 200 equal**; on-load pair **200 of 200 equal**; the cross pair **not** equal: median 3.8e-3, largest 0.276, 5 top answers changed, and the 420 scale scalars that differ in shape (0-d against one element; their bytes are equal, `runs/2026-10-15_s12-session/F/`) | `F2a/` |
| F2-b | JevBench's 231 public items, stored against on-load, one start each | 2 top answers changed (easy 0, original 0, hard 2); accuracy and calibration of the same order | `F2b/` |
| F2-c | the cross pair again with eager execution on both arms | **200 of 200 equal**, although the fingerprints still differ in the 420 shapes: with eager execution the stored and the on-load arm answer identically, so the difference is in the compiled path. Eager is the first step that makes them equal; the other compile-cache step and the 0.30.0 checkout were not run | `F2c/` |
| JevBench at 98,304 | the stored arm started at 98,304 tokens (`--engine` route), JevBench's 231 items, against F2-b's stored answers at 32,768 | below | `jevbench_98304/` |

## JevBench at 98,304 against 32,768 (the stored FP8 31B, the same box)

| Tier | Correct at 32,768 (F2-b) | Correct at 98,304 | ECE at 32,768 | ECE at 98,304 | Brier at 32,768 | Brier at 98,304 | Top answers changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy (48) | 48 | 48 | 0.0194 | 0.0197 | 0.0006 | 0.0006 | 0 |
| original (72) | 71 | 72 | 0.0386 | 0.0350 | 0.0180 | 0.0074 | 1 |
| hard (111) | 91 | 93 | 0.0924 | 0.0915 | 0.2663 | 0.2596 | 2 |
| all 231 | 210 | **213** | | | | | **3** |

- At 98,304 the stored model reads the published figures: 1.000, 1.000 and 0.838 on the three tiers (213 correct), ECE 0.035 on the standard tier and 0.092 on the hard tier (`docs/running.md` and the README quote these).
- The 3 items that changed were 3 of 231, in the direction of the published figures; the two runs' calibration is of the same order. The 32,768 run on this box reads 210; the stored model's 32,768 run in another session (`runs/2026-10-15_s12-session/J/`) read 213, so a difference of three items between runs at the same length is within what restarts and boxes already produce, and the 98,304 run is not evidence that the longer length is better.
- The identity on the 200 suite requests at the same lengths is in `runs/2026-10-15_identity-at-98304/` (the length moves short answers by a restart's size on this card class and not on an H100 PCIe).
