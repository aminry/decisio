# `--pad-policy row` against the served default (session of 2026-10-02)

### l1

| Gate | Result | Verdict |
| --- | --- | --- |
| G1 suite accuracy (1,400) | 76.29% against 76.21%, +0.07 [-0.79, +0.93] points | pass |
| G2 flips | 57 of 1400 (max \|dp\| 0.23) | reported |
| G3 pooled suite ECE | 0.0254 against 0.0201 | pass |
| G4 C2 to C4 | {'C2': True, 'C3': True, 'C4': False} | FAIL |
| G5 JevBench 231 | 83.98%, +0.00 [-1.73, +1.73]; 5 changes; ECE 0.056 against 0.048; I_open A/B 42.3 / 51.9 | pass |
| G6 BANKING77 (3080) | 74.22%, +0.23 [-0.45, +0.91]; 178 changes; ECE 0.0461 against 0.0421 | pass |
| G6 CLINC150+OOS (5500) | 82.29%, -0.13 [-0.67, +0.40]; 298 changes; ECE 0.1891 against 0.1704 | pass |
| G6 GPQA-Diamond (198) | 49.49%, +0.51 [-3.54, +4.55]; 20 changes; ECE 0.1327 against 0.1157 | pass |
| G6 MMLU-Pro (12032) | 60.92%, +0.07 [-0.28, +0.41]; 839 changes; ECE 0.0093 against 0.0093 | pass |
| intents banking77 (head True) | 84.7% against 84.7%, +0.00 [-3.33, +3.33] | pass |
| intents v3_clinc (head True) | 91.0% against 87.0%, +4.00 [+1.00, +8.00] | pass |
| fresh-state server ms (median of 20) | 300 tokens x 1: 38.1 against 48.8; 300 tokens x 4: 106.9 against 107.4; 1000 tokens x 1: 40.9 against 52.1; 1000 tokens x 4: 110.5 against 111.2; 3000 tokens x 1: 79.2 against 91.0; 3000 tokens x 4: 156.5 against 156.7 | reported |
| G8 memory (this card) | Model loading took 33.31 GiB memory and 11.964386 seconds / Using default MoE config. Performance might be sub-optimal! Config file not found at <venv> / Available KV cache memory: 47.11 GiB / GPU KV cache size: 2,016,956 tokens, Maximum concurrency for 32,768 tokens per request: 61.55x | reported |

All gates: **fail** (G4 conformance (C2 to C4)).
