# `--noul-commit` at the served default (session of 2026-10-03)

The served default's yes/no answers on 1,474 items, before and after the transform (`systemone.commit_noul`: P(yes) strictly between 0.20 and 0.80 reported as 0.80 above 0.5 and 0.20 at or below it).
The transform is applied to the served P(yes) in `yesno_readouts.jsonl.gz`; nothing else changes.

| | Accuracy | Committed accuracy | Inside 0.20 to 0.80 | Log loss | ECE, tie-robust |
| --- | ---: | ---: | ---: | ---: | ---: |
| served | 81.4% | 59.6% | 34.3% | 0.4124 | 0.033 |
| `--noul-commit` | 81.4% | 81.4% | 0.0% | 0.4379 | 0.072 |

- Accuracy: the served answer (yes above 0.5) against the gold; unchanged by construction.
- Committed accuracy: right only when P(yes) is at or beyond the band's edge on the right side (at least 0.80 for yes, at most 0.20 for no).
- Log loss: of the gold answer's probability. ECE: tie-robust, 10 equal-mass bins of the answer's confidence, the mean over 200 random orders of tied confidences (rounded to 1e-9), seed 0.
- JevBench v1.5 (`decisio.bench.jevbench_v15` on the harness's 231 published items, which scores a yes/no answer inside the band as no answer): yes/no competence -10.9 to 59.15, I_open A 42.69 to 66.04.

Items: the suite's yes/no tasks (BoolQ, SciFact, SciFact clarified, ToxicChat; 150 each), PAWS (250), Civil Comments (300), Aegis 2.0 (250) and JevBench's 74 yes/no items.
