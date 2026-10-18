# The split fit reproduces Lab 1's S18 temperatures from the same stored reads (2026-10-18)

`python -m decisio.serve.adapter_temperatures fit-split` was run on the stored reads of the first MeTask evaluation (a tuned Gemma 4 31B, S18 of the card sessions: 776 H items and the 500-item dev pool, read through decisio 0.14.0 at the base's temperatures, the dev score items served at the choice temperature), and compared with the fits made from the same reads by the scoring scripts of that evaluation.

All twelve kappas (four variants, three types) and every held-out figure (log loss and ECE before and after, the other half, for the two half variants) are equal; the largest difference of a kappa is 0.000 at the three decimals reported. `reproduction.json` has both sets.

| Variant | Choice | Yes/no | Score | Temperatures to serve (choice, yes/no, score) |
| --- | ---: | ---: | ---: | --- |
| half0 (the headline) | 4.472 | 2.011 | 1.194 | 1.045, 2.612, 4.400 |
| half1 | 4.535 | 1.472 | 1.300 | 1.030, 3.569, 4.041 |
| full | 4.514 | 1.712 | 1.251 | 1.035, 3.068, 4.197 |
| dev | 2.179 | 1.712 | 1.251 | 2.144, 3.068, 4.197 |

The kappas are against decisio's default temperatures (choice 4.672, yes/no and score 5.252): this model's choice answers want a temperature near 1 where the default serves 4.67, and its yes/no answers 2.6 to 3.6 where the default serves 5.25. Served at the defaults, as decisio 0.14.0 does, it hedged on 70% of a benchmark's yes/no items that its top answer had right.

The pools are a program-keyed H pool and a public-source dev pool; no benchmark item is in either. A different model needs its own fit from its own reads; this record shows the tool does what the method does, not that these temperatures suit any other model.

## The same on the second evaluation (BH, BH0 and the references)

Fitted from the stored reads of the second evaluation (a tuned 31B, BH, a variant BH0, MeTask's adapter A and the stored base F; 776 H items and the 500-item dev pool, dev score items as true score questions, so no score-temperature conversion), the tool's `half0` kappas equal the ones the scoring of that evaluation fitted, for all four models and all three types, to the three decimals posted (largest difference 0.000; `reproduction_s20.json`).

| Model | Choice | Yes/no | Score |
| --- | ---: | ---: | ---: |
| A (MeTask's adapter) | 4.472 | 1.992 | 1.167 |
| BH | 4.869 | 2.169 | 1.218 |
| BH0 | 3.807 | 1.838 | 1.235 |
| F (the stored base) | 0.875 | 0.449 | 0.445 |

**A choice refit on dev-like data** (`--variant devhalf0` and `devhalf1`, then `dev`): the choice kappa fitted on the H pool does not transfer to the dev pool (the adapters' dev choice ECE stays near 0.2 after it). Fitted on halves of the dev pool, with the other half held out, BH0's choice kappa is 1.664 and 1.625 (1.641 on the whole pool, a temperature of 2.85 where the H fit gave 1.2), and the held-out choice ECE falls from 0.221 to 0.127 on one half and from 0.294 to 0.095 on the other; BH's kappa is 2.26 and 2.14 (ECE 0.296 to 0.092 and 0.412 to 0.091), A's 2.22 and 2.10, F's 0.93 and 0.88 (ECE barely moves, 0.146 to 0.108 and 0.109 to 0.104). The H pool and the dev pool want different choice temperatures from the same model, by a factor of two to three for the adapters: the temperature file records which pool its choice entry was fitted on.
