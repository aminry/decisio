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
