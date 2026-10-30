# Intent-head floor: 2 examples per option instead of 5 (2026-10-30)

The intent head used to be declined for a whole task when any option had fewer than 5 labelled examples, so two or three rare options out of 77 turned it off.
`intent_head.MIN_PER_OPTION` is now 2.
This record is the offline measurement behind that number and the docs in `docs/tasks.md`.

Method (comparison fixed in writing before any result was read):
- Decisio's own fitting code (`TaskStore.register` and `intent_head.fit_intent_head`, which keep the head's 5-fold cross-validation and its temperature), run offline on the stored Gemma 4 12B and 31B readouts of ten decision sets (seven with 10 or more options, so 42 cells that can have a head: 2 model sizes x 50, 200 and 1,000 labelled examples; 2 or 3 natural draws each; the set's own certification and test pools).
- Arms: the floor of 5 (the old rule), 2 and 3; two heavier designs (a head over the options that meet the floor plus calibration for the rest, and a pooled "other" class) were also run and gave no advantage over the one-number change.
- Measures: accuracy, macro-F1, the certified share at `alpha` 0.05 (`delta` 0.1) and whether the certificate is violated (the 90% Clopper-Pearson lower bound of the certified error exceeds `alpha`).
- Checks: the old rule reproduced the earlier study's taught-server scores in 152 of 152 draws; tasks with fewer than 10 options were identical across arms in 160 of 160 rows; `TaskStore.register` at the new floor gave the offline distribution to 0.0 on five cells.

Result (`summary.json`): over the 42 cells the floor of 2 raised macro-F1 by 1.08 points on average (90% bootstrap interval from 0.30 to 1.98), accuracy by 1.01 and the certified share by 2.77 points; 10 cells rose, 30 did not change, 2 fell by at most 0.39 point; no certificate was violated where the old rule's was not. The gain comes from two sets: BANKING77 at 1,000 examples (31B 73.4 to 81.2, 12B 69.9 to 80.7 macro-F1) and a 27-intent customer-support set at 200 examples (31B 78.2 to 89.7, 12B 77.0 to 90.5).
The floor of 3 is in the same file: +0.85 points, no cell worse, but BANKING77 at 1,000 examples reaches only 77.1 and 75.4.

Also checked here: the floor of 2 on the stored BANKING77 and CLINC150 readouts (`tests/unit/test_intent_head_floor.py`): the head is kept and beats the plain readout on the held-out examples of every intent.

Limits: stored readouts of two Gemma 4 sizes; the gain rests on few cells; a task whose rare options have 0 or 1 examples still serves calibration; the measure is offline, not a new card run.
