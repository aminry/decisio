# Labels-needed estimator: examples certificates

Method: decisio-labels-needed/heldout-bootstrap.v1. Forecast procedure: decisio-certify/holdout-two-arm.v1.

In our tests of the examples-certificate estimator on a 12B model, predictions were within a factor of 1.5 in 629 of 666 evaluated units (94.4%) at alpha 0.10 and 644 of 666 (96.7%) at alpha 0.05. These totals include cases where both prediction and realised label need were beyond the evaluated range. Among units with a finite realised label need, the counts were 82 of 102 (80.4%) and 31 of 36 (86.1%), respectively. Predictions under-estimated the realised label need by more than a factor of 1.5 in 9 of 666 units (1.4%) at alpha 0.10 and 8 of 666 (1.2%) at alpha 0.05.

The evaluation covered 104 held-out evaluation sets (a task in one language each), target certified shares 10%, 30% and 50%, and current label counts 50, 100 and 200. A unit combines an evaluation set, error level, target share and current label count. These aggregate figures are not an accuracy guarantee for an individual task or language. Numeric forecasts are validated only for matching examples-certificate inputs with 50 to 200 current labels at alpha 0.10 or 0.05; no result is transferred to another certification procedure.

At alpha 0.01 and 0.005, accuracy was not measured: the evaluated label need was beyond range. The estimator reports beyond_range with no numeric forecast or accuracy there. The evaluated total-label grid ends at 5,000; beyond_range does not mean a task is impossible to certify.

The estimator resamples the current held-out confidence and correctness pairs, runs the task's certification procedure, and estimates the total labels needed to reach a chosen share. Its 10th/90th percentile range describes forecast variation, not the certificate's confidence level. This measurement does not cover certification from traffic labels or Average error (FDR).
