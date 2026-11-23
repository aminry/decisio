# Labels-needed estimator: certification from traffic labels

Method: decisio-labels-needed/heldout-bootstrap.v1. Forecast procedure: decisio-certify/traffic-labels-fixed-sequence.v1.

At alpha 0.05, predictions were within a factor of 1.5 in 287 of 303 evaluated units (95%) overall, including cases where both prediction and realised label need were beyond the evaluated range. Where a certificate was reachable, the count was 21 of 34 (62%). Predictions under-estimated the realised label need by more than a factor of 1.5 in 4 of 303 units (1.3%). Both accuracy figures describe the measurement; neither guarantees a prediction for an individual task or language.

The evaluation used a 12B model and eligible inputs from 104 held-out evaluation sets (a task in one language each), target shares 10%, 30% and 50%, and current certification-region label counts 50, 100 and 200 where the scored region held at least twice that many answers. Smaller regions supported only the 50-label case. A unit combines an evaluation set, error level, target share and current label count.

Numeric forecasts are validated only at alpha 0.05 for matching inputs, one full-budget certification look at delta 0.1, and a registered region whose share is min(1, 1.5 times the target share). The default region share of 0.45 supports the 30% target; the other targets need their matching region. A forecast does not change the registered certification schedule. Counts include certification-slice labels in that region, not audit, margin or typical labels, registration examples or all answers served. The traffic forecast grid ends at 2,000 such labels, and validation extends only as far as the largest supported label count in each evaluation population.

At alpha 0.10, no numeric forecast is shown: the estimate tends to over-predict the labels needed at this level. At alpha 0.01 and 0.005, the status is beyond_range and accuracy is not measured. Beyond_range does not mean that certification is impossible.

The estimator resamples the current certification-region labels' confidence and correctness, applies the traffic-label certification procedure and estimates the total region labels needed to reach a target share. Its 10th/90th percentile range describes forecast variation, not certificate confidence. These results do not change the separate examples-certificate estimator or provide a labels-needed forecast for Average error (FDR).
