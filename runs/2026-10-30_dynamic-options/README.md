# Dynamic-option teaching: the measurement behind `"options": "dynamic"` (2026-10-30)

An offline evaluation of tasks whose options change with every request, on stored Gemma 4 12B and 31B readouts of five recurring questions: brand from a product title, entity linking, tool choice, product matching and intent routing (1,200 teaching, 800 certification and 1,000 test requests each; two to eight options and a none option; three draws of 200 labels; `alpha` 0.05, `delta` 0.1).
The sources are public datasets with open licences (Amazon Shopping Queries, ZESHEL, ToolACE, Schema-Guided Dialogue); no request or option text is kept in this repository.

- **Decisio alone**: its own distribution with a certificate made on its top probability, whatever the options.
- **Re-ranker**: the same with the re-ranker of `decisio.serve.dynamic` (13 features of an option, a conditional logit, no encoder), the certificate made on its top probability. Evaluated twice: kept whatever its cross-validation says, and as shipped, kept only where its 5-fold cross-validation beats decisio's own distribution.

`summary.json` holds the means.
`tests/unit/test_dynamic.py` reproduces the cells from `tests/data/dynamic_options` (numbers and hashes of the items): the same certified share, error and accuracy as an independent offline implementation of the method, to 1e-9, for all five questions on both bases at the first draw; a run over all fifty cells (2 bases x 5 questions x 5 draws) matched every share, error and accuracy.
With the gate the re-ranker was kept in 14 of 15 draws on the 12B and in 10 of 15 on the 31B (brand on the 31B declined it in all three); a task for which the server declines it is served as decisio alone.

Limits: two model sizes and five question types; the gain comes mostly from entity linking and product matching, where nothing is certified; the features read the option text as the request shows it (a tool's name and description for tool and intent routing).
