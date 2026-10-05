# The rule for ahead, level and behind

Written 2026-10-05 at 21:46 UTC, before any comparison in this record was computed, and committed on its own before the script that applies it.
It does not change after that commit; a change would be a dated amendment at the end, with its reason.

## What is marked

Each cell of decisio's three bases (Qwen3.6-35B-A3B, Gemma 4 12B, Gemma 4 31B) is marked against Jev 1.13.0 as ahead, level or behind.
The other entries' columns are shown without marks.
The same rule applies to every base, every benchmark, every area and the index; no benchmark is dropped, merged or reweighted.

## The values compared

- **A Decision Index benchmark:** the coverage-adjusted primary metric that the index uses, `benchmarks[id].raw` on the board (Jev: `jev.benchmarks[id].raw`, the field EVAL_CARD section 8.1 names) against `index_benchmarks[id].raw` in each of our runs' `scores.json`.
  Both are scored on the same requests of edition 0.2.1.
- **ForecastBench (id 48),** the one lower-is-better metric in the index: its `raw` field is the baseline-relative skill, not the metric, so the Brier loss itself is compared, `jev.results['48'].score` on the board against `benchmarks['48'].score` in our `scores.json`, over the same 10,139 requests, lower being better.
- **An area and the index:** the board's chance-corrected values, `categories[].skill` and `scores.balanced_skill` (Jev: `jev.categories[]`, `jev.scores`), against our `areas[].skill` and `decision_index`.
- **JevBench's 231 published questions:** the number correct of 231.
  Jev's is the JevBench board's `public_accuracy` × 231 (revision v1.4.2.2); ours are the counts in this repository's JevBench records.
- Values are compared unrounded, as the files hold them.

## The interval

For a difference between two systems measured independently on the same n items, the 95% half-width is

```
h = 1.96 × sqrt(2 × m × (1 − m) / n)
```

where m is the mean of the two values being compared, on the metric's 0 to 1 scale.

- m(1 − m) is the largest variance a per-item score between 0 and 1 with mean m can have: the binomial variance for accuracy and an upper bound for per-item F1, nDCG@10 and Brier loss.
  For macro-F1 and single-class F1, which are not means of per-item scores, it is used as a stated approximation.
- n is the smaller of the board's `cases` for the benchmark and the number of scored requests, so that several tracks or several requests per case are not counted as independent items.
- The interval is unpaired: Jev's per-item results are not public, so the pairing that would narrow it cannot be used, and the rule leans toward level.

For an area and the index, the same per-benchmark variance is carried through the board's published formula.
Each benchmark enters as its skill, s = (raw − chance) / (1 − chance) (ForecastBench: (0.25 − Brier) / 0.25), so the variance of a skill difference is 2 m (1 − m) / (n (1 − chance)²) (ForecastBench: 2 m (1 − m) / (n × 0.25²)).
An area is the weighted mean of its benchmarks' skills (gold benchmarks 1.2, the rest 1.0), so its variance is Σ w² var / (Σ w)²; the index weighs the five areas by the board's `suite.area_weights`, so its variance is Σ W² var(area).
The half-width is 1.96 times the square root, times 100 on the 0 to 100 scale; clipping at 0 and 1 is ignored.

For JevBench, n is 231 for the count and 111 for the hard tier.
Jev's hard tier on the JevBench board is over 220 items, 109 of them held out and never published, so it is not the same item set as our 111; that cell is marked "not compared" unless a value on the same 111 is public.

## The decision

- d = ours − Jev for a higher-is-better value, and d = Jev − ours for a lower-is-better one (ForecastBench's Brier loss).
- **Ahead** if d > h, **behind** if d < −h, **level** if |d| ≤ h.
- A cell with no Jev value on the same items is "not compared"; it counts as none of the three.

## Not marked

Latency, price or self-hosting cost and context length are shown with their measurement conditions and not marked: a hosted round trip over the internet and our server time on one card are not the same measurement, and a price is not a benchmark.
The capabilities table states what each side's documentation states; it carries no marks.

## Multiplicity

There are 38 benchmarks per base, each at 95%: if a base were truly level with Jev on all of them, about two cells would still be marked ahead or behind by chance.
The counts are reported as they fall, with no correction.

## Amendment 1 (2026-10-05 21:48 UTC, before any comparison was computed)

- **n also takes Jev's own case count when it is smaller** (`jev.results[id].cases` on the board).
  Reason: a benchmark whose value is one headline track is scored on that track's cases only; iSarcasmEval's value is its track A, English (1,400 cases), while the board's `cases` (4,600) and our scored requests (4,600) count all four tracks.
  Checked by comparing the three counts for all 38 benchmarks, which reads no score: they differ only for GPQA Diamond (198, 198, 196 scored), GSM8K (1,319 cases, two tracks over 2,638 requests), ACOS (400 reviews, 1,565 requests), BPoMP (811 cases, 5,000 requests) and iSarcasmEval (Jev 1,400); the smallest count is used in each.
