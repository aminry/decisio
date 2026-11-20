<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Exact arithmetic for decisions over numbers

A decision often depends on numbers in the text: is this invoice's total right, does this payslip match its rules, is this claim within the limit.
A model reading in one pass is good at checking values it can see and poor at computing exact totals from many lines.
decisio's calculator computes the line values exactly and shows them to the model in a fixed block after the state, so the model checks the values instead of computing them, in the same single read.
The block never states the final figure, a verdict, or whether a claim is right: the decision stays the model's.

It is opt-in per model: the server shows the block only to a model trained with it.

## What the model sees

The rendered state, a newline, then the block:

```text
Shipment SH-310, shipper: Teo Barrand.
...
Calculator (exact values, shown to six decimals):
- Loading time: 89 min x $41.00/h = $60.816667
- Cargo weight: 980 lb = 444.528 kg; 444.528 x $0.35 = $155.5848
- Haul distance: 212 mi = 341.1716 km; 341.1716 x $1.10 = $375.28876
- Cold storage: 2 x $14.25 x 7 days = $199.50
- Missed slot: $40.00 x 2 days = $80.00 (taken off)
- Handling fee: 2 x $8.50 = $17.00
- Claims credit: $25.00 (taken off)
- Taxed base: $703.190227
- Tax charged: $105.478534
```

- The lines are the priced items in the order the state lists them, then percentage adjustments, fees and credits, then the taxed base and the tax (`Tax charged`, or `Tax withheld` for a deduction from pay).
- All arithmetic is exact; values are only shown to six decimals at the end, rounded half to even. Money keeps two to six decimals.
- The format has a version, `calc-v1`. Any change to its bytes is a new version, and a model records the version it was trained with.

## What it reads, and when it shows nothing

The calculator reads structure, not item names: lines such as `Label: 4 items at $12.50 each`, a table of items, or a JSON object's fields, plus the rule sentences that combine them.

- **Items:** a quantity at a price; durations at an hourly rate (with stated rounding to a step, per entry or on the sum); lengths at a rate; a quantity converted with a factor the text states (`1 lb = 0.4536 kg`); per-day charges and day credits over a date range when a rule says whether its last day counts; fees and credits.
- **Rules:** a percentage added to or taken off one named item before tax; the tax rule (its rate, whether it is charged or withheld, and which fees and credits come before or after it); when a credit comes off; the final rounding rule (read, not applied); a definition of the figure asked about.
- **One currency, one record, English (v1).**

**All or nothing.** Every line with a number has to be understood. If anything is uncertain (a sentence with a number or a rule word that matches no rule kind, a label that matches no item or two, a rule whose meaning is not certain, an amount nothing places, two currencies, two records, a date range with no day-count rule), there is no block at all and the response says why.
Counts that carry a multiplier or a number word (`2 dozen eggs at $3.00 each`, `2 packs of six`), and a unit factor that joins two kinds of unit (`1 lb = 0.4536 m`), give no block too (from 0.50.2); a multiplier priced per that same word (`2 dozen at $3.00 a dozen`) is read.
A missing block is the safe failure; a wrong number is the one failure the calculator is built never to make.

## Turning it on

- **The model records it.** A tuned repository records the block format its model was trained with in `decision_config.json`, as `defaults.calculator: "calc-v1"` (written by `python -m decisio.hub tuned-config --calculator calc-v1`). A base and any model that records nothing never see the block.
- **The server:** `--calculator auto` (the default) shows the block when the model records it; `--calculator off` never does. The start prints `CALCULATOR calc-v1 (the repository's defaults)` when it is on.
- **A request:**
  - `"calculator": false` reads the state without the block. Registered tasks on such a server were fitted on reads with the block, so none applies to that request (naming one with `"task"` is a 422).
  - `"calculator": true` asks for it; on a model that records no format the request is answered without the block, with `x-decisio-warning` saying so. It is never shown to a model that was not trained with it.
  - `"calculator_input"` sends the amounts instead of having them read from the text (below).

## Dates and times: `time-v1`

A second block format reads the dates and clock times a state states and writes the facts a model would otherwise have to work out: what a zoned time is in UTC, how much real time lies between two zoned times, and what date "N days after" a date comes to.
It suits rotas, travel logs, shipment and incident timelines, warranty and notice periods.
The block holds facts, never an answer, and is written only where every value is certain.

```text
Calculator, times (exact):
- 13:00 Chicago time, 3 Sep 2026 = 18:00 UTC, 3 Sep 2026
- 01:30 by my watch (Madrid time), 4 Sep 2026 = 23:30 UTC, 3 Sep 2026
- 13:00 Chicago time, 3 Sep 2026 to 01:30 by my watch (Madrid time), 4 Sep 2026: 5 h 30 min
- 31 Aug 2026 + 18 months: 31 Feb 2028 does not exist; February 2028 has 29 days
```

There are four kinds of line:

- **A dated zoned time in UTC.** The zone is an abbreviation with one meaning (CEST, EDT, JST and others), a UTC offset (UTC-4), or a major city ("Chicago time"), read with that zone's rules on the stated date, daylight saving included. A table row's zone cell next to its time cell counts.
- **The real time between two dated zoned times,** across zones and clock changes, with "minus" when the second is earlier.
- **A date plus N days, weeks, months or years,** where the base may be a date or a label the text gives one date for ("1 week after the Start date").
- **A month end that does not exist,** stated and not resolved: the text's own rule decides what happens then.

Anything that could be read two ways gives no line: an abbreviation with several meanings (CST, IST), a zone joined to another ("BST/IST"), a city time without its date or in an hour a clock change skips or repeats, a date without a year when the text names several years, a weekday that contradicts its date.
A text with no date or clock time pays one regex scan.

A model records the format in `defaults.calculator` as for `calc-v1`: `time-v1` for this block alone, or `calc-v1+time-v1` for both (each where it has lines, the billing block first).
Older decisio versions do not know these names and serve such a model without the block.

## Sending the amounts: `calculator_input`

A client that already holds structured amounts can send them, and gets the same block with no reading of the text:

```json
{
  "state": "Shipment SH-310 for Teo Barrand: is the total charge of $469.09 right?",
  "questions": {"total": {"type": "noul", "instructions": "Is the stated total charge right?"}},
  "calculator_input": {
    "currency": "$",
    "items": [
      {"kind": "time", "label": "Loading time", "minutes": [51, 38], "rate": "41.00", "per": "h"},
      {"kind": "convert", "label": "Cargo weight", "quantity": "980", "unit": "lb", "to_unit": "kg",
       "factor": "0.4536", "price": "0.35"},
      {"kind": "days", "label": "Cold storage", "quantity": 2, "price": "14.25", "days": 7}
    ],
    "fees": [{"label": "Handling fee", "price": "8.50", "count": 2}],
    "credits": [{"label": "Claims credit", "amount": "25.00"}],
    "tax": {"rate": "15", "kind": "charged", "outside": []}
  }
}
```

- Item kinds: `count`, `days`, `time` (minutes, a rate per `h` or `min`, optional rounding to a step per entry or on the sum), `length`, `convert` (with the stated factor) and `daycredit`.
- `adjustments`: a percentage of one item, added or taken off before tax. `tax.outside` lists the fees and credits applied after tax; everything else is inside the taxed base.
- Numbers are JSON integers or decimal strings; a JSON float is refused, since it is not exact. Unknown fields are refused. Labels must be unique.
- Bounds: at most 64 items, 32 adjustments, fees, credits or `tax.outside` labels, and 64 minutes or lengths in one item; a number has at most 14 digits before the point and 6 after; a day count is 1 to 3,660.
- Validation is strict and all or nothing: an invalid or inconsistent field is a 422 naming its path (`calculator_input.items[2].price: ...`), on any server.
- The full schema is in the docstring of `decisio.calculator.inputs`.

## What the response says

- `x-decisio-calculator: shown; lines=9; version=calc-v1`, or `none; reason=<reason>` (for example `unparsed-rule`, `ambiguous-label`, `off-by-request`, `not-trained`, `over-limit`); absent when the server shows no block and the request asked for none.
- `usage.input_tokens` counts the block.
- `/health` `systemone.calculator` (when the server shows the block): `version`, `requests`, `shown`, `from_input`, and `none` (a count per reason). No state text is logged.

## How it fits with the rest

- **Every read of the request** sees the same state text: its questions, a certified task's own read, a second read when thinking, both readings with read twice.
- **A task's context** is read with the state (rules may be in the context, amounts in the state); the row is the context, a blank line, the state, then the block.
- **Prefix cache:** the block comes after the state, so the state's prefix is shared as before, and the same state always gets the same block.
- **Never a refusal:** a block that would push the row over the engine's prompt ceiling is left out (`none; reason=over-limit`); the request is read as it would be without the calculator.
- **Tasks and certificates:** the serving-stack and task fingerprints carry the format when the block is on, so a task registered or a certificate made on reads with the block applies only where the block is shown. A task's registration examples are read with the block on such a server.
- **Not here:** requests with images, `--mode packed`, `POST /v1/answer`, and other runtimes (an Ollama listing does not run decisio's server).

## The cascade reader (preview)

The rule reader described above is the default, and it stays the default: nothing turns the cascade reader on unless a server flag or the model's repository asks for it.
The cascade reader is a preview that may show a block for states the rule reader does not read, such as wordings and layouts it has no rule for.
A small role tagger (a token-classification encoder) marks the role of each number in the text, the marked spans are assembled into the same structured amounts as `calculator_input`, and the block is rendered by the same code, so its format and bytes do not change.

**Precision gates** decide whether a block is shown:

- Every number in the text must be marked, every value must come from the text, the units must agree, and a rule the calculator cannot apply exactly (a rounding direction, a limit) gives no block.
- Then one of these must hold, and the response names it as the path: the rule reader reads the state and gives the same block (`rules_agree`), the figures the state itself states are reproduced (`stated_figures`), or every decision of the tagger clears its threshold outside a margin (`threshold`).
- Where the tagger gives no block but the rule reader reads the state, the rule reader's block stands (`rules_only`): the cascade never shows less than the rule reader.
- Where the two give different blocks, there is no block; a tagger failure is no block too, never a failed request.

**Turning it on** (only where the block is shown at all, that is for a model that records `defaults.calculator`):

- `--calculator-reader cascade` with `--calculator-tagger`: a local directory (its `SHA256SUMS` is checked) or `owner/name@<full commit hash>` on the Hugging Face Hub.
- The tagger is loaded once at start, and the start prints `CALCULATOR READER cascade: tagger <name>@<revision> on <device>, thresholds <file or off>`.
- `--calculator-thresholds FILE`: a JSON object with `role`, `link` and `flag` (and `band`, the margin); without it the threshold rule is off, so only `rules_agree`, `stated_figures` and `rules_only` can show a block.
- `--calculator-device auto|cpu|cuda|mps`: `auto` runs the tagger on the serving card for the vLLM backend on CUDA, in the memory vLLM leaves under `--gpu-memory-utilization`, and on the CPU otherwise. The tagger needs `torch` and `transformers`.
- `--calculator-dtype float32|float16|bfloat16`: the tagger's precision; `float32`, the default, is what its thresholds are fitted on.
- The cascade without a tagger, a tagger flag without `--calculator-reader cascade`, a Hub tagger without a full commit hash or an unusable thresholds file stops the start with a message. On a server that shows no block, the cascade is a start note and nothing is loaded.
- A tagger that cannot be loaded never stops the start: torch or transformers missing, a download refused, files that do not match their `SHA256SUMS`, or a device that is not there.
  The server reads with the rule reader, prints `CALCULATOR READER rules (fallback): ...` with the reason, and `/health` `systemone.calculator` carries `reader: "rules"` and `fallback` (`wanted`, `tagger`, `reason`).
  The serving-stack and task fingerprints are then the rule reader's.
- It runs on every backend: on the vLLM backend on CUDA it shares the serving card, and on `--backend mlx` and the CPU stand-in it runs on the CPU (`--calculator-device mps` puts it on a Mac's GPU).
  On a CPU it adds tens of milliseconds to each request whose state it reads, more for a long state.

### From the model's repository

A tuned model can turn the cascade reader on by itself, so a server started with only `--base <owner>/<name>` serves it with no reader flags.
Its `decision_config.json` records the reader in `defaults.calculator_reader`:

```json
"calculator_reader": {
  "reader": "cascade",
  "tagger": "owner/tagger@0123456789abcdef0123456789abcdef01234567",
  "thresholds": {"role": 0.9, "link": 0.6, "flag": 0.8, "band": 0.02}
}
```

- `reader` is `cascade` or `rules`; `rules` takes no other field.
- `tagger` is `owner/name@<full 40-character commit hash>` on the Hugging Face Hub, or for a repository used on one machine a local directory path (absolute, or starting with `./` or `../`).
- `thresholds` holds all four values, each a number from 0 to 1 that the reader accepts (each threshold at least 0.5, with room for the band below 1).
- It needs `defaults.calculator`: the reader applies only where the block is shown, so `--calculator off` or `--mode packed` leaves it unused.
- A malformed record is refused when the repository is read, naming the field.
- **A flag beats the repository.** Any of `--calculator-reader`, `--calculator-tagger` and `--calculator-thresholds` keeps the repository's reader out entirely, so a tagger is never paired with thresholds fitted for another; `--calculator-reader rules` turns it off. `--calculator-device` and `--calculator-dtype` apply to the repository's reader too.
- The start prints the record under `REPOSITORY DEFAULTS` (`calculator_reader`, or `calculator_reader_set_by` when a flag beat it), then the reader's own line, which names the thresholds as the repository's.
- `python -m decisio.hub tuned-config ... --calculator calc-v1 --calculator-reader cascade --calculator-tagger owner/name@<commit> --calculator-thresholds FILE.json` writes it; only the four values of the thresholds file are written.
- An older decisio, which does not know the key, ignores it with a warning and serves the model with the rule reader.

**What it adds** (none of it appears with the rule reader):

- `x-decisio-calculator` names the path of a shown block: `shown; lines=9; version=calc-v1; path=threshold`.
- `x-decisio-stages` carries `calculator=<ms>`, the time the reading took.
- `/health` `systemone.calculator` adds `reader`, `tagger` (`name`, `revision`, `device`), `thresholds` (their values, or `off`) and `paths` (a count per path, `abstain` included).
- The serving-stack fingerprint carries the reader, the tagger's revision and the thresholds, so a certificate made with one reader is suspended under the other (`serving_stack_changed`).
- A request's own `calculator_input` is used as it is, with no reading; a task's registration examples are read with the same reader as requests.

It is a preview: the tagger, its flags and what it reports may change before it can become the default.

## Training rows

The rows a model is trained on must carry the block exactly as serving writes it.
`python -m decisio.calculator render IN.jsonl.gz --out OUT.jsonl.gz` adds the block to each row's state with the same code the server runs (a row the calculator does not read is left unchanged, which teaches the no-block case too), and prints the counts per reason.
Record the decisio version that rendered the rows, and the format in the trained model's repository (`defaults.calculator`).
