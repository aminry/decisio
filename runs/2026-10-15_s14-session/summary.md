# The card session of 2026-10-15 (S14): the auto default context on one H100 PCIe 80 GB

Run on one NVIDIA H100 PCIe 80 GB (81,559 MiB; power limit 350 W, the default 310 W; driver 610.57.04; an Intel Xeon Gold 6438N host), 0.99 card hours; the first-contact gate passed (`first_contact.json`, `gate/`).
Stack: decisio `8a9746dd` (the branch of #196, `gate/branch_head.txt`) on vLLM 0.31.0, the stored FP8 31B (`tachara-ai/decisio-gemma-4-31b`) and the 12B.
The pass rules and predictions were written before the run; `S14_VERDICTS.json` holds the runner's verdict for each cell.
`manifest.json` lists every file with its sha256.

The default under test: a vLLM server started without `--max-model-len` serves the largest multiple of 8,192 that fits the card's KV cache, up to the base's ceiling and not below 32,768 (`docs/running.md`, "The context length").

| Cell | What was run | Result | Files |
| --- | --- | --- | --- |
| B | the 31B at `--max-model-len 32768`, then the 200 single-question requests (the control arm) | engine init 286 s, 387 s to ready | `B/` |
| A | the 31B with no length flag | serves **98,304** ("auto: the base's ceiling of 98,304 fits on this card"): vLLM reported 33.85 GiB of KV cache and a fit of 105,600 tokens (concurrency 1.00 at 105,600); engine init 265 s, 362 s to ready; predicted 90,112 or 98,304 | `A/` |
| C | A against B on the same 200 requests | **200 of 200 exactly equal and every tensor's bytes identical**; B2 (B started again) is also equal to B on 200 of 200 | `C/`, `B2/` |
| E | a prompt of 100,304 and one of 118,304 tokens on A's server, then the first of the 200 requests again | both long prompts answered **422** ("This model's maximum context length is 98304 tokens", although vLLM's own fit was 105,600, so the limit that refused them is the one the server lowered it to); `/health` stayed up; the repeated short request was not exactly equal to its first answer (see below) | `E/` |
| D | the long-context check at 80,000 tokens, 30 items | no errors; 24 right against 26 right on the same items unpadded (rule: at least 23); first question median 18.8 s, second 477 ms | `D/` |
| G | no length flag at `--gpu-memory-utilization` 0.85 and 0.70 | 0.85 started and served **49,152** (vLLM's fit 53,696); 0.70 **did not start**: vLLM's fit was 22,064 tokens and the server refused with its own message ("less than the 32,768 every record was measured at") | `G/` |
| F | no length flag at 0.55 | the start failed (exit 6) with the same message (vLLM's fit 7,968), no server left running | `F/` |
| L12 | the 12B, no length flag | serves **65,536** ("auto: the base's ceiling of 65,536 fits on this card"); 43.28 GiB of KV cache | `L12/` |

## What the card gave for the 31B, by the engine's share of the card

| Engine share | KV cache vLLM reported | vLLM's fit (the longest single request) | Served |
| ---: | ---: | ---: | --- |
| 0.90 (the default) | 33.85 GiB | 105,600 | 98,304 (the ceiling) |
| 0.85 | 29.9 GiB | 53,696 | 49,152 |
| 0.70 | 18.54 GiB | 22,064 | does not start |
| 0.55 | 6.71 GiB | 7,968 | does not start |

The fit is **not linear in the KV memory**: 12% less KV cache (0.90 to 0.85) halves the fit, and the tokens of fit per GiB fall from about 3,100 (at 0.90) to about 1,800 (0.85) to about 1,190 (at 0.70 and 0.55).
This record has not isolated why; the 31B's cache has several groups (vLLM's hybrid KV cache manager), and the length is the largest single request whose groups all fit.
The practical reading is that the length a card serves depends strongly on `--gpu-memory-utilization` and cannot be estimated from one measurement; read `/health` (`max_model_len`, `max_model_len_source`).

## E: the repeated short request

The answer compared is the first of the 200 requests (`boolq:1`), asked again on the server that had answered it in A's pass, against that first answer, which was a first read of the state.
A repeat on a server that has already answered a request can differ from its first answer on the Gemma bases (a first read and a read from the prefix cache are different computations; `EVAL_CARD.md` section 7.5: on a server that had answered the 200 once, asking them again changed 5, by at most 0.031, with no top answer changed), so the comparison as written, exact equality with a first read, is not a test of the refusals.
The size of the difference was not recorded.
A test that separates the two (three answers on a fresh start, with and without the refusals between them) is written for the next card session.

## Not in this record

The identity check of the stored FP8 at 98,304 against 32,768 on another box, which is a separate record.
