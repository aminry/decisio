# The card session of 2026-10-14 (S11): the parts that concern decisio's public documents

Run on one RTX PRO 6000 Blackwell Workstation Edition (600 W default and enforced, driver 595.104.02, AMD EPYC 9534 host); gate passed (12B first question median 283.5 ms against 325 ms at 3,000 tokens).
"Pinned" below is decisio 0.11.1 on vLLM 0.30.0; "0.31.0" is decisio main with vLLM 0.31.0 (#144).
Parts that concern weights decisio does not publish are not in this record. Raw files are as the session wrote them, except `W/summary.json`, which keeps aggregates only (the full verdict names items of a private request set).

| Part | Stack | What was run | Result | Files |
| --- | --- | --- | --- | --- |
| G | 0.31.0 | `tests/gpu/same_forward.py` as it stood (compares a first read with a cached read) | FAIL: 31B 5.52e-4, 12B 6.77e-2 against 1e-4, same option 60 of 60 | `g/` |
| G2 | 0.31.0 | the corrected gate (#167: two cached reads) | PASS: 31B 1.39e-17, 12B 5.55e-17 against 1e-6, same option 60 of 60; the printed first-against-cached numbers equal G's | `G2/` |
| G2b | pinned | the corrected gate | PASS: same 1.39e-17 and 5.55e-17; first-against-cached 31B 1.60e-3, 12B 6.77e-2 | `G2b/` |
| Control | pinned | the default 12B, two fresh starts, 200 single-question requests | equal on 200 of 200, identical fingerprints | `s10/` |
| R1 | pinned | the default 31B, two fresh starts, the same 200 requests | equal on 200 of 200 | `r1/` |
| K | main at the certified-answers branch | the 12B certified-answers restart check | exit 0, but the certificate had no threshold (certified share 0 of 200 at alpha 0.05, accuracy of the 200 questions 0.835): nothing certified, so nothing could flip | `K/` |
| P | 31B pinned, 12B 0.31.0 | 50 questions about a 3,000-token document against one plain read of the same 4,843 tokens | 31B: 1,036 ms against 767 ms (ratio 1.35); 12B sequential 1,550 ms against 467 (3.32); 12B warm 744 ms against 465 (1.60) | `P/` |
| W | 0.31.0 | the 12B, `--multi-question sequential` against `warm`, two fresh starts each, suite4, jev4, travel | inconclusive (the sequential control is too noisy to read): warm 0.48 of sequential's time; warm against warm exact; two sequential starts chose differently on 12 of 1,400, 3 of 231 and 7 of 1,088 questions; warm chose another option than both sequential starts on 4 of 1,400, 0 of 231 and 4 of 1,088 (near ties) | `W/` |
| L | pinned | first and second question by state length, three bases | the table in `docs/running.md`; Qwen has no 32,000-token value (the probe crashed on a missing figure) | `L/` |

Reading, from the files and the session's results (in our private records):
- G failed because it compared a first read with a cached read, which differ by design on Gemma; the corrected gate passes on both versions and the first-against-cached number is of the same order on 0.30.0 and 0.31.0, so it is not new in 0.31.0 (the 0.30.0 run was in a checkout with transformers 5.18.0 against 5.17.0).
- The control repeats exactly on this box, on 0.30.0, on single-question requests; gate W's four-question sequential runs on 0.31.0 do not repeat. The two differ in stack and in request pattern, so neither is named as the cause.
- `box/` (`first_contact.json`, `host.txt`) and `stage_power_clock.md` describe the box and the clock and power by stage; `manifest.json` lists every file with its sha256.
