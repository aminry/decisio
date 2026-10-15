# The per-task head temperature, live, on decisio v0.12.0 (2026-10-15)

On decisio v0.12.0 (`f5e40e0f`) on vLLM 0.31.0, one RTX PRO 6000 Blackwell Workstation Edition at 600 W, a task registered with 200 or 1,000 labelled examples gets an intent head whose answers are served at a per-task temperature that the server fits on examples the head did not see.
The measurement ran in the card session of 2026-10-15 (`runs/2026-10-15_s12-session/`, the teaching-study part) on both Gemma bases.

**One task, one draw.**
The task is a 16-option decision, which of 16 team rules covers a shell command, on a synthetic set that is not published (so the record has numbers and no item text).
There are 300 test items, none of them among the registered examples, and one draw of the 200 or 1,000 examples (draw 0 of an offline grid), natural examples.

## The table

"With" is the live server's answers.
"Without" is the same head fitted on the same examples and served as the softmax of its logits, which is how decisio 0.11.1 served it; it is an offline replica, not a 0.11.1 server.
ECE is tie-robust (10 equal-mass bins, 50 tie orders); the interval is a paired bootstrap of "without minus with" (1,000 resamples over the 300 test items, seed 0).

| Base | Examples | Accuracy with / without | ECE with | ECE without | ECE without minus with (95%) | Log loss with / without | Server temperature | Offline temperature | Registration |
| --- | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| Gemma 4 31B | 200 | 0.807 / 0.800 | 0.061 | 0.167 | +0.105 (+0.035 to +0.130) | 0.739 / 2.100 | 3.845 | 3.816 | 79 s |
| Gemma 4 31B | 1,000 | 0.830 / 0.823 | 0.040 | 0.143 | +0.103 (+0.048 to +0.122) | 0.590 / 1.338 | 3.361 | 3.365 | 393 s |
| Gemma 4 12B | 200 | 0.770 / 0.773 | 0.052 | 0.168 | +0.116 (+0.055 to +0.139) | 0.773 / 1.453 | 2.742 | 2.758 | 40 s |
| Gemma 4 12B | 1,000 | 0.817 / 0.807 | 0.054 | 0.146 | +0.092 (+0.036 to +0.124) | 0.629 / 1.046 | 2.483 | 2.442 | 227 s |

- The temperature the server reports is within 1.7% of the cross-validated temperature fitted offline on the same examples before the feature existed, and the penalty strengths (lambda) are equal: 0.0001, 0.0001 on the 31B and 0.01, 0.001 on the 12B, in the table's order.
- Accuracy is within one point either way; what moves is the confidence (log loss falls by 0.4 to 1.4 nats).
- The interval of the 31B at 200 examples is the widest; all four exclude zero.

## What this does not say

1. **Live and offline are not the same function.**
   With the server's own temperature applied to the offline head, the live answers still differ by a median of 0.011 to 0.023 per item, 155 to 199 of the 300 items differ by more than 0.01, and 7 to 22 argmax answers disagree (2% to 7%).
   The live answers are slightly the better ones on three of the four cells.
   The offline head is a close model of the live one, not the same function.
2. **One task, one draw, 300 test items.**
   The intervals cover the test items only; the draw of examples is fixed, so they do not cover a different draw.
   Across the offline grid's draws and sizes on the 31B, the cross-validated temperature ranged from 3.3 to 4.6 on this task and from 3.3 to 3.8 on BANKING77; the 12B's draws were not analysed offline, so its two live temperatures (2.7 and 2.5) stand alone.
   Nothing here says the result holds for other tasks.
3. **Certification is a different matter.**
   The same registrations certified little: the 31B certified nothing at either size or alpha (alpha 0.02 at 200 examples was refused with HTTP 422, the floor being 149 held-out examples), and the 12B certified at 1,000 examples and alpha 0.05 (threshold 0.9576, 209 of 500 held-out, 4 errors, upper bound 4.3%; on the 300 test items it certified 120 answers with 3 wrong).
   The server certifies on half of the examples; a registration can honestly come back "not certified".
4. **"Without" is a replica.**
   At 96 examples the 31B's offline and live heads agreed to 2e-8 on the 0.11.1 stack, and at 200 examples they differed by up to 0.061 with no changed answer, so the replica is close and not exact.
   A "before" measured live would need one registration on a v0.11.1 server at the same examples (about 0.1 card hours); it has not been run.

## Files

`live_head_temperature.json` (the table's numbers), `head_check_gemma_4_{31b,12b}.json` (the server's temperature and lambda against the offline fit, and the live head against the offline head at the server's temperature), `score_gemma_4_{31b,12b}.json` (the scoring of the live and offline arms); `manifest.json` has the sha256 of each.
The scoring code and the examples are not in this record (the task is not published).
