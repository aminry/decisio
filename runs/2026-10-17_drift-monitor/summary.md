# The traffic monitor on a drift replay: the answer mix against a confidence signal (2026-10-17)

**Status: offline replay of recorded reads; the same stream on a card with the new server is a separate cell.**
The replay is a card run of 2026-10-16: the stored Gemma 4 31B with a CLINC150 task (10 examples per intent, 150 options, a 0.05 certificate whose threshold, 0.0597, certifies every answer), then 5,363 messages in which the out-of-scope share rises from 0 at message 1,000 to 30% at message 4,000. The task has no option for an out-of-scope message, so each is a wrong answer, and each was certified.
The reads were recorded; here they are fed, in order, to the monitor class of this change, with the baseline built the way the server builds it, from the 3,000 certification examples' choices and confidences. No message text is kept.

## What the old monitor did

The answer mix (500-answer window, checked every 100 answers, a line from resampling the examples) sat at a distance of 0.18 to 0.29 against a line of 0.28 from the start, and alerted after message 5,300 (the card's own status checks put it after 5,200). By then 1,008 wrong certified answers had been served since the rise began, and certification stayed on after the alert.
With 150 options a shift spread over many of them barely moves the mix, and a window of 500 over 150 options is noisy: the distance of normal traffic is already near the line.

## The confidence signal

Out-of-scope messages are answered with a lower top probability (median 0.67, against 0.99 in scope), so the share of answers under the examples' 5th percentile (0.786) rises. A CUSUM on that share, with 1.3 times the examples' share as normal and 2.5 times as an alarm, and a limit of 8.78 (a stream at the normal share alarms in 1 of 100 five-thousand-answer runs, simulated), gives:

| | First alert | Wrong certified answers from the rise (message 1,000) to the first alert | To the end of the stream |
| --- | ---: | ---: | ---: |
| Answer mix (the old monitor) | 5,300 | 1,008 | 1,008 and more, certification stays on |
| Confidence signal, report only | 1,697 | 55 | 1,028 |
| Confidence signal, suspending certification | 1,697 | 55 | 55 |

The 4,500 in-scope messages of the stream alone (no drift) raised no alert. Nor did the test split of three BANKING77 draws (3,080 messages each) read through their tasks, with each draw's calibration reads as the baseline (tail share of the traffic 0.041 to 0.047 against 0.050 of the examples); the answer mix, on the same three streams, alerted at its first check (message 500) on normal traffic, because the examples' mix is the training split's and the traffic is the test split's. That is why only the confidence signal suspends certification.

## How the numbers were chosen, and what that means

`detector_variants.txt` lists the settings tried on this stream (tail quantile 5% and 10%, the normal share 1.3 or 1.5 times and the alarm share 2.0 to 3.0 times the examples'), with the alert message, the wrong certified answers before it and whether the in-scope-only messages stayed quiet. The settings used are the fastest of those that were quiet on the in-scope messages. They were therefore chosen on this stream, and the in-scope control and the BANKING77 controls are the only traffic they have been checked against; a second stream they were not chosen on is the test.

## What this does not show

One task, one base, one card, one draw; an out-of-scope share rising to 30%, not a sudden jump or a slow creep of in-distribution change; the confidence signal sees a shift only if the shifted messages are answered with lower confidence, which an overconfident model may not do; the replay is offline on recorded reads, so the new server's own scoring (decisio PR #211) is not in it.
