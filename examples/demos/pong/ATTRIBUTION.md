# Attribution

This directory is a modified copy of **jev-pong** by Ably Labs.

- Upstream: https://github.com/ably-labs/jev-pong
- Commit: d28d6ae34508e0a6aed0b6c4b93f963e650feb4e
- Licence: Apache License 2.0 (the unmodified text is in `LICENSE`)

## What was changed

Every changed file carries a "Changed from upstream" note in its header comment, as Apache-2.0 section 4(b) asks.
In summary:

- The hosted-gateway decider is replaced by a plain `POST <baseUrl>/v1/systemone` (`lib/decide/systemone.ts`), with no key and no vendor SDK.
- The chat-model decider (`lib/decide/chat.ts`) talks to any OpenAI-compatible server instead of the gateway, so a local text model can serve it.
- Lanes are configured (`lib/config/lanes.ts`, `lanes.example.json`) instead of fixed.
- The replay page, the game engine, the lane runner and the recorder are kept.
- Removed: live play, the arena, the explainer pages, the realtime transport, the clip renderer and its native canvas dependency, the analytics, the social card, the brand assets, the design notes, the pre-recorded replay and its published numbers, and the upstream README.
- The accent colour token is renamed from `--jev` to `--accent`; the lane with id `sys1` carries it.
- `p50Ms` was added to the per-lane statistics.
- `lib/decide/prompt.ts`: opt-in prompt variants (`PONG_PROMPT_VARIANT`: `tolerance6`, `offset`) for testing fixes against an oracle; the default question and state are unchanged. `scripts/oracle-replay.ts` and `scripts/teach-examples.ts` score lanes against a perfect-information paddle and draw oracle-labelled examples.
- `lib/decide/log.ts`: with `PONG_DECISION_LOG=<path>` set, each decision's request and full answer are appended to that file.

No trademark or logo of the upstream project's sponsors is used.
Nothing in this directory contains measurements taken from any hosted service.

Changes made in this directory are under the licence of the file they change.
