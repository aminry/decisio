<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Calling a decisio server from other tools

Decisio implements TypeSafe's published System One wire format, so clients written for that format can be pointed at a decisio server.
Five of them were checked against a running server: LangChain, the Vercel AI SDK, n8n, TanStack AI and Pipecat (the table in the [README](../README.md#works-with); the checks and their records are in [`runs/2026-10-08_integration-checks/`](../runs/2026-10-08_integration-checks/manifest.json) and in decisio-examples' `integrations/`).
Decisio is an independent project, not affiliated with or endorsed by TypeSafe or by the makers of these clients; none of them is called with a real TypeSafe key, and the key you pass is ignored by the server.
This page gives, for each, a complete short program or workflow: the imports, the call and the output.

How each snippet was checked: **run against decisio's CPU stand-in** (`--backend hf --model Qwen/Qwen3-0.6B-Base`, so the numbers below mean nothing about the served models) with the package versions named, on 2026-10-14; the n8n workflow was run in n8n 2.35.7 on Node 22 with `n8n execute`.
They are not part of decisio's CI: a new release of one of these clients can change its API; the version pinned beside each snippet is the one that ran.

Every snippet assumes a server on `http://127.0.0.1:8000` (start one as in the [README](../README.md#install-and-run), or on the stand-in with `uv run python -m decisio.serve.vllm_engine --backend hf --model Qwen/Qwen3-0.6B-Base`) and asks the same two questions about one ticket: which queue (a choice) and whether it is urgent (a yes/no).

**The base URL is not the same for every client.** Some append `/v1` themselves and some do not; a wrong one answers 404.

| Client | The base URL you give | Needs a key |
| --- | --- | --- |
| LangChain (`langchain-typesafe`) | `http://host:8000` (the class appends `/v1`) | yes, any string |
| Vercel AI SDK (`@ai-sdk/typesafe-ai`) | `http://host:8000/v1` | yes, any string |
| n8n (HTTP Request node) | the full URL, `http://host:8000/v1/systemone` | no |
| TanStack AI (`@tanstack/ai-typesafe`) | `http://host:8000` | yes, any string |
| Pipecat (`pipecat-ai[jev]`) | `http://host:8000` | yes, any string |

The model name each client sends (`jev-latest` by default in the TypeSafe clients) is ignored by the server and echoed back as the response's `model`.
What each client drops from the answer (the `x-decisio-*` headers, `unknown_probability` and `abstained`) is in the integration records.

## LangChain

`pip install langchain-typesafe==0.0.1a3` (checked at that version; it prints a beta warning).

```python
from langchain_typesafe import Choice, Noul, NoulCriteria, TypeSafeClassifier

clf = TypeSafeClassifier(base_url="http://127.0.0.1:8000", api_key="not-used")
result = clf.invoke(
    {
        "state": {"message": "My card was charged twice."},
        "questions": {
            "queue": Choice(instructions="Which queue?", criteria={"billing": "Payments", "access": "Login problems"}),
            "urgent": Noul(instructions="Is it urgent?", criteria=NoulCriteria(true="needs an answer today", false="can wait")),
        },
    }
)
print(result.answers["queue"].choice, result.answers["urgent"].noul)
```

Output: `billing 0.43047947436151557` (the stand-in's numbers).
`result.answers[name]` is a `ChoiceAnswer` (`choice`, `probabilities`, `confidence`) or a `NoulAnswer` (`noul`); a malformed question is refused by the client before it reaches the server, and a 422 from the server is raised as `TypeSafeUnprocessableEntityError`.

## Vercel AI SDK

`npm install ai@7.0.131 @ai-sdk/typesafe-ai@3.0.15` (Node 22).
`experimental_decide` is the current name; `experimental_evaluate` is a deprecated alias of it.

```js
import { experimental_decide as decide } from "ai";
import { createTypeSafeAi } from "@ai-sdk/typesafe-ai";

const typesafe = createTypeSafeAi({ baseURL: "http://127.0.0.1:8000/v1", apiKey: "not-used" });
const result = await decide({
  model: typesafe.decisionModel("decisio"),
  state: { message: "My card was charged twice." },
  questions: {
    queue: { type: "choice", instructions: "Which queue?", criteria: { billing: "Payments", access: "Login problems" } },
    urgent: { type: "boolean", instructions: "Is it urgent?", criteria: { true: "needs an answer today", false: "can wait" } },
  },
});
console.log(result.answers.queue, result.answers.urgent);
```

Output: the choice answer `{ type: 'choice', choice: 'billing', probabilities: {...} }` and `{ type: 'boolean', probability: 0.43047947436151557 }`.
`confidence` is under `result.providerMetadata.typesafe.confidence`.

## n8n

n8n's built-in HTTP Request node posts the request to the server; no community node is needed.
Import this workflow (Workflows, Import from File; or `n8n import:workflow --input=workflow.json` with the workflow wrapped in a list and given an `id`), change the URL if the server is not on port 8000, and press Execute.
Checked in n8n 2.35.7: the workflow ran to `success` and the node's output held `queue.choice: "billing"` and `urgent.noul: 0.43...`.

```json
{
  "name": "decisio: ask a question about a text",
  "nodes": [
    {
      "parameters": {},
      "id": "11111111-1111-4111-8111-111111111111",
      "name": "Manual trigger",
      "type": "n8n-nodes-base.manualTrigger",
      "typeVersion": 1,
      "position": [0, 0]
    },
    {
      "parameters": {
        "method": "POST",
        "url": "http://127.0.0.1:8000/v1/systemone",
        "sendBody": true,
        "specifyBody": "json",
        "jsonBody": "{\n  \"model\": \"decisio\",\n  \"state\": {\"message\": \"My card was charged twice.\"},\n  \"questions\": {\n    \"queue\": {\"type\": \"choice\", \"instructions\": \"Which queue?\", \"criteria\": {\"billing\": \"Payments\", \"access\": \"Login problems\"}},\n    \"urgent\": {\"type\": \"noul\", \"instructions\": \"Is it urgent?\", \"criteria\": {\"true\": \"needs an answer today\", \"false\": \"can wait\"}}\n  }\n}",
        "options": {}
      },
      "id": "22222222-2222-4222-8222-222222222222",
      "name": "Ask decisio",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.2,
      "position": [240, 0]
    }
  ],
  "connections": {
    "Manual trigger": {"main": [[{"node": "Ask decisio", "type": "main", "index": 0}]]}
  },
  "active": false,
  "settings": {}
}
```

The community node `n8n-nodes-jev` 0.2.3 also works against decisio (its credential's Base URL is the server's, without `/v1`); the HTTP Request node is shown because it needs nothing installed and its request body is the wire format itself.

## TanStack AI

`npm install @tanstack/ai@0.65.1 @tanstack/ai-typesafe@0.1.8` (Node 22).

```js
import { boolean, choice, decide } from "@tanstack/ai";
import { createTypesafeDecider } from "@tanstack/ai-typesafe";

const adapter = createTypesafeDecider("decisio", "not-used", { baseURL: "http://127.0.0.1:8000" });
const result = await decide({
  adapter,
  state: { message: "My card was charged twice." },
  questions: {
    queue: choice({ instructions: "Which queue?", options: { billing: "Payments", access: "Login problems" } }),
    urgent: boolean({ instructions: "Is it urgent?", criteria: { true: "needs an answer today", false: "can wait" } }),
  },
});
console.log(result.queue.value, result.urgent.probability);
```

Output: `billing 0.43047947436151557`.
Each result carries `value`, `probability`, `confidence` and `probabilities`; `meta.usage` maps the server's `usage`.
`typesafeDecider(model)` reads the key from `TYPESAFE_API_KEY` and has no base-URL variable, so use `createTypesafeDecider` with `baseURL` as above.

## Pipecat

`pip install "pipecat-ai[jev]==1.12.0"` (it installs without the audio extras).

```python
import asyncio

from pipecat.classifiers.base_classifier import ChoiceQuestion, YesNoQuestion
from pipecat.classifiers.jev.classifier import JevClassifier


async def main():
    clf = JevClassifier(api_key="not-used", base_url="http://127.0.0.1:8000")
    state = {"message": "My card was charged twice."}
    queue = await clf.choice(state, {"queue": ChoiceQuestion(instructions="Which queue?", options={"billing": "Payments", "access": "Login problems"})})
    urgent = await clf.yes_no(state, {"urgent": YesNoQuestion(instructions="Is it urgent?", yes="needs an answer today", no="can wait")})
    print(queue["queue"], urgent["urgent"])
    await clf.cleanup()


asyncio.run(main())
```

Output: `choice='billing' probabilities={...} confidence=...` and the yes/no result.
A 422 from the server is reported as `HTTP 422` and the server's detail is dropped by the client; the default timeout is 10 seconds, which the first question after a start can exceed on a slow card.

## Not checked here

LangChain's, the AI SDK's, TanStack's and Pipecat's snippets above are the shortest calls that worked; the integration records have the checks for errors, timeouts, the three question types together and the headers.
Registered tasks (`POST /v1/tasks`) and certified answers are decisio's own routes: none of these clients calls them.
