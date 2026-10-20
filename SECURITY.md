<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Security policy

## Reporting a vulnerability

Report security issues through GitHub's private vulnerability reporting on this repository ("Report a vulnerability" under the Security tab).
Do not open a public issue.

You will get an acknowledgement within 5 working days and an assessment within 15.
Fixes for confirmed issues are released as soon as they are ready, with credit to the reporter unless you ask otherwise.

## Supported versions

The latest release on `main` receives security fixes.
Older tags do not.

## Deployment notes

Decisio is a model server, not a hardened public endpoint.

- It is designed to run behind your own proxy on a private network. The server binds to `127.0.0.1` by default; do not expose it directly to the internet.
- There is no authentication on any route. Put authentication, rate limiting and request-size limits in the proxy.
- `POST /v1/tasks`, `POST /v1/tasks/import`, `DELETE /v1/tasks/{id}` and the abstention routes change server state. Restrict them to trusted callers.
- `--debug-readout` exposes raw model readouts; leave it off in production.
- The image route accepts image payloads from requests; size limits belong in the proxy.
- vLLM's own security guidance applies to the engine underneath: https://docs.vllm.ai/en/stable/usage/security/

### Cross-site requests from a web page

Any web page the user of a decisio server opens can make that user's browser send requests to `http://localhost:8000`.
The browser sends a "simple" cross-origin POST, one whose Content-Type is `text/plain`, `application/x-www-form-urlencoded` or `multipart/form-data`, without asking the server first, and the page cannot read the answer.
**Before 0.12.0 the server read a JSON body whatever the Content-Type said, so such a POST was carried out: a page could make the server spend engine time on any route, and could register or replace tasks (`POST /v1/tasks`, `POST /v1/tasks/import`, the abstention routes).**
From 0.12.0 every POST whose Content-Type is not `application/json` (parameters such as `; charset=utf-8` are allowed) is refused with 415 before the request body is read and before any engine work, on every route.
A request with a JSON body already needs the browser's preflight `OPTIONS`, which the server answers with 405, so the POST of a page on another origin is never sent.
The one exception is `multipart/form-data` on `POST /v1/systemone`, which imajev's image clients send: it is accepted only when the request carries an `X-Decisio-Client` header (any value).
A page cannot add a custom header to a cross-site request unless the browser first sends a preflight `OPTIONS` and the server approves it; this server answers every preflight with 405 and sends no `Access-Control-Allow-*` header, so the POST is never sent.
The header says nothing about who sent the request; it only has to be one a page cannot add.
A proxy in front of the server that answers preflights with `Access-Control-Allow-Origin` and `Access-Control-Allow-Headers` re-opens the problem for JSON and multipart alike, so do not configure CORS on it for this server.
With an earlier release, have the proxy refuse a POST whose Content-Type is not `application/json`.
`tests/unit/test_json_only.py` pins the behaviour.

A decisio server is single-tenant.
Its prefix cache and registered tasks are shared by every caller, so response timing can reveal whether another caller recently sent the same text, and one caller's registered task answers another's identical question.
Run one server per trust boundary.

## vLLM advisories

vLLM advisories before 0.31.0 (GHSA-h3rc-6mm3-gc2m, GHSA-p92p-rxj5-7p2x, GHSA-4xqp-c3mv-qff7, GHSA-6cxc-2vcg-w5qc) are not reachable through decisio's routes: decisio uses vLLM as a library, mounts none of its HTTP routes and passes no request field to it as a parameter.
0.12.0 moves to vLLM 0.31.0, which fixes all four.
One call generates more than a token and sets no allowed-token mask: think when unsure (`--think`) lets the served model write reasoning.
Its length is a server setting (`--think-budget`, at most 4,096 tokens), not a request field; a request can only say `think: true` or `false`.
`tests/unit/test_vllm_exposure.py` fails when the source stops matching this.
