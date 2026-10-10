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

A decisio server is single-tenant.
Its prefix cache and registered tasks are shared by every caller, so response timing can reveal whether another caller recently sent the same text, and one caller's registered task answers another's identical question.
Run one server per trust boundary.

## vLLM advisories

vLLM advisories before 0.31.0 (GHSA-h3rc-6mm3-gc2m, GHSA-p92p-rxj5-7p2x, GHSA-4xqp-c3mv-qff7, GHSA-6cxc-2vcg-w5qc) are not reachable through decisio's routes: decisio uses vLLM as a library, mounts none of its HTTP routes and passes no request field to it as a parameter.
0.12.0 moves to vLLM 0.31.0, which fixes all four.
`tests/unit/test_vllm_exposure.py` fails when the source stops matching this.
