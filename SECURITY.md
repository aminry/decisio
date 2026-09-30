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
