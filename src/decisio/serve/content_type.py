# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Refuse a POST that is not JSON, before the server does any work for it.

A web page can make a visitor's browser send a POST to a server on that visitor's own machine (`http://localhost:8000`)
from any origin. The browser sends it without asking the server first, as long as the request is a "simple" one: its
Content-Type is `text/plain`, `application/x-www-form-urlencoded` or `multipart/form-data`. Anything else, including
`application/json`, makes the browser send a preflight `OPTIONS` first, which this server answers with 405, so the
POST is never sent. The page cannot read the answer, but the server has already done the work, and every route here
can cost an engine call. FastAPI parses a JSON body whatever the Content-Type says, so without this check `text/plain`
reaches the engine.

`JsonOnly` answers 415 to any POST whose media type is not `application/json` (parameters such as `; charset=utf-8` are
fine, the case is not significant), on every route, before the application sees the request and before its body is read.

The one exception is `multipart/form-data` on `POST /v1/systemone`, the encoding imajev's image clients send
(docs/api.md), when the request carries an `X-Decisio-Client` header (any value). A page's request cannot carry a custom
header unless the browser first asks the server's permission with a preflight `OPTIONS`, and this server answers every
preflight with 405 and sends no `Access-Control-Allow-*` header, so the browser never sends the POST. The header
proves nothing about the sender; it only has to be one a page cannot add. A proxy that answers preflights with
`Access-Control-Allow-Headers` for it re-opens the hole, as it would for JSON.

Standard library only.
"""

from __future__ import annotations

import json

JSON = "application/json"
MULTIPART = "multipart/form-data"
CLIENT_HEADER = b"x-decisio-client"
MESSAGE = (
    "unsupported content type: this server reads application/json only (a request in another type is refused before "
    "it is read, so that a web page cannot make it do work from a browser; docs/api.md)"
)
MULTIPART_MESSAGE = (
    "multipart/form-data needs an X-Decisio-Client header (any value): a web page cannot add one without a preflight "
    "this server refuses, so a request that has it did not come from a page's simple cross-site POST; docs/api.md"
)


def media_type(headers) -> str:
    """The Content-Type's media type, lower case, without its parameters; "" when there is none."""
    for key, value in headers:
        if key == b"content-type":
            return value.decode("latin-1").split(";")[0].strip().lower()
    return ""


class JsonOnly:
    """ASGI middleware: 415 for a POST that is not `application/json`."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        kind = media_type(scope["headers"])
        if kind == JSON:
            return await self.app(scope, receive, send)
        message = MESSAGE
        if kind == MULTIPART and scope["path"] == "/v1/systemone":
            if any(key == CLIENT_HEADER for key, _ in scope["headers"]):
                return await self.app(scope, receive, send)
            message = MULTIPART_MESSAGE
        body = json.dumps({"detail": message}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 415,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"accept-post", JSON.encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
