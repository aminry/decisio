# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A request body over 32 MiB on `POST /v1/systemone` is a 413, before it is parsed.

System One refuses a body over 32 MiB (the JSON, base64 text and multipart framing included), and so does this route:
a client written against that limit sees the same status and error shape, and a server is protected from a body it
would otherwise read and spool to disk whole. The check is cheap and has two parts. A `Content-Length` over the limit is
refused at once, without reading a byte. A body with no `Content-Length` (chunked), or one that says less than it
sends, is counted as it arrives and refused when it passes the limit, so the form parser never sees more than the limit.

The limit is the request's, not an image's: the route's own image limits (20 MiB and 20 megapixels per image, the
context for the count) are unchanged and wider where they were.

Standard library only.
"""

from __future__ import annotations

import json

MAX_BODY_BYTES = 32 * 1024 * 1024
PATH = "/v1/systemone"
MESSAGE = "Request exceeds size limits."


class _TooLarge(Exception):
    pass


class BodyLimit:
    """ASGI middleware: 413 for a POST to `/v1/systemone` whose body is over `MAX_BODY_BYTES`."""

    def __init__(self, app, limit: int = MAX_BODY_BYTES):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != PATH:
            return await self.app(scope, receive, send)
        for key, value in scope["headers"]:
            if key == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break
                if declared > self.limit:
                    return await self._refuse(send)
        seen = 0
        started = False

        async def counted():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    raise _TooLarge
            return message

        async def watched(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counted, watched)
        except _TooLarge:
            if started:
                raise
            await self._refuse(send)

    async def _refuse(self, send):
        body = json.dumps({"detail": {"error_type": "api_usage_error", "message": MESSAGE}}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})
