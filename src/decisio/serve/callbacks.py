# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Signed callbacks: the server POSTs a JSON body to a URL a task's owner registered (docs/tasks.md, "Average error
(FDR), opt-in": the marks of each closed window).

A callback URL is https, with no user or password in it, and its host must resolve only to public addresses: a
loopback, private, link-local, multicast, reserved or unspecified address is refused, at registration and again at
every delivery, and the connection goes to the address that was checked (the TLS certificate is still checked
against the host name), so a DNS answer that changes between the check and the connection (rebinding) cannot reach
an internal host. Redirects are not followed: a 3xx is a failed delivery.

The body is signed with the task's callback secret: `X-Decisio-Signature: t=<seconds>,v1=<hex>`, the hex being
HMAC-SHA256 under the secret of `<t>.` followed by the body's bytes (`sign`, `verify`).
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import ipaddress
import secrets
import socket
import ssl
import time
from urllib.parse import urlsplit

TIMEOUT = 10.0  # seconds, to connect and for the answer
MAX_BODY = 8 * 1024 * 1024  # a body larger than this is not sent (a window of 250 marks is far smaller)


class CallbackError(ValueError):
    """A callback URL that is refused, or a delivery that could not be made."""


def new_secret() -> str:
    """A callback secret: 32 random bytes as hex (shown once, when it is made)."""
    return secrets.token_hex(32)


def check_url(url) -> str:
    """The URL if it may be a callback (https, a host, no user or password, no fragment), else CallbackError. Its
    host's addresses are checked separately (`public_addresses`), here at registration and at every delivery."""
    if not isinstance(url, str) or not url or len(url) > 2048:
        raise CallbackError("callback.url: an https URL")
    try:
        u = urlsplit(url)
        port = u.port
    except ValueError as e:
        raise CallbackError(f"callback.url: not a URL ({e})") from None
    if u.scheme != "https":
        raise CallbackError("callback.url: https only")
    if not u.hostname:
        raise CallbackError("callback.url: no host")
    if u.username is not None or u.password is not None:
        raise CallbackError("callback.url: no user or password in the URL (the body is signed instead)")
    if u.fragment:
        raise CallbackError("callback.url: no fragment")
    if port is not None and not 1 <= port <= 65535:
        raise CallbackError("callback.url: a port from 1 to 65535")
    return url


def _public(ip: str) -> bool:
    a = ipaddress.ip_address(ip.split("%")[0])
    if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped is not None:
        a = a.ipv4_mapped
    return a.is_global and not (a.is_multicast or a.is_reserved or a.is_loopback or a.is_link_local or a.is_private)


def resolve(host: str, port: int) -> list[str]:
    """The host's addresses (A and AAAA)."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        raise CallbackError(f"callback: {host!r} does not resolve ({e})") from None
    return list(dict.fromkeys(i[4][0] for i in infos))


def public_addresses(url: str) -> list[str]:
    """The URL's host's addresses, all public, else CallbackError: one address that is not public refuses the host
    (an address literal too)."""
    u = urlsplit(check_url(url))
    addrs = resolve(u.hostname, u.port or 443)
    if not addrs:
        raise CallbackError(f"callback: {u.hostname!r} has no address")
    bad = [a for a in addrs if not _public(a)]
    if bad:
        raise CallbackError(f"callback: {u.hostname!r} resolves to an address that is not public ({bad[0]}): refused")
    return addrs


def sign(secret: str, t: int, body: bytes) -> str:
    """The signature header's value: `t=<t>,v1=<hex HMAC-SHA256 of '<t>.' + body under the secret>`."""
    mac = hmac.new(secret.encode(), f"{int(t)}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={int(t)},v1={mac}"


def verify(secret: str, header: str, body: bytes, now: float | None = None, tolerance: float = 300.0) -> bool:
    """What a receiver checks: the signature matches the body under the secret, and `t` is within `tolerance` seconds
    of now (a replayed old delivery fails; the idempotency key catches a repeated one)."""
    try:
        parts = dict(p.split("=", 1) for p in header.split(","))
        t = int(parts["t"])
    except (ValueError, KeyError):
        return False
    if abs((time.time() if now is None else now) - t) > tolerance:
        return False
    return hmac.compare_digest(sign(secret, t, body), f"t={t},v1={parts.get('v1', '')}")


class _PinnedHTTPS(http.client.HTTPSConnection):
    """An HTTPS connection to one checked address, the TLS certificate checked against the URL's host name."""

    def __init__(self, host: str, port: int, address: str, timeout: float, context: ssl.SSLContext):
        super().__init__(host, port, timeout=timeout, context=context)
        self._address = address
        self._ctx = context

    def connect(self) -> None:
        sock = socket.create_connection((self._address, self.port), self.timeout)
        self.sock = self._ctx.wrap_socket(sock, server_hostname=self.host)


def post(url: str, body: bytes, headers: dict, timeout: float = TIMEOUT, context: ssl.SSLContext | None = None) -> int:
    """POST the body to the URL, at an address checked just now; returns the HTTP status (a 3xx is not followed).
    Raises CallbackError when the host is refused or the connection fails."""
    if len(body) > MAX_BODY:
        raise CallbackError(f"callback: body of {len(body)} bytes is over {MAX_BODY}")
    addrs = public_addresses(url)
    u = urlsplit(url)
    path = (u.path or "/") + (f"?{u.query}" if u.query else "")
    conn = _PinnedHTTPS(u.hostname, u.port or 443, addrs[0], timeout, context or ssl.create_default_context())
    try:
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json", **headers})
        resp = conn.getresponse()
        resp.read(65536)
        return int(resp.status)
    except (OSError, http.client.HTTPException) as e:
        raise CallbackError(f"callback: {type(e).__name__}: {e}") from None
    finally:
        conn.close()
