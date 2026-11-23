# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Signed callbacks: the server POSTs a JSON body to a URL a task's owner registered (docs/tasks.md, "Average error
(FDR), opt-in": the marks of each closed window).

A callback URL is https on port 443 or 8443, with no user or password in it, and its host must resolve only to public
addresses: a loopback, private, link-local, multicast, reserved or unspecified address is refused (and an IPv4 address
carried in an IPv6 one, NAT64, 6to4, Teredo or IPv4-mapped, by the same rule), at registration and again at every
delivery, and the connection goes to the address that was checked (the TLS certificate is still checked against the
host name), so a DNS answer that changes between the check and the connection (rebinding) cannot reach an internal
host. Redirects are not followed: a 3xx is a failed delivery. A delivery has one deadline, `DEADLINE` seconds from the
name lookup to the answer's status line, whatever the receiver does; the answer's body is never read.

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
import threading
import time
from urllib.parse import urlsplit

TIMEOUT = 10.0  # seconds, for each socket operation (connect, each read)
DEADLINE = 30.0  # seconds for a whole delivery: the name lookup, the connection, the request and the status line
PORTS = (443, 8443)  # the ports a callback may use
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
    if (port or 443) not in PORTS:
        raise CallbackError(f"callback.url: port 443 or 8443 (the port of an https callback), not {port}")
    return url


_NAT64 = (ipaddress.ip_network("64:ff9b::/96"), ipaddress.ip_network("64:ff9b:1::/48"))


def _embedded(a: ipaddress.IPv6Address) -> list[ipaddress.IPv4Address]:
    """The IPv4 addresses an IPv6 address carries (IPv4-mapped, NAT64, 6to4, Teredo server and client): each is checked
    by the same rule, whatever the standard library's tables say of the IPv6 prefix."""
    out = []
    if a.ipv4_mapped is not None:
        out.append(a.ipv4_mapped)
    if any(a in n for n in _NAT64):
        out.append(ipaddress.IPv4Address(int(a) & 0xFFFFFFFF))
    if a.sixtofour is not None:
        out.append(a.sixtofour)
    if a.teredo is not None:
        out.extend(a.teredo)
    return out


def _public(ip: str) -> bool:
    a = ipaddress.ip_address(ip.split("%")[0])
    if isinstance(a, ipaddress.IPv6Address):
        inner = _embedded(a)
        if inner:
            return all(_public(str(x)) for x in inner)
    return a.is_global and not (a.is_multicast or a.is_reserved or a.is_loopback or a.is_link_local or a.is_private)


def resolve(host: str, port: int, timeout: float = DEADLINE) -> list[str]:
    """The host's addresses (A and AAAA), within `timeout` seconds (the lookup runs on its own thread, since
    getaddrinfo has no timeout of its own)."""
    out: dict = {}

    def lookup():
        try:
            out["infos"] = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as e:
            out["error"] = e

    t = threading.Thread(target=lookup, name="decisio-callback-dns", daemon=True)
    t.start()
    t.join(max(0.0, timeout))
    if t.is_alive():
        raise CallbackError(f"callback: {host!r} did not resolve within {timeout:.0f} s")
    if "error" in out:
        raise CallbackError(f"callback: {host!r} does not resolve ({out['error']})")
    return list(dict.fromkeys(i[4][0] for i in out["infos"]))


def public_addresses(url: str, timeout: float = DEADLINE) -> list[str]:
    """The URL's host's addresses, all public, else CallbackError: one address that is not public refuses the host
    (an address literal too)."""
    u = urlsplit(check_url(url))
    addrs = resolve(u.hostname, u.port or 443, timeout)
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
        self.sock = sock  # held from the start, so the deadline's watchdog can shut it down at any step
        # the handshake after the wrapped socket is held (wrapping detaches the raw one): the watchdog can end it too
        self.sock = self._ctx.wrap_socket(sock, server_hostname=self.host, do_handshake_on_connect=False)
        self.sock.do_handshake()


def _cut(conn: http.client.HTTPConnection, fired: threading.Event) -> None:
    """The deadline's watchdog: shut the connection's socket down, so a blocked connect, send or read returns."""
    fired.set()
    sock = conn.sock
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    conn.close()


def post(url: str, body: bytes, headers: dict, timeout: float = TIMEOUT, context: ssl.SSLContext | None = None,
         deadline: float = DEADLINE) -> int:  # fmt: skip
    """POST the body to the URL, at an address checked just now; returns the HTTP status (a 3xx is not followed; the
    answer's body is not read). The whole delivery, from the name lookup to the status line, has `deadline` seconds,
    whatever the receiver does. Raises CallbackError when the host is refused, the connection fails or the deadline
    passes."""
    if len(body) > MAX_BODY:
        raise CallbackError(f"callback: body of {len(body)} bytes is over {MAX_BODY}")
    start = time.monotonic()
    addrs = public_addresses(url, timeout=deadline)
    left = deadline - (time.monotonic() - start)
    if left <= 0:
        raise CallbackError(f"callback: no answer within {deadline:.0f} s")
    u = urlsplit(url)
    path = (u.path or "/") + (f"?{u.query}" if u.query else "")
    conn = _PinnedHTTPS(
        u.hostname, u.port or 443, addrs[0], min(timeout, left), context or ssl.create_default_context()
    )
    fired = threading.Event()
    watchdog = threading.Timer(left, _cut, args=(conn, fired))
    watchdog.daemon = True
    watchdog.start()
    try:
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json", **headers})
        return int(conn.getresponse().status)
    except (OSError, ValueError, http.client.HTTPException) as e:
        if fired.is_set():
            raise CallbackError(f"callback: no answer within {deadline:.0f} s") from None
        raise CallbackError(f"callback: {type(e).__name__}: {e}") from None
    finally:
        watchdog.cancel()
        conn.close()
