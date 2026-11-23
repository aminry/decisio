# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The average-error marks callback (decisio.serve.callbacks, SystemOne.deliver_marks, the callback routes).

K1  callback URLs: https only, no user or password, no fragment; a host that resolves to any address that is not
    public (loopback, private, link-local, unique-local, an IPv4-mapped private address) is refused, a public one kept
K2  the signature: t and an HMAC-SHA256 of '<t>.' + body under the secret; a changed body, an old t or a malformed
    header fails
K3  a delivery connects to the address checked just now (the TLS name is the host), and a host that resolves to a
    private address at delivery (DNS rebinding) is refused before any connection
K4  through the server: the secret is shown once at registration (never in the listing or the status); each closed
    window is POSTed once, signed, with an idempotency key, its marks as the export gives them; it is recorded and not
    sent again
K5  retries: a failed delivery (5xx, a redirect, a refused host) is tried again after the backoff, then given up after
    the last try and recorded; later windows are still delivered
K6  the routes: PUT sets a URL with a new secret, rotate makes a new secret (deliveries then verify only with it),
    DELETE turns it off; refused URLs are 422s, a task not in the mode or with no callback a 409
K8  a delivery has one deadline: a receiver that answers a byte at a time is cut there, and so is a slow name lookup
K7  registration refuses a callback outside the average-error mode, an http URL and a private host

  uv run pytest -q tests/unit/test_labels_fdr_callback.py
"""

import json
import socket
import time

import pytest

from decisio.serve import callbacks as CB
from decisio.serve import labels as LB

PUBLIC = "93.184.216.34"
DAY = 86_400


def _resolving(monkeypatch, table):
    monkeypatch.setattr(CB, "resolve", lambda host, port, timeout=None: list(table.get(host, [PUBLIC])))


def test_k1_callback_urls():
    for bad in ("http://example.com/h", "https://user:pw@example.com/h", "https://example.com/h#x", "https:///h", "",
                "ftp://example.com", 42):  # fmt: skip
        with pytest.raises(CB.CallbackError):
            CB.check_url(bad)
    assert CB.check_url("https://example.com:8443/hook?x=1") == "https://example.com:8443/hook?x=1"
    for port in (22, 25, 80, 8080):  # a public host's other services are no POST target
        with pytest.raises(CB.CallbackError, match="port 443 or 8443"):
            CB.check_url(f"https://example.com:{port}/hook")


def test_k1_hosts_must_resolve_to_public_addresses_only(monkeypatch):
    for addrs in (["127.0.0.1"], ["10.0.0.5"], ["192.168.1.2"], ["169.254.169.254"], ["::1"], ["fc00::1"],
                  ["fe80::1"], ["::ffff:10.0.0.1"], ["0.0.0.0"], ["224.0.0.1"], [PUBLIC, "10.0.0.1"]):  # fmt: skip
        _resolving(monkeypatch, {"example.com": addrs})
        with pytest.raises(CB.CallbackError, match="not public"):
            CB.public_addresses("https://example.com/h")
    # an IPv4 address inside an IPv6 one is held to the same rule, whatever the standard library's tables say
    # NAT64, local NAT64, 6to4, IPv4-mapped, Teredo
    for inner_private in ("64:ff9b::a00:1", "64:ff9b:1::a00:1", "2002:a00:1::1", "::ffff:127.0.0.1",
                          "2001:0:4136:e378:8000:63bf:3fff:fdd2"):  # fmt: skip
        assert not CB._public(inner_private), inner_private
    assert CB._public("64:ff9b::5db8:d822") and CB._public("2002:5db8:d822::1")  # carrying 93.184.216.34
    _resolving(monkeypatch, {"example.com": [PUBLIC, "2606:2800:220:1::1"]})
    assert CB.public_addresses("https://example.com/h") == [PUBLIC, "2606:2800:220:1::1"]


def test_k2_the_signature():
    body, secret, now = b'{"window": 0}', "s" * 64, time.time()
    header = CB.sign(secret, int(now), body)
    assert header.startswith(f"t={int(now)},v1=") and CB.verify(secret, header, body, now=now)
    assert not CB.verify(secret, header, body + b" ", now=now)  # a changed body
    assert not CB.verify("x" * 64, header, body, now=now)  # another secret
    assert not CB.verify(secret, header, body, now=now + 301)  # an old delivery
    assert not CB.verify(secret, "garbage", body, now=now)


def test_k3_the_connection_goes_to_the_checked_address(monkeypatch):
    seen = []

    def connect(address, timeout):
        seen.append(address)
        raise OSError("stop here")

    monkeypatch.setattr(socket, "create_connection", connect)
    _resolving(monkeypatch, {"hooks.example.com": [PUBLIC]})
    with pytest.raises(CB.CallbackError, match="stop here"):
        CB.post("https://hooks.example.com/h", b"{}", {})
    assert seen == [(PUBLIC, 443)]  # the address checked, never a second lookup of the name
    seen.clear()
    _resolving(monkeypatch, {"hooks.example.com": ["10.1.2.3"]})  # the name now points inside (rebinding)
    with pytest.raises(CB.CallbackError, match="not public"):
        CB.post("https://hooks.example.com/h", b"{}", {})
    assert seen == []  # refused before any connection


def _server(monkeypatch, url="https://hooks.example.com/decisio", status=200):
    from test_labels_served import make
    from test_noul_certified import noul_examples

    monkeypatch.setattr(LB, "FDR_SETTLE", 0)
    _resolving(monkeypatch, {"inside.example.com": ["10.0.0.9"]})
    client, so = make()
    posts = []
    replies = [status] if isinstance(status, int) else list(status)

    def poster(u, body, headers):
        posts.append((u, body, headers))
        r = replies.pop(0) if len(replies) > 1 else replies[0]
        if isinstance(r, Exception):
            raise r
        return r

    so._callback_post = poster
    r = client.post("/v1/tasks", json={"id": "angry", "examples": noul_examples(120),
                                       "labels": {"audit_pi": 1.0, "due_days": 1},
                                       "certify": {"mode": "average_error", "alpha": 0.1, "window": 30,
                                                   "callback": {"url": url}}})  # fmt: skip
    assert r.status_code == 200, r.text
    return client, so, posts, r.json()["average_error"]


def _windows(client, so, n_windows):
    """Serve calibration answers, label them, move past due_days, serve enough for n windows, run the labels turns."""
    from test_labels_served import ask

    rids = [ask(client, f"yes:{0.6 + 0.01 * i:.3f}:cal {i}")[0]["request_id"] for i in range(30)]
    client.post("/v1/tasks/angry/labels", json={"labels": [{"request_id": r, "answer": "yes"} for r in rids]})
    log = so.labels_logs["angry"]
    real = log.clock
    log.clock = lambda: real() + DAY + 60
    for i in range(30 * (n_windows - 1) + 5):
        ask(client, f"yes:{0.9 + 0.002 * (i % 40):.3f}:m {i}")
    for _ in range(n_windows + 1):
        so.flush_labels(force=True)
        so.wait_callbacks()
    return log


def test_k4_one_signed_delivery_per_window(monkeypatch):
    client, so, posts, ae = _server(monkeypatch)
    cb = ae["callback"]
    secret = cb["secret"]
    assert len(secret) == 64 and cb["url"] == "https://hooks.example.com/decisio" and cb["shown_once"]
    assert secret not in client.get("/v1/tasks").text and secret not in client.get("/v1/tasks/angry").text
    log = _windows(client, so, 2)
    assert len(log.fdr_windows) == 2 and len(posts) == 2
    marks = [
        json.loads(x) for x in client.get("/v1/tasks/angry/labels/export", params={"view": "marks"}).text.splitlines()
    ]
    for i, (url, body, headers) in enumerate(posts):
        assert url == "https://hooks.example.com/decisio" and CB.verify(secret, headers["X-Decisio-Signature"], body)
        assert headers["X-Decisio-Event"] == "average_error.window"
        assert headers["Idempotency-Key"] == f"angry/average_error/{log.fdr_windows[i]['id']}"
        p = json.loads(body)
        assert p["window"] == i and p["task"] == "angry" and p["promise"] == LB.fdr_sentence(0.1)
        assert p["marks"] == [m for m in marks if m["window"] == i] and p["size"] == 30
    so.flush_labels(force=True)
    so.wait_callbacks()
    assert len(posts) == 2  # recorded, never sent again
    st = client.get("/v1/tasks/angry/labels/status")
    assert secret not in st.text
    assert st.json()["average_error"]["callback"] == {"url": "https://hooks.example.com/decisio", "delivered": 2,
                                                      "given_up": [], "retrying": [], "waiting": 0}  # fmt: skip


def test_k5_retries_then_given_up(monkeypatch):
    fails = [500, 302, CB.CallbackError("callback: refused"), 503, 500, 500]  # the six tries of window 0
    client, so, posts, _ = _server(monkeypatch, status=[*fails, 200])
    log = _windows(client, so, 1)
    st = so._callback_state["angry"]
    assert len(posts) == 1 and st[0]["attempts"] == 1 and st[0]["next_try"] > time.time() + 20  # backoff, not now
    errors = []
    for _ in range(5):  # the backoff passes (moved by hand) and it is tried again
        st[0]["next_try"] = 0.0
        so.flush_labels(force=True)
        so.wait_callbacks()
        errors.append(st.get(0, {}).get("error"))
    assert len(posts) == 6 and log.fdr_deliveries[0]["ok"] is False and log.fdr_deliveries[0]["attempts"] == 6
    assert "redirects are not followed" in errors[0] and "refused" in errors[1]
    status = client.get("/v1/tasks/angry/labels/status").json()["average_error"]["callback"]
    assert status["given_up"] == [0] and status["delivered"] == 0
    from test_labels_served import ask

    for i in range(30):  # a later window is still delivered
        ask(client, f"yes:0.95:late {i}")
    so.flush_labels(force=True)
    so.wait_callbacks()
    assert len(posts) == 7 and log.fdr_deliveries[1]["ok"] is True


def test_k6_set_rotate_delete(monkeypatch):
    client, so, posts, ae = _server(monkeypatch)
    old = ae["callback"]["secret"]
    r = client.post("/v1/tasks/angry/labels/callback/rotate")
    assert r.status_code == 200, r.text
    new = r.json()["callback"]["secret"]
    assert new != old and len(new) == 64
    log = _windows(client, so, 1)
    ((_, body, headers),) = posts
    assert CB.verify(new, headers["X-Decisio-Signature"], body) and not CB.verify(old, headers["X-Decisio-Signature"],
                                                                                   body)  # fmt: skip
    assert client.put("/v1/tasks/angry/labels/callback", json={"url": "http://hooks.example.com/x"}).status_code == 422
    r = client.put("/v1/tasks/angry/labels/callback", json={"url": "https://inside.example.com/x"})
    assert r.status_code == 422 and "not public" in r.text
    r = client.put("/v1/tasks/angry/labels/callback", json={"url": "https://other.example.com/x"})
    assert r.status_code == 200 and r.json()["callback"]["secret"] not in (old, new)
    assert so.task_store.snapshot("angry")["labels"]["fdr"]["callback"]["url"] == "https://other.example.com/x"
    assert client.delete("/v1/tasks/angry/labels/callback").json() == {"callback": None}
    from test_labels_served import ask

    for i in range(30):
        ask(client, f"yes:0.95:after {i}")
    so.flush_labels(force=True)
    so.wait_callbacks()
    assert len(log.fdr_windows) == 2 and len(posts) == 1  # off: nothing more is sent
    assert "callback" not in client.get("/v1/tasks/angry/labels/status").json()["average_error"]
    assert client.post("/v1/tasks/angry/labels/callback/rotate").status_code == 409
    from test_noul_certified import noul_examples

    client.post("/v1/tasks", json={"id": "calm", "examples": noul_examples(120), "labels": True,
                                   "instructions": "Is the customer calm?"})  # fmt: skip
    assert client.put("/v1/tasks/calm/labels/callback", json={"url": "https://hooks.example.com/x"}).status_code == 409


def test_k7_registration_refusals(monkeypatch):
    from test_labels_served import make
    from test_noul_certified import noul_examples

    _resolving(monkeypatch, {"inside.example.com": ["192.168.0.2"]})
    client, _ = make()
    base = {"id": "x", "examples": noul_examples(120), "labels": True}
    for certify, why in (({"callback": {"url": "https://hooks.example.com/h"}}, "average_error"),
                         ({"mode": "average_error", "callback": {"url": "http://hooks.example.com/h"}}, "https"),
                         ({"mode": "average_error", "callback": {"url": "https://inside.example.com/h"}}, "not public"),
                         ({"mode": "average_error", "callback": "https://hooks.example.com/h"}, "url")):  # fmt: skip
        r = client.post("/v1/tasks", json={**base, "certify": certify})
        assert r.status_code == 422 and why in r.text, (certify, r.text)


def test_k8_a_dribbling_receiver_is_cut_at_the_deadline(monkeypatch):
    import http.client
    import threading

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    stop = threading.Event()

    def dribble():
        c, _ = srv.accept()
        try:
            c.recv(65536)
            for ch in b"HTTP/1.1 200 OK\r\nX-Slow: " + b"a" * 1000:  # one byte every 0.1 s: under the socket timeout
                if stop.is_set():
                    break
                c.sendall(bytes([ch]))
                time.sleep(0.1)
        except OSError:
            pass
        finally:
            c.close()

    threading.Thread(target=dribble, daemon=True).start()

    class Plain(http.client.HTTPConnection):  # the pinned connection, without TLS, to the local receiver
        def __init__(self, host, p, address, timeout, context):
            super().__init__(host, p, timeout=timeout)

        def connect(self):
            self.sock = socket.create_connection(("127.0.0.1", port), self.timeout)

    monkeypatch.setattr(CB, "_PinnedHTTPS", Plain)
    _resolving(monkeypatch, {})
    t0 = time.monotonic()
    with pytest.raises(CB.CallbackError, match="no answer within 1 s"):
        CB.post("https://slow.example.com/h", b"{}", {}, timeout=0.5, deadline=1.0)
    assert time.monotonic() - t0 < 2.5
    stop.set()
    srv.close()
    monkeypatch.undo()  # the real resolve again, over a name lookup that hangs
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: time.sleep(3))
    t0 = time.monotonic()
    with pytest.raises(CB.CallbackError, match="did not resolve within"):
        CB.resolve("hang.example.com", 443, timeout=0.3)
    assert time.monotonic() - t0 < 1.5
