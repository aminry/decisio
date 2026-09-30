# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The names decisio puts on the wire and in its records, and the earlier `rlcd` spellings it still reads.

The package was extracted from a private repository named `rlcd`, whose server wrote `x-rlcd-*` headers, `rlcd-*/1`
record formats and the served name `rlcd-qwen3.6-35b-a3b-letters`; the records in runs/ and task files exported before
the rename carry those spellings. decisio writes only its own names and reads both:

  headers        x-decisio-debug and x-decisio-route are read, with x-rlcd-debug and x-rlcd-route as fallbacks;
                 responses carry x-decisio-server-ms, x-decisio-route and x-decisio-tasks
  record formats decisio-<kind>/1 is written; decisio-<kind>/1 and rlcd-<kind>/1 are both accepted on load
  served name    decisio-qwen3.6-35b-a3b-letters is the default; a task fitted under the old default name is treated as
                 fitted under the new one (`same_fingerprint`)

Standard library only.
"""
from __future__ import annotations

import json

PREFIX, LEGACY_PREFIX = "decisio", "rlcd"
SERVED_NAME = "decisio-qwen3.6-35b-a3b-letters"
LEGACY_SERVED_NAME = "rlcd-qwen3.6-35b-a3b-letters"
DEBUG_KEY = "decisio_debug"                  # the response body's debug field (with --debug-readout)


def header(name: str) -> str:
    """The response header decisio writes, e.g. header("route") == "x-decisio-route"."""
    return f"x-{PREFIX}-{name}"


def request_header(headers, name: str):
    """A request header's value under decisio's name, else under the legacy one, else None."""
    value = headers.get(f"x-{PREFIX}-{name}")
    return value if value is not None else headers.get(f"x-{LEGACY_PREFIX}-{name}")


def record_format(kind: str) -> str:
    """The format decisio writes for a record kind: "task", "task-prior", "intent-head" or "abstention"."""
    return f"{PREFIX}-{kind}/1"


def is_format(value, kind: str) -> bool:
    return value in (f"{PREFIX}-{kind}/1", f"{LEGACY_PREFIX}-{kind}/1")


def check_format(record: dict, kind: str, what: str) -> None:
    """Refuse a record that is not of this kind (in either spelling); a record without a format is refused too."""
    got = record.get("format") if isinstance(record, dict) else None
    if not is_format(got, kind):
        raise ValueError(f"{what}: format {got!r} is not {record_format(kind)!r} (or its earlier spelling "
                         f"{LEGACY_PREFIX}-{kind}/1)")


def canonical_fingerprint(fingerprint):
    """A task store fingerprint with the legacy default served name read as the current default."""
    try:
        d = json.loads(fingerprint)
    except (TypeError, ValueError):
        return fingerprint
    if not isinstance(d, dict):
        return fingerprint
    if d.get("served_name") == LEGACY_SERVED_NAME:
        d["served_name"] = SERVED_NAME
    return json.dumps(d, sort_keys=True)


def same_fingerprint(a, b) -> bool:
    return a == b or canonical_fingerprint(a) == canonical_fingerprint(b)
