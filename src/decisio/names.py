# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The names decisio writes on the wire and in its records, and the earlier spellings it still reads.

  headers        responses carry x-decisio-server-ms, x-decisio-route and x-decisio-tasks; requests may send
                 x-decisio-debug and x-decisio-route, or the earlier x-rlcd-debug and x-rlcd-route
  record formats decisio-<kind>/1 is written; decisio-<kind>/1 and rlcd-<kind>/1 are accepted on load (the records
                 in runs/ use the earlier spelling)
  served name    the default is decisio-qwen3.6-35b-a3b-letters; a task fitted under the earlier default,
                 rlcd-qwen3.6-35b-a3b-letters, is treated as fitted under the current one (`same_fingerprint`)

Standard library only.
"""

from __future__ import annotations

import json

PREFIX, LEGACY_PREFIX = "decisio", "rlcd"
SERVED_NAME = "decisio-qwen3.6-35b-a3b-letters"
LEGACY_SERVED_NAME = "rlcd-qwen3.6-35b-a3b-letters"
DEBUG_KEY = "decisio_debug"  # the response body's debug field (with --debug-readout)


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
        raise ValueError(
            f"{what}: format {got!r} is not {record_format(kind)!r} (or its earlier spelling {LEGACY_PREFIX}-{kind}/1)"
        )


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
    # an entry of servers that briefly carried --describe-options in the fingerprint; it now enters the task key of
    # the questions it changes (decisio.serve.tasks.task_key)
    d.pop("describe_options", None)
    return json.dumps(d, sort_keys=True)


def task_fingerprint(
    *,
    served_name,
    model,
    pad_to,
    pad_where,
    hide_index_keys,
    desnake_labels,
    pad_policy="always",
    prompt_format=None,
    base=None,
) -> str:
    """The task store's fingerprint: a registered task is valid only for the model and the rendering it was fitted
    under. `model` is the checkpoint's name (its source's, when a decisio repository serves a copy of it: decisio.hub).
    A key enters only when it is not the default (`pad_policy` other than always, `prompt_format` facts of a
    non-default format, `base` for a base other than Qwen), so the fingerprints of tasks registered under the defaults
    are unchanged. --describe-options is not here: it enters the task key of the questions it changes
    (decisio.serve.tasks.task_key)."""
    return json.dumps(
        {
            "served_name": served_name,
            "model": model,
            "pad_to": pad_to,
            "pad_where": pad_where,
            "hide_index_keys": hide_index_keys,
            "desnake_labels": desnake_labels,
            **({"pad_policy": pad_policy} if pad_policy != "always" else {}),
            **({"prompt_format": prompt_format} if prompt_format else {}),
            **({"base": base} if base else {}),
        },
        sort_keys=True,
    )


def same_fingerprint(a, b) -> bool:
    return a == b or canonical_fingerprint(a) == canonical_fingerprint(b)
