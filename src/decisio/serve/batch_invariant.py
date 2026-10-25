# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""vLLM's batch-invariant kernels, switched on by a flag or by a tuned repository's defaults (docs/running.md, "Batch
invariance").

vLLM 0.31.0 reads `VLLM_BATCH_INVARIANT=1` from its environment when it starts and then computes every row with
kernels whose result does not depend on what else is in the batch.
A board or a user runs only `--base <repository>`; they do not set an environment variable.
So the setting can be a tuned repository's `defaults.batch_invariant`, or the server's `--batch-invariant on|off`;
this module decides which one wins and sets the variable before the engine is built.

    resolve(flag, repository, environ) -> {"on": bool, "source": "flag" | "repository" | "environment" | "default"}

Precedence: the flag, then the repository, then off. The environment is not a source of its own that can disagree: a
variable that says the opposite of the flag or the repository is refused at start, so /health never misreports what the
engine runs. A variable set to 1 with nothing else saying anything turns it on, and is recorded as such.

Standard library only (the check that vLLM took it, `engine_state`, imports vLLM when there is one).
"""

from __future__ import annotations

VARIABLE = "VLLM_BATCH_INVARIANT"
CHOICES = ("on", "off")


def _env_value(environ) -> bool | None:
    """True / False for 1 / 0 in the environment, None when it is not set; anything else is refused."""
    raw = environ.get(VARIABLE)
    if raw is None or raw == "":
        return None
    if raw not in ("0", "1"):
        raise SystemExit(f"{VARIABLE}={raw!r}: vLLM reads 0 or 1; unset it or use --batch-invariant on|off")
    return raw == "1"


def resolve(flag: str | None, repository: bool | None, environ, backend: str = "vllm") -> dict:
    """Which setting the server runs, where it came from, and the variable set in `environ` before vLLM starts.
    `flag`: "on", "off" or None; `repository`: the tuned repository's `defaults.batch_invariant` or None. Raises
    SystemExit naming the conflict (a variable that says the opposite, a backend without vLLM's kernels)."""
    if flag is not None and flag not in CHOICES:
        raise SystemExit(f"--batch-invariant is on or off, not {flag!r}")
    env = _env_value(environ)
    if flag is not None:
        want, source = flag == "on", "flag"
    elif repository is not None:
        want, source = bool(repository), "repository"
    elif env is not None:
        want, source = env, "environment"
    else:
        want, source = False, "default"
    if backend != "vllm":
        if want and source != "environment":
            raise SystemExit(f"--batch-invariant is for the vLLM server, not --backend {backend}")
        return {"on": False, "source": "default"}
    if env is not None and env != want:
        who = {"flag": f"--batch-invariant {flag}", "repository": "the repository's defaults.batch_invariant"}[source]
        raise SystemExit(
            f"{VARIABLE}={int(env)} in the environment contradicts {who} ({'on' if want else 'off'}): "
            "unset the variable, or change the flag"
        )
    if want:
        environ[VARIABLE] = "1"
    return {"on": want, "source": source}


def engine_state() -> bool | None:
    """Whether the vLLM in this process reads the setting as on (it may have been imported before the variable was set,
    or be a build without the mode); None where vLLM is not importable here."""
    try:
        from vllm import envs
    except Exception:
        return None
    try:
        return bool(envs.VLLM_BATCH_INVARIANT)
    except Exception:
        return None


def start_line(report: dict) -> str:
    where = {
        "flag": "--batch-invariant",
        "repository": "the repository's defaults",
        "environment": f"{VARIABLE} in the environment",
        "default": "the default",
    }[report["source"]]
    return f"BATCH INVARIANT {'on' if report['on'] else 'off'} ({where})" + (
        "" if report.get("confirmed") is None else f"; vLLM reads it as {'on' if report['confirmed'] else 'off'}"
    )
