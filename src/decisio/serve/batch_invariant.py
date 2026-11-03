# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""vLLM's batch-invariant kernels, switched on by a flag or by a tuned repository's defaults (docs/running.md, "Batch
invariance").

vLLM 0.31.0 reads `VLLM_BATCH_INVARIANT=1` from its environment when it starts and then computes every row with
kernels whose result does not depend on what else is in the batch.
A board or a user runs only `--base <repository>`; they do not set an environment variable.
So the setting can be a tuned repository's `defaults.batch_invariant`, the server's `--batch-invariant on|off`, or the
base: it is on by default for a bf16 base (the plain 12B, the 31B in bf16 and a tuned bf16 repository of either),
where it was measured to make a read from the prefix cache give the same bits as a read with the cache off; FP8 bases,
adapters applied at run time, the image route and other bases keep it off until it is measured there. This module
decides which one wins and sets the variable before the engine is built.

    resolve(flag, repository, environ, backend, base_default) -> {"on": bool, "source": ...}
    base_default(precision, fp8_weights, backend, adapters, image_model, mode) -> bool
    merge_engine(engine_json, max_model_len) -> engine_json with max_num_batched_tokens covering the context

Precedence: the flag, then the repository, then the environment, then the base, then off. A variable that says the
opposite of the flag or the repository is refused at start, so /health never misreports what the engine runs; a
variable alone (1 or 0) decides when neither says anything, and is recorded as such, so `VLLM_BATCH_INVARIANT=0` is an
opt-out as much as `--batch-invariant off` is.

The mode keeps the match only for a prompt computed in one engine step: vLLM's chunked prefill splits a prompt longer
than `max_num_batched_tokens`, and a split prompt does not keep it. So when the mode is on and `--engine` names no
`max_num_batched_tokens`, `merge_engine` sets it to the served context (the base's ceiling when the length is chosen
at start), so that every prompt the server accepts is read in one step.

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


def base_default(precision: str | None, fp8_weights: bool, backend: str, adapters, image_model, mode: str) -> bool:
    """Whether the base turns the mode on: a bf16 base served by vLLM in separate mode, its weights not FP8, with no
    adapter applied at run time and no image engine (those are not measured with the mode and the cache yet)."""
    return (
        backend == "vllm"
        and precision == "bf16"
        and not fp8_weights
        and not adapters
        and not image_model
        and mode == "separate"
    )


def resolve(flag: str | None, repository: bool | None, environ, backend: str = "vllm", base_on: bool = False) -> dict:
    """Which setting the server runs, where it came from, and the variable set in `environ` before vLLM starts.
    `flag`: "on", "off" or None; `repository`: the tuned repository's `defaults.batch_invariant` or None; `base_on`:
    `base_default(...)`. Raises SystemExit naming the conflict (a variable that says the opposite of the flag or the
    repository, a backend without vLLM's kernels)."""
    if flag is not None and flag not in CHOICES:
        raise SystemExit(f"--batch-invariant is on or off, not {flag!r}")
    env = _env_value(environ)
    if flag is not None:
        want, source = flag == "on", "flag"
    elif repository is not None:
        want, source = bool(repository), "repository"
    elif env is not None:
        want, source = env, "environment"
    elif base_on:
        want, source = True, "bf16 base"
    else:
        want, source = False, "default"
    if backend != "vllm":
        if want and source in ("flag", "repository"):
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


def merge_engine(engine_json: str, max_model_len: int) -> tuple[str, dict]:
    """`--engine`'s JSON with `max_num_batched_tokens` set to `max_model_len` when it names none (a value given there
    wins), and what was done, for /health and the start line: {"max_num_batched_tokens": n, "set_by": "decisio" |
    "--engine"}."""
    import json

    kw = json.loads(engine_json)
    if "max_num_batched_tokens" in kw:
        return engine_json, {"max_num_batched_tokens": kw["max_num_batched_tokens"], "set_by": "--engine"}
    n = int(kw.get("max_model_len") or max_model_len)
    return json.dumps({**kw, "max_num_batched_tokens": n}), {"max_num_batched_tokens": n, "set_by": "decisio"}


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
        "bf16 base": "the default for a bf16 base",
        "default": "the default",
    }[report["source"]]
    batch = report.get("batching") or {}
    return (
        f"BATCH INVARIANT {'on' if report['on'] else 'off'} ({where})"
        + ("" if report.get("confirmed") is None else f"; vLLM reads it as {'on' if report['confirmed'] else 'off'}")
        + (
            f"; max_num_batched_tokens {batch['max_num_batched_tokens']:,} ({batch['set_by']}): one step a prompt"
            if batch
            else ""
        )
    )
