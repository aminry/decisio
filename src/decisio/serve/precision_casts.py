# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Inductor's precision-cast emulation, on for FP8 bases (docs/running.md, "Precision casts").

vLLM compiles the model with torch's inductor, which may fuse two bf16 operations and drop the rounding to bf16 between
them. Which operations it fuses is not the same on every compile, so two cold starts of the same FP8 model can run
slightly different arithmetic and agree on about 97% of the top answers. `emulate_precision_casts` keeps each rounding
step as eager PyTorch computes it, which narrows the difference between two cold compiles (it does not remove it).
torch reads `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` once, when `torch._inductor.config` is imported. An engine in
another process inherits the variable; an engine in the server's own process may have imported the module already, so
the setting is also written to the module.

    resolve(flag, fp8_weights, environ, backend) -> {"on": bool, "source": "flag", "fp8 base", "environment", "default"}

Precedence: the flag, then the base (on for an FP8 base: the 12B FP8, the 31B served FP8 stored or on load, a tuned
repository of either), then the environment, then off. A variable that says the opposite of the flag or the base's
default is refused at start, so /health never misreports what the engine runs.

Standard library only (`engine_state` and `apply_in_process` touch torch only when it is importable and already loaded).
"""

from __future__ import annotations

import sys

VARIABLE = "TORCHINDUCTOR_EMULATE_PRECISION_CASTS"
CHOICES = ("on", "off", "auto")
FLAG = "--emulate-precision-casts"


def _env_value(environ) -> bool | None:
    """True / False for 1 / 0 in the environment, None when it is not set; anything else is refused."""
    raw = environ.get(VARIABLE)
    if raw is None or raw == "":
        return None
    if raw not in ("0", "1"):
        raise SystemExit(f"{VARIABLE}={raw!r}: torch reads 0 or 1; unset it or use {FLAG} on|off")
    return raw == "1"


def resolve(flag: str | None, fp8_weights: bool, environ, backend: str = "vllm") -> dict:
    """Which setting the server runs, where it came from, and the variable set in `environ` before the engine starts.
    `flag`: "on", "off", "auto" or None (the last two are the same: the base decides); `fp8_weights`: the weights are
    served in FP8. Raises SystemExit naming the conflict (a variable that says the opposite, a backend without
    inductor)."""
    if flag is not None and flag not in CHOICES:
        raise SystemExit(f"{FLAG} is on, off or auto, not {flag!r}")
    if flag == "auto":
        flag = None
    env = _env_value(environ)
    if backend != "vllm":
        if flag == "on":
            raise SystemExit(f"{FLAG} is for the vLLM server, not --backend {backend}")
        return {"on": False, "source": "default"}
    if flag is not None:
        want, source = flag == "on", "flag"
    elif fp8_weights:
        want, source = True, "fp8 base"
    elif env is not None:
        want, source = env, "environment"
    else:
        want, source = False, "default"
    if env is not None and env != want:
        who = {"flag": f"{FLAG} {flag}", "fp8 base": "the default for an FP8 base (on)"}[source]
        hint = f"use {FLAG} off" if source == "fp8 base" else "change the flag"
        raise SystemExit(f"{VARIABLE}={int(env)} in the environment contradicts {who}: unset the variable, or {hint}")
    if want:
        environ[VARIABLE] = "1"
    return {"on": want, "source": source}


def apply_in_process(on: bool) -> None:
    """Write the setting to torch's inductor config where the server's process has imported it already (the variable is
    read once, at import). A process that has not imported it reads the variable when it does."""
    if not on:
        return
    config = sys.modules.get("torch._inductor.config")
    if config is not None:
        config.emulate_precision_casts = True


def engine_state() -> bool | None:
    """What torch's inductor config holds in this process; None where torch is not importable here."""
    try:
        import torch._inductor.config as config
    except Exception:
        return None
    try:
        return bool(config.emulate_precision_casts)
    except Exception:
        return None


def start_line(report: dict) -> str:
    where = {
        "flag": FLAG,
        "fp8 base": "the default for an FP8 base",
        "environment": f"{VARIABLE} in the environment",
        "default": "the default",
    }[report["source"]]
    return f"PRECISION CASTS emulated {'on' if report['on'] else 'off'} ({where})" + (
        "" if report.get("confirmed") is None else f"; torch reads it as {'on' if report['confirmed'] else 'off'}"
    )
