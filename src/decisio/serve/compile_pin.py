# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A compile with no timed choices, an opt-in (docs/running.md, "Deterministic compile").

vLLM compiles the model with torch's inductor and, unless told otherwise, turns on its combo kernels: kernels that fuse
several small ones (the q, k and v norms of an attention layer) and autotune their launch settings (num_warps) by timing
a few choices at the first launch. Two choices that take the same time can sum the same values in another order, so two
cold starts of the same model can compute slightly different arithmetic. It is off unless asked for: on the 12B it makes
cold starts agree, on the 31B they already agreed in the card check and turning it on changes answers.
This setting takes the timed choices away: `deterministic: true` and `combo_kernels: false` in the engine's inductor
compile config. vLLM hands that config to inductor's standalone compile as config patches, which is the route that
reaches the code generation; `TORCHINDUCTOR_DETERMINISTIC=1` in the environment (inductor reads it once, when
`torch._inductor.config` is imported) is set as well but does not reach it under vLLM 0.31.0 (measured by the throughput
study), so the engine config is what the setting stands on. `kernel_meta_report` reads the generated kernels of a
compile and says whether it did.

    resolve(flag, environ, backend) -> {"on": bool, "source": "flag", "environment", "default"}
    merge_engine(engine_json, on) -> engine_json with the compile setting added where the caller gave none

Precedence: the flag, then the environment, then off; a variable that says the opposite of the flag is refused at
start. Standard library only.
"""

from __future__ import annotations

import json
import sys

VARIABLE = "TORCHINDUCTOR_DETERMINISTIC"
CHOICES = ("on", "off")
FLAG = "--deterministic-compile"
ENGINE_SETTING = {"compilation_config": {"inductor_compile_config": {"combo_kernels": False, "deterministic": True}}}


def _env_value(environ) -> bool | None:
    """True / False for 1 / 0 in the environment, None when it is not set; anything else is refused."""
    raw = environ.get(VARIABLE)
    if raw is None or raw == "":
        return None
    if raw not in ("0", "1"):
        raise SystemExit(f"{VARIABLE}={raw!r}: torch reads 1 for on; unset it or use {FLAG} on|off")
    return raw == "1"


def resolve(flag: str | None, environ, backend: str = "vllm") -> dict:
    """Which setting the server runs, where it came from, and the variable set in `environ` before the engine starts.
    `flag`: "on", "off" or None (then the environment decides, else off). Raises SystemExit naming the conflict (a
    variable that says the opposite of the flag, a backend without inductor)."""
    if flag is not None and flag not in CHOICES:
        raise SystemExit(f"{FLAG} is on or off, not {flag!r}")
    env = _env_value(environ)
    if backend != "vllm":
        if flag == "on":
            raise SystemExit(f"{FLAG} is for the vLLM server, not --backend {backend}")
        return {"on": False, "source": "default"}
    if flag is not None:
        want, source = flag == "on", "flag"
    elif env is not None:
        want, source = env, "environment"
    else:
        want, source = False, "default"
    if env is not None and env != want:
        raise SystemExit(
            f"{VARIABLE}={int(env)} in the environment contradicts {FLAG} {flag}: "
            "unset the variable, or change the flag"
        )
    if want:
        environ[VARIABLE] = "1"
    return {"on": want, "source": source}


def merge_engine(engine_json: str, on: bool) -> str:
    """`--engine`'s JSON with `deterministic: true` and `combo_kernels: false` added to the inductor compile config when
    the setting is on and the caller named no value of their own for either (an explicit value wins; the setting never
    overrides a choice)."""
    if not on:
        return engine_json
    kw = json.loads(engine_json)
    compilation = dict(kw.get("compilation_config") or {})
    inductor = dict(compilation.get("inductor_compile_config") or {})
    for key, value in ENGINE_SETTING["compilation_config"]["inductor_compile_config"].items():
        inductor.setdefault(key, value)
    compilation["inductor_compile_config"] = inductor
    return json.dumps({**kw, "compilation_config": compilation})


def kernel_meta_report(*roots) -> dict:
    """Counts over every generated Triton kernel module under `roots` (the vLLM compile cache's `inductor_cache`
    folders, `/tmp/torchinductor_<user>`): how many say `'deterministic': True` and `False` in their `inductor_meta`,
    and how many combo kernels there are. A pinned compile has files > 0, deterministic_true > 0,
    deterministic_false == 0 and combo == 0. Clear the caches before the start under test: a warm start loads the old
    compile's kernels and the report then describes the old compile. Only a CUDA compile writes Triton kernels."""
    import pathlib
    import re

    flag = re.compile(r"'deterministic': (True|False)")
    out = {"files": 0, "deterministic_true": 0, "deterministic_false": 0, "combo": 0}
    for root in roots:
        for path in pathlib.Path(root).rglob("*.py"):
            text = path.read_text(errors="replace")
            if "triton_heuristics" not in text:
                continue
            out["files"] += 1
            for m in flag.finditer(text):
                out["deterministic_true" if m.group(1) == "True" else "deterministic_false"] += 1
            out["combo"] += text.count("'combo_grid_meta'")
    return out


def apply_in_process(on: bool) -> None:
    """Write the setting to torch's inductor config where the server's process has imported it already."""
    if not on:
        return
    config = sys.modules.get("torch._inductor.config")
    if config is not None:
        config.deterministic = True


def engine_state() -> bool | None:
    """What torch's inductor config holds in this process; None where torch is not importable here."""
    try:
        import torch._inductor.config as config
    except Exception:
        return None
    try:
        return bool(config.deterministic)
    except Exception:
        return None


def start_line(report: dict) -> str:
    where = {
        "flag": FLAG,
        "environment": f"{VARIABLE} in the environment",
        "default": "the default",
    }[report["source"]]
    return f"DETERMINISTIC COMPILE {'on' if report['on'] else 'off'} ({where})" + (
        "" if report.get("confirmed") is None else f"; torch reads it as {'on' if report['confirmed'] else 'off'}"
    )
