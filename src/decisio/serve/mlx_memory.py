# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Memory limits of the MLX backend (`--backend mlx`): set at start-up from the machine, the base and the prefix
cache.

MLX keeps the buffers a computation has freed, up to a cache limit, and allows allocation up to a memory limit;
both default to
about 80 percent of the Mac's memory (51.8 GiB on a 64 GB Mac, MLX's `max_recommended_working_set_size`). A
server that
answers many different states therefore grows to that size, not to the size of its weights: a 12B server ended
at about 50 GB
on a 64 GB Mac and, with a second one, froze the machine twice (RLCD experiments/2026-10-14_lab3_games, Amendment 3).
So the server sets both limits itself, before it loads the model:

  cache limit    2 GiB by default (`--mlx-cache-limit-gb`): freed buffers kept for reuse, beyond what is live
  memory limit   by default what the base needs, `weights + working set + prefix cache + cache limit + 2 GiB
  headroom`,
                 rounded up, and never above two thirds of the machine's memory (`--mlx-memory-limit-gb`; 0
                 leaves MLX's own)

The working set is the base's measured peak above its weights at a 32k-token state (`decisio.families`,
`mlx_working_gb`: Qwen 4, Gemma 4 12B 6, Gemma 4 31B 9 GB; docs/running.md). Two thirds of the machine is the
figure the 12B's
prefix-cache budget was already sized against.

The server refuses to start, with the numbers, when the base cannot fit: `weights + working set + cache limit`
over the memory
limit (the default's ceiling, or the flag's), or a `--prefix-cache-mb` that does not fit under the limit. A
prefix cache that
was not asked for explicitly is made smaller to fit instead, and the start-up line says so. After the model is
loaded the
process's physical footprint is checked against the limit too.

The check and the figures in `/health` use the process's **physical footprint** (macOS's `phys_footprint`, what
Activity
Monitor calls memory and what the kernel's memory-pressure decisions follow), not its resident size, which
leaves out part of
MLX's Metal buffers (10.7 GB resident against 14 GB of footprint for a 12B server, Lab 3).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

GIB = 2**30
MIB = 2**20
CEILING = 2 / 3  # of the machine's memory: the most a default limit asks for
DEFAULT_CACHE_LIMIT = 2 * GIB
HEADROOM = 2 * GIB
DEFAULT_WORKING_BYTES = 6 * 10**9  # a base with no measured working set


class MemoryRefused(ValueError):
    """The base does not fit the memory limit: the message says by how much and what to change."""


def gib(n: float) -> str:
    return f"{n / GIB:.1f} GiB"


@dataclass(frozen=True)
class Plan:
    """What the server asks MLX for, and the figures behind it (bytes; `prefix_cache_mb` in MiB)."""

    machine_bytes: int
    weights_bytes: int
    working_bytes: int
    cache_limit_bytes: int
    memory_limit_bytes: int  # 0: not set (MLX's own limit)
    prefix_cache_mb: int  # the budget the engine uses
    prefix_cache_requested_mb: int
    memory_limit_source: str  # "default", "flag" or "mlx"
    cache_limit_source: str  # "default" or "flag"

    @property
    def base_bytes(self) -> int:
        return self.weights_bytes + self.working_bytes + self.cache_limit_bytes


def make_plan(
    *,
    machine_bytes: int,
    weights_bytes: int,
    working_bytes: int,
    prefix_cache_mb: int,
    prefix_explicit: bool = False,
    memory_limit_gb: float | None = None,
    cache_limit_gb: float | None = None,
    name: str = "this base",
) -> Plan:
    """The limits for a base, or MemoryRefused. Pure: no MLX, no files."""
    cache = DEFAULT_CACHE_LIMIT if cache_limit_gb is None else int(cache_limit_gb * GIB)
    base = weights_bytes + working_bytes + cache
    ceiling = int(machine_bytes * CEILING)
    prefix = int(prefix_cache_mb) * MIB
    how = (
        f"{name} needs about {gib(base)} under MLX ({gib(weights_bytes)} of weights, a working set of "
        f"{gib(working_bytes)} at a 32k-token state, and the MLX cache limit of {gib(cache)})"
    )
    if memory_limit_gb == 0:  # MLX's own limit: nothing is set, nothing can be refused
        return Plan(
            machine_bytes,
            weights_bytes,
            working_bytes,
            cache,
            0,
            prefix_cache_mb,
            prefix_cache_mb,
            "mlx",
            "default" if cache_limit_gb is None else "flag",
        )
    if memory_limit_gb is None:
        if base > ceiling:
            raise MemoryRefused(
                f"{how}, and this Mac has {gib(machine_bytes)}: two thirds of it, {gib(ceiling)}, is the most the "
                "server asks MLX for by default. Start a smaller base or conversion, or set --mlx-memory-limit-gb to a "
                "limit you have checked the machine can give (0 leaves MLX's own, about 80 percent of the memory)"
            )
        limit = min(ceiling, base + prefix + HEADROOM)
        limit = min(ceiling, -(-limit // GIB) * GIB)  # whole GiB, up
        source = "default"
    else:
        limit = int(memory_limit_gb * GIB)
        source = "flag"
        if base > limit:
            raise MemoryRefused(
                f"{how}, which is more than the memory limit of {gib(limit)} (--mlx-memory-limit-gb "
                f"{memory_limit_gb:g}). Raise the limit, lower --mlx-cache-limit-gb, or start a smaller base"
            )
    room = max(0, limit - base - HEADROOM)  # what the prefix cache can have beside the base and the headroom
    room_mb = room // MIB
    if prefix_cache_mb > room_mb:
        if prefix_explicit:
            raise MemoryRefused(
                f"--prefix-cache-mb {prefix_cache_mb} does not fit: under the memory limit of {gib(limit)}, {how} and "
                f"{gib(HEADROOM)} of headroom leave {room_mb} MiB for the prefix cache. Lower --prefix-cache-mb or "
                "raise "
                "--mlx-memory-limit-gb"
            )
        prefix_cache_mb = room_mb  # the base's default budget, made smaller to fit
    return Plan(
        machine_bytes,
        weights_bytes,
        working_bytes,
        cache,
        limit,
        prefix_cache_mb,
        int(prefix / MIB),
        source,
        "default" if cache_limit_gb is None else "flag",
    )


def machine_memory_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def weights_bytes(model_dir) -> int:
    """The size of a conversion's weights: its safetensors files (symbolic links into the Hub cache are followed)."""
    return sum(f.stat().st_size for f in Path(model_dir).glob("*.safetensors"))


def _rusage(pid: int | None = None):
    """macOS `proc_pid_rusage` (RUSAGE_INFO_V2) as 64-bit words, or None where it is not available."""
    if sys.platform != "darwin":
        return None
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib", use_errno=True)
        buf = (ctypes.c_uint64 * 64)()
        if (
            lib.proc_pid_rusage(ctypes.c_int(os.getpid() if pid is None else pid), ctypes.c_int(2), ctypes.byref(buf))
            != 0
        ):
            return None
        return buf
    except (OSError, AttributeError):
        return None


def physical_footprint_bytes(pid: int | None = None) -> int | None:
    """The process's physical footprint (`ri_phys_footprint`), or None off macOS."""
    buf = _rusage(pid)
    return int(buf[9]) if buf is not None else None  # after the 16-byte uuid: 8 words precede it


def resident_bytes(pid: int | None = None) -> int | None:
    buf = _rusage(pid)
    return int(buf[8]) if buf is not None else None


def apply(mx, plan: Plan) -> None:
    """Set MLX's limits. The cache limit goes in always; the memory limit unless it is MLX's own."""
    mx.set_cache_limit(int(plan.cache_limit_bytes))
    if plan.memory_limit_bytes:
        mx.set_memory_limit(int(plan.memory_limit_bytes))


def check_after_load(plan: Plan, footprint: int | None) -> None:
    """Refuse when the loaded process is already over the limit (by its physical footprint, not its resident size)."""
    if plan.memory_limit_bytes and footprint is not None and footprint > plan.memory_limit_bytes:
        raise MemoryRefused(
            f"after loading, the server's physical footprint is {gib(footprint)}, above the memory limit of "
            f"{gib(plan.memory_limit_bytes)} (weights {gib(plan.weights_bytes)}). Raise --mlx-memory-limit-gb or start "
            "a smaller base"
        )


def facts(mx, plan: Plan) -> dict:
    """What `/health` reports: the limits as set, MLX's active, cache and peak memory, and the process's physical
    footprint and resident size (bytes; null where the platform does not say)."""
    return {
        **asdict(plan),
        "mlx_active_bytes": int(mx.get_active_memory()),
        "mlx_cache_bytes": int(mx.get_cache_memory()),
        "mlx_peak_bytes": int(mx.get_peak_memory()),
        "physical_footprint_bytes": physical_footprint_bytes(),
        "resident_bytes": resident_bytes(),
    }


def setup(mx, model_dir, family, prefix_cache_mb: int, prefix_explicit: bool, memory_limit_gb, cache_limit_gb) -> Plan:
    """Plan the limits for the conversion in `model_dir` and set them, before the model is loaded."""
    plan = make_plan(
        machine_bytes=machine_memory_bytes(),
        weights_bytes=weights_bytes(model_dir),
        working_bytes=int(getattr(family, "mlx_working_gb", 0) * 1e9) or DEFAULT_WORKING_BYTES,
        prefix_cache_mb=prefix_cache_mb,
        prefix_explicit=prefix_explicit,
        memory_limit_gb=memory_limit_gb,
        cache_limit_gb=cache_limit_gb,
        name=f"the {family.key} base",
    )
    apply(mx, plan)
    return plan


def describe(plan: Plan) -> str:
    limit = gib(plan.memory_limit_bytes) if plan.memory_limit_bytes else "MLX's own"
    note = (
        f"; the prefix cache was cut from {plan.prefix_cache_requested_mb} MiB to fit"
        if plan.prefix_cache_mb != plan.prefix_cache_requested_mb
        else ""
    )
    return (
        f"memory limit {limit} ({plan.memory_limit_source}), cache limit {gib(plan.cache_limit_bytes)} "
        f"({plan.cache_limit_source}), prefix cache {plan.prefix_cache_mb} MiB, weights "
        f"{gib(plan.weights_bytes)}, this "
        f"Mac {gib(plan.machine_bytes)}{note}"
    )
