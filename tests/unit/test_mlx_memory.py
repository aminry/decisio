# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""MLX's memory and cache limits, set at start-up (`decisio.serve.mlx_memory`, `--backend mlx`). A fake `mx`,
no MLX needed.

M1  the plan: the limit a base asks for by default (weights + working set + prefix cache + cache limit + 2
    GiB, whole GiB,
    never above two thirds of the machine), Lab 3's configuration, a prefix cache cut to fit when it was not
    asked for;
M2  the refusals, with the numbers: a base over two thirds of the machine, over an explicit limit, a prefix
    cache asked for
    that does not fit; `--mlx-memory-limit-gb 0` leaves MLX's own limit and refuses nothing;
M3  start-up with a fake mx: the cache limit and the memory limit are set before anything is loaded; the check
    after loading
    uses the physical footprint; /health's figures;
M4  the physical footprint is not the resident size (macOS only);
M5  the flags: in --help, and refused off --backend mlx.

  uv run pytest -q tests/unit/test_mlx_memory.py
"""

import subprocess
import sys

import pytest

from decisio.families import BASES, GEMMA4, GEMMA4_31B, QWEN
from decisio.serve import mlx_memory as mm

GIB, GB = 2**30, 10**9


def plan(machine_gib=64, weights_gb=9.7, working_gb=6, prefix_mb=2048, **kw):
    return mm.make_plan(
        machine_bytes=int(machine_gib * GIB),
        weights_bytes=int(weights_gb * GB),
        working_bytes=int(working_gb * GB),
        prefix_cache_mb=prefix_mb,
        **kw,
    )


def test_m1_the_default_limit_follows_the_base_and_is_capped():
    p = plan(prefix_mb=2048)  # Lab 3's 12B configuration on the 64 GB Mac
    assert p.cache_limit_bytes == 2 * GIB and p.prefix_cache_mb == 2048 and p.memory_limit_source == "default"
    assert p.memory_limit_bytes % GIB == 0
    need = 9.7 * GB + 6 * GB + 2 * GIB + 2048 * 2**20 + 2 * GIB
    assert need <= p.memory_limit_bytes < need + GIB  # whole GiB, up
    assert p.memory_limit_bytes < 24 * GIB  # under what Lab 3 chose by hand for the same base
    big = plan(prefix_mb=7400)  # the 12B's own default prefix cache fits on a 64 GB Mac whole
    assert big.prefix_cache_mb == 7400 and big.memory_limit_bytes <= int(64 * GIB * 2 / 3)
    small = plan(machine_gib=32, prefix_mb=7400)  # on 32 GiB the same base is held to two thirds, the cache cut to fit
    assert small.memory_limit_bytes == int(32 * GIB * 2 / 3) // GIB * GIB or small.memory_limit_bytes <= int(
        32 * GIB * 2 / 3
    )
    assert 0 < small.prefix_cache_mb < 7400 and small.prefix_cache_requested_mb == 7400
    room = small.memory_limit_bytes - small.base_bytes - mm.HEADROOM
    assert small.prefix_cache_mb * 2**20 <= room
    assert plan(cache_limit_gb=1).cache_limit_bytes == GIB and plan(cache_limit_gb=1).cache_limit_source == "flag"


def test_m2_refusals_say_the_numbers():
    with pytest.raises(mm.MemoryRefused) as e:  # the 31B on a 32 GiB Mac
        plan(machine_gib=32, weights_gb=24.9, working_gb=9, name="the gemma-4-31b base")
    msg = str(e.value)
    assert (
        "the gemma-4-31b base needs about" in msg
        and "32.0 GiB" in msg
        and "21.3 GiB" in msg
        and "--mlx-memory-limit-gb" in msg
    )
    with pytest.raises(mm.MemoryRefused) as e:  # an explicit limit below what the base needs
        plan(memory_limit_gb=12)
    assert "--mlx-memory-limit-gb 12" in str(e.value) and "more than the memory limit of 12.0 GiB" in str(e.value)
    with pytest.raises(mm.MemoryRefused) as e:  # a prefix cache asked for that does not fit
        plan(machine_gib=32, prefix_mb=7400, prefix_explicit=True)
    assert "--prefix-cache-mb 7400 does not fit" in str(e.value)
    own = plan(machine_gib=8, memory_limit_gb=0)  # MLX's own limit: nothing set, nothing refused
    assert own.memory_limit_bytes == 0 and own.memory_limit_source == "mlx"
    flag = plan(machine_gib=8, memory_limit_gb=20)  # a flag may exceed the default ceiling: that is its point
    assert flag.memory_limit_bytes == 20 * GIB and flag.memory_limit_source == "flag"


class FakeMx:
    def __init__(self, active=0):
        self.calls, self.active = [], active

    def set_cache_limit(self, n):
        self.calls.append(("cache", n))

    def set_memory_limit(self, n):
        self.calls.append(("memory", n))

    def get_active_memory(self):
        return self.active

    def get_cache_memory(self):
        return 123

    def get_peak_memory(self):
        return self.active + 456


def conversion(tmp_path, gb):
    for i in range(2):  # weights as sparse files: the size is what counts
        with open(tmp_path / f"model-{i}.safetensors", "wb") as f:
            f.truncate(int(gb * GB / 2))
    (tmp_path / "config.json").write_text("{}")
    return tmp_path


def test_m3_start_up_sets_the_limits_and_checks_the_footprint(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "machine_memory_bytes", lambda: 64 * GIB)
    mx = FakeMx(active=5 * GIB)
    p = mm.setup(mx, conversion(tmp_path, 9.7), GEMMA4, 2048, False, None, None)
    assert p.weights_bytes == int(9.7 * GB) and p.working_bytes == 6 * GB
    assert mx.calls == [
        ("cache", 2 * GIB),
        ("memory", p.memory_limit_bytes),
    ]  # the cache limit first, then the memory limit
    off = FakeMx()
    mm.setup(off, tmp_path, GEMMA4, 2048, False, 0, 0.5)  # --mlx-memory-limit-gb 0: only the cache limit is set
    assert off.calls == [("cache", GIB // 2)]
    mm.check_after_load(p, p.memory_limit_bytes - 1)  # under: fine
    mm.check_after_load(p, None)  # a platform that cannot say: nothing to check
    with pytest.raises(mm.MemoryRefused, match="physical footprint"):
        mm.check_after_load(p, p.memory_limit_bytes + 1)
    f = mm.facts(mx, p)
    assert f["memory_limit_bytes"] == p.memory_limit_bytes and f["cache_limit_bytes"] == 2 * GIB
    assert f["mlx_active_bytes"] == 5 * GIB and f["mlx_cache_bytes"] == 123 and f["mlx_peak_bytes"] == 5 * GIB + 456
    assert {"physical_footprint_bytes", "resident_bytes", "weights_bytes", "machine_bytes", "prefix_cache_mb"} <= set(f)
    assert "prefix cache 2048 MiB" in mm.describe(p)


def test_m3_every_base_has_a_working_set():
    assert (QWEN.mlx_working_gb, GEMMA4.mlx_working_gb, GEMMA4_31B.mlx_working_gb) == (4, 6, 9)
    assert all(b.mlx_working_gb > 0 for b in BASES.values())


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS's proc_pid_rusage")
def test_m4_the_footprint_is_read_from_the_kernel_not_from_the_resident_size():
    fp, rss = mm.physical_footprint_bytes(), mm.resident_bytes()
    assert fp > 10 * 2**20 and rss > 10 * 2**20
    assert (
        mm.physical_footprint_bytes(1) is None or mm.physical_footprint_bytes(1) > 0
    )  # another process: allowed or not


@pytest.mark.skipif(sys.platform == "darwin", reason="the other platforms")
def test_m4_off_macos_there_is_no_footprint():
    assert mm.physical_footprint_bytes() is None and mm.resident_bytes() is None


def run_server(*args):
    return subprocess.run(
        [sys.executable, "-m", "decisio.serve.vllm_engine", *args], capture_output=True, text=True, timeout=300
    )


def test_m5_the_flags():
    out = run_server("--help").stdout
    assert "--mlx-memory-limit-gb" in out and "--mlx-cache-limit-gb" in out
    for flag in ("--mlx-memory-limit-gb", "--mlx-cache-limit-gb"):
        r = run_server("--backend", "hf", "--model", "x", flag, "5")
        assert r.returncode != 0 and f"{flag} is for --backend mlx" in r.stderr
