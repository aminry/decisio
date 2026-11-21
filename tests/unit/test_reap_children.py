# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A server's child processes (vLLM's engine cores when the engine runs in a process of its own) end with it.

K1  run_or_die: after main() returns normally (a server stopped by SIGTERM), the processes it started are stopped
K2  run_or_die: after main() raises, the same, and the exit code is 1
K3  reap_children: a child that ignores SIGTERM is killed after the grace period; with no children it returns 0

  uv run pytest -q tests/unit/test_reap_children.py
"""

import json
import subprocess
import sys
import time

import psutil
import pytest

SCRIPT = """
import json, subprocess, sys
from decisio.serve.vllm_engine import run_or_die

def main():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(json.dumps({"child": child.pid}), flush=True)
    if sys.argv[1] == "raise":
        raise RuntimeError("the server failed")
    return 0

sys.exit(run_or_die(main))
"""


def gone(pid):
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


@pytest.mark.parametrize("how,code", [("return", 0), ("raise", 1)])
def test_k1_k2_the_children_end_with_the_server(how, code):
    r = subprocess.run([sys.executable, "-c", SCRIPT, how], capture_output=True, text=True, timeout=120)
    assert r.returncode == code, r.stderr[-2000:]
    child = json.loads(r.stdout.strip().splitlines()[0])["child"]
    deadline = time.time() + 5
    while not gone(child) and time.time() < deadline:
        time.sleep(0.1)
    if not gone(child):
        psutil.Process(child).kill()  # do not leave it behind
        pytest.fail(f"the child {child} outlived the server")


REAP = """
import json, signal, subprocess, sys, time
from decisio.serve.vllm_engine import reap_children

none = reap_children(0.1)
child = subprocess.Popen(
    [sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)"],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
time.sleep(0.5)  # let it install the handler
t0 = time.time()
n = reap_children(0.5)
print(json.dumps({"none": none, "n": n, "s": time.time() - t0, "child": child.pid}))
"""


def test_k3_a_child_ignoring_sigterm_is_killed_after_the_grace():
    # in a process of its own: reap_children stops every child of the calling process
    r = subprocess.run([sys.executable, "-c", REAP], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["none"] == 0 and out["n"] == 1 and 0.5 <= out["s"] < 5
    assert gone(out["child"])
