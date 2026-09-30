# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The patch series under patches/ against the third-party source they are written for (patches/README.md).

  S1  every file of a series applies to the tagged vLLM tree, in order (`git apply --check`, each after the previous
      ones were applied to a scratch copy), in both forms: the full `git format-patch` series and `pkg/`
  S2  `pkg/` is the full series restricted to paths under `vllm/`, hunk for hunk
  S3  the series is inert by default: every changed code path is reached only under the flag (the flag is read from the
      environment, default off), and patches/apply.sh applies exactly the `pkg/` files, each after a dry run
  S4  patches/apply.sh applies the `pkg/` series to an installed vLLM's site-packages (here: a copy of the tagged tree
      standing in for one), refuses a second application and a vLLM of another version

Needs the tagged source tree (`bash scripts/fetch_vllm_source.sh`; VLLM_SRC overrides the location); skipped without it.

    uv run pytest -q tests/unit/test_patches.py
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SERIES = ROOT / "patches" / "vllm-0.30.0" / "suffix-staging"
COMMIT = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
SRC = Path(os.environ.get("VLLM_SRC") or Path.home() / ".cache" / "decisio" / "vllm-v0.30.0")


def git(cwd, *args, check=True):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=check)


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    if not (SRC / ".git").exists():
        pytest.skip(f"no vLLM source tree at {SRC} (bash scripts/fetch_vllm_source.sh)")
    assert git(SRC, "rev-parse", "HEAD").stdout.strip() == COMMIT, "the source tree is not vLLM v0.30.0"
    assert git(SRC, "status", "--porcelain").stdout == "", "the source tree has local changes"
    return SRC


def scratch(tree, tmp_path, name):
    dst = tmp_path / name
    git(tree, "worktree", "add", "--detach", str(dst), COMMIT)
    return dst


def files(sub=""):
    return sorted((SERIES / sub).glob("0*.patch"))


@pytest.mark.parametrize("form", ["", "pkg"])
def test_s1_series_applies_in_order(tree, tmp_path, form):
    assert len(files(form)) == 2
    wt = scratch(tree, tmp_path, f"wt_{form or 'full'}")
    try:
        for p in files(form):
            chk = git(wt, "apply", "--check", str(p), check=False)
            assert chk.returncode == 0, f"{p.name} does not apply to vLLM v0.30.0: {chk.stderr[:400]}"
            git(wt, "apply", str(p))
        changed = set(git(wt, "status", "--porcelain").stdout.split("\n"))
        assert any("vllm/v1/worker/gpu/suffix_staging.py" in c for c in changed)
        if form == "pkg":                                       # the installed-package form touches nothing else
            assert all(c[3:].startswith("vllm/") for c in changed if c)
    finally:
        git(tree, "worktree", "remove", "--force", str(wt))
        shutil.rmtree(wt, ignore_errors=True)


def hunks(path, only_vllm):
    """{file: diff text} of a patch file, optionally only the files under vllm/."""
    out = {}
    for part in re.split(r"(?m)^diff --git ", path.read_text())[1:]:
        name = part.split(" b/", 1)[1].split("\n", 1)[0]
        body = part.split("\n", 1)[1]
        body = re.split(r"(?m)^-- \n", body)[0]                 # the mail footer after the last file
        if not only_vllm or name.startswith("vllm/"):
            out[name] = body
    return out


def test_s2_pkg_is_the_series_restricted_to_vllm():
    for full, pkg in zip(files(""), files("pkg")):
        assert full.name == pkg.name
        a, b = hunks(full, True), hunks(pkg, False)
        assert set(a) == set(b) and all(n.startswith("vllm/") for n in b)
        for name in a:
            assert a[name].rstrip("\n") == b[name].rstrip("\n"), f"{pkg.name}: {name} differs from the full series"


def test_s3_inert_by_default_and_what_apply_applies():
    first = files("pkg")[0].read_text()
    envs = hunks(files("pkg")[0], False)["vllm/envs.py"]
    assert "VLLM_SUFFIX_STAGING" in envs and re.search(r'VLLM_SUFFIX_STAGING", "0"\)', envs), \
        "the flag must default to off"
    assert "VLLM_SUFFIX_STAGING" in first
    apply = (ROOT / "patches" / "apply.sh").read_text()
    assert '"$SERIES"/pkg/0*.patch' in apply and "vllm-0.30.0/suffix-staging" in apply
    assert apply.index("--dry-run") < apply.index("patch -p1 --forward < ")          # every file dry-run first


def fake_python(tmp_path, site, version):
    """A `python` that answers apply.sh's two questions (vLLM's version, its site-packages) without vLLM."""
    py = tmp_path / f"python-{version}"
    py.write_text(f"""#!/bin/bash
case "$2" in *__version__*) echo {version} ;; *) echo {site} ;; esac
""")
    py.chmod(0o755)
    return py


def test_s4_apply_script(tree, tmp_path):
    site = tmp_path / "site-packages"
    shutil.copytree(tree / "vllm", site / "vllm", ignore=shutil.ignore_patterns("__pycache__"))
    apply = ROOT / "patches" / "apply.sh"
    r = subprocess.run(["bash", str(apply), str(fake_python(tmp_path, site, "0.30.0"))], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["applied", files("pkg")[0].name, "applied", files("pkg")[1].name]
    assert (site / "vllm" / "v1" / "worker" / "gpu" / "suffix_staging.py").exists()
    again = subprocess.run(["bash", str(apply), str(fake_python(tmp_path, site, "0.30.0"))],
                           capture_output=True, text=True)
    assert again.returncode != 0 and "does not apply" in again.stderr                  # already applied: refused
    other = subprocess.run(["bash", str(apply), str(fake_python(tmp_path, site, "0.31.0"))],
                           capture_output=True, text=True)
    assert other.returncode != 0 and "is for vllm 0.30.0" in other.stderr
