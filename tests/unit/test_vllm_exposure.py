# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The facts SECURITY.md relies on to say the four vLLM advisories fixed in 0.31.0 are not reachable through a
decisio server, checked in the source (no vLLM, no model):

X1  every `SamplingParams(...)` sets `max_tokens=1` and none sets `min_tokens` (GHSA-4xqp-c3mv-qff7)
X2  every sampling call sets `allowed_token_ids` except the prefix warm-up (`send_warm`) and the health probe, the
    only requests the stale mask could land on (GHSA-6cxc-2vcg-w5qc)
X3  no source names `mm_processor_kwargs`, `trust_remote_code`, `min_tokens` or vLLM's entrypoints, and decisio never
    starts `vllm serve` (GHSA-h3rc-6mm3-gc2m, GHSA-4xqp-c3mv-qff7)
X4  the multimodal ids sent to vLLM are the ones the image engine computes from the decoded pixels, not a request
    field (GHSA-p92p-rxj5-7p2x)
"""

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "decisio"
UNCONSTRAINED = {"send_warm", "probe"}


def sampling_calls():
    """(file, enclosing function, Call) for every SamplingParams(...) in the source."""
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))] + [tree]:
            name = getattr(fn, "name", "<module>")
            for node in ast.iter_child_nodes(fn):
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Call)
                        and getattr(sub.func, "id", getattr(sub.func, "attr", "")) == "SamplingParams"
                    ):
                        yield path, name, sub


def unique_calls():
    seen, out = set(), []
    for path, name, call in sampling_calls():
        key = (path, call.lineno, call.col_offset)
        if key not in seen:
            seen.add(key)
            out.append((path, name, call))
    return out


def test_X0_the_source_has_sampling_calls():
    names = {n for _, n, _ in unique_calls()}
    assert {"score_prompts", "send_warm", "probe"} <= names, names


def test_X1_max_tokens_is_one_and_min_tokens_is_never_set():
    for path, name, call in unique_calls():
        kw = {k.arg: k.value for k in call.keywords}
        where = f"{path.name}:{call.lineno} ({name})"
        assert "min_tokens" not in kw, where
        assert isinstance(kw.get("max_tokens"), ast.Constant) and kw["max_tokens"].value == 1, where


def test_X2_only_the_warm_up_and_the_probe_leave_the_allowed_tokens_unset():
    for path, name, call in unique_calls():
        kw = {k.arg: k.value for k in call.keywords}
        where = f"{path.name}:{call.lineno} ({name})"
        if name in UNCONSTRAINED:
            assert "allowed_token_ids" not in kw, where + ": the doc says this call sets none"
        else:
            assert "allowed_token_ids" in kw, where
            assert not (isinstance(kw["allowed_token_ids"], ast.Constant) and kw["allowed_token_ids"].value is None), (
                where
            )


def test_X3_no_request_controlled_engine_options_and_no_vllm_server():
    pattern = re.compile(
        r"mm_processor_kwargs|trust_remote_code|trust-remote-code|min_tokens|vllm\.entrypoints|vllm serve"
    )
    hits = [
        f"{p.relative_to(SRC)}:{i}"
        for p in sorted(SRC.rglob("*.py"))
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if pattern.search(line)
    ]
    assert hits == [], hits


def test_X4_multimodal_ids_come_from_the_decoded_pixels():
    src = (SRC / "serve" / "image_engine.py").read_text()
    assert 'mm_uuids={"image": ids}' in src and "ids = [self.image_id(img) for img in images]" in src
    body = ast.get_source_segment(
        src, next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "image_id")
    )
    assert "hashlib.sha256(img.tobytes())" in body
    # the only places that put multi_modal_uuids on a prompt take the argument the image engine passes
    for path in sorted(SRC.rglob("*.py")):
        for line in path.read_text().splitlines():
            if '"multi_modal_uuids"' in line:
                assert "mm_uuids" in line, f"{path.name}: {line.strip()}"
