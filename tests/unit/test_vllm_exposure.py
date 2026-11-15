# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The facts SECURITY.md relies on to say the four vLLM advisories fixed in 0.31.0 are not reachable through a
decisio server, checked in the source (no vLLM, no model):

X1  every `SamplingParams(...)` sets `max_tokens=1` and none sets `min_tokens` (GHSA-4xqp-c3mv-qff7), except `think`
    (think when unsure) and `generate_rows` (a Box question's forced reply and its branch decodes), whose
    `max_tokens` is a named argument (`budget`, `max_tokens`) that is always a server-side constant or another
    validated server setting, never a request field
X2  every sampling call sets `allowed_token_ids` except the warm-up (`_prefill`), the health probe, `think` and
    `generate_rows` (free text), the requests the stale mask could land on (GHSA-6cxc-2vcg-w5qc)
X3  no source names `mm_processor_kwargs`, `trust_remote_code`, `min_tokens` or vLLM's entrypoints, and decisio never
    starts `vllm serve` (GHSA-h3rc-6mm3-gc2m, GHSA-4xqp-c3mv-qff7)
X4  the multimodal ids sent to vLLM are the ones the image engine computes from the decoded pixels, not a request
    field (GHSA-p92p-rxj5-7p2x)
"""

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "decisio"
# the warm-up (`send_warm`, whose engine call is `_prefill`), the probe, and the two writers of free text
UNCONSTRAINED = {"_prefill", "probe", "think", "generate_rows"}
# the writers of free text, and the name their own `max_tokens` argument is passed under: `think`'s is its own
# `budget` argument; `generate_rows`'s is its own `max_tokens` argument, which every caller passes a module
# constant for (`decisio.serve.gemma_images.BOX_MAX_TOKENS`, `BOX_BRANCH_MAX_TOKENS`) or the literal 1
FREE_LENGTH = {"think": "budget", "generate_rows": "max_tokens"}


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
    assert {"score_prompts", "_prefill", "probe"} <= names, names


def test_X1_max_tokens_is_one_and_min_tokens_is_never_set():
    for path, name, call in unique_calls():
        kw = {k.arg: k.value for k in call.keywords}
        where = f"{path.name}:{call.lineno} ({name})"
        assert "min_tokens" not in kw, where
        if name in FREE_LENGTH:
            assert isinstance(kw.get("max_tokens"), ast.Name) and kw["max_tokens"].id == FREE_LENGTH[name], where
        else:
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


def test_X1b_the_budget_is_a_validated_server_setting_not_a_request_field():
    from decisio.serve import think

    assert think.MAX_BUDGET == 4096 and think.check_budget(4096, "b") == 4096
    for bad in (4097, 15, True, "512", 512.0):
        try:
            think.check_budget(bad, "b")
        except ValueError:
            continue
        raise AssertionError(bad)
    route = (SRC / "serve" / "systemone.py").read_text()
    assert (
        "self.think_budget" in route
        and 'get("think")' in route
        and "budget" not in route.split('get("think")')[1][:200]
    )


def test_X3_no_request_controlled_engine_options_and_no_vllm_server():
    pattern = re.compile(
        r"mm_processor_kwargs|trust_remote_code|trust-remote-code|min_tokens|vllm\.entrypoints|vllm serve"
    )
    hits = [
        f"{p.relative_to(SRC)}:{i}"
        for p in sorted(SRC.rglob("*.py"))
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if pattern.search(line) and not (p.name == "gemma_images.py" and "image_budget" in line)
    ]  # gemma_images.py sets the soft-token budget from the server's own --image-budget flag, never from a request
    assert hits == [], hits


def test_X4_multimodal_ids_come_from_the_decoded_pixels():
    src = (SRC / "serve" / "image_engine.py").read_text()
    assert 'mm_uuids={"image": ids}' in src and "ids = [self.image_id(img) for img in images]" in src
    shared = (SRC / "serve" / "placed_images.py").read_text()  # the placed layout's ids come from the same function
    assert "ident = [image_id(img) for img in images]" in shared
    body = ast.get_source_segment(
        shared, next(n for n in ast.walk(ast.parse(shared)) if isinstance(n, ast.FunctionDef) and n.name == "image_id")
    )
    assert "hashlib.sha256(img.tobytes())" in body
    # the only places that put multi_modal_uuids on a prompt take the argument the image engine passes
    for path in sorted(SRC.rglob("*.py")):
        for line in path.read_text().splitlines():
            if '"multi_modal_uuids"' in line:
                assert "mm_uuids" in line, f"{path.name}: {line.strip()}"
