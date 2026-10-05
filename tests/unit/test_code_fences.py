# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The rendered documentation names a language on every code fence (scripts/check_code_fences.py)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_code_fences", ROOT / "scripts" / "check_code_fences.py")
fences = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fences)


def test_a_bare_opening_fence_is_found_and_a_closing_one_is_not():
    text = "x\n```bash\nls\n```\n\n```\nbare\n```\n  ~~~\nindented bare\n  ~~~\n````json\n```\nnot a close\n````\n"
    assert fences.bare_fences(text) == [6, 9]


def test_the_rendered_documentation_has_no_bare_fence():
    found = {str(f.relative_to(ROOT)): fences.bare_fences(f.read_text()) for f in fences.default_files()}
    assert not {f: n for f, n in found.items() if n}
