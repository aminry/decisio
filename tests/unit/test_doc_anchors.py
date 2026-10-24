# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Links in the rendered documentation resolve to a file and a heading (scripts/check_doc_anchors.py)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_doc_anchors", ROOT / "scripts" / "check_doc_anchors.py")
anchors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(anchors)


def test_the_anchor_of_a_heading_follows_githubs_rule():
    assert anchors.slug("Read twice (`--read-twice-under N`, `--read-twice-max-questions R`)") == (
        "read-twice---read-twice-under-n---read-twice-max-questions-r"
    )
    assert anchors.slug("The 12B in two precisions") == "the-12b-in-two-precisions"
    assert anchors.slug("What [this](x.md) is, in <b>one</b> line") == "what-this-is-in-one-line"


def test_a_repeated_heading_gets_a_counter_and_a_fenced_one_is_not_a_heading():
    text = "# A\n\n## Same\n\ntext\n\n## Same\n\n```bash\n# not a heading\n```\n"
    assert anchors.anchors(text) == {"a", "same", "same-1"}


def test_a_broken_file_and_a_broken_heading_are_found_and_the_good_links_pass(tmp_path):
    (tmp_path / "other.md").write_text("# Title\n\n## Section two\n")
    page = tmp_path / "page.md"
    page.write_text(
        "# Page\n\n## Own heading\n\n"
        "[ok](other.md#section-two) [ok-same](#own-heading) [web](https://x.test/#nope)\n"
        "[missing file](nope.md#x) [missing heading](other.md#section-three) [missing own](#nothing)\n"
        "```text\n[in a fence](nope.md#x)\n```\n"
    )
    found = anchors.broken(page, {})
    assert [(n, link, why) for n, link, why in found] == [
        (6, "nope.md#x", "no such file"),
        (6, "other.md#section-three", "no such heading"),
        (6, "#nothing", "no such heading"),
    ]


def test_the_rendered_documentation_has_no_broken_link():
    cache = {}
    found = {str(f.relative_to(ROOT)): anchors.broken(f, cache) for f in anchors.default_files()}
    assert not {f: b for f, b in found.items() if b}
