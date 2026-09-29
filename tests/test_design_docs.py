"""The design specification is the deliverable at this stage, so its integrity is tested.

These checks catch the failures that actually happen to a documentation-first
repository: a renamed heading that orphans a table-of-contents entry, a moved
file that breaks a relative link, and a diagram whose source no longer runs.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

DESIGN_DOCS = ("README.md", "system-design/README.md", "system-design/system-design.md")
#: Diagram sources under system-design/; each writes a PNG of the same name beside itself.
DIAGRAMS = ("honing_architecture", "honing_scope", "honing_compositions")

#: Markdown inline links, excluding images and external URLs.
LINK = re.compile(r"(?<!!)\[[^\]]+\]\((?P<target>[^)\s]+)\)")
#: ATX headings, for resolving in-page anchors.
HEADING = re.compile(r"^#{1,6}\s+(?P<text>.+?)\s*$", re.MULTILINE)


def anchor(heading: str) -> str:
    """GitHub's slug for a heading: lowercase, punctuation dropped, spaces hyphenated."""
    text = re.sub(r"`|\*|_", "", heading)
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", "-", text.strip()).lower()


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", DESIGN_DOCS)
def test_the_design_docs_exist_and_are_not_empty(repo_root: Path, relative_path: str) -> None:
    document = repo_root / relative_path
    assert document.is_file(), f"{relative_path} is missing"
    assert document.read_text(encoding="utf-8").strip(), f"{relative_path} is empty"


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", DESIGN_DOCS)
def test_every_relative_link_resolves(repo_root: Path, relative_path: str) -> None:
    document = repo_root / relative_path
    text = document.read_text(encoding="utf-8")
    anchors = {anchor(match["text"]) for match in HEADING.finditer(text)}

    broken = []
    for match in LINK.finditer(text):
        target = match["target"]
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path_part, _, fragment = target.partition("#")
        if path_part:
            if not (document.parent / path_part).exists():
                broken.append(f"missing file: {target}")
        elif fragment and fragment not in anchors:
            broken.append(f"missing anchor: #{fragment}")

    assert not broken, f"{relative_path} has unresolved links: {broken}"


@pytest.mark.unit
def test_the_readme_embeds_the_rendered_diagram(repo_root: Path) -> None:
    readme = (repo_root / "system-design/README.md").read_text(encoding="utf-8")
    assert "![" in readme and "honing_architecture.png" in readme
    assert (repo_root / "system-design/honing_architecture.png").is_file()


@pytest.mark.unit
@pytest.mark.parametrize("image", ["honing_scope.png", "honing_compositions.png"])
def test_the_top_level_readme_embeds_the_scope_diagrams(repo_root: Path, image: str) -> None:
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert f"](system-design/{image})" in readme
    assert (repo_root / "system-design" / image).is_file()


@pytest.mark.unit
@pytest.mark.parametrize("name", DIAGRAMS)
def test_the_diagram_source_writes_beside_itself(repo_root: Path, name: str) -> None:
    """The script must not hard-code an author's absolute path.

    The first one was written in a scratch directory and moved here, which is
    exactly how an absolute output path survives into a repository unnoticed.
    """
    source = (repo_root / f"system-design/{name}.py").read_text(encoding="utf-8")
    assert "Path(__file__)" in source, "the output path must be derived from __file__"
    assert "/tmp/" not in source and "/home/" not in source


@pytest.mark.slow
@pytest.mark.parametrize("name", DIAGRAMS)
def test_the_diagram_regenerates(repo_root: Path, tmp_path: Path, name: str) -> None:
    """Render the diagram in a copy, so a stale committed PNG cannot mask a broken script."""
    pytest.importorskip("matplotlib", reason="diagram rendering needs the 'docs' extra")

    script = tmp_path / f"{name}.py"
    script.write_text(
        (repo_root / f"system-design/{name}.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300)

    assert result.returncode == 0, f"diagram script failed:\n{result.stderr}"
    rendered = script.with_suffix(".png")
    assert rendered.is_file(), "the script did not write a PNG beside itself"
    assert rendered.stat().st_size > 10_000, "the rendered diagram is suspiciously small"
