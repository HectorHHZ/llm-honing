"""The task suite manifest: what a line must hold, how a bad line is reported, and how tasks are selected."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from honing.core import ContractError, Task
from honing.harness.tasks import load_task_suite
from tests.fakes.tasks import task_line, write_manifest


def _refusal(tmp_path: Path, lines: list[dict[str, object] | str]) -> str:
    with pytest.raises(ContractError) as refused:
        load_task_suite(write_manifest(tmp_path, lines))
    return str(refused.value)


# ---------------------------------------------------------------- acceptance: refused with the line number


def test_a_duplicate_task_id_is_refused_with_both_line_numbers(tmp_path: Path) -> None:
    message = _refusal(tmp_path, [task_line("a"), "", task_line("b"), task_line("a")])
    assert message.startswith(f"{tmp_path / 'manifest.jsonl'}:4: ")
    assert "duplicate task_id 'a', first on line 1" in message


def test_an_unknown_split_is_refused_with_the_line_number(tmp_path: Path) -> None:
    message = _refusal(tmp_path, [task_line("a"), task_line("b", split="test")])
    assert message.startswith(f"{tmp_path / 'manifest.jsonl'}:2: ")
    assert "split must be one of ['heldout', 'train', 'val'], got 'test'" in message


def test_a_line_without_a_workspace_key_is_refused_with_the_line_number(tmp_path: Path) -> None:
    line = task_line("b")
    del line["workspace"]
    message = _refusal(tmp_path, [task_line("a"), line])
    assert message.startswith(f"{tmp_path / 'manifest.jsonl'}:2: ")
    assert "missing keys ['workspace']" in message


def test_a_workspace_that_does_not_exist_is_refused_with_the_line_number(tmp_path: Path) -> None:
    message = _refusal(tmp_path, [task_line("a"), task_line("b", workspace="nowhere")])
    assert message.startswith(f"{tmp_path / 'manifest.jsonl'}:2: ")
    assert "workspace 'nowhere' is not a directory" in message


# ---------------------------------------------------------------- other refusals


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("{not json", "not valid JSON"),
        ("[1, 2]", "a manifest line must be a JSON object"),
        (task_line(extra=1), "unknown keys ['extra']"),
        (task_line(task_id=""), "task_id must be a non-empty string"),
        (task_line(prompt=""), "prompt must be a non-empty string"),
        (task_line(prompt=3), "prompt must be a string"),
        (task_line(grader="python3 grade.py"), "grader must be a non-empty list"),
        (task_line(grader=[]), "grader must be a non-empty list"),
        (task_line(grader=["python3", ""]), "grader must be a non-empty list"),
        (task_line(workspace="/tmp"), "workspace must be relative to the manifest"),
        (task_line(workspace="../outside"), "workspace may not leave the manifest's directory"),
        (task_line(grader_files="missing"), "grader_files 'missing' is not a directory"),
        (task_line(metadata=[1]), "metadata must be a JSON object"),
    ],
)
def test_a_malformed_line_is_refused_with_the_line_number(tmp_path: Path, line: object, reason: str) -> None:
    message = _refusal(tmp_path, [task_line("a"), line])
    assert message.startswith(f"{tmp_path / 'manifest.jsonl'}:2: ")
    assert reason in message


def test_a_manifest_with_no_tasks_is_refused(tmp_path: Path) -> None:
    assert "the manifest has no tasks" in _refusal(tmp_path, ["", "  "])


# ---------------------------------------------------------------- reading and selecting


def test_a_suite_reads_every_field_and_selects_by_split_in_file_order(tmp_path: Path) -> None:
    (tmp_path / "hidden").mkdir()
    path = write_manifest(
        tmp_path,
        [
            task_line("v2", split="val"),
            task_line("t1", grader_files="hidden", metadata={"k": [1, "x"]}),
            task_line("v1", split="val"),
            task_line("h1", split="heldout"),
        ],
    )
    suite = load_task_suite(path)
    entry = suite.entry("t1")
    assert entry.workspace == tmp_path / "workspaces" / "t1"
    assert entry.grader == ("python3", "grade.py")
    assert entry.grader_files == tmp_path / "hidden"
    assert entry.metadata == {"k": [1, "x"]}
    assert suite.entry("v2").grader_files is None
    assert suite.entry("v2").metadata == {}
    assert suite.tasks("val") == (Task("v2", "Fix v2."), Task("v1", "Fix v1."))
    assert suite.tasks("val", limit=1) == (Task("v2", "Fix v2."),)
    assert suite.tasks("train") == (Task("t1", "Fix t1."),)
    assert suite.split_of("h1") == "heldout"
    with pytest.raises(ContractError, match="not in the task suite"):
        suite.entry("nope")


def test_the_content_hash_changes_with_a_workspace_file_and_so_does_the_suite_checksum(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, [task_line("a"), task_line("b")])
    (tmp_path / "workspaces" / "a" / "main.py").write_text("x = 1\n")
    before = load_task_suite(path)
    assert before == load_task_suite(path)
    (tmp_path / "workspaces" / "a" / "main.py").write_text("x = 2\n")
    after = load_task_suite(path)
    assert after.entry("a").content_hash != before.entry("a").content_hash
    assert after.entry("b").content_hash == before.entry("b").content_hash
    assert after.checksum != before.checksum


def test_the_content_hash_covers_grader_files(tmp_path: Path) -> None:
    (tmp_path / "hidden").mkdir()
    path = write_manifest(tmp_path, [task_line("a", grader_files="hidden")])
    before = load_task_suite(path).entry("a").content_hash
    (tmp_path / "hidden" / "test_a.py").write_text("assert True\n")
    assert load_task_suite(path).entry("a").content_hash != before


# ---------------------------------------------------------------- disjoint selections


def test_selections_that_share_a_task_are_refused(tmp_path: Path) -> None:
    suite = load_task_suite(write_manifest(tmp_path, [task_line(f"v{n}", split="val") for n in range(4)]))
    val = suite.tasks("val")
    suite.assert_disjoint(val[:2], val[2:])
    # The RA2 dry run's shape: method tasks and gate tasks both drawn from the same split.
    with pytest.raises(ContractError, match="'v0' is in selections 0 and 1"):
        suite.assert_disjoint(val, val)


def test_a_held_out_task_may_not_be_selected(tmp_path: Path) -> None:
    suite = load_task_suite(write_manifest(tmp_path, [task_line("t"), task_line("h", split="heldout")]))
    with pytest.raises(ContractError, match="'h' is held out"):
        suite.assert_disjoint(suite.tasks("train"), suite.tasks("heldout"))


def test_a_selected_task_must_be_in_the_suite(tmp_path: Path) -> None:
    suite = load_task_suite(write_manifest(tmp_path, [task_line("t")]))
    with pytest.raises(ContractError, match="not in the task suite"):
        suite.assert_disjoint([Task("other", "p")])


SEPARATORS = "next line\u0085and\u2028paragraph"


@pytest.mark.parametrize("ending", ["\n", "\r\n"])
def test_unicode_line_separators_inside_a_line_do_not_split_it(tmp_path: Path, ending: str) -> None:
    write_manifest(tmp_path, [task_line("a"), task_line("b")])  # creates the workspaces
    lines = [task_line("a", prompt=SEPARATORS), task_line("b", prompt="Fix b.")]
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(line, ensure_ascii=False) + ending for line in lines), encoding="utf-8")
    assert SEPARATORS in path.read_text(encoding="utf-8")
    suite = load_task_suite(path)
    assert suite.entry("a").prompt == SEPARATORS
    assert [entry.task_id for entry in suite.entries] == ["a", "b"]


def test_a_bad_line_after_unicode_line_separators_is_reported_on_its_own_line(tmp_path: Path) -> None:
    write_manifest(tmp_path, [task_line("a")])
    path = tmp_path / "manifest.jsonl"
    text = json.dumps(task_line("a", prompt=SEPARATORS), ensure_ascii=False) + "\n" + "{not json\n"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ContractError) as refused:
        load_task_suite(path)
    assert str(refused.value).startswith(f"{path}:2: not valid JSON")


def test_an_integer_longer_than_python_converts_is_refused_with_the_line_number(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, [task_line("a")])
    path.write_text(path.read_text() + '{"task_id": 1' + "0" * 5000 + "}\n")
    with pytest.raises(ContractError) as refused:
        load_task_suite(path)
    assert str(refused.value).startswith(f"{path}:2: not valid JSON")
