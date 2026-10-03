"""A harness method that proposes a fixed list of edits, one per move.

It has no search and no model: each harness move proposes the next edit on the
list (a file to add or replace), and the gate decides. It is the baseline an
evolution method has to beat (a person's edit, judged by the same gate), and
the way to put a specific harness change, such as a new skill or a different
toolset, through a campaign on purpose.

A rejected edit is not retried; the next move proposes the next edit.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from honing.core import (
    EvaluationResult,
    EvolveContext,
    HarnessFile,
    HarnessFileKind,
    HarnessProposal,
    HarnessSpec,
    MethodRequirements,
    ProposeResult,
    SelectionDecision,
    require,
)
from honing.release import WinLossSelector

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ScriptedEdit:
    """Set ``path`` to ``content`` (added, or replaced if present), with a note saying why."""

    path: str
    kind: HarnessFileKind
    content: str
    note: str = ""


class ScriptedMethod:
    """Satisfies :class:`~honing.core.EvolveMethod` by proposing ``edits`` in order."""

    name = "scripted"
    state_schema_version = SCHEMA_VERSION

    def __init__(self, edits: Sequence[ScriptedEdit], min_win_margin: int = 0) -> None:
        require(len(edits) > 0, "a scripted method needs at least one edit")
        self.edits = tuple(edits)
        self.requires = MethodRequirements(file_kinds=frozenset(edit.kind for edit in self.edits))
        self.selector = WinLossSelector(min_win_margin)

    def initial_state(self, seed: HarnessSpec) -> Mapping[str, object]:
        return {"schema_version": SCHEMA_VERSION, "next": 0, "verdicts": []}

    def propose(self, context: EvolveContext) -> ProposeResult:
        index = int(str(context.state["next"]))
        state = {**context.state, "next": index + 1}
        if index >= len(self.edits):
            return ProposeResult(proposal=None, state=context.state)
        edit = self.edits[index]
        spec = context.current.with_file(edit.path, HarnessFile(kind=edit.kind, content=edit.content))
        if spec == context.current:
            return ProposeResult(proposal=None, state=state)
        return ProposeResult(proposal=HarnessProposal(spec=spec, note=edit.note or f"set {edit.path}"), state=state)

    def record(
        self,
        state: Mapping[str, object],
        proposal: HarnessProposal,
        evaluation: EvaluationResult,
        decision: SelectionDecision,
    ) -> Mapping[str, object]:
        verdicts = [*list(state["verdicts"]), decision.outcome]  # type: ignore[call-overload]
        return {**state, "verdicts": verdicts}


def method(context: object, edits: Sequence[Mapping[str, object]], min_win_margin: int = 0) -> ScriptedMethod:
    """Factory for a pipeline config. An edit is ``{"path", "kind", "content" or "json", "note"}``.

    ``json`` takes a JSON value and writes it as the file's content, for settings files.
    """
    parsed = []
    for edit in edits:
        content = edit.get("content")
        if content is None:
            content = json.dumps(edit["json"], indent=2) + "\n"
        parsed.append(
            ScriptedEdit(
                path=str(edit["path"]),
                kind=edit["kind"],  # type: ignore[arg-type]
                content=str(content),
                note=str(edit.get("note", "")),
            )
        )
    return ScriptedMethod(parsed, min_win_margin)
