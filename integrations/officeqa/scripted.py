"""A harness method that proposes a fixed list of edits, one per move.

It has no search and no model; the gate judges each edit. It is the baseline a
search method must beat, and the way to put a chosen change, such as a new skill
or another toolset, through a campaign. A rejected edit is not retried.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from honing.core import (
    ContractError,
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
    """Sets one harness file, adding it or replacing it.

    Attributes:
        path: The file's path in the harness.
        kind: The file's kind.
        content: The file's new content.
        note: Why, recorded on the proposal.
    """

    path: str
    kind: HarnessFileKind
    content: str
    note: str = ""


class ScriptedMethod:
    """Satisfies :class:`~honing.core.EvolveMethod` by proposing ``edits`` in order.

    Args:
        edits: The edits, one per harness move.
        min_win_margin: The win/loss selector's margin.

    Raises:
        ContractError: ``edits`` is empty.
    """

    name = "scripted"
    state_schema_version = SCHEMA_VERSION

    def __init__(self, edits: Sequence[ScriptedEdit], min_win_margin: int = 0) -> None:
        require(len(edits) > 0, "a scripted method needs at least one edit")
        self._edits = tuple(edits)
        self.requires = MethodRequirements(file_kinds=frozenset(edit.kind for edit in self._edits))
        self.selector = WinLossSelector(min_win_margin)

    @property
    def edits(self) -> tuple[ScriptedEdit, ...]:
        """The edits, in the order they are proposed."""
        return self._edits

    def initial_state(self, seed: HarnessSpec) -> Mapping[str, object]:
        """Returns the state before any edit: the next edit to propose, and no verdicts yet."""
        return {"schema_version": SCHEMA_VERSION, "next": 0, "verdicts": []}

    def propose(self, context: EvolveContext) -> ProposeResult:
        """Proposes the next edit applied to the head's harness, or nothing once all are spent.

        Args:
            context: The move's context.

        Returns:
            The proposal, or none when the edits are spent or the edit changes nothing.
        """
        index = int(str(context.state["next"]))
        if index >= len(self._edits):
            return ProposeResult(proposal=None, state=context.state)
        edit = self._edits[index]
        state = {**context.state, "next": index + 1}
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
        """Appends the gate's verdict to the state.

        Args:
            state: The state ``propose`` returned.
            proposal: The proposal the gate judged.
            evaluation: The gate's evaluation.
            decision: The gate's decision.

        Returns:
            The new state.
        """
        return {**state, "verdicts": [*list(state["verdicts"]), decision.outcome]}  # type: ignore[call-overload]


def method(context: object, edits: Sequence[Mapping[str, object]], min_win_margin: int = 0) -> ScriptedMethod:
    """Builds the method for a pipeline config.

    Args:
        context: The pipeline context; its ``root`` fills ``{root}`` in file paths.
        edits: Each ``{"path", "kind", "note"}`` plus exactly one of ``content``
            (text), ``json`` (a value written as JSON, for settings), or
            ``file`` (a local file whose text is used).
        min_win_margin: The win/loss selector's margin.

    Returns:
        The method.

    Raises:
        ContractError: An edit has none or several of ``content``, ``json``, and ``file``.
    """
    root = str(getattr(context, "root", "."))
    parsed = []
    for edit in edits:
        sources = [key for key in ("content", "json", "file") if key in edit]
        if len(sources) != 1:
            raise ContractError(f"edit {edit.get('path')!r} needs exactly one of content, json, and file")
        if sources == ["json"]:
            content = json.dumps(edit["json"], indent=2) + "\n"
        elif sources == ["file"]:
            content = Path(os.path.expandvars(str(edit["file"]).format(root=root))).expanduser().read_text("utf-8")
        else:
            content = str(edit["content"])
        parsed.append(
            ScriptedEdit(
                path=str(edit["path"]),
                kind=edit["kind"],  # type: ignore[arg-type]
                content=content,
                note=str(edit.get("note", "")),
            )
        )
    return ScriptedMethod(parsed, min_win_margin)
