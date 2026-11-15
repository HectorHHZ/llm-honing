"""An episode's decision traces as compact text for a model to reflect on.

GEPA's reflective dataset, Meta-Harness's trace files and System 0 read the
same rendering, and the same traces always render to the same text.

The text opens with the episode, its outcome, how many key events it had and
the questions asked. Then come windows of consecutive decisions around the key
events (deaths, self-hits and missed shots). A window holds ``before``
decisions before the one an event followed, that decision, and ``after``
decisions after it, and windows that overlap merge. An episode without a key
event shows its last decisions. Each decision shows its time, the probability
of every option, the choice, what the controller allowed and applied when that
changed, and the events that followed it.

State is shown by field, a field being one sentence of a line of the state.
A decision lists only the fields that changed since the decision before it
with the same questions, and the first decision of each set of questions lists
them all. The controller's constraints and applied action are shown when they
changed since the decision before. So a decision's text does not depend on
which other decisions are shown.

The text never exceeds ``max_tokens`` as :func:`estimate_tokens` counts. Over
the cap, decisions leave the windows one at a time: missed shots' windows
before deaths', and in each the decision farthest from its event first. The
decision a death or a self-hit followed always stays. If those decisions are
still over the cap, the windows are filled again without state fields, and as
a last resort the text is cut at the cap.

This module imports only the standard library and :mod:`honing.core`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from honing.core import DecisionTrace, KeyEvent, require

DEFAULT_MAX_TOKENS = 6000
DEFAULT_BEFORE = 4
DEFAULT_AFTER = 2
CUT_MARK = "[cut at the token cap]"
# A window's class: windows around deaths and self-hits are kept over those around missed shots.
_CLASS: dict[KeyEvent, int] = {"death": 0, "self_hit": 0, "missed_shot": 1}
_LABEL: dict[KeyEvent, str] = {"death": "death", "self_hit": "self-hit", "missed_shot": "missed shot"}
_TOKEN = re.compile(r"\w+|[^\w\s]")
_SENTENCE_END = re.compile(r"(?<=\.)\s+")


def estimate_tokens(text: str) -> int:
    """Count each word, number and punctuation mark as one token.

    It needs no tokenizer and gives the same count everywhere. A subword
    tokenizer splits long words and merges some punctuation, so its count for
    the same text differs. The cap bounds this estimate.
    """
    return len(_TOKEN.findall(text))


def render_traces(
    traces: Sequence[DecisionTrace],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    before: int = DEFAULT_BEFORE,
    after: int = DEFAULT_AFTER,
) -> str:
    """One episode's traces, in step order, as text of at most ``max_tokens`` estimated tokens."""
    require(max_tokens > estimate_tokens(CUT_MARK), f"max_tokens must exceed {estimate_tokens(CUT_MARK)}")
    require(before >= 0 and after >= 0, "a window's before and after are counts of decisions, >= 0")
    if not traces:
        return "No decisions were recorded in this episode.\n"
    require(
        [trace.step for trace in traces] == list(range(len(traces)))
        and len({trace.episode_id for trace in traces}) == 1,
        "render one episode's traces, every step in order",
    )
    ranks = _ranks(traces, before, after)
    # Highest rank first, and among equal ranks the earliest decision first.
    leaving = sorted((i for i in ranks if ranks[i] != (0, 0)), key=lambda i: (ranks[i], -i), reverse=True)
    for fields in (True, False):
        text = _fit(traces, _blocks(traces, fields=fields), sorted(ranks), leaving, max_tokens, fields=fields)
        if estimate_tokens(text) <= max_tokens:
            return text
    return _cut(text, max_tokens)


def _fit(
    traces: Sequence[DecisionTrace],
    blocks: Sequence[Sequence[str]],
    shown: list[int],
    leaving: Sequence[int],
    max_tokens: int,
    *,
    fields: bool,
) -> str:
    """The text of the ``shown`` decisions, which leave in ``leaving``'s order until it is within the cap."""
    text = _join(traces, blocks, shown, fields=fields)
    for step in leaving:
        if estimate_tokens(text) <= max_tokens:
            break
        shown.remove(step)
        text = _join(traces, blocks, shown, fields=fields)
    return text


def _ranks(traces: Sequence[DecisionTrace], before: int, after: int) -> dict[int, tuple[int, int]]:
    """Every decision in a window, ranked by its window's class, then by its distance from the window's event."""
    events = [(step, _CLASS[event.key]) for step, trace in enumerate(traces) for event in trace.events if event.key]
    if not events:
        events = [(len(traces) - 1, 0)]
    ranks: dict[int, tuple[int, int]] = {}
    for anchor, kind in events:
        for step in range(max(0, anchor - before), min(len(traces), anchor + after + 1)):
            ranks[step] = min(ranks.get(step, (kind, abs(step - anchor))), (kind, abs(step - anchor)))
    return ranks


def _blocks(traces: Sequence[DecisionTrace], *, fields: bool) -> list[list[str]]:
    """Each decision's lines. What it shows of the state and the controller is what changed since the one before."""
    blocks = []
    states: dict[tuple[str, ...], set[tuple[str, str]]] = {}
    controller: tuple[str | None, str | None] | None = None
    for step, trace in enumerate(traces):
        lines = [f"decision {step} at {trace.time_seconds:.3f} s:"]
        asked = tuple(key for key, _ in trace.decision.questions)
        current = _fields(trace)
        if fields:
            lines.extend(_state(current, states.get(asked)))
        states[asked] = {(section, field) for section, items in current for field in items}
        lines.extend(_answers(trace))
        if (trace.constraints, trace.applied) != controller:
            controller = (trace.constraints, trace.applied)
            lines.extend(
                f"  {name}: {value}"
                for name, value in zip(("constraints", "applied"), controller, strict=True)
                if value
            )
        lines.extend(_events(trace))
        blocks.append(lines)
    return blocks


def _join(
    traces: Sequence[DecisionTrace], blocks: Sequence[Sequence[str]], shown: Sequence[int], *, fields: bool
) -> str:
    lines = _header(traces, len(shown), fields=fields)
    previous = -1
    for step in shown:
        if step > previous + 1:
            lines.append(_gap(traces, previous + 1, step - 1))
        lines.extend(blocks[step])
        previous = step
    if previous < len(traces) - 1:
        lines.append(_gap(traces, previous + 1, len(traces) - 1))
    return "\n".join(lines) + "\n"


def _header(traces: Sequence[DecisionTrace], shown: int, *, fields: bool) -> list[str]:
    first = traces[0]
    names = [_count(len(traces), "decision")]
    names += [f"release {first.release_id}"] if first.release_id else []
    names += [f"harness {first.harness_id}"] if first.harness_id else []
    score = f" Score {first.grade.score:g}." if first.grade is not None else ""
    counts = dict.fromkeys(_LABEL, 0)
    for trace in traces:
        for event in trace.events:
            if event.key:
                counts[event.key] += 1
    found = [_count(n, _LABEL[key]) for key, n in counts.items() if n]
    around = "around the key events" if found else "at the end of the episode"
    state = (
        "Each lists the state fields that changed since the decision before it with the same questions."
        if fields
        else "State fields are left out to stay within the token cap."
    )
    lines = [
        f"Episode {first.episode_id}: {', '.join(names)}.",
        f"Outcome: {first.outcome or 'not recorded'}.{score}",
        f"Key events: {', '.join(found) if found else 'none'}.",
        f"Shown: {_count(shown, 'decision')} {around}. {state}",
        "Questions:",
    ]
    asked: dict[str, str] = {}
    for trace in traces:
        for key, question in trace.decision.questions:
            if key not in asked:
                asked[key] = f"  {key}, {question.kind} of {len(question.options)} options: {question.instructions}"
    return lines + list(asked.values())


def _gap(traces: Sequence[DecisionTrace], first: int, last: int) -> str:
    start, end = traces[first].time_seconds, traces[last].time_seconds
    when = f"at {start:.3f} s" if first == last else f"{start:.3f} s to {end:.3f} s"
    return f"({_count(last - first + 1, 'decision')} not shown, {when})"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _fields(trace: DecisionTrace) -> list[tuple[str, list[str]]]:
    """The state's fields by section, each listed once: its lines split at the end of every sentence."""
    return [
        (
            section,
            list(
                dict.fromkeys(
                    field
                    for line in lines
                    for row in line.split("\n")
                    for field in _SENTENCE_END.split(row.strip())
                    if field
                )
            ),
        )
        for section, lines in trace.decision.parts
    ]


def _state(current: Sequence[tuple[str, Sequence[str]]], before: set[tuple[str, str]] | None) -> list[str]:
    lines = []
    for section, items in current:
        if not items:
            continue
        changed = [field for field in items if before is None or (section, field) not in before]
        total = _count(len(items), "field")
        which = f"all {total}" if before is None else f"{len(changed)} of {total} changed"
        lines.append(f"  {section}, {which}{':' if changed else ''}")
        lines.extend(f"    {field}" for field in changed)
    return lines


def _answers(trace: DecisionTrace) -> list[str]:
    probabilities = dict(trace.decision.behavior_probs or ())
    chosen = dict(trace.decision.chosen)
    return [
        f"  {key}: {', '.join(f'{option} {p:.3f}' for option, p in probabilities[key])}; chose {chosen[key]}"
        for key, _ in trace.decision.questions
    ]


def _events(trace: DecisionTrace) -> list[str]:
    return [
        (
            f"  then {event.time_seconds:.3f} s, {_LABEL[event.key]}: {event.text}"
            if event.key
            else f"  then {event.time_seconds:.3f} s: {event.text}"
        )
        for event in trace.events
    ]


def _cut(text: str, max_tokens: int) -> str:
    """The text up to its last token within the cap, keeping room for the mark that says so."""
    tokens = list(_TOKEN.finditer(text))
    end = tokens[max_tokens - estimate_tokens(CUT_MARK) - 1].end()
    return f"{text[:end]}\n{CUT_MARK}\n"


__all__ = ["CUT_MARK", "DEFAULT_AFTER", "DEFAULT_BEFORE", "DEFAULT_MAX_TOKENS", "estimate_tokens", "render_traces"]
