"""An episode's decision traces as compact text for a model to reflect on.

GEPA's reflective dataset, Meta-Harness's trace files and System 0 read the
same rendering, and the same traces and cap always render to the same text.

The text opens with the episode, the release it ran under (or, when no
release is known, the first characters of the policy checksum), its outcome,
how many key events of each label it had and the questions asked. Then come
windows of consecutive decisions around the key events, measured in episode
time. An event's window holds the decision it followed and every decision
asked from ``before_seconds`` before the event to ``after_seconds`` after it,
and windows that overlap or touch merge into one. The defaults are sized for
a tank round: 2 s before an event reaches past a shell's flight of 1.1 to
1.7 s to the decisions that could have dodged it, and 0.5 s after shows how
the model went on. An episode without a key event shows the decisions asked
in the last ``before_seconds`` up to its last one. Each decision shows its
time, the probability of every option, the choice, what the controller
allowed and applied, and the events that followed it.

State is shown by field, a field being one sentence of a line of the state.
The first decision of each window lists every field, the constraints and the
applied action, so a reader can rebuild the state without the hidden
decisions. A later decision lists the fields that are new since the decision
shown before it in its window with the same questions, or all of them when
there is none. It also lists as gone each field of that decision that no new
field took the place of. A new field takes the place of an old one when both
begin with the same name: the first two words of the text before the first
digit, sign, or one of ``=:([{``. So "Me at 2." replaces "Me at 1.", and
"Enemy shell incoming." is gone when nothing begins with "Enemy shell". The
constraints and applied action are shown when they changed since the decision
shown before. A decision's text therefore depends on which decisions are shown
before it in its window.

The text never exceeds ``max_tokens`` as :func:`estimate_tokens` counts. Over
the cap, decisions leave the windows one at a time: those around events with a
higher priority number first, and among equals the decision asked longest
before or after the one its event followed first, then the one more decisions
away from it, then the earliest. A decision that leaves from inside a window
splits it in two. The decision an event of priority 0 followed always stays,
and in an episode without such an event, the decision that would leave last.
If those decisions are still over the cap, the windows are filled again
without state fields, and as a last resort the text is cut at the cap.

This module imports only the standard library and :mod:`honing.core`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from honing.core import DecisionTrace, require

DEFAULT_MAX_TOKENS = 6000
DEFAULT_BEFORE_SECONDS = 2.0
DEFAULT_AFTER_SECONDS = 0.5
# Enough of a policy checksum to tell policies apart: the tank runner names run directories with as many.
POLICY_CHECKSUM_CHARS = 12
CUT_MARK = "[cut at the token cap]"
_TOKEN = re.compile(r"\w+|[^\w\s]")
_SENTENCE_END = re.compile(r"(?<=\.)\s+")
_NAME_END = re.compile(r"[\d=:(\[{+-]")

Fields = list[tuple[str, list[str]]]


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
    before_seconds: float = DEFAULT_BEFORE_SECONDS,
    after_seconds: float = DEFAULT_AFTER_SECONDS,
) -> str:
    """One episode's traces, in step order, as text of at most ``max_tokens`` estimated tokens."""
    require(max_tokens > estimate_tokens(CUT_MARK), f"max_tokens must exceed {estimate_tokens(CUT_MARK)}")
    require(
        all(math.isfinite(seconds) and seconds >= 0.0 for seconds in (before_seconds, after_seconds)),
        "a window's before_seconds and after_seconds are finite and >= 0",
    )
    if not traces:
        return "No decisions were recorded in this episode.\n"
    require(
        [trace.step for trace in traces] == list(range(len(traces)))
        and len({trace.episode_id for trace in traces}) == 1,
        "render one episode's traces, every step in order",
    )
    times = [trace.time_seconds for trace in traces]
    require(times == sorted(times), "an episode's decisions are in time order")
    ranks = _ranks(traces, before_seconds, after_seconds)
    # Highest rank first, and among equal ranks the earliest decision first.
    leaving = sorted((i for i in ranks if ranks[i] != (0, 0.0, 0)), key=lambda i: (ranks[i], -i), reverse=True)
    if len(leaving) == len(ranks):
        leaving.pop()
    fields = [_fields(trace) for trace in traces]
    for with_fields in (True, False):
        layout = _Layout(traces, fields if with_fields else None)
        shown = sorted(ranks)
        for step in leaving:
            if layout.tokens(shown) <= max_tokens:
                break
            shown.remove(step)
        text = "\n".join(layout.lines(shown)) + "\n"
        if estimate_tokens(text) <= max_tokens:
            return text
    return _cut(text, max_tokens)


def _ranks(
    traces: Sequence[DecisionTrace], before_seconds: float, after_seconds: float
) -> dict[int, tuple[int, float, int]]:
    """Every decision in a window, ranked by its event's priority, then by its distance from the decision it followed.

    The distance is first the time between the two decisions, then the number
    of decisions between them, so the decision an event followed ranks first
    in its window and one asked at the same time comes next.
    """
    anchors = [
        (step, event.time_seconds, event.priority)
        for step, trace in enumerate(traces)
        for event in trace.events
        if event.priority is not None
    ]
    if not anchors:
        anchors = [(len(traces) - 1, traces[-1].time_seconds, 0)]
    ranks: dict[int, tuple[int, float, int]] = {}
    for anchor, time, priority in anchors:
        first = anchor
        while first > 0 and traces[first - 1].time_seconds >= time - before_seconds:
            first -= 1
        last = anchor
        while last + 1 < len(traces) and traces[last + 1].time_seconds <= time + after_seconds:
            last += 1
        asked = traces[anchor].time_seconds
        for step in range(first, last + 1):
            rank = (priority, abs(traces[step].time_seconds - asked), abs(step - anchor))
            ranks[step] = min(ranks.get(step, rank), rank)
    return ranks


class _Layout:
    """The text of any set of shown decisions. A decision's lines depend on where its window starts.

    ``fields`` holds each decision's state fields, or is ``None`` to leave the
    state out. A decision's lines depend on where its window starts only
    through two facts: whether the window holds the decision before it, and
    whether it holds the last one before it with the same questions. So each
    decision's lines and their token count are worked out at most four times,
    and trying many sets of shown decisions stays cheap.
    """

    def __init__(self, traces: Sequence[DecisionTrace], fields: Sequence[Fields] | None) -> None:
        self._traces = traces
        self._fields = fields
        self._intro, self._key_events = _intro(traces)
        self._questions = _questions(traces)
        self._fixed_tokens = sum(estimate_tokens(line) for line in (*self._intro, *self._questions))
        # For each decision, the last one before it with the same questions, or -1.
        last: dict[tuple[str, ...], int] = {}
        self._same_before: list[int] = []
        for step, trace in enumerate(traces):
            self._same_before.append(last.get(_asked(trace), -1))
            last[_asked(trace)] = step
        self._blocks: dict[tuple[int, bool, bool], tuple[list[str], int]] = {}

    def tokens(self, shown: Sequence[int]) -> int:
        """The estimated tokens of :meth:`lines`, which equal the sum of each line's, since no token spans a line."""
        total = self._fixed_tokens + estimate_tokens(self._shown(len(shown)))
        for step, start, gap in self._walk(shown):
            total += self._block(step, start)[1] + (estimate_tokens(gap) if gap else 0)
        return total + sum(estimate_tokens(gap) for gap in self._tail(shown))

    def lines(self, shown: Sequence[int]) -> list[str]:
        lines = [*self._intro, self._shown(len(shown)), *self._questions]
        for step, start, gap in self._walk(shown):
            if gap:
                lines.append(gap)
            lines.extend(self._block(step, start)[0])
        return lines + self._tail(shown)

    def _walk(self, shown: Sequence[int]) -> list[tuple[int, int, str | None]]:
        """Each shown step with the step its window starts at, and the gap line before it if one is hidden."""
        walked = []
        previous, start = -1, 0
        for step in shown:
            gap = None
            if step > previous + 1:
                gap, start = _gap(self._traces, previous + 1, step - 1), step
            walked.append((step, start, gap))
            previous = step
        return walked

    def _tail(self, shown: Sequence[int]) -> list[str]:
        last = shown[-1] if shown else -1
        return [_gap(self._traces, last + 1, len(self._traces) - 1)] if last < len(self._traces) - 1 else []

    def _shown(self, shown: int) -> str:
        around = "around the key events" if self._key_events else "at the end of the episode"
        state = (
            "The first in each window lists the whole state, and each later one the state fields that changed"
            " since the decision shown before it with the same questions."
            if self._fields is not None
            else "State fields are left out to stay within the token cap."
        )
        return f"Shown: {_count(shown, 'decision')} {around}. {state}"

    def _block(self, step: int, start: int) -> tuple[list[str], int]:
        key = (step, step > start, self._same_before[step] >= start)
        if key not in self._blocks:
            lines = _block(self._traces, self._fields, step, start)
            self._blocks[key] = (lines, sum(estimate_tokens(line) for line in lines))
        return self._blocks[key]


def _block(traces: Sequence[DecisionTrace], fields: Sequence[Fields] | None, step: int, start: int) -> list[str]:
    """One decision's lines, in a window that starts at ``start`` and shows every decision from there on."""
    trace = traces[step]
    lines = [f"decision {step} at {trace.time_seconds:.3f} s:"]
    if fields is not None:
        asked = _asked(trace)
        same = next((j for j in range(step - 1, start - 1, -1) if _asked(traces[j]) == asked), None)
        lines.extend(_state(fields[step], None if same is None else fields[same]))
    lines.extend(_answers(trace))
    shown_before = traces[step - 1] if step > start else None
    controller = (trace.constraints, trace.applied)
    if shown_before is None or (shown_before.constraints, shown_before.applied) != controller:
        lines.extend(
            f"  {name}: {value}" for name, value in zip(("constraints", "applied"), controller, strict=True) if value
        )
    lines.extend(_events(trace))
    return lines


def _asked(trace: DecisionTrace) -> tuple[str, ...]:
    return tuple(key for key, _ in trace.decision.questions)


def _intro(traces: Sequence[DecisionTrace]) -> tuple[list[str], bool]:
    """The header's lines on the episode, its outcome and its key events, and whether it had any."""
    first = traces[0]
    names = [_count(len(traces), "decision")]
    if first.release_id:
        names.append(f"release {first.release_id}")
    elif first.policy_checksum:
        # The release names the policy and the harness; without it, the weights' checksum names the policy.
        names.append(f"policy checksum {first.policy_checksum[:POLICY_CHECKSUM_CHARS]}")
    names += [f"harness {first.harness_id}"] if first.harness_id else []
    score = f" Score {first.grade.score:g}." if first.grade is not None else ""
    counts: Counter[str] = Counter()
    lowest: dict[str, int] = {}
    for trace in traces:
        for event in trace.events:
            if event.label is not None and event.priority is not None:
                counts[event.label] += 1
                lowest[event.label] = min(lowest.get(event.label, event.priority), event.priority)
    # Labels by priority, then in the order they first happened.
    labels = sorted(counts, key=lambda label: lowest[label])
    found = ", ".join(f"{label} ({counts[label]})" for label in labels)
    lines = [
        f"Episode {first.episode_id}: {', '.join(names)}.",
        f"Outcome: {first.outcome or 'not recorded'}.{score}",
        f"Key events: {found or 'none'}.",
    ]
    return lines, bool(counts)


def _questions(traces: Sequence[DecisionTrace]) -> list[str]:
    asked: dict[str, str] = {}
    for trace in traces:
        for key, question in trace.decision.questions:
            if key not in asked:
                asked[key] = f"  {key}, {question.kind} of {len(question.options)} options: {question.instructions}"
    return ["Questions:", *asked.values()]


def _gap(traces: Sequence[DecisionTrace], first: int, last: int) -> str:
    start, end = traces[first].time_seconds, traces[last].time_seconds
    when = f"at {start:.3f} s" if first == last else f"{start:.3f} s to {end:.3f} s"
    return f"({_count(last - first + 1, 'decision')} not shown, {when})"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _fields(trace: DecisionTrace) -> Fields:
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


def _state(current: Fields, before: Fields | None) -> list[str]:
    """Each section's fields: all of them without ``before``, else those new since it and those gone from it."""
    lines = []
    if before is None:
        for section, items in current:
            if items:
                lines.append(f"  {section}, all {_count(len(items), 'field')}:")
                lines.extend(f"    {field}" for field in items)
        return lines
    now, earlier = dict(current), dict(before)
    for section in [*now, *(section for section in earlier if section not in now)]:
        items = now.get(section, [])
        new, gone = _diff(items, earlier.get(section, []))
        if not items and not gone:
            continue
        summary = f"{len(new)} of {_count(len(items), 'field')} changed{f', {len(gone)} gone' if gone else ''}"
        lines.append(f"  {section}, {summary}{':' if new or gone else ''}")
        lines.extend(f"    {field}" for field in new)
        lines.extend(f"    gone: {field}" for field in gone)
    return lines


def _diff(items: Sequence[str], old: Sequence[str]) -> tuple[list[str], list[str]]:
    """The fields new since ``old``, and the fields of ``old`` that no new field took the place of."""
    present, earlier = set(items), set(old)
    new = [field for field in items if field not in earlier]
    room = Counter(_name(field) for field in new)
    gone = []
    for field in old:
        if field in present:
            continue
        if room[_name(field)]:
            room[_name(field)] -= 1
        else:
            gone.append(field)
    return new, gone


def _name(field: str) -> str:
    """What a field is about: the first two words of its text before the first digit, sign, or one of ``=:([{``."""
    return " ".join(_NAME_END.split(field, maxsplit=1)[0].split()[:2])


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
            f"  then {event.time_seconds:.3f} s, {event.label}: {event.text}"
            if event.label is not None
            else f"  then {event.time_seconds:.3f} s: {event.text}"
        )
        for event in trace.events
    ]


def _cut(text: str, max_tokens: int) -> str:
    """The text up to its last token within the cap, keeping room for the mark that says so."""
    tokens = list(_TOKEN.finditer(text))
    end = tokens[max_tokens - estimate_tokens(CUT_MARK) - 1].end()
    return f"{text[:end]}\n{CUT_MARK}\n"


__all__ = [
    "CUT_MARK",
    "DEFAULT_AFTER_SECONDS",
    "DEFAULT_BEFORE_SECONDS",
    "DEFAULT_MAX_TOKENS",
    "POLICY_CHECKSUM_CHARS",
    "estimate_tokens",
    "render_traces",
]
