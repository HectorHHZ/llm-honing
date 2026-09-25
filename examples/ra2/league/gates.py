"""Promotion rules consume actual learned-policy match outcomes, never proxies."""

import math
from collections import defaultdict


def wilson_lower(wins, total, z=1.959963984540054):
    if total <= 0:
        return 0.0
    p = wins / total
    return (p + z * z / (2 * total) - z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / (
        1 + z * z / total
    )


def tick_budget_matches(record, spec):
    """Longer-horizon diagnostics must not count toward a shorter-budget gate."""
    return (
        record.get("maxTicks") == spec["maxTicks"]
        and type(record.get("tick")) is int
        and 0 <= record["tick"] <= spec["maxTicks"]
    )


def evaluate_gate(records, protocol):
    spec = protocol["sealedTest"]
    expected = {(m, s, seat) for m in spec["maps"] for s in spec["seeds"] for seat in spec["seats"]}
    actual = [(r["map"], r["seed"], r["seat"]) for r in records]
    failures = []
    if len(actual) != len(set(actual)) or set(actual) != expected:
        failures.append("incomplete_or_duplicate_matrix")
    if len(records) < spec["minimumGames"]:
        failures.append("insufficient_games")
    wins = 0
    maps, seats = defaultdict(list), defaultdict(list)
    champions = set()
    for r in records:
        valid = (
            r.get("controller") == "model"
            and r.get("normalView") is True
            and r.get("fallbackCount") == 0
            and r.get("auditPassed") is True
            and r.get("replayVerified") is True
            and r.get("error") is None
        )
        if not valid:
            failures.append("unverified_or_non_model_match")
        if any(
            r.get(key) != protocol[key]
            for key in ("engineCommit", "baselineRevision", "cadenceTicks", "engineActionCap")
        ):
            failures.append("protocol_mismatch")
        if not tick_budget_matches(r, spec):
            failures.append("tick_budget_mismatch")
        if r.get("decisionInterval") != protocol["cadenceTicks"]:
            failures.append("decision_cadence_mismatch")
        if r.get("split") != "sealedTest" or not r.get("checkpointSha256"):
            failures.append("missing_test_provenance")
        champions.add(r.get("checkpointSha256"))
        win = valid and r.get("ended") is True and not r.get("truncated") and r.get("outcome", {}).get(r["seat"]) == 1
        wins += win
        maps[r["map"]].append(win)
        seats[r["seat"]].append(win)
    if len(champions) != 1:
        failures.append("mixed_checkpoints")
    rate = wins / len(records) if records else 0.0
    lower = wilson_lower(wins, len(records))
    if rate < spec["minimumWinRate"] or lower < spec["minimumWilsonLower95"]:
        failures.append("overall_win_gate")
    if any(sum(v) / len(v) < spec["minimumPerMapWinRate"] for v in maps.values()):
        failures.append("map_win_gate")
    if any(sum(v) / len(v) < spec["minimumPerSeatWinRate"] for v in seats.values()):
        failures.append("seat_win_gate")
    return {
        "passed": not failures,
        "games": len(records),
        "wins": wins,
        "winRate": rate,
        "wilsonLower95": lower,
        "failures": sorted(set(failures)),
    }
