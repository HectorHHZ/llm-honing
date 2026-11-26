"""The full-HK raw-key adapter of the common Exp 0 tools (integrations/_shared/exp0/adapter.py): `game_id`, `rerender`, `replay`.

  --adapter integrations/hollow-knight-forestdango/exp0/v0/hk_adapter.py:HKKeysAdapter      (relative to the repository root, or an absolute path)

rerender(state, history): the request the CLEAN harness renders at one decision, from the exact harness state saved
right before it (public_states.jsonl line = state_codec.snapshot of the Agent: its view of the visible observation,
its memory and its numbers). The harness snapshot named in the state is loaded from this directory
(harness/<name>/, the pinned copy); no fault switch, no network, no game, no model. For a no_injection, weights or
controller decision it equals the logged request byte for byte; at a harness-fault decision it equals the logged
clean counterpart. `history` is not needed (the state carries the memory) and is ignored.

replay(run_dir): the strict harness replay of the whole attempt on its recorded observations (replay_check.py):
every live render equals the logged clean request, every snapshot re-renders it, every bridge window the harness asks
for equals the logged one. The HK stand-in for native replay: {"verified": bool, "status": ...}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# an own entry at the end, always (even when the directory is already on the path): the common loader removes the
# entry it inserted once this file is imported, and the re-render imports state_codec / core / harness lazily
sys.path.append(str(HERE))


class HKKeysAdapter:
    game_id = "hollow-knight-forestdango"

    def rerender(self, state: dict, history: list[dict]) -> dict:
        import state_codec

        from core.laya_multi import build_request

        name = state.get("harness") or "v9_keys"
        if not (HERE / "harness" / name / "agent.py").exists():
            raise FileNotFoundError(f"harness snapshot {name} is not in {HERE / 'harness'}")
        return state_codec.rerender(state, f"harness.{name}", build_request)

    def replay(self, run_dir: Path) -> dict:
        """Strict harness replay (always), plus the native replay on the deterministic build when its report exists
        (ROOT/native_replay/<run>.json, written by native_replay.py: the recorded native input sent again to a new
        game; every debug state per window, visible observation per window and state per frame identical). With a
        passing native replay the method is `native_replay`; without one it is the harness replay alone."""
        import replay_check

        run_dir = Path(run_dir)
        report = replay_check.check(run_dir, "harness." + self._harness(run_dir))
        if not report["ok"]:
            raise ValueError(f"harness replay differs: {report['mismatches'][:2]}")
        result = json.loads((run_dir / "result.json").read_text())
        harness = {
            "decisions": report["checked"],
            "bridge_windows": report["windows_replayed"],
            "prefix_steps": report.get("prefix_steps", 0),
            "live_render_exact": report["live_render_exact"],
            "snapshot_rerender_exact": report["snapshot_rerender_exact"],
        }
        native_file = run_dir.parent / "native_replay" / f"{run_dir.name}.json"
        if not native_file.exists():
            return {
                "verified": True,
                "status": result["status"],
                "method": "harness_replay_on_recorded_observations",
                **harness,
            }
        native = json.loads(native_file.read_text())
        if not native.get("ok") or native.get("run") != run_dir.name or native.get("drawn"):
            raise ValueError(f"native replay differs or is not this attempt's: {native_file.name}")
        compared = {r["what"]: {"identical": r["identical"], "rows": r["rows_attempt"]} for r in native["results"]}
        return {
            "verified": True,
            "status": result["status"],
            "method": "native_replay",
            "native": {
                "procedure": "the recorded native input (every bridge command from the boot on) sent again to a "
                "new game of the same deterministic build; debug state per window, visible "
                "observation per window and state per frame compared record by record",
                "commands": native["commands"],
                "compared": compared,
                "report": f"native_replay/{run_dir.name}.json",
            },
            "harness_replay": harness,
        }

    @staticmethod
    def _harness(run_dir: Path) -> str:
        manifest = json.loads((Path(run_dir) / "extra" / "manifest.json").read_text())
        return manifest["harness"]["name"]
