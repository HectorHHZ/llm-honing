"""The per-game adapter interface the common Exp 0 v0 tools call, and its loader.

A game supplies one small adapter object. The common tools never import game code otherwise.

Required:

``game_id: str``
    The dataset game id (same as ``collection.json`` ``game_id``).

``rerender(state: dict, history: list[dict]) -> dict``
    Re-render one decision's ``/v1/decide`` request with the **clean** harness from the saved public
    state of that decision (the ``state`` of its line in ``public_states.jsonl``) and the public history
    of the earlier rows (``contract.public_history_entry`` of every earlier row, in order). It must be
    deterministic and use no fault switch, no network and no game process. For every decision of a
    ``no_injection`` episode it must return a request equal to the logged one (``probes.py selfcheck``
    tests this); at a harness-fault trigger it returns the request the clean harness would have sent.
    The returned request includes the same ``id`` the collector sent.

Optional:

``replay(run_dir: pathlib.Path) -> dict``
    Native replay verification for games that can replay an episode deterministically (Griddly did).
    Return ``{"verified": bool, "status": <native status>, ...}`` or raise ``ValueError`` when the replay
    differs. Without it the audit records ``{"method": "contract_record"}``: the native outcome is taken
    from the recorded native events, checked for internal consistency, not replayed.

An adapter is named on the command line as ``path/to/file.py:attribute`` or ``package.module:attribute``.
The attribute may be an instance, a class (instantiated with no arguments) or a function returning one.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GameAdapter(Protocol):
    game_id: str

    def rerender(self, state: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]: ...


def load_adapter(spec: str) -> GameAdapter:
    """Load ``file.py:attr`` or ``module:attr`` and return the adapter instance."""
    if ":" not in spec:
        raise ValueError("an adapter is named as path/to/file.py:attribute or package.module:attribute")
    where, attribute = spec.rsplit(":", 1)
    if where.endswith(".py"):
        path = Path(where).resolve()
        name = f"_exp0_adapter_{abs(hash(str(path)))}"
        module_spec = importlib.util.spec_from_file_location(name, path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"cannot load adapter file {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        sys.path.insert(0, str(path.parent))
        try:
            module_spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(path.parent))
    else:
        module = importlib.import_module(where)
    target = getattr(module, attribute)
    adapter = target() if inspect.isclass(target) or (callable(target) and not hasattr(target, "rerender")) else target
    if not isinstance(adapter, GameAdapter):
        raise TypeError(f"{spec} does not provide game_id and rerender(state, history)")
    return adapter
