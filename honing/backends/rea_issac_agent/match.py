"""Agent-only decision proxy around the frozen REA-Issac match recorder."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from honing.backends.rea_issac.catalog import DIRECTIONS, OPTIONS
from honing.backends.rea_issac.match import run_match
from honing.backends.rea_issac.model import DecisionClient, validate_reply
from honing.backends.rea_issac.replay import audit_run
from honing.backends.rea_issac.scenarios import ReaIssacScenario
from honing.backends.rea_issac_agent.controller import AgentController
from honing.backends.rea_issac_agent.faults import CONTROLLER_SWAP_MOVE_AXES, swap_move_axes, verify_clean_baseline
from honing.core import HarnessSpec

FORMAT = "rea-issac-agent-record-v1"
REVERSE_DIRECTIONS = {vector: name for name, vector in DIRECTIONS.items()}


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _observation(request: dict[str, Any]) -> dict[str, Any]:
    state = request["parts"]["essential"][0]
    marker = "Public observation: "
    line = state.split(marker, 1)[1].split("\n", 1)[0]
    observation = json.loads(line)
    if not isinstance(observation, dict) or request["id"].split("-t")[-1] != str(observation["tick"]):
        raise ValueError("agent request lacks the expected public observation")
    return observation


def _proxy_reply(original: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    choices = {
        "move": REVERSE_DIRECTIONS[action["move_x"], action["move_y"]],
        "shoot": REVERSE_DIRECTIONS[action["shoot_x"], action["shoot_y"]],
        "bomb": "YES" if action.get("bomb", False) else "NO",
    }
    response = json.loads(json.dumps(original))
    for name, options in OPTIONS.items():
        choice = choices[name]
        response["answers"][name] = {
            "choice": choice,
            "probabilities": {option: float(option == choice) for option in options},
        }
    return response


@contextmanager
def _agent_server(upstream: DecisionClient, mode: str, records: Path, controller_fault: str | None) -> Iterator[str]:
    if controller_fault not in (None, CONTROLLER_SWAP_MOVE_AXES):
        raise ValueError(f"unsupported controller fault: {controller_fault}")
    controller = AgentController(mode)
    health = upstream.health()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/v1/health":
                self.send_error(404)
                return
            self._send(health)

        def do_POST(self) -> None:
            if self.path != "/v1/decide":
                self.send_error(404)
                return
            try:
                length = int(self.headers["Content-Length"])
                if not 0 < length <= 4 * 1024 * 1024:
                    raise ValueError("agent request size is invalid")
                request = json.loads(self.rfile.read(length))
                selected, original = upstream.decide(request)
                before_fault, reason = controller.apply(_observation(request), selected)
                before_fault = {**before_fault, "bomb": before_fault.get("bomb", False)}
                executed = swap_move_axes(before_fault) if controller_fault else before_fault
                response = _proxy_reply(original, executed)
                validate_reply(request, response)
                record = {
                    "request_id": request["id"],
                    "request_sha256": _digest(request),
                    "model_response": original,
                    "selected_action": selected,
                    "executed_action": executed,
                    "controller_reason": reason,
                }
                if controller_fault:
                    record.update(
                        controller_fault=controller_fault,
                        controller_before_fault=before_fault,
                        fault_triggered=executed != before_fault,
                    )
                with records.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                self._send(response)
            except (KeyError, TypeError, ValueError, OSError) as error:
                self.send_error(502, str(error))

        def _send(self, value: dict[str, Any]) -> None:
            body = json.dumps(value, sort_keys=True, allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def run_agent_match(
    scenario: ReaIssacScenario,
    out: Path,
    spec: HarnessSpec,
    *,
    decision_server: str,
    policy_checksum: str,
    controller_mode: str,
    controller_fault: str | None = None,
) -> dict[str, Any]:
    """Keep v1 match/replay unchanged; save the original model replies beside its recording."""
    if controller_fault:
        verify_clean_baseline()
    upstream = DecisionClient(decision_server)
    with _agent_server(upstream, controller_mode, out / "agent_decisions.jsonl", controller_fault) as proxy:
        result = run_match(scenario, out, spec, policy="model", decision_server=proxy, policy_checksum=policy_checksum)
    metadata = {
        "format": FORMAT,
        "controller_mode": controller_mode,
        "controller_sha256": hashlib.sha256((Path(__file__).parent / "controller.py").read_bytes()).hexdigest(),
        "agent_decisions_sha256": hashlib.sha256((out / "agent_decisions.jsonl").read_bytes()).hexdigest(),
    }
    if controller_fault:
        metadata["controller_fault"] = controller_fault
    (out / "agent.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    audit_agent_run(out)
    return result


def audit_agent_run(out: Path) -> dict[str, Any]:
    """Audit the v1 game recording and its agent-only decision record."""
    audit_run(out)
    metadata = json.loads((out / "agent.json").read_text())
    records_path = out / "agent_decisions.jsonl"
    if (
        metadata["format"] != FORMAT
        or metadata["controller_sha256"]
        != hashlib.sha256((Path(__file__).parent / "controller.py").read_bytes()).hexdigest()
        or metadata["agent_decisions_sha256"] != hashlib.sha256(records_path.read_bytes()).hexdigest()
    ):
        raise ValueError("agent metadata changed")
    controller = AgentController(metadata["controller_mode"])
    controller_fault = metadata.get("controller_fault")
    if controller_fault not in (None, CONTROLLER_SWAP_MOVE_AXES):
        raise ValueError("unsupported recorded controller fault")
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    decisions = [json.loads(line) for line in (out / "decisions.jsonl").read_text().splitlines()]
    if len(records) != len(decisions):
        raise ValueError("agent decision count differs from match")
    for record, decision in zip(records, decisions, strict=True):
        request = decision["request"]
        selected = validate_reply(request, record["model_response"])
        before_fault, reason = controller.apply(_observation(request), selected)
        before_fault = {**before_fault, "bomb": before_fault.get("bomb", False)}
        executed = swap_move_axes(before_fault) if controller_fault else before_fault
        if controller_fault and (
            record.get("controller_fault") != controller_fault
            or record.get("controller_before_fault") != before_fault
            or record.get("fault_triggered") != (executed != before_fault)
        ):
            raise ValueError("controller fault record differs from clean action")
        if (
            record["request_id"] != request["id"]
            or record["request_sha256"] != _digest(request)
            or record["selected_action"] != selected
            or record["executed_action"] != executed
            or record["controller_reason"] != reason
            or decision["response"] != _proxy_reply(record["model_response"], executed)
            or decision["action"] != executed
        ):
            raise ValueError("agent selected/executed record differs from match")
    return {
        "verified": True,
        "decisions": len(records),
        "controller_mode": metadata["controller_mode"],
        "controller_fault": controller_fault,
        "fault_triggers": sum(record.get("fault_triggered", False) for record in records),
    }
