from __future__ import annotations

import json
from pathlib import Path

from envharness_gaia2.report import LEAK_TERMS, command_functions, leaked_blocked_writes

from helpers import write_scenario_output

APPS = Path(__file__).resolve().parents[2] / "cli" / "gaia2_cli" / "apps"
DELETE = "calendar delete-event --event-id {}"


def _rollout(tmp_path: Path, commands, n_blocked: int, written_ids, blocked_ids=None) -> Path:
    path = write_scenario_output(tmp_path / "scenario_universe_29_x", success=False, commands=commands)
    argvs = [None] * n_blocked if blocked_ids is None else [["delete-event", "--event-id", i] for i in blocked_ids]
    (path / "env_rules_applied.jsonl").write_text("".join(
        json.dumps({"t": 1.0, "app": "calendar", "subcommand": "delete-event", "kind": "blocked", "reason": "rate",
                    **({} if argv is None else {"argv": argv})}) + "\n"
        for argv in argvs))
    (path / "events.jsonl").write_text("".join(
        json.dumps({"t": 1.0, "app": "Calendar", "fn": "delete_calendar_event", "args": {"event_id": i}, "w": True}) + "\n"
        for i in written_ids))
    return path


def test_command_functions_maps_subcommands_to_logged_functions():
    functions = command_functions(APPS)
    assert functions[("calendar", "delete-event")] == "delete_calendar_event"
    assert functions[("cabs", "order-ride")] == "order_ride"
    assert functions[("calendar", "delete-calendar-event-by-attendee")] == "delete_calendar_event_by_attendee"


def test_concurrent_allowed_call_is_not_a_leak(tmp_path):
    # Three parallel deletes: the first runs (one write), the other two are blocked.
    path = _rollout(tmp_path, [DELETE.format(i) for i in "abc"], n_blocked=2, written_ids=["a"])
    assert leaked_blocked_writes(path, command_functions(APPS)) == 0


def test_write_from_a_blocked_call_is_a_leak(tmp_path):
    path = _rollout(tmp_path, [DELETE.format("a"), "calendar delete-event --help"], n_blocked=1, written_ids=["a"])
    assert leaked_blocked_writes(path, command_functions(APPS)) == 1


def test_allowed_write_in_a_loop_with_blocked_calls_is_not_a_leak(tmp_path):
    # One loop text runs three deletes: a runs, b and c are blocked.
    loop = "for i in a b c; do calendar delete-event --event-id $i; done"
    path = _rollout(tmp_path, [loop], n_blocked=2, written_ids=["a"], blocked_ids=["b", "c"])
    assert leaked_blocked_writes(path, command_functions(APPS)) == 0


def test_write_matching_a_blocked_call_is_a_leak(tmp_path):
    path = _rollout(tmp_path, [DELETE.format("a")], n_blocked=1, written_ids=["a"], blocked_ids=["a"])
    assert leaked_blocked_writes(path, command_functions(APPS)) == 1


def test_oracle_function_schema_key_is_not_a_leak_term():
    text = '"oracle_function": "add_calendar_event", "oracle_f\\n[5] $ chats ... oracle={"X": 2} oracle_events oracle_fx'
    assert len(LEAK_TERMS["oracle"].findall(text)) == 3
