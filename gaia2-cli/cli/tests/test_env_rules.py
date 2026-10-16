# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Tests for the environment-side rules hook around the app CLIs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from gaia2_cli import env_rules
from gaia2_cli.apps import calendar as calendar_app
from gaia2_cli.shared import set_app

from conftest import read_events

ADD_EVENT = [
    "add-event",
    "--title",
    "Team Standup",
    "--start-datetime",
    "2024-01-15 10:00:00",
    "--end-datetime",
    "2024-01-15 11:00:00",
]
SEARCH = ["search-events", "--query", "Standup"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    (state / "events.jsonl").touch()
    home = tmp_path / "rules"
    home.mkdir()
    monkeypatch.setenv("GAIA2_STATE_DIR", str(state))
    monkeypatch.setenv("GAIA2_ENV_RULES_HOME", str(home))
    monkeypatch.delenv("GAIA2_EVENT_ID", raising=False)
    set_app("Calendar")
    return state, home


def _run(monkeypatch, capsys, argv: list[str]) -> tuple[int, str, str]:
    monkeypatch.setattr(sys, "argv", ["calendar", *argv])
    with pytest.raises(SystemExit) as exc:
        env_rules.run_app("calendar", calendar_app.cli)
    out, err = capsys.readouterr()
    code = exc.value.code
    return (0 if code is None else code), out, err


def _rules(home: Path, body: str) -> None:
    (home / "rules.py").write_text(body)


def _applied(home: Path) -> list[dict]:
    path = home / "applied.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_no_rules_is_passthrough(env, monkeypatch, capsys):
    state, home = env
    code, out, _ = _run(monkeypatch, capsys, ADD_EVENT)
    assert code == 0
    assert json.loads(out)["status"] == "success"
    assert len(read_events(state)) == 1
    assert not (home / "state.json").exists()


def test_event_id_bypasses_rules(env, monkeypatch, capsys):
    state, home = env
    _rules(home, "class _Rules(Rules):\n    def filter_action(self, action, env_state):\n        return Blocked(reason='no')\n")
    monkeypatch.setenv("GAIA2_EVENT_ID", "Event-ENV-1")
    code, out, _ = _run(monkeypatch, capsys, ADD_EVENT)
    assert code == 0 and json.loads(out)["status"] == "success"
    assert _applied(home) == []


def test_blocked_call_never_executes(env, monkeypatch, capsys):
    state, home = env
    _rules(
        home,
        "class _Rules(Rules):\n"
        "    def filter_action(self, action, env_state):\n"
        "        if env_state['call_index_app'] == 1:\n"
        "            return Blocked(reason='Error: calendar service temporarily unavailable')\n"
        "        return action\n",
    )
    code, out, err = _run(monkeypatch, capsys, ADD_EVENT)
    assert code == 1
    assert out == ""
    assert "temporarily unavailable" in err
    assert read_events(state) == []
    [entry] = _applied(home)
    assert entry["kind"] == "blocked" and entry["app"] == "calendar" and entry["subcommand"] == "add-event"

    code, out, _ = _run(monkeypatch, capsys, ADD_EVENT)  # second call goes through
    assert code == 0 and len(read_events(state)) == 1


def test_read_output_can_be_filtered(env, monkeypatch, capsys):
    state, home = env
    _run(monkeypatch, capsys, ADD_EVENT)
    _rules(
        home,
        "class _Rules(Rules):\n"
        "    def filter_observation(self, obs, env_state):\n"
        "        if env_state['subcommand'] == 'search-events':\n"
        "            return Observation(text='[]\\n')\n"
        "        return obs\n",
    )
    code, out, _ = _run(monkeypatch, capsys, SEARCH)
    assert code == 0 and out == "[]\n"
    assert [entry["kind"] for entry in _applied(home)] == ["modified"]


def test_write_output_rewrite_is_ignored(env, monkeypatch, capsys):
    state, home = env
    _rules(
        home,
        "class _Rules(Rules):\n"
        "    def filter_observation(self, obs, env_state):\n"
        "        return Observation(text='Error: failed\\n')\n",
    )
    code, out, _ = _run(monkeypatch, capsys, ADD_EVENT)
    assert code == 0
    assert json.loads(out)["status"] == "success"
    assert len(read_events(state)) == 1
    assert [entry["kind"] for entry in _applied(home)] == ["write_observation_ignored"]


def test_rules_errors_fail_open(env, monkeypatch, capsys):
    state, home = env
    _rules(home, "class _Rules(Rules):\n    def filter_action(self, action, env_state):\n        raise RuntimeError('boom')\n")
    code, out, _ = _run(monkeypatch, capsys, ADD_EVENT)
    assert code == 0 and json.loads(out)["status"] == "success"
    assert "boom" in (home / "errors.log").read_text()

    _rules(home, "this is not python(")
    code, out, _ = _run(monkeypatch, capsys, ADD_EVENT)
    assert code == 0 and len(read_events(state)) == 2


def test_app_errors_keep_exit_codes(env, monkeypatch, capsys):
    _, home = env
    _rules(home, "class _Rules(Rules):\n    pass\n")
    code, _, err = _run(monkeypatch, capsys, ["get-event", "--event-id", "missing"])
    assert code != 0
    code, _, err = _run(monkeypatch, capsys, ["add-event", "--bogus"])
    assert code == 2 and "No such option" in err


def test_loader_contract():
    with pytest.raises(env_rules.RulesCodeError):
        env_rules.load_rules_class("x = 1\n")
    with pytest.raises(env_rules.RulesCodeError):
        env_rules.load_rules_class(
            "class _Rules(Rules):\n    def modify_transition(self, a, r, s):\n        return r\n"
        )
    assert issubclass(env_rules.load_rules_class("class _Rules(Rules):\n    pass\n"), env_rules.Rules)


def test_entry_points_cover_all_apps():
    import tomllib

    scripts = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )["project"]["scripts"]
    apps = {k: v for k, v in scripts.items() if not k.startswith("gaia2-")}
    assert len(apps) == 10
    for name, target in apps.items():
        module, _, attr = target.partition(":")
        assert module == "gaia2_cli.env_rules"
        assert callable(getattr(env_rules, attr)), name


def test_subcommand_counter_ignores_help_probes(env, monkeypatch, capsys):
    state, home = env
    _rules(
        home,
        "class _Rules(Rules):\n"
        "    def filter_action(self, action, env_state):\n"
        "        if env_state['subcommand'] == 'add-event' and env_state['call_index_subcommand'] == 1:\n"
        "            return Blocked(reason='Error: calendar service temporarily unavailable')\n"
        "        return action\n",
    )
    _run(monkeypatch, capsys, ["--help"])
    _run(monkeypatch, capsys, ["add-event", "--help"])
    _run(monkeypatch, capsys, SEARCH)
    code, _, err = _run(monkeypatch, capsys, ADD_EVENT)  # first real add-event: blocked
    assert code == 1 and "temporarily unavailable" in err
    code, out, _ = _run(monkeypatch, capsys, ADD_EVENT)  # retry goes through
    assert code == 0 and sum(e["w"] for e in read_events(state)) == 1
    saved = json.loads((home / "state.json").read_text())
    assert saved["per_subcommand"] == {"calendar.search-events": 1, "calendar.add-event": 2}
    assert saved["per_app"]["calendar"] == 5
