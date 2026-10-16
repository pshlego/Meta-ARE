# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Environment-side rules for the Gaia2 app CLIs (EnvHarness "Rules", A/O axes).

Every app CLI entry point goes through :func:`run_app`. Without a rules file
(or for ENV actions, which carry ``GAIA2_EVENT_ID``) it calls the original
click group unchanged, so seed behaviour is byte-identical.

With ``<rules home>/rules.py`` present, a ``_Rules(Rules)`` class defined in
it may intervene on agent calls:

* ``filter_action(action, env_state) -> Action | Blocked`` runs before the
  command. ``Blocked`` stops it: the app is never executed (nothing reaches
  ``events.jsonl``), the reason is printed to stderr like a normal app error
  and the process exits 1.
* ``filter_observation(obs, env_state) -> Observation`` rewrites the stdout
  the agent sees. A command that wrote app state always shows its original
  output; a rewrite there is ignored and recorded as a violation.

``modify_transition`` (changing an executed result) is deliberately not
supported: making a successful write look failed invites a retry that the
judge would count twice. Express "transient failures" as ``Blocked``.

The rules file lives under ``/var/gaia2`` (gaia2-only, 0700), so the agent
can neither read nor change it. Every intervention is appended to
``applied.jsonl``; errors in rules code fail open and go to ``errors.log``.
This module is stdlib-only; it must stay importable in the slim image.
"""

from __future__ import annotations

import contextlib
import fcntl
import importlib
import io
import json
import os
import shlex
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_RULES_HOME = "/var/gaia2/env_rules"
_REASON_LOG_CHARS = 200


@dataclass
class Action:
    """Mirrors ``envharness.core.types.Action``."""

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Blocked:
    """Mirrors ``envharness.core.types.Blocked``."""

    reason: str = ""
    kind: str = "blocked"


@dataclass
class Observation:
    """Mirrors ``envharness.core.types.Observation``."""

    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class Rules:
    """Base class for rules code; override either hook."""

    def filter_action(self, action: Action, env_state: dict[str, Any]) -> Action | Blocked:
        return action

    def filter_observation(self, obs: Observation, env_state: dict[str, Any]) -> Observation:
        return obs


class RulesCodeError(Exception):
    """The rules source does not satisfy the ``_Rules(Rules)`` contract."""


def load_rules_class(source: str) -> type[Rules]:
    """Compile rules source in the same namespace contract as envharness ``load_rules_subclass``."""
    namespace: dict[str, Any] = {
        "Rules": Rules,
        "Action": Action,
        "Blocked": Blocked,
        "Observation": Observation,
    }
    try:
        exec(compile(source, "<env_rules>", "exec"), namespace)  # noqa: S102 - trusted, verified host-side
    except Exception as exc:  # pragma: no cover - surfaced through errors.log
        raise RulesCodeError(f"rules code failed to compile/execute: {exc}") from exc
    cls = namespace.get("_Rules")
    if not isinstance(cls, type) or not issubclass(cls, Rules):
        raise RulesCodeError("rules code must define `class _Rules(Rules)`")
    if "modify_transition" in vars(cls):
        raise RulesCodeError("modify_transition is not supported by Gaia2 env rules")
    return cls


def rules_home() -> Path:
    return Path(os.environ.get("GAIA2_ENV_RULES_HOME", DEFAULT_RULES_HOME))


def _append(path: Path, line: str) -> None:
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        pass


def _record(home: Path, app: str, subcommand: str, kind: str, reason: str = "") -> None:
    entry = {
        "t": time.time(),
        "app": app,
        "subcommand": subcommand,
        "kind": kind,
        "reason": reason[:_REASON_LOG_CHARS],
    }
    _append(home / "applied.jsonl", json.dumps(entry) + "\n")


def _error(home: Path, app: str, context: str) -> None:
    _append(
        home / "errors.log",
        f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {app} {context}\n{traceback.format_exc()}\n",
    )


def _next_call_index(home: Path, app: str) -> tuple[int, int]:
    """Increment and return (per-app, total) call counters, 1-based."""
    state_path = home / "state.json"
    with open(home / "state.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = {}
            per_app = state.setdefault("per_app", {})
            per_app[app] = int(per_app.get(app, 0)) + 1
            state["total"] = int(state.get("total", 0)) + 1
            state_path.write_text(json.dumps(state), encoding="utf-8")
            return per_app[app], state["total"]
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _subcommand(argv: list[str]) -> str:
    return next((arg for arg in argv if not arg.startswith("-")), "")


def _write_count() -> int:
    from gaia2_cli.shared import events_file

    try:
        with open(events_file(), encoding="utf-8") as handle:
            return sum(1 for line in handle if '"w": true' in line)
    except (OSError, RuntimeError):
        return 0


def _invoke(group: Any, app: str, argv: list[str]) -> int:
    """Run a click group like standalone mode does, returning the exit code."""
    import click

    try:
        result = group.main(args=argv, prog_name=app, standalone_mode=False)
        return result if isinstance(result, int) else 0
    except click.exceptions.Exit as exc:
        return exc.exit_code
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.Abort:
        print("Aborted!", file=sys.stderr)
        return 1
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1


def run_app(app: str, group: Any) -> None:
    """Entry point shared by every app CLI."""
    home = rules_home()
    rules_file = home / "rules.py"
    if os.environ.get("GAIA2_EVENT_ID") or not rules_file.is_file():
        group()  # unchanged seed behaviour
        return

    argv = sys.argv[1:]
    subcommand = _subcommand(argv)
    try:
        rules = load_rules_class(rules_file.read_text(encoding="utf-8"))()
        app_index, total_index = _next_call_index(home, app)
    except Exception:
        _error(home, app, "rules load failed; running unchanged")
        group()
        return

    env_state = {
        "app": app,
        "subcommand": subcommand,
        "call_index_app": app_index,
        "call_index_total": total_index,
    }
    action = Action(
        name=f"{app}.{subcommand}" if subcommand else app,
        kwargs={"argv": list(argv), "subcommand": subcommand, "text": shlex.join([app, *argv])},
    )
    try:
        decision = rules.filter_action(action, env_state)
    except Exception:
        _error(home, app, "filter_action raised; running unchanged")
        decision = action
    if isinstance(decision, Blocked) or getattr(decision, "kind", None) == "blocked":
        reason = str(getattr(decision, "reason", "") or "Error: request failed")
        _record(home, app, subcommand, "blocked", reason)
        print(reason, file=sys.stderr)
        sys.exit(1)
    if isinstance(decision, Action) and decision.kwargs.get("argv") != argv:
        # Rewriting what actually executes could corrupt judged writes.
        _record(home, app, subcommand, "action_rewrite_ignored")

    writes_before = _write_count()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = _invoke(group, app, argv)
    original = buffer.getvalue()
    wrote = _write_count() > writes_before

    shown = original
    try:
        filtered = rules.filter_observation(
            Observation(text=original, data={"exit_code": code, "wrote": wrote}), env_state
        )
        text = getattr(filtered, "text", original)
        if isinstance(text, str) and text != original:
            if wrote:
                _record(home, app, subcommand, "write_observation_ignored")
            else:
                shown = text
                _record(home, app, subcommand, "modified")
    except Exception:
        _error(home, app, "filter_observation raised; showing original output")

    sys.stdout.write(shown)
    sys.stdout.flush()
    sys.exit(code)


def _entry(app: str, module: str):
    def main() -> None:
        run_app(app, importlib.import_module(module).cli)

    main.__name__ = f"{app.replace('-', '_')}_main"
    return main


calendar_main = _entry("calendar", "gaia2_cli.apps.calendar")
contacts_main = _entry("contacts", "gaia2_cli.apps.contacts")
emails_main = _entry("emails", "gaia2_cli.apps.email")
messages_main = _entry("messages", "gaia2_cli.apps.messages")
chats_main = _entry("chats", "gaia2_cli.apps.chats")
rent_a_flat_main = _entry("rent-a-flat", "gaia2_cli.apps.apartment")
city_main = _entry("city", "gaia2_cli.apps.city")
cabs_main = _entry("cabs", "gaia2_cli.apps.cab")
shopping_main = _entry("shopping", "gaia2_cli.apps.shopping")
cloud_drive_main = _entry("cloud-drive", "gaia2_cli.apps.files")
