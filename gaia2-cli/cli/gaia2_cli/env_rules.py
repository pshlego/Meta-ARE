# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Environment-side rules for the Gaia2 app CLIs (EnvHarness Contract: A/T/O).

Every app CLI entry point goes through :func:`run_app`. Without a rules file
(or for ENV actions, which carry ``GAIA2_EVENT_ID``) it calls the original
click group unchanged, so seed behaviour is byte-identical.

With ``<rules home>/rules.py`` present, a ``_Rules(Rules)`` class defined in
it intervenes on agent calls exactly like envharness ``harnesses/rules.py``:

* ``filter_action(action, env_state) -> Action | Blocked`` runs before the
  command (A). ``Blocked`` stops it: the app is never executed (nothing
  reaches ``events.jsonl``), the reason is printed to stderr like a normal
  app error and the process exits 1. A returned ``Action`` whose
  ``kwargs["argv"]`` differs is what actually executes.
* ``modify_transition(action, response, env_state) -> EnvResponse`` (T)
  transforms the executed command's result: ``response.observation.text`` is
  its stdout, ``response.info`` holds ``exit_code``, ``stderr`` and ``wrote``.
* ``filter_observation(obs, env_state) -> Observation`` (O) transforms the
  stdout the agent sees.

Which interventions are acceptable is the designer's (prompt-level) concern,
as in the original benchmarks; every intervention is recorded in
``applied.jsonl`` (with ``wrote`` for state-changing calls) so judge integrity
can be audited. ``env_state["extras"]`` is a JSON scratch dict persisted
across calls. The rules file lives under ``/var/gaia2`` (gaia2-only, 0700), so
the agent can neither read nor change it; errors in rules code fail open and
go to ``errors.log``. This module is stdlib-only.
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


@dataclass
class EnvResponse:
    """Mirrors ``envharness.core.types.EnvResponse`` for one app call.

    ``observation.text`` is the command's stdout; ``info`` carries
    ``exit_code``, ``stderr`` and ``wrote``.
    """

    observation: Observation = field(default_factory=Observation)
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class EnvState(dict):
    """``env_state`` readable as ``env_state.app`` or ``env_state["app"]``.

    envharness bridges hand hooks a typed state object (attribute access,
    which ``_DEFAULT_SYSTEM`` teaches); this keeps that and the dict form.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class Extras(dict):
    """``env_state["extras"]``: always truthy, so ``env_state.get("extras") or {}``
    still returns (and persists) this dict when it is empty."""

    def __bool__(self) -> bool:
        return True


class Rules:
    """Base class for rules code; override any hook."""

    def filter_action(self, action: Action, env_state: dict[str, Any]) -> Action | Blocked:
        return action

    def modify_transition(self, action: Action, response: EnvResponse, env_state: dict[str, Any]) -> EnvResponse:
        return response

    def filter_observation(self, obs: Observation, env_state: dict[str, Any]) -> Observation:
        return obs


class RulesCodeError(Exception):
    """The rules source does not satisfy the ``_Rules(Rules)`` contract."""


def _public_getattr(obj: Any, name: str, *default: Any) -> Any:
    """``getattr`` for rules code: public attribute names only (no dunder escapes)."""
    if not isinstance(name, str) or name.startswith("_"):
        raise AttributeError(f"rules code may only read public attributes, not {name!r}")
    return getattr(obj, name, *default)


def _public_hasattr(obj: Any, name: str) -> bool:
    if not isinstance(name, str) or name.startswith("_"):
        return False
    return hasattr(obj, name)


def load_rules_class(source: str) -> type[Rules]:
    """Compile rules source in the same namespace contract as envharness ``load_rules_subclass``."""
    namespace: dict[str, Any] = {
        "Rules": Rules,
        "Action": Action,
        "Blocked": Blocked,
        "Observation": Observation,
        "EnvResponse": EnvResponse,
        "getattr": _public_getattr,
        "hasattr": _public_hasattr,
    }
    try:
        exec(compile(source, "<env_rules>", "exec"), namespace)  # noqa: S102 - trusted, verified host-side
    except Exception as exc:  # pragma: no cover - surfaced through errors.log
        raise RulesCodeError(f"rules code failed to compile/execute: {exc}") from exc
    cls = namespace.get("_Rules")
    if not isinstance(cls, type) or not issubclass(cls, Rules):
        raise RulesCodeError("rules code must define `class _Rules(Rules)`")
    return cls


def rules_home() -> Path:
    return Path(os.environ.get("GAIA2_ENV_RULES_HOME", DEFAULT_RULES_HOME))


def _append(path: Path, line: str) -> None:
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        pass


def _record(home: Path, app: str, subcommand: str, kind: str, argv: list[str], **extra: Any) -> None:
    entry = {"t": time.time(), "app": app, "subcommand": subcommand, "kind": kind, "argv": argv, **extra}
    if "reason" in entry:
        entry["reason"] = str(entry["reason"])[:_REASON_LOG_CHARS]
    _append(home / "applied.jsonl", json.dumps(entry) + "\n")


def _error(home: Path, app: str, context: str) -> None:
    _append(
        home / "errors.log",
        f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {app} {context}\n{traceback.format_exc()}\n",
    )


def _next_call_index(state: dict, app: str, subcommand: str, is_help: bool) -> tuple[int, int, int]:
    """Increment and return (per-app, total, per-subcommand) call counters (1-based).

    Help/schema probes count toward the app and total counters but not toward
    the per-subcommand counter (0 for them), so rules can target "the first
    real call of X" even when the agent explores with ``--help`` first.
    """
    per_app = state.setdefault("per_app", {})
    per_app[app] = int(per_app.get(app, 0)) + 1
    state["total"] = int(state.get("total", 0)) + 1
    per_sub = state.setdefault("per_subcommand", {})
    key = f"{app}.{subcommand}"
    if not is_help:
        per_sub[key] = int(per_sub.get(key, 0)) + 1
    return per_app[app], state["total"], (0 if is_help else per_sub[key])


@contextlib.contextmanager
def _hook_phase(home: Path, env_state: "EnvState"):
    """Run hooks with ``extras`` read, changed and stored under the state lock.

    The student often runs app calls in parallel; holding the lock from reading
    ``extras`` to storing it makes each hook phase see every earlier call's
    changes, as in the original's sequential episode (counters, one-shots).
    The app itself runs outside the lock.
    """
    with _locked_state(home) as state:
        stored = state.get("extras")
        env_state["extras"] = Extras(stored if isinstance(stored, dict) else {})
        yield state
        try:
            state["extras"] = json.loads(json.dumps(env_state.get("extras")))
        except (TypeError, ValueError):
            _append(home / "errors.log", "env_state['extras'] is not JSON-serializable; not saved\n")


@contextlib.contextmanager
def _locked_state(home: Path):
    state_path = home / "state.json"
    with open(home / "state.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = {}
            yield state
            state_path.write_text(json.dumps(state), encoding="utf-8")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _subcommand(argv: list[str]) -> str:
    return next((arg for arg in argv if not arg.startswith("-")), "")


def _is_help(argv: list[str], subcommand: str) -> bool:
    return subcommand in ("", "schema") or any(arg in ("--help", "-h") for arg in argv)


def _write_count() -> int:
    """Writes this process logged so far; concurrent calls' writes in events.jsonl do not count."""
    from gaia2_cli import shared

    return shared.writes_logged


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


def _argv_of(decision: Any, fallback: list[str]) -> list[str]:
    argv = getattr(decision, "kwargs", {}).get("argv") if isinstance(getattr(decision, "kwargs", None), dict) else None
    if isinstance(argv, (list, tuple)) and all(isinstance(arg, str) for arg in argv):
        return list(argv)
    return fallback


def run_app(app: str, group: Any) -> None:
    """Entry point shared by every app CLI."""
    home = rules_home()
    rules_file = home / "rules.py"
    if os.environ.get("GAIA2_EVENT_ID") or not rules_file.is_file():
        group()  # unchanged seed behaviour
        return

    argv = sys.argv[1:]
    subcommand = _subcommand(argv)
    is_help = _is_help(argv, subcommand)
    try:
        rules = load_rules_class(rules_file.read_text(encoding="utf-8"))()
    except Exception:
        _error(home, app, "rules load failed; running unchanged")
        group()
        return

    env_state = EnvState({"app": app, "subcommand": subcommand, "argv": list(argv), "is_help": is_help})
    action = Action(
        name=f"{app}.{subcommand}" if subcommand else app,
        kwargs={"argv": list(argv), "subcommand": subcommand, "text": shlex.join([app, *argv])},
    )

    # A: transform or reject before execution.
    with _hook_phase(home, env_state) as state:
        app_index, total_index, subcommand_index = _next_call_index(state, app, subcommand, is_help)
        env_state.update(call_index_app=app_index, call_index_total=total_index,
                         call_index_subcommand=subcommand_index)
        try:
            decision = rules.filter_action(action, env_state)
        except Exception:
            _error(home, app, "filter_action raised; running unchanged")
            decision = action
    if isinstance(decision, Blocked) or getattr(decision, "kind", None) == "blocked":
        reason = str(getattr(decision, "reason", "") or "Error: request failed")
        _record(home, app, subcommand, "blocked", argv, reason=reason)
        print(reason, file=sys.stderr)
        sys.exit(1)
    exec_argv = _argv_of(decision, argv)
    if exec_argv != argv:
        _record(home, app, subcommand, "action_rewritten", argv, new_argv=exec_argv)
    executed = decision if isinstance(decision, Action) else action
    env_state["executed_argv"] = exec_argv

    writes_before = _write_count()
    out_buffer, err_buffer = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out_buffer), contextlib.redirect_stderr(err_buffer):
        code = _invoke(group, app, exec_argv)
    stdout, stderr = out_buffer.getvalue(), err_buffer.getvalue()
    wrote = _write_count() > writes_before

    # T: transform the executed command's result.
    response = EnvResponse(
        observation=Observation(text=stdout, data={"exit_code": code, "wrote": wrote}),
        info={"exit_code": code, "stderr": stderr, "wrote": wrote},
    )
    with _hook_phase(home, env_state):
        stdout, code, stderr = _transition_and_observation(
            home, rules, app, subcommand, argv, executed, response, env_state, wrote)

    if stderr and not stderr.endswith("\n"):
        stderr += "\n"  # a CLI's messages end with a newline; keep stdout on its own line
    sys.stderr.write(stderr)
    sys.stderr.flush()
    sys.stdout.write(stdout)
    sys.stdout.flush()
    sys.exit(code)


def _transition_and_observation(home: Path, rules: Rules, app: str, subcommand: str, argv: list[str],
                                executed: Action, response: EnvResponse, env_state: "EnvState",
                                wrote: bool) -> tuple[str, int, str]:
    """T then O on an executed command; returns the (stdout, exit code, stderr) the agent gets."""
    stdout, code, stderr = response.observation.text, response.info["exit_code"], response.info["stderr"]
    try:
        transformed = rules.modify_transition(executed, response, env_state)
        info = getattr(transformed, "info", None) or {}
        new_stdout = getattr(getattr(transformed, "observation", None), "text", stdout)
        new_code = info.get("exit_code", code)
        new_stderr = info.get("stderr", stderr)
        if not (isinstance(new_stdout, str) and isinstance(new_stderr, str) and isinstance(new_code, int)):
            raise TypeError("modify_transition must keep text str, info.exit_code int and info.stderr str")
        if (new_stdout, new_code, new_stderr) != (stdout, code, stderr):
            _record(home, app, subcommand, "transition_modified", argv, wrote=wrote,
                    exit_code=[code, new_code], stdout_changed=new_stdout != stdout,
                    stderr_changed=new_stderr != stderr)
            stdout, code, stderr = new_stdout, new_code, new_stderr
    except Exception:
        _error(home, app, "modify_transition raised; keeping the executed result")

    # O: transform what the agent sees.
    try:
        filtered = rules.filter_observation(
            Observation(text=stdout, data={"exit_code": code, "wrote": wrote}), env_state
        )
        text = getattr(filtered, "text", stdout)
        if not isinstance(text, str):
            raise TypeError("filter_observation must return an Observation with str text")
        if text != stdout:
            _record(home, app, subcommand, "observation_modified", argv, wrote=wrote)
            stdout = text
    except Exception:
        _error(home, app, "filter_observation raised; showing the unfiltered output")
    return stdout, code, stderr


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
