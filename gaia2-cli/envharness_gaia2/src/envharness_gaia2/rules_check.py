"""Security checks for designer-proposed components (``rules_code``, ``in_env_actions``).

Rules run inside the gaia2-cli container via ``gaia2_cli.env_rules`` as the
gaia2 user, next to the scenario file (which holds the oracle). As in the
original benchmarks, *which* A/T/O interventions are acceptable is a
prompt-level concern; this gate only enforces what the prompt cannot: no file,
OS or network access, no scenario identifiers, the ``_Rules(Rules)`` contract,
and Setup actions that are single app CLI calls. A violation is handled like
envharness's ``RulesCodeError``: the K rollouts become error traces.
"""

from __future__ import annotations

import ast
import builtins
import re
import shlex
from collections.abc import Iterable, Sequence

from envharness.core.types import Action

MAX_BYTES = 8 * 1024
MAX_SETUP_ACTIONS = 20
ALLOWED_IMPORTS = frozenset({"re", "json", "random", "hashlib", "itertools", "textwrap"})
HOOKS = frozenset({"filter_action", "modify_transition", "filter_observation"})
FORBIDDEN_METHODS = frozenset({"reset", "step", "observe", "evaluate", "save_state", "from_state", "step_reward"})
FORBIDDEN_NAMES = frozenset(
    {
        "open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars",
        "setattr", "delattr", "input", "breakpoint", "exit", "quit",
    }
)
FORBIDDEN_MODULE_ROOTS = frozenset({"os", "sys", "subprocess", "socket", "shutil", "pathlib", "io", "builtins"})
APPS = ("calendar", "contacts", "emails", "messages", "chats", "rent-a-flat", "city", "cabs", "shopping", "cloud-drive")
_SHELL_CHARS = frozenset(";&|<>()")


def split_command(command: str) -> list[str]:
    """shlex split that also separates shell punctuation (``a;b`` -> ``a ; b``)."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
    lexer.whitespace_split = True
    return list(lexer)
PRE_BOUND = frozenset({"Rules", "Action", "Blocked", "Observation", "EnvResponse", "getattr", "hasattr"})
_SCENARIO_ID = re.compile(r"scenario_universe_\d+_[0-9a-z]+")


# getattr/hasattr are allowed: the container binds versions that refuse "_"-prefixed names.


class RulesCheckError(ValueError):
    """A component violates the Gaia2 env-rules security contract."""


def check_rules_code(code: str, *, forbidden_ids: Iterable[str] = ()) -> None:
    """Raise RulesCheckError with an actionable message on the first violation."""
    if not code.strip():
        return  # pass-through
    if len(code.encode("utf-8")) > MAX_BYTES:
        raise RulesCheckError(f"rules_code is larger than {MAX_BYTES} bytes")
    _no_identifiers(code, forbidden_ids, "rules_code")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise RulesCheckError(f"rules_code does not parse: {exc}") from exc

    rules_class = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_Rules"), None
    )
    if rules_class is None:
        raise RulesCheckError("rules_code must define `class _Rules(Rules):`")
    if not any(isinstance(base, ast.Name) and base.id == "Rules" for base in rules_class.bases):
        raise RulesCheckError("_Rules must subclass Rules")
    methods = {item.name for item in rules_class.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
    bad = sorted(methods & FORBIDDEN_METHODS)
    if bad:
        raise RulesCheckError(f"_Rules may override only {sorted(HOOKS)}; remove {bad}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                if name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise RulesCheckError(f"rules_code may import only {sorted(ALLOWED_IMPORTS)}; got `{name}`")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise RulesCheckError(f"rules_code may not use `{node.id}`")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            raise RulesCheckError(f"rules_code may not use dunder name `{node.id}`")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            raise RulesCheckError(f"rules_code may not access dunder attribute `{node.attr}`")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_MODULE_ROOTS:
            raise RulesCheckError(f"rules_code may not reference `{node.id}`")

    undefined = _undefined_names(tree)
    if undefined:
        name = undefined[0]
        hint = f"add `import {name}` at the top of rules_code" if name in ALLOWED_IMPORTS else "define it first"
        raise RulesCheckError(
            f"rules_code uses `{name}` but never defines or imports it (NameError at run time; only "
            f"{sorted(PRE_BOUND - {'getattr', 'hasattr'})} are pre-bound): {hint}"
        )


def _undefined_names(tree: ast.AST) -> list[str]:
    """Names read anywhere but bound nowhere in the module (a NameError when the line runs).

    Module-wide and scope-blind on purpose: it only flags names that no statement
    binds, e.g. ``json.loads`` without ``import json``, which a hook's own
    ``try/except`` would otherwise turn into a silent no-op.
    """
    bound = set(PRE_BOUND) | set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
    return sorted({node.id for node in ast.walk(tree)
                   if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in bound})


def setup_commands(actions: Sequence[Action], *, forbidden_ids: Iterable[str] = ()) -> list[str]:
    """Validate ``in_env_actions`` (``exec`` app CLI calls) and return their command strings."""
    if len(actions) > MAX_SETUP_ACTIONS:
        raise RulesCheckError(f"at most {MAX_SETUP_ACTIONS} in_env_actions")
    commands = []
    for action in actions:
        if action.name != "exec":
            raise RulesCheckError(f'in_env_actions items use the exec tool: name "exec", got {action.name!r}')
        command = str((action.kwargs or {}).get("command") or "").strip()
        try:
            argv = split_command(command)
        except ValueError as exc:
            raise RulesCheckError(f"in_env_actions command does not parse: {command!r} ({exc})") from exc
        if not argv or argv[0] not in APPS:
            raise RulesCheckError(f"each in_env_actions command must be one app CLI call ({', '.join(APPS)}): {command!r}")
        if any(token and set(token) <= _SHELL_CHARS for token in argv):
            raise RulesCheckError(f"in_env_actions commands are single app calls, not shell pipelines: {command!r}")
        _no_identifiers(command, forbidden_ids, "in_env_actions")
        commands.append(command)
    return commands


def _no_identifiers(text: str, forbidden_ids: Iterable[str], what: str) -> None:
    if _SCENARIO_ID.search(text) or any(sid and sid in text for sid in forbidden_ids):
        raise RulesCheckError(f"{what} must not reference GAIA2 scenario identifiers")
