"""Static checks for designer-proposed environment rules (``rules_code``).

Rules run inside the container via ``gaia2_cli.env_rules`` (filter_action and
filter_observation only). This gate rejects anything outside that contract
before any rollout is spent; a rejection becomes REFINE feedback.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable

MAX_BYTES = 8 * 1024
ALLOWED_IMPORTS = frozenset({"re", "json", "random", "hashlib", "itertools", "textwrap"})
ALLOWED_HOOKS = frozenset({"filter_action", "filter_observation"})
FORBIDDEN_METHODS = frozenset({"modify_transition", "reset", "step", "observe", "save_state", "from_state"})
FORBIDDEN_NAMES = frozenset(
    {
        "open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars",
        "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit",
    }
)
FORBIDDEN_MODULE_ROOTS = frozenset({"os", "sys", "subprocess", "socket", "shutil", "pathlib", "io", "builtins"})
_SCENARIO_ID = re.compile(r"scenario_universe_\d+_[0-9a-z]+")


class RulesCheckError(ValueError):
    """Rules code violates the Gaia2 env-rules contract."""


def check_rules_code(code: str, *, forbidden_ids: Iterable[str] = ()) -> None:
    """Raise RulesCheckError with an actionable message on the first violation."""
    if not code.strip():
        return  # pass-through candidate
    if len(code.encode("utf-8")) > MAX_BYTES:
        raise RulesCheckError(f"rules_code is larger than {MAX_BYTES} bytes")
    if _SCENARIO_ID.search(code) or any(sid and sid in code for sid in forbidden_ids):
        raise RulesCheckError("rules_code must not reference GAIA2 scenario identifiers")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise RulesCheckError(f"rules_code does not parse: {exc}") from exc

    rules_class = None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _check_import(node)
        elif isinstance(node, ast.ClassDef) and node.name == "_Rules":
            rules_class = node
    if rules_class is None:
        raise RulesCheckError("rules_code must define `class _Rules(Rules):`")
    if not any(isinstance(base, ast.Name) and base.id == "Rules" for base in rules_class.bases):
        raise RulesCheckError("_Rules must subclass Rules")
    hooks = {
        item.name
        for item in rules_class.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    bad = sorted(hooks & FORBIDDEN_METHODS)
    if bad:
        raise RulesCheckError(
            f"_Rules may only override filter_action/filter_observation; remove {bad} "
            "(modify_transition is unsupported: never alter executed writes)"
        )
    if not hooks & ALLOWED_HOOKS:
        raise RulesCheckError("_Rules must override filter_action and/or filter_observation")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _check_import(node)
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise RulesCheckError(f"rules_code may not use `{node.id}`")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            raise RulesCheckError(f"rules_code may not access dunder attribute `{node.attr}`")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_MODULE_ROOTS:
            raise RulesCheckError(f"rules_code may not reference `{node.id}`")


def _check_import(node: ast.Import | ast.ImportFrom) -> None:
    names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
    for name in names:
        root = name.split(".")[0]
        if root not in ALLOWED_IMPORTS:
            raise RulesCheckError(
                f"rules_code may import only {sorted(ALLOWED_IMPORTS)}; got `{name}`"
            )
