"""Regenerate prompts/app_cli_summary.md from the app CLIs' click definitions.

The designer's counterpart of the original bridges' full tool schemas: every
subcommand with its exact options and arguments (what `<app> schema` and
`<app> <subcommand> --help` show the student). Run from gaia2-cli/:

    PYTHONPATH=cli:core envharness_gaia2/.venv/bin/python -m envharness_gaia2.app_cli_summary
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path

import click

APPS = [
    ("calendar", "gaia2_cli.apps.calendar"),
    ("contacts", "gaia2_cli.apps.contacts"),
    ("emails", "gaia2_cli.apps.email"),
    ("messages", "gaia2_cli.apps.messages"),
    ("chats", "gaia2_cli.apps.chats"),
    ("rent-a-flat", "gaia2_cli.apps.apartment"),
    ("city", "gaia2_cli.apps.city"),
    ("cabs", "gaia2_cli.apps.cab"),
    ("shopping", "gaia2_cli.apps.shopping"),
    ("cloud-drive", "gaia2_cli.apps.files"),
]
OUT = Path(__file__).resolve().parents[2] / "prompts" / "app_cli_summary.md"


def _short(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _param(param: click.Parameter) -> str:
    if isinstance(param, click.Argument):
        return f"<{param.name.upper()}>" if param.required else f"[{param.name.upper()}]"
    flag = max(param.opts, key=len)
    if param.is_flag:
        text = flag
    else:
        metavar = param.type.name.upper() if hasattr(param.type, "name") else "VALUE"
        if isinstance(param.type, click.Choice):
            metavar = "|".join(map(str, param.type.choices))
        text = f"{flag} {metavar}"
    if param.multiple:
        text += " (repeatable)"
    if param.required:
        text += " (required)"
    elif param.default not in (None, False, ()) and not callable(param.default):
        text += f" (default {json.dumps(param.default)})"
    return text


def _writes(cmd: click.Command) -> bool:
    try:
        return "write=True" in inspect.getsource(cmd.callback)
    except (OSError, TypeError):
        return False


def render() -> str:
    lines = [
        "# Gaia2 app CLIs available to the agent (via OpenClaw `exec`)",
        "",
        "Usage: `<app> <subcommand> [ARGS] [--option VALUE ...]`. Every subcommand with its exact",
        "arguments and options (the same information `<app> schema` and `<app> <subcommand> --help`",
        "give the student). An option that is not listed is rejected by the CLI (exit 2,",
        "\"No such option\"). [write] = the subcommand changes app state.",
    ]
    for app, module in APPS:
        group = importlib.import_module(module).cli
        lines += ["", f"## `{app}`"]
        for name, cmd in group.commands.items():
            if name == "schema" or getattr(cmd, "hidden", False):
                continue
            mark = " [write]" if _writes(cmd) else ""
            lines.append(f"- `{app} {name}`{mark} — {_short(cmd.help, 110)}")
            for param in cmd.params:
                if getattr(param, "hidden", False) or param.name == "help":
                    continue
                help_text = _short(getattr(param, "help", "") or "", 90)
                lines.append(f"  - `{_param(param)}`" + (f" — {help_text}" if help_text else ""))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        sys.exit(0 if OUT.read_text(encoding="utf-8") == text else 1)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({len(text)} chars)")
