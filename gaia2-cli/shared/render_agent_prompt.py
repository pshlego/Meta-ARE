#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Render the agent system prompt and exec-approvals from a scenario.

Reads the AGENTS_TEMPLATE.md template, resolves the scenario's apps via
the app registry, fills in tool names/descriptions, and writes the final
AGENTS.md and exec-approvals.json.

Usage:
    python3 render_agent_prompt.py \\
        --scenario /var/gaia2/custom_scenario.json \\
        --template /opt/AGENTS_TEMPLATE.md \\
        --output-prompt /home/agent/AGENTS.md \\
        --output-approvals /home/agent/.openclaw/exec-approvals.json

Called by openclaw-setup.sh at container startup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_MESSAGING_CLI_NAMES = {"messages", "chats"}


def render(
    scenario_path: str,
    template_path: str,
    output_prompt: str | None = None,
    output_approvals: str | None = None,
    exec_tool: str = "exec",
    workspace_dir: str | None = None,
    workspace_out: str | None = None,
) -> str:
    """Render the agent prompt from template + scenario.

    When ``workspace_dir`` is given, every ``*.md`` in it is rendered with the
    same placeholders into ``workspace_out`` (an external harness's workspace
    bootstrap files).

    Returns the rendered prompt text.
    """
    from gaia2_cli.app_registry import resolve_scenario_tools

    result = resolve_scenario_tools(scenario_path)
    tools = result["tools"]

    # Build tool list for template
    tool_lines = []
    for cli_name, description in sorted(tools.items()):
        tool_lines.append(f"- `{cli_name}` — {description}")
    tool_list = "\n".join(tool_lines)

    # Build messaging tools list for rule 9
    messaging = [k for k in sorted(tools.keys()) if k in _MESSAGING_CLI_NAMES]
    messaging_tools = ", ".join(messaging) if messaging else "messages, chats"

    def fill(template: str) -> str:
        text = template.replace("{{TOOL_LIST}}", tool_list)
        text = text.replace("{{MESSAGING_TOOLS}}", messaging_tools)
        return text.replace("{{EXEC_TOOL}}", exec_tool)

    # Read and fill template
    prompt = fill(Path(template_path).read_text())

    if workspace_dir and workspace_out:
        out_dir = Path(workspace_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dir.chmod(0o755)
        for source in sorted(Path(workspace_dir).glob("*.md")):
            target = out_dir / source.name
            target.write_text(fill(source.read_text()))
            target.chmod(0o644)

    # Write prompt
    if output_prompt:
        Path(output_prompt).parent.mkdir(parents=True, exist_ok=True)
        Path(output_prompt).write_text(prompt)

    # Write exec-approvals.json
    if output_approvals:
        approvals = _build_exec_approvals(tools)
        Path(output_approvals).parent.mkdir(parents=True, exist_ok=True)
        Path(output_approvals).write_text(json.dumps(approvals, indent=2) + "\n")

    return prompt


def _build_exec_approvals(_tools: dict[str, str]) -> dict:
    """Build exec-approvals.json for unrestricted Gaia2 exec usage."""
    return {
        "version": 1,
        "defaults": {"security": "full", "ask": "off", "askFallback": "full"},
        "agents": {
            "main": {
                "security": "full",
                "ask": "off",
                "askFallback": "full",
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render agent prompt from scenario")
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON")
    parser.add_argument(
        "--template",
        default="/opt/AGENTS_TEMPLATE.md",
        help="Path to prompt template",
    )
    parser.add_argument("--output-prompt", help="Write rendered prompt to this path")
    parser.add_argument(
        "--output-approvals", help="Write exec-approvals.json to this path"
    )
    parser.add_argument(
        "--exec-tool",
        default="exec",
        help="Tool name for shell execution (exec for OpenClaw, terminal for Hermes)",
    )
    parser.add_argument(
        "--workspace-dir",
        help="Harness workspace dir whose *.md files are rendered too",
    )
    parser.add_argument(
        "--workspace-out", help="Output dir for the rendered workspace files"
    )
    args = parser.parse_args()

    prompt = render(
        args.scenario,
        args.template,
        output_prompt=args.output_prompt,
        output_approvals=args.output_approvals,
        exec_tool=args.exec_tool,
        workspace_dir=args.workspace_dir,
        workspace_out=args.workspace_out,
    )

    # If no output paths given, print to stdout
    if not args.output_prompt:
        print(prompt)


if __name__ == "__main__":
    main()
