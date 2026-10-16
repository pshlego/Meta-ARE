# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Externalized OpenClaw harness: digest, structural check and config overlay.

A harness directory (``gaia2-cli/harness/openclaw`` in the repository, mounted
read-only at ``/opt/gaia2_harness`` in the container) holds everything an
optimizer may change about the OpenClaw agent::

    HARNESS.json             {"schema_version": "gaia2-openclaw-harness/v1"}
    AGENTS_TEMPLATE.md       system-prompt template rendered to ~/AGENTS.md
    openclaw.overlay.json    restricted overlay merged into openclaw.json
    workspace/<NAME>.md      SOUL/TOOLS/IDENTITY/USER/HEARTBEAT bootstrap files
    skills/<name>/SKILL.md   workspace skills (+ optional scripts/)
    plugins/<id>/...         local OpenClaw plugins (openclaw.plugin.json + JS)

This module is stdlib-only so the container entrypoint, the runner and the
AutoSaddler verifier share one implementation of the rules. The overlay policy
lives next to the harness (``harness/policy/openclaw_overlay_policy.json``)
and is baked into the image, so a candidate harness cannot widen it.

CLI::

    python3 harness_tool.py digest DIR
    python3 harness_tool.py check DIR --policy POLICY [--phase steering]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

HARNESS_SCHEMA_VERSION = "gaia2-openclaw-harness/v1"
POLICY_SCHEMA_VERSION = "gaia2-openclaw-overlay-policy/v1"
WORKSPACE_FILES = ("SOUL.md", "TOOLS.md", "IDENTITY.md", "USER.md", "HEARTBEAT.md")
PLACEHOLDERS = ("{{TOOL_LIST}}", "{{MESSAGING_TOOLS}}", "{{EXEC_TOOL}}")
REQUIRED_TEMPLATE_PLACEHOLDERS = ("{{TOOL_LIST}}", "{{EXEC_TOOL}}")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_PLACEHOLDER_RE = re.compile(r"\{\{[^{}]*\}\}")
_SKILL_SCRIPT_SUFFIXES = (".sh", ".py")
_PLUGIN_SUFFIXES = (".js", ".mjs", ".cjs", ".json")
_IGNORED_NAMES = (".gitkeep",)
DEFAULT_LIMITS = {
    "markdown_file_chars": 20_000,
    "markdown_total_chars": 60_000,
    "max_skills": 16,
    "max_plugins": 4,
    "plugin_file_bytes": 65_536,
}


class HarnessError(ValueError):
    """Raised when a harness directory or overlay violates the contract."""


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------


def harness_files(root: Path) -> list[PurePosixPath]:
    """Return every regular file below ``root`` (relative, sorted)."""
    root = Path(root)
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise HarnessError(f"symlinks are not allowed in a harness: {path.relative_to(root)}")
        if path.is_file():
            files.append(PurePosixPath(path.relative_to(root).as_posix()))
    return files


def digest(root: Path) -> str:
    """sha256 over the sorted (path, file sha256) list of a harness directory."""
    root = Path(root)
    records = [
        {"path": rel.as_posix(), "sha256": hashlib.sha256((root / rel).read_bytes()).hexdigest()}
        for rel in harness_files(root)
    ]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Structural check
# ---------------------------------------------------------------------------


def load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise HarnessError(f"unsupported overlay policy: {path}")
    if not isinstance(policy.get("allowed"), list):
        raise HarnessError("overlay policy needs an 'allowed' list")
    return policy


def check(root: Path, policy: dict[str, Any], *, phase: str | None = None) -> None:
    """Validate a harness directory; raise HarnessError on the first problem."""
    root = Path(root)
    limits = {**DEFAULT_LIMITS, **(policy.get("limits") or {})}
    files = harness_files(root)
    names = {rel.as_posix() for rel in files}

    manifest = _json_file(root / "HARNESS.json", "HARNESS.json")
    if manifest != {"schema_version": HARNESS_SCHEMA_VERSION}:
        raise HarnessError(f"HARNESS.json must be exactly {{\"schema_version\": \"{HARNESS_SCHEMA_VERSION}\"}}")
    for required in ("AGENTS_TEMPLATE.md", "openclaw.overlay.json"):
        if required not in names:
            raise HarnessError(f"missing required harness file: {required}")

    skills: set[str] = set()
    plugins: set[str] = set()
    markdown_total = 0
    for rel in files:
        parts = rel.parts
        text_path = root / rel
        if rel.name in _IGNORED_NAMES:
            continue
        if parts[0] in ("HARNESS.json", "AGENTS_TEMPLATE.md", "openclaw.overlay.json") and len(parts) == 1:
            pass
        elif parts[0] == "workspace" and len(parts) == 2 and parts[1] in WORKSPACE_FILES:
            pass
        elif parts[0] == "skills" and len(parts) >= 3 and _NAME_RE.match(parts[1]):
            skills.add(parts[1])
            if not (
                (len(parts) == 3 and parts[2] == "SKILL.md")
                or (len(parts) == 4 and parts[2] == "scripts" and rel.suffix in _SKILL_SCRIPT_SUFFIXES)
            ):
                raise HarnessError(f"unexpected file in skill: {rel}")
            if phase == "steering" and parts[2] == "scripts":
                raise HarnessError(f"steering harnesses may not add skill scripts: {rel}")
        elif parts[0] == "plugins" and len(parts) == 3 and _NAME_RE.match(parts[1]):
            plugins.add(parts[1])
            if rel.suffix not in _PLUGIN_SUFFIXES:
                raise HarnessError(f"unexpected file in plugin: {rel}")
            if text_path.stat().st_size > limits["plugin_file_bytes"]:
                raise HarnessError(f"plugin file too large: {rel}")
        else:
            raise HarnessError(f"file outside the harness layout: {rel}")

        try:
            text = text_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise HarnessError(f"harness files must be UTF-8: {rel}") from exc
        if rel.suffix == ".md":
            markdown_total += len(text)
            if len(text) > limits["markdown_file_chars"]:
                raise HarnessError(
                    f"{rel} has {len(text)} chars; limit is {limits['markdown_file_chars']}"
                )
            unknown = sorted(set(_PLACEHOLDER_RE.findall(text)) - set(PLACEHOLDERS))
            if unknown and parts[0] != "skills":
                raise HarnessError(f"{rel} uses unknown placeholders: {unknown}")

    if markdown_total > limits["markdown_total_chars"]:
        raise HarnessError(
            f"harness markdown totals {markdown_total} chars; limit is {limits['markdown_total_chars']}"
        )
    template = (root / "AGENTS_TEMPLATE.md").read_text(encoding="utf-8")
    for placeholder in REQUIRED_TEMPLATE_PLACEHOLDERS:
        if placeholder not in template:
            raise HarnessError(f"AGENTS_TEMPLATE.md must keep the {placeholder} placeholder")
    if len(skills) > limits["max_skills"]:
        raise HarnessError(f"at most {limits['max_skills']} skills are allowed")
    if len(plugins) > limits["max_plugins"]:
        raise HarnessError(f"at most {limits['max_plugins']} plugins are allowed")
    for name in sorted(skills):
        _check_skill(root / "skills" / name, name)
    for plugin_id in sorted(plugins):
        _check_plugin(root / "plugins" / plugin_id, plugin_id)

    overlay = _json_file(root / "openclaw.overlay.json", "openclaw.overlay.json")
    validate_overlay(overlay, policy, plugin_ids=plugins)
    if phase == "steering":
        steering_violations = [
            rel.as_posix()
            for rel in files
            if rel.suffix != ".md" and rel.name not in _IGNORED_NAMES
            and rel.as_posix() not in ("HARNESS.json", "openclaw.overlay.json")
        ]
        if steering_violations:
            raise HarnessError(f"steering harnesses may hold Markdown only: {steering_violations}")


def _check_skill(skill_dir: Path, name: str) -> None:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        raise HarnessError(f"skill {name} has no SKILL.md")
    text = skill_md.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        raise HarnessError(f"skills/{name}/SKILL.md needs YAML frontmatter")
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "\t")):
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip().strip("\"'")
    if fields.get("name") != name:
        raise HarnessError(f"skills/{name}/SKILL.md frontmatter name must be {name!r}")
    if not fields.get("description"):
        raise HarnessError(f"skills/{name}/SKILL.md frontmatter needs a description")


def _check_plugin(plugin_dir: Path, plugin_id: str) -> None:
    manifest = _json_file(plugin_dir / "openclaw.plugin.json", f"plugins/{plugin_id}/openclaw.plugin.json")
    if manifest.get("id") != plugin_id:
        raise HarnessError(f"plugins/{plugin_id}/openclaw.plugin.json id must be {plugin_id!r}")
    if not isinstance(manifest.get("configSchema"), dict):
        raise HarnessError(f"plugins/{plugin_id}/openclaw.plugin.json needs a configSchema object")
    entries = [p for p in plugin_dir.iterdir() if p.suffix in (".js", ".mjs", ".cjs")]
    if not entries:
        raise HarnessError(f"plugins/{plugin_id} has no JavaScript entry file")


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------


def _leaf_paths(value: Any, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict) and value:
        out = []
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise HarnessError("overlay keys must be non-empty strings")
            out.extend(_leaf_paths(item, (*prefix, key)))
        return out
    return [(prefix, value)]


def _rule_matches(rule_path: tuple[str, ...], path: tuple[str, ...]) -> bool:
    if len(path) < len(rule_path):
        return False
    return all(r == "*" or r == p for r, p in zip(rule_path, path))


def validate_overlay(
    overlay: Any, policy: dict[str, Any], *, plugin_ids: set[str] | None = None
) -> None:
    """Every overlay leaf must sit under an allowed policy path and pass its bounds."""
    if not isinstance(overlay, dict):
        raise HarnessError("openclaw.overlay.json must contain a JSON object")
    rules = [(tuple(rule["path"].split(".")), rule) for rule in policy["allowed"]]
    for path, value in _leaf_paths(overlay):
        if not path:
            continue
        if value is None:
            raise HarnessError(f"overlay value at {'.'.join(path)} must not be null")
        matching = [rule for rule_path, rule in rules if _rule_matches(rule_path, path)]
        if not matching:
            raise HarnessError(f"overlay path {'.'.join(path)} is not allowed by the policy")
        rule = max(matching, key=lambda item: len(item["path"].split(".")))
        exact = len(rule["path"].split(".")) == len(path)
        if exact:
            _check_value(".".join(path), value, rule)
        if rule.get("plugin_scoped") and plugin_ids is not None:
            plugin = path[2] if len(path) > 2 else None
            if plugin not in plugin_ids:
                raise HarnessError(
                    f"overlay path {'.'.join(path)} must name a harness plugin ({sorted(plugin_ids)})"
                )


def _check_value(label: str, value: Any, rule: dict[str, Any]) -> None:
    kind = rule.get("type")
    checks = {
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "string": lambda v: isinstance(v, str),
        "boolean": lambda v: isinstance(v, bool),
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
    }
    if kind is not None and not checks[kind](value):
        raise HarnessError(f"overlay value at {label} must be a {kind}")
    if "enum" in rule and value not in rule["enum"]:
        raise HarnessError(f"overlay value at {label} must be one of {rule['enum']}")
    if "min" in rule and value < rule["min"]:
        raise HarnessError(f"overlay value at {label} must be >= {rule['min']}")
    if "max" in rule and value > rule["max"]:
        raise HarnessError(f"overlay value at {label} must be <= {rule['max']}")
    if kind == "array" and "items_enum" in rule:
        bad = [item for item in value if item not in rule["items_enum"]]
        if bad:
            raise HarnessError(f"overlay value at {label} has unsupported items: {bad}")


def merge_overlay(config: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``overlay`` into ``config``: dicts merge, other values replace."""
    merged = dict(config)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_overlay(merged[key], value)
        else:
            merged[key] = value
    return merged


def _json_file(path: Path, label: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HarnessError(f"missing {label}") from exc
    except json.JSONDecodeError as exc:
        raise HarnessError(f"{label} is not valid JSON: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_digest = sub.add_parser("digest")
    p_digest.add_argument("dir")
    p_check = sub.add_parser("check")
    p_check.add_argument("dir")
    p_check.add_argument("--policy", required=True)
    p_check.add_argument("--phase", choices=("capability", "steering"), default=None)
    args = parser.parse_args(argv)
    try:
        if args.command == "digest":
            print(digest(Path(args.dir)))
        else:
            check(Path(args.dir), load_policy(Path(args.policy)), phase=args.phase)
            print("ok")
    except (HarnessError, OSError) as exc:
        print(f"harness error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
