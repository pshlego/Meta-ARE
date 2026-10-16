# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Generate OpenClaw runtime config from environment variables.

This module keeps provider-resolution logic out of shell so the config matrix
can be unit-tested directly. ``openclaw-setup.sh`` stays as thin bootstrap glue
that invokes this script and exports the resolved environment back into the
calling shell.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

OPENCLAW_VERSION = "2026.4.1"
DEFAULT_PROVIDER = "anthropic"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_COMPAT_CONTEXT_WINDOW = 262_144
DEFAULT_OPENAI_COMPAT_MAX_TOKENS = 65_535
DEFAULT_ANTHROPIC_CONTEXT_WINDOW = 200_000
DEFAULT_ANTHROPIC_MAX_TOKENS = 64_000
DEFAULT_GOOGLE_CONTEXT_WINDOW = 1_048_576
DEFAULT_GOOGLE_MAX_TOKENS = 65_536
# Without it, a runaway command holds the run for OpenClaw's 30-minute exec
# default, long past the daemon's idle timeout. The agent can still ask for
# more on a single call (exec timeout=...).
EXEC_TIMEOUT_SECONDS = 60
HEARTBEAT_CONTENT = """Check the System messages above for environment notifications and act on them.
Reply HEARTBEAT_OK only if nothing needs attention.
"""


class SetupError(RuntimeError):
    """Raised when config generation cannot proceed."""


@dataclass(frozen=True)
class ProviderSetup:
    """Resolved provider configuration for one OpenClaw session."""

    provider: str
    model: str
    api_key: str
    provider_config: dict[str, Any] | None
    shell_exports: OrderedDict[str, str]

    @property
    def uses_custom_provider(self) -> bool:
        return self.provider_config is not None


@dataclass(frozen=True)
class SetupOutputs:
    """All generated runtime artifacts derived from the environment."""

    provider_setup: ProviderSetup
    openclaw_config: dict[str, Any]
    auth_profiles: dict[str, Any]
    warnings: tuple[str, ...]


def first_nonempty(*values: str | None, default: str = "") -> str:
    for value in values:
        if value not in (None, ""):
            return value
    return default


def normalize_base_url(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    return value.strip().rstrip("/") or None


def build_openai_completions_provider(
    base_url: str,
    api_key: str,
    model: str,
    *,
    reasoning: bool = False,
    context_window: int = 200_000,
    max_tokens: int = 8_192,
) -> dict[str, Any]:
    return {
        "baseUrl": base_url,
        "apiKey": api_key,
        "api": "openai-completions",
        "authHeader": True,
        "models": [
            {
                "id": model,
                "name": model,
                "api": "openai-completions",
                "reasoning": reasoning,
                "input": ["text", "image"],
                "contextWindow": context_window,
                "maxTokens": max_tokens,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            }
        ],
    }


def build_openai_responses_provider(base_url: str, model: str) -> dict[str, Any]:
    return {
        "baseUrl": base_url,
        "api": "openai-responses",
        "models": [
            {
                "id": model,
                "name": model,
                "reasoning": True,
                "input": ["text", "image"],
                "contextWindow": 272_000,
                "maxTokens": 128_000,
            }
        ],
    }


def build_anthropic_messages_provider(base_url: str, model: str) -> dict[str, Any]:
    return {
        "baseUrl": base_url,
        "api": "anthropic-messages",
        "models": [
            {
                "id": model,
                "name": model,
                "reasoning": True,
                "input": ["text", "image"],
                "contextWindow": DEFAULT_ANTHROPIC_CONTEXT_WINDOW,
                "maxTokens": DEFAULT_ANTHROPIC_MAX_TOKENS,
            }
        ],
    }


def build_google_generative_ai_provider(base_url: str, model: str) -> dict[str, Any]:
    return {
        "baseUrl": base_url,
        "api": "google-generative-ai",
        "authHeader": True,
        "models": [
            {
                "id": model,
                "name": model,
                "reasoning": True,
                "input": ["text", "image"],
                "contextWindow": DEFAULT_GOOGLE_CONTEXT_WINDOW,
                "maxTokens": DEFAULT_GOOGLE_MAX_TOKENS,
            }
        ],
    }


def resolve_provider_setup(env: Mapping[str, str]) -> ProviderSetup:
    provider = first_nonempty(env.get("PROVIDER"), default=DEFAULT_PROVIDER)
    shell_exports: OrderedDict[str, str] = OrderedDict()

    if provider == "anthropic":
        model = first_nonempty(
            env.get("MODEL"), env.get("ANTHROPIC_MODEL"), default="claude-sonnet-4-6"
        )
        api_key = first_nonempty(env.get("API_KEY"), env.get("ANTHROPIC_API_KEY"))
        base_url = normalize_base_url(
            first_nonempty(env.get("ANTHROPIC_BASE_URL"), env.get("BASE_URL"))
        )
        provider_config = (
            None
            if base_url is None
            else build_anthropic_messages_provider(base_url, model)
        )
        shell_exports["ANTHROPIC_API_KEY"] = api_key
    elif provider == "openai":
        model = first_nonempty(
            env.get("MODEL"), env.get("OPENAI_MODEL"), default="gpt-4o"
        )
        api_key = first_nonempty(env.get("API_KEY"), env.get("OPENAI_API_KEY"))
        base_url = normalize_base_url(
            first_nonempty(env.get("OPENAI_BASE_URL"), env.get("BASE_URL"))
        )
        provider_config = (
            None
            if base_url is None
            else build_openai_responses_provider(base_url, model)
        )
        shell_exports["OPENAI_API_KEY"] = api_key
    elif provider == "google":
        model = first_nonempty(
            env.get("MODEL"),
            env.get("GEMINI_MODEL"),
            env.get("GOOGLE_MODEL"),
            default="gemini-3.1-pro-preview",
        )
        api_key = first_nonempty(
            env.get("API_KEY"),
            env.get("GEMINI_API_KEY"),
            env.get("GOOGLE_API_KEY"),
        )
        base_url = normalize_base_url(
            first_nonempty(
                env.get("GOOGLE_BASE_URL"),
                env.get("GEMINI_BASE_URL"),
                env.get("BASE_URL"),
            )
        )
        provider_config = (
            None
            if base_url is None
            else build_google_generative_ai_provider(base_url, model)
        )
        shell_exports["GEMINI_API_KEY"] = api_key
        shell_exports["GOOGLE_API_KEY"] = api_key
    elif provider == "openrouter":
        model = first_nonempty(
            env.get("MODEL"),
            env.get("OPENROUTER_MODEL"),
            default="moonshotai/kimi-k2.5",
        )
        api_key = first_nonempty(env.get("API_KEY"), env.get("OPENROUTER_API_KEY"))
        base_url = normalize_base_url(
            first_nonempty(
                env.get("OPENROUTER_BASE_URL"), default=DEFAULT_OPENROUTER_BASE_URL
            )
        )
        provider_config = build_openai_completions_provider(
            base_url,
            api_key,
            model,
            reasoning=True,
            context_window=DEFAULT_OPENAI_COMPAT_CONTEXT_WINDOW,
            max_tokens=DEFAULT_OPENAI_COMPAT_MAX_TOKENS,
        )
    elif provider in {"openai-compat", "openai-completions"}:
        model = first_nonempty(
            env.get("MODEL"),
            env.get("OPENAI_COMPAT_MODEL"),
            env.get("OPENAI_COMPLETIONS_MODEL"),
        )
        api_key = first_nonempty(
            env.get("API_KEY"),
            env.get("OPENAI_COMPAT_API_KEY"),
            env.get("OPENAI_COMPLETIONS_API_KEY"),
        )
        base_url = normalize_base_url(
            first_nonempty(
                env.get("OPENAI_COMPAT_BASE_URL"),
                env.get("OPENAI_COMPLETIONS_BASE_URL"),
                env.get("BASE_URL"),
            )
        )
        if not model:
            raise SetupError(f"MODEL is required for provider '{provider}'")
        if base_url is None:
            raise SetupError(f"BASE_URL is required for provider '{provider}'")
        provider_config = build_openai_completions_provider(
            base_url,
            api_key,
            model,
            reasoning=True,
            context_window=DEFAULT_OPENAI_COMPAT_CONTEXT_WINDOW,
            max_tokens=DEFAULT_OPENAI_COMPAT_MAX_TOKENS,
        )
    else:
        raise SetupError(f"Unknown provider '{provider}'")

    exports = OrderedDict()
    exports["PROVIDER"] = provider
    exports["MODEL"] = model
    exports["API_KEY"] = api_key
    for key, value in shell_exports.items():
        exports[key] = value
    exports["OPENCLAW_GATEWAY_TOKEN"] = first_nonempty(
        env.get("OPENCLAW_GATEWAY_TOKEN"), default="gaia2-notif"
    )
    exports["OPENCLAW_HOOKS_TOKEN"] = first_nonempty(
        env.get("OPENCLAW_HOOKS_TOKEN"), default="gaia2-hooks"
    )

    return ProviderSetup(
        provider=provider,
        model=model,
        api_key=api_key,
        provider_config=provider_config,
        shell_exports=exports,
    )


def resolve_context_tokens(env: Mapping[str, str]) -> int | None:
    """Return the context window override from ``OPENCLAW_CONTEXT_WINDOW``."""
    value = first_nonempty(env.get("OPENCLAW_CONTEXT_WINDOW"))
    if not value:
        return None
    try:
        tokens = int(value)
    except ValueError:
        tokens = 0
    if tokens < 1:
        raise SetupError(
            f"OPENCLAW_CONTEXT_WINDOW must be a positive integer, got {value!r}"
        )
    return tokens


def build_openclaw_config(
    provider_setup: ProviderSetup,
    env: Mapping[str, str],
    *,
    now: str | None = None,
) -> dict[str, Any]:
    if now is None:
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    gateway_port = first_nonempty(env.get("OPENCLAW_GATEWAY_PORT"))
    thinking_default = first_nonempty(env.get("THINKING"), default="low")
    timeout_seconds = int(
        first_nonempty(env.get("AGENT_TIMEOUT_SECONDS"), default="1200")
    )
    agent_defaults: dict[str, Any] = {
        "skipBootstrap": True,
        "thinkingDefault": thinking_default,
        "timeoutSeconds": timeout_seconds,
        "model": {"primary": f"{provider_setup.provider}/{provider_setup.model}"},
        "workspace": "/home/agent",
        "maxConcurrent": 4,
        "subagents": {"maxConcurrent": 8},
        "envelopeTimestamp": "off",
        "envelopeElapsed": "off",
        "heartbeat": {"every": "24h"},
        "llm": {"idleTimeoutSeconds": 300},
    }
    context_tokens = resolve_context_tokens(env)
    if context_tokens is not None:
        # Agent-level window OpenClaw compacts against, whatever the model's.
        agent_defaults["contextTokens"] = context_tokens
    gateway: dict[str, Any] = {
        "bind": "loopback",
        "mode": "local",
        "auth": {
            "mode": "token",
            "token": provider_setup.shell_exports["OPENCLAW_GATEWAY_TOKEN"],
        },
        "controlUi": {"allowInsecureAuth": True},
    }
    if gateway_port:
        gateway["port"] = int(gateway_port)

    return {
        "models": (
            {"providers": {provider_setup.provider: provider_setup.provider_config}}
            if provider_setup.provider_config is not None
            else {}
        ),
        "agents": {"defaults": agent_defaults},
        "tools": {
            "allow": ["exec"],
            "exec": {
                "security": "full",
                "ask": "off",
                "safeBins": [],
                "timeoutSec": EXEC_TIMEOUT_SECONDS,
            },
            "elevated": {"enabled": False},
        },
        "channels": {"defaults": {"heartbeat": {"showOk": True, "showAlerts": True}}},
        "commands": {"native": "auto", "nativeSkills": "auto"},
        "plugins": {"entries": {}},
        "messages": {
            "messagePrefix": "none",
            "ackReactionScope": "group-mentions",
            "queue": {"mode": "steer"},
        },
        "hooks": {
            "enabled": True,
            "token": provider_setup.shell_exports["OPENCLAW_HOOKS_TOKEN"],
            "path": "/hooks",
        },
        "gateway": gateway,
        "wizard": {
            "lastRunAt": now,
            "lastRunVersion": OPENCLAW_VERSION,
            "lastRunCommand": "doctor",
            "lastRunMode": "local",
        },
        "meta": {
            "lastTouchedVersion": OPENCLAW_VERSION,
            "lastTouchedAt": now,
        },
    }


def build_auth_profiles(provider_setup: ProviderSetup) -> dict[str, Any]:
    if provider_setup.uses_custom_provider and provider_setup.api_key:
        return {provider_setup.provider: {"apiKey": provider_setup.api_key}}
    return {}


HARNESS_TOOL_PATH = "/opt/harness_tool.py"
HARNESS_POLICY_PATH = "/opt/gaia2_harness_policy.json"
WORKSPACE_BOOTSTRAP_FILES = ("SOUL.md", "TOOLS.md", "IDENTITY.md", "USER.md", "HEARTBEAT.md")


def _load_harness_tool(env: Mapping[str, str]):
    import importlib.util

    path = first_nonempty(env.get("GAIA2_HARNESS_TOOL"), default=HARNESS_TOOL_PATH)
    spec = importlib.util.spec_from_file_location("gaia2_harness_tool", path)
    if spec is None or spec.loader is None:
        raise SetupError(f"cannot load harness tool from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def apply_harness(config: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Merge an external harness (GAIA2_HARNESS_DIR) into the OpenClaw config.

    The overlay is validated against the policy baked into the image (keys the
    benchmark depends on -- provider, auth, tool allowlist, gateway -- are not
    overridable) and harness plugins are wired into plugins.load.paths.
    Without GAIA2_HARNESS_DIR the config is returned unchanged.
    """
    harness_value = first_nonempty(env.get("GAIA2_HARNESS_DIR"))
    if not harness_value:
        return config
    harness_dir = Path(harness_value)
    tool = _load_harness_tool(env)
    policy = tool.load_policy(
        Path(first_nonempty(env.get("GAIA2_HARNESS_POLICY"), default=HARNESS_POLICY_PATH))
    )
    overlay = json.loads((harness_dir / "openclaw.overlay.json").read_text(encoding="utf-8"))
    plugins_root = harness_dir / "plugins"
    plugin_ids = sorted(
        path.name
        for path in (plugins_root.iterdir() if plugins_root.is_dir() else ())
        if (path / "openclaw.plugin.json").is_file()
    )
    try:
        tool.validate_overlay(overlay, policy, plugin_ids=set(plugin_ids))
    except ValueError as exc:
        raise SetupError(f"invalid harness overlay: {exc}") from exc
    merged = tool.merge_overlay(config, overlay)
    # The root entrypoint stages root-owned plugin copies (OpenClaw rejects
    # plugins owned by other uids, e.g. a bind mount's host user).
    stage_value = first_nonempty(env.get("GAIA2_HARNESS_STAGE_DIR"))
    if stage_value and (Path(stage_value) / "plugins").is_dir():
        plugins_root = Path(stage_value) / "plugins"
    if plugin_ids:
        plugins = merged.setdefault("plugins", {})
        load = plugins.setdefault("load", {})
        load["paths"] = [*load.get("paths", []), *(str(plugins_root / pid) for pid in plugin_ids)]
        entries = plugins.setdefault("entries", {})
        for plugin_id in plugin_ids:
            entries[plugin_id] = {**entries.get(plugin_id, {}), "enabled": True}
    return merged


def generate_setup(env: Mapping[str, str], *, now: str | None = None) -> SetupOutputs:
    provider_setup = resolve_provider_setup(env)
    warnings: list[str] = []
    if not provider_setup.api_key:
        warnings.append(
            f"API_KEY not set (provider={provider_setup.provider}). OpenClaw LLM calls will fail."
        )
    return SetupOutputs(
        provider_setup=provider_setup,
        openclaw_config=apply_harness(
            build_openclaw_config(provider_setup, env, now=now), env
        ),
        auth_profiles=build_auth_profiles(provider_setup),
        warnings=tuple(warnings),
    )


def render_shell_exports(shell_exports: Mapping[str, str]) -> str:
    return (
        "\n".join(
            f"export {key}={shlex.quote(value)}" for key, value in shell_exports.items()
        )
        + "\n"
    )


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_runtime_files(home: Path, outputs: SetupOutputs) -> None:
    openclaw_dir = home / ".openclaw"
    auth_dir = openclaw_dir / "agents" / "main" / "agent"

    atomic_write_text(
        openclaw_dir / "openclaw.json",
        json.dumps(outputs.openclaw_config, indent=2) + "\n",
    )
    atomic_write_text(
        auth_dir / "auth-profiles.json",
        json.dumps(outputs.auth_profiles, indent=2) + "\n",
    )

    # An external harness's workspace files are rendered (placeholders filled)
    # by the root entrypoint into GAIA2_HARNESS_STAGE_DIR/workspace.
    stage_value = os.environ.get("GAIA2_HARNESS_STAGE_DIR", "")
    staged = Path(stage_value) / "workspace" if stage_value else None
    defaults = {name: "" for name in WORKSPACE_BOOTSTRAP_FILES}
    defaults["HEARTBEAT.md"] = HEARTBEAT_CONTENT
    for name in WORKSPACE_BOOTSTRAP_FILES:
        source = staged / name if staged is not None else None
        content = (
            source.read_text(encoding="utf-8")
            if source is not None and source.is_file()
            else defaults[name]
        )
        atomic_write_text(home / name, content)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--home", help="Target home directory for generated OpenClaw files."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    home_value = args.home or os.environ.get("HOME")
    if not home_value:
        print("[openclaw-setup] Error: HOME is not set", file=sys.stderr)
        return 1

    try:
        outputs = generate_setup(os.environ)
        write_runtime_files(Path(home_value), outputs)
    except (SetupError, ValueError) as exc:
        print(f"[openclaw-setup] Error: {exc}", file=sys.stderr)
        return 1

    for warning in outputs.warnings:
        print(f"[openclaw-setup] Warning: {warning}", file=sys.stderr)

    sys.stdout.write(render_shell_exports(outputs.provider_setup.shell_exports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
