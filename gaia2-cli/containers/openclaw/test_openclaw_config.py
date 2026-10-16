# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import openclaw_config

FIXED_NOW = "2026-04-08T21:00:00.000Z"
SCRIPT = Path(__file__).resolve().with_name("openclaw_config.py")


def test_openai_without_base_url_uses_builtin_provider() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "openai",
            "API_KEY": "openai-key",
            "MODEL": "gpt-5.4",
        },
        now=FIXED_NOW,
    )

    assert outputs.provider_setup.provider == "openai"
    assert outputs.provider_setup.provider_config is None
    assert outputs.provider_setup.shell_exports["OPENAI_API_KEY"] == "openai-key"
    assert outputs.openclaw_config["models"] == {}
    assert outputs.auth_profiles == {}
    assert (
        outputs.openclaw_config["agents"]["defaults"]["model"]["primary"]
        == "openai/gpt-5.4"
    )


def test_openai_with_base_url_uses_custom_responses_provider() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "openai",
            "API_KEY": "openai-key",
            "MODEL": "custom-openai-model",
            "BASE_URL": "https://proxy.example/v1/",
        },
        now=FIXED_NOW,
    )

    provider_cfg = outputs.openclaw_config["models"]["providers"]["openai"]
    assert provider_cfg == {
        "baseUrl": "https://proxy.example/v1",
        "api": "openai-responses",
        "models": [
            {
                "id": "custom-openai-model",
                "name": "custom-openai-model",
                "reasoning": True,
                "input": ["text", "image"],
                "contextWindow": 272000,
                "maxTokens": 128000,
            }
        ],
    }
    assert outputs.auth_profiles == {"openai": {"apiKey": "openai-key"}}


def test_google_without_base_url_uses_builtin_provider() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "google",
            "GEMINI_API_KEY": "google-key",
        },
        now=FIXED_NOW,
    )

    assert outputs.provider_setup.provider == "google"
    assert outputs.provider_setup.model == "gemini-3.1-pro-preview"
    assert outputs.provider_setup.provider_config is None
    assert outputs.provider_setup.shell_exports["GEMINI_API_KEY"] == "google-key"
    assert outputs.provider_setup.shell_exports["GOOGLE_API_KEY"] == "google-key"
    assert outputs.openclaw_config["models"] == {}
    assert outputs.auth_profiles == {}


def test_google_with_base_url_uses_custom_native_provider() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "google",
            "API_KEY": "custom-api-key",
            "MODEL": "custom-google-model",
            "BASE_URL": "https://provider.example/google/v1/",
        },
        now=FIXED_NOW,
    )

    provider_cfg = outputs.openclaw_config["models"]["providers"]["google"]
    assert provider_cfg == {
        "baseUrl": "https://provider.example/google/v1",
        "api": "google-generative-ai",
        "authHeader": True,
        "models": [
            {
                "id": "custom-google-model",
                "name": "custom-google-model",
                "reasoning": True,
                "input": ["text", "image"],
                "contextWindow": 1048576,
                "maxTokens": 65536,
            }
        ],
    }
    assert outputs.auth_profiles == {"google": {"apiKey": "custom-api-key"}}


def test_anthropic_without_base_url_uses_builtin_provider() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "anthropic-key",
        },
        now=FIXED_NOW,
    )

    assert outputs.provider_setup.provider == "anthropic"
    assert outputs.provider_setup.model == "claude-sonnet-4-6"
    assert outputs.provider_setup.provider_config is None
    assert outputs.provider_setup.shell_exports["ANTHROPIC_API_KEY"] == "anthropic-key"
    assert outputs.openclaw_config["models"] == {}
    assert outputs.auth_profiles == {}


def test_anthropic_with_base_url_uses_custom_messages_provider() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "anthropic",
            "API_KEY": "anthropic-key",
            "MODEL": "custom-anthropic-model",
            "BASE_URL": "https://proxy.example/anthropic/",
        },
        now=FIXED_NOW,
    )

    provider_cfg = outputs.openclaw_config["models"]["providers"]["anthropic"]
    assert provider_cfg == {
        "baseUrl": "https://proxy.example/anthropic",
        "api": "anthropic-messages",
        "models": [
            {
                "id": "custom-anthropic-model",
                "name": "custom-anthropic-model",
                "reasoning": True,
                "input": ["text", "image"],
                "contextWindow": 200000,
                "maxTokens": 64000,
            }
        ],
    }
    assert outputs.auth_profiles == {"anthropic": {"apiKey": "anthropic-key"}}


def test_default_provider_is_anthropic_builtin() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "ANTHROPIC_API_KEY": "anthropic-key",
        },
        now=FIXED_NOW,
    )

    assert outputs.provider_setup.provider == "anthropic"
    assert outputs.provider_setup.model == "claude-sonnet-4-6"
    assert outputs.provider_setup.shell_exports["ANTHROPIC_API_KEY"] == "anthropic-key"
    assert outputs.provider_setup.provider_config is None
    assert outputs.openclaw_config["models"] == {}
    assert outputs.auth_profiles == {}


def test_openrouter_uses_custom_chat_completions_defaults() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "openrouter-key",
        },
        now=FIXED_NOW,
    )

    provider_cfg = outputs.openclaw_config["models"]["providers"]["openrouter"]
    model_cfg = provider_cfg["models"][0]
    assert outputs.provider_setup.model == "moonshotai/kimi-k2.5"
    assert provider_cfg["baseUrl"] == openclaw_config.DEFAULT_OPENROUTER_BASE_URL
    assert model_cfg["reasoning"] is True
    assert model_cfg["contextWindow"] == 262144
    assert model_cfg["maxTokens"] == 65535


def test_openai_compat_requires_base_url_and_model() -> None:
    with pytest.raises(openclaw_config.SetupError, match="MODEL is required"):
        openclaw_config.generate_setup(
            {
                "PROVIDER": "openai-compat",
                "BASE_URL": "https://compat.example/v1",
            },
            now=FIXED_NOW,
        )

    with pytest.raises(openclaw_config.SetupError, match="BASE_URL is required"):
        openclaw_config.generate_setup(
            {
                "PROVIDER": "openai-compat",
                "MODEL": "compat-model",
            },
            now=FIXED_NOW,
        )


def test_openai_compat_uses_custom_chat_completions_provider() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "openai-compat",
            "API_KEY": "compat-key",
            "MODEL": "compat-model",
            "BASE_URL": "https://compat.example/v1/",
        },
        now=FIXED_NOW,
    )

    provider_cfg = outputs.openclaw_config["models"]["providers"]["openai-compat"]
    assert outputs.provider_setup.provider == "openai-compat"
    assert outputs.openclaw_config["agents"]["defaults"]["model"]["primary"] == (
        "openai-compat/compat-model"
    )
    assert provider_cfg == {
        "baseUrl": "https://compat.example/v1",
        "apiKey": "compat-key",
        "api": "openai-completions",
        "authHeader": True,
        "models": [
            {
                "id": "compat-model",
                "name": "compat-model",
                "api": "openai-completions",
                "reasoning": True,
                "input": ["text", "image"],
                "contextWindow": 262144,
                "maxTokens": 65535,
                "cost": {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                },
            }
        ],
    }
    assert outputs.auth_profiles == {"openai-compat": {"apiKey": "compat-key"}}


def test_openai_completions_alias_matches_openai_compat() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "openai-completions",
            "API_KEY": "compat-key",
            "MODEL": "compat-model",
            "BASE_URL": "https://compat.example/v1/",
        },
        now=FIXED_NOW,
    )

    assert outputs.provider_setup.provider == "openai-completions"
    assert (
        outputs.openclaw_config["models"]["providers"]["openai-completions"]["api"]
        == "openai-completions"
    )


def test_unknown_provider_raises_setup_error() -> None:
    with pytest.raises(openclaw_config.SetupError):
        openclaw_config.generate_setup({"PROVIDER": "bogus"}, now=FIXED_NOW)


def test_write_runtime_files_creates_expected_outputs(tmp_path: Path) -> None:
    home = tmp_path / "home"
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "openai",
            "API_KEY": "openai-key",
            "MODEL": "custom-openai-model",
            "BASE_URL": "https://proxy.example/v1",
        },
        now=FIXED_NOW,
    )

    openclaw_config.write_runtime_files(home, outputs)

    openclaw_json = json.loads((home / ".openclaw" / "openclaw.json").read_text())
    auth_profiles = json.loads(
        (
            home / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"
        ).read_text()
    )

    assert openclaw_json == outputs.openclaw_config
    assert auth_profiles == outputs.auth_profiles
    assert (home / "HEARTBEAT.md").read_text() == openclaw_config.HEARTBEAT_CONTENT
    for name in ("SOUL.md", "TOOLS.md", "IDENTITY.md", "USER.md"):
        assert (home / name).read_text() == ""


def test_cli_writes_files_and_emits_shell_exports(tmp_path: Path) -> None:
    home = tmp_path / "home"
    env = os.environ.copy()
    env.update(
        {
            "PROVIDER": "anthropic",
            "API_KEY": "anthropic-key",
            "BASE_URL": "https://proxy.example/anthropic",
            "HOME": str(home),
        }
    )

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--home", str(home)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "export PROVIDER=anthropic" in proc.stdout
    assert "export ANTHROPIC_API_KEY=anthropic-key" in proc.stdout

    openclaw_json = json.loads((home / ".openclaw" / "openclaw.json").read_text())
    assert (
        openclaw_json["models"]["providers"]["anthropic"]["api"] == "anthropic-messages"
    )


def test_context_window_override_sets_agent_context_tokens() -> None:
    outputs = openclaw_config.generate_setup(
        {
            "PROVIDER": "anthropic",
            "API_KEY": "anthropic-key",
            "OPENCLAW_CONTEXT_WINDOW": "64000",
        },
        now=FIXED_NOW,
    )

    assert outputs.openclaw_config["agents"]["defaults"]["contextTokens"] == 64000


def test_context_tokens_are_left_to_the_model_by_default() -> None:
    outputs = openclaw_config.generate_setup(
        {"PROVIDER": "anthropic", "API_KEY": "anthropic-key"},
        now=FIXED_NOW,
    )

    assert "contextTokens" not in outputs.openclaw_config["agents"]["defaults"]


@pytest.mark.parametrize("value", ["0", "-5", "lots"])
def test_invalid_context_window_override_is_rejected(value: str) -> None:
    with pytest.raises(openclaw_config.SetupError, match="OPENCLAW_CONTEXT_WINDOW"):
        openclaw_config.generate_setup(
            {
                "PROVIDER": "anthropic",
                "API_KEY": "anthropic-key",
                "OPENCLAW_CONTEXT_WINDOW": value,
            },
            now=FIXED_NOW,
        )


def test_exec_commands_time_out_after_a_minute_by_default() -> None:
    outputs = openclaw_config.generate_setup(
        {"PROVIDER": "anthropic", "API_KEY": "anthropic-key"},
        now=FIXED_NOW,
    )

    assert outputs.openclaw_config["tools"]["exec"]["timeoutSec"] == 60


# ---------------------------------------------------------------------------
# External harness (GAIA2_HARNESS_DIR)
# ---------------------------------------------------------------------------

GAIA2_CLI = Path(__file__).resolve().parents[2]
SEED_HARNESS = GAIA2_CLI / "harness" / "openclaw"
HARNESS_ENV = {
    "GAIA2_HARNESS_TOOL": str(GAIA2_CLI / "shared" / "harness_tool.py"),
    "GAIA2_HARNESS_POLICY": str(
        GAIA2_CLI / "harness" / "policy" / "openclaw_overlay_policy.json"
    ),
}
COMPAT_ENV = {
    "PROVIDER": "openai-compat",
    "API_KEY": "k",
    "MODEL": "m",
    "BASE_URL": "http://127.0.0.1:1/v1",
}


def _harness_copy(tmp_path: Path) -> Path:
    import shutil

    target = tmp_path / "harness"
    shutil.copytree(SEED_HARNESS, target)
    return target


def test_seed_harness_leaves_config_unchanged() -> None:
    plain = openclaw_config.generate_setup(COMPAT_ENV, now=FIXED_NOW)
    seeded = openclaw_config.generate_setup(
        {**COMPAT_ENV, **HARNESS_ENV, "GAIA2_HARNESS_DIR": str(SEED_HARNESS)},
        now=FIXED_NOW,
    )
    assert seeded.openclaw_config == plain.openclaw_config


def test_seed_workspace_files_match_defaults() -> None:
    assert (SEED_HARNESS / "workspace" / "HEARTBEAT.md").read_text() == (
        openclaw_config.HEARTBEAT_CONTENT
    )


def test_harness_overlay_and_plugins_are_merged(tmp_path: Path) -> None:
    harness = _harness_copy(tmp_path)
    (harness / "openclaw.overlay.json").write_text(
        json.dumps(
            {
                "messages": {"queue": {"mode": "followup"}},
                "plugins": {"entries": {"guard": {"config": {"strict": True}}}},
            }
        )
    )
    plugin = harness / "plugins" / "guard"
    plugin.mkdir()
    (plugin / "openclaw.plugin.json").write_text(
        json.dumps({"id": "guard", "configSchema": {"type": "object"}})
    )
    (plugin / "index.js").write_text("export default {id: 'guard', register() {}};\n")

    config = openclaw_config.generate_setup(
        {**COMPAT_ENV, **HARNESS_ENV, "GAIA2_HARNESS_DIR": str(harness)}, now=FIXED_NOW
    ).openclaw_config

    assert config["messages"]["queue"] == {"mode": "followup"}
    assert config["messages"]["messagePrefix"] == "none"
    assert config["plugins"]["load"]["paths"] == [str(plugin)]
    assert config["plugins"]["entries"]["guard"] == {
        "config": {"strict": True},
        "enabled": True,
    }
    assert config["tools"]["allow"] == ["exec"]


def test_harness_overlay_cannot_override_benchmark_keys(tmp_path: Path) -> None:
    harness = _harness_copy(tmp_path)
    (harness / "openclaw.overlay.json").write_text('{"tools": {"allow": ["exec", "read"]}}')
    with pytest.raises(openclaw_config.SetupError, match="invalid harness overlay"):
        openclaw_config.generate_setup(
            {**COMPAT_ENV, **HARNESS_ENV, "GAIA2_HARNESS_DIR": str(harness)},
            now=FIXED_NOW,
        )


def test_write_runtime_files_uses_staged_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "stage"
    (stage / "workspace").mkdir(parents=True)
    (stage / "workspace" / "SOUL.md").write_text("Be precise.\n")
    monkeypatch.setenv("GAIA2_HARNESS_STAGE_DIR", str(stage))
    home = tmp_path / "home"
    outputs = openclaw_config.generate_setup(COMPAT_ENV, now=FIXED_NOW)
    openclaw_config.write_runtime_files(home, outputs)
    assert (home / "SOUL.md").read_text() == "Be precise.\n"
    assert (home / "TOOLS.md").read_text() == ""
    assert (home / "HEARTBEAT.md").read_text() == openclaw_config.HEARTBEAT_CONTENT
