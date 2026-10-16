# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import json
import shutil
from pathlib import Path

import click
import pytest
from gaia2_runner import cli as runner_cli
from gaia2_runner.config import load_runner_toml_config
from gaia2_runner.harness import (
    HARNESS_POLICY_PATH,
    check_and_digest,
    harness_tool,
)

GAIA2_CLI = Path(__file__).resolve().parents[2]
SEED = GAIA2_CLI / "harness" / "openclaw"


@pytest.fixture
def harness(tmp_path: Path) -> Path:
    target = tmp_path / "openclaw"
    shutil.copytree(SEED, target)
    return target


def _check(path: Path, phase: str | None = None) -> None:
    tool = harness_tool()
    tool.check(path, tool.load_policy(HARNESS_POLICY_PATH), phase=phase)


def test_seed_harness_matches_official_defaults() -> None:
    assert (SEED / "AGENTS_TEMPLATE.md").read_bytes() == (
        GAIA2_CLI / "shared" / "AGENTS_TEMPLATE.md"
    ).read_bytes()
    assert json.loads((SEED / "openclaw.overlay.json").read_text()) == {}
    for name in ("SOUL.md", "TOOLS.md", "IDENTITY.md", "USER.md"):
        assert (SEED / "workspace" / name).read_text() == ""
    _check(SEED)
    _check(SEED, phase="steering")


def test_digest_is_content_addressed(harness: Path) -> None:
    before = check_and_digest(harness)
    assert before == check_and_digest(SEED)
    (harness / "workspace" / "SOUL.md").write_text("Be careful.\n")
    assert check_and_digest(harness) != before


@pytest.mark.parametrize(
    "overlay",
    [
        {"tools": {"allow": ["exec", "read"]}},
        {"models": {"providers": {}}},
        {"agents": {"defaults": {"model": {"primary": "x/y"}}}},
        {"agents": {"defaults": {"thinkingDefault": "high"}}},
        {"messages": {"queue": {"mode": "bogus"}}},
        {"tools": {"exec": {"timeoutSec": 100000}}},
        {"plugins": {"load": {"paths": ["/tmp"]}}},
        {"plugins": {"entries": {"not-a-harness-plugin": {"config": {}}}}},
        {"agents": {"defaults": {"timeoutSeconds": None}}},
    ],
)
def test_overlay_policy_rejects_benchmark_owned_or_invalid_keys(
    harness: Path, overlay: dict
) -> None:
    (harness / "openclaw.overlay.json").write_text(json.dumps(overlay))
    with pytest.raises(ValueError):
        _check(harness)


def test_overlay_policy_accepts_behavior_knobs(harness: Path) -> None:
    (harness / "openclaw.overlay.json").write_text(
        json.dumps(
            {
                "agents": {
                    "defaults": {
                        "timeoutSeconds": 1800,
                        "compaction": {"mode": "safeguard", "memoryFlush": {"enabled": False}},
                    }
                },
                "messages": {"queue": {"mode": "followup"}},
                "tools": {"exec": {"timeoutSec": 120}, "loopDetection": {"enabled": True}},
                "skills": {"allowBundled": [], "entries": {"weather": {"enabled": False}}},
            }
        )
    )
    _check(harness)


def _add_plugin(harness: Path, plugin_id: str = "guard") -> Path:
    plugin = harness / "plugins" / plugin_id
    plugin.mkdir(parents=True)
    (plugin / "openclaw.plugin.json").write_text(
        json.dumps({"id": plugin_id, "configSchema": {"type": "object"}})
    )
    (plugin / "index.js").write_text("export default { id: 'guard', register() {} };\n")
    return plugin


def _add_skill(harness: Path, name: str = "calendar-care", scripts: bool = False) -> None:
    skill = harness / "skills" / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Check calendar conflicts first.\n---\n\nBody\n"
    )
    if scripts:
        (skill / "scripts").mkdir()
        (skill / "scripts" / "helper.sh").write_text("echo hi\n")


def test_skills_and_plugins_are_validated(harness: Path) -> None:
    _add_skill(harness)
    _add_plugin(harness)
    (harness / "openclaw.overlay.json").write_text(
        json.dumps({"plugins": {"entries": {"guard": {"config": {"strict": True}}}}})
    )
    _check(harness)

    bad = harness / "skills" / "broken"
    bad.mkdir()
    (bad / "SKILL.md").write_text("no frontmatter\n")
    with pytest.raises(ValueError, match="frontmatter"):
        _check(harness)


def test_plugin_manifest_id_must_match_directory(harness: Path) -> None:
    plugin = _add_plugin(harness)
    (plugin / "openclaw.plugin.json").write_text(json.dumps({"id": "other", "configSchema": {}}))
    with pytest.raises(ValueError, match="id must be"):
        _check(harness)


def test_steering_phase_allows_markdown_only(harness: Path) -> None:
    _add_skill(harness)
    _check(harness, phase="steering")
    _add_plugin(harness)
    with pytest.raises(ValueError, match="Markdown only"):
        _check(harness, phase="steering")


def test_layout_and_placeholder_rules(harness: Path) -> None:
    (harness / "notes.txt").write_text("x")
    with pytest.raises(ValueError, match="outside the harness layout"):
        _check(harness)
    (harness / "notes.txt").unlink()

    template = harness / "AGENTS_TEMPLATE.md"
    template.write_text(template.read_text().replace("{{TOOL_LIST}}", ""))
    with pytest.raises(ValueError, match="TOOL_LIST"):
        _check(harness)


def test_build_execution_config_mounts_harness(
    harness: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, *_ = runner_cli._build_execution_config(
        image="localhost/gaia2-oc:latest",
        runtime="docker",
        timeout=600,
        health_timeout=60,
        provider="openai-compat",
        model="m",
        api_key="k",
        base_url="http://127.0.0.1:1/v1",
        thinking="high",
        judge_model="j",
        judge_provider="openai-compat",
        judge_base_url="http://127.0.0.1:1/v1",
        judge_api_key="k",
        volumes=(),
        notification_mode="message",
        time_speed=None,
        judge_time_check="off",
        max_scenario_duration=7200,
        harness_dir=str(harness),
    )
    digest = check_and_digest(harness)
    assert f"{harness.resolve()}:/opt/gaia2_harness:ro" in config.extra_volumes
    assert config.container_env["GAIA2_HARNESS_DIR"] == "/opt/gaia2_harness"
    assert config.container_env["GAIA2_HARNESS_DIGEST"] == digest
    assert config.container_env["GAIA2_JUDGE_TIME_CHECK"] == "off"
    assert config.result_metadata["harness_digest"] == digest
    assert config.max_scenario_duration == 7200


def test_build_execution_config_rejects_invalid_harness(harness: Path) -> None:
    (harness / "openclaw.overlay.json").write_text('{"tools": {"allow": ["read"]}}')
    with pytest.raises(click.UsageError, match="Invalid harness"):
        runner_cli._build_execution_config(
            image="localhost/gaia2-oc:latest",
            runtime="docker",
            timeout=600,
            health_timeout=60,
            provider="openai-compat",
            model="m",
            api_key="k",
            base_url="http://127.0.0.1:1/v1",
            thinking="high",
            judge_model="j",
            judge_provider="openai-compat",
            judge_base_url=None,
            judge_api_key="k",
            volumes=(),
            notification_mode="message",
            time_speed=None,
            harness_dir=str(harness),
        )


def test_toml_reads_harness_time_check_and_duration(tmp_path: Path, harness: Path) -> None:
    (tmp_path / "scenario.json").write_text(
        json.dumps({"metadata": {"definition": {"scenario_id": "s"}}, "events": []})
    )
    path = tmp_path / "eval.toml"
    path.write_text(
        f"""
[target]
scenario = "scenario.json"

[agent]
image = "localhost/gaia2-oc:latest"
provider = "openai-compat"
model = "m"
harness_dir = "{harness}"

[judge]
provider = "openai-compat"
model = "j"
time_check = "off"

[run]
timeout = 1800
max_scenario_duration = 7200
"""
    )
    config = load_runner_toml_config(str(path))
    assert config.agent.harness_dir == str(harness.resolve())
    assert config.judge.time_check == "off"
    assert config.run.max_scenario_duration == 7200
