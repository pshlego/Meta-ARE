# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import click
import pytest
from gaia2_runner import cli as runner_cli
from gaia2_runner.config import load_runner_toml_config
from gaia2_runner.launcher import LocalLauncher


def _build(**overrides):
    kwargs = dict(
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
    )
    kwargs.update(overrides)
    return runner_cli._build_execution_config(**kwargs)[0]


def test_env_rules_dir_is_mounted_and_pinned(tmp_path: Path) -> None:
    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / "rules.py").write_text("class _Rules(Rules):\n    pass\n")
    digest = "sha256:" + hashlib.sha256((rules / "rules.py").read_bytes()).hexdigest()
    config = _build(env_rules_dir=str(rules))
    assert f"{rules.resolve()}:/opt/gaia2_env_rules_src:ro" in config.extra_volumes
    assert config.container_env["GAIA2_ENV_RULES_DIR"] == "/opt/gaia2_env_rules_src"
    assert config.container_env["GAIA2_ENV_RULES_DIGEST"] == digest
    assert config.result_metadata["env_rules_digest"] == digest


def test_env_rules_dir_requires_rules_file_and_openclaw(tmp_path: Path) -> None:
    with pytest.raises(click.UsageError, match="no rules.py"):
        _build(env_rules_dir=str(tmp_path))
    (tmp_path / "rules.py").write_text("")
    with pytest.raises(click.UsageError, match="OpenClaw"):
        _build(image="localhost/gaia2-hermes:latest", env_rules_dir=str(tmp_path))


def test_without_env_rules_nothing_is_added() -> None:
    config = _build()
    assert "GAIA2_ENV_RULES_DIR" not in config.container_env
    assert "env_rules_digest" not in config.result_metadata


def test_toml_env_rules_dir(tmp_path: Path) -> None:
    (tmp_path / "scenario.json").write_text(
        json.dumps({"metadata": {"definition": {"scenario_id": "s"}}, "events": []})
    )
    (tmp_path / "rules").mkdir()
    path = tmp_path / "eval.toml"
    path.write_text(
        """
[target]
scenario = "scenario.json"
[agent]
image = "localhost/gaia2-oc:latest"
provider = "openai-compat"
model = "m"
env_rules_dir = "rules"
[judge]
provider = "openai-compat"
model = "j"
"""
    )
    assert load_runner_toml_config(str(path)).agent.env_rules_dir == str((tmp_path / "rules").resolve())


def test_container_labels_from_host_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []
    launcher = LocalLauncher(runtime="docker")
    monkeypatch.setattr(
        launcher, "_run", lambda args, **kw: calls.append(args) or type("R", (), {"stdout": "cid\n"})()
    )
    monkeypatch.setenv("GAIA2_CONTAINER_LABELS", "envh.run=r1,envh.batch=b2")
    launcher.launch("img", str(tmp_path / "s.json"), env={"GAIA2_CONTAINER_LABELS": "x=y", "A": "1"})
    argv = calls[0]
    assert argv.count("--label") == 2
    assert "envh.run=r1" in argv and "envh.batch=b2" in argv
    assert "GAIA2_CONTAINER_LABELS=x=y" not in argv and "A=1" in argv
