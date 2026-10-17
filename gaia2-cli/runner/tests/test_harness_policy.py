# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import click
import pytest
from gaia2_runner.config import load_runner_toml_config

from .test_env_rules_config import _build

GAIA2_CLI = Path(__file__).resolve().parents[2]
SEED = GAIA2_CLI / "harness" / "openclaw"


def _harness(tmp_path: Path, n_skills: int) -> Path:
    harness = tmp_path / "harness"
    shutil.copytree(SEED, harness)
    for i in range(n_skills):
        skill = harness / "skills" / f"skill-{i}"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"---\nname: skill-{i}\ndescription: \"d{i}\"\n---\n\n# S{i}\n\nbody\n")
    return harness


def _policy(tmp_path: Path, max_skills: int) -> Path:
    policy = json.loads((GAIA2_CLI / "harness" / "policy" / "openclaw_overlay_policy.json").read_text())
    policy["limits"]["max_skills"] = max_skills
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    return path


def test_default_policy_limits_skills(tmp_path: Path) -> None:
    with pytest.raises(click.UsageError, match="at most 16 skills"):
        _build(harness_dir=str(_harness(tmp_path, 17)))


def test_harness_policy_is_used_on_host_and_mounted_over_the_image_policy(tmp_path: Path) -> None:
    harness = _harness(tmp_path, 17)
    policy = _policy(tmp_path, 128)
    config = _build(harness_dir=str(harness), harness_policy=str(policy))
    assert f"{policy.resolve()}:/opt/gaia2_harness_policy.json:ro" in config.extra_volumes
    expected = "sha256:" + hashlib.sha256(policy.read_bytes()).hexdigest()
    assert config.result_metadata["harness_policy_digest"] == expected


def test_harness_policy_requires_harness_dir(tmp_path: Path) -> None:
    with pytest.raises(click.UsageError, match="requires"):
        _build(harness_policy=str(_policy(tmp_path, 128)))


def test_without_harness_policy_nothing_is_mounted(tmp_path: Path) -> None:
    config = _build(harness_dir=str(_harness(tmp_path, 1)))
    assert not any("gaia2_harness_policy" in volume for volume in config.extra_volumes)
    assert "harness_policy_digest" not in config.result_metadata


def test_toml_harness_policy(tmp_path: Path) -> None:
    (tmp_path / "scenario.json").write_text(
        json.dumps({"metadata": {"definition": {"scenario_id": "s"}}, "events": []})
    )
    _policy(tmp_path, 128)
    path = tmp_path / "eval.toml"
    path.write_text(
        """
[target]
scenario = "scenario.json"
[agent]
image = "localhost/gaia2-oc:latest"
provider = "openai-compat"
model = "m"
harness_policy = "policy.json"
[judge]
provider = "openai-compat"
model = "j"
"""
    )
    assert load_runner_toml_config(str(path)).agent.harness_policy == str((tmp_path / "policy.json").resolve())
