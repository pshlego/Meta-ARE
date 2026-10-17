# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Runner-side access to the external OpenClaw harness tooling.

The implementation lives in ``gaia2-cli/shared/harness_tool.py`` (also baked
into the OpenClaw image) so the runner, the container and AutoSaddler apply
identical digest and validation rules.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_GAIA2_CLI_ROOT = Path(__file__).resolve().parents[2]
HARNESS_TOOL_PATH = _GAIA2_CLI_ROOT / "shared" / "harness_tool.py"
HARNESS_POLICY_PATH = _GAIA2_CLI_ROOT / "harness" / "policy" / "openclaw_overlay_policy.json"
CONTAINER_HARNESS_DIR = "/opt/gaia2_harness"
CONTAINER_HARNESS_POLICY = "/opt/gaia2_harness_policy.json"


def harness_tool() -> ModuleType:
    name = "gaia2_shared_harness_tool"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HARNESS_TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {HARNESS_TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_and_digest(harness_dir: str | Path, policy: str | Path | None = None) -> str:
    """Validate a harness directory against ``policy`` (default: the repository policy); return its digest."""
    tool = harness_tool()
    tool.check(Path(harness_dir), tool.load_policy(Path(policy) if policy else HARNESS_POLICY_PATH))
    return tool.digest(Path(harness_dir))


def policy_digest(policy: str | Path) -> str:
    return "sha256:" + hashlib.sha256(Path(policy).read_bytes()).hexdigest()
