# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""EnvHarness Stage (Setup) for the Gaia2 app CLIs.

``gaia2-env-setup SETUP_JSON LOG_JSONL`` runs the designer's ``in_env_actions``
before the agent starts. Each action is one app CLI call (``{"command":
"<app> <subcommand> ..."}``), executed without a shell as an environment
event: ``GAIA2_EVENT_ID=Event-ENV-setup-<i>`` makes ``log_action`` tag it, the
daemon classifies ``Event-ENV-*`` ids as ENV, and the judge compares only
AGENT writes, so Setup reshapes the initial state without counting as the
agent's actions (env_rules also bypasses rules for such calls). Results go to
``LOG_JSONL``; a failing action is logged, not fatal. Stdlib-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time

APPS = (
    "calendar",
    "contacts",
    "emails",
    "messages",
    "chats",
    "rent-a-flat",
    "city",
    "cabs",
    "shopping",
    "cloud-drive",
)
MAX_ACTIONS = 20
_LOG_CHARS = 500


BUNDLE_FILES = ("rules.py", "setup.json")


def bundle_digest(directory: str) -> str:
    """Digest of an env-rules bundle (``rules.py`` and/or ``setup.json``).

    Mirrored by gaia2-runner and the envharness host so the container can
    verify it received exactly what was designed.
    """
    digest = hashlib.sha256()
    for name in BUNDLE_FILES:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                digest.update(f"{name}:{hashlib.sha256(handle.read()).hexdigest()}\n".encode())
    return "sha256:" + digest.hexdigest()


class SetupError(ValueError):
    """An action is not a single app CLI call."""


def parse_actions(payload: object) -> list[list[str]]:
    """Validate ``{"commands": [str, ...]}`` and return argv lists."""
    if not isinstance(payload, dict) or not isinstance(payload.get("commands"), list):
        raise SetupError('setup must be {"commands": [...]}')
    commands = payload["commands"]
    if len(commands) > MAX_ACTIONS:
        raise SetupError(f"at most {MAX_ACTIONS} setup actions")
    argvs = []
    for command in commands:
        if not isinstance(command, str):
            raise SetupError("each setup action must be a command string")
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
        lexer.whitespace_split = True
        argv = list(lexer)
        if not argv or argv[0] not in APPS:
            raise SetupError(f"setup action must start with an app CLI ({', '.join(APPS)}): {command!r}")
        if any(token and set(token) <= set(";&|<>()") for token in argv):
            raise SetupError(f"setup actions are single app calls, not shell pipelines: {command!r}")
        argvs.append(argv)
    return argvs


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: gaia2-env-setup SETUP_JSON LOG_JSONL", file=sys.stderr)
        return 2
    setup_path, log_path = args
    with open(setup_path, encoding="utf-8") as handle:
        actions = parse_actions(json.load(handle))
    with open(log_path, "a", encoding="utf-8") as log:
        for index, action in enumerate(actions):
            env = {**os.environ, "GAIA2_EVENT_ID": f"Event-ENV-setup-{index}"}
            try:
                done = subprocess.run(action, env=env, capture_output=True, text=True, timeout=30, check=False)
                record = {"exit_code": done.returncode, "stdout": done.stdout[:_LOG_CHARS],
                          "stderr": done.stderr[:_LOG_CHARS]}
            except (OSError, subprocess.TimeoutExpired) as exc:
                record = {"exit_code": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"[:_LOG_CHARS]}
            log.write(json.dumps({"t": time.time(), "index": index, "command": shlex.join(action), **record}) + "\n")
            log.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
