# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Hook configuration: loading and parsing hook.json files."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class HookHandler:
    """
    Level 3: Single hook handler definition (from hook.json).

    Supported types:
    - "prompt": Single LLM call with a rendered prompt template
    - "agent": GC-SDK agentic session with tool access for deep validation
    - "reminder": Static hint injected once per tool name (no LLM call)
    """

    type: Literal["prompt", "agent", "reminder"]
    prompt: str = ""
    reminder: str = ""
    timeout: int = 7200


@dataclass
class HookMatcherGroup:
    """Level 2: A matcher group that filters tool calls and triggers handlers."""

    matcher: str
    hooks: list[HookHandler]
    description: str = ""
    _compiled: re.Pattern = field(init=False, repr=False)  # type: ignore[type-arg]

    def __post_init__(self) -> None:
        self._compiled = re.compile(self.matcher)

    def matches(self, tool_name: str) -> bool:
        return bool(self._compiled.search(tool_name))


@dataclass
class HookConfig:
    """Level 1: Complete hook configuration loaded from hook.json."""

    pre_tool_use: list[HookMatcherGroup] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str) -> "HookConfig":
        with open(path) as f:
            data = json.load(f)

        config = cls()
        for matcher_group in data.get("hooks", {}).get("PreToolUse", []):
            handlers: list[HookHandler] = []
            for h in matcher_group.get("hooks", []):
                handler_type = h.get("type")
                if handler_type not in ("prompt", "agent", "reminder"):
                    logger.warning(
                        f"Skipping unsupported hook handler type: {handler_type!r}. "
                        f"Only 'prompt', 'agent', and 'reminder' types are supported."
                    )
                    continue
                if handler_type == "reminder":
                    reminder_text = h.get("reminder", "")
                    if not reminder_text:
                        logger.warning(
                            "Skipping reminder hook with empty 'reminder' field."
                        )
                        continue
                    handlers.append(
                        HookHandler(
                            type="reminder",
                            reminder=reminder_text,
                        )
                    )
                else:
                    handlers.append(
                        HookHandler(
                            type=handler_type,
                            prompt=h["prompt"],
                            timeout=h.get("timeout", 7200),
                        )
                    )
            if handlers:
                config.pre_tool_use.append(
                    HookMatcherGroup(
                        matcher=matcher_group["matcher"],
                        hooks=handlers,
                        description=matcher_group.get("description", ""),
                    )
                )
        return config

    @property
    def has_pre_tool_use_hooks(self) -> bool:
        return len(self.pre_tool_use) > 0
