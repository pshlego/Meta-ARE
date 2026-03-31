# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Hook I/O types: PreToolUseHookInput and HookDecision."""

import json
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class PreToolUseHookInput:
    """Input assembled at the interception point and passed to the hook handler."""

    hook_event_name: str = "PreToolUse"
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    execution_trace: str = ""
    agent_thought: str = ""
    task_description: str = ""

    def render_template(self, template: str) -> str:
        """Substitute $VARIABLE placeholders with actual runtime values."""
        replacements = {
            "$TOOL_NAME": self.tool_name,
            "$TOOL_ARGUMENTS": json.dumps(self.tool_input, indent=2, default=str),
            "$EXECUTION_TRACE": self.execution_trace,
            "$AGENT_THOUGHT": self.agent_thought,
            "$TASK_DESCRIPTION": self.task_description,
        }
        result = template
        for var, val in replacements.items():
            result = result.replace(var, val)
        return result


@dataclass
class HookDecision:
    """
    Decision returned by a hook handler after LLM evaluation.

    Only "allow" and "deny" are supported. "ask" from Claude Code is treated
    as "deny" since Meta-ARE agents run autonomously.
    """

    permission_decision: Literal["allow", "deny"] = "allow"
    permission_decision_reason: str = ""
    system_message: str = ""
    additional_context: str = ""
    updated_input: Optional[dict] = None

    @classmethod
    def from_llm_response(cls, response_text: str) -> "HookDecision":
        """Parse LLM JSON response into HookDecision."""
        # Try to extract JSON from the response (LLM may wrap it in markdown)
        cleaned = response_text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1]
            cleaned = cleaned.split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1]
            cleaned = cleaned.split("```", 1)[0].strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            import re

            json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    logger.warning(
                        f"Hook LLM did not return valid JSON. Defaulting to allow. "
                        f"Response: {response_text[:200]}"
                    )
                    return cls(permission_decision="allow")
            else:
                logger.warning(
                    f"Hook LLM did not return valid JSON. Defaulting to allow. "
                    f"Response: {response_text[:200]}"
                )
                return cls(permission_decision="allow")

        decision = data.get("decision", "allow")
        feedback = data.get("feedback", "")

        if decision == "ask":
            logger.warning(
                "Hook LLM returned 'ask' decision, which is not supported. "
                "Treating as 'deny'."
            )
            decision = "deny"
            if not feedback:
                feedback = (
                    "Hook requested user clarification (ask), but Meta-ARE does not "
                    "support interactive approval. Defaulting to deny."
                )

        if decision == "deny":
            return cls(
                permission_decision="deny",
                permission_decision_reason=feedback,
                system_message=feedback,
            )
        return cls(permission_decision="allow")

    @property
    def is_denied(self) -> bool:
        return self.permission_decision == "deny"

    def to_hint_text(self) -> str:
        """Convert feedback to hint content for injection into the agent."""
        parts = []
        if self.system_message:
            parts.append(self.system_message)
        if self.additional_context:
            parts.append(self.additional_context)
        return "\n".join(parts)
