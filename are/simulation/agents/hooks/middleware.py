# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Hook middleware: intercepts tool calls and evaluates PreToolUse hooks."""

import logging

from are.simulation.agents.hooks.config import HookConfig
from are.simulation.agents.hooks.types import HookDecision, PreToolUseHookInput
from are.simulation.agents.llm.llm_engine import LLMEngine

logger = logging.getLogger(__name__)


class HookMiddleware:
    """Middleware that intercepts tool calls and evaluates PreToolUse hooks."""

    def __init__(
        self,
        config: HookConfig,
        llm_engine: LLMEngine,
        workspace_path: str | None = None,
    ):
        self.config = config
        self.llm_engine = llm_engine
        # workspace_path is used as cwd for agent-type hooks (GC-SDK sessions)
        self.workspace_path = workspace_path

    def evaluate_pre_tool_use(self, hook_input: PreToolUseHookInput) -> HookDecision:
        """
        Evaluate all matching PreToolUse hooks for a given tool call.

        Priority rule: deny > allow. If ANY hook returns deny, the final
        decision is deny.
        """
        if not self.config.has_pre_tool_use_hooks:
            return HookDecision(permission_decision="allow")

        final_decision = HookDecision(permission_decision="allow")

        for matcher_group in self.config.pre_tool_use:
            if not matcher_group.matches(hook_input.tool_name):
                continue

            logger.info(
                f"[Hook] Matched PreToolUse hook '{matcher_group.description or matcher_group.matcher}' "
                f"for tool '{hook_input.tool_name}'"
            )

            for handler in matcher_group.hooks:
                if handler.type not in ("prompt", "agent"):
                    continue

                # Render prompt template with runtime values
                rendered_prompt = hook_input.render_template(handler.prompt)

                # Log the rendered prompt
                logger.info(
                    f"[Hook] Rendered prompt for tool '{hook_input.tool_name}' "
                    f"(type={handler.type}):\n"
                    f"{'=' * 60}\n{rendered_prompt}\n{'=' * 60}"
                )

                # Dispatch to the appropriate evaluator
                if handler.type == "prompt":
                    logger.info(
                        f"[Hook] Sending prompt to LLM for evaluation "
                        f"(tool={hook_input.tool_name})"
                    )
                    raw_response = self._call_llm(rendered_prompt)
                elif handler.type == "agent":
                    logger.info(
                        f"[Hook] Spawning GC-SDK agent session for evaluation "
                        f"(tool={hook_input.tool_name})"
                    )
                    raw_response = self._call_gc_sdk_agent(
                        rendered_prompt, timeout=handler.timeout
                    )

                decision = HookDecision.from_llm_response(raw_response)

                # Log the raw response
                logger.info(
                    f"[Hook] Raw {handler.type} response for tool "
                    f"'{hook_input.tool_name}':\n"
                    f"{'=' * 60}\n{raw_response}\n{'=' * 60}"
                )

                logger.info(
                    f"[Hook] LLM decision: {decision.permission_decision}"
                    + (
                        f" — reason: {decision.permission_decision_reason[:200]}"
                        if decision.is_denied
                        else ""
                    )
                )

                # Priority: deny > allow
                if decision.is_denied:
                    final_decision = decision
                    break

            if final_decision.is_denied:
                break

        return final_decision

    def _call_llm(self, prompt: str) -> str:
        """Call the same LLM backbone used by the agent."""
        messages = [
            {"role": "user", "content": prompt},
        ]
        try:
            response_text, _metadata = self.llm_engine(messages, stop_sequences=[])
            return response_text
        except Exception as e:
            logger.error(f"[Hook] LLM call failed: {e}. Defaulting to allow.")
            return '{"decision": "allow"}'

    def _call_gc_sdk_agent(self, prompt: str, timeout: int = 7200) -> str:
        """Spawn a GC-SDK agentic session for deep validation with tool access."""
        from are.simulation.agents.hooks.gc_sdk_client import run_gc_sdk_agent

        cwd = self.workspace_path or "."
        try:
            response_text = run_gc_sdk_agent(
                cwd=cwd,
                prompt=prompt,
                timeout=float(timeout),
            )
            return response_text
        except Exception as e:
            logger.error(
                f"[Hook] GC-SDK agent call failed: {e}. Defaulting to allow."
            )
            return '{"decision": "allow"}'
