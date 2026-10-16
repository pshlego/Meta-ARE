"""EnvHarness designer for GAIA2/OpenClaw: LLM client + LLMHarnessAgent subclass."""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Iterable

from envharness.agents.harness_agent import (
    PROPOSE_TOOL,
    HarnessAgentContext,
    LLMHarnessAgent,
    _candidate_from_args,
)
from envharness.core.types import Candidate, Trace
from envharness.infra.llm import ChatResponse, LLMClient, Message, ToolCall, _message_to_dict

from envharness_gaia2 import PROMPTS_DIR
from envharness_gaia2.rules_check import RulesCheckError, check_rules_code


class ResponsesClient(LLMClient):
    """``LLMClient`` over the OpenAI Responses API.

    The Copilot endpoint used by the optimizer serves gpt-5.5 only through
    ``/responses`` (``/chat/completions`` returns ``unsupported_api_for_model``),
    so chat-style messages and function tools are translated here.
    """

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str = "dummy",
        reasoning_effort: str | None = None,
        max_retries: int = 6,
        retry_initial_delay: float = 2.0,
        timeout: float = 1800.0,
        usage_log: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.usage_log = usage_log

        self.model_id = model
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.retry_initial_delay = retry_initial_delay
        self._client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)

    def chat(self, messages, tools=None, tool_choice="auto", temperature=0.7, max_tokens=None, **kwargs):
        del temperature  # reasoning models reject sampling knobs
        instructions = "\n\n".join(m.content for m in messages if m.role == "system")
        input_items: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                input_items.append(
                    {"type": "function_call_output", "call_id": message.tool_call_id or "", "output": message.content}
                )
                continue
            if message.content:
                input_items.append({"role": message.role, "content": message.content})
            for tool_call in message.tool_calls or []:
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments),
                    }
                )
        request: dict[str, Any] = {"model": self.model_id, "input": input_items}
        if instructions:
            request["instructions"] = instructions
        if tools:
            request["tools"] = [_responses_tool(tool) for tool in tools]
            request["tool_choice"] = _responses_tool_choice(tool_choice)
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}
        if max_tokens:
            request["max_output_tokens"] = int(max_tokens)

        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.responses.create(**request)
                break
            except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
                if attempt == self.max_retries:
                    raise
                self._sleep(attempt, exc)
            except APIStatusError as exc:
                if exc.status_code < 500 or attempt == self.max_retries:
                    raise
                self._sleep(attempt, exc)
        if self.usage_log and getattr(response, "usage", None) is not None:
            usage = response.usage
            record = {
                "ts": time.time(),
                "model": self.model_id,
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "reasoning_tokens": getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0),
            }
            with open(self.usage_log, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        calls = []
        for item in response.output:
            if getattr(item, "type", "") == "function_call":
                try:
                    arguments = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {"_raw": item.arguments}
                calls.append(ToolCall(id=item.call_id, name=item.name, arguments=arguments))
        return ChatResponse(content=response.output_text or "", tool_calls=calls, raw=response)

    def _sleep(self, attempt: int, exc: Exception) -> None:
        delay = min(60.0, self.retry_initial_delay * (2**attempt))
        print(f"[ResponsesClient retry {attempt + 1}] {type(exc).__name__}: sleeping {delay:.0f}s", file=sys.stderr)
        time.sleep(delay)


def _responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function", tool)
    return {
        "type": "function",
        "name": function["name"],
        "description": function.get("description", ""),
        "parameters": function.get("parameters", {"type": "object"}),
        "strict": False,
    }


def _responses_tool_choice(choice: Any) -> Any:
    if isinstance(choice, dict) and "function" in choice:
        return {"type": "function", "name": choice["function"]["name"]}
    return choice


def make_client(optimizer, *, log_path, role: str) -> LLMClient:
    """Optimizer LLM client with every call JSONL-logged (envharness LoggingLLMClient)."""
    from envharness.infra.llm import LoggingLLMClient

    if optimizer.wire_api == "responses":
        factory = "envharness_gaia2.designer:ResponsesClient"
        kwargs = {"model": optimizer.model, "api_base": optimizer.api_base, "reasoning_effort": optimizer.reasoning_effort,
                  "usage_log": str(log_path.with_name("optimizer_usage.jsonl"))}
    else:
        factory = "envharness.infra.llm:LiteLLMClient"
        kwargs = {"model": f"openai/{optimizer.model}", "api_base": optimizer.api_base, "api_key": "dummy"}
        if optimizer.reasoning_effort:
            kwargs["reasoning_effort"] = optimizer.reasoning_effort
    return LoggingLLMClient(
        inner_factory=factory, inner_kwargs=kwargs, log_path=str(log_path), role=role,
        truncate_messages_to_chars=None,
    )


class RulesRejected(RuntimeError):
    """The designer could not produce valid rules within the fix-up budget."""


class Gaia2HarnessAgent(LLMHarnessAgent):
    """LLMHarnessAgent with a GAIA2 system prompt, band skip and validated rules."""

    def __init__(
        self,
        client: LLMClient,
        *,
        k: int,
        band: tuple[float, float],
        forbidden_ids: Iterable[str] = (),
        max_fix_attempts: int = 2,
    ) -> None:
        lo, hi = band
        extra = (PROMPTS_DIR / "designer_extra.md").read_text(encoding="utf-8").format(k=k, lo=lo, hi=hi)
        super().__init__(
            client,
            system_prompt=(PROMPTS_DIR / "designer_system.md").read_text(encoding="utf-8"),
            extra_instructions=extra,
        )
        self.band = band
        self.forbidden_ids = tuple(forbidden_ids)
        self.max_fix_attempts = max_fix_attempts

    def propose(self, ctx: HarnessAgentContext) -> Candidate:
        lo, hi = self.band
        if ctx.baseline is not None and lo <= ctx.baseline.sr <= hi:
            return Candidate(rationale=f"skip-in-band (baseline sr={ctx.baseline.sr:.2f})")
        return self._validated(super().propose(ctx), ctx, intro="Propose a Candidate now.")

    def refine(self, candidate: Candidate, traces: list[Trace], ctx: HarnessAgentContext) -> Candidate:
        refined = self._validated(
            super().refine(candidate, traces, ctx), ctx, intro="Refine the previous candidate based on the K rollouts."
        )
        if not refined.rules_code.strip():
            # An empty refinement would only replay the baseline; end the task.
            raise RulesRejected(f"refinement produced no rules ({refined.rationale[:120]})")
        return refined

    def _validated(self, candidate: Candidate, ctx: HarnessAgentContext, *, intro: str) -> Candidate:
        for attempt in range(self.max_fix_attempts + 1):
            problem = self._problem(candidate)
            if problem is None:
                return candidate
            if attempt == self.max_fix_attempts:
                break
            messages = self._base_messages(ctx, intro=intro)
            messages.append(
                Message(
                    role="user",
                    content=(
                        "Your previous candidate was rejected by the static checker before any rollout:\n"
                        f"{problem}\n\nPrevious rules_code:\n```python\n{candidate.rules_code}\n```\n"
                        "Fix it (or skip with an empty rules_code). Same propose_candidate schema."
                    ),
                )
            )
            response = self.client.chat(
                messages, tools=[PROPOSE_TOOL],
                tool_choice={"type": "function", "function": {"name": "propose_candidate"}},
            )
            if not response.tool_calls:
                break
            candidate = _candidate_from_args(response.tool_calls[0].arguments)
        return Candidate(rationale=f"skip-invalid-rules: {self._problem(candidate)}")

    def _problem(self, candidate: Candidate) -> str | None:
        if candidate.in_env_actions:
            return "in_env_actions (Setup) are disabled in this experiment; leave them empty."
        try:
            check_rules_code(candidate.rules_code, forbidden_ids=self.forbidden_ids)
        except RulesCheckError as exc:
            return str(exc)
        return None

    @staticmethod
    def _render_steps_for_mutator(
        steps: list,
        max_steps: int = 12,
        max_think_chars: int = 600,
        max_obs_chars: int = 600,
        max_cmd_chars: int = 300,
    ) -> str:
        """Fixes the upstream renderer, which drops pydantic ``Action`` commands."""
        shown = steps[-max_steps:]
        start = len(steps) - len(shown)
        out = []
        if start:
            out.append(f"    [showing last {len(shown)} of {len(steps)} steps; first {start} omitted]")
        for offset, step in enumerate(shown, start=start + 1):
            action = getattr(step, "raw_action", None)
            kwargs = (action.get("kwargs") if isinstance(action, dict) else getattr(action, "kwargs", None)) or {}
            command = str(kwargs.get("command") or kwargs.get("text") or "")[:max_cmd_chars]
            observation = getattr(step, "filtered_observation", None) or getattr(step, "raw_observation", None)
            text = (observation.get("text", "") if isinstance(observation, dict) else getattr(observation, "text", "")) or ""
            reasoning = (getattr(step, "policy_raw_response", None) or "").strip()
            block = [f"    Step {offset}:"]
            if reasoning:
                block.append(f"      <reasoning>{reasoning[:max_think_chars]}</reasoning>")
            block.append(f"      <cmd>{command}</cmd>")
            if getattr(step, "blocked_reason", None):
                block.append(f"      BLOCKED: {step.blocked_reason}")
            block.append(f"      <out>{text.strip()[:max_obs_chars]}</out>")
            out.append("\n".join(block))
        return "\n\n".join(out)
