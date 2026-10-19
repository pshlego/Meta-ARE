"""EnvHarness designer for GAIA2/OpenClaw: optimizer LLM client + the original LLMHarnessAgent."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Iterable

from envharness.agents.harness_agent import LLMHarnessAgent
from envharness.infra.llm import ChatResponse, LLMClient, LoggingLLMClient, ToolCall

from envharness_gaia2 import PROMPTS_DIR


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


def make_client(optimizer, *, log_path, role: str, reasoning_effort: str | None = None) -> LLMClient:
    """Optimizer LLM client with every call JSONL-logged (envharness LoggingLLMClient)."""
    effort = reasoning_effort or optimizer.reasoning_effort
    if optimizer.wire_api == "responses":
        factory = "envharness_gaia2.designer:ResponsesClient"
        kwargs = {"model": optimizer.model, "api_base": optimizer.api_base, "reasoning_effort": effort,
                  "usage_log": str(log_path.with_name("optimizer_usage.jsonl"))}
    else:
        factory = "envharness.infra.llm:LiteLLMClient"
        kwargs = {"model": f"openai/{optimizer.model}", "api_base": optimizer.api_base, "api_key": "dummy"}
        if effort:
            kwargs["reasoning_effort"] = effort
    return ThreadSafeLoggingLLMClient(
        inner_factory=factory, inner_kwargs=kwargs, log_path=str(log_path), role=role,
        truncate_messages_to_chars=None,
    )


class ThreadSafeLoggingLLMClient(LoggingLLMClient):
    """LoggingLLMClient (same JSONL record) whose appends are serialized across threads.

    Bank induction runs pairs in a thread pool; the upstream client appends
    each (often >PIPE_BUF) record without a lock, so records could interleave.
    """

    _lock = threading.Lock()

    def chat(self, messages, tools=None, tool_choice="auto", temperature=0.7, max_tokens=None, **kwargs):
        started = time.time()
        error, exc = None, None
        try:
            resp = self.inner.chat(messages, tools=tools, tool_choice=tool_choice, temperature=temperature,
                                   max_tokens=max_tokens, **kwargs)
        except Exception as e:  # noqa: BLE001 - logged, then re-raised
            exc, error = e, f"{type(e).__name__}: {e}"
            resp = ChatResponse(content="", tool_calls=[])
        record = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(),
            "role": self.role,
            "model": self.model_id,
            "duration_ms": int((time.time() - started) * 1000),
            "messages": [self._serialize_message(m) for m in messages],
            "tools": tools,
            "tool_choice": tool_choice,
            "response": {
                "content": resp.content,
                "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in resp.tool_calls],
            },
            "error": error,
        }
        line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
        with self._lock, open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line)
        if exc is not None:
            raise exc
        return resp


class Gaia2HarnessAgent(LLMHarnessAgent):
    """The original LLMHarnessAgent (``_DEFAULT_SYSTEM`` + the benchmark's
    ``extra_instructions``, as every envharness corpus config does), with one
    rendering fix: the upstream step renderer drops the command of pydantic
    ``Action`` steps (``harness_agent.py:858``). SWE-bench still shows commands
    because its observations start with ``$ <cmd>``; GAIA2 observations do not,
    so the command is handed to the upstream renderer in the dict form it reads.
    Everything else (format, limits, truncation) is the upstream renderer's.
    """

    def __init__(self, client: LLMClient, *, k: int, band: tuple[float, float]) -> None:
        lo, hi = band
        extra = (PROMPTS_DIR / "designer_extra.md").read_text(encoding="utf-8")
        for key, value in {"k": k, "lo": lo, "hi": hi}.items():
            extra = extra.replace("{" + key + "}", str(value))
        super().__init__(client, extra_instructions=extra)

    @staticmethod
    def _render_steps_for_mutator(steps: list, **kwargs) -> str:
        """Upstream renderer over steps whose command it can read.

        The student's final reply is its last action, shown like a command.
        Reasoning is passed through unchanged: upstream shows only ``<think>`` blocks, so a
        function-calling student's plain-text reasoning is not shown, as in the original.
        """
        shown = []
        for step in steps:
            action = getattr(step, "raw_action", None)
            action_kwargs = (action.get("kwargs") if isinstance(action, dict) else getattr(action, "kwargs", None)) or {}
            command = str(action_kwargs.get("command") or action_kwargs.get("text") or "")
            reasoning = (getattr(step, "policy_raw_response", None) or "").strip()
            shown.append(SimpleNamespace(
                raw_action={"name": getattr(action, "name", ""), "kwargs": {"command": command}},
                policy_raw_response=reasoning or None,
                filtered_observation=getattr(step, "filtered_observation", None),
                raw_observation=getattr(step, "raw_observation", None),
                info=getattr(step, "info", None) or {},
            ))
        return LLMHarnessAgent._render_steps_for_mutator(shown, **kwargs)
