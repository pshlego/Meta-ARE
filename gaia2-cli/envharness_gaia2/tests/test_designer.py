from __future__ import annotations

import json
from types import SimpleNamespace

from envharness.agents.harness_agent import HarnessAgentContext
from envharness.core.types import Action, BaselineSnapshot, Observation, Step
from envharness.infra.llm import ChatResponse, Message, ScriptedClient, ToolCall

from envharness.agents.harness_agent import LLMHarnessAgent

from envharness_gaia2.designer import Gaia2HarnessAgent, ResponsesClient
from envharness_gaia2.orchestrator import ENV_STATE_SCHEMA, EXEC_TOOL_SCHEMA

GOOD = "class _Rules(Rules):\n    def filter_action(self, action, env_state):\n        return action\n"


def _ctx(sr: float) -> HarnessAgentContext:
    return HarnessAgentContext(
        tool_schemas=[EXEC_TOOL_SCHEMA], env_state_schema=ENV_STATE_SCHEMA, task_description="t", task_id=3,
        baseline=BaselineSnapshot(n=3, n_success=round(sr * 3), sr=sr),
    )


def test_responses_client_translates_messages_and_tools(monkeypatch):
    captured = {}

    def create(**request):
        captured.update(request)
        return SimpleNamespace(
            output=[SimpleNamespace(type="reasoning"),
                    SimpleNamespace(type="function_call", call_id="c1", name="propose_candidate",
                                    arguments=json.dumps({"rationale": "r", "rules_code": GOOD}))],
            output_text="",
        )

    client = ResponsesClient(model="gpt-5.5", api_base="http://127.0.0.1:1/v1", reasoning_effort="xhigh")
    monkeypatch.setattr(client._client.responses, "create", create)
    tool = {"type": "function", "function": {"name": "propose_candidate", "description": "d",
                                             "parameters": {"type": "object"}}}
    response = client.chat(
        [Message(role="system", content="SYS"), Message(role="user", content="hi")],
        tools=[tool], tool_choice={"type": "function", "function": {"name": "propose_candidate"}}, temperature=0.7,
    )
    assert captured["instructions"] == "SYS"
    assert captured["input"] == [{"role": "user", "content": "hi"}]
    assert captured["tools"][0]["name"] == "propose_candidate" and "function" not in captured["tools"][0]
    assert captured["tool_choice"] == {"type": "function", "name": "propose_candidate"}
    assert captured["reasoning"] == {"effort": "xhigh"} and "temperature" not in captured
    assert response.tool_calls[0].arguments["rules_code"] == GOOD


def test_agent_is_the_original_with_the_gaia2_block():
    from envharness.agents.harness_agent import _DEFAULT_SYSTEM

    agent = Gaia2HarnessAgent(ScriptedClient(responder=lambda m, t: None), k=5, band=(0.4, 0.6))
    assert agent.system_prompt.startswith(_DEFAULT_SYSTEM)
    extra = agent.system_prompt.split("--- experiment-specific instructions ---", 1)[1]
    assert "PHASE 1" in extra and "PHASE 4" in extra and "GRADING INTEGRITY" in extra
    assert "[0.4, 0.6]" in extra and "K=5" in extra and "{" + "lo}" not in extra
    assert type(agent).propose is LLMHarnessAgent.propose and type(agent).refine is LLMHarnessAgent.refine
    assert type(agent).decide is LLMHarnessAgent.decide


def test_logging_client_appends_whole_records_from_threads(tmp_path):
    import concurrent.futures as cf

    from envharness_gaia2.designer import ThreadSafeLoggingLLMClient

    client = ThreadSafeLoggingLLMClient(
        inner_factory="envharness.infra.llm:ScriptedClient",
        inner_kwargs={"script": [{"name": "noop", "kwargs": {"blob": "x" * 50_000}} for _ in range(16)]},
        log_path=str(tmp_path / "calls.jsonl"), role="induce", truncate_messages_to_chars=None,
    )
    with cf.ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: client.chat([Message(role="user", content="y" * 20_000)]), range(16)))
    lines = (tmp_path / "calls.jsonl").read_text().splitlines()
    assert len(lines) == 16 and all(json.loads(line)["role"] == "induce" for line in lines)


def test_step_renderer_shows_commands_in_the_upstream_format():
    step = Step(raw_action=Action(name="exec", kwargs={"command": "contacts get-contacts --offset 0"}),
                filtered_observation=Observation(text="x" * 1000), policy_raw_response="look up contacts")
    text = Gaia2HarnessAgent._render_steps_for_mutator([step])
    assert "<action>contacts get-contacts --offset 0</action>" in text
    assert "look up contacts" not in text  # upstream shows only <think> blocks (FC policies: none)
    assert "Observation: " + "x" * 400 + " ...[truncated]" in text  # upstream obs limit


def test_step_renderer_is_upstream_apart_from_the_command():
    # Steps in the dict form the upstream renderer reads commands from.
    steps = [SimpleNamespace(raw_action={"name": "exec", "kwargs": {"command": f"cabs list-rides --offset {i}"}},
                             filtered_observation=Observation(text=f"out {i}" * 99), raw_observation=None,
                             policy_raw_response=f"<think>t{i}</think>", info={}) for i in range(14)]
    assert Gaia2HarnessAgent._render_steps_for_mutator(steps) == LLMHarnessAgent._render_steps_for_mutator(steps)


def test_reply_step_renders_the_answer():
    step = Step(raw_action=Action(name="reply", kwargs={"text": "Booked the cab."}), filtered_observation=Observation(text=""))
    assert "<action>Booked the cab.</action>" in Gaia2HarnessAgent._render_steps_for_mutator([step])
