from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from envharness.agents.harness_agent import HarnessAgentContext
from envharness.core.types import Action, BaselineSnapshot, Candidate, Observation, Step
from envharness.infra.llm import ChatResponse, Message, ScriptedClient, ToolCall

from envharness_gaia2.designer import Gaia2HarnessAgent, ResponsesClient, RulesRejected
from envharness_gaia2.orchestrator import ENV_STATE_SCHEMA, EXEC_TOOL_SCHEMA

GOOD = "class _Rules(Rules):\n    def filter_action(self, action, env_state):\n        return action\n"
BAD = "import os\nclass _Rules(Rules):\n    def filter_action(self, action, env_state):\n        return action\n"


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


def _propose(code: str) -> ChatResponse:
    return ChatResponse(content="", tool_calls=[ToolCall(id="t", name="propose_candidate",
                                                         arguments={"rationale": "why", "rules_code": code})])


def test_band_skip_and_rules_fixup():
    calls = []

    def responder(messages, tools):
        calls.append(messages)
        return _propose(BAD if len(calls) == 1 else GOOD)

    agent = Gaia2HarnessAgent(ScriptedClient(responder=responder), k=3, band=(0.3, 0.7))
    assert agent.propose(_ctx(0.67)).rationale.startswith("skip-in-band")
    assert calls == []
    candidate = agent.propose(_ctx(0.0))
    assert candidate.rules_code == GOOD and len(calls) == 2
    assert "rejected by the static checker" in calls[1][-1].content
    assert "GAIA2" in calls[0][0].content and "experiment-specific" in calls[0][0].content


def test_invalid_rules_after_budget_become_skip_and_refine_raises():
    agent = Gaia2HarnessAgent(ScriptedClient(responder=lambda m, t: _propose(BAD)), k=3, band=(0.3, 0.7))
    candidate = agent.propose(_ctx(1.0))
    assert candidate.rules_code == "" and candidate.rationale.startswith("skip-invalid-rules")
    with pytest.raises(RulesRejected):
        agent.refine(Candidate(rules_code=GOOD), [], _ctx(1.0))


def test_in_env_actions_are_rejected():
    agent = Gaia2HarnessAgent(ScriptedClient(responder=lambda m, t: _propose(GOOD)), k=3, band=(0.3, 0.7))
    assert agent._problem(Candidate(rules_code=GOOD, in_env_actions=[Action(name="x")])) is not None


def test_step_renderer_shows_commands():
    step = Step(raw_action=Action(name="exec", kwargs={"command": "contacts get-contacts --offset 0"}),
                blocked_reason="Error: temporarily unavailable",
                filtered_observation=Observation(text="[]"), policy_raw_response="look up contacts")
    text = Gaia2HarnessAgent._render_steps_for_mutator([step])
    assert "<cmd>contacts get-contacts --offset 0</cmd>" in text
    assert "BLOCKED: Error: temporarily unavailable" in text and "look up contacts" in text
