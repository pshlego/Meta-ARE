from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest
from envharness.core.types import Candidate

from envharness_gaia2.gaia2_run import disposition_of
from envharness_gaia2.trace_convert import FORBIDDEN_FILES, convert, failure_category, user_request

from helpers import FIXTURES, write_scenario_output


def _convert(path: Path):
    return convert(path, task_idx=4, task_id="gaia2", iteration_id="it", candidate=Candidate(), capability="search")


def test_convert_real_fixture_without_opening_oracle_files(monkeypatch):
    opened = []
    real_open = builtins.open

    def spy(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (opened.append(str(self)), real_open(self, encoding="utf-8").read())[1])
    trace = _convert(FIXTURES / "scenario_universe_29_0wlkxc")
    assert not [p for p in opened if Path(p).name in FORBIDDEN_FILES]
    assert trace.success is True and trace.rollout_seed == 4
    assert trace.duration_steps == len(trace.steps) > 5
    assert trace.steps[-1].raw_action.name == "reply" and trace.steps[-1].raw_action.kwargs["text"]
    assert trace.steps[0].info["task"].startswith("Find the person with whom I have the most conversations")
    first = trace.steps[0]
    assert first.raw_action.name == "exec" and first.raw_action.kwargs["command"].startswith("chats --help")
    assert first.filtered_observation.text.startswith("Task: Find the person with whom I have the most conversations")
    assert "Sender (untrusted metadata)" not in first.filtered_observation.text
    assert first.blocked_reason is None
    blocked = [s for s in trace.steps if s.blocked_reason]
    assert len(blocked) == 1 and blocked[0].filtered_action is None
    assert blocked[0].raw_action.kwargs["command"].startswith("contacts get-contacts")


def test_failure_reasons_are_reduced_to_categories(tmp_path):
    path = write_scenario_output(tmp_path / "scenario_universe_29_x", success=False)
    trace = _convert(path)
    assert trace.success is False and trace.error is None
    dumped = trace.model_dump_json()
    assert "oracle" not in dumped and "ORACLE-SECRET" not in dumped
    assert trace.steps[0].info["failure_category"] == "tool_count_mismatch"
    assert failure_category({"success": False, "failure_reasons": ["Tool count mismatch: agent={}, oracle={'a': 1}"]}) == "no_tool_calls"
    assert failure_category({"success": None, "timed_out": True}) == "timeout"
    assert user_request("Sender (untrusted metadata):\n```json\n{}\n```\n\n[Tue 2024-10-15 07:00 UTC] Do it.") == "Do it."


def test_disposition_requires_matching_digests():
    ok = {"success": True, "harness_digest": "h"}
    assert disposition_of(ok, [200], {"digest": "h"}, "h", None)[0] == "success"
    assert disposition_of(ok, [200], {"digest": "x"}, "h", None)[0] == "execution_error"
    assert disposition_of({**ok, "env_rules_digest": "r"}, [200], {"digest": "h"}, "h", "r")[0] == "success"
    assert disposition_of(ok, [200], {"digest": "h"}, "h", "r")[0] == "execution_error"
    stalled = {"success": None, "harness_digest": "h", "error": "No judgment (no turn boundary detected)"}
    assert disposition_of(stalled, [200], {"digest": "h"}, "h", None)[0] == "task_failure"
    assert disposition_of(stalled, [], {"digest": "h"}, "h", None)[0] == "execution_error"
    # stalled because the LLM endpoint failed mid-rollout (envh3n r1/task_003): infrastructure
    outage = disposition_of(stalled, [200, 200, 502, 502], {"digest": "h"}, "h", None)
    assert outage[0] == "execution_error" and "HTTP 502" in outage[1]
    assert disposition_of(stalled, [502, 502, 200, None], {"digest": "h"}, "h", None)[0] == "task_failure"


def test_interventions_attach_by_exact_argv_and_rules_errors_mark_the_trace(tmp_path):
    commands = ("contacts get-contacts --help && contacts get-contacts --offset 0",
                "cd /tmp; contacts get-contacts --offset 10 | head", "calendar get-events")
    path = write_scenario_output(tmp_path / "scenario_universe_29_x", success=True, commands=commands)
    applied = [
        {"app": "contacts", "subcommand": "get-contacts", "kind": "blocked", "argv": ["get-contacts", "--offset", "10"],
         "reason": "busy"},
        {"app": "contacts", "subcommand": "get-contacts", "kind": "observation_modified", "argv": ["get-contacts", "--offset", "0"],
         "wrote": False},
        {"app": "cabs", "subcommand": "order-ride", "kind": "blocked", "argv": ["order-ride"], "reason": "x"},
    ]
    (path / "env_rules_applied.jsonl").write_text("".join(json.dumps(e) + "\n" for e in applied))
    trace = _convert(path)
    assert [s.blocked_reason for s in trace.steps[:3]] == [None, "busy", None]
    assert trace.steps[0].info["interventions"][0]["kind"] == "observation_modified"
    assert trace.steps[0].info["unattributed_interventions"] == 1
    assert trace.error is None
    assert trace.success is True
    (path / "env_rules_errors.log").write_text(
        "[ts] contacts filter_observation raised; showing the unfiltered output\nTraceback (most recent call last):\n"
        '  File "<env_rules>", line 3, in filter_observation\nAttributeError: x has no attribute app\n\n'
        "[ts] calendar filter_action raised; running unchanged\nTraceback (most recent call last):\nKeyError: 'y'\n")
    # as envharness's runner: the episode fails with "env.step raised: <Type>: <msg>" (the first one)
    errored = _convert(path)
    assert errored.success is False and errored.final_reward == 0.0
    assert errored.error == "env.step raised: AttributeError: x has no attribute app (in filter_observation)"


def test_outputs_before_compaction_and_interventions_by_time(tmp_path):
    # A script runs the app with argv the command text does not show; OpenClaw later
    # compacts the output and strips "_" from the provider's tool-call id.
    sd = tmp_path / "scenario_universe_29_y"
    sd.mkdir()
    script = "python3 - <<'PY'\nimport subprocess\nfor off in (0, 3):\n    subprocess.run(['chats', 'list-recent-conversations', '--offset', str(off)])\nPY"
    call = {"id": "call_A1", "type": "function", "function": {"name": "exec", "arguments": json.dumps({"command": script})}}
    asked = {"role": "assistant", "content": "page through chats", "tool_calls": [{**call, "id": "callA1"}]}

    def entry(created, messages, reply):
        return {"type": "llm_call", "http_status": 200, "request": {"messages": messages},
                "raw_response": json.dumps({"created": created, "choices": [{"message": reply}]})}

    user = {"role": "user", "content": "Find my chats."}
    lines = [
        entry(100, [user], {"role": "assistant", "content": "page through chats", "tool_calls": [call]}),
        entry(110, [user, asked, {"role": "tool", "tool_call_id": "callA1", "content": "[{\"id\": 1}]"}],
              {"role": "assistant", "content": "thinking"}),
        entry(120, [user, asked, {"role": "tool", "tool_call_id": "callA1",
                                  "content": "[compacted: tool output removed to free context]"}],
              {"role": "assistant", "content": "Done."}),
    ]
    (sd / "trace.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    (sd / "result.json").write_text(json.dumps({"success": True, "agent_response": "Done."}))
    (sd / "env_rules_applied.jsonl").write_text(json.dumps(
        {"t": 105.5, "app": "chats", "subcommand": "list-recent-conversations", "kind": "blocked",
         "argv": ["list-recent-conversations", "--offset", "3"], "reason": "Error: busy"}) + "\n")
    trace = _convert(sd)
    step = trace.steps[0]
    assert step.filtered_observation.text.endswith("[{\"id\": 1}]")
    assert step.blocked_reason == "Error: busy" and "unattributed_interventions" not in step.info
