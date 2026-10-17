"""Convert one gaia2-runner scenario output directory into an envharness ``Trace``.

Only agent-visible artifacts are read: ``result.json`` (verdict fields only),
``trace.jsonl`` (the model's own requests/responses) and the env-rules logs
(``env_rules_applied.jsonl``, ``env_rules_errors.log``). Oracle-bearing files are never opened, and the
judge's ``failure_reasons`` (which embed oracle tool counts and expected
values) are reduced to a category.

Conversation reconstruction is ported from AutoSaddler
``plugins/gaia2_openclaw/evaluator.py`` (``_llm_calls``/``_conversation``).
"""

from __future__ import annotations

import json
import re
import shlex
import uuid
from pathlib import Path
from typing import Any

from envharness.core.types import Action, Candidate, Observation, Step, Trace

READ_FILES = ("result.json", "trace.jsonl", "env_rules_applied.jsonl", "env_rules_errors.log")
FORBIDDEN_FILES = ("daemon_judgments.jsonl", "eventd.log", "trace.html")
_MAX_OUTPUT_CHARS = 4000
_FAILURE_CATEGORIES = (
    (re.compile(r"Tool count mismatch", re.I), "tool_count_mismatch"),
    (re.compile(r"SMU count mismatch", re.I), "user_message_count_mismatch"),
    (re.compile(r"soft|llm check|judge_output", re.I), "soft_check_failed"),
    (re.compile(r"Cannot match|not matched|mismatch", re.I), "action_mismatch"),
    (re.compile(r"Scenario timeout|timed out|timeout", re.I), "timeout"),
)


def failure_category(result: dict[str, Any]) -> str | None:
    """A coarse failure label without oracle content (counts, expected values)."""
    if result.get("success") is True:
        return None
    reasons = result.get("failure_reasons")
    text = " ".join(str(item) for item in reasons) if isinstance(reasons, list) else ""
    if "agent={}" in text.replace(" ", ""):
        return "no_tool_calls"
    for pattern, label in _FAILURE_CATEGORIES:
        if text and pattern.search(text):
            return label
    if result.get("timed_out"):
        return "timeout"
    if result.get("success") is None:
        return "no_judgment"
    return "judge_rejected"


def _read_text(scenario_dir: Path, name: str) -> str | None:
    if name not in READ_FILES:  # defence in depth: only allow-listed artifacts
        raise PermissionError(f"trace_convert may not read {name}")
    path = scenario_dir / name
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None


def _llm_calls(scenario_dir: Path) -> list[dict[str, Any]]:
    text = _read_text(scenario_dir, "trace.jsonl") or ""
    calls = []
    for line in text.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict) or entry.get("type", "llm_call") != "llm_call":
            continue
        response = entry.get("raw_response")
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                response = {}
        request = entry.get("request") if isinstance(entry.get("request"), dict) else {}
        calls.append(
            {
                "http_status": entry.get("http_status"),
                "messages": request.get("messages") if isinstance(request.get("messages"), list) else [],
                "response": response if isinstance(response, dict) else {},
                "usage": (response or {}).get("usage") if isinstance(response, dict) else None,
            }
        )
    return calls


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", part.get("content", ""))) if isinstance(part, dict) else str(part)
            for part in content
        )
    return "" if content is None else str(content)


def conversation(scenario_dir: Path) -> list[dict[str, Any]]:
    """Messages of the last successful call plus its response (what the agent saw and did)."""
    successful = [call for call in _llm_calls(scenario_dir) if call["http_status"] == 200]
    if not successful:
        return []
    last = successful[-1]
    messages = [m for m in last["messages"] if isinstance(m, dict)]
    choices = last["response"].get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        final = choices[0].get("message")
        if isinstance(final, dict):
            messages.append({"role": "assistant", **final})
    return messages


_ENVELOPE = re.compile(r"^Sender \(untrusted metadata\):\s*```.*?```\s*", re.S)
_TIMESTAMP = re.compile(r"^\[[^\]]{8,40}\]\s*")


def user_request(text: str) -> str:
    """Strip OpenClaw's sender-metadata envelope and timestamp from a user message."""
    return _TIMESTAMP.sub("", _ENVELOPE.sub("", text.strip()), count=1).strip()


def _exec_command(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function") or {}
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    if isinstance(arguments, dict):
        return str(arguments.get("command", ""))
    return ""


APPS = frozenset({"calendar", "contacts", "emails", "messages", "chats", "rent-a-flat", "city", "cabs",
                  "shopping", "cloud-drive"})
_SEPARATORS = frozenset({"|", "||", "&&", ";", "&", "(", ")", "{", "}", "then", "do", "else"})


def app_invocations(command: str) -> list[tuple[str, list[str]]]:
    """``(app, argv)`` of every app CLI call in a shell command (best effort)."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        tokens = command.split()
    calls: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for token in tokens:
        if token in _SEPARATORS or token.startswith((">", "<", "2>")):
            current = None
            continue
        name = Path(token).name
        if current is None and name in APPS:
            current = (name, [])
            calls.append(current)
        elif current is not None:
            current[1].append(token)
    return calls


def _applied(scenario_dir: Path) -> list[dict[str, Any]]:
    text = _read_text(scenario_dir, "env_rules_applied.jsonl") or ""
    entries = []
    for line in text.splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def convert(
    scenario_dir: Path,
    *,
    task_idx: int,
    task_id: str,
    iteration_id: str,
    candidate: Candidate,
    capability: str,
) -> Trace:
    """Build a Trace whose Steps are the agent's ``exec`` calls plus its final reply.

    Env-rules interventions are attached to the step whose command ran the
    same app with the same argv (first unconsumed match); a blocked call sets
    ``blocked_reason`` as in envharness ``Rules``. Errors raised by rules code
    (fail-open in the container) make the rollout an error trace, as a hook
    exception does in the original runner.
    """
    scenario_dir = Path(scenario_dir)
    result = json.loads(_read_text(scenario_dir, "result.json") or "{}")
    messages = conversation(scenario_dir)
    pending = [e for e in _applied(scenario_dir) if isinstance(e, dict)]
    outputs: dict[str, str] = {
        str(m.get("tool_call_id")): _text(m.get("content"))
        for m in messages
        if m.get("role") == "tool" and m.get("tool_call_id")
    }
    task_text = user_request(next((_text(m.get("content")) for m in messages if m.get("role") == "user"), ""))
    base_info = {"scenario_id": scenario_dir.name, "capability": capability}

    steps: list[Step] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        reasoning = _text(message.get("content"))
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            command = _exec_command(tool_call)
            output = outputs.get(str(tool_call.get("id")), "")[:_MAX_OUTPUT_CHARS]
            action = Action(name="exec", kwargs={"command": command, "text": command})
            interventions = []
            for app, argv in app_invocations(command):
                match = next((e for e in pending if e.get("app") == app and e.get("argv") == argv), None)
                if match is not None:
                    pending.remove(match)
                    interventions.append(match)
            blocked = next((e for e in interventions if e.get("kind") == "blocked"), None)
            blocked_reason = str(blocked.get("reason") or "blocked") if blocked else None
            info = dict(base_info)
            if interventions:
                info["interventions"] = [
                    {k: v for k, v in e.items() if k in ("app", "subcommand", "kind", "wrote", "exit_code", "new_argv")}
                    for e in interventions
                ]
            steps.append(
                Step(
                    raw_action=action,
                    filtered_action=None if blocked_reason else action,
                    blocked_reason=blocked_reason,
                    raw_observation=Observation(text=output),
                    filtered_observation=Observation(text=output),
                    policy_raw_response=reasoning[:_MAX_OUTPUT_CHARS] or None,
                    info=info,
                )
            )
    final_reply = str(result.get("agent_response") or "")
    if final_reply or not steps:
        # The student's final answer is its last action (cf. OfficeQA's answer()).
        reply = Action(name="reply", kwargs={"text": final_reply[:_MAX_OUTPUT_CHARS]})
        steps.append(Step(raw_action=reply, filtered_action=reply, raw_observation=Observation(text=""),
                          filtered_observation=Observation(text=""), info=dict(base_info)))
    first = steps[0].filtered_observation
    steps[0].filtered_observation = Observation(text=f"Task: {task_text}\n\n{first.text if first else ''}".rstrip())
    steps[0].info["task"] = task_text
    if pending:
        steps[0].info["unattributed_interventions"] = len(pending)

    success = result.get("success") is True
    category = failure_category(result)
    steps[0].info["failure_category"] = category
    error = None if result.get("success") is not None else (category or "no_judgment")
    rules_errors = (_read_text(scenario_dir, "env_rules_errors.log") or "").strip()
    if rules_errors:
        error = f"rules_error: {rules_errors.splitlines()[0][:200]}"
    return Trace(
        episode_id=uuid.uuid4().hex,
        iteration_id=iteration_id,
        task_id=task_id,
        candidate=candidate,
        rollout_seed=task_idx,
        steps=steps,
        final_reward=1.0 if success else 0.0,
        success=success,
        duration_steps=len(steps),
        error=error,
    )


def missing_trace(*, task_idx: int, task_id: str, iteration_id: str, candidate: Candidate, reason: str) -> Trace:
    """Placeholder for a rollout that produced no result (counted as an error, not a failure)."""
    return Trace(
        episode_id=uuid.uuid4().hex,
        iteration_id=iteration_id,
        task_id=task_id,
        candidate=candidate,
        rollout_seed=task_idx,
        success=False,
        error=f"execution_error: {reason}"[:300],
    )
