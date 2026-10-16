"""Skill induction from EnvRigger traces (EnvHarness cascade + paired-diff).

Pair selection ports ``categorize_with_cascade`` from envharness
``experiments/swebench/bank_distillation/build_ours_bank.py`` (SWE-bench
dependencies removed):

  Tier1  baseline failure x accepted-variation success
  Tier2  baseline failure x success of the last exploration candidate
  Tier3  baseline failure x baseline success
  (no baseline failure) -> success-only induction from a baseline success

The ``orig`` arm uses Tier3 / success-only (original-environment traces only).
Induction prompts adapt envharness ``PAIRED_DIFF_SI`` / ``SUCCESSFUL_SI``;
their output is parsed with envharness ``parse_memory_items``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from envharness.core.types import Trace
from envharness.infra.llm import LLMClient, Message
from envharness.reasoning_bank.bank import MemoryItem
from envharness.reasoning_bank.induce import parse_memory_items

from envharness_gaia2 import PROMPTS_DIR

_MAX_TRAJ_CHARS = 9000


@dataclass
class InductionUnit:
    task_idx: int
    scenario_id: str
    tier: str
    fail: Trace | None
    succ: Trace


def _has_rules(trace: Trace) -> bool:
    return bool((trace.candidate.rules_code or "").strip())


def select_units(
    traces: Sequence[Trace], *, task_idx: int, scenario_id: str, pairs_per_task: int, arm: str = "ours"
) -> tuple[list[InductionUnit], str]:
    """Cascade pair selection for one task's traces; returns (units, tier label)."""
    usable = [t for t in traces if t.steps and not t.error]
    base_fail = [t for t in usable if t.kind == "baseline" and not t.success]
    base_succ = [t for t in usable if t.kind == "baseline" and t.success]
    acc_succ = [t for t in usable if t.kind == "accepted" and _has_rules(t) and t.success]
    expl_succ = [t for t in usable if t.kind == "exploration" and _has_rules(t) and t.success]
    if arm == "orig":
        acc_succ, expl_succ = [], []

    if not base_fail:
        if base_succ:
            shortest = min(base_succ, key=lambda t: t.duration_steps)
            return [InductionUnit(task_idx, scenario_id, "S_success_only", None, shortest)], "S_success_only"
        return [], "skip_no_traces"
    if acc_succ:
        succ, tier = acc_succ, "1_accepted"
    elif expl_succ:
        last = expl_succ[-1].candidate_id
        succ, tier = [t for t in expl_succ if t.candidate_id == last], "2_exploration_last"
    elif base_succ:
        succ, tier = base_succ, "3_baseline_fallback"
    else:
        return [], "skip_no_succ"
    units = []
    for fail in base_fail:
        for good in succ:
            units.append(InductionUnit(task_idx, scenario_id, tier, fail, good))
            if len(units) >= pairs_per_task:
                return units, tier
    return units, tier


def task_query(trace: Trace) -> str:
    if not trace.steps or trace.steps[0].filtered_observation is None:
        return ""
    first = trace.steps[0].filtered_observation.text
    return first.split("\n\n", 1)[0].removeprefix("Task: ").strip()


def format_trajectory(trace: Trace, *, max_obs_chars: int = 500) -> str:
    """Commands, (filtered) outputs, blocked reasons and the final reply."""
    lines = []
    for index, step in enumerate(trace.steps, start=1):
        command = str(step.raw_action.kwargs.get("command") or step.raw_action.kwargs.get("text") or "")
        text = step.filtered_observation.text if step.filtered_observation else ""
        if index == 1:
            text = text.split("\n\n", 1)[1] if "\n\n" in text else ""
        reasoning = (step.policy_raw_response or "").strip()
        if reasoning:
            lines.append(f"[{index}] thought: {reasoning[:300]}")
        lines.append(f"[{index}] $ {command}")
        if step.blocked_reason:
            lines.append(f"    ! {step.blocked_reason}")
        if text.strip():
            lines.append(f"    > {text.strip()[:max_obs_chars]}")
    final = trace.steps[-1].info.get("final_reply") if trace.steps else None
    if final:
        lines.append(f"FINAL REPLY TO USER: {final[:800]}")
    text = "\n".join(lines)
    return text if len(text) <= _MAX_TRAJ_CHARS else text[:3000] + "\n...[middle omitted]...\n" + text[-(_MAX_TRAJ_CHARS - 3000):]


def induce(client: LLMClient, unit: InductionUnit, *, max_items: int = 3) -> list[MemoryItem]:
    query = task_query(unit.succ) or (task_query(unit.fail) if unit.fail else "")
    if unit.fail is not None:
        system = (PROMPTS_DIR / "induce_paired_diff.md").read_text(encoding="utf-8")
        user = (
            f"Task: {query[:2000]}\n\nFAILURE trajectory:\n{format_trajectory(unit.fail)}\n\n"
            f"SUCCESS trajectory:\n{format_trajectory(unit.succ)}\n\n"
            "Compare. Extract atomic skills. If no clean difference, emit NOTHING."
        )
    else:
        system = (PROMPTS_DIR / "induce_success.md").read_text(encoding="utf-8")
        user = f"User request: {query[:2000]}\n\nTrajectory:\n{format_trajectory(unit.succ)}"
    response = client.chat([Message(role="system", content=system), Message(role="user", content=user)])
    items = []
    for parsed in parse_memory_items(response.content or "")[:max_items]:
        items.append(
            MemoryItem(
                title=parsed["title"].strip(),
                description=parsed["description"].strip(),
                content=parsed["content"].strip(),
                embedding=[],
                source={
                    "task_idx": unit.task_idx,
                    "scenario_id": unit.scenario_id,
                    "tier": unit.tier,
                    "succ_episode_id": unit.succ.episode_id,
                    "fail_episode_id": unit.fail.episode_id if unit.fail else None,
                },
            )
        )
    return items
