from __future__ import annotations

import pytest
from envharness.core.types import Action, Candidate, Observation, Step, Trace
from envharness.infra.llm import ChatResponse, ScriptedClient
from envharness.reasoning_bank.bank import MemoryItem

from envharness_gaia2.induce import format_trajectory, induce, select_units
from envharness_gaia2.skills import SkillRenderError, consolidate, render, select_new_skills

from helpers import HARNESS_TOOL, POLICY, SEED_HARNESS

RULES = "class _Rules(Rules):\n    pass\n"


def _trace(kind: str, success: bool, *, rules: str = "", cid: str = "c", steps: int = 2) -> Trace:
    step_list = [
        Step(raw_action=Action(name="exec", kwargs={"command": f"calendar get-events {i}"}),
             filtered_observation=Observation(text=("Task: Move my meeting.\n\n" if i == 0 else "") + f"out {i}"))
        for i in range(steps)
    ]
    return Trace(episode_id=f"{kind}-{success}-{cid}", iteration_id="i", task_id="t", candidate=Candidate(rules_code=rules),
                 candidate_id=cid, kind=kind, success=success, steps=step_list, duration_steps=steps, rollout_seed=1)


def _memory(n: int) -> str:
    return "\n".join(f"# Memory Item {i}\n## Title Check overlaps {i}\n## Description When booking.\n## Content List events first {i}."
                     for i in range(1, n + 1))


def test_cascade_tiers():
    base_fail, base_succ = _trace("baseline", False), _trace("baseline", True)
    acc = _trace("accepted", True, rules=RULES)
    expl_a, expl_b = _trace("exploration", True, rules=RULES, cid="a"), _trace("exploration", True, rules=RULES, cid="b")
    units, tier = select_units([base_fail, base_succ, acc], task_idx=1, scenario_id="s", pairs_per_task=2)
    assert tier == "1_accepted" and units[0].succ is acc and units[0].fail is base_fail
    units, tier = select_units([base_fail, expl_a, expl_b], task_idx=1, scenario_id="s", pairs_per_task=2)
    assert tier == "2_exploration_last" and [u.succ.candidate_id for u in units] == ["b"]
    units, tier = select_units([base_fail, base_succ], task_idx=1, scenario_id="s", pairs_per_task=2)
    assert tier == "3_baseline_fallback"
    units, tier = select_units([base_succ], task_idx=1, scenario_id="s", pairs_per_task=2)
    assert tier == "S_success_only" and units[0].fail is None
    units, tier = select_units([base_fail, base_fail, base_fail, acc, acc], task_idx=1, scenario_id="s", pairs_per_task=2)
    assert len(units) == 2
    units, tier = select_units([base_fail, base_succ, acc], task_idx=1, scenario_id="s", pairs_per_task=2, arm="orig")
    assert tier == "3_baseline_fallback"


def test_induce_paired_diff_parses_memory_items():
    seen = {}

    def responder(messages, tools):
        seen["system"], seen["user"] = messages[0].content, messages[1].content
        return ChatResponse(content=_memory(2))

    units, _ = select_units([_trace("baseline", False), _trace("accepted", True, rules=RULES)],
                            task_idx=1, scenario_id="scenario_universe_29_x", pairs_per_task=1)
    items = induce(ScriptedClient(responder=responder), units[0])
    assert [i.title for i in items] == ["Check overlaps 1", "Check overlaps 2"]
    assert items[0].source["tier"] == "1_accepted" and items[0].embedding == []
    assert "FAILURE trajectory" in seen["user"] and "Task: Move my meeting." in seen["user"]
    assert "$ calendar get-events 0" in format_trajectory(units[0].succ)


def _item(i: int) -> MemoryItem:
    return MemoryItem(title=f"Check overlaps {i}", description="When booking: verify.", content=f"List events first {i}.",
                      embedding=[], source={})


def test_select_and_consolidate_limits():
    new = select_new_skills(ScriptedClient(responder=lambda m, t: ChatResponse(content=_memory(5))),
                            [_item(i) for i in range(6)], [], max_new=3, source={"batch": 1})
    assert len(new) == 3 and new[0].source == {"batch": 1}
    assert select_new_skills(ScriptedClient(script=[]), [], [], max_new=3, source={}) == []
    merged = consolidate(ScriptedClient(responder=lambda m, t: ChatResponse(content=_memory(2))),
                         [_item(i) for i in range(5)], max_skills=2, source={})
    assert len(merged) == 2
    with pytest.raises(SkillRenderError):
        consolidate(ScriptedClient(responder=lambda m, t: ChatResponse(content="nothing")), [_item(1)], max_skills=2, source={})


def test_render_validates_with_harness_tool(tmp_path):
    out = tmp_path / "harness"
    digest = render([_item(1), _item(2)], seed_dir=SEED_HARNESS, out_dir=out, policy=POLICY, harness_tool=HARNESS_TOOL)
    skills = sorted(p.parent.name for p in (out / "skills").rglob("SKILL.md"))
    assert skills == ["check-overlaps-1", "check-overlaps-2"]
    text = (out / "skills" / "check-overlaps-1" / "SKILL.md").read_text()
    assert text.startswith("---\nname: check-overlaps-1\ndescription: \"When booking: verify.\"\n---")
    assert digest.startswith("sha256:")
    empty = render([], seed_dir=SEED_HARNESS, out_dir=tmp_path / "seed", policy=POLICY, harness_tool=HARNESS_TOOL)
    import subprocess
    import sys

    seed_digest = subprocess.run([sys.executable, str(HARNESS_TOOL), "digest", str(SEED_HARNESS)],
                                 capture_output=True, text=True, check=True).stdout.strip()
    assert empty == seed_digest


def test_render_rejects_leaks_and_oversize(tmp_path):
    leaky = MemoryItem(title="Leak", description="d", content="see scenario_universe_29_abc", embedding=[], source={})
    with pytest.raises(SkillRenderError, match="scenario identifiers"):
        render([leaky], seed_dir=SEED_HARNESS, out_dir=tmp_path / "a", policy=POLICY, harness_tool=HARNESS_TOOL)
    big = [MemoryItem(title=f"Big {i}", description="d", content="x" * 15000, embedding=[], source={}) for i in range(5)]
    with pytest.raises(SkillRenderError, match="budget"):
        render(big, seed_dir=SEED_HARNESS, out_dir=tmp_path / "b", policy=POLICY, harness_tool=HARNESS_TOOL)
