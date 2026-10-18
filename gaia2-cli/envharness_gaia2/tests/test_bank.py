from __future__ import annotations

import pytest
from envharness.core.types import Action, Candidate, FailureAnalysis, Observation, Step, Trace
from envharness.infra.llm import ChatResponse, ScriptedClient
from envharness.reasoning_bank.bank import MemoryItem

from envharness_gaia2.bank import (
    build,
    categorize_with_cascade,
    format_trajectory,
    induce_paired_diff_atomic,
    induce_pick,
    plan_orig,
    plan_ours,
    plan_pairs,
)
from envharness_gaia2.skills import SkillRenderError, render

from helpers import HARNESS_TOOL, POLICY, SEED_HARNESS

RULES = "class _Rules(Rules):\n    pass\n"


def _trace(kind, success, *, rules="", cid="c", eid=None, label=None) -> dict:
    steps = [Step(raw_action=Action(name="exec", kwargs={"command": "calendar get-events"}),
                  filtered_observation=Observation(text="Task: Move my meeting.\n\nout"),
                  policy_raw_response="check the calendar", info={"task": "Move my meeting."}),
             Step(raw_action=Action(name="reply", kwargs={"text": "Done."}), filtered_observation=Observation(text=""))]
    trace = Trace(episode_id=eid or f"{kind}-{success}-{cid}", iteration_id="i", task_id="t",
                  candidate=Candidate(rules_code=rules, rationale="friction on calendar"), candidate_id=cid, kind=kind,
                  success=success, steps=steps, duration_steps=2, rollout_seed=1,
                  failure_analysis=FailureAnalysis(primary_axis="A", label=label) if label else None)
    return trace.model_dump(mode="json")


def test_cascade_matches_the_original_tiers():
    bf, bs = _trace("baseline", False), _trace("baseline", True)
    acc = _trace("accepted", True, rules=RULES)
    ea, eb = _trace("exploration", True, rules=RULES, cid="a"), _trace("exploration", True, rules=RULES, cid="b")
    assert categorize_with_cascade([bf, bs, acc])[2] == "1_accepted"
    fail, succ, tier = categorize_with_cascade([bf, ea, eb])
    assert tier == "2_exploration_last" and [t["candidate_id"] for t in succ] == ["b"]
    assert categorize_with_cascade([bf, bs])[2] == "3_baseline_fallback"
    assert categorize_with_cascade([bs, acc])[2] == "skip_no_base_fail"
    assert categorize_with_cascade([bf])[2] == "skip_no_succ"
    # only the first 5 baselines decide base_fail (max_baseline_rollouts)
    assert categorize_with_cascade([*[_trace("baseline", True, eid=str(i)) for i in range(5)], bf])[2] == "skip_no_base_fail"


def test_pair_caps_and_orig_balance():
    traces = {"r1/task_000": [*[_trace("baseline", False, eid=f"f{i}") for i in range(3)],
                              *[_trace("accepted", True, rules=RULES, eid=f"a{i}", label="ambiguity") for i in range(3)],
                              _trace("baseline", True, eid="s0")]}
    ours = plan_ours(traces, max_pairs_per_task=5)
    assert len(ours.work) == 5 and ours.rows[0]["tier"] == "1_accepted"
    assert {w.failure_label for w in ours.work} == {"ambiguity"} and ours.work[0].task_text == "Move my meeting."
    orig = plan_orig(traces, seed=0)
    # cap = min(n_cascade=3*3, n_intra=3*1) -> 3 intra-baseline pairs
    assert len(orig.work) == 3 and {w.succ["episode_id"] for w in orig.work} == {"s0"}
    assert orig.rows[0]["cap"] == 3


def _long(trace: dict, n: int) -> dict:
    trace["steps"] = trace["steps"] * n
    return trace


def test_alfworld_pair_selection():
    # hardening accept (baseline 3/3 succeeds): ours pairs the accepted rollouts, orig the baseline ones
    hard = [*[_trace("baseline", True, eid=f"b{i}") for i in range(3)],
            _long(_trace("accepted", True, rules=RULES, eid="a-long"), 3), _trace("accepted", True, rules=RULES, eid="a-short"),
            _trace("accepted", False, rules=RULES, eid="a-f1"), _long(_trace("accepted", False, rules=RULES, eid="a-f2"), 2)]
    # rejected candidate: ours falls back to the baseline rollouts (all failures -> single_fail)
    rejected = [*[_trace("baseline", False, eid=f"f{i}") for i in range(3)],
                _trace("exploration", True, rules=RULES, eid="x")]
    traces = {"r1/task_000": hard, "r1/task_001": rejected}
    ours = plan_pairs(traces, condition="ours")
    assert [(r["tier"], r["induction"]) for r in ours.rows] == [("accepted", "paired_diff"),
                                                                 ("baseline_fallback", "single_fail")]
    assert ours.work[0].succ["episode_id"] == "a-short" and ours.work[0].fail["episode_id"] == "a-f2"
    assert ours.work[1].succ is None and ours.work[1].fail["episode_id"].startswith("f")
    orig = plan_pairs(traces, condition="orig")
    assert [(r["tier"], r["induction"]) for r in orig.rows] == [("baseline", "single_succ"), ("baseline", "single_fail")]


def test_alfworld_pair_prompts():
    seen = []

    def responder(messages, tools):
        seen.append((messages[0].content, messages[1].content))
        return ChatResponse(content="# Memory Item 1\n## Title Page past the first result\n## Description d\n## Content c")

    traces = {"r1/task_000": [_trace("accepted", True, rules=RULES, eid="s"), _trace("accepted", False, rules=RULES, eid="f")],
              "r1/task_001": [_trace("baseline", False, eid="f0")]}
    plan = plan_pairs(traces, condition="ours")
    client = ScriptedClient(responder=responder)
    induce_pick(client, plan.work[0], existing_lesson_titles=["Old lesson"])
    induce_pick(client, plan.work[1])
    (sys_pair, user_pair), (sys_fail, user_fail) = seen
    assert "COMPARING two trajectories on the same task in the same environment" in sys_pair
    assert user_pair.startswith("Task: Move my meeting.") and "FAILURE trajectory:" in user_pair
    assert "EXISTING LESSONS" in user_pair and "Old lesson" in user_pair
    assert "GAIA2 personal-assistant environment" in sys_fail and "ALFWorld" not in sys_fail
    assert "attempted to resolve the task but failed" in sys_fail and user_fail.startswith("User query: Move my meeting.")
    items, stats = build(plan, client, lambda texts: [[0.1] for _ in texts], condition="ours")
    assert stats["pairs"] == 2 and [it.source["induction"] for it in items] == ["paired_diff"]  # same title deduped
    assert stats["tiers"] == {"accepted": 1, "baseline_fallback": 1}


def test_induction_prompt_threads_diagnosis_and_existing_titles():
    seen = {}

    def responder(messages, tools):
        seen["system"], seen["user"] = messages[0].content, messages[1].content
        return ChatResponse(content="# Memory Item 1\n## Title Read the event before booking\n"
                                    "## Description When a ride depends on an event.\n## Content Run `calendar get-event`.")

    items = induce_paired_diff_atomic(ScriptedClient(responder=responder), task_text="Move my meeting.",
                                      baseline_trajectory_text="B", mutated_trajectory_text="M",
                                      failure_label="ambiguity", mutation_rationale="refuse first booking",
                                      existing_lesson_titles=["Old lesson"])
    assert items[0]["title"] == "Read the event before booking"
    assert "ambiguity" in seen["system"] and "refuse first booking" in seen["system"] and "GAIA2" in seen["system"]
    assert "Old lesson" in seen["user"] and "BASELINE trajectory (FAILED" in seen["user"]


def test_build_embeds_and_dedups():
    traces = {"r1/task_000": [_trace("baseline", False), _trace("accepted", True, rules=RULES)]}
    client = ScriptedClient(responder=lambda m, t: ChatResponse(
        content="# Memory Item 1\n## Title Same title\n## Description d\n## Content c\n"
                "# Memory Item 2\n## Title same TITLE\n## Description d\n## Content c"))
    items, stats = build(plan_ours(traces), client, lambda texts: [[0.1, 0.2] for _ in texts], condition="cascade")
    assert len(items) == 1 and items[0].embedding == [0.1, 0.2] and items[0].source["tier"] == "1_accepted"
    assert stats["items_before_dedup"] == 2 and stats["tiers"] == {"1_accepted": 1}


def test_trajectory_rendering_shows_reasoning_actions_and_reply():
    text = format_trajectory(_trace("baseline", False)["steps"])
    assert "Reasoning: check the calendar" in text and "<action>calendar get-events</action>" in text
    assert "<action>reply to user: Done.</action>" in text


def test_render_whole_bank_into_skills(tmp_path):
    items = [MemoryItem(title=f"Skill {i}", description="d", content="c", embedding=[]) for i in range(20)]
    digest = render(items, seed_dir=SEED_HARNESS, out_dir=tmp_path / "h", policy=POLICY, harness_tool=HARNESS_TOOL)
    assert digest.startswith("sha256:") and len(list((tmp_path / "h").glob("skills/*/SKILL.md"))) == 20
    leaky = [MemoryItem(title="Leak", description="d", content="see scenario_universe_29_abc", embedding=[])]
    with pytest.raises(SkillRenderError):
        render(leaky, seed_dir=SEED_HARNESS, out_dir=tmp_path / "x", policy=POLICY, harness_tool=HARNESS_TOOL)
