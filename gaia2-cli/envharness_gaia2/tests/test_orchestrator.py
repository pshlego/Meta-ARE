from __future__ import annotations

from envharness.agents.harness_agent import ScriptedHarnessAgent
from envharness.core.types import Candidate, DecideResult, Decision
from envharness.orchestration.budget import CappedAdaptive
from envharness.orchestration.objectives import DifficultyZone
from envharness.orchestration.orchestrator import OrchestratorConfig
from envharness.orchestration.runner import EnvSpec, PolicySpec
from envharness.orchestration.storage import TraceStore

from envharness_gaia2.orchestrator import ENV_IMPORT_PATH, ENV_STATE_SCHEMA, EXEC_TOOL_SCHEMA, Gaia2Orchestrator

from helpers import SEED_HARNESS, FakeGaia2Runner, cases

RULES = "class _Rules(Rules):\n    def filter_action(self, action, env_state):\n        return action\n"


def _orchestrator(tmp_path, agent, runner, task_cases, task_ids):
    return Gaia2Orchestrator(
        env_spec=EnvSpec(import_path=ENV_IMPORT_PATH),
        policy_spec=PolicySpec(client_factory="envharness.infra.llm:LiteLLMClient", client_kwargs={"model": "m"},
                               temperature=1.0),
        harness_agent=agent,
        objective=DifficultyZone(target_band=(0.3, 0.7)),
        budget=CappedAdaptive(max_k=2),
        trace_store=TraceStore(tmp_path / "traces.jsonl"),
        tool_schemas=[EXEC_TOOL_SCHEMA],
        env_state_schema=ENV_STATE_SCHEMA,
        config=OrchestratorConfig(task_id="gaia2-test", task_description="t", n_tasks=len(task_ids), k_per_candidate=3,
                                  explicit_task_ids=task_ids, compute_baseline=True, baseline_cache_dir=None,
                                  mutator_baseline_mode="full", skip_passthrough_candidates=True),
        log_dir=tmp_path / "log",
        gaia2=runner,
        cases=task_cases,
        harness_dir=SEED_HARNESS,
        work_dir=tmp_path / "work",
    )


def test_baseline_propose_rollout_accept(tmp_path):
    task_cases = cases(tmp_path, 3)
    # Baseline always fails; with rules, 2 of 3 rollouts succeed (in band).
    runner = FakeGaia2Runner(lambda sid, rules, rep: bool(rules) and rep < 2)

    def script(kind, payload):
        if kind == "propose":
            assert payload["ctx"].baseline.sr == 0.0
            assert payload["ctx"].baseline_traces, "full baseline trajectories must reach the designer"
            return Candidate(rules_code=RULES, rationale="exercise retries")
        if kind == "decide":
            return DecideResult(decision=Decision.ACCEPT, rationale="in band")
        raise AssertionError(kind)

    orch = _orchestrator(tmp_path, ScriptedHarnessAgent(script), runner, task_cases, [2])
    accepted = orch.run()
    traces = orch.trace_store.all()
    baseline = [t for t in traces if t.kind == "baseline"]
    assert len(baseline) == 3 and {t.candidate_id for t in baseline} == {"baseline"}
    assert [t.rollout_idx for t in baseline] == [0, 1, 2]
    assert len(accepted) == 3 and {t.kind for t in accepted} == {"accepted"}
    assert {t.rollout_seed for t in traces} == {2} and {t.task_id for t in traces} == {"gaia2-test"}
    assert sum(t.success for t in accepted) == 2
    assert [c["cases"] for c in runner.calls] == [["scenario_universe_29_t2"]] * 2
    assert runner.calls[0]["rules"] is None and runner.calls[1]["rules"] == RULES
    assert all(c["k"] == 3 for c in runner.calls)
    assert (tmp_path / "log" / "checkpoints").is_dir()


def test_passthrough_candidate_skips_rollouts(tmp_path):
    task_cases = cases(tmp_path, 2)
    runner = FakeGaia2Runner(lambda sid, rules, rep: rep == 0)

    def script(kind, payload):
        return Candidate(rationale="skip-in-band")

    orch = _orchestrator(tmp_path, ScriptedHarnessAgent(script), runner, task_cases, [0, 1])
    orch.run()
    assert len(runner.calls) == 2  # baselines only
    assert {t.kind for t in orch.trace_store.all()} == {"baseline"}


def test_stalled_agent_is_a_failure_not_an_error(tmp_path):
    from envharness_gaia2.gaia2_run import ScenarioResult
    from helpers import write_scenario_output

    task_cases = cases(tmp_path, 1)
    orch = _orchestrator(tmp_path, ScriptedHarnessAgent(lambda k, p: Candidate()), FakeGaia2Runner(lambda *a: False),
                         task_cases, [0])
    path = write_scenario_output(tmp_path / "out" / "scenario_universe_29_t0", success=None)
    stalled = ScenarioResult("scenario_universe_29_t0", 0, "task_failure", None, path, "h", None, "agent stalled")
    trace = orch._trace(stalled, 0, task_cases[0], Candidate(), "it")
    assert trace.error is None and trace.success is False
    assert trace.steps[0].info["failure_category"].startswith("agent_stalled")
    infra = ScenarioResult("scenario_universe_29_t0", 0, "execution_error", None, path, "h", None, "infrastructure")
    assert orch._trace(infra, 0, task_cases[0], Candidate(), "it").error.startswith("execution_error")
