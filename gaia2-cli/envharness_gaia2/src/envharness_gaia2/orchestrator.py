"""EnvRigger over gaia2-runner: envharness ``Orchestrator`` with K rollouts per runner call.

The upstream orchestrator runs one ``EpisodeRunner.run(spec)`` per rollout. A
GAIA2 rollout is a container scenario, and gaia2-runner already runs K of them
in parallel with ``pass_at=K``; so baseline and candidate rollouts are
overridden to issue one ``run_batch(..., k=K)`` call and convert its outputs
into envharness Traces stamped exactly like the parent implementation
(candidate_id, rollout_idx, kind, iteration_id, task_id, rollout_seed).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from envharness.core.types import Candidate, Trace
from envharness.orchestration.orchestrator import Orchestrator
from envharness.orchestration.runner import EpisodeRunner, EpisodeSpec

from envharness_gaia2.gaia2_run import CaseRef, Gaia2Runner, ScenarioResult
from envharness_gaia2.trace_convert import convert, missing_trace

ENV_IMPORT_PATH = "envharness_gaia2.env:Gaia2ScenarioEnv"

EXEC_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "exec",
        "description": "OpenClaw's only tool: run a shell command. GAIA2 apps are CLI commands.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}

ENV_STATE_SCHEMA = (
    "env_state passed to filter_action/filter_observation:\n"
    '  app: str              # "calendar", "contacts", "emails", "messages", "chats",\n'
    '                        # "rent-a-flat", "city", "cabs", "shopping", "cloud-drive"\n'
    '  subcommand: str       # e.g. "get-contacts", "add-event" ("" or "schema" for probes)\n'
    "  is_help: bool         # True for --help / schema probes\n"
    "  call_index_app: int   # 1-based count of the student's calls to this app (incl. probes)\n"
    "  call_index_total: int # 1-based count of all the student's app calls (incl. probes)\n"
    "  call_index_subcommand: int  # 1-based count of real calls of this app+subcommand (0 for probes)\n"
    "Rules objects keep no state across calls (one process per app call); use the counters.\n"
    "Observation.data for filter_observation: {exit_code: int, wrote: bool}"
)


class _UnusedRunner(EpisodeRunner):
    def run(self, spec: EpisodeSpec) -> Trace:  # pragma: no cover - never called
        raise RuntimeError("Gaia2Orchestrator runs rollouts through gaia2-runner batches")


class Gaia2Orchestrator(Orchestrator):
    def __init__(
        self,
        *args,
        gaia2: Gaia2Runner,
        cases: Sequence[CaseRef],
        harness_dir: Path,
        work_dir: Path,
        on_rollouts: Callable[[int], None] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("runner", _UnusedRunner())
        super().__init__(*args, **kwargs)
        self.gaia2 = gaia2
        self.cases = list(cases)
        self.harness_dir = Path(harness_dir)
        self.work_dir = Path(work_dir)
        self.on_rollouts = on_rollouts

    # -- overrides --------------------------------------------------------------
    def _rollout_baseline_k(self, task_idx: int, task_id: int) -> list[Trace]:
        traces = self._run(
            task_id,
            Candidate(rules_code="", in_env_actions=[]),
            iteration_id=f"baseline-{task_idx:04d}",
            out_dir=self.work_dir / f"task_{task_id:03d}" / "baseline",
        )
        for k, trace in enumerate(traces):
            trace.candidate_id = "baseline"
            trace.rollout_idx = k
            self._log_rollout(f"baseline-{task_idx:04d}", 0, "baseline", k, task_id, 0.0, trace)
        return traces

    def _rollout_k(
        self,
        candidate: Candidate,
        candidate_id: str,
        iteration_id: str,
        task_idx: int,
        attempt: int,
        ctx_task_id: int,
    ) -> list[Trace]:
        traces = self._run(
            ctx_task_id,
            candidate,
            iteration_id=iteration_id,
            out_dir=self.work_dir / f"task_{ctx_task_id:03d}" / f"cand_{attempt}_{candidate_id}",
        )
        for k, trace in enumerate(traces):
            trace.candidate_id = candidate_id
            trace.rollout_idx = k
            trace.kind = "exploration"  # promoted to "accepted" by _iterate on ACCEPT
            self._log_rollout(iteration_id, attempt, candidate_id, k, ctx_task_id, 0.0, trace)
        return traces

    # -- helpers ------------------------------------------------------------------
    def _run(self, task_id: int, candidate: Candidate, *, iteration_id: str, out_dir: Path) -> list[Trace]:
        case = self.cases[task_id]
        k = self.config.k_per_candidate
        results = self.gaia2.run_batch(
            [case],
            harness_dir=self.harness_dir,
            env_rules_code=candidate.rules_code or None,
            k=k,
            out_dir=out_dir,
        )
        if self.on_rollouts is not None:
            self.on_rollouts(len(results))
        by_rep = {r.repetition: r for r in results if r.scenario_id == case.scenario_id}
        return [self._trace(by_rep.get(k_), task_id, case, candidate, iteration_id) for k_ in range(k)]

    def _trace(
        self,
        result: ScenarioResult | None,
        task_id: int,
        case: CaseRef,
        candidate: Candidate,
        iteration_id: str,
    ) -> Trace:
        if result is None or result.scenario_dir is None or result.disposition == "execution_error":
            return missing_trace(
                task_idx=task_id,
                task_id=self.config.task_id,
                iteration_id=iteration_id,
                candidate=candidate,
                reason=result.reason if result else "no result",
            )
        trace = convert(
            result.scenario_dir,
            task_idx=task_id,
            task_id=self.config.task_id,
            iteration_id=iteration_id,
            candidate=candidate,
            capability=case.capability,
        )
        trace.policy_model_id = self.policy_spec.client_kwargs.get("model", "")
        if result.disposition == "task_failure" and trace.error:
            # A stalled agent (no turn boundary / scenario timeout after real
            # model calls) is a task failure, not an execution error: keep it
            # in success statistics and in skill induction.
            trace.steps[0].info["failure_category"] = f"agent_stalled:{trace.error}"
            trace.error = None
        return trace
