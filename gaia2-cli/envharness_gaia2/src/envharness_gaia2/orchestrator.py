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
from envharness_gaia2.rules_check import RulesCheckError, check_rules_code, setup_commands
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

ENV_STATE_SCHEMA = """\
GAIA2 bridge (OpenClaw student). The student's only tool is `exec` (a shell); the
GAIA2 apps are CLI commands: calendar, contacts, emails, messages, chats,
rent-a-flat, city, cabs, shopping, cloud-drive (`<app> <subcommand> --opt value`).
Your Rules hooks run on every student call of an app CLI (one process per call):

  filter_action(action, env_state) -> Action | Blocked     (A, before execution)
    action.name == "<app>.<subcommand>"; action.kwargs = {argv: [...], subcommand, text}
    argv = everything after the app name, starting with the subcommand:
      `calendar get-events --limit 5` -> argv == ["get-events", "--limit", "5"]
    Blocked(reason=...): the app does NOT run; `reason` is printed on stderr and the
      command exits 1 (looks like an app error to the student).
    The student sees stderr, then stdout, in one tool output (a missing final newline
    on stderr is added).
    Return Action(name=action.name, kwargs={**action.kwargs, "argv": [...]}) to run a
      different argv instead. Only kwargs["argv"] decides what runs (to run another
      subcommand of the same app, put it first in argv); name, subcommand and text are
      informational, and the app cannot change.
  modify_transition(action, response, env_state) -> EnvResponse   (T, after execution)
    response.observation.text = the command's stdout (JSON text for most subcommands;
      this bridge has no info["result"]: parse the text with json.loads). The apps
      print json.dumps(value, indent=2) + "\\n" (non-ASCII escaped as \\uXXXX): write
      changed JSON the same way, or the output no longer looks like the app's.
    response.info = {exit_code: int, stderr: str, wrote: bool}   # wrote: it changed app state
    Return EnvResponse(observation=Observation(text=...), info={...}) to change what the
      command reports (stdout / stderr / exit_code); the executed effect stays.
      reward, terminated, truncated and observation.data are ignored.
  filter_observation(obs, env_state) -> Observation                (O)
    obs.text = stdout shown to the student; obs.data = {exit_code, wrote}

env_state (env_state.app or env_state["app"]; read-only except extras):
  app: str; subcommand: str; argv: list[str]
  is_help: bool                      # True for probes (see below)
  call_index_app: int                # 1-based count of the student's calls to this app (incl. probes)
  call_index_total: int              # 1-based count of all app calls (incl. probes)
  call_index_subcommand: int         # 1-based count of real calls of this app+subcommand (0 for probes)
  executed_argv: list[str]           # (T/O only) argv that actually ran
  extras: dict                       # JSON scratch dict persisted across calls (write here)
Each call is a new process: self attributes do not persist, keep state in extras.
The counters include calls the CLI then rejects (e.g. a missing required option).
The student may run several app calls at once; they are still applied one at a time
(A, execution, T and O of one call finish before the next call starts), so every hook
sees extras as left by all earlier calls.
Only Rules, Action, Blocked, Observation, EnvResponse are pre-bound. Every module you
use must be imported at the top of rules_code (e.g. `import json` before json.loads),
and only re, json, random, hashlib, itertools and textwrap may be imported; no file,
OS or network access. A static check rejects (and the rollouts then fail with
RulesCheckError): rules_code over 8192 bytes (UTF-8), a name that is used but never
imported or defined, open, eval,
exec, compile, globals, locals, vars, setattr, delattr, input, any name or attribute
starting with "__", and references to os/sys/subprocess/socket/shutil/pathlib/io.
getattr/hasattr work for public names only. env_state, obs.data and response.info
are dicts (env_state also allows attribute access): use env_state.get("key").
Not reachable by hooks: the user's request and notifications (delivered by OpenClaw).
Probes also go through all three hooks: `<app> --help`, `<app> schema` and
`<app> <subcommand> --help`, which the student uses to learn the CLIs (they print
usage and change nothing). `<app> <subcommand> --help` has the same action.name and
subcommand as a real call of that subcommand; only is_help tells them apart.
For `<app> --help` and `<app> schema`, subcommand is "" and "schema".

Setup (in_env_actions, S0): app CLI calls run by the ENVIRONMENT before the student
starts, e.g. {"name": "exec", "kwargs_json": "{\\"command\\": \\"contacts add-new-contact --first-name Ana\\"}"}.
One app call per item (no shell syntax), at most 20. They are environment events:
they change the initial app state but are not the student's actions. They run
instantly at the scenario's start time, before any scheduled environment event and
before the student starts: they cannot wait for or trigger later events, read
commands among them have no effect, and their output is not shown to anyone. A
failing command (e.g. an invalid option) is skipped, so use the exact options listed
below.
"""


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
        forbidden_ids: Sequence[str] = (),
        on_rollouts: Callable[[int], None] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("runner", _UnusedRunner())
        super().__init__(*args, **kwargs)
        self.gaia2 = gaia2
        self.cases = list(cases)
        self.harness_dir = Path(harness_dir)
        self.work_dir = Path(work_dir)
        self.forbidden_ids = tuple(forbidden_ids)
        self.on_rollouts = on_rollouts

    # -- overrides --------------------------------------------------------------
    def _rollout_baseline_k(self, task_idx: int, task_id: int) -> list[Trace]:
        traces = self._run(
            task_id,
            Candidate(rules_code="", in_env_actions=[]),
            iteration_id=f"baseline-{task_idx:04d}",
            out_dir=self.work_dir / "baseline",
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
            out_dir=self.work_dir / f"cand_{attempt}_{candidate_id}",
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
        try:
            check_rules_code(candidate.rules_code, forbidden_ids=self.forbidden_ids)
            setup = setup_commands(candidate.in_env_actions, forbidden_ids=self.forbidden_ids)
        except RulesCheckError as exc:
            # Like envharness's RulesCodeError: the episodes fail before the env boots.
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "rules_check_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
            return [
                missing_trace(task_idx=task_id, task_id=self.config.task_id, iteration_id=iteration_id,
                              candidate=candidate, reason=f"RulesCheckError: {exc}")
                for _ in range(k)
            ]
        results = self.gaia2.run_batch(
            [case],
            harness_dir=self.harness_dir,
            env_rules_code=candidate.rules_code or None,
            setup_commands=setup,
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
        if result.disposition == "task_failure" and trace.error and not trace.error.startswith("rules_error"):
            # A stalled agent (no turn boundary / scenario timeout after real
            # model calls) is a task failure, not an execution error: keep it
            # in success statistics and in skill induction.
            trace.steps[0].info["failure_category"] = f"agent_stalled:{trace.error}"
            trace.error = None
        return trace
