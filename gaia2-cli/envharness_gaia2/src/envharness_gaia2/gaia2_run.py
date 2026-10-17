"""Run GAIA2 scenarios on OpenClaw through gaia2-runner (one call = K rollouts).

Ported from AutoSaddler ``plugins/gaia2_openclaw`` (runner.py ``_toml``,
``_stage_cases``, ``_process_timeout``, ``_remove_containers``; evaluator.py
``_disposition``; config.py ``wrap_docker_group``), adapted to batches driven by
the EnvHarness orchestrator.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_PER_SCENARIO_OVERHEAD_SECONDS = 120
_STARTUP_GRACE_SECONDS = 600
_AGENT_STALL_MARKERS = ("no turn boundary", "idle timeout or agent stuck", "Scenario timeout")


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bundle_digest(directory: Path) -> str:
    """Mirror of ``gaia2_cli.env_setup.bundle_digest`` (rules.py and/or setup.json)."""
    digest = hashlib.sha256()
    for name in ("rules.py", "setup.json"):
        path = Path(directory) / name
        if path.is_file():
            digest.update(f"{name}:{hashlib.sha256(path.read_bytes()).hexdigest()}\n".encode())
    return "sha256:" + digest.hexdigest()


def wrap_docker_group(argv: list[str], docker_group: str | None) -> list[str]:
    """Run ``argv`` with the docker group (``sg``) when the login session lacks it."""
    if docker_group is None:
        return argv
    return ["sg", docker_group, "-c", shlex.join(argv)]


@dataclass(frozen=True)
class CaseRef:
    """One GAIA2 scenario of the experiment's dataset."""

    scenario_id: str
    capability: str
    source: Path
    sha256: str


@dataclass(frozen=True)
class RunSettings:
    meta_are_repo: Path
    dataset_root: Path
    image: str
    docker_group: str | None
    model: str
    model_provider: str
    model_endpoint: str | None
    reasoning_effort: str | None
    thinking: str
    azure_config_dir: Path | None
    judge_model: str
    judge_provider: str
    judge_endpoint: str | None
    judge_time_check: str
    scenario_timeout_seconds: int
    max_scenario_duration: int | None
    health_timeout_seconds: int
    process_completion_grace_seconds: int
    max_concurrent: int
    infrastructure_retries: int

    @property
    def runner_project(self) -> Path:
        return self.meta_are_repo / "gaia2-cli" / "runner"


@dataclass
class ScenarioResult:
    scenario_id: str
    repetition: int
    disposition: str  # success | task_failure | execution_error | missing
    success: bool | None
    scenario_dir: Path | None
    harness_digest: str | None
    env_rules_digest: str | None
    reason: str


class Gaia2Runner:
    """Executes ``gaia2-runner run-config`` for a set of cases with a harness snapshot."""

    def __init__(self, settings: RunSettings, *, harness_digest_fn, label_prefix: str) -> None:
        self.settings = settings
        self.harness_digest_fn = harness_digest_fn
        self.label_prefix = label_prefix
        self.invocations = 0
        self.rollouts = 0  # scenario runs actually launched (excludes resumed attempts)

    # -- public ----------------------------------------------------------------
    def run_batch(
        self,
        cases: Sequence[CaseRef],
        *,
        harness_dir: Path,
        env_rules_code: str | None,
        k: int,
        out_dir: Path,
        setup_commands: Sequence[str] = (),
    ) -> list[ScenarioResult]:
        """Run every case ``k`` times; retry infrastructure failures (``infrastructure_retries``)."""
        out_dir = Path(out_dir).resolve()
        bundle = (env_rules_code, tuple(setup_commands))
        results = self._run_attempt(cases, harness_dir, bundle, k, out_dir / "attempt-1")
        for retry in range(self.settings.infrastructure_retries):
            failed_ids = sorted({r.scenario_id for r in results if r.disposition in ("execution_error", "missing")})
            if not failed_ids:
                break
            retry_cases = [case for case in cases if case.scenario_id in failed_ids]
            retried = self._run_attempt(
                retry_cases, harness_dir, bundle, k, out_dir / f"attempt-{retry + 2}"
            )
            by_key = {(r.scenario_id, r.repetition): r for r in retried}
            results = [by_key.get((r.scenario_id, r.repetition), r) if r.scenario_id in failed_ids else r for r in results]
        return results

    # -- one gaia2-runner invocation ------------------------------------------------
    def _run_attempt(
        self,
        cases: Sequence[CaseRef],
        harness_dir: Path,
        bundle: tuple[str | None, tuple[str, ...]],
        k: int,
        attempt_dir: Path,
    ) -> list[ScenarioResult]:
        env_rules_code, setup = bundle
        attempt_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = attempt_dir / "dataset"
        output_dir = attempt_dir / "output"
        harness_dir = Path(harness_dir).resolve()
        _make_world_readable(harness_dir)
        harness_digest = self.harness_digest_fn(harness_dir)
        rules_dir = None
        rules_digest = None
        has_rules = env_rules_code is not None and bool(env_rules_code.strip())
        if has_rules or setup:
            rules_dir = attempt_dir / "env_rules"
            if rules_dir.exists():
                shutil.rmtree(rules_dir)
            rules_dir.mkdir()
            if has_rules:
                (rules_dir / "rules.py").write_text(env_rules_code, encoding="utf-8")
            if setup:
                (rules_dir / "setup.json").write_text(json.dumps({"commands": list(setup)}, indent=2) + "\n",
                                                      encoding="utf-8")
            rules_digest = bundle_digest(rules_dir)
            _make_world_readable(rules_dir)
        self._stage_cases(cases, dataset_dir)
        config_path = attempt_dir / "run.toml"
        config_path.write_text(
            self._toml(dataset_dir, output_dir, harness_dir, rules_dir, k), encoding="utf-8"
        )
        label = f"{self.label_prefix}={hashlib.sha256(str(attempt_dir).encode()).hexdigest()[:16]}"
        argv = wrap_docker_group(
            [
                "uv", "run", "--project", str(self.settings.runner_project), "--python", "3.12",
                "gaia2-runner", "run-config", "--config", str(config_path),
            ],
            self.settings.docker_group,
        )
        command = {
            "argv": argv,
            "run_toml_sha256": sha256_file(config_path),
            "harness_digest": harness_digest,
            "env_rules_digest": rules_digest,
            "label": label,
            "cases": [case.scenario_id for case in cases],
            "k": k,
        }
        command_digest = hashlib.sha256(json.dumps(command, sort_keys=True).encode()).hexdigest()
        (attempt_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
        completion_path = attempt_dir / "completion.json"
        completion = _read_json(completion_path)
        if not (completion and completion.get("command_sha256") == command_digest):
            if output_dir.exists():
                shutil.rmtree(output_dir)
            environment = os.environ.copy()
            if self.settings.azure_config_dir is not None:
                environment["GAIA2_AZURE_CONFIG_DIR"] = str(self.settings.azure_config_dir)
            environment["GAIA2_CONTAINER_LABELS"] = label
            self.invocations += 1
            self.rollouts += len(cases) * k
            try:
                completed = subprocess.run(
                    argv,
                    cwd=self.settings.meta_are_repo,
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=self._process_timeout(len(cases), k),
                    check=False,
                )
                returncode, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
            except subprocess.TimeoutExpired as exc:
                returncode = -9
                stdout = exc.stdout if isinstance(exc.stdout, str) else ""
                stderr = (exc.stderr if isinstance(exc.stderr, str) else "") + "\n[process timeout]"
            (attempt_dir / "stdout.log").write_text(stdout or "", encoding="utf-8")
            (attempt_dir / "stderr.log").write_text(stderr or "", encoding="utf-8")
            self.remove_containers(label)
            completion_path.write_text(
                json.dumps({"returncode": returncode, "command_sha256": command_digest}) + "\n",
                encoding="utf-8",
            )
        return self._collect(cases, k, output_dir, harness_digest, rules_digest)

    def remove_containers(self, label: str) -> None:
        """Remove only containers labelled for this invocation."""
        listed = subprocess.run(
            wrap_docker_group(["docker", "ps", "-aq", "--filter", f"label={label}"], self.settings.docker_group),
            text=True,
            capture_output=True,
            check=False,
        )
        ids = listed.stdout.split()
        if ids:
            subprocess.run(
                wrap_docker_group(["docker", "rm", "-f", *ids], self.settings.docker_group),
                capture_output=True,
                check=False,
            )

    def _process_timeout(self, case_count: int, k: int) -> float:
        waves = (case_count * k + self.settings.max_concurrent - 1) // self.settings.max_concurrent
        per = self.settings.scenario_timeout_seconds + self.settings.health_timeout_seconds + _PER_SCENARIO_OVERHEAD_SECONDS
        return waves * per + self.settings.process_completion_grace_seconds + _STARTUP_GRACE_SECONDS

    def _stage_cases(self, cases: Sequence[CaseRef], dataset_dir: Path) -> None:
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        for case in cases:
            if sha256_file(case.source) != case.sha256:
                raise ValueError(f"scenario digest drift: {case.source}")
            destination = dataset_dir / case.capability / f"{case.scenario_id}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(case.source, destination)
            destination.chmod(0o644)
            destination.parent.chmod(0o755)
            if sha256_file(destination) != case.sha256:
                raise ValueError(f"staged scenario digest drift: {destination}")
        dataset_dir.chmod(0o755)

    def _toml(self, dataset_dir: Path, output_dir: Path, harness_dir: Path, rules_dir: Path | None, k: int) -> str:
        s = self.settings
        q = json.dumps
        lines = [
            "[target]",
            f"dataset_root = {q(str(dataset_dir))}",
            "",
            "[agent]",
            f"image = {q(s.image)}",
            'runtime = "docker"',
            f"provider = {q(s.model_provider)}",
            f"model = {q(s.model)}",
            f"thinking = {q(s.thinking)}",
            f"harness_dir = {q(str(harness_dir))}",
        ]
        if rules_dir is not None:
            lines.append(f"env_rules_dir = {q(str(rules_dir))}")
        if s.model_endpoint:
            lines.append(f"base_url = {q(s.model_endpoint)}")
        if s.reasoning_effort:
            lines.append(f"reasoning_effort = {q(s.reasoning_effort)}")
        lines += [
            "",
            "[judge]",
            f"provider = {q(s.judge_provider)}",
            f"model = {q(s.judge_model)}",
            f"time_check = {q(s.judge_time_check)}",
        ]
        if s.judge_endpoint:
            lines.append(f"base_url = {q(s.judge_endpoint)}")
        if s.azure_config_dir is not None:
            lines += ["", "[azure]", f"config_dir = {q(str(s.azure_config_dir))}"]
        lines += [
            "",
            "[run]",
            f"timeout = {s.scenario_timeout_seconds}",
            f"health_timeout = {s.health_timeout_seconds}",
            f"concurrency = {s.max_concurrent}",
            f"pass_at = {k}",
            "retry = false",
            f"output_dir = {q(str(output_dir))}",
            'log_level = "INFO"',
        ]
        if s.max_scenario_duration is not None:
            lines.append(f"max_scenario_duration = {s.max_scenario_duration}")
        return "\n".join(lines) + "\n"

    def _collect(
        self,
        cases: Sequence[CaseRef],
        k: int,
        output_dir: Path,
        harness_digest: str,
        rules_digest: str | None,
    ) -> list[ScenarioResult]:
        found: dict[tuple[str, int], Path] = {}
        if output_dir.is_dir():
            for result_path in output_dir.rglob("result.json"):
                scenario_dir = result_path.parent
                parts = scenario_dir.relative_to(output_dir).parts
                run = next((int(p[4:]) for p in parts if p.startswith("run_") and p[4:].isdigit()), 1)
                found[(scenario_dir.name, run - 1)] = scenario_dir
        results = []
        for case in cases:
            for rep in range(k):
                scenario_dir = found.get((case.scenario_id, rep))
                if scenario_dir is None:
                    results.append(
                        ScenarioResult(case.scenario_id, rep, "missing", None, None, None, None, "no result.json")
                    )
                    continue
                result = _read_json(scenario_dir / "result.json") or {}
                disposition, reason = disposition_of(
                    result,
                    llm_statuses(scenario_dir / "trace.jsonl"),
                    _read_json(scenario_dir / "harness_applied.json"),
                    harness_digest,
                    rules_digest,
                )
                results.append(
                    ScenarioResult(
                        case.scenario_id,
                        rep,
                        disposition,
                        result.get("success") if disposition != "execution_error" else None,
                        scenario_dir,
                        result.get("harness_digest"),
                        result.get("env_rules_digest"),
                        reason,
                    )
                )
        return results


def disposition_of(
    result: dict,
    llm_statuses_: Sequence[int | None],
    applied: dict | None,
    expected_harness_digest: str,
    expected_rules_digest: str | None,
) -> tuple[str, str]:
    """success / task_failure / execution_error (ported from AutoSaddler ``_disposition``)."""
    if result.get("harness_digest") != expected_harness_digest or (applied or {}).get("digest") != expected_harness_digest:
        return "execution_error", "harness digest mismatch"
    if result.get("env_rules_digest") != expected_rules_digest:
        return "execution_error", "env rules digest mismatch"
    success = result.get("success")
    if success is True:
        return "success", "judge accepted"
    if success is False:
        return "task_failure", "judge rejected"
    error = str(result.get("error") or "")
    reachable = any(status == 200 for status in llm_statuses_)
    if reachable and (result.get("timed_out") or any(marker in error for marker in _AGENT_STALL_MARKERS)):
        return "task_failure", f"agent stalled: {error[:200]}"
    return "execution_error", f"infrastructure: {error[:200] or 'no judgment'}"


def llm_statuses(trace_path: Path) -> list[int | None]:
    if not trace_path.is_file():
        return []
    statuses = []
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            statuses.append(json.loads(line).get("http_status"))
        except (json.JSONDecodeError, AttributeError):
            continue
    return statuses


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _make_world_readable(root: Path) -> None:
    root.chmod(root.stat().st_mode | 0o755)
    for path in root.rglob("*"):
        path.chmod(path.stat().st_mode | (0o755 if path.is_dir() else 0o644))
