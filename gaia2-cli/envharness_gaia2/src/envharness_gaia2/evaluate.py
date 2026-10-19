"""Stage 3: evaluate harnesses (nobank / orig / ours_r*) on a held-out manifest.

An arm maps every case to the harness its student runs with: one harness for
the whole manifest (nobank, or a whole bank in skills/), or the scenario's
retrieved top-k skills. Cases that share a harness share one gaia2-runner
call; the calls run in parallel with at most ``max_concurrent`` containers in
total.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from envharness_gaia2.gaia2_run import CaseRef, Gaia2Runner, ScenarioResult


@dataclass(frozen=True)
class Arm:
    harness_for: Callable[[CaseRef], tuple[Path, list[str]]]  # case -> (harness dir, injected skill titles)
    n_bank: int  # items in the arm's bank (0 for nobank)


@dataclass(frozen=True)
class _Job:
    arm: str
    harness_dir: Path
    cases: tuple[CaseRef, ...]
    out_dir: Path


def run_jobs(runner: Gaia2Runner, jobs: Sequence[_Job], *, k: int, max_concurrent: int) -> list[list[ScenarioResult]]:
    """Run the jobs concurrently; a job holds min(max_concurrent, cases x k) container slots."""
    cond = threading.Condition()
    free = [max_concurrent]

    def one(job: _Job) -> list[ScenarioResult]:
        need = min(max_concurrent, len(job.cases) * k)
        with cond:
            cond.wait_for(lambda: free[0] >= need)
            free[0] -= need
        try:
            return runner.run_batch(list(job.cases), harness_dir=job.harness_dir, env_rules_code=None, k=k,
                                    out_dir=job.out_dir, concurrency=need)
        finally:
            with cond:
                free[0] += need
                cond.notify_all()

    with cf.ThreadPoolExecutor(max_workers=max(1, max_concurrent)) as pool:
        return list(pool.map(one, jobs))


def evaluate(runner: Gaia2Runner, cases: Sequence[CaseRef], *, arms: dict[str, Arm], k: int, out_dir: Path,
             digest_fn: Callable[[Path], str], max_concurrent: int) -> dict:
    """Run every case ``k`` times per arm; returns {arm: {cases, valid, success_rate, ...}}."""
    jobs: list[_Job] = []
    plans: dict[str, dict] = {}
    for name, arm in arms.items():
        groups: dict[Path, list[CaseRef]] = {}
        skills: dict[str, list[str]] = {}
        for case in cases:
            harness_dir, titles = arm.harness_for(case)
            groups.setdefault(Path(harness_dir), []).append(case)
            skills[case.scenario_id] = titles
        single = len(groups) == 1
        for harness_dir, members in groups.items():
            jobs.append(_Job(name, harness_dir, tuple(members),
                             out_dir / name if single else out_dir / name / harness_dir.name))
        plans[name] = {"groups": groups, "skills": skills, "single": single, "n_bank": arm.n_bank}

    outcomes = run_jobs(runner, jobs, k=k, max_concurrent=max_concurrent)

    summary: dict[str, dict] = {}
    for name, plan in plans.items():
        expected = {harness_dir: digest_fn(harness_dir) for harness_dir in plan["groups"]}
        harness_of = {case.scenario_id: harness_dir for harness_dir, members in plan["groups"].items()
                      for case in members}
        per_case: dict[str, list] = {}
        for job, results in zip(jobs, outcomes):
            if job.arm != name:
                continue
            for result in results:
                per_case.setdefault(result.scenario_id, []).append({
                    "repetition": result.repetition,
                    "disposition": result.disposition,
                    "success": result.success,
                    "harness_digest": result.harness_digest,
                    "expected_harness_digest": expected[job.harness_dir],
                    "result_dir": str(result.scenario_dir) if result.scenario_dir else None,
                })
        valid = [r for rs in per_case.values() for r in rs if r["disposition"] in ("success", "task_failure")]
        # experiments/*/reproduce.py: SR = won / all attempts; an errored or missing attempt counts as a failure
        attempts = len(cases) * k
        won = sum(1 for rs in per_case.values() for r in rs if r["disposition"] == "success" and r["success"])
        only = next(iter(plan["groups"])) if plan["single"] else None
        summary[name] = {
            "harness_dir": str(only) if only else None,
            "harness_digest": expected[only] if only else None,
            "n_skills": plan["n_bank"],
            "case_harness": {sid: str(path) for sid, path in harness_of.items()},
            "case_skills": plan["skills"],
            "cases": per_case,
            "attempts": attempts,
            "valid": len(valid),
            "digest_ok": all(r["harness_digest"] == r["expected_harness_digest"] for r in valid),
            "success_rate": (won / attempts) if attempts else None,
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
