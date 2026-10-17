"""Stage 3: evaluate fixed harnesses (nobank / orig / ours_r*) on a held-out manifest."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Sequence

from envharness_gaia2.gaia2_run import CaseRef, Gaia2Runner


def evaluate(runner: Gaia2Runner, cases: Sequence[CaseRef], *, harnesses: dict[str, Path], k: int, out_dir: Path,
             digest_fn: Callable[[Path], str]) -> dict:
    """Run every case ``k`` times per harness; returns {arm: {cases, valid, success_rate, ...}}."""
    summary: dict[str, dict] = {}
    for name, harness_dir in harnesses.items():
        expected = digest_fn(harness_dir)
        n_skills = len(list(Path(harness_dir).glob("skills/*/SKILL.md")))
        results = runner.run_batch(list(cases), harness_dir=harness_dir, env_rules_code=None, k=k, out_dir=out_dir / name)
        per_case: dict[str, list] = {}
        for result in results:
            per_case.setdefault(result.scenario_id, []).append({
                "repetition": result.repetition,
                "disposition": result.disposition,
                "success": result.success,
                "harness_digest": result.harness_digest,
                "result_dir": str(result.scenario_dir) if result.scenario_dir else None,
            })
        valid = [r for rs in per_case.values() for r in rs if r["disposition"] in ("success", "task_failure")]
        summary[name] = {
            "harness_dir": str(harness_dir),
            "harness_digest": expected,
            "n_skills": n_skills,
            "cases": per_case,
            "valid": len(valid),
            "digest_ok": all(r["harness_digest"] == expected for r in valid),
            "success_rate": (sum(1 for r in valid if r["success"]) / len(valid)) if valid else None,
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
