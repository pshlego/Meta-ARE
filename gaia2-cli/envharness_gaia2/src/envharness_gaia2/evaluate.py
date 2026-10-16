"""Evaluate fixed harness snapshots on a manifest (e.g. seed vs final on dev)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from envharness_gaia2.gaia2_run import CaseRef, Gaia2Runner


def evaluate(runner: Gaia2Runner, cases: Sequence[CaseRef], *, harnesses: dict[str, Path], k: int, out_dir: Path) -> dict:
    """Run every case ``k`` times per harness; returns {name: {scenario_id: [success, ...], ...}}."""
    summary: dict[str, dict] = {}
    for name, harness_dir in harnesses.items():
        results = runner.run_batch(list(cases), harness_dir=harness_dir, env_rules_code=None, k=k, out_dir=out_dir / name)
        per_case: dict[str, list] = {}
        for result in results:
            per_case.setdefault(result.scenario_id, []).append(
                {
                    "repetition": result.repetition,
                    "disposition": result.disposition,
                    "success": result.success,
                    "harness_digest": result.harness_digest,
                    "result_dir": str(result.scenario_dir) if result.scenario_dir else None,
                }
            )
        valid = [r for rs in per_case.values() for r in rs if r["disposition"] in ("success", "task_failure")]
        summary[name] = {
            "harness_dir": str(harness_dir),
            "cases": per_case,
            "valid": len(valid),
            "success_rate": (sum(1 for r in valid if r["success"]) / len(valid)) if valid else None,
        }
    (out_dir / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
