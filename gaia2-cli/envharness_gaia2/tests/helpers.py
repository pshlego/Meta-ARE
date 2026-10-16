"""Fakes shared by the envharness_gaia2 tests (no docker, no API)."""

from __future__ import annotations

import json
from pathlib import Path

from envharness_gaia2.gaia2_run import CaseRef, ScenarioResult

GAIA2_CLI = Path(__file__).resolve().parents[2]
SEED_HARNESS = GAIA2_CLI / "harness" / "openclaw"
POLICY = GAIA2_CLI / "harness" / "policy" / "openclaw_overlay_policy.json"
HARNESS_TOOL = GAIA2_CLI / "shared" / "harness_tool.py"
FIXTURES = Path(__file__).parent / "fixtures"


def write_scenario_output(path: Path, *, success: bool | None, commands=("contacts get-contacts",), digest="sha256:h",
                          rules_digest=None, failure="Tool count mismatch: agent={'X': 1}, oracle={'X': 2}") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    result = {"scenario_id": path.name, "success": success, "harness_digest": digest, "agent_response": "done"}
    if rules_digest:
        result["env_rules_digest"] = rules_digest
    if success is False:
        result["failure_reasons"] = [failure]
    (path / "result.json").write_text(json.dumps(result))
    (path / "harness_applied.json").write_text(json.dumps({"digest": digest}))
    messages = [
        {"role": "system", "content": "You are a personal assistant running inside OpenClaw."},
        {"role": "user", "content": "Sender (untrusted metadata):\n```json\n{}\n```\n\n[Tue 2024-10-15 07:00 UTC] Book a ride."},
    ]
    for i, command in enumerate(commands):
        messages.append({"role": "assistant", "content": f"thinking {i}",
                         "tool_calls": [{"id": f"c{i}", "type": "function",
                                         "function": {"name": "exec", "arguments": json.dumps({"command": command})}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"output {i}"})
    entry = {"type": "llm_call", "http_status": 200, "request": {"model": "m", "messages": messages},
             "raw_response": json.dumps({"choices": [{"message": {"role": "assistant", "content": "All done."}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 2}})}
    (path / "trace.jsonl").write_text(json.dumps(entry) + "\n")
    (path / "daemon_judgments.jsonl").write_text("ORACLE-SECRET oracle_events match_details\n")
    return path


class FakeGaia2Runner:
    """Stands in for Gaia2Runner: success is decided by a callback per (scenario, rules, rep)."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.rollouts = 0
        self.calls = []

    def run_batch(self, cases, *, harness_dir, env_rules_code, k, out_dir):
        import subprocess
        import sys

        digest = subprocess.run([sys.executable, str(HARNESS_TOOL), "digest", str(harness_dir)],
                                text=True, capture_output=True, check=True).stdout.strip()
        self.calls.append({"cases": [c.scenario_id for c in cases], "harness_dir": str(harness_dir),
                           "digest": digest, "rules": env_rules_code, "k": k})
        self.rollouts += len(cases) * k
        results = []
        for case in cases:
            for rep in range(k):
                success = self.outcome(case.scenario_id, env_rules_code, rep)
                path = write_scenario_output(Path(out_dir) / f"run_{rep + 1}" / case.capability / case.scenario_id,
                                             success=success, digest=digest)
                results.append(ScenarioResult(case.scenario_id, rep,
                                              "success" if success else "task_failure", success, path,
                                              digest, None, ""))
        return results


def cases(tmp_path: Path, n: int) -> list[CaseRef]:
    out = []
    for i in range(n):
        source = tmp_path / "data" / "search" / f"scenario_universe_29_t{i}.json"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("{}")
        from envharness_gaia2.gaia2_run import sha256_file

        out.append(CaseRef(f"scenario_universe_29_t{i}", "search", source, sha256_file(source)))
    return out
