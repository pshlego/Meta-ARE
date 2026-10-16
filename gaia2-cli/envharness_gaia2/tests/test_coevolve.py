from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from envharness.infra.llm import ChatResponse, ScriptedClient, ToolCall

from envharness_gaia2.coevolve import CoEvolution, epoch_batches
from envharness_gaia2.config import ConfigError, load_config

from helpers import FakeGaia2Runner

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke_cloudgpt.yaml"
RULES = "class _Rules(Rules):\n    def filter_action(self, action, env_state):\n        return action\n"


def _config(tmp_path: Path, **budget) -> Path:
    raw = yaml.safe_load(CONFIG.read_text())
    raw["storage"]["run_root"] = str(tmp_path / "runs")
    raw["task_agent"]["azure_config_dir"] = None
    raw["coevolution"]["rounds"] = 2
    raw["coevolution"]["task_selection"]["batch_size"] = 3
    raw["budget"].update(budget)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _designer():
    def responder(messages, tools):
        name = tools[0]["function"]["name"]
        if name == "propose_candidate":
            return ChatResponse(content="", tool_calls=[ToolCall(id="p", name=name, arguments={"rationale": "friction", "rules_code": RULES})])
        return ChatResponse(content="", tool_calls=[ToolCall(id="d", name=name, arguments={"decision": "accept", "rationale": "ok"})])

    return ScriptedClient(responder=responder)


def _inducer():
    counter = {"n": 0}

    def responder(messages, tools):
        counter["n"] += 1
        n = counter["n"]
        return ChatResponse(content=f"# Memory Item 1\n## Title Verify before writing {n}\n## Description When changing state.\n## Content Re-read the record first {n}.")

    return ScriptedClient(responder=responder)


def _outcome(sid, rules, rep):
    # baseline fails on every task except one success; rules lift 2/3.
    return (rep < 2) if rules else (rep == 0 and sid.endswith("0wlkxc"))


def test_two_rounds_lineage_resume_and_budget(tmp_path):
    cfg = load_config(_config(tmp_path, max_rollouts=1000, max_iterations=3), verify_image=False)
    run_dir = cfg.run_root / "r1"
    run_dir.mkdir(parents=True)
    runner = FakeGaia2Runner(_outcome)
    result = CoEvolution(cfg, run_dir, gaia2=runner, designer_client=_designer(), induce_client=_inducer()).run()

    assert result["stopped"] == "budget" and result["batches"] == 3
    lineage = [json.loads(l) for l in (run_dir / "lineage.jsonl").read_text().splitlines()]
    assert [(r["round"], r["batch"]) for r in lineage] == [(1, 1), (1, 2), (2, 1)]
    # Each batch's baselines ran on the previous batch's snapshot.
    seed_digest = result["seed_digest"]
    assert lineage[0]["input_harness_digest"] == seed_digest
    assert lineage[1]["input_harness_digest"] == lineage[0]["skills_digest"] != seed_digest
    assert lineage[2]["input_harness_digest"] == lineage[1]["skills_digest"]
    baseline_digests = [c["digest"] for c in runner.calls if c["rules"] is None]
    assert baseline_digests[0] == seed_digest and lineage[0]["skills_digest"] in baseline_digests
    assert all(1 <= len(json.loads((run_dir / "rounds" / f"r{r['round']}" / f"b{r['batch']}" / "selected.json").read_text())) <= 3
               for r in lineage)
    harness = Path(result["harness_dir"])
    assert (harness / "skills").is_dir() and len(list(harness.glob("skills/*/SKILL.md"))) == lineage[-1]["n_skills"]
    assert result["final_eval"]["seed"]["valid"] == 1 and result["final_eval"]["final"]["valid"] == 1
    assert result["rollouts"] == sum(r["rollouts"] for r in lineage)
    assert (run_dir / "traces.jsonl").read_text().count('"kind":"baseline"') == 3 * 3 * 3

    # Resume: finished batches are skipped; raising the cap continues with the remaining one.
    cfg2 = load_config(_config(tmp_path, max_rollouts=1000, max_iterations=5), verify_image=False)
    runner2 = FakeGaia2Runner(_outcome)
    result2 = CoEvolution(cfg2, run_dir, gaia2=runner2, designer_client=_designer(), induce_client=_inducer()).run()
    assert result2["batches"] == 4 and result2["stopped"] == "completed"
    assert {tuple(c["cases"]) for c in runner2.calls if c["rules"] is None} <= {
        (cfg2.train_cases[t].scenario_id,) for t in epoch_batches(6, 3, 44)[1]
    } | {tuple(c.scenario_id for c in cfg2.dev_cases)}


def test_consolidation_when_bank_exceeds_cap(tmp_path):
    raw = yaml.safe_load(_config(tmp_path, max_rollouts=1000, max_iterations=2).read_text())
    raw["coevolution"]["max_skills"] = 2
    path = tmp_path / "c2.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path, verify_image=False)
    run_dir = cfg.run_root / "r2"
    run_dir.mkdir(parents=True)

    def inducer(messages, tools):
        return ChatResponse(content="\n".join(
            f"# Memory Item {i}\n## Title Skill {i} {len(messages[1].content)}\n## Description d\n## Content c {i}" for i in range(1, 4)))

    result = CoEvolution(cfg, run_dir, gaia2=FakeGaia2Runner(_outcome), designer_client=_designer(),
                         induce_client=ScriptedClient(responder=inducer)).run()
    lineage = [json.loads(l) for l in (run_dir / "lineage.jsonl").read_text().splitlines()]
    assert any(r["consolidated"] for r in lineage)
    assert all(r["n_skills"] <= 2 for r in lineage)
    assert result["bank_file"].startswith("bank_consolidated_")


def test_config_is_strict(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    raw["run"]["max_concurrent"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="max_concurrent"):
        load_config(path, verify_image=False)
    raw = yaml.safe_load(CONFIG.read_text())
    raw["envrigger"]["bogus"] = 1
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):
        load_config(path, verify_image=False)


def test_report_runs_on_a_finished_run(tmp_path, monkeypatch):
    from envharness_gaia2 import report as report_mod

    cfg = load_config(_config(tmp_path, max_rollouts=1000, max_iterations=2), verify_image=False)
    run_dir = cfg.run_root / "r3"
    run_dir.mkdir(parents=True)
    (run_dir / "config.resolved.json").write_text(json.dumps({"config": cfg.raw}))
    CoEvolution(cfg, run_dir, gaia2=FakeGaia2Runner(_outcome), designer_client=_designer(),
                induce_client=_inducer()).run()
    monkeypatch.setattr(report_mod.subprocess, "run", _no_docker(report_mod.subprocess.run))
    path = report_mod.Report(run_dir, {}).write()
    text = path.read_text()
    assert "## 5. Skill bank updates — PASS" in text
    # Fakes carry no OpenClaw system prompt, so only the digest chain of check 6 applies here.
    assert "r1b2 baseline harness_digest == r1b1 snapshot: True" in text
    assert "## 8. Budget — PASS" in text


def _no_docker(real_run):
    import subprocess

    def run(argv, *args, **kwargs):
        if argv and argv[0] == "sg":
            return subprocess.CompletedProcess(argv, 0, "", "")
        return real_run(argv, *args, **kwargs)

    return run
