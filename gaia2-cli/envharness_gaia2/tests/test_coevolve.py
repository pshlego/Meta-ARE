from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from envharness.infra.llm import ChatResponse, ScriptedClient, ToolCall

from envharness_gaia2.coevolve import Pipeline
from envharness_gaia2.config import ConfigError, load_config

from helpers import FakeGaia2Runner

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke_cloudgpt.yaml"
RULES = "class _Rules(Rules):\n    def filter_action(self, action, env_state):\n        return action\n"


def _config(tmp_path: Path, **overrides) -> Path:
    raw = yaml.safe_load(CONFIG.read_text())
    raw["storage"]["run_root"] = str(tmp_path / "runs")
    raw["task_agent"]["azure_config_dir"] = None
    raw["envrigger"]["k_per_candidate"] = 3
    raw.pop("retrieval", None)  # whole-bank mode; tests/test_retrieval.py adds the section back
    for dotted, value in overrides.items():
        section, key = dotted.split(".")
        raw[section][key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _designer():
    def responder(messages, tools):
        name = tools[0]["function"]["name"]
        if name == "propose_candidate":
            return ChatResponse(content="", tool_calls=[ToolCall(id="p", name=name, arguments={
                "rationale": "refuse the first contacts call", "rules_code": RULES})])
        return ChatResponse(content="", tool_calls=[ToolCall(id="d", name=name, arguments={
            "decision": "accept", "rationale": "in band", "failure_analysis": {"primary_axis": "A", "label": "retry"}})])

    return ScriptedClient(responder=responder)


def _inducer(log):
    def responder(messages, tools):
        log.append(messages[1].content)
        n = len(log)
        return ChatResponse(content=f"# Memory Item 1\n## Title Lesson {n}\n## Description When booking.\n## Content Run `contacts search-contacts` {n}.")

    return ScriptedClient(responder=responder)


def _outcome(sid, rules, rep):
    # baseline: rep 0 succeeds only (sr 1/3 -> below band); rules: 2 of 3 succeed.
    return rep < 2 if rules else rep == 0


def _pipeline(cfg, run_dir, runner, log):
    return Pipeline(cfg, run_dir, gaia2=runner, designer_client=_designer(), induce_client=_inducer(log),
                    embedder=lambda texts: [[1.0, 0.0] for _ in texts])


@pytest.mark.parametrize("mode", ["alfworld_pair", "swebench_cascade"])
def test_two_rounds_resume_and_eval(tmp_path, mode):
    cfg = load_config(_config(tmp_path, **{"bank.mode": mode}), verify_image=False)
    run_dir = cfg.run_root / "r1"
    run_dir.mkdir(parents=True)
    runner, log = FakeGaia2Runner(_outcome), []
    result = _pipeline(cfg, run_dir, runner, log).run()

    lineage = [json.loads(line) for line in (run_dir / "lineage.jsonl").read_text().splitlines()]
    assert [r["round"] for r in lineage] == [1, 2] and result["stopped"] is None
    n_tasks = len(cfg.train_cases)
    assert all((run_dir / "rounds" / f"r{r}" / "corpus" / f"task_{t:03d}" / "DONE").is_file()
               for r in (1, 2) for t in range(n_tasks))
    # round 2's students carry bank_1 (baseline harness digest == r1 harness digest)
    r1_digest = (run_dir / "rounds" / "r1" / "harness.digest").read_text().strip()
    assert lineage[1]["input_harness_digest"] == r1_digest == lineage[0]["output_harness_digest"]
    r1_harness = str(run_dir / "rounds" / "r1" / "harness")
    corpus_r2 = [c for c in runner.calls if c["harness_dir"] == r1_harness]
    assert len(corpus_r2) >= n_tasks + 1 and {c["digest"] for c in corpus_r2} == {r1_digest}
    # augment mode: round 2 induction sees round 1's titles
    assert any("EXISTING LESSONS" in u and "Lesson 1" in u for u in log)
    bank1 = (run_dir / "banks" / "ours_r1.jsonl").read_text().splitlines()
    bank2 = (run_dir / "banks" / "ours_r2.jsonl").read_text().splitlines()
    assert len(bank2) > len(bank1) > 0 and (run_dir / "banks" / "orig.jsonl").is_file()
    assert len(list((run_dir / "rounds" / "r2" / "harness").glob("skills/*/SKILL.md"))) == len(bank2)
    # eval arms
    summary = result["eval"]
    assert set(summary) == {"nobank", "orig", "ours_r1", "ours_r2"}
    assert all(arm["valid"] == len(cfg.dev_cases) and arm["digest_ok"] for arm in summary.values())
    assert summary["ours_r2"]["n_skills"] == len(bank2) and summary["nobank"]["n_skills"] == 0
    assert result["rollouts"] == sum(c["k"] * len(c["cases"]) for c in runner.calls) - result["eval_rollouts"]

    # resume: everything is committed, nothing reruns
    calls_before = len(runner.calls)
    again = Pipeline(cfg, run_dir, gaia2=runner, designer_client=_designer(), induce_client=_inducer([]),
                     embedder=lambda texts: [[1.0, 0.0] for _ in texts]).run()
    assert len(runner.calls) - calls_before == len(summary)  # eval reuses runner caching only in the real runner
    assert again["rollouts"] == result["rollouts"]


def test_budget_stops_between_tasks(tmp_path):
    cfg = load_config(_config(tmp_path, **{"budget.max_rollouts": 7}), verify_image=False)
    run_dir = cfg.run_root / "r2"
    run_dir.mkdir(parents=True)
    runner = FakeGaia2Runner(_outcome)
    result = _pipeline(cfg, run_dir, runner, []).run()
    assert result["stopped"].startswith("budget (round 1")
    assert len(list((run_dir / "rounds" / "r1" / "corpus").glob("task_*/DONE"))) == 2  # 6 rollouts each (3 + 3)
    # a partial round is not committed: no bank from it, only the seed is evaluated
    assert not (run_dir / "rounds" / "r1" / "DONE").exists() and not (run_dir / "banks").exists()
    assert not (run_dir / "rounds" / "r2").exists() and set(result["eval"]) == {"nobank"}

    # --resume with a raised cap finishes round 1 (the two committed tasks are not rerun) and round 2
    bigger = load_config(_config(tmp_path, **{"budget.max_rollouts": 200}), verify_image=False)
    again = _pipeline(bigger, run_dir, runner, []).run()
    assert again["stopped"] is None and set(again["eval"]) == {"nobank", "orig", "ours_r1", "ours_r2"}
    assert again["rollouts"] == 2 * len(cfg.train_cases) * 6
    assert json.loads((run_dir / "budget.json").read_text())["max_rollouts"] == 200


def test_runner_gets_the_skillbank_policy(tmp_path):
    from envharness_gaia2.gaia2_run import Gaia2Runner

    cfg = load_config(_config(tmp_path), verify_image=False)
    assert cfg.run.harness_policy == cfg.harness_policy
    toml = Gaia2Runner(cfg.run, harness_digest_fn=str, label_prefix="t")._toml(
        tmp_path / "d", tmp_path / "o", tmp_path / "h", None, 1)
    assert f'harness_policy = "{cfg.harness_policy}"' in toml


def test_config_is_strict(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    raw["run"]["max_concurrent"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="max_concurrent"):
        load_config(path, verify_image=False)
    for section, key, value in (("envrigger", "bogus", 1), ("envrigger", "target_band", [0.5, 0.5]),
                                ("coevolution", "rounds", 0)):
        raw = yaml.safe_load(CONFIG.read_text())
        raw[section][key] = value
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ConfigError):
            load_config(path, verify_image=False)


def test_report_runs_on_a_finished_run(tmp_path, monkeypatch):
    from envharness_gaia2 import report as report_mod

    cfg = load_config(_config(tmp_path), verify_image=False)
    run_dir = cfg.run_root / "r3"
    run_dir.mkdir(parents=True)
    (run_dir / "config.resolved.json").write_text(json.dumps({"config": cfg.raw}))
    _pipeline(cfg, run_dir, FakeGaia2Runner(_outcome), []).run()
    real_run = report_mod.subprocess.run
    monkeypatch.setattr(report_mod.subprocess, "run", lambda argv, *a, **k: (
        report_mod.subprocess.CompletedProcess(argv, 0, "", "") if argv[0] == "sg" else real_run(argv, *a, **k)))
    text = report_mod.Report(run_dir, {}).write().read_text()
    for title in ("5. Skill banks", "8. Budget", "9. Evaluation"):
        assert f"## {title}" in text and text.split(f"## {title}")[1].split("\n")[0].endswith("PASS"), title
    assert "corpus ran on r1's bank harness" in text


def test_corpus_task_without_traces_is_retried(tmp_path, monkeypatch):
    from envharness_gaia2 import coevolve

    cfg = load_config(_config(tmp_path, **{"coevolution.rounds": 1}), verify_image=False)
    run_dir = cfg.run_root / "retry"
    run_dir.mkdir(parents=True)
    real_run, seen = coevolve.Gaia2Orchestrator.run, []

    def flaky(self):
        seen.append(self.config.explicit_task_ids[0])
        if seen.count(self.config.explicit_task_ids[0]) == 1:
            return None  # an aborted task: no traces.jsonl
        return real_run(self)

    monkeypatch.setattr(coevolve.Gaia2Orchestrator, "run", flaky)
    _pipeline(cfg, run_dir, FakeGaia2Runner(_outcome), []).run()
    assert seen == [t for t in range(len(cfg.train_cases)) for _ in range(2)]  # experiments/swebench CORPUS_RETRIES


def test_eval_success_rate_counts_errors_as_failures(tmp_path):
    from envharness_gaia2.evaluate import Arm, evaluate
    from envharness_gaia2.gaia2_run import ScenarioResult

    class Runner:
        def run_batch(self, cases, *, harness_dir, env_rules_code, k, out_dir, concurrency=None, setup_commands=()):
            return [ScenarioResult(c.scenario_id, 0, "execution_error" if i == 0 else "success", None if i == 0 else True,
                                   None, "d", None, "") for i, c in enumerate(cases)]

    from helpers import cases as make_cases

    summary = evaluate(Runner(), make_cases(tmp_path, 4), arms={"a": Arm(lambda c: (tmp_path, []), 0)}, k=1,
                       out_dir=tmp_path / "eval", digest_fn=lambda d: "d", max_concurrent=2)
    assert summary["a"]["valid"] == 3 and summary["a"]["attempts"] == 4 and summary["a"]["success_rate"] == 0.75
