from __future__ import annotations

import json
import threading
import time
import zlib
from pathlib import Path

import pytest
import yaml
from envharness.reasoning_bank.bank import MemoryItem

from envharness_gaia2.coevolve import Pipeline
from envharness_gaia2.config import ConfigError, RetrievalConfig, load_config
from envharness_gaia2.evaluate import _Job, run_jobs
from envharness_gaia2.gaia2_run import CaseRef, ScenarioResult
from envharness_gaia2.retrieval import Retriever, selection_key, task_query

from helpers import FakeGaia2Runner
from test_coevolve import _config, _designer, _inducer, _outcome


def _embed(texts):
    """Deterministic 4-d vectors that differ per text."""
    out = []
    for text in texts:
        h = zlib.crc32(text.encode())
        out.append([1.0 + (h & 0xFF) / 64, ((h >> 8) & 0xFF) / 64, ((h >> 16) & 0xFF) / 64, ((h >> 24) & 0xFF) / 64])
    return out


def _scenario(tmp_path: Path, name: str, message: str, *, extra_user: bool = False) -> CaseRef:
    events = [
        {"event_type": "USER", "dependencies": [], "action": {"app": "AgentUserInterface",
         "function": "send_message_to_agent", "args": [{"name": "content", "value": message}]}},
        {"class_name": "OracleEvent", "event_type": "AGENT", "dependencies": ["x"],
         "action": {"function": "order_ride", "args": [{"name": "start_location", "value": "ORACLE"}]}},
    ]
    if extra_user:
        events.append({"event_type": "USER", "dependencies": ["y"], "action": {
            "function": "send_message_to_agent", "args": [{"name": "content", "value": "later turn"}]}})
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"events": events}))
    return CaseRef(name, "search", path, "sha256:x")


def test_task_query_is_the_first_user_request_only(tmp_path):
    case = _scenario(tmp_path, "s1", "  Book me a cab. " + "x" * 3000, extra_user=True)
    query = task_query(case.source)
    assert query.startswith("Book me a cab.") and len(query) == 2000
    assert "ORACLE" not in query and "later turn" not in query
    (tmp_path / "bad.json").write_text(json.dumps({"events": []}))
    with pytest.raises(ValueError):
        task_query(tmp_path / "bad.json")


def test_retriever_uses_bank_retrieve_and_caches_queries(tmp_path):
    bank = [MemoryItem(title=f"T{i}", description="d", content="c", embedding=v)
            for i, v in enumerate(_embed([f"item {i}" for i in range(8)]))]
    calls = []

    def embed(texts):
        calls.append(texts)
        return _embed(texts)

    case = _scenario(tmp_path, "s1", "Find the cheapest flat.")
    cache = tmp_path / "q.json"
    for mode in ("cosine", "mmr"):
        retriever = Retriever(RetrievalConfig(3, mode, 0.5), embed, cache)
        chosen = retriever.select(bank, case)
        assert len(chosen) == 3 and len({it.title for it in chosen}) == 3
        assert chosen == retriever.select(bank, case)
    assert len(calls) == 1  # the query was embedded once; the cache file served the rest
    assert json.loads(cache.read_text())["s1"]["embedding"] == _embed(["Find the cheapest flat."])[0]
    assert Retriever(RetrievalConfig(3, "mmr", 0.5), embed, cache).select([], case) == []
    assert selection_key(bank[:2]) == selection_key(bank[:2]) != selection_key(bank[1:3])


def test_config_retrieval_section_is_optional_and_validated(tmp_path):
    cfg = load_config(_config(tmp_path), verify_image=False)
    assert cfg.retrieval is None
    base = yaml.safe_load(Path(_config(tmp_path)).read_text())
    for value, error in (({"top_k": 5, "mode": "mmr", "mmr_lambda": 0.5}, None),
                         ({"top_k": 0, "mode": "mmr", "mmr_lambda": 0.5}, ConfigError),
                         ({"top_k": 5, "mode": "bm25", "mmr_lambda": 0.5}, ConfigError),
                         ({"top_k": 5, "mode": "mmr"}, ConfigError)):
        raw = dict(base, retrieval=value)
        path = tmp_path / "r.yaml"
        path.write_text(yaml.safe_dump(raw))
        if error:
            with pytest.raises(error):
                load_config(path, verify_image=False)
        else:
            assert load_config(path, verify_image=False).retrieval == RetrievalConfig(5, "mmr", 0.5)


def test_run_jobs_caps_containers_across_parallel_calls(tmp_path):
    lock, state = threading.Lock(), {"now": 0, "peak": 0, "concurrency": []}

    class Runner:
        def run_batch(self, cases, *, harness_dir, env_rules_code, k, out_dir, concurrency=None, setup_commands=()):
            with lock:
                state["now"] += concurrency  # containers this call may run at once
                state["peak"] = max(state["peak"], state["now"])
                state["concurrency"].append(concurrency)
            time.sleep(0.02)
            with lock:
                state["now"] -= concurrency
            return [ScenarioResult(c.scenario_id, r, "success", True, None, "d", None, "") for c in cases
                    for r in range(k)]

    jobs = [_Job("a", tmp_path / f"h{i}", tuple(_scenario(tmp_path, f"s{i}_{j}", "q") for j in range(n)),
                 tmp_path / f"o{i}") for i, n in enumerate([1, 1, 3, 1, 9, 1, 2, 1, 1, 1])]
    out = run_jobs(Runner(), jobs, k=1, max_concurrent=5)
    assert [len(r) for r in out] == [len(j.cases) for j in jobs]
    assert state["peak"] <= 5 and max(state["concurrency"]) == 5 and min(state["concurrency"]) == 1


def test_pipeline_with_retrieval(tmp_path):
    path = _config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["retrieval"] = {"top_k": 2, "mode": "mmr", "mmr_lambda": 0.5}
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path, verify_image=False)
    run_dir = cfg.run_root / "ret"
    run_dir.mkdir(parents=True)
    runner, log = FakeGaia2Runner(_outcome), []
    result = Pipeline(cfg, run_dir, gaia2=runner, designer_client=_designer(), induce_client=_inducer(log),
                      embedder=_embed).run()

    seed_digest = (run_dir / "seed_harness.digest").read_text().strip()
    bank1 = [json.loads(line)["title"] for line in (run_dir / "banks" / "ours_r1.jsonl").read_text().splitlines()]
    bank2 = [json.loads(line)["title"] for line in (run_dir / "banks" / "ours_r2.jsonl").read_text().splitlines()]
    assert len(bank1) > 2
    assert not (run_dir / "rounds" / "r1" / "harness").exists()  # no whole-bank render in retrieval mode
    for task_dir in sorted((run_dir / "rounds" / "r1" / "corpus").glob("task_*")):
        record = json.loads((task_dir / "retrieval.json").read_text())
        assert record["skills"] == [] and record["harness_digest"] == seed_digest  # round 1: empty bank
    r2 = sorted((run_dir / "rounds" / "r2" / "corpus").glob("task_*"))
    for task_dir in r2:
        record = json.loads((task_dir / "retrieval.json").read_text())
        assert len(record["skills"]) == 2 and set(record["skills"]) <= set(bank1)
        assert {c["digest"] for c in runner.calls if c["harness_dir"] == record["harness_dir"]} == {record["harness_digest"]}
        names = sorted(p.parent.name for p in Path(record["harness_dir"]).glob("skills/*/SKILL.md"))
        assert len(names) == 2
    lineage = [json.loads(line) for line in (run_dir / "lineage.jsonl").read_text().splitlines()]
    assert lineage[1]["input_bank_sha256"] == lineage[0]["output_bank_sha256"]

    summary = result["eval"]
    assert set(summary) == {"nobank", "orig", "ours_r1", "ours_r2"}
    assert all(arm["valid"] == len(cfg.dev_cases) and arm["digest_ok"] for arm in summary.values())
    assert summary["ours_r2"]["n_skills"] == len(bank2)
    assert all(len(t) == 2 and set(t) <= set(bank2) for t in summary["ours_r2"]["case_skills"].values())
    assert all(t == [] for t in summary["nobank"]["case_skills"].values())
    assert result["rollouts"] + result["eval_rollouts"] == sum(c["k"] * len(c["cases"]) for c in runner.calls)

    # resume: corpus is committed and query vectors are cached -> same selections
    before = len(runner.calls)
    again = Pipeline(cfg, run_dir, gaia2=runner, designer_client=_designer(), induce_client=_inducer([]),
                     embedder=lambda texts: pytest.fail("query re-embedded on resume")).run()
    assert again["eval"]["ours_r2"]["case_skills"] == summary["ours_r2"]["case_skills"]
    assert all(c["rules"] is None for c in runner.calls[before:])  # only eval reran (fake runner has no cache)


def test_report_checks_per_scenario_injection(tmp_path, monkeypatch):
    from envharness_gaia2 import report as report_mod

    path = _config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["retrieval"] = {"top_k": 2, "mode": "mmr", "mmr_lambda": 0.5}
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path, verify_image=False)
    run_dir = cfg.run_root / "rep"
    run_dir.mkdir(parents=True)
    (run_dir / "config.resolved.json").write_text(json.dumps({"config": cfg.raw}))
    Pipeline(cfg, run_dir, gaia2=FakeGaia2Runner(_outcome), designer_client=_designer(), induce_client=_inducer([]),
             embedder=_embed).run()
    real_run = report_mod.subprocess.run
    monkeypatch.setattr(report_mod.subprocess, "run", lambda argv, *a, **k: (
        report_mod.subprocess.CompletedProcess(argv, 0, "", "") if argv[0] == "sg" else real_run(argv, *a, **k)))
    text = report_mod.Report(run_dir, {}).write().read_text()
    assert "## 5. Skill banks" in text and text.split("## 5. Skill banks")[1].split("\n")[0].endswith("PASS")
    assert "retrieval top_k=2 (mmr" in text and "bad: none" in text
    assert "r2 corpus: 6 tasks ran with the top-2 of r1's bank" in text and "problems: none" in text
    assert "2-2 skills per scenario" in text


def test_prompt_injection_pastes_the_original_memory_block(tmp_path):
    from envharness.prompts.alfworld_skill_prompt import build_memory_block

    from envharness_gaia2.skills import TEMPLATE, injected_titles, render
    from helpers import HARNESS_TOOL, POLICY, SEED_HARNESS

    items = [MemoryItem(title=f"Check {{{{x}}}} item {i}", description=f"when {i}", content=f"do {i}", embedding=[1.0]) for i in (2, 1)]
    digest = render(items, seed_dir=SEED_HARNESS, out_dir=tmp_path / "h", policy=POLICY, harness_tool=HARNESS_TOOL,
                    injection="prompt")
    text = (tmp_path / "h" / TEMPLATE).read_text()
    seed = (SEED_HARNESS / TEMPLATE).read_text()
    block = build_memory_block(items, style="soft")
    assert text.startswith(seed.rstrip("\n")) and "Below are some memory items" in text
    assert "explicitly state whether any insight applies" in text  # the original per-step gate
    assert text.index("item 2") < text.index("item 1")  # retrieval order, as the original
    assert "{{x}}" not in text and block.replace("{{", "{ {").replace("}}", "} }") in text  # no template placeholders
    assert not list((tmp_path / "h").glob("skills/*/SKILL.md")) and digest.startswith("sha256:")
    assert injected_titles(tmp_path / "h") == ["Check { {x} } item 2", "Check { {x} } item 1"]
    # no items: the seed harness unchanged (same digest as the skill_md seed)
    empty = render([], seed_dir=SEED_HARNESS, out_dir=tmp_path / "e", policy=POLICY, harness_tool=HARNESS_TOOL,
                   injection="prompt")
    assert empty == render([], seed_dir=SEED_HARNESS, out_dir=tmp_path / "e2", policy=POLICY, harness_tool=HARNESS_TOOL)


def test_injection_and_evaluation_options_are_validated(tmp_path):
    base = yaml.safe_load(Path(_config(tmp_path)).read_text())
    ret = {"top_k": 5, "mode": "mmr", "mmr_lambda": 0.5}
    for extra, error in (({"retrieval": ret, "injection": {"mode": "prompt", "style": "soft"}}, None),
                         ({"injection": {"mode": "prompt", "style": "soft"}}, ConfigError),  # prompt needs retrieval
                         ({"retrieval": ret, "injection": {"mode": "paste", "style": "soft"}}, ConfigError),
                         ({"evaluation": {"orig_arm": "no"}}, ConfigError)):
        path = tmp_path / "o.yaml"
        path.write_text(yaml.safe_dump(dict(base, **extra)))
        if error:
            with pytest.raises(error):
                load_config(path, verify_image=False)
        else:
            cfg = load_config(path, verify_image=False)
            assert cfg.injection.mode == "prompt" and cfg.orig_arm is True
    assert load_config(_config(tmp_path), verify_image=False).injection.mode == "skill_md"


def test_pipeline_prompt_injection_without_orig_arm(tmp_path):
    from envharness_gaia2.skills import TEMPLATE, injected_titles

    path = _config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw.update(retrieval={"top_k": 2, "mode": "mmr", "mmr_lambda": 0.5},
               injection={"mode": "prompt", "style": "soft"}, evaluation={"orig_arm": False})
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path, verify_image=False)
    run_dir = cfg.run_root / "prompt"
    run_dir.mkdir(parents=True)
    (run_dir / "config.resolved.json").write_text(json.dumps({"config": cfg.raw}))
    runner = FakeGaia2Runner(_outcome)
    result = Pipeline(cfg, run_dir, gaia2=runner, designer_client=_designer(), induce_client=_inducer([]),
                      embedder=_embed).run()
    assert set(result["eval"]) == {"nobank", "ours_r1", "ours_r2"}  # control = nobank only
    assert not (run_dir / "banks" / "orig.jsonl").exists()
    bank1 = [json.loads(line)["title"] for line in (run_dir / "banks" / "ours_r1.jsonl").read_text().splitlines()]
    for task_dir in sorted((run_dir / "rounds" / "r2" / "corpus").glob("task_*")):
        record = json.loads((task_dir / "retrieval.json").read_text())
        harness = Path(record["harness_dir"])
        assert injected_titles(harness) == record["skills"] and set(record["skills"]) <= set(bank1)
        assert "Below are some memory items" in (harness / TEMPLATE).read_text()
        assert not list(harness.glob("skills/*/SKILL.md"))
    for sid, titles in result["eval"]["ours_r2"]["case_skills"].items():
        assert injected_titles(Path(result["eval"]["ours_r2"]["case_harness"][sid])) == titles


def test_report_checks_prompt_injection(tmp_path, monkeypatch):
    from envharness_gaia2 import report as report_mod

    path = _config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw.update(retrieval={"top_k": 2, "mode": "mmr", "mmr_lambda": 0.5},
               injection={"mode": "prompt", "style": "soft"}, evaluation={"orig_arm": False})
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path, verify_image=False)
    run_dir = cfg.run_root / "rep"
    run_dir.mkdir(parents=True)
    (run_dir / "config.resolved.json").write_text(json.dumps({"config": cfg.raw}))
    Pipeline(cfg, run_dir, gaia2=FakeGaia2Runner(_outcome), designer_client=_designer(), induce_client=_inducer([]),
             embedder=_embed).run()
    real_run = report_mod.subprocess.run
    monkeypatch.setattr(report_mod.subprocess, "run", lambda argv, *a, **k: (
        report_mod.subprocess.CompletedProcess(argv, 0, "", "") if argv[0] == "sg" else real_run(argv, *a, **k)))
    text = report_mod.Report(run_dir, {}).write().read_text()
    for section in ("## 5. Skill banks", "## 9. Evaluation"):
        assert text.split(section)[1].split("\n")[0].endswith("PASS"), section
    assert "injection prompt: " in text and "bad: none" in text
