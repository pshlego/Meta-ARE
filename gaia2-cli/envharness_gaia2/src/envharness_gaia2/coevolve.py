"""Co-evolve an OpenClaw skill bank with EnvHarness environment variations on GAIA2.

    uv run python -m envharness_gaia2.coevolve --config configs/smoke_cloudgpt.yaml --run-id <id> [--resume]

For each round, the training tasks are shuffled into batches. Per batch:
EnvRigger (baseline K -> propose rules -> K rollouts -> decide, up to max_k)
runs task by task with the current harness snapshot (seed + rendered skills);
skills are induced from the batch's traces (cascade paired-diff), up to
``max_new_skills_per_batch`` are added to the bank (consolidated above
``max_skills``), and the bank is rendered into the next snapshot. A batch is
committed with a DONE marker, so ``--resume`` continues after the last
finished batch. Finally seed and final harnesses are evaluated on dev.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from envharness.agents.harness_agent import NoopHarnessAgent
from envharness.orchestration.budget import CappedAdaptive
from envharness.orchestration.objectives import DifficultyZone
from envharness.orchestration.orchestrator import OrchestratorConfig
from envharness.orchestration.runner import EnvSpec, PolicySpec
from envharness.orchestration.storage import TraceStore

from envharness_gaia2 import PROMPTS_DIR
from envharness_gaia2.config import Config, load_config
from envharness_gaia2.designer import Gaia2HarnessAgent, make_client
from envharness_gaia2.evaluate import evaluate
from envharness_gaia2.gaia2_run import Gaia2Runner
from envharness_gaia2.induce import induce, select_units
from envharness_gaia2.orchestrator import ENV_IMPORT_PATH, ENV_STATE_SCHEMA, EXEC_TOOL_SCHEMA, Gaia2Orchestrator
from envharness_gaia2.skills import (
    SkillRenderError,
    consolidate,
    load_bank,
    render,
    save_bank,
    select_new_skills,
)

TASK_DESCRIPTION = (
    "GAIA2 personal-assistant scenario solved by OpenClaw (fixed harness; only its skills evolve). "
    "The user request is the 'Task:' line at the start of each trajectory.\n\n"
)


def harness_tool_path(cfg: Config) -> Path:
    return cfg.meta_are_repo / "gaia2-cli" / "shared" / "harness_tool.py"


def harness_digest(cfg: Config, harness_dir: Path) -> str:
    return subprocess.run(
        [sys.executable, str(harness_tool_path(cfg)), "digest", str(harness_dir)],
        text=True, capture_output=True, check=True,
    ).stdout.strip()


def epoch_batches(n: int, batch_size: int, seed: int) -> list[list[int]]:
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


class CoEvolution:
    def __init__(self, cfg: Config, run_dir: Path, *, gaia2: Gaia2Runner | None = None,
                 designer_client=None, induce_client=None) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.gaia2 = gaia2 or Gaia2Runner(
            cfg.run, harness_digest_fn=lambda d: harness_digest(cfg, d), label_prefix=f"envh.{run_dir.name}"
        )
        self.designer_client = designer_client or make_client(
            cfg.optimizer, log_path=run_dir / "designer_calls.jsonl", role="designer"
        )
        self.induce_client = induce_client or make_client(
            cfg.optimizer, log_path=run_dir / "induce_calls.jsonl", role="induce"
        )
        self.state_path = run_dir / "budget.json"
        self.state = self._load_state()

    # -- state -----------------------------------------------------------------
    def _load_state(self) -> dict:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"rollouts": 0, "batches": 0, "bank_file": "bank.jsonl", "harness_dir": None,
                "max_rollouts": self.cfg.max_rollouts, "max_iterations": self.cfg.max_iterations}

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

    def _exhausted(self) -> bool:
        return self.state["batches"] >= self.cfg.max_iterations or self.state["rollouts"] >= self.cfg.max_rollouts

    def _log_lineage(self, record: dict) -> None:
        with open(self.run_dir / "lineage.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    # -- main loop -------------------------------------------------------------
    def run(self) -> dict:
        cfg = self.cfg
        seed_snapshot = self.run_dir / "seed_harness"
        if self.state["harness_dir"] is None:
            digest = render([], seed_dir=cfg.seed_harness_dir, out_dir=seed_snapshot, policy=cfg.harness_policy,
                            harness_tool=harness_tool_path(cfg))
            self.state.update(harness_dir=str(seed_snapshot), harness_digest=digest, seed_digest=digest)
            self._save_state()
        stopped = None
        for round_no in range(1, cfg.rounds + 1):
            batches = epoch_batches(len(cfg.train_cases), cfg.batch_size, cfg.selection_seed + round_no)
            for batch_no, task_ids in enumerate(batches, start=1):
                batch_dir = self.run_dir / "rounds" / f"r{round_no}" / f"b{batch_no}"
                if (batch_dir / "DONE").is_file():
                    continue
                if self._exhausted():
                    stopped = "budget"
                    break
                self._run_batch(round_no, batch_no, task_ids, batch_dir)
            if stopped:
                break
        final = self.final_eval()
        return {"stopped": stopped or "completed", **self.state, "final_eval": final}

    def _run_batch(self, round_no: int, batch_no: int, task_ids: list[int], batch_dir: Path) -> None:
        cfg = self.cfg
        started = time.time()
        rollouts_before = self.gaia2.rollouts
        batch_dir.mkdir(parents=True, exist_ok=True)
        harness_dir = Path(self.state["harness_dir"])
        store_path = batch_dir / "traces.jsonl"
        if store_path.exists():  # an interrupted attempt of this batch: start it over
            store_path.unlink()
        store = TraceStore(store_path)
        agent = (
            Gaia2HarnessAgent(self.designer_client, k=cfg.k, band=cfg.target_band, forbidden_ids=cfg.forbidden_ids)
            if cfg.arm == "ours"
            else NoopHarnessAgent()
        )
        orchestrator = Gaia2Orchestrator(
            env_spec=EnvSpec(import_path=ENV_IMPORT_PATH),
            policy_spec=PolicySpec(
                client_factory="envharness.infra.llm:LiteLLMClient",
                client_kwargs={"model": cfg.run.model},
                temperature=1.0,  # nominal; satisfies the band preflight (rollouts are sampled by gaia2-runner)
            ),
            harness_agent=agent,
            objective=DifficultyZone(target_band=cfg.target_band),
            budget=CappedAdaptive(max_k=cfg.max_k),
            trace_store=store,
            tool_schemas=[EXEC_TOOL_SCHEMA],
            env_state_schema=ENV_STATE_SCHEMA + "\n\n" + (PROMPTS_DIR / "app_cli_summary.md").read_text(encoding="utf-8"),
            config=OrchestratorConfig(
                task_id=f"gaia2-openclaw:{self.run_dir.name}",
                task_description=TASK_DESCRIPTION,
                n_tasks=len(task_ids),
                k_per_candidate=cfg.k,
                explicit_task_ids=list(task_ids),
                compute_baseline=True,
                baseline_cache_dir=None,
                mutator_baseline_mode="full",
                skip_passthrough_candidates=cfg.skip_passthrough_candidates,
                max_episode_steps=200,
            ),
            log_dir=batch_dir,
            gaia2=self.gaia2,
            cases=cfg.train_cases,
            harness_dir=harness_dir,
            work_dir=batch_dir,
        )
        orchestrator.run()
        traces = store.all()

        # -- induce candidate skills per task (cascade) --------------------------
        candidates = []
        induce_log = []
        for task_id in task_ids:
            case = cfg.train_cases[task_id]
            units, tier = select_units(
                [t for t in traces if t.rollout_seed == task_id],
                task_idx=task_id, scenario_id=case.scenario_id, pairs_per_task=cfg.pairs_per_task, arm=cfg.arm,
            )
            for unit in units:
                items = induce(self.induce_client, unit)
                for item in items:
                    item.source.update(round=round_no, batch=batch_no)
                candidates.extend(items)
                induce_log.append({"task_idx": task_id, "tier": unit.tier, "n_items": len(items),
                                   "items": [i.to_dict() for i in items]})
            if not units:
                induce_log.append({"task_idx": task_id, "tier": tier, "n_items": 0, "items": []})
        (batch_dir / "induce.jsonl").write_text("".join(json.dumps(r) + "\n" for r in induce_log), encoding="utf-8")

        # -- update the bank ----------------------------------------------------------
        bank = load_bank(self.run_dir / self.state["bank_file"])
        source = {"round": round_no, "batch": batch_no}
        new = select_new_skills(self.induce_client, candidates, bank,
                                max_new=cfg.max_new_skills_per_batch, source=source)
        (batch_dir / "selected.json").write_text(json.dumps([i.to_dict() for i in new], indent=2) + "\n", encoding="utf-8")
        bank = bank + new
        consolidated = False
        if len(bank) > cfg.max_skills:
            bank, consolidated = consolidate(self.induce_client, bank, max_skills=cfg.max_skills, source=source), True
        out_harness = batch_dir / "harness"
        try:
            digest = render(bank, seed_dir=cfg.seed_harness_dir, out_dir=out_harness, policy=cfg.harness_policy,
                            harness_tool=harness_tool_path(cfg), forbidden_ids=cfg.forbidden_ids)
        except SkillRenderError:
            bank, consolidated = consolidate(self.induce_client, bank, max_skills=max(1, len(bank) - 1), source=source), True
            digest = render(bank, seed_dir=cfg.seed_harness_dir, out_dir=out_harness, policy=cfg.harness_policy,
                            harness_tool=harness_tool_path(cfg), forbidden_ids=cfg.forbidden_ids)
        if consolidated:
            self.state["bank_file"] = f"bank_consolidated_r{round_no}b{batch_no}.jsonl"
            save_bank(self.run_dir / self.state["bank_file"], bank)
        else:
            with open(self.run_dir / self.state["bank_file"], "a", encoding="utf-8") as handle:
                for item in new:
                    handle.write(json.dumps(item.to_dict()) + "\n")
        (batch_dir / "harness.digest").write_text(digest + "\n", encoding="utf-8")

        with open(self.run_dir / "traces.jsonl", "a", encoding="utf-8") as handle:
            for trace in traces:
                handle.write(trace.model_dump_json() + "\n")
        used = self.gaia2.rollouts - rollouts_before
        self.state.update(
            rollouts=self.state["rollouts"] + used,
            batches=self.state["batches"] + 1,
            harness_dir=str(out_harness),
            harness_digest=digest,
        )
        self._log_lineage(
            {
                "round": round_no, "batch": batch_no, "task_ids": task_ids,
                "scenario_ids": [cfg.train_cases[t].scenario_id for t in task_ids],
                "input_harness_digest": harness_digest(cfg, harness_dir),
                "skills_digest": digest, "added": [i.title for i in new], "n_skills": len(bank),
                "consolidated": consolidated, "rollouts": used, "elapsed_s": round(time.time() - started, 1),
            }
        )
        self._save_state()
        (batch_dir / "DONE").write_text(json.dumps({"digest": digest}) + "\n", encoding="utf-8")

    def final_eval(self) -> dict:
        out = self.run_dir / "final_eval"
        if (out / "summary.json").is_file():
            return json.loads((out / "summary.json").read_text(encoding="utf-8"))
        harnesses = {"seed": self.run_dir / "seed_harness", "final": Path(self.state["harness_dir"])}
        before = self.gaia2.rollouts
        summary = evaluate(self.gaia2, self.cfg.dev_cases, harnesses=harnesses, k=self.cfg.eval_repetitions, out_dir=out)
        self.state["eval_rollouts"] = self.gaia2.rollouts - before
        self._save_state()
        return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_dir = cfg.run_root / args.run_id
    if run_dir.exists() and not args.resume:
        raise SystemExit(f"{run_dir} exists; pass --resume to continue it")
    if args.resume and not run_dir.exists():
        raise SystemExit(f"{run_dir} does not exist; nothing to resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = run_dir / "config.resolved.json"
    if not resolved.exists():
        resolved.write_text(json.dumps(
            {
                "config_path": str(cfg.path),
                "config": cfg.raw,
                "meta_are_head": subprocess.run(["git", "-C", str(cfg.meta_are_repo), "rev-parse", "HEAD"],
                                                text=True, capture_output=True).stdout.strip(),
                "image": cfg.image,
                "image_id": cfg.image_id,
                "seed_harness_digest": harness_digest(cfg, cfg.seed_harness_dir),
                "dataset_digest": cfg.dataset_digest,
                "train": [c.scenario_id for c in cfg.train_cases],
                "development": [c.scenario_id for c in cfg.dev_cases],
            }, indent=2, default=str) + "\n", encoding="utf-8")
    result = CoEvolution(cfg, run_dir).run()
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("stopped", "rollouts", "batches", "harness_digest")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
