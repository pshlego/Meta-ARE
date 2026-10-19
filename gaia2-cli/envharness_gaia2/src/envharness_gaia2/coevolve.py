"""EnvHarness on GAIA2 with OpenClaw: corpus -> skill bank -> eval, over co-evolution rounds.

    uv run python -m envharness_gaia2.coevolve --config configs/smoke_cloudgpt.yaml --run-id <id> [--resume]

Mirrors an envharness ``experiments/<bench>/reproduce.py``:

* Stage 1 (corpus): the original Orchestrator, one run per train task
  (``run_harness.py --n-tasks 1``), baseline K -> propose -> K rollouts ->
  decide up to ``max_k``.
* Stage 2 (banks): ``ours`` from the round's traces (``bank.mode``: SWE-bench cascade or ALFWorld pair build),
  ``orig`` from round 1's baseline traces (intra-baseline or ALFWorld baseline-pair build).
* Stage 3 (eval): nobank / orig / ours_r<r> on the held-out manifest, one
  attempt per instance. With ``retrieval`` each scenario's student gets the
  bank's top-k items for its first user request in OpenClaw ``skills/`` (the
  original evals' ``Bank.retrieve``); without it the whole bank is rendered.

Co-evolution (``coevolution.rounds`` > 1) chains original parts only: round r's
student carries ``bank_{r-1}`` in its skills, and round r's bank adds the new
lessons of the augment-mode build (``--existing-bank``); with ``retrieval`` the
student of each round r task carries the top-k of ``bank_{r-1}`` for that task. ``rounds: 1`` is the
original single pass. With ``coevolution.batch_size`` each round runs its own
disjoint batch of train tasks; with ``evaluation.per_round`` each round's bank is
evaluated on the dev manifest right after the round. Eval results are only written
to ``eval/`` and never reach the designer or the induction LLM. Tasks and round
banks commit with DONE markers and eval arms in ``eval/summary.json``, so
``--resume`` continues where the run stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from envharness.orchestration.budget import CappedAdaptive
from envharness.orchestration.objectives import DifficultyZone
from envharness.orchestration.orchestrator import OrchestratorConfig
from envharness.orchestration.runner import EnvSpec, PolicySpec
from envharness.orchestration.storage import TraceStore

from envharness_gaia2 import PROMPTS_DIR
from envharness_gaia2.bank import (
    build, dedup_by_title, load_bank, openai_embedder, plan_orig, plan_ours, plan_pairs, save_bank,
)
from envharness_gaia2.config import Config, load_config
from envharness_gaia2.designer import Gaia2HarnessAgent, make_client
from envharness_gaia2.evaluate import Arm, evaluate
from envharness_gaia2.gaia2_run import Gaia2Runner
from envharness_gaia2.gaia2_run import CaseRef
from envharness_gaia2.orchestrator import ENV_IMPORT_PATH, ENV_STATE_SCHEMA, EXEC_TOOL_SCHEMA, Gaia2Orchestrator
from envharness_gaia2.retrieval import Retriever, selection_key
from envharness_gaia2.skills import render

CORPUS_RETRIES = 2  # experiments/swebench/reproduce.py: re-run a task whose traces.jsonl is empty

TASK_DESCRIPTION = (
    "Mutate GAIA2 train scenarios where the OpenClaw Policy is currently weak; only emit "
    "mutations that exercise failure-recovery skills WITHOUT shifting the env's surface "
    "presentation. The induced memory items should capture transferable assistant "
    "techniques (app CLI usage, verification, disambiguation) that close the gap on "
    "held-out GAIA2 scenarios."
)


def harness_tool_path(cfg: Config) -> Path:
    return cfg.meta_are_repo / "gaia2-cli" / "shared" / "harness_tool.py"


def bank_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes() if path.is_file() else b"").hexdigest()


def harness_digest(cfg: Config, harness_dir: Path) -> str:
    return subprocess.run(
        [sys.executable, str(harness_tool_path(cfg)), "digest", str(harness_dir)],
        text=True, capture_output=True, check=True,
    ).stdout.strip()


class Pipeline:
    def __init__(self, cfg: Config, run_dir: Path, *, gaia2: Gaia2Runner | None = None,
                 designer_client=None, induce_client=None, embedder=None) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.gaia2 = gaia2 or Gaia2Runner(
            cfg.run, harness_digest_fn=lambda d: harness_digest(cfg, d), label_prefix=f"envh.{run_dir.name}"
        )
        self.designer_client = designer_client or make_client(
            cfg.optimizer, log_path=run_dir / "designer_calls.jsonl", role="designer"
        )
        self.induce_client = induce_client or make_client(
            cfg.optimizer, log_path=run_dir / "induce_calls.jsonl", role="induce",
            reasoning_effort=cfg.optimizer.induce_reasoning_effort,
        )
        self.embed = embedder or openai_embedder(cfg.bank.embed_model, cfg.optimizer.api_base)
        self.retriever = (Retriever(cfg.retrieval, self.embed, run_dir / "query_embeddings.json")
                          if cfg.retrieval else None)
        self.state_path = run_dir / "budget.json"
        self.state = (json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.is_file()
                      else {"rollouts": 0, "eval_rollouts": 0, "max_rollouts": cfg.max_rollouts, "stopped": None})

    # -- bookkeeping ---------------------------------------------------------------
    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

    def _spent(self) -> int:
        """Rollouts that count against ``budget.max_rollouts``."""
        return self.state["rollouts"] + (self.state["eval_rollouts"] if self.cfg.budget_includes_eval else 0)

    def _count(self, before: int, key: str = "rollouts") -> None:
        self.state[key] += self.gaia2.rollouts - before
        self._save_state()

    def _render(self, items, out_dir: Path) -> str:
        digest = render(items, seed_dir=self.cfg.seed_harness_dir, out_dir=out_dir, policy=self.cfg.harness_policy,
                        harness_tool=harness_tool_path(self.cfg), forbidden_ids=self.cfg.forbidden_ids,
                        injection=self.cfg.injection.mode, style=self.cfg.injection.style)
        (out_dir.parent / f"{out_dir.name}.digest").write_text(digest + "\n", encoding="utf-8")
        return digest

    def _student_harness(self, bank, case: CaseRef, root: Path) -> tuple[Path, list[str]]:
        """Seed + the bank's top-k for ``case`` (retrieval mode); equal selections share a harness."""
        chosen = self.retriever.select(bank, case)
        if self.cfg.injection.mode == "skill_md":  # OpenClaw lists skills by name; the prompt keeps retrieval order
            chosen = sorted(chosen, key=lambda item: item.title)
        if not chosen:
            return self.run_dir / "seed_harness", []
        harness = root / selection_key(chosen)
        if not (root / f"{harness.name}.digest").is_file():
            self._render(chosen, harness)
        return harness, [item.title for item in chosen]

    # -- main --------------------------------------------------------------------------
    def run(self) -> dict:
        seed_dir = self.run_dir / "seed_harness"
        if not (self.run_dir / "seed_harness.digest").is_file():
            self.state["seed_digest"] = self._render([], seed_dir)
            self._save_state()
        if self.cfg.batch_size is not None:
            (self.run_dir / "batches.json").write_text(json.dumps({
                f"r{r}": [self.cfg.train_cases[t].scenario_id for t in self.cfg.round_tasks(r)]
                for r in range(1, self.cfg.rounds + 1)}, indent=2) + "\n", encoding="utf-8")
        if self.state["stopped"] and self._spent() < self.cfg.max_rollouts:
            # resumed with a raised budget: continue where the cap stopped the run
            self.state.update(stopped=None, max_rollouts=self.cfg.max_rollouts)
            self._save_state()
        harness, bank = seed_dir, []
        for round_no in range(1, self.cfg.rounds + 1):
            round_dir = self.run_dir / "rounds" / f"r{round_no}"
            if not (round_dir / "DONE").is_file():
                self._round(round_no, round_dir, harness, bank)
            if not (round_dir / "DONE").is_file():
                break  # the budget stopped this round's corpus: no bank from a partial round
            if self.retriever is None:
                harness = round_dir / "harness"
            bank = load_bank(self.run_dir / "banks" / f"ours_r{round_no}.jsonl")
            if self.cfg.per_round_eval:
                self.evaluate()  # this round's bank (and nobank/orig after round 1) on the dev manifest
            if self.state["stopped"]:
                break
        summary = self.evaluate()  # updates eval_rollouts: evaluate before snapshotting the state
        return {**self.state, "eval": summary}

    def _round(self, round_no: int, round_dir: Path, harness: Path, prior_bank: list) -> None:
        started = time.time()
        before = self.state["rollouts"]
        input_digest = harness_digest(self.cfg, harness) if self.retriever is None else None
        input_bank = bank_sha256(self.run_dir / "banks" / f"ours_r{round_no - 1}.jsonl") if round_no > 1 else None
        traces_by_task = self.corpus(round_no, round_dir / "corpus", harness, prior_bank)
        if len(traces_by_task) < len(self.cfg.round_tasks(round_no)):
            return  # stopped by the budget; --resume with a larger budget.max_rollouts finishes the round

        bank_cfg = self.cfg.bank
        if bank_cfg.mode == "alfworld_pair":
            ours_plan, ours_condition = plan_pairs(traces_by_task, condition="ours"), "ours"
        else:
            ours_plan, ours_condition = plan_ours(traces_by_task, max_pairs_per_task=bank_cfg.max_pairs_per_task,
                                                  max_baseline_rollouts=bank_cfg.max_baseline_rollouts,
                                                  tier2_last_only=bank_cfg.tier2_last_only), "cascade"
        # induce_pair (alfworld_pair) writes every induced item; title dedup is the swebench_cascade
        # step. Across rounds the merged bank is deduplicated when dedup_titles is set.
        within_round_dedup = bank_cfg.dedup_titles and bank_cfg.mode != "alfworld_pair"
        new, ours_stats = build(
            ours_plan, self.induce_client, self.embed, condition=ours_condition,
            existing_titles=[item.title for item in prior_bank], concurrency=self.cfg.bank.concurrency,
            dedup_titles=within_round_dedup,
        )
        bank = dedup_by_title([*prior_bank, *new]) if self.cfg.bank.dedup_titles else [*prior_bank, *new]
        save_bank(self.run_dir / "banks" / f"ours_r{round_no}_new.jsonl", new)
        save_bank(self.run_dir / "banks" / f"ours_r{round_no}.jsonl", bank)
        stats = {"ours": ours_stats}
        if round_no == 1 and self.cfg.orig_arm:
            if bank_cfg.mode == "alfworld_pair":
                orig_plan, orig_condition = plan_pairs(traces_by_task, condition="orig"), "orig"
            else:
                orig_plan, orig_condition = plan_orig(traces_by_task, seed=bank_cfg.orig_seed), "intra_baseline"
            orig, orig_stats = build(orig_plan, self.induce_client, self.embed, condition=orig_condition,
                                     concurrency=bank_cfg.concurrency, dedup_titles=within_round_dedup)
            save_bank(self.run_dir / "banks" / "orig.jsonl", orig)
            stats["orig"] = orig_stats
        (round_dir / "bank_build.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

        # Retrieval mode never renders the whole bank (it may exceed the skill limits); the bank file is the output.
        digest = self._render(bank, round_dir / "harness") if self.retriever is None else None
        output_bank = bank_sha256(self.run_dir / "banks" / f"ours_r{round_no}.jsonl")
        with open(self.run_dir / "lineage.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "round": round_no, "input_harness_digest": input_digest, "output_harness_digest": digest,
                "input_bank_sha256": input_bank, "output_bank_sha256": output_bank,
                "n_tasks_run": len(traces_by_task), "new_items": [item.title for item in new],
                "n_skills": len(bank), "rollouts": self.state["rollouts"] - before,
                "stopped": self.state["stopped"], "elapsed_s": round(time.time() - started, 1),
            }) + "\n")
        (round_dir / "DONE").write_text(json.dumps({"digest": digest, "bank_sha256": output_bank}) + "\n",
                                        encoding="utf-8")

    # -- Stage 1 -----------------------------------------------------------------------
    def corpus(self, round_no: int, corpus_dir: Path, harness: Path, prior_bank=()) -> dict[str, list[dict]]:
        cfg = self.cfg
        out: dict[str, list[dict]] = {}
        for task_id in cfg.round_tasks(round_no):
            task_dir = corpus_dir / f"task_{task_id:03d}"
            if not (task_dir / "DONE").is_file():
                if self._spent() >= cfg.max_rollouts:
                    self.state["stopped"] = f"budget (round {round_no}, before task {task_id})"
                    self._save_state()
                    break
                self._corpus_task(task_id, task_dir, harness, prior_bank)
            out[f"r{round_no}/task_{task_id:03d}"] = [
                t.model_dump(mode="json") for t in TraceStore(task_dir / "traces.jsonl").all()
            ]
        return out

    def _corpus_task(self, task_id: int, task_dir: Path, harness: Path, prior_bank=()) -> None:
        """One orchestrator run per task; like experiments/swebench/reproduce.py, a task whose
        traces.jsonl comes back empty (the orchestrator aborted it) is re-run up to CORPUS_RETRIES times."""
        for attempt in range(1 + CORPUS_RETRIES):
            used = self._corpus_task_once(task_id, task_dir, harness, prior_bank)
            if TraceStore(task_dir / "traces.jsonl").all():
                break
            print(f"[corpus] task {task_id}: no traces (attempt {attempt + 1}/{1 + CORPUS_RETRIES})", file=sys.stderr)
        (task_dir / "DONE").write_text(json.dumps({"harness_digest": harness_digest(self.cfg, used)})
                                       + "\n", encoding="utf-8")

    def _corpus_task_once(self, task_id: int, task_dir: Path, harness: Path, prior_bank=()) -> Path:
        """Run the task's orchestrator once; returns the student harness it used."""
        cfg = self.cfg
        if task_dir.exists():  # an interrupted attempt of this task: start it over
            shutil.rmtree(task_dir)
        task_dir.mkdir(parents=True)
        if self.retriever is not None:
            harness, titles = self._student_harness(prior_bank, cfg.train_cases[task_id],
                                                    task_dir.parent.parent / "student_harnesses")
            (task_dir / "retrieval.json").write_text(json.dumps({
                "bank_size": len(prior_bank), "skills": titles, "harness_dir": str(harness),
                "harness_digest": harness_digest(cfg, harness),
            }, indent=2) + "\n", encoding="utf-8")
        base, before = self.state["rollouts"], self.gaia2.rollouts

        def count_rollouts(_: int) -> None:  # saved per batch, so an interrupted task's rollouts still count
            self.state["rollouts"] = base + self.gaia2.rollouts - before
            self._save_state()

        orchestrator = Gaia2Orchestrator(
            env_spec=EnvSpec(import_path=ENV_IMPORT_PATH),
            policy_spec=PolicySpec(
                client_factory="envharness.infra.llm:LiteLLMClient",
                client_kwargs={"model": cfg.run.model},
                temperature=1.0,  # nominal; satisfies the band preflight (OpenClaw samples the rollouts)
            ),
            harness_agent=Gaia2HarnessAgent(self.designer_client, k=cfg.k, band=cfg.target_band),
            objective=DifficultyZone(target_band=cfg.target_band, window=20),  # experiments/swebench/corpus.yaml
            budget=CappedAdaptive(max_k=cfg.max_k),
            trace_store=TraceStore(task_dir / "traces.jsonl"),
            tool_schemas=[EXEC_TOOL_SCHEMA],
            env_state_schema=ENV_STATE_SCHEMA + "\n" + (PROMPTS_DIR / "app_cli_summary.md").read_text(encoding="utf-8"),
            config=OrchestratorConfig(
                task_id=f"gaia2-openclaw:{self.run_dir.name}",
                task_description=TASK_DESCRIPTION,
                n_tasks=1,
                k_per_candidate=cfg.k,
                explicit_task_ids=[task_id],
                compute_baseline=True,
                baseline_cache_dir=None,
                mutator_baseline_mode="full",
                skip_passthrough_candidates=cfg.skip_passthrough_candidates,
            ),
            log_dir=task_dir,
            gaia2=self.gaia2,
            cases=cfg.train_cases,
            harness_dir=harness,
            work_dir=task_dir,
            forbidden_ids=cfg.forbidden_ids,
            on_rollouts=count_rollouts,
        )
        orchestrator.run()
        count_rollouts(0)
        return harness

    # -- Stage 3 -----------------------------------------------------------------------
    def evaluate(self) -> dict:
        """Evaluate every arm not yet in ``eval/summary.json`` (committed arms are kept as they are)."""
        out = self.run_dir / "eval"
        done = json.loads((out / "summary.json").read_text(encoding="utf-8")) if (out / "summary.json").is_file() else {}
        seed = self.run_dir / "seed_harness"
        arms = {"nobank": Arm(lambda case: (seed, []), 0)}
        banks: dict[str, tuple[Path, Path | None]] = {}  # arm -> (bank file, whole-bank harness)
        orig_bank = self.run_dir / "banks" / "orig.jsonl"
        if self.cfg.orig_arm and orig_bank.is_file():
            harness = out / "harness_orig"
            if self.retriever is None and not (out / "harness_orig.digest").is_file():
                self._render(load_bank(orig_bank), harness)
            banks["orig"] = (orig_bank, harness)
        for round_dir in sorted((self.run_dir / "rounds").glob("r*"), key=lambda p: int(p.name[1:])):
            if (round_dir / "DONE").is_file():
                banks[f"ours_{round_dir.name}"] = (self.run_dir / "banks" / f"ours_{round_dir.name}.jsonl",
                                                    round_dir / "harness")
        for name, (bank_file, whole) in banks.items():
            items = load_bank(bank_file)
            if self.retriever is None:
                titles = [item.title for item in items]
                arms[name] = Arm(lambda case, h=whole, t=titles: (h, t), len(items))
            else:
                root = out / "harnesses" / name
                arms[name] = Arm(lambda case, b=items, r=root: self._student_harness(b, case, r), len(items))
        todo = {name: arm for name, arm in arms.items() if name not in done}
        if not todo:
            return done
        before = self.gaia2.rollouts
        new = evaluate(self.gaia2, self.cfg.dev_cases, arms=todo, k=self.cfg.eval_repetitions, out_dir=out,
                       digest_fn=lambda d: harness_digest(self.cfg, d), max_concurrent=self.cfg.run.max_concurrent)
        self._count(before, "eval_rollouts")
        summary = {name: {**done, **new}[name] for name in arms if name in done or name in new}
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
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
    with open(run_dir / "resumes.jsonl", "a", encoding="utf-8") as handle:
        head = subprocess.run(["git", "-C", str(cfg.meta_are_repo), "rev-parse", "HEAD"],
                              text=True, capture_output=True).stdout.strip()
        handle.write(json.dumps({"t": time.time(), "resume": args.resume, "meta_are_head": head}) + "\n")
    result = Pipeline(cfg, run_dir).run()
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("rollouts", "eval_rollouts", "stopped")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
