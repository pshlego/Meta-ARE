"""Strict YAML schema for an envharness-gaia2 run (corpus -> bank -> eval, optionally over rounds)."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from envharness_gaia2.gaia2_run import CaseRef, RunSettings, sha256_file, wrap_docker_group

SCHEMA_VERSION = "envharness-gaia2/v2"
_SECTIONS: dict[str, set[str] | None] = {
    "schema_version": None,
    "meta_are_repo": None,
    "seed_harness_dir": None,
    "harness_policy": None,
    "image": None,
    "image_id": None,
    "container_runtime": None,
    "docker_group": None,
    "data": {
        "dataset_root", "dataset_source_descriptor", "dataset_source_revision",
        "train_manifest", "development_manifest", "test_manifest",
    },
    "task_agent": {"provider", "model", "base_url", "reasoning_effort", "thinking", "azure_config_dir"},
    "judge": {"provider", "model", "base_url", "time_check"},
    "run": {
        "scenario_timeout_seconds", "max_scenario_duration", "health_timeout_seconds",
        "process_completion_grace_seconds", "max_concurrent", "infrastructure_retries", "eval_repetitions",
    },
    "optimizer_llm": {"model", "api_base", "wire_api", "reasoning_effort", "induce_reasoning_effort"},
    "envrigger": {"k_per_candidate", "max_k", "target_band", "skip_passthrough_candidates"},
    "bank": {"mode", "max_pairs_per_task", "max_baseline_rollouts", "tier2_last_only", "dedup_titles", "embed_model",
             "concurrency", "orig_seed"},
    "coevolution": {"rounds"},
    "budget": {"max_rollouts"},
    "storage": {"run_root"},
    "retrieval": {"top_k", "mode", "mmr_lambda"},
    "injection": {"mode", "style"},
    "evaluation": {"orig_arm"},
}
# Optional sections. Without ``retrieval`` the whole bank is rendered into skills/; without
# ``injection`` items go in as SKILL.md files; without ``evaluation`` the orig arm is evaluated.
_OPTIONAL = {"retrieval", "injection", "evaluation"}
# Optional keys inside a section (absent = the behaviour before they existed):
#   coevolution.batch_size  each round runs its own disjoint batch of train tasks (absent: every round runs all)
#   coevolution.batch_seed  shuffle the train manifest with this seed before batching (null: manifest order)
#   evaluation.per_round    evaluate each round's bank on the dev manifest as soon as the round is committed
#   budget.include_eval     budget.max_rollouts counts eval rollouts too (absent: corpus rollouts only)
#   envrigger.designer      false: no environment design; each task runs only its K baseline rollouts in the
#                           unmodified environment, so every bank is a ReasoningBank-style bank of those
#                           rollouts (the original's ``orig`` build, round by round) (absent: true)
_OPTIONAL_KEYS: dict[str, set[str]] = {
    "envrigger": {"designer"},
    "coevolution": {"batch_size", "batch_seed"},
    "evaluation": {"per_round"},
    "budget": {"include_eval"},
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class OptimizerLLMConfig:
    model: str
    api_base: str
    wire_api: str
    reasoning_effort: str | None
    induce_reasoning_effort: str | None


@dataclass(frozen=True)
class BankConfig:
    mode: str  # swebench_cascade | alfworld_pair (bank.BANK_MODES)
    max_pairs_per_task: int
    max_baseline_rollouts: int | None
    tier2_last_only: bool
    dedup_titles: bool
    embed_model: str
    concurrency: int
    orig_seed: int


@dataclass(frozen=True)
class RetrievalConfig:
    """Per-scenario top-k skills, as every original eval does (``Bank.retrieve``)."""

    top_k: int
    mode: str  # cosine | mmr
    mmr_lambda: float


@dataclass(frozen=True)
class InjectionConfig:
    """How injected bank items reach the student (skills.INJECTION_MODES)."""

    mode: str = "skill_md"  # skill_md: OpenClaw SKILL.md files | prompt: the original memory block in AGENTS.md
    style: str = "soft"  # build_memory_block style (experiments/alfworld/reasoning_bank_eval.yaml inject_style)


@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict[str, Any]
    meta_are_repo: Path
    seed_harness_dir: Path
    harness_policy: Path
    image: str
    image_id: str
    docker_group: str | None
    run: RunSettings
    eval_repetitions: int
    train_cases: tuple[CaseRef, ...]
    dev_cases: tuple[CaseRef, ...]
    dataset_digest: str
    optimizer: OptimizerLLMConfig
    k: int
    max_k: int
    target_band: tuple[float, float]
    skip_passthrough_candidates: bool
    bank: BankConfig
    rounds: int
    max_rollouts: int
    run_root: Path
    retrieval: RetrievalConfig | None = None  # None: the whole bank goes into skills/
    injection: InjectionConfig = InjectionConfig()
    orig_arm: bool = True  # evaluate the bank induced from unmutated baseline rollouts (the original's ablation)
    batch_size: int | None = None  # None: every round runs every train task
    batch_seed: int | None = None  # None: batches follow the train manifest order
    per_round_eval: bool = False  # evaluate ours_r<r> right after round r (else once, after the last round)
    budget_includes_eval: bool = False  # max_rollouts caps corpus + eval rollouts (else corpus only)
    designer: bool = True  # False: baseline rollouts only, no environment design (ReasoningBank-style run)

    def round_tasks(self, round_no: int) -> list[int]:
        """Indices into ``train_cases`` that round ``round_no`` runs."""
        if self.batch_size is None:
            return list(range(len(self.train_cases)))
        order = list(range(len(self.train_cases)))
        if self.batch_seed is not None:
            random.Random(self.batch_seed).shuffle(order)
        return sorted(order[(round_no - 1) * self.batch_size: round_no * self.batch_size])

    @property
    def forbidden_ids(self) -> tuple[str, ...]:
        return tuple(case.scenario_id for case in (*self.train_cases, *self.dev_cases))


def load_config(path: str | Path, *, verify_image: bool = True) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping")
    unknown = sorted(set(raw) - set(_SECTIONS))
    missing = sorted(set(_SECTIONS) - set(raw) - _OPTIONAL)
    if unknown or missing:
        raise ConfigError(f"config keys: unknown={unknown} missing={missing}")
    for section, keys in _SECTIONS.items():
        if keys is None or section not in raw:
            continue
        value = raw[section]
        optional = _OPTIONAL_KEYS.get(section, set())
        if not isinstance(value, dict) or not keys <= set(value) <= keys | optional:
            raise ConfigError(f"[{section}] must have {sorted(keys)} (optional {sorted(optional)}); "
                              f"got {sorted(value or {})}")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION}")
    if raw["container_runtime"] != "docker":
        raise ConfigError("container_runtime must be docker")

    base = path.parent

    def p(value: str) -> Path:
        candidate = Path(value).expanduser()
        return (candidate if candidate.is_absolute() else base / candidate).resolve()

    repo = p(raw["meta_are_repo"])
    seed = repo / raw["seed_harness_dir"]
    policy = repo / raw["harness_policy"]
    for required in (seed, policy):
        if not required.exists():
            raise ConfigError(f"missing {required}")
    if verify_image:
        _verify_image(raw["image"], raw["image_id"], raw["docker_group"])

    data = raw["data"]
    dataset_root = p(data["dataset_root"])
    descriptor = json.loads(p(data["dataset_source_descriptor"]).read_text(encoding="utf-8"))
    if descriptor.get("source_revision") != data["dataset_source_revision"]:
        raise ConfigError("dataset descriptor revision mismatch")
    records = {item["scenario_id"]: item for item in descriptor.get("files", [])}
    train = _cases(p(data["train_manifest"]), dataset_root, records)
    dev = _cases(p(data["development_manifest"]), dataset_root, records)
    test_ids = set(_manifest_ids(p(data["test_manifest"])))
    overlap = ({c.scenario_id for c in train} & {c.scenario_id for c in dev}) | (
        {c.scenario_id for c in (*train, *dev)} & test_ids
    )
    if overlap:
        raise ConfigError(f"splits overlap: {sorted(overlap)}")
    dataset_digest = "sha256:" + hashlib.sha256(
        json.dumps(sorted((c.scenario_id, c.sha256) for c in (*train, *dev))).encode()
    ).hexdigest()

    agent, judge, run = raw["task_agent"], raw["judge"], raw["run"]
    settings = RunSettings(
        meta_are_repo=repo,
        dataset_root=dataset_root,
        image=raw["image"],
        docker_group=raw["docker_group"],
        model=agent["model"],
        model_provider=agent["provider"],
        model_endpoint=agent["base_url"],
        reasoning_effort=agent["reasoning_effort"],
        thinking=agent["thinking"],
        azure_config_dir=p(agent["azure_config_dir"]) if agent["azure_config_dir"] else None,
        judge_model=judge["model"],
        judge_provider=judge["provider"],
        judge_endpoint=judge["base_url"],
        judge_time_check=str(judge["time_check"]),
        scenario_timeout_seconds=int(run["scenario_timeout_seconds"]),
        max_scenario_duration=run["max_scenario_duration"],
        health_timeout_seconds=int(run["health_timeout_seconds"]),
        process_completion_grace_seconds=int(run["process_completion_grace_seconds"]),
        max_concurrent=int(run["max_concurrent"]),
        infrastructure_retries=int(run["infrastructure_retries"]),
        harness_policy=policy,
    )
    if settings.max_concurrent < 2:
        # concurrency 1 makes gaia2-runner use fixed host ports (8090/18789),
        # which collide with any other runner on the host.
        raise ConfigError("run.max_concurrent must be >= 2 (port allocation)")
    opt = raw["optimizer_llm"]
    if opt["wire_api"] not in ("chat_completions", "responses"):
        raise ConfigError("optimizer_llm.wire_api must be chat_completions or responses")
    env, bank, budget = raw["envrigger"], raw["bank"], raw["budget"]
    band = tuple(float(x) for x in env["target_band"])
    if len(band) != 2 or not 0.0 <= band[0] < band[1] <= 1.0:
        raise ConfigError("envrigger.target_band must be [lo, hi] with 0 <= lo < hi <= 1 (DifficultyZone)")
    use_designer = env.get("designer", True)
    if not isinstance(use_designer, bool):
        raise ConfigError("envrigger.designer must be true or false")
    if not use_designer and not env["skip_passthrough_candidates"]:
        raise ConfigError("envrigger.designer false needs skip_passthrough_candidates true "
                          "(the empty candidate must end the task after its baseline)")
    if bank["mode"] not in ("swebench_cascade", "alfworld_pair"):
        raise ConfigError("bank.mode must be swebench_cascade or alfworld_pair")
    retrieval = None
    if "retrieval" in raw:
        ret = raw["retrieval"]
        retrieval = RetrievalConfig(int(ret["top_k"]), str(ret["mode"]), float(ret["mmr_lambda"]))
        if retrieval.top_k < 1 or retrieval.mode not in ("cosine", "mmr") or not 0.0 <= retrieval.mmr_lambda <= 1.0:
            raise ConfigError("retrieval: top_k >= 1, mode cosine|mmr, 0 <= mmr_lambda <= 1")
    inj = raw.get("injection") or {"mode": "skill_md", "style": "soft"}
    injection = InjectionConfig(str(inj["mode"]), str(inj["style"]))
    if injection.mode not in ("skill_md", "prompt") or injection.style not in ("soft", "legacy"):
        raise ConfigError("injection: mode skill_md|prompt, style soft|legacy")
    if injection.mode == "prompt" and retrieval is None:
        raise ConfigError("injection.mode prompt pastes the retrieved top-k (as the original evals do): "
                          "add a retrieval section")
    evaluation = raw.get("evaluation") or {"orig_arm": True}
    orig_arm, per_round_eval = evaluation["orig_arm"], evaluation.get("per_round", False)
    include_eval = budget.get("include_eval", False)
    for name, flag in (("evaluation.orig_arm", orig_arm), ("evaluation.per_round", per_round_eval),
                       ("budget.include_eval", include_eval)):
        if not isinstance(flag, bool):
            raise ConfigError(f"{name} must be true or false")
    coevo = raw["coevolution"]
    rounds = int(coevo["rounds"])
    if rounds < 1:
        raise ConfigError("coevolution.rounds must be >= 1 (1 = the original single pass)")
    batch_size = coevo.get("batch_size")
    batch_seed = coevo.get("batch_seed")
    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size < 1 or rounds * batch_size > len(train):
            raise ConfigError(f"coevolution.batch_size must be an int >= 1 with rounds x batch_size <= "
                              f"{len(train)} train tasks (disjoint batches)")
    if batch_seed is not None and (batch_size is None or not isinstance(batch_seed, int)):
        raise ConfigError("coevolution.batch_seed must be an int, and needs batch_size")
    return Config(
        path=path,
        raw=raw,
        meta_are_repo=repo,
        seed_harness_dir=seed,
        harness_policy=policy,
        image=raw["image"],
        image_id=raw["image_id"],
        docker_group=raw["docker_group"],
        run=settings,
        eval_repetitions=int(run["eval_repetitions"]),
        train_cases=train,
        dev_cases=dev,
        dataset_digest=dataset_digest,
        optimizer=OptimizerLLMConfig(opt["model"], opt["api_base"], opt["wire_api"], opt["reasoning_effort"],
                                     opt["induce_reasoning_effort"]),
        k=int(env["k_per_candidate"]),
        max_k=int(env["max_k"]),
        target_band=band,  # type: ignore[arg-type]
        skip_passthrough_candidates=bool(env["skip_passthrough_candidates"]),
        bank=BankConfig(
            mode=str(bank["mode"]),
            max_pairs_per_task=int(bank["max_pairs_per_task"]),
            max_baseline_rollouts=None if bank["max_baseline_rollouts"] is None else int(bank["max_baseline_rollouts"]),
            tier2_last_only=bool(bank["tier2_last_only"]),
            dedup_titles=bool(bank["dedup_titles"]),
            embed_model=str(bank["embed_model"]),
            concurrency=int(bank["concurrency"]),
            orig_seed=int(bank["orig_seed"]),
        ),
        rounds=rounds,
        max_rollouts=int(budget["max_rollouts"]),
        run_root=p(raw["storage"]["run_root"]),
        retrieval=retrieval,
        injection=injection,
        orig_arm=orig_arm,
        batch_size=batch_size,
        batch_seed=batch_seed,
        per_round_eval=per_round_eval,
        budget_includes_eval=include_eval,
        designer=use_designer,
    )


def _manifest_ids(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        return [case_id for ids in value.values() for case_id in ids]
    return list(value)


def _cases(manifest: Path, root: Path, records: dict[str, dict]) -> tuple[CaseRef, ...]:
    cases = []
    for scenario_id in _manifest_ids(manifest):
        record = records.get(scenario_id)
        if record is None:
            raise ConfigError(f"{scenario_id} is not in the dataset descriptor")
        source = root / record["path"]
        if sha256_file(source) != record["sha256"]:
            raise ConfigError(f"dataset digest drift: {source}")
        cases.append(CaseRef(scenario_id, Path(record["path"]).parts[0], source, record["sha256"]))
    return tuple(cases)


def _verify_image(image: str, image_id: str, docker_group: str | None) -> None:
    completed = subprocess.run(
        wrap_docker_group(["docker", "image", "inspect", image, "--format", "{{.Id}}"], docker_group),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or completed.stdout.strip() != image_id:
        raise ConfigError(f"image {image} id {completed.stdout.strip()!r} != pinned {image_id}")
