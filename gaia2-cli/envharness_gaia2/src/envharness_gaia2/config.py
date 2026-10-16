"""Strict YAML schema for an envharness-gaia2 co-evolution run."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from envharness_gaia2.gaia2_run import CaseRef, RunSettings, sha256_file, wrap_docker_group

SCHEMA_VERSION = "envharness-gaia2/v1"
_SECTIONS: dict[str, set[str] | None] = {
    "schema_version": None,
    "arm": None,
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
    "optimizer_llm": {"model", "api_base", "wire_api", "reasoning_effort"},
    "envrigger": {"k_per_candidate", "max_k", "target_band", "skip_passthrough_candidates", "allow_setup"},
    "coevolution": {"rounds", "task_selection", "max_new_skills_per_batch", "max_skills", "pairs_per_task"},
    "budget": {"max_rollouts", "max_iterations"},
    "storage": {"run_root"},
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class OptimizerLLMConfig:
    model: str
    api_base: str
    wire_api: str
    reasoning_effort: str | None


@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict[str, Any]
    arm: str
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
    allow_setup: bool
    rounds: int
    batch_size: int
    selection_seed: int
    max_new_skills_per_batch: int
    max_skills: int
    pairs_per_task: int
    max_rollouts: int
    max_iterations: int
    run_root: Path

    @property
    def forbidden_ids(self) -> tuple[str, ...]:
        return tuple(case.scenario_id for case in (*self.train_cases, *self.dev_cases))


def load_config(path: str | Path, *, verify_image: bool = True) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping")
    unknown = sorted(set(raw) - set(_SECTIONS))
    missing = sorted(set(_SECTIONS) - set(raw))
    if unknown or missing:
        raise ConfigError(f"config keys: unknown={unknown} missing={missing}")
    for section, keys in _SECTIONS.items():
        if keys is None:
            continue
        value = raw[section]
        if not isinstance(value, dict) or set(value) != keys:
            raise ConfigError(f"[{section}] must have exactly {sorted(keys)}; got {sorted(value or {})}")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION}")
    if raw["arm"] not in ("ours", "orig"):
        raise ConfigError("arm must be ours or orig")
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
    )
    if settings.max_concurrent < 2:
        # concurrency 1 makes gaia2-runner use fixed host ports (8090/18789),
        # which collide with any other runner on the host.
        raise ConfigError("run.max_concurrent must be >= 2 (port allocation)")
    opt = raw["optimizer_llm"]
    if opt["wire_api"] not in ("chat_completions", "responses"):
        raise ConfigError("optimizer_llm.wire_api must be chat_completions or responses")
    env, coevo, budget = raw["envrigger"], raw["coevolution"], raw["budget"]
    band = tuple(float(x) for x in env["target_band"])
    if len(band) != 2 or not 0.0 <= band[0] <= band[1] <= 1.0:
        raise ConfigError("envrigger.target_band must be [lo, hi] within [0, 1]")
    if env["allow_setup"]:
        raise ConfigError("allow_setup is not supported (Setup/S0 is an optional extension)")
    selection = coevo["task_selection"]
    if selection.get("type") != "epoch_shuffled" or set(selection) != {"type", "batch_size", "seed"}:
        raise ConfigError("coevolution.task_selection must be {type: epoch_shuffled, batch_size, seed}")
    return Config(
        path=path,
        raw=raw,
        arm=raw["arm"],
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
        optimizer=OptimizerLLMConfig(opt["model"], opt["api_base"], opt["wire_api"], opt["reasoning_effort"]),
        k=int(env["k_per_candidate"]),
        max_k=int(env["max_k"]),
        target_band=band,  # type: ignore[arg-type]
        skip_passthrough_candidates=bool(env["skip_passthrough_candidates"]),
        allow_setup=False,
        rounds=int(coevo["rounds"]),
        batch_size=int(selection["batch_size"]),
        selection_seed=int(selection["seed"]),
        max_new_skills_per_batch=int(coevo["max_new_skills_per_batch"]),
        max_skills=int(coevo["max_skills"]),
        pairs_per_task=int(coevo["pairs_per_task"]),
        max_rollouts=int(budget["max_rollouts"]),
        max_iterations=int(budget["max_iterations"]),
        run_root=p(raw["storage"]["run_root"]),
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
