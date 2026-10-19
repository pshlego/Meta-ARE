"""Skill-bank builds: ports of the envharness Stage-2 bank builders (``bank.mode``).

``swebench_cascade`` -- ``experiments/swebench/bank_distillation``:

* ``ours`` = ``build_ours_bank.py --pair-mode cascade``: per task, baseline
  failures x the cascade's successes (Tier 1 accepted, Tier 2 last
  exploration candidate, Tier 3 baseline; skip without a baseline failure),
  capped at ``max_pairs_per_task``; ``existing_titles`` is the augment mode
  (``--existing-bank``) used for later co-evolution rounds.
* ``orig`` = ``build_orig_bank.py --pair-mode intra_baseline``: baseline
  failure x baseline success, capped per task at ``min(n_cascade, n_intra)``
  and subsampled deterministically.
* Both distil with the paired-diff atomic prompt (``induce.py``), threading the
  designer's diagnosis, embed ``title: description`` and dedup by title.

``alfworld_pair`` -- ``scripts/induce_pair.py`` (ALFWorld):

* ``orig`` = the task's baseline rollouts; ``ours`` = its accepted rollouts when
  a candidate was accepted, else the baseline rollouts (cascade fallback).
* Per task one pair: the shortest success and the longest failure. Both ->
  ``induce_paired_diff`` (same environment); one of them -> ReasoningBank's
  SUCCESSFUL_SI / FAILED_SI single-trajectory induction.
* Co-evolution keeps the SWE-bench augment mode (existing titles in the prompt)
  and title dedup on top.

Only the trajectory rendering and the prompts' domain wording are GAIA2
specific. Traces are handled as dicts (``Trace.model_dump``), like the original
scripts read them from ``traces.jsonl``.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import random
import sys
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from envharness.infra.llm import LLMClient, Message
from envharness.reasoning_bank.bank import MemoryItem
from envharness.reasoning_bank.induce import PAIRED_DIFF_SI, parse_memory_items
from envharness.reasoning_bank.induce import format_trajectory as reasoning_bank_trajectory
from envharness.third_party.reasoning_bank.prompts import FAILED_SI, SUCCESSFUL_SI

from envharness_gaia2 import PROMPTS_DIR
from envharness_gaia2.designer import is_fatal_optimizer_error

BANK_MODES = ("swebench_cascade", "alfworld_pair")
TIER_LABELS = ("1_accepted", "2_exploration_last", "3_baseline_fallback", "skip_no_base_fail", "skip_no_succ")


# -- pair selection (verbatim logic of build_ours_bank.py / build_orig_bank.py) ----
def extract_mutator_diagnosis(traces: list[dict]) -> tuple[str, str]:
    label = "unspecified"
    rationale = "unspecified"
    for t in traces:
        if t.get("kind") != "accepted":
            continue
        cand = t.get("candidate") or {}
        fa = t.get("failure_analysis") or {}
        if fa.get("label"):
            label = fa["label"]
        rat = cand.get("rationale")
        if rat and "skip" not in rat.lower():
            rationale = rat
            break
    return label, rationale


def _cand_id(t: dict):
    cand = t.get("candidate") or {}
    return t.get("candidate_id") or cand.get("id") or cand.get("candidate_id") or t.get("iteration_id")


def _has_mut(t: dict) -> bool:
    cand = t.get("candidate") or {}
    return bool((cand.get("rules_code") or cand.get("mutation_code") or "").strip() or cand.get("in_env_actions"))


def categorize_with_cascade(traces: list[dict], tier2_last_only: bool = True, max_baseline_rollouts: int | None = 5):
    """Returns (base_fail, selected_succ, tier_label) -- ``build_ours_bank.categorize_with_cascade``.

    A Setup-only candidate (``in_env_actions`` without ``rules_code``) counts as
    a mutation too; the original only had ``rules_code`` mutations in practice.
    """
    base_fail, base_succ, acc_succ, expl_succ = [], [], [], []
    n_baseline = 0
    for t in traces:
        if not t.get("steps"):
            continue
        k = t.get("kind")
        s = bool(t.get("success"))
        if k == "baseline":
            n_baseline += 1
            if max_baseline_rollouts is not None and n_baseline > max_baseline_rollouts:
                continue
            (base_succ if s else base_fail).append(t)
        elif k == "accepted" and _has_mut(t) and s:
            acc_succ.append(t)
        elif k == "exploration" and _has_mut(t) and s:
            expl_succ.append(t)
    if not base_fail:
        return None, None, "skip_no_base_fail"
    if acc_succ:
        return base_fail, acc_succ, "1_accepted"
    if expl_succ:
        if tier2_last_only:
            last_cid = _cand_id(expl_succ[-1])
            expl_succ = [t for t in expl_succ if _cand_id(t) == last_cid]
        return base_fail, expl_succ, "2_exploration_last"
    if base_succ:
        return base_fail, base_succ, "3_baseline_fallback"
    return None, None, "skip_no_succ"


def categorize(traces: list[dict]):
    """``build_orig_bank.categorize`` (all baseline rollouts)."""
    base_fail, base_succ, acc_succ, expl_succ = [], [], [], []
    for t in traces:
        if not t.get("steps"):
            continue
        k = t.get("kind")
        s = bool(t.get("success"))
        if k == "baseline":
            (base_succ if s else base_fail).append(t)
        elif k == "accepted" and _has_mut(t) and s:
            acc_succ.append(t)
        elif k == "exploration" and _has_mut(t) and s:
            expl_succ.append(t)
    return base_fail, base_succ, acc_succ, expl_succ


def select_succ_for_mode(mode: str, acc_succ, expl_succ, base_succ):
    if mode == "cascade":
        if acc_succ:
            return acc_succ, "1_accepted"
        if expl_succ:
            return expl_succ, "2_exploration_last"
        if base_succ:
            return base_succ, "3_baseline_fallback"
        return [], "skip_no_succ"
    if mode == "intra_baseline":
        if base_succ:
            return base_succ, "intra_baseline"
        return [], "skip_no_base_succ"
    raise ValueError(f"unknown mode: {mode}")


@dataclass
class PairWork:
    task_key: str
    tier: str
    base: dict
    succ: dict
    task_text: str
    failure_label: str
    rationale: str


@dataclass
class BankPlan:
    work: list[PairWork | PickWork] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)


def task_text_of(traces: list[dict]) -> str:
    for t in traces:
        steps = t.get("steps") or []
        if steps and (steps[0].get("info") or {}).get("task"):
            return steps[0]["info"]["task"]
    return ""


def plan_ours(traces_by_task: dict[str, list[dict]], *, max_pairs_per_task: int = 5,
              max_baseline_rollouts: int | None = 5, tier2_last_only: bool = True) -> BankPlan:
    plan = BankPlan()
    for key, traces in sorted(traces_by_task.items()):
        base_fail, sel_succ, tier = categorize_with_cascade(
            traces, tier2_last_only=tier2_last_only, max_baseline_rollouts=max_baseline_rollouts
        )
        row = {"task": key, "tier": tier, "pairs": 0}
        plan.rows.append(row)
        if base_fail is None:
            continue
        label, rationale = extract_mutator_diagnosis([t for t in traces if t.get("kind") == "accepted"])
        text = task_text_of(traces)
        for b in base_fail:
            for s in sel_succ:
                if max_pairs_per_task and row["pairs"] >= max_pairs_per_task:
                    break
                plan.work.append(PairWork(key, tier, b, s, text, label, rationale))
                row["pairs"] += 1
            if max_pairs_per_task and row["pairs"] >= max_pairs_per_task:
                break
    return plan


def plan_orig(traces_by_task: dict[str, list[dict]], *, seed: int = 0) -> BankPlan:
    """``build_orig_bank.py --pair-mode intra_baseline`` with the balanced cap."""
    plan = BankPlan()
    for key, traces in sorted(traces_by_task.items()):
        bf, bs, accs, expls = categorize(traces)
        if not bf:
            plan.rows.append({"task": key, "tier": "skip_no_base_fail", "pairs": 0})
            continue
        casc_succ, casc_tier = select_succ_for_mode("cascade", accs, expls, bs)
        intra_succ, _ = select_succ_for_mode("intra_baseline", accs, expls, bs)
        n_casc, n_intra = len(bf) * len(casc_succ), len(bf) * len(intra_succ)
        if n_casc == 0 or n_intra == 0:
            plan.rows.append({"task": key, "tier": "skip_not_common", "pairs": 0})
            continue
        cap = min(n_casc, n_intra)
        all_pairs = sorted(((b, s) for b in bf for s in intra_succ),
                           key=lambda p: (p[0].get("episode_id"), p[1].get("episode_id")))
        if len(all_pairs) > cap:
            rng = random.Random(seed + zlib.crc32(key.encode()) % 10_000)
            all_pairs = rng.sample(all_pairs, cap)
        label, rationale = extract_mutator_diagnosis([t for t in traces if t.get("kind") == "accepted"])
        text = task_text_of(traces)
        plan.rows.append({"task": key, "tier": "intra_baseline", "cascade_tier": casc_tier,
                          "n_cascade": n_casc, "n_intra": n_intra, "cap": cap, "pairs": len(all_pairs)})
        plan.work.extend(PairWork(key, "intra_baseline", b, s, text, label, rationale) for b, s in all_pairs)
    return plan


# -- trajectory rendering (GAIA2 counterpart of format_trajectory_swebench) --------
def format_trajectory(steps: list[dict], max_cmd_chars: int = 250, max_obs_chars: int = 400,
                      max_steps_shown: int = 18, max_think_chars: int = 600) -> str:
    """Render a student trajectory as the original renders a SWE-bench one.

    OpenClaw's reasoning is untagged assistant text, i.e. the original's
    "raw response present but no tags" branch (``Reasoning:`` + ``<action>``).
    """
    n_total = len(steps)
    shown = steps[-max_steps_shown:] if n_total > max_steps_shown else steps
    start = n_total - len(shown)
    out = []
    for i, s in enumerate(shown):
        idx = start + i + 1
        ra = s.get("raw_action") or {}
        kwargs = ra.get("kwargs") or {}
        if ra.get("name") == "reply":
            cmd = "reply to user: " + str(kwargs.get("text") or "").strip()
        else:
            cmd = str(kwargs.get("command") or "").strip()
        if len(cmd) > max_cmd_chars:
            cmd = cmd[:max_cmd_chars] + " ...[truncated]"
        fo = s.get("filtered_observation") or s.get("raw_observation") or {}
        obs = ((fo.get("text") if isinstance(fo, dict) else "") or "").strip()
        if len(obs) > max_obs_chars:
            obs = obs[:max_obs_chars] + " ...[truncated]"
        raw = (s.get("policy_raw_response") or "").strip()
        block = [f"Step {idx}:"]
        if raw:
            block.append(f"Reasoning: {raw[:max_think_chars]}")
        block.append(f"<action>{cmd}</action>")
        if s.get("blocked_reason"):
            block.append(f"Blocked: {s['blocked_reason']}")
        if obs:
            block.append(f"Observation: {obs}")
        out.append("\n".join(block))
    return "\n\n".join(out)


# -- distillation (induce_paired_diff_atomic) -----------------------------------------
def existing_lessons_block(titles: Sequence[str] | None) -> str:
    """The augment mode's (``--existing-bank``) prompt block."""
    if not titles:
        return ""
    existing_str = "\n".join(f"- {t}" for t in titles)
    return (
        "\n\n## EXISTING LESSONS (from prior rounds) — DO NOT RE-EMIT THESE\n"
        "The following lessons are ALREADY in the bank. Emit ONLY techniques "
        "NOT covered by these. If the trajectory difference is fully captured "
        f"by an existing lesson, emit NOTHING.\n{existing_str}\n"
    )


def induce_paired_diff_atomic(
    client: LLMClient,
    *,
    task_text: str,
    baseline_trajectory_text: str,
    mutated_trajectory_text: str,
    failure_label: str | None = None,
    mutation_rationale: str | None = None,
    existing_lesson_titles: Sequence[str] | None = None,
    max_items: int = 3,
    retries: int = 4,
) -> list[dict]:
    label = (failure_label or "unspecified").strip()
    summary = (mutation_rationale or "unspecified").strip()[:400]
    system = (PROMPTS_DIR / "induce_paired_diff_atomic.md").read_text(encoding="utf-8")
    system = system.replace("{failure_label}", label).replace("{mutation_summary}", summary).strip()
    existing = existing_lessons_block(existing_lesson_titles)
    user = (
        f"User request:\n{task_text[:3000]}\n\n"
        f"BASELINE trajectory (FAILED, env unmutated):\n{baseline_trajectory_text[:7000]}\n\n"
        f"MUTATED trajectory (SUCCEEDED, env perturbed by Mutator):\n{mutated_trajectory_text[:7000]}\n"
        f"{existing}"
        "\nCompare. Extract atomic recovery skills per the constraints. "
        "If no clean technique-level difference, emit NOTHING. "
        "If a difference exists but is already covered by an existing lesson above, also emit NOTHING."
    )
    for attempt in range(retries):
        try:
            response = client.chat(
                [Message(role="system", content=system), Message(role="user", content=user)], temperature=0.0
            )
            return parse_memory_items(response.content or "")[:max_items]
        except Exception as exc:  # noqa: BLE001 - mirrors the original's retry-then-empty policy
            if is_fatal_optimizer_error(exc):
                raise  # auth/permission/budget errors fail every call (induce.py)
            if attempt == retries - 1:
                print(f"[bank] induction failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                return []
            time.sleep(2**attempt)
    return []


def openai_embedder(model: str, api_base: str, api_key: str = "dummy") -> Callable[[list[str]], list[list[float]]]:
    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key=api_key)

    def embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 10):
            for attempt in range(5):
                try:
                    response = client.embeddings.create(model=model, input=texts[i : i + 10])
                    out.extend(d.embedding for d in response.data)
                    break
                except Exception:  # noqa: BLE001
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)
        return out

    return embed


def _induce_pair(work: PairWork, client: LLMClient, embed, condition: str, existing_titles) -> list[MemoryItem]:
    items = induce_paired_diff_atomic(
        client,
        task_text=work.task_text,
        baseline_trajectory_text=format_trajectory(work.base.get("steps") or []),
        mutated_trajectory_text=format_trajectory(work.succ.get("steps") or []),
        failure_label=work.failure_label,
        mutation_rationale=work.rationale,
        existing_lesson_titles=existing_titles,
    )
    if not items:
        return []
    try:
        embeddings = embed([f"{it['title']}: {it['description']}" for it in items])
    except Exception as exc:  # noqa: BLE001 - the original drops the pair when embedding fails
        print(f"[bank] embedding failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return []
    return [
        MemoryItem(
            title=it["title"], description=it["description"], content=it["content"], embedding=emb,
            source={
                "task": work.task_key, "condition": condition, "tier": work.tier,
                "induction_mode": f"paired_diff_atomic_{condition}_{work.tier}",
                "baseline_episode_id": work.base.get("episode_id"),
                "mutated_episode_id": work.succ.get("episode_id"),
                "mutated_candidate_id": _cand_id(work.succ),
                "mutator_failure_label": work.failure_label,
            },
        )
        for it, emb in zip(items, embeddings)
    ]


# -- alfworld_pair: scripts/induce_pair.py -------------------------------------------
_ALFWORLD_DOMAIN = "a text-based household environment (ALFWorld)"
GAIA2_DOMAIN = ("a GAIA2 personal-assistant environment (calendar, contacts, emails, messages, chats, "
                "rent-a-flat, city, cabs, shopping and cloud-drive, operated through app CLI commands)")


def _gaia2_si(si: str) -> str:
    """ReasoningBank prompt with only the domain words changed, as the original adapts it."""
    if _ALFWORLD_DOMAIN not in si:
        raise ValueError("ReasoningBank prompt no longer names the ALFWorld domain")
    return si.replace(_ALFWORLD_DOMAIN, GAIA2_DOMAIN)


@dataclass
class PickWork:
    task_key: str
    tier: str  # "baseline" (orig), "accepted" or "baseline_fallback" (ours)
    succ: dict | None
    fail: dict | None
    task_text: str

    @property
    def induction(self) -> str:
        if self.succ and self.fail:
            return "paired_diff"
        return "single_succ" if self.succ else "single_fail"


def pick_pair(rollouts: list[dict]) -> tuple[dict | None, dict | None]:
    """``induce_pair._pick_pair``: shortest success, longest failure."""
    succ = [r for r in rollouts if r.get("success") and r.get("steps")]
    fail = [r for r in rollouts if not r.get("success") and r.get("steps")]
    succ.sort(key=lambda r: len(r["steps"]))
    fail.sort(key=lambda r: -len(r["steps"]))
    return (succ[0] if succ else None), (fail[0] if fail else None)


def plan_pairs(traces_by_task: dict[str, list[dict]], *, condition: str) -> BankPlan:
    """``induce_pair.main``: orig from baseline rollouts; ours from accepted ones, else baseline."""
    if condition not in ("orig", "ours"):
        raise ValueError(f"unknown condition: {condition}")
    plan = BankPlan()
    for key, traces in sorted(traces_by_task.items()):
        baseline = [t for t in traces if t.get("kind") == "baseline"]
        accepted = [t for t in traces if t.get("kind") == "accepted"]
        if condition == "orig":
            rollouts, tier = baseline, "baseline"
        elif accepted:
            rollouts, tier = accepted, "accepted"
        else:
            rollouts, tier = baseline, "baseline_fallback"
        succ, fail = pick_pair(rollouts)
        if not succ and not fail:
            plan.rows.append({"task": key, "tier": "skip_no_steps", "pairs": 0})
            continue
        work = PickWork(key, tier, succ, fail, task_text_of(traces))
        plan.rows.append({"task": key, "tier": tier, "induction": work.induction, "pairs": 1})
        plan.work.append(work)
    return plan


def induce_pick(client: LLMClient, work: PickWork, *, existing_lesson_titles: Sequence[str] | None = None,
                max_items: int = 3, retries: int = 4) -> list[dict]:
    """``induce_pair._items_for_task``: paired-diff when both exist, else single-trajectory induction."""
    existing = existing_lessons_block(existing_lesson_titles)
    if work.succ and work.fail:
        system = PAIRED_DIFF_SI
        user = (
            f"Task: {work.task_text[:2000]}\n\n"
            f"FAILURE trajectory:\n{reasoning_bank_trajectory(work.fail.get('steps') or [])[:7000]}\n\n"
            f"SUCCESS trajectory:\n{reasoning_bank_trajectory(work.succ.get('steps') or [])[:7000]}\n"
            f"{existing}"
            "\nCompare. Extract atomic skills. If no clean difference, emit NOTHING."
        )
    else:
        trace = work.succ or work.fail
        system = _gaia2_si(SUCCESSFUL_SI if work.succ else FAILED_SI)
        user = f"User query: {work.task_text}\n\nTrajectory:\n{reasoning_bank_trajectory(trace.get('steps') or [])}{existing}"
    for attempt in range(retries):
        try:
            response = client.chat(
                [Message(role="system", content=system), Message(role="user", content=user)], temperature=0.0
            )
            return parse_memory_items(response.content or "")[:max_items]
        except Exception as exc:  # noqa: BLE001 - mirrors the original's retry-then-empty policy
            if is_fatal_optimizer_error(exc):
                raise  # auth/permission/budget errors fail every call (induce.py)
            if attempt == retries - 1:
                print(f"[bank] induction failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                return []
            time.sleep(2**attempt)
    return []


def _induce_pick(work: PickWork, client: LLMClient, embed, condition: str, existing_titles) -> list[MemoryItem]:
    items = induce_pick(client, work, existing_lesson_titles=existing_titles)
    if not items:
        return []
    # induce_pair embeds after induction and lets an embedding failure (after retries) stop the build
    embeddings = embed([f"## {it['title']}\n{it['description']}\n{it['content']}" for it in items])
    return [
        MemoryItem(
            title=it["title"], description=it["description"], content=it["content"], embedding=emb,
            source={
                "task": work.task_key, "condition": condition, "tier": work.tier, "induction": work.induction,
                "succ_episode_id": (work.succ or {}).get("episode_id", ""),
                "fail_episode_id": (work.fail or {}).get("episode_id", ""),
                "succ_candidate_id": _cand_id(work.succ) if work.succ and work.succ.get("kind") != "baseline" else None,
                "task_query": work.task_text,
            },
        )
        for it, emb in zip(items, embeddings)
    ]


def dedup_by_title(items: list[MemoryItem]) -> list[MemoryItem]:
    seen: set[str] = set()
    out = []
    for it in items:
        key = (it.title or "").strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def build(plan: BankPlan, client: LLMClient, embed, *, condition: str, existing_titles: Sequence[str] = (),
          concurrency: int = 4, dedup_titles: bool = True) -> tuple[list[MemoryItem], dict]:
    """Distil every planned pair (in parallel) and return (items, stats)."""
    items: list[MemoryItem] = []
    empty = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(_induce_pick if isinstance(w, PickWork) else _induce_pair, w, client, embed, condition,
                               list(existing_titles)) for w in plan.work]
        for future in futures:  # submission order keeps the bank deterministic
            got = future.result()
            empty += not got
            items.extend(got)
    if plan.work and not items and not existing_titles:
        # augment mode (later rounds) may legitimately emit nothing new. induce_pair.py: "[FATAL] induction produced 0 items ... refusing to hand empty bank(s) to
        # downstream stages" (an empty arm would silently evaluate as nobank).
        raise RuntimeError(f"induction produced 0 items for {condition} from {len(plan.work)} pairs")
    before = len(items)
    if dedup_titles:
        items = dedup_by_title(items)
    stats = {"condition": condition, "pairs": len(plan.work), "empty_pairs": empty, "items_before_dedup": before,
             "items": len(items), "rows": plan.rows,
             "tiers": {label: sum(1 for r in plan.rows if r["tier"] == label) for label in
                       sorted({r["tier"] for r in plan.rows})}}
    return items, stats


def save_bank(path: Path, items: Sequence[MemoryItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(it.to_dict()) + "\n" for it in items), encoding="utf-8")


def load_bank(path: Path) -> list[MemoryItem]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            v = json.loads(line)
            out.append(MemoryItem(title=v["title"], description=v.get("description", ""), content=v.get("content", ""),
                                  embedding=v.get("embedding") or [], source=v.get("source") or {}))
    return out
