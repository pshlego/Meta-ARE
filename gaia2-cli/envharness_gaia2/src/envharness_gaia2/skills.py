"""Skill bank: selection, consolidation and rendering into the OpenClaw harness.

The bank (JSONL of envharness ``MemoryItem``) is the source of truth; each
batch renders it into ``skills/<slug>/SKILL.md`` of a copy of the seed
harness. OpenClaw lists every workspace skill (name + description) in its
system prompt and the agent reads a SKILL.md on demand, so there is no
retrieval step; the bank is capped and consolidated instead.
"""

from __future__ import annotations

import filecmp
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

from envharness.infra.llm import LLMClient, Message
from envharness.reasoning_bank.bank import MemoryItem
from envharness.reasoning_bank.induce import parse_memory_items

from envharness_gaia2 import PROMPTS_DIR

_SCENARIO_ID = re.compile(r"scenario_universe_\d+")
_MD_FILE_LIMIT = 20_000
_MD_TOTAL_LIMIT = 60_000


class SkillRenderError(RuntimeError):
    pass


# -- bank I/O ------------------------------------------------------------------
def load_bank(path: Path) -> list[MemoryItem]:
    if not path.is_file():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            items.append(MemoryItem(**{k: value.get(k) for k in ("title", "description", "content")},
                                    embedding=[], source=value.get("source") or {}))
    return items


def save_bank(path: Path, items: Iterable[MemoryItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item.to_dict()) + "\n" for item in items), encoding="utf-8")


def _items_text(items: Sequence[MemoryItem]) -> str:
    return "\n".join(
        f"[{i}] {item.title}\n    when: {item.description}\n    how:  {item.content}" for i, item in enumerate(items, 1)
    )


# -- LLM curation --------------------------------------------------------------
def select_new_skills(
    client: LLMClient, candidates: Sequence[MemoryItem], existing: Sequence[MemoryItem], *, max_new: int, source: dict
) -> list[MemoryItem]:
    if not candidates or max_new <= 0:
        return []
    system = (PROMPTS_DIR / "select_new_skills.md").read_text(encoding="utf-8").replace("{max_new}", str(max_new))
    user = (
        f"CURRENT skills ({len(existing)}):\n{_items_text(existing) or '(none)'}\n\n"
        f"CANDIDATE snippets ({len(candidates)}):\n{_items_text(candidates)}"
    )
    response = client.chat([Message(role="system", content=system), Message(role="user", content=user)])
    return _items(parse_memory_items(response.content or "")[:max_new], source)


def consolidate(client: LLMClient, items: Sequence[MemoryItem], *, max_skills: int, source: dict) -> list[MemoryItem]:
    system = (PROMPTS_DIR / "consolidate.md").read_text(encoding="utf-8").replace("{max_skills}", str(max_skills))
    response = client.chat([Message(role="system", content=system), Message(role="user", content=_items_text(items))])
    consolidated = _items(parse_memory_items(response.content or "")[:max_skills], source)
    if not consolidated:
        raise SkillRenderError("consolidation returned no parseable skills")
    return consolidated


def _items(parsed: Sequence[dict], source: dict) -> list[MemoryItem]:
    return [
        MemoryItem(title=p["title"].strip(), description=p["description"].strip(), content=p["content"].strip(),
                   embedding=[], source=dict(source))
        for p in parsed
        if p.get("title") and p.get("content")
    ]


# -- rendering -----------------------------------------------------------------
def slugify(title: str, taken: set[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:48].strip("-") or "skill"
    if not slug[0].isalnum():
        slug = "s-" + slug
    candidate, n = slug, 2
    while candidate in taken:
        candidate = f"{slug[:44]}-{n}"
        n += 1
    taken.add(candidate)
    return candidate


def skill_markdown(item: MemoryItem, name: str) -> str:
    description = " ".join(item.description.split()).replace('"', "'")
    return (
        f"---\nname: {name}\ndescription: \"{description}\"\n---\n\n"
        f"# {item.title.strip()}\n\n{item.content.strip()}\n"
    )


def skills_budget(seed_dir: Path) -> int:
    """Characters left for skills under the policy's harness-wide Markdown limit."""
    used = sum(len(p.read_text(encoding="utf-8")) for p in seed_dir.rglob("*.md") if "skills" not in p.parts)
    return _MD_TOTAL_LIMIT - used


def render(
    items: Sequence[MemoryItem],
    *,
    seed_dir: Path,
    out_dir: Path,
    policy: Path,
    harness_tool: Path,
    forbidden_ids: Iterable[str] = (),
) -> str:
    """Write seed + skills to ``out_dir``, validate, and return the harness digest."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(seed_dir, out_dir)
    skills_dir = out_dir / "skills"
    for child in skills_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
    taken: set[str] = set()
    total = 0
    for item in items:
        name = slugify(item.title, taken)
        text = skill_markdown(item, name)
        if len(text) > _MD_FILE_LIMIT:
            raise SkillRenderError(f"skill {name} exceeds {_MD_FILE_LIMIT} chars")
        (skills_dir / name).mkdir()
        (skills_dir / name / "SKILL.md").write_text(text, encoding="utf-8")
        total += len(text)
    budget = skills_budget(seed_dir)
    if total > budget:
        raise SkillRenderError(f"skills total {total} chars exceeds the remaining Markdown budget {budget}")

    blob = "\n".join(p.read_text(encoding="utf-8") for p in skills_dir.rglob("SKILL.md"))
    leaked = [sid for sid in forbidden_ids if sid and sid in blob]
    if _SCENARIO_ID.search(blob) or leaked:
        raise SkillRenderError(f"skills mention scenario identifiers: {leaked[:3]}")
    _assert_only_skills_differ(seed_dir, out_dir)

    check = subprocess.run(
        [sys.executable, str(harness_tool), "check", str(out_dir), "--policy", str(policy)],
        text=True, capture_output=True, check=False,
    )
    if check.returncode != 0:
        raise SkillRenderError(f"harness_tool check failed: {check.stderr.strip() or check.stdout.strip()}")
    digest = subprocess.run(
        [sys.executable, str(harness_tool), "digest", str(out_dir)], text=True, capture_output=True, check=True
    ).stdout.strip()
    return digest


def _assert_only_skills_differ(seed_dir: Path, out_dir: Path) -> None:
    def files(root: Path) -> set[str]:
        return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and "skills" != p.relative_to(root).parts[0]}

    seed_files, out_files = files(seed_dir), files(out_dir)
    if seed_files != out_files:
        raise SkillRenderError(f"non-skill files differ: {sorted(seed_files ^ out_files)}")
    changed = [f for f in seed_files if not filecmp.cmp(seed_dir / f, out_dir / f, shallow=False)]
    if changed:
        raise SkillRenderError(f"non-skill files changed: {changed}")
