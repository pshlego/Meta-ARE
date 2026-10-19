"""Render a skill bank into the OpenClaw harness (the original ``skill_dir`` injection mode).

The whole bank (no retrieval) becomes ``skills/<slug>/SKILL.md`` in a copy of
the seed harness. OpenClaw lists every workspace skill (name + description)
in its system prompt and the student reads a SKILL.md on demand, like the
original eval's ``skill_dir`` mode (``officeqa/reasoning_bank_eval.py``).
Size limits come from the harness policy file (``limits``).
"""

from __future__ import annotations

import filecmp
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from envharness.reasoning_bank.bank import MemoryItem

_SCENARIO_ID = re.compile(r"scenario_universe_\d+")


class SkillRenderError(RuntimeError):
    pass


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
    # one line; no backslash (a YAML escape) or double quote inside the double-quoted scalar
    description = " ".join(item.description.split()).replace("\\", "/").replace('"', "'")
    return f"---\nname: {name}\ndescription: \"{description}\"\n---\n\n# {item.title.strip()}\n\n{item.content.strip()}\n"


def render(
    items: Sequence[MemoryItem],
    *,
    seed_dir: Path,
    out_dir: Path,
    policy: Path,
    harness_tool: Path,
    forbidden_ids: Iterable[str] = (),
) -> str:
    """Write seed + skills to ``out_dir``, validate with ``harness_tool check``, return the digest."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(seed_dir, out_dir)
    skills_dir = out_dir / "skills"
    for child in skills_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
    taken: set[str] = set()
    for item in items:
        name = slugify(item.title, taken)
        (skills_dir / name).mkdir()
        (skills_dir / name / "SKILL.md").write_text(skill_markdown(item, name), encoding="utf-8")

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
    return subprocess.run(
        [sys.executable, str(harness_tool), "digest", str(out_dir)], text=True, capture_output=True, check=True
    ).stdout.strip()


def policy_limits(policy: Path) -> dict:
    return json.loads(Path(policy).read_text(encoding="utf-8")).get("limits") or {}


def _assert_only_skills_differ(seed_dir: Path, out_dir: Path) -> None:
    def files(root: Path) -> set[str]:
        return {p.relative_to(root).as_posix() for p in root.rglob("*")
                if p.is_file() and p.relative_to(root).parts[0] != "skills"}

    seed_files, out_files = files(seed_dir), files(out_dir)
    if seed_files != out_files:
        raise SkillRenderError(f"non-skill files differ: {sorted(seed_files ^ out_files)}")
    changed = [f for f in seed_files if not filecmp.cmp(seed_dir / f, out_dir / f, shallow=False)]
    if changed:
        raise SkillRenderError(f"non-skill files changed: {changed}")
