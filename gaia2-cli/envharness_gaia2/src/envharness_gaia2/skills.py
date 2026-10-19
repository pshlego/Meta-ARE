"""Render bank items into a copy of the seed OpenClaw harness, in one of two injection modes.

``skill_md``: each item becomes ``skills/<slug>/SKILL.md``. OpenClaw lists every
workspace skill (name + description) in its system prompt and the student reads
a SKILL.md on demand, like the original eval's ``skill_dir`` mode
(``officeqa/reasoning_bank_eval.py``).

``prompt``: the items are pasted into the student's system prompt
(``AGENTS_TEMPLATE.md`` -> AGENTS.md) with the original ALFWorld eval's
``build_memory_block`` (header, per-step relevance gate, ``### Insight i``
blocks in retrieval order), as every original eval injects its top-k.

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

from envharness.prompts.alfworld_skill_prompt import build_memory_block
from envharness.reasoning_bank.bank import MemoryItem

_SCENARIO_ID = re.compile(r"scenario_universe_\d+")
INJECTION_MODES = ("skill_md", "prompt")
TEMPLATE = "AGENTS_TEMPLATE.md"
_INSIGHT = re.compile(r"^### Insight \d+: (.*)$", re.MULTILINE)


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


def prompt_template(seed_template: str, items: Sequence[MemoryItem], style: str = "soft") -> str:
    """Seed AGENTS_TEMPLATE.md + the original memory block (``{{`` kept literal for the renderer)."""
    if not items:
        return seed_template
    block = build_memory_block(items, style=style).replace("{{", "{ {").replace("}}", "} }")
    return seed_template.rstrip("\n") + "\n\n" + block


def injected_titles(harness: Path) -> list[str]:
    """Titles a rendered harness gives the student: its memory block, else its SKILL.md files."""
    titles = _INSIGHT.findall((harness / TEMPLATE).read_text(encoding="utf-8")) if (harness / TEMPLATE).is_file() else []
    if titles:
        return titles
    out = []
    for path in sorted(harness.glob("skills/*/SKILL.md")):
        head = next((line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("# ")), "# ")
        out.append(head[2:].strip())
    return out


def render(
    items: Sequence[MemoryItem],
    *,
    seed_dir: Path,
    out_dir: Path,
    policy: Path,
    harness_tool: Path,
    forbidden_ids: Iterable[str] = (),
    injection: str = "skill_md",
    style: str = "soft",
) -> str:
    """Write seed + the items to ``out_dir``, validate with ``harness_tool check``, return the digest."""
    if injection not in INJECTION_MODES:
        raise SkillRenderError(f"unknown injection mode {injection!r}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(seed_dir, out_dir)
    skills_dir = out_dir / "skills"
    for child in skills_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
    expected_template = None
    if injection == "prompt":
        expected_template = prompt_template((seed_dir / TEMPLATE).read_text(encoding="utf-8"), items, style)
        (out_dir / TEMPLATE).write_text(expected_template, encoding="utf-8")
    else:
        taken: set[str] = set()
        for item in items:
            name = slugify(item.title, taken)
            (skills_dir / name).mkdir()
            (skills_dir / name / "SKILL.md").write_text(skill_markdown(item, name), encoding="utf-8")

    blob = "\n".join(p.read_text(encoding="utf-8") for p in skills_dir.rglob("SKILL.md")) + (expected_template or "")
    leaked = [sid for sid in forbidden_ids if sid and sid in blob]
    if _SCENARIO_ID.search(blob) or leaked:
        raise SkillRenderError(f"skills mention scenario identifiers: {leaked[:3]}")
    _assert_only_skills_differ(seed_dir, out_dir, expected_template)
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


def _assert_only_skills_differ(seed_dir: Path, out_dir: Path, expected_template: str | None = None) -> None:
    def files(root: Path) -> set[str]:
        return {p.relative_to(root).as_posix() for p in root.rglob("*")
                if p.is_file() and p.relative_to(root).parts[0] != "skills"}

    seed_files, out_files = files(seed_dir), files(out_dir)
    if seed_files != out_files:
        raise SkillRenderError(f"non-skill files differ: {sorted(seed_files ^ out_files)}")
    changed = [f for f in seed_files if not filecmp.cmp(seed_dir / f, out_dir / f, shallow=False)]
    if expected_template is not None and TEMPLATE in changed:
        if (out_dir / TEMPLATE).read_text(encoding="utf-8") != expected_template:
            raise SkillRenderError(f"{TEMPLATE} is not the seed template plus the memory block")
        changed.remove(TEMPLATE)
    if changed:
        raise SkillRenderError(f"non-skill files changed: {changed}")
