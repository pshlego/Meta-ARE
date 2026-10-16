"""Smoke report: evaluate the section-9 checklist on a finished run directory.

    uv run python -m envharness_gaia2.report --run-dir <run> [--protected-image TAG=ID ...]

Writes ``smoke_report.md`` (one section per check, with evidence paths) and
prints a one-line verdict per check.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import re
import subprocess
import sys
from pathlib import Path

LEAK_TERMS = ("oracle", "match_details", "id_mapping", "daemon_judgments")
_SCENARIO_ID = re.compile(r"scenario_universe_\d+_[0-9a-z]+")


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _results(root: Path) -> list[Path]:
    return sorted(root.rglob("result.json")) if root.is_dir() else []


class Report:
    def __init__(self, run_dir: Path, protected_images: dict[str, str]) -> None:
        self.run = run_dir
        self.protected_images = protected_images
        self.config = _json(run_dir / "config.resolved.json")
        self.state = _json(run_dir / "budget.json")
        self.lineage = _jsonl(run_dir / "lineage.jsonl")
        self.sections: list[tuple[str, bool | None, list[str]]] = []

    def add(self, title: str, ok: bool | None, lines: list[str]) -> None:
        self.sections.append((title, ok, lines))

    # 1 ---------------------------------------------------------------------------
    def seed_eval(self) -> None:
        summary = _json(self.run / "final_eval" / "summary.json")
        seed = summary.get("seed", {})
        rows = [r for rs in seed.get("cases", {}).values() for r in rs]
        valid = all(r["disposition"] in ("success", "task_failure") for r in rows) and bool(rows)
        digest_ok = all(r.get("harness_digest") == self.state.get("seed_digest") for r in rows)
        self.add("1. Seed evaluation is valid", valid and digest_ok,
                 [f"- {sid}: {[(r['disposition'], r['success']) for r in rs]}" for sid, rs in seed.get("cases", {}).items()]
                 + [f"- harness_digest == seed digest ({self.state.get('seed_digest')}): {digest_ok}",
                    f"- evidence: {self.run / 'final_eval' / 'summary.json'}"])

    # 2 + 3 -----------------------------------------------------------------------
    def envrigger_and_judge(self) -> None:
        round1 = self.run / "rounds" / "r1"
        traces = _jsonl(self.run / "traces.jsonl")
        lines, ok = [], True
        tasks = sorted({t for r in self.lineage if r["round"] == 1 for t in r["task_ids"]})
        for task in tasks:
            n_base = sum(1 for t in traces if t.get("rollout_seed") == task and t.get("kind") == "baseline")
            lines.append(f"- task {task}: baseline traces (all rounds) = {n_base}")
            ok &= n_base >= 3
        decisions = []
        for batch in sorted(p for p in round1.iterdir() if p.is_dir()) if round1.is_dir() else []:
            for event in _jsonl(batch / "orchestrator.jsonl"):
                if event.get("kind") in ("task_skipped_passthrough", "mutator_decision", "task_aborted", "candidate_proposed"):
                    decisions.append(f"{batch.name}: {event['kind']} {event.get('decision') or event.get('rationale', '')[:80]}")
        lines += [f"  - {d}" for d in decisions] or ["  - (no designer events)"]
        variant_results = [p for p in _results(round1) if "/cand_" in str(p)]
        with_digest = sum(1 for p in variant_results if _json(p).get("env_rules_digest"))
        applied = [(p, e) for p in round1.rglob("env_rules_applied.jsonl") for e in _jsonl(p)]
        interventions = [e for _, e in applied if e.get("kind") in ("blocked", "modified")]
        lines += [f"- variant rollouts: {len(variant_results)}, with env_rules_digest: {with_digest}",
                  f"- interventions (blocked/modified): {len(interventions)} "
                  f"({sum(e['kind'] == 'blocked' for e in interventions)} blocked)"]
        ok &= bool(decisions) and with_digest == len(variant_results) and bool(interventions)
        self.add("2. EnvRigger ran (baseline -> design -> variant rollouts -> decision)", ok, lines)

        violations = [e for _, e in applied if e.get("kind") == "write_observation_ignored"]
        leaked_blocks = 0
        for path, entry in applied:
            if entry.get("kind") != "blocked":
                continue
            events = _jsonl(path.parent / "events.jsonl")
            leaked_blocks += sum(1 for ev in events if abs(float(ev.get("t", 0)) - float(entry["t"])) < 0.05
                                 and str(ev.get("app", "")).lower().startswith(entry["app"][:4]))
        errors = [p for p in self.run.rglob("env_rules_errors.log") if p.stat().st_size > 0]
        self.add("3. Judge integrity", leaked_blocks == 0 and not errors,
                 [f"- blocked calls with a matching events.jsonl entry: {leaked_blocks}",
                  f"- ignored write-output rewrites: {len(violations)}",
                  f"- non-empty env_rules_errors.log: {[str(p) for p in errors] or 'none'}"])

    # 4 ---------------------------------------------------------------------------
    def designer_rendering(self) -> None:
        calls = _jsonl(self.run / "designer_calls.jsonl")
        texts = [json.dumps(c.get("messages", [])) for c in calls]
        with_cmd = sum(1 for t in texts if re.search(r"<cmd>[^<]+</cmd>", t))
        empty_action = sum(t.count("<action></action>") for t in texts)
        self.add("4. Designer sees real commands", with_cmd > 0 and empty_action == 0,
                 [f"- designer calls: {len(calls)}, with non-empty <cmd>: {with_cmd}, empty <action>: {empty_action}"])

    # 5 + 6 -----------------------------------------------------------------------
    def bank_and_coevolution(self) -> None:
        tool = Path(self.config.get("config", {}).get("meta_are_repo", "")) / "gaia2-cli" / "shared" / "harness_tool.py"
        policy = Path(self.config.get("config", {}).get("meta_are_repo", "")) / self.config.get("config", {}).get("harness_policy", "")
        seed = self.run / "seed_harness"
        lines, ok = [], True
        for record in self.lineage:
            batch = self.run / "rounds" / f"r{record['round']}" / f"b{record['batch']}"
            selected = json.loads((batch / "selected.json").read_text()) if (batch / "selected.json").is_file() else []
            harness = batch / "harness"
            check = subprocess.run([sys.executable, str(tool), "check", str(harness), "--policy", str(policy)],
                                   capture_output=True, text=True)
            outside = [p.relative_to(seed).as_posix() for p in seed.rglob("*")
                       if p.is_file() and p.relative_to(seed).parts[0] != "skills"
                       and not filecmp.cmp(p, harness / p.relative_to(seed), shallow=False)]
            lines.append(f"- r{record['round']}b{record['batch']}: selected={len(selected)} n_skills={record['n_skills']} "
                         f"consolidated={record['consolidated']} check={'ok' if check.returncode == 0 else check.stderr.strip()} "
                         f"non-skill diffs={outside or 'none'}")
            ok &= 0 <= len(selected) <= 3 and check.returncode == 0 and not outside
        self.add("5. Skill bank updates", ok and bool(self.lineage), lines)

        lines, ok = [], True
        names_seen = cat_seen = False
        for prev, nxt in zip(self.lineage, self.lineage[1:]):
            batch = self.run / "rounds" / f"r{nxt['round']}" / f"b{nxt['batch']}"
            digests = {_json(p).get("harness_digest") for p in _results(batch) if "/baseline/" in str(p)}
            match = digests == {prev["skills_digest"]}
            ok &= match
            lines.append(f"- r{nxt['round']}b{nxt['batch']} baseline harness_digest == r{prev['round']}b{prev['batch']} snapshot: {match}")
            skill_names = [p.parent.name for p in (self.run / "rounds" / f"r{prev['round']}" / f"b{prev['batch']}" / "harness" / "skills").glob("*/SKILL.md")]
            for trace in batch.rglob("trace.jsonl"):
                text = trace.read_text(encoding="utf-8", errors="replace")
                if skill_names and all(name in text for name in skill_names[:1]):
                    names_seen = True
                if re.search(r"cat [^\"]*skills/[a-z0-9-]+/SKILL\.md", text):
                    cat_seen = True
        lines += [f"- skill names present in later system prompts: {names_seen}",
                  f"- agent read a SKILL.md (cat .../SKILL.md): {cat_seen}"]
        self.add("6. Co-evolution (next batch runs on the updated skills)", ok and names_seen and len(self.lineage) > 1, lines)

    # 7 ---------------------------------------------------------------------------
    def leaks(self) -> None:
        lines, ok = [], True
        for name in ("designer_calls.jsonl", "induce_calls.jsonl"):
            path = self.run / name
            text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
            counts = {term: text.count(term) for term in LEAK_TERMS}
            ids = len(_SCENARIO_ID.findall(text))
            lines.append(f"- {name}: {counts}, scenario ids: {ids}")
            ok &= not any(counts.values()) and ids == 0
        skills_text = "\n".join(p.read_text() for p in self.run.rglob("harness/skills/*/SKILL.md"))
        skill_ids = len(_SCENARIO_ID.findall(skills_text))
        lines.append(f"- scenario ids in rendered skills: {skill_ids}")
        self.add("7. No oracle/scenario leakage into optimizer inputs or skills", ok and skill_ids == 0, lines)

    # 8-11 ------------------------------------------------------------------------
    def budget_eval_cleanup(self) -> None:
        total = int(self.state.get("rollouts", 0)) + int(self.state.get("eval_rollouts", 0))
        self.add("8. Budget", total <= self.state.get("max_rollouts", 200) and self.state.get("batches", 0) <= self.state.get("max_iterations", 5),
                 [f"- rollouts: co-evolution {self.state.get('rollouts')} + final eval {self.state.get('eval_rollouts')} = {total} "
                  f"(cap {self.state.get('max_rollouts')}); batches {self.state.get('batches')} (cap {self.state.get('max_iterations')})"])
        summary = _json(self.run / "final_eval" / "summary.json")
        rows = ["| harness | valid rollouts | success rate | per case |", "|---|---|---|---|"]
        for name, data in summary.items():
            per_case = {sid: [r["success"] for r in rs] for sid, rs in data.get("cases", {}).items()}
            rows.append(f"| {name} | {data.get('valid')} | {data.get('success_rate')} | {per_case} |")
        valid = all(data.get("valid") for data in summary.values()) and set(summary) == {"seed", "final"}
        self.add("9. Final evaluation (val_smoke)", valid, rows)

        label = f"label=envh.{self.run.name}"
        left = subprocess.run(["sg", "docker", "-c", f"docker ps -aq --filter {label}"], capture_output=True, text=True).stdout.split()
        lines = [f"- containers left with {label}: {len(left)}"]
        ok = not left
        for tag, expected in self.protected_images.items():
            actual = subprocess.run(["sg", "docker", "-c", f"docker image inspect {tag} --format '{{{{.Id}}}}'"],
                                    capture_output=True, text=True).stdout.strip()
            lines.append(f"- {tag}: {actual} (expected {expected}) -> {actual == expected}")
            ok &= actual == expected
        self.add("10. Cleanup and protected image tags", ok, lines)

        task_in = task_out = 0
        for trace in self.run.rglob("trace.jsonl"):
            for row in _jsonl(trace):
                raw = row.get("raw_response")
                try:
                    usage = (json.loads(raw) if isinstance(raw, str) else raw or {}).get("usage") or {}
                except (json.JSONDecodeError, AttributeError):
                    usage = {}
                task_in += int(usage.get("prompt_tokens") or 0)
                task_out += int(usage.get("completion_tokens") or 0)
        opt = _jsonl(self.run / "optimizer_usage.jsonl")
        elapsed = sum(r.get("elapsed_s", 0) for r in self.lineage)
        self.add("11. Cost and time", None, [
            f"- task agent tokens: input {task_in:,}, output {task_out:,}",
            f"- optimizer tokens ({len(opt)} calls): input {sum(r['input_tokens'] for r in opt):,}, "
            f"output {sum(r['output_tokens'] for r in opt):,} (reasoning {sum(r['reasoning_tokens'] for r in opt):,})",
            f"- co-evolution wall time: {elapsed / 3600:.2f} h over {len(self.lineage)} batches",
        ])

    def write(self) -> Path:
        self.seed_eval()
        self.envrigger_and_judge()
        self.designer_rendering()
        self.bank_and_coevolution()
        self.leaks()
        self.budget_eval_cleanup()
        out = ["# EnvHarness × OpenClaw GAIA2 smoke report", "",
               f"- run: `{self.run}`", f"- Meta-ARE HEAD: `{self.config.get('meta_are_head')}`",
               f"- image: `{self.config.get('image')}` ({self.config.get('image_id')})",
               f"- seed digest: `{self.state.get('seed_digest')}`, final digest: `{self.state.get('harness_digest')}`", ""]
        for title, ok, lines in self.sections:
            mark = "PASS" if ok else ("INFO" if ok is None else "FAIL")
            out += [f"## {title} — {mark}", "", *lines, ""]
            print(f"{mark:4}  {title}")
        path = self.run / "smoke_report.md"
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--protected-image", action="append", default=[], help="TAG=IMAGE_ID")
    args = parser.parse_args(argv)
    protected = dict(item.split("=", 1) for item in args.protected_image)
    print(Report(Path(args.run_dir), protected).write())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
