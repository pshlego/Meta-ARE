"""Smoke report: evaluate the plan-v2 section-9 checklist on a finished run directory.

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

# "oracle_function" is a key of every app's public `<app> schema` output (the agent sees it too),
# not oracle data; it may also appear cut short ("oracle_f") where long tool outputs are truncated.
# Every other "oracle" occurrence counts.
LEAK_TERMS = {
    "oracle": re.compile(r"oracle(?!_f(?:unction|unctio|uncti|unct|unc|un|u)?\b)"),
    "match_details": re.compile("match_details"),
    "id_mapping": re.compile("id_mapping"),
    "daemon_judgments": re.compile("daemon_judgments"),
}
# CLI app name -> module under gaia2-cli/cli/gaia2_cli/apps (as in gaia2_cli.env_rules).
_APP_MODULES = {"calendar": "calendar", "contacts": "contacts", "emails": "email", "messages": "messages",
                "chats": "chats", "rent-a-flat": "apartment", "city": "city", "cabs": "cab",
                "shopping": "shopping", "cloud-drive": "files"}
_COMMAND_DEF = re.compile(r'^@cli\.command\("([^"]+)"[^)]*\).*?^def (\w+)\(', re.M | re.S)
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


def command_functions(apps_dir: Path) -> dict[tuple[str, str], str]:
    """(app, subcommand) -> function name logged to events.jsonl (the click callback name)."""
    mapping = {}
    for app, module in _APP_MODULES.items():
        path = apps_dir / f"{module}.py"
        if path.is_file():
            for command, function in _COMMAND_DEF.findall(path.read_text(encoding="utf-8")):
                mapping[(app, command)] = function
    return mapping


def leaked_blocked_writes(scenario_dir: Path, functions: dict[tuple[str, str], str]) -> int:
    """Writes in events.jsonl that only a blocked call could have produced.

    A blocked call exits before the app runs, so for every blocked (app, subcommand)
    the writes of its function cannot exceed the agent's invocations of that
    subcommand minus the blocked ones. Invocations are counted from the exec
    commands (an upper bound: a failed `a && b` chain counts b too), so the check
    never flags a concurrent allowed call of the same subcommand. A loop runs one
    command text many times, so only writes carrying every argument value of some
    blocked call are counted.
    """
    from envharness_gaia2.trace_convert import _exec_command, conversation

    blocked: dict[tuple[str, str], list[list[str] | None]] = {}
    for entry in _jsonl(scenario_dir / "env_rules_applied.jsonl"):
        if entry.get("kind") == "blocked":
            key = (entry.get("app", ""), entry.get("subcommand", ""))
            argv = entry.get("argv")
            values = None if argv is None else [
                tok.split("=", 1)[1] if tok.startswith("--") and "=" in tok else tok
                for tok in argv[1:] if not (tok.startswith("-") and "=" not in tok)]
            blocked.setdefault(key, []).append(values)
    if not blocked:
        return 0

    def could_be(event: dict, values: list[str] | None) -> bool:
        args = {str(v) for v in (event.get("args") or {}).values()}
        return values is None or all(v in args for v in values)

    events = _jsonl(scenario_dir / "events.jsonl")
    commands = [_exec_command(call) for message in conversation(scenario_dir)
                for call in message.get("tool_calls") or [] if isinstance(call, dict)]
    leaked = 0
    for (app, subcommand), calls in blocked.items():
        n_blocked = len(calls)
        function = functions.get((app, subcommand))
        writes = sum(1 for ev in events if ev.get("w") is True and ev.get("fn") == function
                     and any(could_be(ev, values) for values in calls))
        if function is None or writes == 0:
            continue
        call = re.compile(rf"(?<![\w-]){re.escape(app)}\s+{re.escape(subcommand)}(?![\w-])(?![^;&|]*--help)")
        invocations = sum(len(call.findall(command)) for command in commands)
        leaked += max(0, writes - max(0, invocations - n_blocked))
    return leaked


def _results(root: Path) -> list[Path]:
    return sorted(root.rglob("result.json")) if root.is_dir() else []


INTERVENTION_KINDS = ("blocked", "action_rewritten", "transition_modified", "observation_modified")


class Report:
    def __init__(self, run_dir: Path, protected_images: dict[str, str]) -> None:
        self.run = run_dir
        self.protected_images = protected_images
        self.config = _json(run_dir / "config.resolved.json")
        self.state = _json(run_dir / "budget.json")
        self.lineage = _jsonl(run_dir / "lineage.jsonl")
        self.eval = _json(run_dir / "eval" / "summary.json")
        self.rounds = sorted((p for p in (run_dir / "rounds").glob("r*") if p.is_dir()), key=lambda p: int(p.name[1:]))
        repo = Path(self.config.get("config", {}).get("meta_are_repo", ""))
        self.repo = repo
        self.sections: list[tuple[str, bool | None, list[str]]] = []

    def add(self, title: str, ok: bool | None, lines: list[str]) -> None:
        self.sections.append((title, ok, lines))

    # 1 ---------------------------------------------------------------------------
    def seed_eval(self) -> None:
        arm = self.eval.get("nobank", {})
        rows = [r for rs in arm.get("cases", {}).values() for r in rs]
        valid = bool(rows) and all(r["disposition"] in ("success", "task_failure") for r in rows)
        digest_ok = arm.get("harness_digest") == self.state.get("seed_digest") and arm.get("digest_ok", False)
        self.add("1. Seed (nobank) evaluation is valid", valid and digest_ok,
                 [f"- {sid}: {[(r['disposition'], r['success']) for r in rs]}" for sid, rs in arm.get("cases", {}).items()]
                 + [f"- harness_digest == seed digest ({self.state.get('seed_digest')}): {digest_ok}",
                    f"- evidence: {self.run / 'eval' / 'summary.json'}"])

    # 2 + 3 -----------------------------------------------------------------------
    def envrigger_and_judge(self) -> None:
        k = int(self.config.get("config", {}).get("envrigger", {}).get("k_per_candidate", 5))
        lines, ok = [], True
        decisions = 0
        for round_dir in self.rounds:
            for task_dir in sorted((round_dir / "corpus").glob("task_*")):
                traces = _jsonl(task_dir / "traces.jsonl")
                n_base = sum(1 for t in traces if t.get("kind") == "baseline")
                events = [e for e in _jsonl(task_dir / "orchestrator.jsonl")
                          if e.get("kind") in ("task_skipped_passthrough", "mutator_decision", "task_aborted")]
                decisions += len(events)
                summary = ", ".join(f"{e['kind'].replace('mutator_', '')}:{e.get('decision') or (e.get('rationale') or '')[:40]}"
                                    for e in events) or "-"
                kinds = sorted({t.get("kind") for t in traces if t.get("kind") != "baseline"})
                lines.append(f"- {round_dir.name}/{task_dir.name}: baseline={n_base}/{k} decisions=[{summary}] other_kinds={kinds}")
                ok &= n_base == k and (task_dir / "DONE").is_file()
        variant_results = [p for p in _results(self.run / "rounds") if "/cand_" in str(p)]
        with_digest = sum(1 for p in variant_results if _json(p).get("env_rules_digest"))
        applied = [e for p in (self.run / "rounds").rglob("env_rules_applied.jsonl") for e in _jsonl(p)]
        setups = [e for p in (self.run / "rounds").rglob("env_rules_setup.jsonl") for e in _jsonl(p)]
        by_kind = {kind: sum(1 for e in applied if e.get("kind") == kind) for kind in INTERVENTION_KINDS}
        setup_ok = sum(1 for e in setups if e.get("exit_code") == 0)
        lines += [f"- variant rollouts: {len(variant_results)}, with env_rules_digest: {with_digest}",
                  f"- interventions by lever: {by_kind}",
                  f"- Stage (setup) actions: {len(setups)} executed, {setup_ok} succeeded, "
                  f"{len(setups) - setup_ok} failed (designer command errors; see env_rules_setup.jsonl)"]
        ok &= decisions > 0 and with_digest == len(variant_results) and (sum(by_kind.values()) + len(setups)) > 0
        self.add("2. EnvRigger ran (baseline K -> design -> variant rollouts -> decision)", ok, lines)

        functions = command_functions(self.repo / "gaia2-cli" / "cli" / "gaia2_cli" / "apps")
        rollouts = sorted({p.parent for p in (self.run / "rounds").rglob("env_rules_applied.jsonl")
                           if any(e.get("kind") == "blocked" for e in _jsonl(p))})
        leaked = sum(leaked_blocked_writes(d, functions) for d in rollouts)
        setup_logs = list((self.run / "rounds").rglob("env_rules_setup.jsonl"))
        setup_ran = setup_tagged = setup_writes = 0
        for log in setup_logs:
            ids = {str(ev.get("event_id")) for ev in _jsonl(log.parent / "events.jsonl")}
            writes = [ev for ev in _jsonl(log.parent / "events.jsonl")
                      if str(ev.get("event_id", "")).startswith("Event-ENV-setup-") and ev.get("w")]
            setup_writes += len(writes)
            for entry in _jsonl(log):
                if entry.get("exit_code") == 0:
                    setup_ran += 1
                    setup_tagged += f"Event-ENV-setup-{entry.get('index')}" in ids
        rules_errors = [p for p in (self.run / "rounds").rglob("env_rules_errors.log") if p.stat().st_size > 0]
        error_traces = sum(1 for p in (self.run / "rounds").rglob("traces.jsonl") for t in _jsonl(p)
                           if str(t.get("error") or "").startswith("rules_error"))
        monitors = {
            "write shown as failed (T exit_code 0->!=0 on wrote)": sum(
                1 for e in applied if e.get("kind") == "transition_modified" and e.get("wrote")
                and (e.get("exit_code") or [0, 0])[0] == 0 and (e.get("exit_code") or [0, 0])[1] != 0),
            "write output rewritten (T/O on wrote)": sum(
                1 for e in applied if e.get("kind") in ("transition_modified", "observation_modified") and e.get("wrote")),
            "write command rewritten (A)": sum(
                1 for e in applied if e.get("kind") == "action_rewritten" and _is_write(e, functions, self.run)),
        }
        self.add("3. Judge integrity", leaked == 0 and setup_tagged == setup_ran and error_traces >= len(rules_errors),
                 [f"- rollouts with blocked calls: {len(rollouts)}; writes only a blocked call could have produced: {leaked}",
                  f"- successful setup actions: {setup_ran}, logged as Event-ENV-setup-* (ENV) events: {setup_tagged} "
                  f"({setup_writes} setup writes, excluded from the judge's AGENT writes)"
                  + ("" if setup_ran else " -- no successful setup in this run; Stage end-to-end is not exercised here"),
                  f"- non-empty env_rules_errors.log: {len(rules_errors)}; traces marked rules_error: {error_traces}",
                  f"- integrity monitors (observed, not a failure condition): {monitors}"])

    # 4 ---------------------------------------------------------------------------
    def designer_rendering(self) -> None:
        calls = _jsonl(self.run / "designer_calls.jsonl")
        texts = [json.dumps(c.get("messages", [])) for c in calls]
        with_cmd = sum(1 for t in texts if re.search(r"<cmd>[^<]+</cmd>", t))
        empty = sum(t.count("<cmd></cmd>") for t in texts)
        self.add("4. Designer sees real commands", with_cmd > 0 and empty == 0,
                 [f"- designer calls: {len(calls)}, with a non-empty <cmd>: {with_cmd}, empty <cmd>: {empty}"])

    # 5 + 6 -----------------------------------------------------------------------
    def banks_and_injection(self) -> None:
        tool = self.repo / "gaia2-cli" / "shared" / "harness_tool.py"
        policy = self.repo / self.config.get("config", {}).get("harness_policy", "")
        seed = self.run / "seed_harness"
        lines, ok = [], True
        for round_dir in self.rounds:
            stats = _json(round_dir / "bank_build.json")
            for name, st in stats.items():
                lines.append(f"- {round_dir.name} {name}: pairs={st.get('pairs')} empty_pairs={st.get('empty_pairs')} "
                             f"items={st.get('items')} (before dedup {st.get('items_before_dedup')}) tiers={st.get('tiers')}")
            harness = round_dir / "harness"
            check = subprocess.run([sys.executable, str(tool), "check", str(harness), "--policy", str(policy)],
                                   capture_output=True, text=True)
            outside = [p.relative_to(seed).as_posix() for p in seed.rglob("*")
                       if p.is_file() and p.relative_to(seed).parts[0] != "skills"
                       and not filecmp.cmp(p, harness / p.relative_to(seed), shallow=False)]
            bank = _jsonl(self.run / "banks" / f"{'ours_' + round_dir.name}.jsonl")
            dims = {len(item.get("embedding") or []) for item in bank}
            n_skills = len(list(harness.glob("skills/*/SKILL.md")))
            lines.append(f"- {round_dir.name} bank: {len(bank)} items, embedding dims {sorted(dims)}, rendered skills "
                         f"{n_skills}, harness check {'ok' if check.returncode == 0 else check.stderr.strip()}, "
                         f"non-skill diffs {outside or 'none'}")
            ok &= check.returncode == 0 and not outside and n_skills == len(bank) and (not bank or dims == {max(dims)} and max(dims) > 0)
        orig = _jsonl(self.run / "banks" / "orig.jsonl")
        lines.append(f"- orig bank: {len(orig)} items")
        mode = self.config.get("config", {}).get("bank", {}).get("mode", "swebench_cascade")
        source = {"swebench_cascade": "SWE-bench bank_distillation", "alfworld_pair": "ALFWorld induce_pair"}.get(mode, mode)
        self.add(f"5. Skill banks ({source} port)", ok and bool(self.rounds), lines)

        lines, ok = [], True
        for prev, nxt in zip(self.rounds, self.rounds[1:]):
            expected = (prev / "harness.digest").read_text().strip() if (prev / "harness.digest").is_file() else None
            digests = {_json(p).get("harness_digest") for p in _results(nxt / "corpus") if "/baseline/" in str(p)}
            match = digests == {expected}
            lines.append(f"- {nxt.name} corpus ran on {prev.name}'s bank harness ({expected}): {match}")
            ok &= match
        cat_seen = False
        for arm, data in self.eval.items():
            names = [p.parent.name for p in Path(data.get("harness_dir", "")).glob("skills/*/SKILL.md")]
            traces = [p for p in (self.run / "eval" / arm).rglob("trace.jsonl")]
            text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in traces)
            listed = sum(1 for name in names if name in text)
            cat_seen |= bool(re.search(r"skills/[a-z0-9-]+/SKILL\.md", text))
            lines.append(f"- eval {arm}: {data.get('n_skills')} skills, listed in the student's prompt: {listed}/{len(names)}, "
                         f"digest ok: {data.get('digest_ok')}")
            ok &= listed == len(names) and bool(data.get("digest_ok"))
        for record in self.lineage[1:]:
            lines.append(f"- round {record['round']} augment-mode new items: {len(record.get('new_items', []))}")
        lines.append(f"- student read a SKILL.md in eval: {cat_seen}")
        self.add("6. Injection and co-evolution", ok, lines)

    # 7 ---------------------------------------------------------------------------
    def leaks(self) -> None:
        lines, ok = [], True
        for name in ("designer_calls.jsonl", "induce_calls.jsonl"):
            path = self.run / name
            text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
            counts = {term: len(pattern.findall(text)) for term, pattern in LEAK_TERMS.items()}
            counts["oracle_function (public schema key, not counted)"] = text.count("oracle_function")
            ids = len(_SCENARIO_ID.findall(text))
            lines.append(f"- {name}: {counts}, scenario ids: {ids}")
            ok &= not any(counts[term] for term in LEAK_TERMS) and ids == 0
        skills_text = "\n".join(p.read_text() for p in self.run.rglob("skills/*/SKILL.md"))
        skill_ids = len(_SCENARIO_ID.findall(skills_text))
        lines.append(f"- scenario ids in rendered skills: {skill_ids}")
        self.add("7. No oracle/scenario leakage into optimizer inputs or skills", ok and skill_ids == 0, lines)

    # 8-11 ------------------------------------------------------------------------
    def budget_eval_cleanup(self) -> None:
        total = int(self.state.get("rollouts", 0)) + int(self.state.get("eval_rollouts", 0))
        cap = self.state.get("max_rollouts", 200)
        self.add("8. Budget", total <= cap,
                 [f"- rollouts: corpus {self.state.get('rollouts')} + eval {self.state.get('eval_rollouts')} = {total} "
                  f"(cap {cap}); stopped: {self.state.get('stopped')}"])
        rows = ["| arm | skills | valid | success rate | per case |", "|---|---|---|---|---|"]
        for name, data in self.eval.items():
            per_case = {sid: [r["success"] for r in rs] for sid, rs in data.get("cases", {}).items()}
            rows.append(f"| {name} | {data.get('n_skills')} | {data.get('valid')} | {data.get('success_rate')} | {per_case} |")
        expected_arms = {"nobank", "orig", *(f"ours_{r.name}" for r in self.rounds)}
        valid = set(self.eval) == expected_arms and all(data.get("valid") for data in self.eval.values())
        self.add("9. Evaluation (held-out; smoke checks validity only)", valid,
                 rows + [f"- arms expected {sorted(expected_arms)}, got {sorted(self.eval)}"])

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
        roles = {name: len(_jsonl(self.run / f"{name}_calls.jsonl")) for name in ("designer", "induce")}
        elapsed = sum(r.get("elapsed_s", 0) for r in self.lineage)
        self.add("11. Cost and time", None, [
            f"- task agent tokens: input {task_in:,}, output {task_out:,}",
            f"- optimizer calls {roles}; tokens: input {sum(r['input_tokens'] for r in opt):,}, "
            f"output {sum(r['output_tokens'] for r in opt):,} (reasoning {sum(r['reasoning_tokens'] for r in opt):,})",
            f"- rounds wall time: {elapsed / 3600:.2f} h over {len(self.lineage)} rounds (eval not included)",
        ])

    def write(self) -> Path:
        self.seed_eval()
        self.envrigger_and_judge()
        self.designer_rendering()
        self.banks_and_injection()
        self.leaks()
        self.budget_eval_cleanup()
        out = ["# EnvHarness × OpenClaw GAIA2 smoke report (plan v2)", "",
               f"- run: `{self.run}`", f"- Meta-ARE HEAD at start: `{self.config.get('meta_are_head')}`",
               f"- image: `{self.config.get('image')}` ({self.config.get('image_id')})",
               f"- seed digest: `{self.state.get('seed_digest')}`", ""]
        for title, ok, lines in self.sections:
            mark = "PASS" if ok else ("INFO" if ok is None else "FAIL")
            out += [f"## {title} — {mark}", "", *lines, ""]
            print(f"{mark:4}  {title}")
        notes = self.run / "smoke_notes.md"  # operator notes (aborted attempts, resumes), kept across regenerations
        if notes.is_file():
            out += ["## Run notes", "", notes.read_text(encoding="utf-8").strip(), ""]
        path = self.run / "smoke_report.md"
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        return path


def _is_write(entry: dict, functions: dict[tuple[str, str], str], run: Path) -> bool:
    """Whether an intervened subcommand is a state-changing one (seen as a write anywhere in the run)."""
    function = functions.get((entry.get("app"), entry.get("subcommand")))
    if function is None:
        return False
    if function not in _WRITE_CACHE:
        _WRITE_CACHE[function] = any(ev.get("fn") == function and ev.get("w")
                                     for p in run.rglob("events.jsonl") for ev in _jsonl(p))
    return _WRITE_CACHE[function]


_WRITE_CACHE: dict[str, bool] = {}


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
