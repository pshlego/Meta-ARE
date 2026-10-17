# envharness_gaia2 — EnvHarness on GAIA2 with OpenClaw as the student

A port of the original EnvHarness pipeline (arXiv 2608.19880; envharness `fab7d57`,
SWE-bench variant) to GAIA2, following plan v2
(`/mnt/data2/v-parksungho_icml/ENVHARNESS_OPENCLAW_SKILLBANK_PLAN_v2.md`): the original code is
followed as is; only the bridge, the designer's domain block and the domain wording of the
induction prompt are GAIA2-specific.

| EnvHarness (original) | here |
|---|---|
| Orchestrator, LLMHarnessAgent (`_DEFAULT_SYSTEM` + `extra_instructions`), DifficultyZone, CappedAdaptive | used unchanged; only the two rollout functions run K OpenClaw rollouts through `gaia2-runner` (`pass_at=K`) |
| Contract `Rules` (A transform/reject, T, O) | `gaia2_cli/env_rules.py` around every app CLI in the container: same hook contract, stdout/stderr/exit code as the response |
| Stage `Setup` (`in_env_actions`) | `gaia2_cli/env_setup.py`: app CLI calls run before the agent as `Event-ENV-setup-*` events (the judge compares AGENT writes only) |
| Chain `Link` | not used (as in every original benchmark) |
| domain constraints | `prompts/designer_extra.md` (SWE-bench PHASE 1–4 format, GRADING INTEGRITY); only security is enforced in code (`rules_check.py`) |
| `bank_distillation` (cascade ours / intra-baseline orig, paired-diff atomic) | `bank.py` + `prompts/induce_paired_diff_atomic.md` |
| eval `skill_dir` mode | the whole bank rendered into OpenClaw `skills/` (no retrieval); arms nobank / orig / ours_r* |
| co-evolution ("round after round"; no driver upstream) | `coevolution.rounds`: round r's student carries bank_{r-1}; round r's bank adds the augment-mode (`--existing-bank`) lessons; `rounds: 1` = the original single pass |

## Run

```bash
cd gaia2-cli/envharness_gaia2
uv sync --extra dev && uv run pytest            # no docker / API needed
RUN_ID=envh2-oc-smoke-$(date -u +%Y%m%dT%H%M%SZ)
nohup uv run python -m envharness_gaia2.coevolve --config configs/smoke_cloudgpt.yaml --run-id $RUN_ID \
  > /mnt/data2/v-parksungho_icml/working_dir/$RUN_ID.log 2>&1 &
# continue an interrupted run (tasks, rounds and eval arms commit with DONE markers):
uv run python -m envharness_gaia2.coevolve --config configs/smoke_cloudgpt.yaml --run-id $RUN_ID --resume
uv run python -m envharness_gaia2.report --run-dir <run_root>/$RUN_ID \
  --protected-image localhost/gaia2-oc:7279049a322b=sha256:1a7d3e5c... --protected-image localhost/gaia2-oc:latest=sha256:1a7d3e5c...
```

Run directory: `seed_harness/`, `rounds/r<r>/{corpus/task_<i>/{baseline,cand_<n>_<id>}/attempt-*/output/...,
traces.jsonl, orchestrator.jsonl, DONE}, bank_build.json, harness/, DONE}`, `banks/{ours_r<r>,ours_r<r>_new,orig}.jsonl`,
`eval/{nobank,orig,ours_r<r>}/`, `lineage.jsonl`, `designer_calls.jsonl`, `induce_calls.jsonl`,
`optimizer_usage.jsonl`, `budget.json`, `result.json`, `smoke_report.md`.

Optimizer inputs never include oracle data: trace conversion reads only `result.json`,
`trace.jsonl` and the env-rules logs, and judge failure reasons are reduced to categories.

## Plan v2 pre-checks (2026-10-17)

| ID | Result |
|---|---|
| Q0 | No other runs, no TB2 watchdog, no gaia2 containers; protected `:7279049a322b` = `:latest` = `sha256:1a7d3e5c…`. |
| Q1 | Copilot `/v1/embeddings` `text-embedding-3-small`: 1536-dim vectors. |
| Q2 | Docker (`scenario_universe_29_0wlkxc`, an image built from the same env-rules code as `envh-097a373eab7f`): a Setup `contacts add-new-contact` ran at sim time 2024-10-15 07:00:00, logged as `Event-ENV-setup-0` (write), no eventd errors; the judge's verdict concerned only the reply content (no tool-count mismatch from the setup write). Found and fixed: `prepare_faketime_file` was not re-runnable under `fs.protected_regular`. |
| Q3 | Same run: A argv rewrite (`chats list-recent-conversations --limit 3`) executed, T turned the first `contacts get-contacts` into exit 1 with a transient error; both recorded with argv; empty `env_rules_errors.log`. |
| Q4 | OpenClaw 2026.4.1 defaults: 150 skills / 30000 chars in the prompt (`DEFAULT_MAX_SKILLS_IN_PROMPT`, `DEFAULT_MAX_SKILLS_PROMPT_CHARS`); the smoke report checks every rendered skill is listed. |
| Q5 | Induction over the Responses API: `reasoning_effort=low` parses; gpt-5.5 rejects `minimal`. |

## v2 smoke result (2026-10-17)

`envh2-oc-smoke-20261017T103232Z` (2 rounds × 6 train tasks, K=5, max_k 1): all 10 checks PASS
(11 = cost info); 110 corpus + 4 eval rollouts (cap 200); 1 accepted candidate; levers used A
(block/argv rewrite), T and S0; banks ours_r1 11 → ours_r2 12 (augment), orig 13; round 2 ran on
bank_1 (digest checked, all skills listed in the prompt). Report with run notes (two aborted
attempts and their fixes `663ab77`, `2adad8a`; designer observations):
`reports/envh2-oc-smoke-20261017T103232Z.md`. Stage end to end in the loop (scripted designer,
`0wlkxc`, K=2): both setup writes ran as `Event-ENV-setup-*`, rollouts judged successful.

## v1 Phase 0 checks (2026-10-16)

| ID | Result |
|---|---|
| P0 | No other AutoSaddler run, watchdog not running, no gaia2 containers. Meta-ARE `feat/gaia2-openclaw-harness` @ `7279049` (pushed to origin); images `:7279049a322b` = `:latest` = `sha256:1a7d3e5c…`. |
| P1 | Copilot `http://127.0.0.1:4148/v1` does **not** serve gpt-5.5 on `/chat/completions` (`unsupported_api_for_model`). The Responses API works with function tools and `reasoning.effort=xhigh` → `optimizer_llm.wire_api: responses`, `designer.ResponsesClient`. |
| P2 | ENV actions always carry `GAIA2_EVENT_ID` (eventd → adapter `execute_cli_action` → `/usr/local/bin/<app>`); agent calls go through setuid `gaia2-exec`, which now clears `GAIA2_EVENT_ID` so the agent cannot forge it or bypass rules. |
| P3 | `/var/gaia2` is `gaia2:gaia2 0700`; rules live in `/var/gaia2/env_rules` (0700) — unreadable by `agent`. |
| P4 | App errors: plain text on stderr, exit 1 (`cli_error`); click usage errors exit 2. Blocked calls mimic the former. |
| P5 | Containers are `gaia2-<scenario>-<uuid8>`; added `GAIA2_CONTAINER_LABELS` so each runner invocation is labelled `envh.<run-id>=<hash>` and only its containers are removed. **Note:** `concurrency = 1` makes gaia2-runner use fixed host ports 8090/18789 — two such runners collide; the pipeline requires `max_concurrent >= 2`. |
| P6 | `trace.jsonl` last successful call holds the whole conversation (system, user with OpenClaw's sender envelope, assistant `exec` tool calls, tool outputs); fixture in `tests/fixtures/` (oracle files excluded; `daemon_judgments.jsonl` there is a decoy used to assert it is never opened). |

Env-rules docker check (4-5, `scenario_universe_29_0wlkxc`): without rules `success=true`
with the seed digest and no env-rules artifacts; with example rules (first `contacts` call
refused, long `messages`/`emails` lists cut to 3) `success=true`, `env_rules_applied.jsonl`
recorded 1 blocked + 2 modified, empty `env_rules_errors.log`, and the agent's conversation
never mentions the rules.

## v1 smoke result (2026-10-16)

`envh-oc-smoke-20261016T104706Z`: 2 rounds × 2 batches, 93 co-evolution + 2 eval rollouts (cap 200),
all 10 checks PASS (11 = cost info); seed → 8 skills; val_smoke seed 0/1, final 0/1 (smoke scale, validity only).
Report with run notes (aborted first attempt, mid-run fix + resume): `reports/envh-oc-smoke-20261016T104706Z.md`.
