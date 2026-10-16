# envharness_gaia2 — EnvHarness skill-bank co-evolution for OpenClaw on GAIA2

Reproduces the EnvHarness procedure (arXiv 2608.19880) on GAIA2 with OpenClaw as the
student's harness. Only the OpenClaw harness's `skills/` directory evolves; everything
else stays the seed (`gaia2-cli/harness/openclaw`).

| EnvHarness | here |
|---|---|
| student = LLM + skill bank | OpenClaw (seed harness) + `skills/<name>/SKILL.md` |
| Rules (A/T/O) | `gaia2_cli/env_rules.py` inside the container: `filter_action` (block before execution) and `filter_observation` (rewrite read outputs). No `modify_transition` (would corrupt judged writes). |
| Setup (S0) | disabled (`allow_setup: false`) |
| EnvRigger | `Gaia2Orchestrator` = envharness `Orchestrator`; K rollouts = one `gaia2-runner` call with `pass_at=K` |
| acceptance band | K=3, band [0.3, 0.7]; in-band baselines are skipped |
| skill induction | cascade (Tier1 accepted / Tier2 last exploration / Tier3 baseline, success-only fallback) + paired-diff prompts |
| bank update | ≤ 3 new skills per batch, consolidated above `max_skills`; rendered to `skills/` (no retrieval: OpenClaw lists all skills) |

## Run

```bash
cd gaia2-cli/envharness_gaia2
uv sync --extra dev && uv run pytest            # no docker / API needed
RUN_ID=envh-oc-smoke-$(date -u +%Y%m%dT%H%M%SZ)
nohup uv run python -m envharness_gaia2.coevolve --config configs/smoke_cloudgpt.yaml --run-id $RUN_ID \
  > /mnt/data2/v-parksungho_icml/working_dir/$RUN_ID.log 2>&1 &
# continue an interrupted run:
uv run python -m envharness_gaia2.coevolve --config configs/smoke_cloudgpt.yaml --run-id $RUN_ID --resume
# checklist report:
uv run python -m envharness_gaia2.report --run-dir <run_root>/$RUN_ID \
  --protected-image localhost/gaia2-oc:7279049a322b=sha256:1a7d3e5c... --protected-image localhost/gaia2-oc:latest=sha256:1a7d3e5c...
```

Run directory: `config.resolved.json`, `traces.jsonl`, `rounds/r{R}/b{B}/{task_<i>/{baseline,cand_<n>_<id>}/attempt-*/output/..., induce.jsonl, selected.json, harness/, harness.digest, orchestrator.jsonl, DONE}`, `bank*.jsonl`, `lineage.jsonl`, `designer_calls.jsonl`, `induce_calls.jsonl`, `optimizer_usage.jsonl`, `budget.json`, `final_eval/`, `smoke_report.md`.

Optimizer LLM inputs never include oracle data: trace conversion reads only `result.json`,
`trace.jsonl` and `env_rules_applied.jsonl`, and judge failure reasons are reduced to
categories (`tool_count_mismatch`, `action_mismatch`, ...).

## Phase 0 checks (2026-10-16)

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
