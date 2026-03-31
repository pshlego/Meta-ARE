#!/usr/bin/env bash
# ============================================================================
# run_hook_experiment_agent.sh — Run val scenarios with agent-type hook handlers
#
# Automatically generates an agent-type variant of hook_from_train_dataset.json
# (replacing all "type": "prompt" with "type": "agent"), then runs val scenarios
# with it.
#
# Usage:
#   bash scripts/run_hook_experiment_agent.sh
# ============================================================================
set -euo pipefail

# ---- Configuration ----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${REPO_ROOT}/.venv/bin/activate"

# Dataset file
VAL_IDS_FILE="${REPO_ROOT}/failure_diagnosis/val_dataset_hook.json"

# Source dataset directory
SCENARIO_SRC_DIR="${REPO_ROOT}/datasets_local/gaia2/mini/validation"

# Original hook config (prompt-based)
HOOK_CONFIG_PROMPT="${REPO_ROOT}/hooks/hook_from_train_dataset.json"

# Output root
OUTPUT_ROOT="${REPO_ROOT}/eval_hook_experiment"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="${OUTPUT_ROOT}/${TIMESTAMP}_agent"

# Model config
MODEL="gpt-5-20250807"
PROVIDER="azure"
REASONING_EFFORT="low"
JUDGE_MODEL="Llama-3.3-70B-Instruct"
JUDGE_PROVIDER="azure"
SCENARIO_TIMEOUT=7200
MAX_CONCURRENT=4
NUM_RUNS=1
TRACE_FORMAT="both"

# ---- Helper functions -------------------------------------------------------

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

create_scenario_dataset() {
    local ids_file="$1"
    local dataset_dir="$2"

    mkdir -p "${dataset_dir}/mini/validation"

    local ids
    ids=$(python3 -c "import json; print('\n'.join(json.load(open('${ids_file}'))))")

    local count=0
    for sid in ${ids}; do
        local src_file
        src_file=$(find "${SCENARIO_SRC_DIR}" -name "*${sid}*" -type f 2>/dev/null | head -1)
        if [[ -z "${src_file}" ]]; then
            log "WARNING: Scenario file not found for ${sid}, skipping"
            continue
        fi
        local fname
        fname=$(basename "${src_file}")
        ln -sf "${src_file}" "${dataset_dir}/mini/validation/${fname}"
        count=$((count + 1))
    done

    log "Created dataset with ${count} scenarios in ${dataset_dir}/mini/validation/"
}

# ---- Main -------------------------------------------------------------------

main() {
    log "Starting agent-hook experiment (val_with_hook_agent)"
    log "Experiment directory: ${EXPERIMENT_DIR}"

    # Validate input files
    for f in "${VAL_IDS_FILE}" "${HOOK_CONFIG_PROMPT}"; do
        if [[ ! -f "${f}" ]]; then
            log "ERROR: Required file not found: ${f}"
            exit 1
        fi
    done

    # Generate agent-type hook config by replacing "type": "prompt" → "type": "agent"
    local hook_config_agent="${EXPERIMENT_DIR}/hook_agent.json"
    mkdir -p "${EXPERIMENT_DIR}"

    python3 -c "
import json, sys

with open('${HOOK_CONFIG_PROMPT}') as f:
    config = json.load(f)

count = 0
for matcher_group in config.get('hooks', {}).get('PreToolUse', []):
    for hook in matcher_group.get('hooks', []):
        if hook.get('type') == 'prompt':
            hook['type'] = 'agent'
            count += 1

with open('${hook_config_agent}', 'w') as f:
    json.dump(config, f, indent=2)

print(f'Converted {count} prompt handler(s) to agent type')
print(f'Saved to: ${hook_config_agent}')
"
    log "Agent hook config generated at: ${hook_config_agent}"

    # Create val dataset
    local val_dataset="${EXPERIMENT_DIR}/datasets/val"
    log "Creating val dataset..."
    create_scenario_dataset "${VAL_IDS_FILE}" "${val_dataset}"
    log ""

    # Run experiment
    local name="val_with_hook_agent"
    local output_dir="${EXPERIMENT_DIR}/${name}"
    local logfile="${EXPERIMENT_DIR}/logs/${name}.log"
    mkdir -p "${output_dir}" "${EXPERIMENT_DIR}/logs"

    log "============================================"
    log "Experiment: ${name}"
    log "  Dataset:  ${val_dataset}"
    log "  Hook:     ${hook_config_agent} (agent type)"
    log "  Output:   ${output_dir}"
    log "  Log:      ${logfile}"
    log "============================================"

    local cmd=(
        are-benchmark run
        --agent default
        --dataset "${val_dataset}"
        --config mini
        --split validation
        --model "${MODEL}"
        --provider "${PROVIDER}"
        --reasoning_effort "${REASONING_EFFORT}"
        --judge_model "${JUDGE_MODEL}"
        --judge_provider "${JUDGE_PROVIDER}"
        --output_dir "${output_dir}"
        --scenario_timeout "${SCENARIO_TIMEOUT}"
        --num_runs "${NUM_RUNS}"
        --max_concurrent_scenarios "${MAX_CONCURRENT}"
        --trace_dump_format "${TRACE_FORMAT}"
        --hook-config "${hook_config_agent}"
    )

    log "Running: ${cmd[*]}"
    "${cmd[@]}" 2>&1 | tee "${logfile}"

    local exit_code=${PIPESTATUS[0]}
    if [[ ${exit_code} -ne 0 ]]; then
        log "WARNING: Experiment ${name} exited with code ${exit_code}"
    else
        log "Experiment ${name} completed successfully"
    fi

    # Summarize results
    if [[ -f "${output_dir}/output.jsonl" ]]; then
        log "Results summary for ${name}:"
        python3 -c "
import json
results = [json.loads(l) for l in open('${output_dir}/output.jsonl')]
total = len(results)
success = sum(1 for r in results if r.get('score', 0) > 0 or r.get('metadata', {}).get('status') == 'success')
print(f'  Total: {total}, Success: {success}, Rate: {success/total*100:.1f}%' if total > 0 else '  No results')
for r in results:
    sid = r.get('task_id', 'unknown')
    score = r.get('score', 'N/A')
    status = r.get('metadata', {}).get('status', 'N/A')
    print(f'  {sid}: score={score}, status={status}')
" 2>/dev/null || log "  (Could not parse results)"
    fi

    # Hook stats
    if [[ -f "${logfile}" ]]; then
        local hook_matches hook_denies hook_allows
        hook_matches=$(grep -c "\[Hook\] Matched" "${logfile}" 2>/dev/null || echo "0")
        hook_denies=$(grep -c "\[Hook\] LLM decision: deny" "${logfile}" 2>/dev/null || echo "0")
        hook_allows=$(grep -c "\[Hook\] LLM decision: allow" "${logfile}" 2>/dev/null || echo "0")
        local agent_sessions
        agent_sessions=$(grep -c "\[Hook\] Spawning GC-SDK agent" "${logfile}" 2>/dev/null || echo "0")
        log "Hook stats for ${name}:"
        log "  Matched: ${hook_matches}, Allow: ${hook_allows}, Deny: ${hook_denies}"
        log "  GC-SDK agent sessions: ${agent_sessions}"
    fi

    log ""
    log "============================================"
    log "EXPERIMENT COMPLETE"
    log "  Log:    ${logfile}"
    log "  Traces: ${output_dir}/"
    log "  Hook config (agent): ${hook_config_agent}"
    log "============================================"
}

main "$@"
