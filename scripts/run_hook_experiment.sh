#!/usr/bin/env bash
# ============================================================================
# run_hook_experiment.sh — Run hook ablation experiments on train/val datasets
#
# Runs 4 experiments:
#   1. train scenarios WITHOUT hook
#   2. train scenarios WITH hook
#   3. val scenarios WITHOUT hook
#   4. val scenarios WITH hook
#
# Usage:
#   bash scripts/run_hook_experiment.sh
# ============================================================================
set -euo pipefail

# ---- Configuration ----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${REPO_ROOT}/.venv/bin/activate"

# Dataset files (JSON arrays of scenario IDs)
TRAIN_IDS_FILE="${REPO_ROOT}/failure_diagnosis/train_dataset_hook.json"
VAL_IDS_FILE="${REPO_ROOT}/failure_diagnosis/val_dataset_hook.json"

# Source dataset directory (all scenarios live here)
SCENARIO_SRC_DIR="${REPO_ROOT}/datasets_local/gaia2/mini/validation"

# Hook config
HOOK_CONFIG="${REPO_ROOT}/hooks/hook_from_train_dataset.json"

# Output root
OUTPUT_ROOT="${REPO_ROOT}/eval_hook_experiment"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="${OUTPUT_ROOT}/${TIMESTAMP}"

# Model config (matching run_gaia2_full.sh)
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

# Create a temporary dataset directory with symlinks to selected scenarios
create_scenario_dataset() {
    local ids_file="$1"
    local dataset_dir="$2"

    mkdir -p "${dataset_dir}/mini/validation"

    # Read scenario IDs from JSON array
    local ids
    ids=$(python3 -c "import json; print('\n'.join(json.load(open('${ids_file}'))))")

    local count=0
    for sid in ${ids}; do
        # Find the scenario file in the source directory
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

# Run a single experiment
run_experiment() {
    local name="$1"          # e.g., "train_with_hook"
    local dataset_dir="$2"   # path to temporary dataset directory
    local use_hook="$3"      # "true" or "false"

    local output_dir="${EXPERIMENT_DIR}/${name}"
    local logfile="${EXPERIMENT_DIR}/logs/${name}.log"
    mkdir -p "${output_dir}" "${EXPERIMENT_DIR}/logs"

    log "============================================"
    log "Experiment: ${name}"
    log "  Dataset:  ${dataset_dir}"
    log "  Hook:     ${use_hook}"
    log "  Output:   ${output_dir}"
    log "  Log:      ${logfile}"
    log "============================================"

    # Build the command
    local cmd=(
        are-benchmark run
        --agent default
        --dataset "${dataset_dir}"
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
    )

    # Add hook config if enabled
    if [[ "${use_hook}" == "true" ]]; then
        cmd+=(--hook-config "${HOOK_CONFIG}")
    fi

    # Run and log
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

    # Count hook activations if hook was used
    if [[ "${use_hook}" == "true" && -f "${logfile}" ]]; then
        local hook_matches hook_denies hook_allows
        hook_matches=$(grep -c "\[Hook\] Matched" "${logfile}" 2>/dev/null || echo "0")
        hook_denies=$(grep -c "\[Hook\] LLM decision: deny" "${logfile}" 2>/dev/null || echo "0")
        hook_allows=$(grep -c "\[Hook\] LLM decision: allow" "${logfile}" 2>/dev/null || echo "0")
        log "Hook stats for ${name}:"
        log "  Matched: ${hook_matches}, Allow: ${hook_allows}, Deny: ${hook_denies}"
    fi

    echo ""
}

# ---- Main -------------------------------------------------------------------

main() {
    log "Starting hook ablation experiment"
    log "Experiment directory: ${EXPERIMENT_DIR}"
    log "Hook config: ${HOOK_CONFIG}"
    log ""

    # Validate input files
    for f in "${TRAIN_IDS_FILE}" "${VAL_IDS_FILE}" "${HOOK_CONFIG}"; do
        if [[ ! -f "${f}" ]]; then
            log "ERROR: Required file not found: ${f}"
            exit 1
        fi
    done

    # Create temporary dataset directories
    local train_dataset="${EXPERIMENT_DIR}/datasets/train"
    local val_dataset="${EXPERIMENT_DIR}/datasets/val"

    log "Creating train dataset..."
    create_scenario_dataset "${TRAIN_IDS_FILE}" "${train_dataset}"
    log ""

    log "Creating val dataset..."
    create_scenario_dataset "${VAL_IDS_FILE}" "${val_dataset}"
    log ""

    # Run 4 experiments
    log "========== EXPERIMENT 1/4: Train WITHOUT hook =========="
    run_experiment "train_no_hook" "${train_dataset}" "false"

    log "========== EXPERIMENT 2/4: Train WITH hook =========="
    run_experiment "train_with_hook" "${train_dataset}" "true"

    log "========== EXPERIMENT 3/4: Val WITHOUT hook =========="
    run_experiment "val_no_hook" "${val_dataset}" "false"

    log "========== EXPERIMENT 4/4: Val WITH hook =========="
    run_experiment "val_with_hook" "${val_dataset}" "true"

    # ---- Final comparison report -------------------------------------------
    log ""
    log "============================================"
    log "EXPERIMENT COMPLETE — Summary"
    log "============================================"
    log ""

    python3 - <<'PYEOF'
import json
import os
import sys

experiment_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EXPERIMENT_DIR", "")
if not experiment_dir:
    print("No experiment dir")
    sys.exit(0)

experiments = ["train_no_hook", "train_with_hook", "val_no_hook", "val_with_hook"]
results = {}

for exp in experiments:
    output_file = os.path.join(experiment_dir, exp, "output.jsonl")
    if not os.path.exists(output_file):
        results[exp] = {"total": 0, "success": 0, "scenarios": {}}
        continue
    scenarios = {}
    with open(output_file) as f:
        for line in f:
            r = json.loads(line)
            sid = r.get("task_id", "unknown")
            score = r.get("score", 0)
            status = r.get("metadata", {}).get("status", "unknown")
            scenarios[sid] = {"score": score, "status": status}
    total = len(scenarios)
    success = sum(1 for s in scenarios.values() if s["score"] > 0 or s["status"] == "success")
    results[exp] = {"total": total, "success": success, "scenarios": scenarios}

# Print comparison table
print(f"{'Experiment':<25} {'Total':>6} {'Success':>8} {'Rate':>8}")
print("-" * 50)
for exp in experiments:
    r = results[exp]
    rate = f"{r['success']/r['total']*100:.1f}%" if r['total'] > 0 else "N/A"
    print(f"{exp:<25} {r['total']:>6} {r['success']:>8} {rate:>8}")

# Print per-scenario comparison
print()
print("Per-scenario comparison (score):")
all_sids = set()
for exp in experiments:
    all_sids.update(results[exp]["scenarios"].keys())

header = f"{'Scenario':<45}"
for exp in experiments:
    short = exp.replace("_no_hook", "-").replace("_with_hook", "+")
    header += f" {short:>10}"
print(header)
print("-" * len(header))

for sid in sorted(all_sids):
    row = f"{sid:<45}"
    for exp in experiments:
        s = results[exp]["scenarios"].get(sid, {})
        score = s.get("score", "—")
        if isinstance(score, float):
            score = f"{score:.1f}"
        row += f" {str(score):>10}"
    print(row)

# Save comparison as JSON
comparison_path = os.path.join(experiment_dir, "comparison.json")
with open(comparison_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nComparison saved to {comparison_path}")
PYEOF

    log ""
    log "All logs: ${EXPERIMENT_DIR}/logs/"
    log "All traces: ${EXPERIMENT_DIR}/{train,val}_{no,with}_hook/"
    log "Comparison: ${EXPERIMENT_DIR}/comparison.json"
}

# Export EXPERIMENT_DIR for the python script
export EXPERIMENT_DIR
main "$@"
