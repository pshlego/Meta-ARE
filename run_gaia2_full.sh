#!/usr/bin/env bash
# ============================================================================
# run_gaia2_full.sh — Run GAIA2 benchmark across all 10 universes (21–30)
#
# Agent:  default
# Model:  gpt-5-20250807 with reasoning_effort="low"
# Judge:  Llama-3.3-70B-Instruct via azure (CloudGPT)
# ============================================================================
set -euo pipefail

# ---- Activate virtual environment -------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.venv/bin/activate"

# ---- Configuration ----------------------------------------------------------
DATASET_ROOT="/Data2/v-parksungho/Meta-ARE/datasets_local/gaia2"
SPLIT_ROOT="/Data2/v-parksungho/Meta-ARE/datasets_local/gaia2_by_universe"
OUTPUT_ROOT="/Data2/v-parksungho/Meta-ARE/eval_full_gaia2"
LOG_DIR="${OUTPUT_ROOT}/logs"

MODEL="gpt-5-20250807"
PROVIDER="azure"
REASONING_EFFORT="low"

JUDGE_MODEL="Llama-3.3-70B-Instruct"
JUDGE_PROVIDER="azure"

SCENARIO_TIMEOUT=7200
MAX_CONCURRENT=4
NUM_RUNS=1
TRACE_FORMAT="both"

DIMENSIONS=("adaptability" "ambiguity" "execution" "search" "time")
UNIVERSES=(21 22 23 24 25 26 27 28 29 30)

# ---- Step 1: Split dataset by universe (symlinks) --------------------------
split_dataset() {
    echo "============================================"
    echo "Step 1: Splitting dataset by universe"
    echo "============================================"

    for u in "${UNIVERSES[@]}"; do
        for dim in "${DIMENSIONS[@]}"; do
            target_dir="${SPLIT_ROOT}/universe_${u}/${dim}/validation"
            mkdir -p "${target_dir}"

            src_dir="${DATASET_ROOT}/${dim}/validation"
            count=0
            for f in "${src_dir}"/*scenario_universe_${u}_*.json; do
                [ -e "$f" ] || continue
                fname=$(basename "$f")
                if [ ! -e "${target_dir}/${fname}" ]; then
                    ln -s "$f" "${target_dir}/${fname}"
                fi
                count=$((count + 1))
            done
            # echo "  universe_${u}/${dim}: ${count} scenarios"
        done

        # Print per-universe total
        total=$(find "${SPLIT_ROOT}/universe_${u}" -name "*.json" | wc -l)
        echo "  universe_${u}: ${total} scenarios total"
    done

    grand_total=$(find "${SPLIT_ROOT}" -name "*.json" | wc -l)
    echo ""
    echo "  Grand total: ${grand_total} scenario symlinks"
    echo ""
}

# ---- Step 2: Run benchmarks per universe ------------------------------------
run_universe() {
    local u=$1
    local universe_dataset="${SPLIT_ROOT}/universe_${u}"
    local universe_output="${OUTPUT_ROOT}/universe_${u}"
    local logfile="${LOG_DIR}/universe_${u}.log"

    mkdir -p "${universe_output}" "${LOG_DIR}"

    echo "============================================"
    echo "Running universe ${u}"
    echo "  Dataset: ${universe_dataset}"
    echo "  Output:  ${universe_output}"
    echo "  Log:     ${logfile}"
    echo "============================================"

    are-benchmark run \
        --agent default \
        --dataset "${universe_dataset}" \
        --split validation \
        --model "${MODEL}" \
        --provider "${PROVIDER}" \
        --reasoning_effort "${REASONING_EFFORT}" \
        --judge_model "${JUDGE_MODEL}" \
        --judge_provider "${JUDGE_PROVIDER}" \
        --output_dir "${universe_output}" \
        --scenario_timeout "${SCENARIO_TIMEOUT}" \
        --num_runs "${NUM_RUNS}" \
        --max_concurrent_scenarios "${MAX_CONCURRENT}" \
        --trace_dump_format "${TRACE_FORMAT}" \
        --enable-checkpoints \
        2>&1 | tee "${logfile}"

    echo ""
    echo "Universe ${u} complete. Log saved to ${logfile}"
    echo ""
}

# ---- Main -------------------------------------------------------------------
main() {
    echo "GAIA2 Full Benchmark Run"
    echo "  Model:             ${MODEL} (reasoning_effort=${REASONING_EFFORT})"
    echo "  Judge:             ${JUDGE_MODEL} (${JUDGE_PROVIDER})"
    echo "  Universes:         ${UNIVERSES[*]}"
    echo "  Max concurrent:    ${MAX_CONCURRENT}"
    echo "  Scenario timeout:  ${SCENARIO_TIMEOUT}s"
    echo "  Num runs:          ${NUM_RUNS}"
    echo "  Output root:       ${OUTPUT_ROOT}"
    echo ""

    # Split dataset
    split_dataset

    # Run each universe sequentially
    for u in "${UNIVERSES[@]}"; do
        run_universe "${u}"
    done

    echo "============================================"
    echo "All universes complete!"
    echo "Results in: ${OUTPUT_ROOT}"
    echo "============================================"
}

# Allow running a single universe: ./run_gaia2_full.sh 25
if [ $# -ge 1 ]; then
    # Still split dataset (fast, idempotent)
    split_dataset
    for arg in "$@"; do
        run_universe "${arg}"
    done
else
    main
fi
