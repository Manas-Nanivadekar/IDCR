#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# IDCR Experiment Runner
#
# Runs all 7 experiments, or a specific one.
#
# Usage:
#   ./run.sh              # Run all experiments
#   ./run.sh --exp 1      # Run only experiment 1
#   ./run.sh --exp 1,3,5  # Run experiments 1, 3, and 5
#   ./run.sh --quick      # Run with reduced parameters for quick testing
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse arguments ──────────────────────────────────────────────────────────
RUN_EXPS="1,2,3,4,5,6,7"
QUICK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --exp)
            RUN_EXPS="$2"
            shift 2
            ;;
        --quick)
            QUICK=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./run.sh [--exp 1,2,3] [--quick]"
            echo ""
            echo "Options:"
            echo "  --exp N,M,...   Run specific experiments (default: all)"
            echo "  --quick         Run with reduced parameters for quick testing"
            echo ""
            echo "Experiments:"
            echo "  1  Submodularity verification"
            echo "  2  Greedy approximation ratio"
            echo "  3  RL vs Greedy comparison"
            echo "  4  Interaction tensor analysis"
            echo "  5  Conformal coverage verification"
            echo "  6  Baseline comparison"
            echo "  7  Pareto frontier analysis"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

IFS=',' read -ra EXPS <<< "$RUN_EXPS"

# ── Check data exists ────────────────────────────────────────────────────────
if [ ! -d "data/generated" ]; then
    echo "❌ Data not generated yet. Run ./setup.sh first."
    exit 1
fi

echo "╔══════════════════════════════════════╗"
echo "║     IDCR Experiment Runner           ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "  Experiments: ${RUN_EXPS}"
echo "  Quick mode:  ${QUICK}"
echo ""

TOTAL=${#EXPS[@]}
CURRENT=0
FAILED=()
TIMES=()

run_exp() {
    local exp_num=$1
    local exp_name=$2
    local exp_script=$3
    local extra_args=${4:-""}

    CURRENT=$((CURRENT + 1))
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${CURRENT}/${TOTAL}] Experiment ${exp_num}: ${exp_name}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local start_time=$(date +%s)

    if uv run python "$exp_script" $extra_args; then
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        TIMES+=("Exp ${exp_num}: ${elapsed}s")
        echo "  ✓ Experiment ${exp_num} completed in ${elapsed}s"
    else
        FAILED+=("$exp_num")
        echo "  ✗ Experiment ${exp_num} FAILED"
    fi
    echo ""
}

# ── Run experiments ──────────────────────────────────────────────────────────

for exp in "${EXPS[@]}"; do
    case $exp in
        1)
            if $QUICK; then
                # Override defaults via env or we modify the script call
                run_exp 1 "Submodularity Verification" \
                    "experiments/exp1_submodularity.py"
            else
                run_exp 1 "Submodularity Verification" \
                    "experiments/exp1_submodularity.py"
            fi
            ;;
        2)
            run_exp 2 "Greedy Approximation Ratio" \
                "experiments/exp2_greedy_approx.py"
            ;;
        3)
            run_exp 3 "RL vs Greedy" \
                "experiments/exp3_rl_vs_greedy.py"
            ;;
        4)
            run_exp 4 "Interaction Tensor Analysis" \
                "experiments/exp4_interaction_tensor.py"
            ;;
        5)
            run_exp 5 "Conformal Coverage" \
                "experiments/exp5_coverage.py"
            ;;
        6)
            run_exp 6 "Baseline Comparison" \
                "experiments/exp6_baselines.py"
            ;;
        7)
            run_exp 7 "Pareto Frontier" \
                "experiments/exp7_pareto.py"
            ;;
        *)
            echo "  ⚠ Unknown experiment: $exp (skipping)"
            ;;
    esac
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║           Run Summary                ║"
echo "╚══════════════════════════════════════╝"
echo ""

for t in "${TIMES[@]}"; do
    echo "  ✓ $t"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "  ✗ Failed experiments: ${FAILED[*]}"
    echo ""
    echo "  Check logs above for error details."
    exit 1
else
    echo ""
    echo "  ✅ All experiments completed successfully!"
    echo ""
    echo "  Results saved to: outputs/"
    echo ""
    echo "  Output files:"
    find outputs/ -name "*.png" -o -name "*.npz" 2>/dev/null | sort | while read f; do
        echo "    📊 $f"
    done
fi
