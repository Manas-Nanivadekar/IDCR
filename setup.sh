#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# IDCR Setup Script
#
# Sets up the environment and generates all synthetic data.
# Run once before running experiments.
#
# Usage:  chmod +x setup.sh && ./setup.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════╗"
echo "║        IDCR Environment Setup        ║"
echo "╚══════════════════════════════════════╝"

# ── 1. Check Python & uv ─────────────────────────────────────────────────────
echo ""
echo "━━━ Step 1: Checking prerequisites ━━━"

if ! command -v uv &>/dev/null; then
    echo "❌ 'uv' not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "  ✓ uv found: $(uv --version)"

if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 not found."
    exit 1
fi
echo "  ✓ python3 found: $(python3 --version)"

# ── 2. Install dependencies ──────────────────────────────────────────────────
echo ""
echo "━━━ Step 2: Installing dependencies ━━━"
uv sync
echo "  ✓ Dependencies installed"

# ── 3. Verify installation ───────────────────────────────────────────────────
echo ""
echo "━━━ Step 3: Verifying installation ━━━"
uv run python -c "
import numpy, scipy, torch, matplotlib, seaborn, tqdm, hydra
print(f'  numpy={numpy.__version__}')
print(f'  scipy={scipy.__version__}')
print(f'  torch={torch.__version__}')
print(f'  matplotlib={matplotlib.__version__}')
"
echo "  ✓ All core packages verified"

# ── 4. Run tests ─────────────────────────────────────────────────────────────
echo ""
echo "━━━ Step 4: Running unit tests ━━━"
uv run pytest tests/ -v --tb=short
echo "  ✓ All tests passed"

# ── 5. Create output directories ─────────────────────────────────────────────
echo ""
echo "━━━ Step 5: Creating directories ━━━"
mkdir -p data/generated/{splits,ground_truth,synergy}
mkdir -p data/llm_cache
mkdir -p data/generated_docs
mkdir -p outputs/{exp1_submodularity,exp2_greedy_approx,exp3_rl_vs_greedy}
mkdir -p outputs/{exp4_interaction_tensor,exp5_coverage,exp6_baselines,exp7_pareto}
echo "  ✓ Directories created"

# ── 6. Generate synthetic data ────────────────────────────────────────────────
echo ""
echo "━━━ Step 6: Generating synthetic data ━━━"
uv run python scripts/generate_data.py

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║      ✅ Setup complete!              ║"
echo "║                                      ║"
echo "║  Next: ./run.sh                      ║"
echo "║    or: ./run.sh --exp 1              ║"
echo "╚══════════════════════════════════════╝"
