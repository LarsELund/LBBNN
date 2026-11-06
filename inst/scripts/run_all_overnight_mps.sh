#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p results_overnight/{logic,kepler,kepler_real,nonlinear}

echo "=== Starting Full Overnight Symbolic LBBNN Benchmark on MPS ==="

# 1. Bayesian Logic Regression (100 seeds)
echo "[1/4] Running Bayesian Logic Regression (100 seeds)..."
Rscript inst/scripts/run_benchmark_cli.R --benchmark logic --seeds 1:100 --workers 1 --device mps --out_dir results_overnight/logic >> results_overnight/all.log 2>&1

# 2. Kepler Synthetic Law (100 seeds)
echo "[2/4] Running Kepler Synthetic Discovery (100 seeds)..."
Rscript inst/scripts/run_benchmark_cli.R --benchmark kepler --seeds 1:100 --workers 1 --device mps --out_dir results_overnight/kepler >> results_overnight/all.log 2>&1

# 3. Kepler Real Exoplanet Data (100 seeds)
echo "[3/4] Running Kepler Real Exoplanet Benchmark (100 seeds)..."
Rscript inst/scripts/run_benchmark_cli.R --benchmark kepler_real --seeds 1:100 --workers 1 --device mps --out_dir results_overnight/kepler_real >> results_overnight/all.log 2>&1

# 4. Friedman #1 Nonlinear Benchmark (100 seeds)
echo "[4/4] Running Friedman #1 Benchmark (100 seeds)..."
Rscript inst/scripts/run_benchmark_cli.R --benchmark nonlinear --seeds 1:100 --workers 1 --device mps --out_dir results_overnight/nonlinear >> results_overnight/all.log 2>&1

echo "=== All Overnight Benchmarks Completed Successfully! ==="
