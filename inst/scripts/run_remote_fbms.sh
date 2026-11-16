#!/usr/bin/env bash

# =============================================================================
# inst/scripts/run_remote_fbms.sh
# Cluster execution wrapper for FBMS-BLR / FBMS-BGNLM using multicore
# gmjmcmc.parallel (jonlachmann/FBMS@JCS >= 1.4.1) and native BMA predictions.
# Note: Intentionally avoids `set -u` so Lmod /etc/profile and module load R work cleanly.
#
# Usage:
#   ./inst/scripts/run_remote_fbms.sh \
#     [BENCHMARK=all] [SEEDS=1:100] [FBMS_CORES=20] [FBMS_RUNS=40] \
#     [FBMS_P=30] [PRIORS=explore] [MERGE_PRIOR=none] [OUT_DIR=results_baselines] [OUTER_WORKERS=3]
# =============================================================================

BENCHMARK=${1:-all}
SEEDS=${2:-1:100}
FBMS_CORES=${3:-20}
FBMS_RUNS=${4:-40}
FBMS_P=${5:-30}
PRIORS=${6:-default}
MERGE_PRIOR=${7:-}
OUT_DIR=${8:-results_baselines}
OUTER_WORKERS=${9:-1}

mkdir -p "$OUT_DIR"
echo $$ > "$OUT_DIR/fbms.pid"

# 1. Remove any stale custom ~/.R/Makevars that could override EasyBuild R's Makeconf
rm -f "$HOME/.R/Makevars" 2>/dev/null || true

# 2. Strip micromamba/conda from PATH to prevent compiler/ABI conflicts with cluster R
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -E "micromamba|conda" | paste -sd ':' -)

# 3. Source system profiles so module / Lmod is available in non-interactive subshells
[ -f /etc/profile ] && source /etc/profile 2>/dev/null || true
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh 2>/dev/null || true
[ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash 2>/dev/null || true

# 4. Load ONLY the cluster R module (avoids GCCcore-14.3 ABI segfaults)
if command -v module >/dev/null 2>&1; then
  module purge 2>/dev/null || true
  module load R 2>/dev/null || true
fi

RSCRIPT_BIN=$(command -v Rscript || echo "Rscript")

# Single-threaded BLAS inside each forked parallel GMJMCMC chain to avoid thread oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "=================================================================="
echo "[INFO] Starting Parallel FBMS@JCS Run"
echo "       Host:            $(hostname)"
echo "       PID:             $$"
echo "       Date:            $(date)"
echo "       Rscript:         $RSCRIPT_BIN"
echo "       Benchmark:       $BENCHMARK"
echo "       Seeds:           $SEEDS"
echo "       Priors:          $PRIORS"
echo "       Merge Prior:     ${MERGE_PRIOR:-auto}"
echo "       Outer Workers:   $OUTER_WORKERS"
echo "       FBMS Cores/Seed: $FBMS_CORES"
echo "       Total Threads:   $((OUTER_WORKERS * FBMS_CORES))"
echo "       FBMS Runs/Seed:  $FBMS_RUNS"
echo "       FBMS Pop P:      $FBMS_P"
echo "       Out Dir:         $OUT_DIR"
echo "=================================================================="

echo "[STEP 1/2] Ensuring jonlachmann/FBMS@JCS (>= 1.4.1) and baseline packages..."
"$RSCRIPT_BIN" inst/scripts/install_baselines.R || echo "[WARN] Package installer returned code $?, proceeding..."

echo "[STEP 2/2] Running parallel FBMS@JCS with native BMA predictions..."
EXTRA_ARGS=()
if [ -n "$MERGE_PRIOR" ]; then
  EXTRA_ARGS+=(--merge_prior "$MERGE_PRIOR")
fi

"$RSCRIPT_BIN" inst/scripts/run_remote_fbms_update.R \
  --benchmark "$BENCHMARK" \
  --seeds "$SEEDS" \
  --priors "$PRIORS" \
  --outer_workers "$OUTER_WORKERS" \
  --fbms_cores "$FBMS_CORES" \
  --fbms_runs "$FBMS_RUNS" \
  --fbms_P "$FBMS_P" \
  --out_dir "$OUT_DIR" \
  --resume TRUE \
  "${EXTRA_ARGS[@]}"

echo "[OK] FBMS@JCS cluster job finished at $(date)."
