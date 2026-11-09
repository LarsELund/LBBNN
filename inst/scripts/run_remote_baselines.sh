#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# inst/scripts/run_remote_baselines.sh
# Self-contained server execution script for running the Neuro-Symbolic LBBNN
# baseline comparison suite on Linux HPC servers (e.g. nam-shub-02.uio.no).
# =============================================================================

BENCHMARK=${1:-all}
SEEDS=${2:-1:100}
WORKERS=${3:-16}
OUT_DIR=${4:-results_baselines}

mkdir -p "$OUT_DIR"
echo $$ > "$OUT_DIR/baselines.pid"

# 1. Load cluster modules if available (sets up R 4.5 + FlexiBLAS toolchain)
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load R 2>/dev/null || true
  module load FlexiBLAS 2>/dev/null || true
  module load Python 2>/dev/null || true
fi

# 2. Locate libflexiblas.so.3 on the system and ensure it is in LD_LIBRARY_PATH
FLEXI_LIB=$(find /cluster/software /software /opt /mn/sarpanitu/ansatte-u2 -name "libflexiblas.so.3" 2>/dev/null | head -n 1 || true)
if [ -n "$FLEXI_LIB" ]; then
  FLEXI_DIR=$(dirname "$FLEXI_LIB")
  export LD_LIBRARY_PATH="${FLEXI_DIR}:${LD_LIBRARY_PATH:-}"
fi

# 3. Choose Rscript: prioritize module R 4.5 (where FBMS/BAS are already built), then micromamba
if command -v Rscript >/dev/null 2>&1; then
  RSCRIPT_BIN=$(command -v Rscript)
elif [ -x "/mn/sarpanitu/ansatte-u2/aliaksah/micromamba/envs/r-symbolic/bin/Rscript" ]; then
  RSCRIPT_BIN="/mn/sarpanitu/ansatte-u2/aliaksah/micromamba/envs/r-symbolic/bin/Rscript"
else
  RSCRIPT_BIN="Rscript"
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "=================================================================="
echo "[INFO] Starting Neuro-Symbolic LBBNN Baseline Suite"
echo "       Host:            $(hostname)"
echo "       PID:             $$"
echo "       Date:            $(date)"
echo "       Rscript:         $RSCRIPT_BIN"
echo "       LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-none}"
echo "       Benchmark:       $BENCHMARK"
echo "       Seeds:           $SEEDS"
echo "       Workers:         $WORKERS"
echo "       Out Dir:         $OUT_DIR"
echo "=================================================================="

echo "[STEP 1/2] Verifying R and Python baseline packages..."
"$RSCRIPT_BIN" inst/scripts/install_baselines.R

echo "[STEP 2/2] Running baseline benchmarks and LMM contrasts..."
"$RSCRIPT_BIN" inst/scripts/run_baselines_cli.R \
  --benchmark "$BENCHMARK" \
  --seeds "$SEEDS" \
  --workers "$WORKERS" \
  --out_dir "$OUT_DIR" \
  --overnight_dir "results_overnight" \
  --resume TRUE

echo "[OK] All baseline tasks and LMM statistical evaluations finished at $(date)."
