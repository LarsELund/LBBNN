#!/usr/bin/env bash

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

# 1. Remove any stale custom ~/.R/Makevars that could override EasyBuild R's Makeconf
rm -f "$HOME/.R/Makevars" 2>/dev/null || true

# 2. Strip micromamba/conda from PATH to prevent compiler/ABI conflicts with cluster R
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -E "micromamba|conda" | paste -sd ':' -)

# 3. Source system profiles so module / Lmod is available in non-interactive subshells
[ -f /etc/profile ] && source /etc/profile 2>/dev/null || true
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh 2>/dev/null || true
[ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash 2>/dev/null || true

# 4. Load cluster R and FlexiBLAS modules ONLY (do NOT load Python/GCCcore-14.3 which conflicts with R's GCCcore-11.2)
if command -v module >/dev/null 2>&1; then
  module purge 2>/dev/null || true
  module load R 2>/dev/null || true
  module load FlexiBLAS 2>/dev/null || true
fi

# 5. Add known FlexiBLAS library directories to LD_LIBRARY_PATH
for d in \
  /opt/uio/modules/rhel8/easybuild/software/FlexiBLAS/*/lib \
  /cluster/software/FlexiBLAS/*/lib \
  /software/FlexiBLAS/*/lib \
  /opt/eb/software/FlexiBLAS/*/lib \
  /cluster/software/EB/FlexiBLAS/*/lib; do
  if [ -d "$d" ]; then
    export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
  fi
done

RSCRIPT_BIN=$(command -v Rscript || echo "Rscript")

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
echo "       Benchmark:       $BENCHMARK"
echo "       Seeds:           $SEEDS"
echo "       Workers:         $WORKERS"
echo "       Out Dir:         $OUT_DIR"
echo "=================================================================="

echo "[STEP 1/2] Verifying R baseline packages..."
"$RSCRIPT_BIN" inst/scripts/install_baselines.R || echo "[WARN] Package installer returned code $?, proceeding..."

echo "[STEP 2/2] Running baseline benchmarks and LMM contrasts..."
"$RSCRIPT_BIN" inst/scripts/run_baselines_cli.R \
  --benchmark "$BENCHMARK" \
  --seeds "$SEEDS" \
  --workers "$WORKERS" \
  --out_dir "$OUT_DIR" \
  --overnight_dir "results_overnight" \
  --resume TRUE

echo "[OK] All baseline tasks and LMM statistical evaluations finished at $(date)."
