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

# 1. Source system profiles so module / Lmod is available in subshell
if [ -f /etc/profile ]; then source /etc/profile 2>/dev/null || true; fi
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh 2>/dev/null || true; fi
if [ -f /usr/share/Modules/init/bash ]; then source /usr/share/Modules/init/bash 2>/dev/null || true; fi
if [ -f ~/.bashrc ]; then source ~/.bashrc 2>/dev/null || true; fi

# 2. Load cluster modules if available
if command -v module >/dev/null 2>&1; then
  module load R 2>/dev/null || true
  module load FlexiBLAS 2>/dev/null || true
  module load Python 2>/dev/null || true
fi

# 3. Add known FlexiBLAS and R EasyBuild library directories to LD_LIBRARY_PATH
for d in \
  /cluster/software/FlexiBLAS/*/lib \
  /software/FlexiBLAS/*/lib \
  /opt/eb/software/FlexiBLAS/*/lib \
  /cluster/software/EB/FlexiBLAS/*/lib \
  /mn/sarpanitu/*/*/FlexiBLAS/*/lib; do
  if [ -d "$d" ]; then
    export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
  fi
done

# Fallback: if libflexiblas.so.3 is still not found, locate it
if ! ldconfig -p 2>/dev/null | grep -q libflexiblas; then
  FLEXI_FOUND=$(find /cluster/software /software /opt -name "libflexiblas.so.3" 2>/dev/null | head -n 1 || true)
  if [ -n "$FLEXI_FOUND" ]; then
    export LD_LIBRARY_PATH="$(dirname "$FLEXI_FOUND"):${LD_LIBRARY_PATH:-}"
  fi
fi

# 4. Choose Rscript: prioritize module R 4.5 (where FBMS/BAS are already built), then micromamba
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
