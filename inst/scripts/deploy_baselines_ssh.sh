#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# inst/scripts/deploy_baselines_ssh.sh
# One-command SSH deployment and management utility for running the baseline
# comparison suite on nam-shub-02.uio.no (RHEL 8, error-free, no liblantern.so).
#
# Usage:
#   ./inst/scripts/deploy_baselines_ssh.sh setup               # Sync code & install R/Python baseline packages
#   ./inst/scripts/deploy_baselines_ssh.sh start [1:100] [16]  # Launch background run via nohup on server
#   ./inst/scripts/deploy_baselines_ssh.sh status              # Check running jobs & checkpoint counts
#   ./inst/scripts/deploy_baselines_ssh.sh pull                # Pull completed results_baselines/ back to local Mac
# =============================================================================

ACTION=${1:-status}
SEEDS=${2:-1:100}
WORKERS=${3:-16}

REMOTE_HOST="nam-shub-02.uio.no"
REMOTE_USER="aliaksah"
REMOTE_DIR="~/SymolicLBBNN"

sync_to_remote() {
  echo "[SYNC] Syncing code, data, and precomputed LBBNN checkpoints to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}..."
  rsync -avz --exclude '.git' --exclude '*.o' --exclude '*.so' \
    DESCRIPTION NAMESPACE R inst data results_overnight \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
}

case "$ACTION" in
  setup)
    sync_to_remote
    echo "[SETUP] Setting up host compilers and baseline packages on ${REMOTE_HOST}..."
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "
      mkdir -p ~/.R
      cat << 'EOF' > ~/.R/Makevars
CC = gcc -fPIC
CXX = g++ -fPIC
CXX11 = g++ -fPIC
CXX14 = g++ -fPIC
CXX17 = g++ -fPIC
FC = gfortran -fPIC
F77 = gfortran -fPIC
EOF
      ENV_BIN=/mn/sarpanitu/ansatte-u2/aliaksah/micromamba/envs/r-symbolic/bin
      if [ -d \"\$ENV_BIN\" ]; then
        ln -sf /usr/bin/gcc \"\$ENV_BIN/x86_64-conda-linux-gnu-cc\" 2>/dev/null || true
        ln -sf /usr/bin/g++ \"\$ENV_BIN/x86_64-conda-linux-gnu-c++\" 2>/dev/null || true
        ln -sf /usr/bin/gfortran \"\$ENV_BIN/x86_64-conda-linux-gnu-gfortran\" 2>/dev/null || true
      fi

      MAMBA_BIN=\$(which micromamba 2>/dev/null || find ~/.local ~/micromamba /mn/sarpanitu/ansatte-u2/aliaksah -name micromamba -type f 2>/dev/null | head -n 1 || true)
      if [ -n \"\$MAMBA_BIN\" ] && [ -x \"\$MAMBA_BIN\" ]; then
        echo '[SETUP] Installing precompiled binary R baseline packages via conda-forge...'
        \"\$MAMBA_BIN\" install -y -n r-symbolic -c conda-forge \
          r-glmnet r-earth r-mgcv r-ranger r-jsonlite r-matrix r-survival r-rcpp r-rcppeigen r-bh r-data.table r-gensa r-tolerance || true
      fi

      export PATH=\"/mn/sarpanitu/ansatte-u2/aliaksah/micromamba/envs/r-symbolic/bin:\$PATH\"
      cd ${REMOTE_DIR} && chmod +x inst/scripts/*.sh && \
      Rscript inst/scripts/install_baselines.R
    "
    ;;
  start)
    sync_to_remote
    echo "[START] Launching baseline suite (seeds=${SEEDS}, workers=${WORKERS}) on ${REMOTE_HOST}..."
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "cd ${REMOTE_DIR} && chmod +x inst/scripts/*.sh && mkdir -p results_baselines && nohup ./inst/scripts/run_remote_baselines.sh all '${SEEDS}' '${WORKERS}' results_baselines > results_baselines/baselines_master.log 2>&1 < /dev/null & sleep 1 && echo '[OK] Launched in background. Remote PID:' && cat results_baselines/baselines.pid 2>/dev/null || true"
    ;;
  status)
    echo "[STATUS] Checking remote progress on ${REMOTE_HOST}..."
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "cd ${REMOTE_DIR} && \
      if [ -f results_baselines/baselines.pid ]; then \
        PID=\$(cat results_baselines/baselines.pid); \
        echo \"=== Baseline Master Process (PID: \$PID) ===\"; \
        ps -p \$PID -o pid,etime,pcpu,pmem,comm 2>/dev/null || echo 'Master process not currently active.'; \
      fi && \
      echo '=== Seed Progress by Benchmark ===' && \
      for b in logic kepler kepler_real nonlinear; do \
        cnt=\$(ls results_baselines/\$b/checkpoints/*.rds 2>/dev/null | wc -l || echo 0); \
        echo \"  - \$b: \$cnt seeds completed\"; \
      done && \
      echo '=== Recent Log Output ===' && \
      tail -n 25 results_baselines/baselines_master.log 2>/dev/null || echo 'No log file found yet.'"
    ;;
  pull)
    echo "[PULL] Pulling results_baselines/ from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}..."
    mkdir -p results_baselines
    rsync -avz "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/results_baselines/" results_baselines/
    echo "[OK] Pulled results_baselines/ locally."
    ;;
  *)
    echo "Usage: $0 {setup|start|status|pull} [SEEDS] [WORKERS]"
    exit 1
    ;;
esac
