#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# inst/scripts/deploy_baselines_ssh.sh
# One-command SSH deployment and management utility for running the baseline
# comparison suite on nam-shub-02.uio.no (RHEL 8, error-free, no liblantern.so).
#
# Usage:
#   ./inst/scripts/deploy_baselines_ssh.sh setup                     # Sync code & install R/Python baseline packages
#   ./inst/scripts/deploy_baselines_ssh.sh start [1:100] [16]        # Launch full baseline run via nohup on server
#   ./inst/scripts/deploy_baselines_ssh.sh fbms [1:100] [40] [25]    # Update ONLY FBMS (default JCS prior) via nohup
#   ./inst/scripts/deploy_baselines_ssh.sh fbms_priors [1:100] [20] [30] [40] [2] # Explore all priors (2 seeds x 20 cores = 40 CPUs)
#   ./inst/scripts/deploy_baselines_ssh.sh fbms_logic_custom [1:100] [20] [25] [40] [2] # Run Ex7 custom BLR mlik estimators on logic
#   ./inst/scripts/deploy_baselines_ssh.sh stop                      # Stop running FBMS sweep process tree cleanly
#   ./inst/scripts/deploy_baselines_ssh.sh status                    # Check running jobs & checkpoint counts
#   ./inst/scripts/deploy_baselines_ssh.sh pull                      # Pull completed results_baselines/ back to local Mac
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
  fbms)
    FBMS_CORES=${3:-40}
    FBMS_P=${4:-25}
    FBMS_RUNS=${5:-40}
    OUTER_WORKERS=${6:-1}
    sync_to_remote
    echo "[FBMS] Launching parallel FBMS@JCS update (seeds=${SEEDS}, outer=${OUTER_WORKERS}, cores=${FBMS_CORES}, runs=${FBMS_RUNS}, P=${FBMS_P}) on ${REMOTE_HOST}..."
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "cd ${REMOTE_DIR} && chmod +x inst/scripts/*.sh && mkdir -p results_baselines && nohup ./inst/scripts/run_remote_fbms.sh all '${SEEDS}' '${FBMS_CORES}' '${FBMS_RUNS}' '${FBMS_P}' default default results_baselines '${OUTER_WORKERS}' > results_baselines/fbms_update.log 2>&1 < /dev/null & sleep 1 && echo '[OK] Launched FBMS update in background. Remote PID:' && cat results_baselines/fbms.pid 2>/dev/null || true"
    ;;
  fbms_priors)
    FBMS_CORES=${3:-20}
    FBMS_P=${4:-30}
    FBMS_RUNS=${5:-40}
    OUTER_WORKERS=${6:-2}
    sync_to_remote
    echo "[FBMS-PRIORS] Launching multi-prior FBMS@JCS exploration (seeds=${SEEDS}, outer=${OUTER_WORKERS} x cores=${FBMS_CORES} = $((OUTER_WORKERS * FBMS_CORES)) threads, runs=${FBMS_RUNS}, P=${FBMS_P}) on ${REMOTE_HOST}..."
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "cd ${REMOTE_DIR} && chmod +x inst/scripts/*.sh && mkdir -p results_baselines && nohup ./inst/scripts/run_remote_fbms.sh all '${SEEDS}' '${FBMS_CORES}' '${FBMS_RUNS}' '${FBMS_P}' explore none results_baselines '${OUTER_WORKERS}' > results_baselines/fbms_update.log 2>&1 < /dev/null & sleep 1 && echo '[OK] Launched FBMS prior sweep in background. Remote PID:' && cat results_baselines/fbms.pid 2>/dev/null || true"
    ;;
  fbms_logic_custom)
    FBMS_CORES=${3:-20}
    FBMS_P=${4:-25}
    FBMS_RUNS=${5:-40}
    OUTER_WORKERS=${6:-2}
    sync_to_remote
    echo "[FBMS-LOGIC-CUSTOM] Launching Ex7 custom BLR mlik estimators (custom_logic_jeffreys, custom_logic_tcch; seeds=${SEEDS}, outer=${OUTER_WORKERS} x cores=${FBMS_CORES} = $((OUTER_WORKERS * FBMS_CORES)) threads, runs=${FBMS_RUNS}, P=${FBMS_P}, N=500)..."
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "cd ${REMOTE_DIR} && chmod +x inst/scripts/*.sh && mkdir -p results_baselines && nohup ./inst/scripts/run_remote_fbms.sh logic '${SEEDS}' '${FBMS_CORES}' '${FBMS_RUNS}' '${FBMS_P}' 'custom_logic_jeffreys,custom_logic_tcch' none results_baselines '${OUTER_WORKERS}' > results_baselines/fbms_logic_custom.log 2>&1 < /dev/null & sleep 1 && echo '[OK] Launched Ex7 custom logic sweep in background. Remote PID:' && cat results_baselines/fbms.pid 2>/dev/null || true"
    ;;
  stop)
    echo "[STOP] Stopping active FBMS job tree on ${REMOTE_HOST} (completed checkpoints are kept)..."
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "cd ${REMOTE_DIR} && \
      kill_tree() { \
        local p=\$1; \
        for c in \$(ps -o pid= --ppid \$p 2>/dev/null); do kill_tree \$c; done; \
        kill -TERM \$p 2>/dev/null || true; \
      }; \
      if [ -f results_baselines/fbms.pid ]; then \
        PID=\$(cat results_baselines/fbms.pid); \
        kill_tree \$PID; \
        sleep 1; \
        pkill -9 -u \$USER -f 'run_remote_fbms_update.R' 2>/dev/null || true; \
        echo \"[OK] Stopped FBMS process tree (root PID \$PID).\"; \
      else \
        pkill -TERM -u \$USER -f 'run_remote_fbms' 2>/dev/null || true; \
        echo '[OK] Sent TERM to any matching run_remote_fbms processes.'; \
      fi"
    ;;
  status)
    echo "[STATUS] Checking remote progress on ${REMOTE_HOST}..."
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "cd ${REMOTE_DIR} && \
      for pidfile in results_baselines/baselines.pid results_baselines/fbms.pid; do \
        if [ -f \"\$pidfile\" ]; then \
          PID=\$(cat \"\$pidfile\"); \
          echo \"=== \$pidfile (PID: \$PID) ===\"; \
          ps -p \$PID -o pid,etime,pcpu,pmem,comm 2>/dev/null || echo 'Process not currently active.'; \
        fi; \
      done && \
      echo '=== Seed Progress by Benchmark ===' && \
      for b in logic kepler kepler_real nonlinear; do \
        cnt=\$(ls results_baselines/\$b/checkpoints/*.rds 2>/dev/null | wc -l || echo 0); \
        fcnt=\$(ls results_baselines/\$b/fbms_checkpoints/*.rds 2>/dev/null | wc -l || echo 0); \
        pcnt=\$(find results_baselines/\$b/fbms_priors -name '*.rds' 2>/dev/null | wc -l || echo 0); \
        echo \"  - \$b: \$cnt total seeds | \$fcnt FBMS default | \$pcnt FBMS prior-sweep ckpts\"; \
      done && \
      echo '=== Recent FBMS Log Output ===' && \
      tail -n 25 results_baselines/fbms_update.log 2>/dev/null || tail -n 25 results_baselines/baselines_master.log 2>/dev/null || echo 'No log file found yet.'"
    ;;
  pull)
    echo "[PULL] Pulling results_baselines/ (excluding *.log) from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}..."
    mkdir -p results_baselines
    rsync -avz --exclude '*.log' "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/results_baselines/" results_baselines/
    echo "[OK] Pulled results_baselines/ locally."
    ;;
  *)
    echo "Usage: $0 {setup|start|fbms|fbms_priors|fbms_logic_custom|stop|status|pull} [SEEDS] [CORES] [P] [RUNS] [OUTER]"
    exit 1
    ;;
esac
