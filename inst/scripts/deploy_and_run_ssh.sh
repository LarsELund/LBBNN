#!/usr/bin/env bash
# =============================================================================
# Remote SSH Deployment & Execution Toolkit (No Slurm)
# Usage:
#   ./inst/scripts/deploy_and_run_ssh.sh start [benchmark] [seeds]
#   ./inst/scripts/deploy_and_run_ssh.sh status
#   ./inst/scripts/deploy_and_run_ssh.sh pull
# =============================================================================
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-inst/scripts/remote_config.yml}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Error: Config file $CONFIG_FILE not found." >&2
  exit 1
fi

parse_yaml() {
  local key="$1"
  awk -F':' -v k="$key" '$1 ~ "^[[:space:]]*"k"[[:space:]]*$" {
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2);
    gsub(/^"|"$|^'\''|'\''$/, "", $2);
    print $2
  }' "$CONFIG_FILE"
}

SSH_HOST="$(parse_yaml ssh_host)"
SSH_USER="$(parse_yaml ssh_user)"
REMOTE_DIR="$(parse_yaml remote_dir)"
RSCRIPT_PATH="$(parse_yaml rscript_path)"
N_WORKERS="$(parse_yaml n_workers)"
DEVICE="$(parse_yaml device)"
RUNNER_MODE="$(parse_yaml runner_mode)"
ENV_MODULES="$(parse_yaml env_modules)"

CMD="${1:-status}"
BENCHMARK="${2:-all}"
SEEDS="${3:-1:100}"
REMOTE_TARGET="${SSH_USER}@${SSH_HOST}"

# Prefix command with module loading if specified in config
MOD_PREFIX=""
if [[ -n "${ENV_MODULES:-}" ]]; then
  MOD_PREFIX="${ENV_MODULES} && "
fi

case "$CMD" in
  start)
    echo "[1/3] Syncing feature/symbolic-bgnlm-lbbnn workspace to ${REMOTE_TARGET}:${REMOTE_DIR} ..."
    ssh "$REMOTE_TARGET" "mkdir -p ${REMOTE_DIR}/results/checkpoints"
    rsync -avz --exclude '.git' --exclude 'results' --exclude '.Rproj.user' \
      ./ "${REMOTE_TARGET}:${REMOTE_DIR}/"

    echo "[2/3] Installing/updating LBBNN package on remote server ..."
    ssh "$REMOTE_TARGET" "${MOD_PREFIX}cd ${REMOTE_DIR} && ${RSCRIPT_PATH} -e \"if (requireNamespace('pak', quietly=TRUE)) pak::local_install('.') else if (requireNamespace('remotes', quietly=TRUE)) remotes::install_local('.', upgrade='never')\""

    echo "[3/3] Launching detached benchmark run via ${RUNNER_MODE} (${BENCHMARK}, seeds ${SEEDS}, device ${DEVICE}) ..."
    RUN_CMD="${MOD_PREFIX}cd ${REMOTE_DIR} && ${RSCRIPT_PATH} inst/scripts/run_benchmark_cli.R --benchmark ${BENCHMARK} --seeds ${SEEDS} --workers ${N_WORKERS} --device ${DEVICE} --out_dir results"

    if [[ "${RUNNER_MODE}" == "tmux" ]]; then
      ssh "$REMOTE_TARGET" "if command -v tmux >/dev/null 2>&1; then tmux kill-session -t symbolic_lbbnn_bench 2>/dev/null || true; tmux new-session -d -s symbolic_lbbnn_bench \"bash -l -c '${RUN_CMD} > results/run.log 2>&1'\"; echo 'Started in tmux session: symbolic_lbbnn_bench'; else nohup bash -l -c '${RUN_CMD}' > ${REMOTE_DIR}/results/run.log 2>&1 & echo \$! > ${REMOTE_DIR}/results/run.pid; echo 'Started via nohup (PID: \$(cat ${REMOTE_DIR}/results/run.pid))'; fi"
    else
      # Default: pure detached nohup process with PID tracking
      ssh "$REMOTE_TARGET" "cd ${REMOTE_DIR} && nohup bash -l -c '${RUN_CMD}' > results/run.log 2>&1 & echo \$! > results/run.pid && echo \"Job launched via nohup with PID \$(cat results/run.pid)\""
    fi
    ;;

  status)
    echo "=== Checking Remote Benchmark Status on ${REMOTE_TARGET} ==="
    ssh "$REMOTE_TARGET" "cd ${REMOTE_DIR} 2>/dev/null && \
      if [[ -f results/run.pid ]]; then \
        PID=\$(cat results/run.pid); \
        if ps -p \$PID > /dev/null 2>&1; then \
          echo \"[RUNNING] Process PID \$PID is active.\"; \
        else \
          echo \"[FINISHED/STOPPED] Process PID \$PID is no longer running.\"; \
        fi; \
      fi; \
      echo 'Completed seed checkpoints:' && \
      ls -1 results/checkpoints/*.rds 2>/dev/null | wc -l || true && \
      echo '--- Last 20 lines of results/run.log ---' && \
      tail -n 20 results/run.log 2>/dev/null || echo 'No run.log found yet.'"
    ;;

  stop)
    echo "=== Stopping Remote Benchmark Process on ${REMOTE_TARGET} ==="
    ssh "$REMOTE_TARGET" "cd ${REMOTE_DIR} 2>/dev/null && \
      if [[ -f results/run.pid ]]; then \
        PID=\$(cat results/run.pid); \
        kill -9 \$PID 2>/dev/null && echo \"Killed PID \$PID\" || echo \"PID \$PID not running\"; \
        rm -f results/run.pid; \
      fi"
    ;;

  pull)
    echo "=== Pulling results/ from ${REMOTE_TARGET}:${REMOTE_DIR}/results/ ==="
    mkdir -p results
    rsync -avz "${REMOTE_TARGET}:${REMOTE_DIR}/results/" ./results/
    echo "=== Pulled summary files: ==="
    ls -la results/summary_* 2>/dev/null || true
    ;;

  *)
    echo "Usage: $0 {start|status|pull} [benchmark] [seeds]" >&2
    exit 1
    ;;
esac

