#!/bin/bash
set -euo pipefail

# Print a ready-to-run SSH -L tunnel command for the 5090 box based on the latest
# TAU2Bench experts deployments in this repo checkout (H100 head node).
#
# Reads:
#   slurm/tau2_expert_logs/expert-1/rl_toolorch_expert1_<jobid>/node_ip.txt
#   slurm/tau2_expert_logs/expert-23/rl_toolorch_expert23_<jobid>/node_ip.txt
#
# Usage (on H100 head node):
#   bash oss-ToolOrchestra/scripts/experts/print_5090_tunnel_cmd_tau2_5090.sh [HEAD_SSH_HOST]
#
# Then copy the printed ssh command to 5090 and run it under tmux.

HEAD_SSH_HOST="${1:-H100-Together}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OSS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${OSS_DIR}/.." && pwd)"

pick_latest_node_ip() {
  local glob="$1"
  local d
  d="$(ls -1dt ${glob} 2>/dev/null | head -n 1 || true)"
  if [[ -z "${d}" ]]; then
    echo ""
    return 0
  fi
  cat "${d}/node_ip.txt" 2>/dev/null | head -n 1 || true
}

EXPERT1_IP="$(pick_latest_node_ip "${REPO_ROOT}/slurm/tau2_expert_logs/expert-1/rl_toolorch_expert1_*/")"
EXPERT23_IP="$(pick_latest_node_ip "${REPO_ROOT}/slurm/tau2_expert_logs/expert-23/rl_toolorch_expert23_*/")"

if [[ -z "${EXPERT1_IP}" || -z "${EXPERT23_IP}" ]]; then
  echo "ERROR: could not find node_ip.txt for tau2_5090 expert jobs." >&2
  echo "Expected directories under:" >&2
  echo "  ${REPO_ROOT}/slurm/tau2_expert_logs/expert-1/rl_toolorch_expert1_<jobid>/" >&2
  echo "  ${REPO_ROOT}/slurm/tau2_expert_logs/expert-23/rl_toolorch_expert23_<jobid>/" >&2
  exit 1
fi

cat <<EOF
# Run on 5090 (under tmux). Assumes you can SSH to the H100 head node as: ${HEAD_SSH_HOST}
ssh -N -o ExitOnForwardFailure=yes \\
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \\
  -L 1910:${EXPERT1_IP}:1910 -L 1911:${EXPERT1_IP}:1911 -L 1912:${EXPERT1_IP}:1912 -L 1913:${EXPERT1_IP}:1913 \\
  -L 1904:${EXPERT23_IP}:1904 -L 1905:${EXPERT23_IP}:1905 -L 1906:${EXPERT23_IP}:1906 \\
  -L 1920:${EXPERT23_IP}:1920 \\
  ${HEAD_SSH_HOST}

# Ready check (on 5090):
for p in 1910 1911 1912 1913 1904 1905 1906 1920; do
  echo -n "p=\$p "
  curl -sf --max-time 2 http://127.0.0.1:\$p/health && echo OK || echo FAIL
done
EOF
