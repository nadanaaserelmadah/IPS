#!/bin/bash
# E5: run the remaining sessions of one seed unattended: wait for the running kernel, then launch the next session
# with next_session.sh on the next account of the plan, until the run reports done.
#   chain.sh SEED FIRST_KERNEL WORKDIR ACCOUNT:N [ACCOUNT:N ...]
#   e.g. PREFIX=vsvio-e5-1gpu GPU_IDS=0 chain.sh 0 nadaelmadah/vsvio-e5-1gpu-s0-c1 /tmp/x \
#          nadaelmadah:2 nadanasserelmadah:3 nadanasserelmadah:4 sallyhaggag:5
# A session killed from outside (rc -9) is continued from its last resume.pth (ALLOW_KILLED=1).
set -uo pipefail
SEED=$1 CUR=$2 WORK=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
PREFIX=${PREFIX:-vsvio-e5}
log() { echo "$(date '+%m-%d %H:%M') $*"; }
log "chain start: seed $SEED from $CUR, PREFIX=$PREFIX GPU_IDS=${GPU_IDS:-all} SPLIT=${SPLIT:-clean}, plan: $*"
status() { KAGGLE_CONFIG_DIR=~/.kaggle-accounts/${1%%/*} kaggle kernels status "$1" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*' | cut -d. -f2; }
wait_kernel() {
  local st
  while true; do
    st=$(status "$1")
    [[ $st == RUNNING || $st == QUEUED || -z $st ]] || { echo "$st"; return; }
    sleep 600
  done
}
for step in "$@"; do
  acct=${step%%:*} n=${step##*:}
  st=$(wait_kernel "$CUR"); log "$CUR finished: $st"
  d=$WORK/chain_probe_$(basename "$CUR"); mkdir -p "$d"
  KAGGLE_CONFIG_DIR=~/.kaggle-accounts/${CUR%%/*} kaggle kernels output "$CUR" -p "$d" --file-pattern '^e5/(status\.json|driver_log\.txt)$' >/dev/null 2>&1
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('$d/e5/status.json')).get('done') else 1)" 2>/dev/null; then
    log "run done after $CUR"; exit 0
  fi
  grep -E "resumed from|time budget|rc " "$d/e5/driver_log.txt" | tail -3
  log "launching session $n on $acct"
  ALLOW_KILLED=1 "$HERE/next_session.sh" "$SEED" "$CUR" "$acct" "$n" "$WORK" || { log "next_session.sh failed"; exit 1; }
  CUR=$acct/$PREFIX-s$SEED-c$n
  sleep 300
done
st=$(wait_kernel "$CUR"); log "$CUR finished: $st (last planned session)"
