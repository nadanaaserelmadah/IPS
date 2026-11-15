#!/bin/bash
# E5: keep a chain.sh run alive across reboots. Run by launchd at login and every 30 min (see the plist note in
# scripts/kaggle_kernels/README.md); does nothing while the chain runs or once the run is done.
#   chain_supervisor.sh STATE_FILE
# STATE_FILE (bash) sets SEED, PREFIX, GPU_IDS, SPLIT, FIRST (the session-1 kernel), PLAN (array of ACCOUNT:N for
# sessions 2..), WORK (work folder) and LOG (the chain's log). The current position is read from Kaggle: the last
# planned kernel that exists is where chain.sh resumes waiting.
set -uo pipefail
STATE=$1
# shellcheck source=/dev/null
source "$STATE"
HERE=$(cd "$(dirname "$0")" && pwd)
stamp() { echo "$(date '+%m-%d %H:%M') supervisor: $*" >> "$LOG"; }

pgrep -f "chain.sh $SEED .*/$PREFIX-s$SEED-c" >/dev/null && exit 0
grep -q "run done after .*/$PREFIX-s$SEED-c" "$LOG" 2>/dev/null && exit 0

cur=$FIRST rest=()
for step in "${PLAN[@]}"; do
  acct=${step%%:*} n=${step##*:} k=${step%%:*}/$PREFIX-s$SEED-c${step##*:}
  if KAGGLE_CONFIG_DIR=~/.kaggle-accounts/$acct kaggle kernels status "$k" 2>&1 | grep -q 'KernelWorkerStatus'; then
    cur=$k rest=()
  else
    rest+=("$step")
  fi
done
stamp "chain not running; restarting from $cur, remaining: ${rest[*]:-none}"
cd "$WORK" || exit 1
export PREFIX GPU_IDS SPLIT
python3 - "$HERE/chain.sh" "$SEED" "$cur" "$WORK" "$LOG" "${rest[@]}" <<'EOF'
import os, sys
chain, seed, cur, work, log, *rest = sys.argv[1:]
if os.fork():
    sys.exit(0)
os.setsid()
pid = os.fork()
if pid:  # keep the Mac awake while the chain runs
    os.execvp("caffeinate", ["caffeinate", "-i", "-w", str(pid)])
f = open(log, "a")
os.dup2(f.fileno(), 1); os.dup2(f.fileno(), 2)
os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
os.execvp(chain, [chain, seed, cur, work, *rest])
EOF
