#!/bin/bash
# E5: launch the next session of a VS-VIO seed after the previous one stopped on its time budget.
#   next_session.sh SEED PREV_KERNEL TARGET_ACCOUNT N WORKDIR
#   e.g. next_session.sh 0 hagagovic/vsvio-e5-s0 hagagovic 2 /tmp/x       (same account: kernel_sources)
#        next_session.sh 0 hagagovic/vsvio-e5-s0-c2 amrelmadah 3 /tmp/x   (other account: resume dataset)
# Same account: the new kernel mounts the previous kernel's output. Other account (weekly GPU quota spent):
# the previous output (e5/ with resume.pth) is downloaded and uploaded as <target>/vsvio-e5-s<SEED>-resume.
# The target account needs <target>/vsvio-e5-inputs. The kernel script is the master with only the seed changed.
# Env: PREFIX (default vsvio-e5) names the kernels, folders and resume datasets of a run; GPU_IDS (default all)
# must match the run's first session (e.g. PREFIX=vsvio-e5-1gpu GPU_IDS=0 for the one-GPU run); SPLIT (default
# clean) likewise (authors = training set with 08).
set -euo pipefail
SEED=$1 PREV=$2 TGT=$3 N=$4 WORK=$5
HERE=$(cd "$(dirname "$0")" && pwd)
SRC_ACCT=${PREV%%/*}
PREFIX=${PREFIX:-vsvio-e5} GPU=${GPU_IDS:-all} SPLIT=${SPLIT:-clean}
kg() { KAGGLE_CONFIG_DIR=~/.kaggle-accounts/$1 kaggle "${@:2}" 2>&1 | grep -v "outdated \`kaggle\` version"; }

st=$(kg "$SRC_ACCT" kernels status "$PREV" | grep -o 'KernelWorkerStatus\.[A-Z_]*')
[[ $st == *COMPLETE || ( -n "${ALLOW_KILLED:-}" && $st == *ERROR ) ]] || { echo "previous kernel $PREV is $st"; exit 1; }

DL=$WORK/${PREFIX}_s${SEED}_after_$(basename "$PREV")
mkdir -p "$DL"
if [[ $TGT == "$SRC_ACCT" ]]; then
  # same account: the next kernel mounts the previous output itself (and checks resume.pth when it loads it), so
  # only the small files are needed here; a 580 MB download over a flaky link is what stopped the chain twice
  for try in 1 2 3; do
    kg "$SRC_ACCT" kernels output "$PREV" -p "$DL" --file-pattern '^e5/(status\.json|driver_log\.txt)$' --force | tail -1 || true
    [[ -f $DL/e5/status.json ]] && break; sleep 30
  done
else
  # other account: resume.pth has to come down and go up again. Kaggle cuts downloads short at times (0-byte
  # resume.pth, three times in E5): retry until it is whole
  for try in 1 2 3 4 5 6; do
    kg "$SRC_ACCT" kernels output "$PREV" -p "$DL" --file-pattern "^e5/" --force | tail -1 || true
    size=$(stat -f%z "$DL/e5/resume.pth" 2>/dev/null || stat -c%s "$DL/e5/resume.pth" 2>/dev/null || echo 0)
    [[ -f $DL/e5/status.json && $size -gt 100000000 ]] && break
    echo "download incomplete (resume.pth $size bytes), retry $try"; sleep 60
  done
fi
SAME=$([[ $TGT == "$SRC_ACCT" ]] && echo 1 || echo 0) ALLOW_KILLED=${ALLOW_KILLED:-} GPU=$GPU SPLIT=$SPLIT python3 - "$DL/e5" <<'EOF'
import json, os, sys
from pathlib import Path
d = Path(sys.argv[1]); st = json.loads((d / "status.json").read_text()); last = st["sessions"][-1]
print(f"seed {st['seed']}, sessions {len(st['sessions'])}, last: rc {last.get('rc')}, stopped before epoch {last.get('stopped_before_epoch')}")
assert not st.get("done") and not st["smoke"], "nothing to continue"
assert st.get("split", "clean") == os.environ["SPLIT"], f"run used split {st.get('split', 'clean')}, SPLIT is {os.environ['SPLIT']}"
assert st.get("gpu_ids", "all") == os.environ["GPU"], f"run used gpu_ids {st.get('gpu_ids', 'all')}, GPU_IDS is {os.environ['GPU']}"
# A session killed from outside (rc -9) still leaves the last resume.pth intact: it is replaced atomically after
# every epoch. ALLOW_KILLED=1 continues from it; the next session's log states the epoch it resumes at.
clean = "stopped_before_epoch" in last and last.get("rc") == 0
assert clean or (os.environ.get("ALLOW_KILLED") and last.get("rc") == -9), "last session did not stop cleanly on its budget"
assert os.environ["SAME"] == "1" or (d / "resume.pth").stat().st_size > 1e8, "resume.pth missing or truncated"
EOF

slug=${PREFIX}-s${SEED}-c${N}
if [[ $PREFIX == vsvio-e5 ]]; then K=$HERE/../e5_vsvio_s${SEED}_c${N}; else K=$HERE/../${PREFIX//-/_}_s${SEED}_c${N}; fi
mkdir -p "$K"
sed "s/os.environ.get(\"E5_SEED\", \"0\")/os.environ.get(\"E5_SEED\", \"$SEED\")/" "$HERE/vsvio_e5_train.py" | sed "s/os.environ.get(\"E5_EXPECT_RESUME\", \"0\")/os.environ.get(\"E5_EXPECT_RESUME\", \"1\")/" \
  | sed "s/os.environ.get(\"E5_GPU_IDS\", \"all\")/os.environ.get(\"E5_GPU_IDS\", \"$GPU\")/" \
  | sed "s/os.environ.get(\"E5_SPLIT\", \"clean\")/os.environ.get(\"E5_SPLIT\", \"$SPLIT\")/" > "$K/vsvio_e5_train.py"
if [[ $TGT == "$SRC_ACCT" ]]; then
  ds="\"hocop1/kitti-odometry\", \"$TGT/vsvio-e5-inputs\""; ks="\"$PREV\""
else
  R=$WORK/resume_ds_${PREFIX}_s${SEED}_c${N}
  rm -rf "$R"; mkdir -p "$R"; cp -R "$DL/e5" "$R/e5"
  echo "{\"title\": \"${PREFIX}-s${SEED}-resume\", \"id\": \"$TGT/${PREFIX}-s${SEED}-resume\", \"licenses\": [{\"name\": \"other\"}]}" > "$R/dataset-metadata.json"
  if kg "$TGT" datasets status "$TGT/${PREFIX}-s${SEED}-resume" | grep -q ready; then
    kg "$TGT" datasets version -p "$R" --dir-mode zip -m "session $N" | tail -1
  else
    kg "$TGT" datasets create -p "$R" --dir-mode zip | tail -1
  fi
  until kg "$TGT" datasets status "$TGT/${PREFIX}-s${SEED}-resume" | grep -q ready; do sleep 30; done
  ds="\"hocop1/kitti-odometry\", \"$TGT/vsvio-e5-inputs\", \"$TGT/${PREFIX}-s${SEED}-resume\""; ks=""
fi
cat > "$K/kernel-metadata.json" <<EOF
{"id": "$TGT/$slug", "title": "$slug", "code_file": "vsvio_e5_train.py", "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_tpu": false, "enable_internet": true, "dataset_sources": [$ds], "kernel_sources": [$ks], "competition_sources": [], "model_sources": []}
EOF
kg "$TGT" kernels push -p "$K" | tail -1
