#!/bin/bash
# Audit paper: rebuild the scoring inputs that scripts/audit/score_released.py (and e5_run.sh) need for KITTI
# 05 / 07 / 10, from their public sources. Same steps as the original E1-E3 build (2026-11-09).
#   build_score_inputs.sh OUT
# OUT/gyro       mhaggag/kitti-gyro-feats (Kaggle dataset): <seq>_gyro_R.npy, calibration json, + max-gap files below
# OUT/raw        KITTI raw OXTS for the three test drives (scripts/kitti/fetch_oxts.py)
# OUT/npz        packed OXTS (scripts/kitti/oxts_pack.py)
# OUT/raw_calib  KITTI raw calibration of 2011_09_30
# OUT/suma       SuMa poses (scripts/kitti/fetch_suma_poses.py)
# OUT/geo_test   two-view direction on 05/07/10 (kernel mhaggag/kitti-geo-direction-test); gate v3 reads kgeo/lag60
set -euo pipefail
OUT=$(mkdir -p "$1" && cd "$1" && pwd)
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-python3}
"$PY" -c "import hydra, numpy" 2>/dev/null || { echo "PY=$PY needs hydra-core (the scorer imports VIFT); see the venv note in e5_run.sh"; exit 1; }
kg() { KAGGLE_CONFIG_DIR=~/.kaggle-accounts/mhaggag kaggle "$@" 2>&1 | grep -v "outdated \`kaggle\` version" || true; }

mkdir -p "$OUT/gyro" && (cd "$OUT/gyro" && kg datasets download -q --unzip mhaggag/kitti-gyro-feats)
cd "$ROOT"
"$PY" scripts/kitti/fetch_oxts.py "$OUT/raw" 2011_09_30_drive_0018 2011_09_30_drive_0027 2011_09_30_drive_0034
"$PY" scripts/kitti/oxts_pack.py "$OUT/raw" "$OUT/npz" 05,07,10
"$PY" scripts/kitti/fetch_suma_poses.py "$OUT/suma" 05,07,10
mkdir -p "$OUT/raw_calib" && (cd "$OUT/raw_calib" &&
  curl -s -O https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_calib.zip && unzip -o -q 2011_09_30_calib.zip)
mkdir -p "$OUT/geo_test" && (cd "$OUT/geo_test" && kg kernels output mhaggag/kitti-geo-direction-test -p . | tail -1)
"$PY" - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts/kitti")
import gyro_hybrid as h
S = Path(sys.argv[1])
lag = json.loads((S / "gyro/calib_2011_09_30.json").read_text())["lag_s"]
for s in ("05", "07", "10"):
    g = h.max_gaps(S / "npz" / f"{s}.npz", lag)
    np.save(S / "gyro" / f"{s}_gyro_maxgap_s.npy", g)
    print(s, "lag", lag, "frames", len(g), "gap frames", int((g > h.GAP_S).sum()), "gyro_R", np.load(S / "gyro" / f"{s}_gyro_R.npy").shape)
assert (S / "geo_test/kgeo/lag60").is_dir() and (S / "raw_calib/2011_09_30").is_dir()
print("score inputs ready:", S)
EOF
