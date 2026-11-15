#!/bin/bash
# Audit paper, E5: from the finished kernels to the tables and the figure, in one go.
#   e5_run.sh OUT SCORE_INPUTS [account/kernel ...]
# With kernels: downloads each finished kernel's e5/ folder (no .pth) to OUT/runs/<kernel>/e5. Without: uses the
# e5/ folders already under OUT/runs/*/ (e.g. a mock run). SCORE_INPUTS holds geo_test/kgeo/lag60, npz, gyro,
# raw_calib and suma (build them with scripts/audit/build_score_inputs.sh). PY: python with numpy, matplotlib and
# hydra-core (the scorer imports VIFT), e.g. python3 -m venv --system-site-packages V && V/bin/pip install hydra-core
# omegaconf natsort rich lightning.
# Writes OUT/e5_select.txt, OUT/e5_scored.txt, OUT/e5_analysis.txt and OUT/audit_e5.png/.pdf.
set -euo pipefail
OUT=$1 IN=$2; shift 2
PY=${PY:-python3}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT/runs"
for k in "$@"; do
  st=$(KAGGLE_CONFIG_DIR=~/.kaggle-accounts/${k%%/*} kaggle kernels status "$k" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*')
  [[ $st == *COMPLETE ]] || { echo "$k is $st"; exit 1; }
  KAGGLE_CONFIG_DIR=~/.kaggle-accounts/${k%%/*} kaggle kernels output "$k" -p "$OUT/runs/${k#*/}" \
    --file-pattern '^e5/(status\.json|evals\.jsonl|driver_log\.txt|train_e5\.diff|epochs/.*\.npy)$' 2>&1 | tail -1
done
RUNS=("$OUT"/runs/*/e5)
echo "runs: ${RUNS[*]}"
"$PY" "$ROOT/scripts/audit/e5_select.py" "$OUT/farm" "${RUNS[@]}" | tee "$OUT/e5_select.txt"
"$PY" "$ROOT/scripts/audit/score_released.py" "$OUT"/farm/*/*/poses -- "$IN/geo_test/kgeo/lag60" "$IN/npz" \
  "$IN/gyro" "$IN/raw_calib" "$IN/suma" > "$OUT/e5_scored.txt"
"$PY" "$ROOT/scripts/audit/e5_analysis.py" "$OUT/e5_scored.txt" \
  "$ROOT/results_cache/2026-11-09/audit_E1_E3_released_models.txt" \
  "$ROOT/results_cache/2026-11-10/audit_E4_lagfix_6seeds.txt" "$OUT/audit_e5.png" "${RUNS[@]}" | tee "$OUT/e5_analysis.txt"
