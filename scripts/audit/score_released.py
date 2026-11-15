"""Audit paper, E1-E3: score released learned-VIO models (dumps from scripts/kaggle_kernels/kernel_ulvio,
kernel_nasvio, kernel_vsvio_gpu) on KITTI 05 / 07 / 10 against OXTS GT and SuMa.

Rows per model: network; oracle GT rotation + network translation; oracle network rotation + GT translation;
gyro rotation (true timing) + network translation; + accelerometer length (scripts/kitti/cross_model.py), then
the frozen gate v3 (variant C5b, sigma_net 0.829, scripts/gate/run_final.kitti) with nothing changed.
The models trained on 08, so 08 is reported apart, as training-set performance, network only.
  python scripts/audit/score_released.py DUMP_DIR [DUMP_DIR ...] -- GEO_DIR NPZ_DIR GYRO_DIR RAW_CALIB_DIR SUMA_DIR
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/kitti"))
sys.path.insert(0, str(ROOT / "scripts/gate"))
import gyro_hybrid as h  # noqa: E402
import run_final  # noqa: E402


def main():
    i = sys.argv.index("--")
    dumps = [Path(d) for d in sys.argv[1:i]]
    geo, npz_dir, gyro_dir, raw_calib, suma = (Path(a) for a in sys.argv[i + 1:i + 6])
    # gyro_hybrid.load_gap returns None when a gap file is missing, and the scoring then runs without the OXTS-gap
    # fallback and comes out far lower (VS-VIO policy, gate v3: 0.77 instead of 1.27). Refuse instead.
    missing = [s for s in ("05", "07", "10") if not (gyro_dir / f"{s}_gyro_maxgap_s.npy").is_file()]
    assert not missing and h.USE_GAP, f"gyro gap files missing for {missing} in {gyro_dir} (build_score_inputs.sh)"
    print("## Oracle swap and sensor rows (t_rel % / r_rel deg/100 m, mean of 05 / 07 / 10)\n", flush=True)
    subprocess.check_call([sys.executable, str(ROOT / "scripts/kitti/cross_model.py"), *map(str, dumps), "--",
                           str(npz_dir), str(gyro_dir), str(raw_calib), str(suma)])
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        runs = {"/".join(d.parts[-3:]): [str(d)] for d in dumps}  # 3 parts: run dirs often share kitti_rpmg/poses_best
        assert len(runs) == len(dumps), "dump directories must differ in their last three path parts"
        json.dump(runs, f)
    run_final.kitti("C5b", 0.829, f.name, geo, npz_dir, gyro_dir, raw_calib, suma)
    print("\n## 08 (in every released model's training split), network only, OXTS\n\n| model | t_rel % | r_rel |\n|---|---|---|")
    for d in dumps:
        if (d / "08_gt_poses.npy").is_file():
            gt = np.load(d / "08_gt_poses.npy").astype(np.float64)
            est = np.load(d / "08_estimated_poses.npy").astype(np.float64)[: len(gt)]
            t, r = h.score(est, gt)[:2]
            print(f"| {'/'.join(d.parts[-3:])} | {t:.2f} | {r:.2f} |")


if __name__ == "__main__":
    main()
