"""Re-score KITTI results against SuMa lidar-SLAM poses (independent of the OXTS GPS/INS unit).

The review objection: KITTI's ground truth comes from the same OXTS unit as our gyro / accelerometer,
so r_rel 0.18 and the accelerometer step-length gains could be partly by construction. SemanticKITTI's
SuMa poses (fetch_suma_poses.py) are estimated from the Velodyne alone. Each prediction is scored
against both references with the same KITTI metric (t_rel %, r_rel deg/100 m over 100-800 m):
network, network translation + gyro rotation, and + accelerometer step length (imu_scale_kitti.py,
W = 500). The gyro / accelerometer calibration is unchanged (fitted on training drives).
Also reported: the OXTS GT itself scored against SuMa (how far the two references disagree).
  python scripts/kitti/score_vs_suma.py RUNS_JSON NPZ_DIR GYRO_DIR RAW_CALIB_DIR SUMA_DIR OXTS_GT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gyro_hybrid as h  # noqa: E402
import imu_scale_kitti as isk  # noqa: E402

SEQS = ("05", "07", "10")


def main():
    runs = json.loads(Path(sys.argv[1]).read_text())
    npz_dir, gyro_dir, raw_calib, suma_dir, oxts_dir = (Path(a) for a in sys.argv[2:7])
    refs = {s: {"OXTS": isk.gt6_from_poses(oxts_dir / f"{s}.txt"), "SuMa": isk.gt6_from_poses(suma_dir / f"{s}.txt")} for s in SEQS}
    ref_gap = [h.score(refs[s]["OXTS"], refs[s]["SuMa"]) for s in SEQS]
    print("Reference disagreement, OXTS GT scored against SuMa: "
          + ", ".join(f"{s} {t:.2f}% / {r:.2f}" for s, (t, r) in zip(SEQS, ref_gap))
          + f", mean {np.mean([t for t, _ in ref_gap]):.2f}% / {np.mean([r for _, r in ref_gap]):.2f}")
    print("\n| recipe | runs | reference | network | + gyro rotation | + accelerometer length (W=500) |")
    print("|---|---|---|---|---|---|")
    for recipe, dirs in runs.items():
        res = {ref: {k: [] for k in ("net", "gyro", "imu")} for ref in ("OXTS", "SuMa")}
        for d in dirs:
            d = Path(d)
            per = {ref: {k: [] for k in ("net", "gyro", "imu")} for ref in ("OXTS", "SuMa")}
            for s in SEQS:
                gt_dump = np.load(d / f"{s}_gt_poses.npy").astype(np.float64)
                est = np.load(d / f"{s}_estimated_poses.npy").astype(np.float64)[: len(gt_dump)]
                n = len(gt_dump)
                assert np.allclose(gt_dump, refs[s]["OXTS"][:n], atol=1e-4), (d, s)
                gR, gap = np.load(gyro_dir / f"{s}_gyro_R.npy"), h.load_gap(gyro_dir, s)
                hyb = h.hybrid(est, gR, gap)
                e6, gap2 = isk.run_lengths(npz_dir, gyro_dir, raw_calib, s, est, 500)
                acc = h.hybrid(e6, gR, gap2)
                for ref in ("OXTS", "SuMa"):
                    gt = refs[s][ref][:n]
                    for k, e in (("net", est), ("gyro", hyb), ("imu", acc)):
                        per[ref][k].append(h.score(e, gt))
            for ref in per:
                for k in per[ref]:
                    res[ref][k].append(np.mean(per[ref][k], 0))

        def fmt(v):
            v = np.array(v)
            t = f"{v[:, 0].mean():.2f} ± {v[:, 0].std(ddof=1):.2f}" if len(v) > 1 else f"{v[0, 0]:.2f}"
            return f"{t} / {v[:, 1].mean():.2f}"
        for ref in ("OXTS", "SuMa"):
            print(f"| {recipe} | {len(dirs)} | {ref} | {fmt(res[ref]['net'])} | {fmt(res[ref]['gyro'])} | **{fmt(res[ref]['imu'])}** |", flush=True)


if __name__ == "__main__":
    main()
