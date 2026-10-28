"""Cross-model check: the oracle swap and the sensor fixes on a learned VIO we did not train (VS-VIO, authors'
weights), KITTI 05 / 07 / 10, against OXTS GT and SuMa.

Rows: network; oracle GT rotation + network translation; oracle network rotation + GT translation;
gyro rotation + network translation; + accelerometer length (W = 500, network direction).
  python scripts/kitti/cross_model.py DUMP_DIR [DUMP_DIR ...] -- NPZ_DIR GYRO_DIR RAW_CALIB_DIR SUMA_DIR
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gyro_hybrid as h  # noqa: E402
import imu_scale_kitti as isk  # noqa: E402

SEQS = ("05", "07", "10")


def main():
    i = sys.argv.index("--")
    dumps = [Path(d) for d in sys.argv[1:i]]
    npz_dir, gyro_dir, raw_calib, suma = (Path(a) for a in sys.argv[i + 1:i + 5])
    print("| model | reference | network | oracle: GT rot + net t | oracle: net rot + GT t | gyro rot + net t | + accel length |")
    print("|---|---|---|---|---|---|---|")
    for d in dumps:
        rows = {ref: {k: [] for k in ("net", "gtR", "gtT", "gyro", "acc")} for ref in ("OXTS", "SuMa")}
        for s in SEQS:
            gt = np.load(d / f"{s}_gt_poses.npy").astype(np.float64)
            n = len(gt)
            est = np.load(d / f"{s}_estimated_poses.npy").astype(np.float64)[:n]
            gR, gap = np.load(gyro_dir / f"{s}_gyro_R.npy"), h.load_gap(gyro_dir, s)
            gtR = np.hstack([gt[:, :3], est[:, 3:]])
            gtT = np.hstack([est[:, :3], gt[:, 3:]])
            hyb = h.hybrid(est, gR, gap)
            e6, gap2 = isk.run_lengths(npz_dir, gyro_dir, raw_calib, s, est, 500)
            acc = h.hybrid(e6, gR, gap2)
            refs = {"OXTS": gt, "SuMa": isk.gt6_from_poses(suma / f"{s}.txt")[:n]}
            for ref, g in refs.items():
                for k, e in (("net", est), ("gtR", gtR), ("gtT", gtT), ("gyro", hyb), ("acc", acc)):
                    rows[ref][k].append(h.score(e, g))
        for ref in rows:
            cells = []
            for k in ("net", "gtR", "gtT", "gyro", "acc"):
                v = np.array(rows[ref][k])
                cells.append(f"{v[:, 0].mean():.2f} / {v[:, 1].mean():.2f}")
            print(f"| {d.parent.name}/{d.name} | {ref} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
