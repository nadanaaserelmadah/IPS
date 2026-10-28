"""KITTI baseline requested in review: does the network add anything once rotation and length are measured?

Pose per frame = gyro rotation + translation of accelerometer length (imu_scale_kitti, W = 500) along a
fixed direction (camera forward, +z), i.e. no network at all. Windows the accelerometer cannot resolve
(ill-conditioned, e.g. stops / constant speed, or OXTS gap) take either the median accelerometer length
(no network at all) or the network's length (isolates the direction).
Compared with the same pipeline using the network's direction (Table 7h) on the same predictions'
GT; scored against OXTS GT and against SuMa (fetch_suma_poses.py).
  python scripts/kitti/network_free_kitti.py DUMP_DIR NPZ_DIR GYRO_DIR RAW_CALIB_DIR SUMA_DIR [GEO_DIR]
GEO_DIR (optional) holds <s>_geo_dir.npz from geo_direction_kitti.py: geometric direction instead of +z.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gyro_hybrid as h  # noqa: E402
import imu_scale_kitti as isk  # noqa: E402

SEQS = ("05", "07", "10")


def lengths(npz_dir, gyro_dir, raw_calib, s, dirs):
    calib = json.loads((gyro_dir / f"calib_{isk.DAY[s]}.json").read_text())
    p_c, _ = isk.cam_in_imu(raw_calib, isk.DAY[s])
    lens, _, _ = isk.seq_lengths(npz_dir / f"{s}.npz", calib, p_c, dirs, 500, gap=h.load_gap(gyro_dir, s))
    return lens


def main():
    dump, npz_dir, gyro_dir, raw_calib, suma = (Path(a) for a in sys.argv[1:6])
    geo = Path(sys.argv[6]) if len(sys.argv) > 6 else None
    rows = {}
    for s in SEQS:
        gt = np.load(dump / f"{s}_gt_poses.npy").astype(np.float64)
        n = len(gt)
        est = np.load(dump / f"{s}_estimated_poses.npy").astype(np.float64)[:n]
        refs = {"OXTS": gt, "SuMa": isk.gt6_from_poses(suma / f"{s}.txt")[:n]}
        gR, gap = np.load(gyro_dir / f"{s}_gyro_R.npy"), h.load_gap(gyro_dir, s)
        variants = {"forward (+z)": np.tile([0.0, 0.0, 1.0], (n, 1))}
        te = est[:, 3:]
        variants["network direction"] = te / np.linalg.norm(te, axis=1, keepdims=True)
        if geo is not None:
            d = np.load(geo / f"{s}_geo_dir.npz")["d"][:n]
            ok = ~np.isnan(d[:, 0])
            variants["geometric direction (network where missing)"] = np.where(ok[:, None], d, variants["network direction"])
            fwd = np.tile([0.0, 0.0, 1.0], (n, 1))
            variants["geometric direction (+z where missing; no network)"] = np.where(ok[:, None], d, fwd)
        net_len = np.linalg.norm(te, axis=1)
        for name, dirs in variants.items():
            L = lengths(npz_dir, gyro_dir, raw_calib, s, dirs)
            for fill_name, fill in (("median accel. length", np.full(n, np.nanmedian(L))), ("network length", net_len)):
                e6 = est.copy()
                e6[:, 3:] = dirs * np.where(np.isnan(L), fill, L)[:, None]
                hyb = h.hybrid(e6, gR, gap)
                for ref, g in refs.items():
                    rows.setdefault((f"{name}; unresolved windows: {fill_name}", ref), []).append(h.score(hyb, g))
    print(f"\n{dump}")
    print("| direction | reference | 05 | 07 | 10 | mean t_rel % / r_rel |\n|---|---|---|---|---|---|")
    for (name, ref), v in rows.items():
        v = np.array(v)
        print(f"| {name} | {ref} | " + " | ".join(f"{x:.2f}" for x in v[:, 0]) + f" | **{v[:, 0].mean():.2f}** / {v[:, 1].mean():.2f} |")


if __name__ == "__main__":
    main()
