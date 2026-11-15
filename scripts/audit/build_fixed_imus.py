"""Audit paper, E4: build lag-corrected KITTI imus/*.mat in the exact VS-VIO format.

VS-VIO's imus/<s>.mat (`imu_data_interp`, 10 * n_frames - 9 rows: ax ay az wx wy wz) is the raw 100 Hz OXTS
linearly interpolated at 10 equal sub-steps of each real image_02 interval, with no time offset
(scripts/kitti/oxts_vs_mat.py). The samples therefore span the true ~104 ms interval, but they lag the motion
seen by the images and the GT poses by ~60 ms. Here the same interpolation is done at t_img + lag, with the lag
of each recording day from scripts/kitti/gyro_hybrid.py calibrate (fitted on training drives only; 2011_09_30:
+60 ms, 2011_10_03: +62.5 ms). Nothing else changes (format, sub-steps, columns, gap handling), so a model fed
these files differs from the released pipeline only in the IMU-camera alignment.

Check: with --lag0 the script must reproduce the released files (gyro RMSE ~1e-4 rad/s, the residual of
OXTS rounding); it compares against REF_DIR when given and prints the RMSE.
  python scripts/audit/build_fixed_imus.py NPZ_DIR CALIB_DIR OUT_DIR [--ref REF_DIR] [--lag0]
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

SEQS = ("00", "01", "02", "04", "05", "06", "07", "08", "09", "10")
DAY = {"00": "2011_10_03", "01": "2011_10_03", "02": "2011_10_03"} | {s: "2011_09_30" for s in ("04", "05", "06", "07", "08", "09", "10")}
COLS = [11, 12, 13, 17, 18, 19]  # OXTS ax ay az wx wy wz (vehicle frame x fwd, y left, z up)


def build(npz: Path, lag: float) -> np.ndarray:
    z = np.load(npz)
    t_ox, ox, t_img = z["t_ox"], z["ox"], z["t_img"]
    tq = (t_img[:-1, None] + np.arange(10)[None] * np.diff(t_img)[:, None] / 10).ravel()
    tq = np.r_[tq, t_img[-1]] + lag
    return np.stack([np.interp(tq, t_ox, ox[:, k]) for k in COLS], 1)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    npz_dir, calib_dir, out = (Path(a) for a in args[:3])
    ref = Path(sys.argv[sys.argv.index("--ref") + 1]) if "--ref" in sys.argv else None
    lag0 = "--lag0" in sys.argv
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for s in SEQS:
        if not (npz_dir / f"{s}.npz").is_file():
            print(f"{s}: no npz, skipped")
            continue
        lag = 0.0 if lag0 else json.loads((calib_dir / f"calib_{DAY[s]}.json").read_text())["lag_s"]
        m = build(npz_dir / f"{s}.npz", lag)
        sio.savemat(out / f"{s}.mat", {"imu_data_interp": m})
        meta[s] = {"lag_s": lag, "rows": len(m)}
        msg = f"{s}: lag {lag * 1e3:+.1f} ms, {m.shape}"
        if ref is not None and (ref / f"{s}.mat").is_file():
            r = sio.loadmat(ref / f"{s}.mat")["imu_data_interp"]
            assert r.shape == m.shape, (s, r.shape, m.shape)
            e = np.sqrt(np.mean((m - r) ** 2, 0))
            msg += f" | RMSE vs released: acc {np.round(e[:3], 4)} m/s^2, gyro {np.round(e[3:], 5)} rad/s"
        print(msg, flush=True)
    (out / "build_info.json").write_text(json.dumps({"lag0": lag0, "cols": COLS, "seqs": meta}, indent=2))


if __name__ == "__main__":
    main()
