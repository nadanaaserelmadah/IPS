"""TUM-VI: gyro rotation over each camera interval vs motion-capture rotation (independent of the IMU).

KITTI's GT comes from the same OXTS unit as its gyro, so the KITTI gyro result (Table 7d, r_rel 0.18,
per-frame median 0.023-0.037 deg) could be partly by construction. TUM-VI's mocap0 is an optical
motion-capture pose of the IMU body, so agreement here is independent evidence.
Per sequence: gyro (imu0, 200 Hz) integrated with the trapezoidal rule over each cam0 interval
(20 Hz); mocap orientation SLERP-interpolated at the interval ends; a constant time offset (scan
-20..20 ms) and an IMU->mocap-body rotation (Kabsch) fitted on the sequence's first half and
scored on the second half. Reports per-frame angle error and the drift of rotations composed over
2 s / 10 s windows, plus the same with nominal (fixed 5 ms) sub-steps.
  python scripts/tum_vi/gyro_vs_mocap.py ROOT [OUT_JSON]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def read_csv(p: Path, usecols=None) -> np.ndarray:
    return np.loadtxt(p, delimiter=",", comments="#", usecols=usecols, ndmin=2)


def find_sequences(root: Path):
    for imu in sorted(root.rglob("imu0/data.csv")):
        mav = imu.parent.parent
        if (mav / "mocap0/data.csv").is_file() and (mav / "cam0/data.csv").is_file():
            yield mav


def so3_log(R):
    return Rotation.from_matrix(R).as_rotvec()


def kabsch(a, b):
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def gyro_rel(t_imu, w, t_cam, lag, nominal_dt=None):
    out = []
    for a, b in zip(t_cam[:-1] + lag, t_cam[1:] + lag):
        tk = np.r_[a, t_imu[(t_imu > a) & (t_imu < b)], b]
        wk = np.stack([np.interp(tk, t_imu, w[:, c]) for c in range(3)], 1)
        R = np.eye(3)
        for k in range(len(tk) - 1):
            dt = nominal_dt if nominal_dt is not None else tk[k + 1] - tk[k]
            R = R @ Rotation.from_rotvec(0.5 * (wk[k] + wk[k + 1]) * dt).as_matrix()
        out.append(R)
    return np.array(out)


def analyse(mav: Path) -> dict:
    imu, moc, cam = read_csv(mav / "imu0/data.csv"), read_csv(mav / "mocap0/data.csv"), read_csv(mav / "cam0/data.csv", usecols=[0])  # second column is the image file name
    t_imu, w = imu[:, 0] * 1e-9, imu[:, 1:4]
    t_moc, q_wxyz = moc[:, 0] * 1e-9, moc[:, 4:8]
    t_cam = cam[:, 0] * 1e-9
    keep = np.r_[True, np.diff(t_moc) > 0]
    t_moc, q_wxyz = t_moc[keep], q_wxyz[keep]
    rot_moc = Rotation.from_quat(q_wxyz[:, [1, 2, 3, 0]])  # scipy wants x, y, z, w
    slerp = Slerp(t_moc, rot_moc)
    # camera intervals whose ends both have a mocap sample within 15 ms (mocap only covers the room)
    gap = np.abs(t_moc[np.clip(np.searchsorted(t_moc, t_cam), 0, len(t_moc) - 1)] - t_cam)
    ok = (gap < 0.015) & (t_cam > t_moc[0] + 0.05) & (t_cam < t_moc[-1] - 0.05) & (t_cam > t_imu[0] + 0.05) & (t_cam < t_imu[-1] - 0.05)
    valid = ok[:-1] & ok[1:]
    idx = np.where(valid)[0]
    Rm_abs = slerp(np.clip(t_cam, t_moc[0], t_moc[-1])).as_matrix()
    Rm = np.einsum("nji,njk->nik", Rm_abs[:-1], Rm_abs[1:])  # R_i^T R_{i+1}, body frame
    half = idx[: len(idx) // 2], idx[len(idx) // 2:]

    def fit(lag, nominal=None):
        Rg = gyro_rel(t_imu, w, t_cam, lag, nominal)
        a, b = np.array([so3_log(r) for r in Rg[half[0]]]), np.array([so3_log(r) for r in Rm[half[0]]])
        X = kabsch(a, b)
        return float(np.median(np.linalg.norm((X @ a.T).T - b, axis=1))), X, Rg

    res = {"sequence": mav.parent.name, "frames_with_mocap": int(len(idx)), "cam_dt_median_ms": float(np.median(np.diff(t_cam)) * 1e3)}
    for name, nominal in (("true_dt", None), ("nominal_5ms", 0.005)):
        lags = np.arange(-0.02, 0.0201, 0.0025)
        errs = [fit(l, nominal)[0] for l in lags]
        lag = float(lags[int(np.argmin(errs))])
        _, X, Rg = fit(lag, nominal)
        Rg = np.einsum("ij,njk,lk->nil", X, Rg, X)  # X R X^T: IMU frame -> mocap body frame
        test = half[1]
        e = np.degrees([np.linalg.norm(so3_log(Rg[i].T @ Rm[i])) for i in test])
        mag = np.degrees([np.linalg.norm(so3_log(Rm[i])) for i in test])
        drift = {}
        for horizon in (40, 200):  # 2 s and 10 s at 20 Hz, contiguous runs of valid frames only
            d = []
            for s0 in range(0, len(test) - horizon, horizon // 2):
                seg = test[s0:s0 + horizon]
                if seg[-1] - seg[0] != horizon - 1:
                    continue
                A, B = np.eye(3), np.eye(3)
                for i in seg:
                    A, B = A @ Rg[i], B @ Rm[i]
                d.append(np.degrees(np.linalg.norm(so3_log(A.T @ B))))
            drift[f"{horizon // 20}s"] = {"median_deg": float(np.median(d)) if d else None, "n": len(d)}
        res[name] = {"lag_ms": lag * 1e3, "extrinsic_angle_deg": float(np.degrees(np.linalg.norm(so3_log(X)))),
                     "per_frame_err_median_deg": float(np.median(e)), "per_frame_err_mean_deg": float(np.mean(e)),
                     "per_frame_rotation_median_deg": float(np.median(mag)), "drift": drift}
    return res


if __name__ == "__main__":
    root = Path(sys.argv[1])
    out = [analyse(m) for m in find_sequences(root)]
    for r in out:
        t, n = r["true_dt"], r["nominal_5ms"]
        print(f"{r['sequence']}: {r['frames_with_mocap']} frames, cam dt {r['cam_dt_median_ms']:.2f} ms | true dt: lag {t['lag_ms']:+.1f} ms, "
              f"extr {t['extrinsic_angle_deg']:.2f} deg, per-frame err median {t['per_frame_err_median_deg']:.4f} deg "
              f"(motion {t['per_frame_rotation_median_deg']:.3f} deg/frame), drift 2 s {t['drift']['2s']['median_deg']}, 10 s {t['drift']['10s']['median_deg']} deg "
              f"| nominal 5 ms: err {n['per_frame_err_median_deg']:.4f}, drift 10 s {n['drift']['10s']['median_deg']}")
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(out, indent=2))
