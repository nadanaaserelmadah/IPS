"""TUM-VI: accelerometer step length vs motion capture (ground truth independent of the IMU).

KITTI's GT positions come from the same OXTS GPS/INS unit whose accelerometer gives our step length
(RESULTS Table 7h), so the KITTI gain could be partly by construction. TUM-VI's mocap0 is optical
motion capture, so agreement here is independent evidence for the estimator itself.
Same estimator as scripts/euroc/imu_scale.py (VINS-style linear alignment over windows of W=20
contiguous 100 ms edges for v_0, gravity, accelerometer bias (ridge) and the step lengths), with the
direction of each step taken from mocap (the length is what is tested), in the IMU frame (the
mocap body is the IMU up to ~0.1-0.2 deg, Table 7f). Gyro bias: mean over the static start.
Self-contained (numpy + scipy) so it runs on a plain Kaggle CPU image.
  python scripts/tum_vi/imu_length_vs_mocap.py ROOT [OUT_JSON]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

WINDOW = 20
EDGE_S = 0.1
RIDGE_BA = 1.0
COND_MAX = 1e4


def read_csv(p: Path, usecols=None):
    return np.loadtxt(p, delimiter=",", comments="#", usecols=usecols, ndmin=2)


def exp(w):
    return Rotation.from_rotvec(w).as_matrix()


def static_bias(t, gyro, acc, block_s=0.25, min_s=0.5):
    n = max(int(round(block_s / np.median(np.diff(t)))), 2)
    end = 0
    for s in range(0, len(t) - n, n):
        g, a = gyro[s:s + n], np.linalg.norm(acc[s:s + n], axis=1)
        if np.linalg.norm(g - g.mean(0), axis=1).max() > 0.1 or a.std() > 0.3:
            break
        end = s + n
    return (gyro[:end].mean(0), t[end - 1] - t[0]) if end and t[end - 1] - t[0] >= min_s else (np.zeros(3), 0.0)


def preint(t, gyro, acc, b, t0, t1):
    tk = np.r_[t0, t[(t > t0) & (t < t1)], t1]
    ws = np.stack([np.interp(tk, t, gyro[:, j]) for j in range(3)], 1) - b
    As = np.stack([np.interp(tk, t, acc[:, j]) for j in range(3)], 1)
    R, al, be, Ja, Jb = np.eye(3), np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3))
    for k in range(len(tk) - 1):
        dt = tk[k + 1] - tk[k]
        Rn = R @ exp(0.5 * (ws[k] + ws[k + 1]) * dt)
        a = 0.5 * (R @ As[k] + Rn @ As[k + 1])
        Rm = 0.5 * (R + Rn)
        al += be * dt + 0.5 * a * dt * dt
        Ja += Jb * dt + 0.5 * Rm * dt * dt
        be += a * dt
        Jb += Rm * dt
        R = Rn
    return R, al, be, Ja, Jb


def solve(Rk, d, pre, dts):
    M = len(dts)
    n = 9 + M
    T = np.concatenate([[0.0], np.cumsum(dts)])
    rows, rhs = [], []
    sRb, sRJ = np.zeros(3), np.zeros((3, 3))
    for k in range(M):
        _, al, be, Ja, Jb = pre[k]
        A = np.zeros((3, n))
        A[:, 9 + k] = Rk[k] @ d[k]
        A[:, 0:3] = -np.eye(3) * dts[k]
        A[:, 3:6] = -np.eye(3) * (T[k] * dts[k] + 0.5 * dts[k] ** 2)
        A[:, 6:9] = sRJ * dts[k] + Rk[k] @ Ja
        rows.append(A)
        rhs.append(sRb * dts[k] + Rk[k] @ al)
        sRb, sRJ = sRb + Rk[k] @ be, sRJ + Rk[k] @ Jb
    A, y = np.vstack(rows), np.concatenate(rhs)
    cond = np.linalg.cond(A)
    A = np.vstack([A, np.hstack([np.zeros((3, 6)), RIDGE_BA * np.eye(3), np.zeros((3, M))])])
    x = np.linalg.lstsq(A, np.concatenate([y, np.zeros(3)]), rcond=None)[0]
    return x[9:], cond, np.linalg.norm(x[3:6])


def analyse(mav: Path) -> dict:
    imu, moc = read_csv(mav / "imu0/data.csv"), read_csv(mav / "mocap0/data.csv")
    cam = read_csv(mav / "cam0/data.csv", usecols=[0])[:, 0] * 1e-9
    t, gyro, acc = imu[:, 0] * 1e-9, imu[:, 1:4], imu[:, 4:7]
    tm, pm, q = moc[:, 0] * 1e-9, moc[:, 1:4], moc[:, 4:8]
    keep = np.r_[True, np.diff(tm) > 0]
    tm, pm, q = tm[keep], pm[keep], q[keep]
    slerp = Slerp(tm, Rotation.from_quat(q[:, [1, 2, 3, 0]]))
    b, static_s = static_bias(t, gyro, acc)
    edges_t = cam[::2]  # 100 ms edges, as on EuRoC
    near = np.abs(tm[np.clip(np.searchsorted(tm, edges_t), 0, len(tm) - 1)] - edges_t) < 0.015
    ok_t = near & (edges_t > max(tm[0], t[0]) + 0.05) & (edges_t < min(tm[-1], t[-1]) - 0.05)
    valid = ok_t[:-1] & ok_t[1:]
    # gyro attitude chain over all edges (world = IMU frame at the first edge)
    t0s, t1s = edges_t[:-1], edges_t[1:]
    pres = {}
    ratios, gts, ests, conds, gs = [], [], [], [], []
    runs = np.split(np.arange(len(valid)), np.where(np.diff(valid.astype(int)) != 0)[0] + 1)
    for run in runs:
        if not valid[run[0]] or len(run) < WINDOW // 2:
            continue
        for s in range(0, len(run) - WINDOW // 2 + 1, WINDOW):
            idx = run[s:s + WINDOW]
            if len(idx) < WINDOW // 2:
                continue
            for i in idx:
                if i not in pres:
                    pres[i] = preint(t, gyro, acc, b, t0s[i], t1s[i])
            Rk, R = [], np.eye(3)
            for i in idx:
                Rk.append(R)
                R = R @ pres[i][0]
            Rmoc0 = slerp(t0s[idx]).as_matrix()
            p0 = np.stack([np.interp(t0s[idx], tm, pm[:, j]) for j in range(3)], 1)
            p1 = np.stack([np.interp(t1s[idx], tm, pm[:, j]) for j in range(3)], 1)
            dp = p1 - p0
            d_body = np.einsum("nji,nj->ni", Rmoc0, dp)  # mocap displacement in the body frame at t0
            L = np.linalg.norm(d_body, axis=1)
            d = d_body / np.maximum(L, 1e-9)[:, None]
            lens, cond, gn = solve(Rk, d, [pres[i] for i in idx], t1s[idx] - t0s[idx])
            conds.append(cond)
            gs.append(gn)
            if cond >= COND_MAX:
                continue
            m = L > 0.01
            ratios.extend((lens[m] / L[m]).tolist())
            gts.extend(L[m].tolist())
            ests.extend(lens[m].tolist())
    r = np.array(ratios)
    q1, q3 = (np.percentile(r, [25, 75]) if len(r) else (np.nan, np.nan))
    return {"sequence": mav.parent.name, "static_start_s": float(static_s), "edges_scored": int(len(r)),
            "windows": len(conds), "windows_well_conditioned": int(np.sum(np.array(conds) < COND_MAX)),
            "len_ratio_median": float(np.median(r)) if len(r) else None, "len_ratio_iqr": [float(q1), float(q3)],
            "abs_len_err_median_cm": float(np.median(np.abs(np.array(ests) - np.array(gts))) * 100) if len(r) else None,
            "gt_step_median_cm": float(np.median(gts) * 100) if len(r) else None,
            "g_norm_median": float(np.median(gs)) if gs else None,
            "total_len_ratio": float(np.sum(ests) / np.sum(gts)) if len(r) else None}


def find_sequences(root: Path):
    for imu in sorted(root.rglob("imu0/data.csv")):
        mav = imu.parent.parent
        if (mav / "mocap0/data.csv").is_file() and (mav / "cam0/data.csv").is_file():
            yield mav


if __name__ == "__main__":
    out = [analyse(m) for m in find_sequences(Path(sys.argv[1]))]
    print("| sequence | edges | windows ok | |l|/|t_mocap| median | IQR | total length ratio | |err| median cm (step cm) | |g| | static s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in out:
        if r["len_ratio_median"] is None:
            print(f"| {r['sequence']} | 0 | {r['windows_well_conditioned']}/{r['windows']} | – | – | – | – | – | {r['static_start_s']:.1f} |")
            continue
        print(f"| {r['sequence']} | {r['edges_scored']} | {r['windows_well_conditioned']}/{r['windows']} | {r['len_ratio_median']:.3f} | "
              f"{r['len_ratio_iqr'][0]:.2f}–{r['len_ratio_iqr'][1]:.2f} | {r['total_len_ratio']:.3f} | "
              f"{r['abs_len_err_median_cm']:.2f} ({r['gt_step_median_cm']:.1f}) | {r['g_norm_median']:.2f} | {r['static_start_s']:.1f} |")
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(out, indent=2))
