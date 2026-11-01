"""EuRoC: metric step length from the accelerometer, given per-edge directions (cam0 + IMU only).

The geometric direction (geo_direction.py) needs one camera; its length does not. Here the length
comes from the IMU, as in VINS-style visual-inertial alignment, so the sensors match the learned
baselines (cam0 + IMU). Over windows of M contiguous 100 ms edges we solve one linear least-squares
problem for the initial velocity v_0, gravity g (world), accelerometer bias b_a (body, ridge-
regularised) and the M step lengths l_k:
  camera displacement   R_k R_bs d_k l_k
    = v_k dt_k + 1/2 g dt_k^2 + R_k (alpha_k - J_alpha_k b_a) + (R_{k+1} - R_k) p_bs
  v_k = v_0 + g T_k + sum_{j<k} R_j (beta_j - J_beta_j b_a)
with R_k the gyro attitude (static-start bias), alpha / beta the preintegrated position / velocity
over edge k and p_bs the cam0 lever arm. World yaw is arbitrary; g absorbs the initial tilt.
Nothing is fitted on test ground truth.
  python scripts/euroc/imu_scale.py EUROC_ROOT TS_DIR [GEO_DIR] [SEQ ...]
Without GEO_DIR the GT directions are used (validation of the length estimate only).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gyro_hybrid_euroc as gh  # noqa: E402
from shape_diag import rotvec  # noqa: E402

EDGE_NS = gh.EDGE_NS
WINDOW = 20
RIDGE_BA = 1.0  # weight pulling b_a towards 0 (m/s^2 units; the bias is ~0.1)
LAST_RMS = float("nan")


def attitude(t_ns, gyro, b):
    """World-from-body rotation at every IMU sample (trapezoid, R(t_0) = I)."""
    R = np.empty((len(t_ns), 3, 3))
    R[0] = np.eye(3)
    for i in range(1, len(t_ns)):
        R[i] = R[i - 1] @ gh.so3_exp(0.5 * (gyro[i - 1] + gyro[i] - 2 * b) * (t_ns[i] - t_ns[i - 1]) * 1e-9)
    return R


def att_at(t_ns, Rs, t):
    i = int(np.clip(np.searchsorted(t_ns, t) - 1, 0, len(t_ns) - 2))
    f = (t - t_ns[i]) / (t_ns[i + 1] - t_ns[i])
    return Rs[i] @ gh.so3_exp(f * rotvec(Rs[i].T @ Rs[i + 1]))


def preintegrate(t_ns, gyro, acc, b, t0, t1):
    """alpha, beta (body frame at t0) and their Jacobians w.r.t. the accelerometer bias."""
    i0, i1 = np.searchsorted(t_ns, t0), np.searchsorted(t_ns, t1)
    ts = np.concatenate([[t0], t_ns[i0:i1], [t1]]).astype(np.float64)
    ws = np.stack([np.interp(ts, t_ns, gyro[:, j]) for j in range(3)], 1) - b
    As = np.stack([np.interp(ts, t_ns, acc[:, j]) for j in range(3)], 1)
    R, al, be, Ja, Jb = np.eye(3), np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3))
    for k in range(len(ts) - 1):
        dt = (ts[k + 1] - ts[k]) * 1e-9
        R_next = R @ gh.so3_exp(0.5 * (ws[k] + ws[k + 1]) * dt)
        a = 0.5 * (R @ As[k] + R_next @ As[k + 1])
        Rm = 0.5 * (R + R_next)
        al += be * dt + 0.5 * a * dt * dt
        Ja += Jb * dt + 0.5 * Rm * dt * dt
        be += a * dt
        Jb += Rm * dt
        R = R_next
    return al, be, Ja, Jb


def solve_window(Rk, Rk1, d_cam, pre, dts, R_bs, p_bs):
    M = len(dts)
    n = 9 + M  # v0, g, b_a, l_1..l_M
    rows, rhs = [], []
    T = np.concatenate([[0.0], np.cumsum(dts)])
    sum_Rb = np.zeros(3)
    sum_RJb = np.zeros((3, 3))
    for k in range(M):
        al, be, Ja, Jb = pre[k]
        A = np.zeros((3, n))
        A[:, 9 + k] = Rk[k] @ R_bs @ d_cam[k]
        A[:, 0:3] = -np.eye(3) * dts[k]
        A[:, 3:6] = -np.eye(3) * (T[k] * dts[k] + 0.5 * dts[k] ** 2)
        A[:, 6:9] = sum_RJb * dts[k] + Rk[k] @ Ja
        rows.append(A)
        rhs.append(sum_Rb * dts[k] + Rk[k] @ al + (Rk1[k] - Rk[k]) @ p_bs)
        sum_Rb = sum_Rb + Rk[k] @ be
        sum_RJb = sum_RJb + Rk[k] @ Jb
    A = np.vstack(rows + [np.hstack([np.zeros((3, 6)), RIDGE_BA * np.eye(3), np.zeros((3, M))])])
    y = np.concatenate(rhs + [np.zeros(3)])
    x = np.linalg.lstsq(A, y, rcond=None)[0]
    global LAST_RMS
    LAST_RMS = float(np.sqrt(np.mean((A[: 3 * M] @ x - y[: 3 * M]) ** 2)))  # alignment misfit (m), sensor-only quality
    return x[9:], x[3:6], x[6:9]


def imu_lengths(root: Path, seq: str, t_end: np.ndarray, d_cam: np.ndarray, window: int = WINDOW, g_tol: float | None = None,
                rms_out: np.ndarray | None = None):
    """Step lengths per edge (NaN where not solved). g_tol: reject windows whose solved |g| is more than
    g_tol m/s^2 from 9.81 (physical plausibility check; 0.5 was fixed before looking at any result)."""
    mav = root / seq / "mav0"
    imu = np.loadtxt(mav / "imu0/data.csv", delimiter=",", comments="#")
    t_ns, gyro, acc = imu[:, 0].astype(np.int64), imu[:, 1:4], imu[:, 4:7]
    b = gh.bias_for(seq, t_ns, gyro, acc)
    T_bs = gh.load_T_BS(root / seq)
    R_bs, p_bs = T_bs[:3, :3], T_bs[:3, 3]
    Rs = attitude(t_ns, gyro, b)
    t0s = t_end - EDGE_NS
    # TUM-VI frame times drift off the 50 ms grid: an edge starts at the previous edge's end when within 10 ms
    near = np.abs(t0s[1:] - t_end[:-1]) < 10_000_000
    t0s[1:][near] = t_end[:-1][near]
    # contiguous runs of edges (a dropped window breaks the chain)
    breaks = np.where(t0s[1:] != t_end[:-1])[0] + 1
    lengths = np.full(len(t_end), np.nan)
    gnorm = []
    for run in np.split(np.arange(len(t_end)), breaks):
        starts = list(range(0, len(run), window))
        if len(starts) > 1 and len(run) - starts[-1] < window // 2:
            starts.pop()  # merge a short tail into the previous window
        for si, s in enumerate(starts):
            e = starts[si + 1] if si + 1 < len(starts) else len(run)
            idx = run[s:e]
            if len(idx) < 5:
                continue
            Rk = [att_at(t_ns, Rs, t0s[i]) for i in idx]
            Rk1 = [att_at(t_ns, Rs, t_end[i]) for i in idx]
            pre = [preintegrate(t_ns, gyro, acc, b, t0s[i], t_end[i]) for i in idx]
            dts = (t_end[idx] - t0s[idx]) * 1e-9
            l, g, ba = solve_window(Rk, Rk1, d_cam[idx], pre, dts, R_bs, p_bs)
            gnorm.append(np.linalg.norm(g))
            if rms_out is not None:
                rms_out[idx] = LAST_RMS
            if g_tol is not None and abs(gnorm[-1] - 9.81) > g_tol:
                continue
            lengths[idx] = l
    return lengths, np.array(gnorm)


def main():
    root, ts_dir = Path(sys.argv[1]), Path(sys.argv[2])
    rest = sys.argv[3:]
    geo_dir = Path(rest.pop(0)) if rest and Path(rest[0]).is_dir() else None
    seqs = rest or ["MH_05_difficult", "V2_02_medium"]
    rng = np.random.default_rng(0)
    for seq in seqs:
        t_end = np.load(ts_dir / f"{seq}_timestamps.npy").astype(np.int64)
        gtp = np.load(ts_dir / f"{seq}_gt_poses.npy").astype(np.float64)
        gxyz = np.load(ts_dir / f"{seq}_gt_xyz.npy").reshape(-1, 3).astype(np.float64)
        tg = gtp[:, 3:]
        ng = np.linalg.norm(tg, axis=1)
        moving = ng > 0.01
        Rg = [gh.euler_to_rotation_matrix(p[:3]) for p in gtp]
        variants = {"GT direction": tg / np.maximum(ng, 1e-9)[:, None]}
        noisy = variants["GT direction"] + rng.normal(0, np.radians(2.0), tg.shape)
        variants["GT direction + 2 deg noise"] = noisy / np.linalg.norm(noisy, axis=1, keepdims=True)
        if geo_dir is not None:
            z = np.load(geo_dir / f"{seq}_geo.npz")
            d = z["d"]
            ok = ~np.isnan(d[:, 0])
            variants["geometric direction (GT where missing)"] = np.where(ok[:, None], d, variants["GT direction"])
        print(f"\n## {seq} (floor {gh.ate(np.zeros_like(gxyz), gxyz)[0]:.2f} m)")
        print("| directions | |l|/|t_gt| median (moving) | IQR | |g| median | SE3 (R_gt + d·l) | Sim3 | scale |\n|---|---|---|---|---|---|---|")
        for name, d in variants.items():
            l, gn = imu_lengths(root, seq, t_end, d)
            r = l[moving] / ng[moving]
            q1, q3 = np.nanpercentile(r, [25, 75])
            t = d * np.nan_to_num(l)[:, None]
            se3, sim3, sc = gh.ate(gh.compose(Rg, t), gxyz)
            print(f"| {name} | {np.nanmedian(r):.3f} | {q1:.2f}–{q3:.2f} | {np.median(gn):.2f} | {se3:.2f} | {sim3:.2f} | {sc:.2f} |")


if __name__ == "__main__":
    main()
