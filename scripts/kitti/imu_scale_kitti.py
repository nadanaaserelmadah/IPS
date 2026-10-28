"""KITTI: translation length from the raw OXTS accelerometer (same sensors as VIFT), gyro rotation.

With gyro rotation (gyro_hybrid.py) the remaining KITTI error is the network's step length: it
over-estimates distance by 1.1-1.8% on 05 / 07 / 10 (own-scale oracle 1.40%). Here the length comes
from the accelerometer, VINS-style: over windows of W contiguous image intervals we solve one linear
least-squares problem for the initial velocity v_0, gravity g (world), accelerometer bias b_a
(ridge-regularised) and either
  free : one length per edge, or
  scale: one factor s on the network's lengths (l_k = s |t_net,k|),
from
  R_k R_ic d_k l_k = v_k dt_k + 1/2 g dt_k^2 + R_k (alpha_k - J_alpha_k b_a) + (R_{k+1} - R_k) p_c
  v_k = v_0 + g T_k + sum_{j<k} R_j (beta_j - J_beta_j b_a)
R_k: gyro attitude (world = IMU at the first sample); R_ic: camera->IMU rotation (gyro_hybrid calib,
fitted on training sequences only); p_c: camera position in the IMU frame (KITTI raw calib files);
intervals shifted by the calibrated lag. OXTS velocities are NOT used (they are the GT source).
A constant-velocity window leaves the length unobservable; such windows (condition number above
COND_MAX) keep the network length.
  python scripts/kitti/imu_scale_kitti.py NPZ_DIR CALIB_DIR RAW_CALIB_DIR DUMP_DIR [DUMP_DIR ...]
  python scripts/kitti/imu_scale_kitti.py select NPZ_DIR CALIB_DIR RAW_CALIB_DIR POSES_DIR   (window choice, training seqs)
  python scripts/kitti/imu_scale_kitti.py batch NPZ_DIR CALIB_DIR RAW_CALIB_DIR RUNS_JSON   (all recipes, W=500 / 300, ensembles)
  python scripts/kitti/imu_scale_kitti.py sens NPZ_DIR CALIB_DIR RAW_CALIB_DIR POSES_DIR RUNS_JSON   (COND_MAX sensitivity)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gyro_hybrid as h  # noqa: E402
import gyro_rotation_test as g  # noqa: E402

SEQS = ("05", "07", "10")
DAY = {"00": "2011_10_03", "01": "2011_10_03", "02": "2011_10_03", "04": "2011_09_30", "05": "2011_09_30",
       "06": "2011_09_30", "07": "2011_09_30", "08": "2011_09_30", "09": "2011_09_30", "10": "2011_09_30"}
RIDGE_BA = 1.0
COND_MAX = 1e4
_PRE = {}
LAST_RMS = float("nan")


def read_rt(path: Path):
    vals = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                vals[k.strip()] = np.array([float(x) for x in v.split()])
            except ValueError:
                pass
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = vals["R"].reshape(3, 3), vals["T"]
    return T


def cam_in_imu(raw_calib: Path, day: str):
    T_velo_imu = read_rt(raw_calib / day / "calib_imu_to_velo.txt")
    T_cam_velo = read_rt(raw_calib / day / "calib_velo_to_cam.txt")
    T_cam_imu = T_cam_velo @ T_velo_imu
    return np.linalg.inv(T_cam_imu)[:3, 3], T_cam_imu[:3, :3]


def preint(t_ox, w, a, b_g, t0, t1):
    tk = np.r_[t0, t_ox[(t_ox > t0) & (t_ox < t1)], t1]
    wk = np.stack([np.interp(tk, t_ox, w[:, c]) for c in range(3)], 1) - b_g
    ak = np.stack([np.interp(tk, t_ox, a[:, c]) for c in range(3)], 1)
    R, al, be, Ja, Jb = np.eye(3), np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3))
    for k in range(len(tk) - 1):
        dt = tk[k + 1] - tk[k]
        Rn = R @ g.so3_exp(0.5 * (wk[k] + wk[k + 1]) * dt)
        acc = 0.5 * (R @ ak[k] + Rn @ ak[k + 1])
        Rm = 0.5 * (R + Rn)
        al += be * dt + 0.5 * acc * dt * dt
        Ja += Jb * dt + 0.5 * Rm * dt * dt
        be += acc * dt
        Jb += Rm * dt
        R = Rn
    return R, al, be, Ja, Jb


def solve(Rk, Rk1, dirs_imu, pre, dts, p_c, base_len=None):
    """Lengths for one window; base_len given -> one scale factor instead of free lengths."""
    M = len(dts)
    n_l = 1 if base_len is not None else M
    n = 9 + n_l
    T = np.concatenate([[0.0], np.cumsum(dts)])
    rows, rhs = [], []
    sRb, sRJ = np.zeros(3), np.zeros((3, 3))
    for k in range(M):
        _, al, be, Ja, Jb = pre[k]
        A = np.zeros((3, n))
        col = Rk[k] @ dirs_imu[k]
        if base_len is None:
            A[:, 9 + k] = col
        else:
            A[:, 9] = col * base_len[k]
        A[:, 0:3] = -np.eye(3) * dts[k]
        A[:, 3:6] = -np.eye(3) * (T[k] * dts[k] + 0.5 * dts[k] ** 2)
        A[:, 6:9] = sRJ * dts[k] + Rk[k] @ Ja
        rows.append(A)
        rhs.append(sRb * dts[k] + Rk[k] @ al + (Rk1[k] - Rk[k]) @ p_c)
        sRb, sRJ = sRb + Rk[k] @ be, sRJ + Rk[k] @ Jb
    A = np.vstack(rows)
    y = np.concatenate(rhs)
    cond = np.linalg.cond(A)
    A = np.vstack([A, np.hstack([np.zeros((3, 6)), RIDGE_BA * np.eye(3), np.zeros((3, n_l))])])
    x = np.linalg.lstsq(A, np.concatenate([y, np.zeros(3)]), rcond=None)[0]
    lens = x[9:] if base_len is None else x[9] * base_len
    global LAST_RMS
    LAST_RMS = float(np.sqrt(np.mean((A[: len(y)] @ x - y) ** 2)))  # alignment misfit (m), sensor-only quality
    return lens, cond, np.linalg.norm(x[3:6])


def seq_lengths(npz: Path, calib: dict, p_c, dirs_cam, window, base_len=None, gap=None, rms_out=None, cond_out=None):
    """rms_out / cond_out (optional ndarrays, len n): filled per edge with its window's alignment misfit / condition number."""
    z = np.load(npz)
    t_ox, ox, t_img = z["t_ox"], z["ox"], z["t_img"]
    w, a = ox[:, 17:20], ox[:, 11:14]
    lag, R_ci = calib["lag_s"], np.array(calib["R_cam_imu"])
    R_ic = R_ci.T
    b_g = np.zeros(3)  # KITTI drives start moving; the gyro bias is left to the per-window fit of b_a only
    t0s, t1s = t_img[:-1] + lag, t_img[1:] + lag
    n = len(dirs_cam)
    key = (str(npz), lag, n)
    if key not in _PRE:  # preintegration does not depend on the directions; cache it per sequence
        pre = [preint(t_ox, w, a, b_g, t0s[i], t1s[i]) for i in range(n)]
        # attitude chain at the interval boundaries (world = IMU frame at the first interval start)
        Rb = [np.eye(3)]
        for i in range(n):
            Rb.append(Rb[-1] @ pre[i][0])
        _PRE[key] = (pre, Rb)
    pre, Rb = _PRE[key]
    dirs_imu = dirs_cam @ R_ic.T
    out = np.full(n, np.nan)
    conds, gs = [], []
    for s in range(0, n, window):
        idx = np.arange(s, min(s + window, n))
        if len(idx) < max(10, window // 2) and s > 0:
            idx = np.arange(max(0, n - window), n)  # last short window: refit on the tail
        if gap is not None and (gap[idx] > h.GAP_S).any():
            continue
        lens, cond, gn = solve([Rb[i] for i in idx], [Rb[i + 1] for i in idx], dirs_imu[idx], [pre[i] for i in idx],
                               t1s[idx] - t0s[idx], p_c, None if base_len is None else base_len[idx])
        conds.append(cond)
        gs.append(gn)
        if rms_out is not None:
            rms_out[idx] = LAST_RMS
        if cond_out is not None:
            cond_out[idx] = cond
        if cond < COND_MAX:
            out[idx] = lens
    return out, np.array(conds), np.array(gs)


def main():
    npz_dir, calib_dir, raw_calib = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    dumps = [Path(d) for d in sys.argv[4:]]
    for window in (300, 500):
        print(f"\n### window {window} intervals (~{window * 0.104:.0f} s)")
        print("| translation | 05 | 07 | 10 | mean t_rel % | |g| median | windows used |")
        print("|---|---|---|---|---|---|---|")
        res = {}
        for di, dump in enumerate(dumps + [None]):
            for s in SEQS:
                ref = dumps[0]
                gt = np.load(ref / f"{s}_gt_poses.npy").astype(np.float64)
                gR = np.load(calib_dir / f"{s}_gyro_R.npy")
                gap = h.load_gap(calib_dir, s)
                calib = json.loads((calib_dir / f"calib_{DAY[s]}.json").read_text())
                p_c, _ = cam_in_imu(raw_calib, DAY[s])
                tg = gt[:, 3:]
                if dump is None:  # oracle: GT direction, IMU length
                    dirs, base, est, tag = tg / np.linalg.norm(tg, axis=1, keepdims=True), None, gt.copy(), "GT dir"
                    variants = {"GT direction x IMU length (free)": None}
                else:
                    est = np.load(dump / f"{s}_estimated_poses.npy").astype(np.float64)[: len(gt)]
                    te = est[:, 3:]
                    dirs = te / np.linalg.norm(te, axis=1, keepdims=True)
                    base = np.linalg.norm(te, axis=1)
                    variants = {"network (gyro rot.)": "net", "network dir x IMU length (free)": None,
                                "network length x IMU scale per window": base}
                for name, mode in variants.items():
                    key = (name if dump is None else f"{name} [{dump.parent.parent.name or dump.name}]")
                    if isinstance(mode, str):
                        e6 = est
                        ok_frac = 1.0
                    else:
                        lens, conds, gs = seq_lengths(npz_dir / f"{s}.npz", calib, p_c, dirs, window,
                                                      base_len=mode, gap=gap)
                        ok = ~np.isnan(lens)
                        fallback = np.linalg.norm(est[:, 3:], axis=1)
                        L = np.where(ok, lens, fallback)
                        e6 = est.copy()
                        e6[:, 3:] = dirs * L[:, None]
                        ok_frac = float(ok.mean())
                        res.setdefault(key, {}).setdefault("g", []).extend(gs.tolist())
                    t_rel, r_rel = h.score(h.hybrid(e6, gR, gap), gt)
                    res.setdefault(key, {})[s] = t_rel
                    res[key].setdefault("ok", []).append(ok_frac)
        for key, r in res.items():
            m = np.mean([r[s] for s in SEQS])
            gmed = np.median(r["g"]) if r.get("g") else float("nan")
            print(f"| {key} | {r['05']:.2f} | {r['07']:.2f} | {r['10']:.2f} | **{m:.2f}** | {gmed:.2f} | {np.mean(r['ok']):.0%} |")



def gt6_from_poses(poses_txt: Path):
    P = np.loadtxt(poses_txt).reshape(-1, 3, 4)
    T = np.tile(np.eye(4), (len(P), 1, 1))
    T[:, :3, :] = P
    rel = [np.linalg.inv(T[i]) @ T[i + 1] for i in range(len(T) - 1)]
    return np.array([np.r_[g.euler_from_matrix(r[:3, :3]), r[:3, 3]] for r in rel])


def select(npz_dir: Path, calib_dir: Path, raw_calib: Path, poses_dir: Path, seqs=("00", "01", "02", "04", "06", "08", "09")):
    """Window choice on TRAINING sequences only: GT direction, IMU length everywhere (no fallback)."""
    global COND_MAX
    keep, COND_MAX = COND_MAX, np.inf
    print("| window | " + " | ".join(seqs) + " | mean t_rel % |\n|---|" + "---|" * (len(seqs) + 1))
    for window in (50, 100, 200, 300, 500, 1000):
        row = []
        for s in seqs:
            gt = gt6_from_poses(poses_dir / f"{s}.txt")
            calib = json.loads((calib_dir / f"calib_{DAY[s]}.json").read_text())
            p_c, _ = cam_in_imu(raw_calib, DAY[s])
            gap = h.load_gap(calib_dir, s)
            n = min(len(gt), len(np.load(calib_dir / f"{s}_gyro_R.npy")))
            gt = gt[:n]
            dirs = gt[:, 3:] / np.maximum(np.linalg.norm(gt[:, 3:], axis=1, keepdims=True), 1e-9)
            lens, _, _ = seq_lengths(npz_dir / f"{s}.npz", calib, p_c, dirs, window, gap=None)
            e6 = gt.copy()
            e6[:, 3:] = dirs * np.nan_to_num(lens, nan=0.0)[:, None]
            row.append(h.score(h.hybrid(e6, np.load(calib_dir / f"{s}_gyro_R.npy")[:n], gap), gt)[0])
        print(f"| {window} | " + " | ".join(f"{r:.2f}" for r in row) + f" | **{np.mean(row):.2f}** |", flush=True)
    COND_MAX = keep


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "select":
    select(*[Path(a) for a in sys.argv[2:6]])
    sys.exit(0)


def run_lengths(npz_dir, calib_dir, raw_calib, s, est, window):
    """t_rel of one prediction (05/07/10 seq s): network translation vs network direction x IMU length."""
    calib = json.loads((calib_dir / f"calib_{DAY[s]}.json").read_text())
    p_c, _ = cam_in_imu(raw_calib, DAY[s])
    gap = h.load_gap(calib_dir, s)
    te = est[:, 3:]
    nt = np.linalg.norm(te, axis=1)
    dirs = te / np.maximum(nt, 1e-9)[:, None]
    lens, _, _ = seq_lengths(npz_dir / f"{s}.npz", calib, p_c, dirs, window, gap=gap)
    e6 = est.copy()
    e6[:, 3:] = dirs * np.where(np.isnan(lens), nt, lens)[:, None]
    return e6, gap


def batch(npz_dir, calib_dir, raw_calib, runs_json, windows=(500, 300)):
    """Every recipe of runs.json: per-run and ensemble t_rel with gyro rotation, network vs IMU length."""
    runs = json.loads(Path(runs_json).read_text())
    print("| recipe | runs | network (gyro rot.) | " + " | ".join(f"IMU length W={w}" for w in windows)
          + " | ensemble network | " + " | ".join(f"ensemble IMU W={w}" for w in windows) + " |")
    print("|---|---|---|" + "---|" * (2 * len(windows) + 1))
    for recipe, dirs in runs.items():
        per = {k: [] for k in ["net"] + list(windows)}
        ens = {}
        loaded = {s: [] for s in SEQS}
        for d in dirs:
            d = Path(d)
            row = {k: [] for k in per}
            for s in SEQS:
                gt = np.load(d / f"{s}_gt_poses.npy").astype(np.float64)
                est = np.load(d / f"{s}_estimated_poses.npy").astype(np.float64)[: len(gt)]
                loaded[s].append((est, gt))
                gR = np.load(calib_dir / f"{s}_gyro_R.npy")
                gap = h.load_gap(calib_dir, s)
                row["net"].append(h.score(h.hybrid(est, gR, gap), gt)[0])
                for w in windows:
                    e6, gap = run_lengths(npz_dir, calib_dir, raw_calib, s, est, w)
                    row[w].append(h.score(h.hybrid(e6, gR, gap), gt)[0])
            for k in per:
                per[k].append(np.mean(row[k]))
        if len(dirs) > 1:
            for k in ["net"] + list(windows):
                vals = []
                for s in SEQS:
                    gt = loaded[s][0][1]
                    mean = np.mean([e for e, _ in loaded[s]], 0)
                    gR = np.load(calib_dir / f"{s}_gyro_R.npy")
                    if k == "net":
                        e6, gap = mean, h.load_gap(calib_dir, s)
                    else:
                        e6, gap = run_lengths(npz_dir, calib_dir, raw_calib, s, mean, k)
                    vals.append(h.score(h.hybrid(e6, gR, gap), gt)[0])
                ens[k] = np.mean(vals)

        def fmt(v):
            v = np.array(v)
            return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v[0]:.2f}"
        print(f"| {recipe} | {len(dirs)} | {fmt(per['net'])} | " + " | ".join(f"**{fmt(per[w])}**" for w in windows)
              + " | " + (" | ".join(f"{ens[k]:.2f}" for k in ["net"] + list(windows)) if ens else " | ".join(["–"] * (len(windows) + 1)))
              + " |", flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "batch":
    batch(*[Path(a) for a in sys.argv[2:6]])
    sys.exit(0)


def sensitivity(npz_dir, calib_dir, raw_calib, poses_dir, runs_json, window=500, conds=(1e3, 1e4, 1e5, 1e6)):
    """COND_MAX sensitivity. Training drives (GT direction): share of edges kept and the length error of the
    kept ones. Test: t_rel of the two headline recipes per threshold (1e4 was fixed beforehand)."""
    global COND_MAX
    keep = COND_MAX
    train = ("00", "01", "02", "04", "06", "08", "09")
    print("| COND_MAX | train: edges using IMU | train: median abs(l/l_gt - 1) of kept edges | test: with-08 recipe (3) | test: no-08 recipe + mirror (3) |")
    print("|---|---|---|---|---|")
    runs = json.loads(Path(runs_json).read_text())
    heads = [runs["WITH 08: our recipe (all seeds)"], runs["no 08: our recipe + mirror"]]
    for c in conds:
        COND_MAX = c
        used, errs = [], []
        for s in train:
            gt = gt6_from_poses(poses_dir / f"{s}.txt")
            n = min(len(gt), len(np.load(calib_dir / f"{s}_gyro_R.npy")))
            gt = gt[:n]
            calib = json.loads((calib_dir / f"calib_{DAY[s]}.json").read_text())
            p_c, _ = cam_in_imu(raw_calib, DAY[s])
            L = np.linalg.norm(gt[:, 3:], axis=1)
            dirs = gt[:, 3:] / np.maximum(L, 1e-9)[:, None]
            lens, _, _ = seq_lengths(npz_dir / f"{s}.npz", calib, p_c, dirs, window, gap=h.load_gap(calib_dir, s))
            ok = ~np.isnan(lens) & (L > 0.1)
            used.append(np.mean(~np.isnan(lens)))
            errs.extend(np.abs(lens[ok] / L[ok] - 1).tolist())
        tests = []
        for dirs_list in heads:
            vals = []
            for d in dirs_list:
                d = Path(d)
                row = []
                for s in SEQS:
                    gt = np.load(d / f"{s}_gt_poses.npy").astype(np.float64)
                    est = np.load(d / f"{s}_estimated_poses.npy").astype(np.float64)[: len(gt)]
                    e6, gap = run_lengths(npz_dir, calib_dir, raw_calib, s, est, window)
                    row.append(h.score(h.hybrid(e6, np.load(calib_dir / f"{s}_gyro_R.npy"), gap), gt)[0])
                vals.append(np.mean(row))
            tests.append(f"{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}")
        print(f"| {c:.0e} | {np.mean(used):.0%} | {np.median(errs) * 100:.1f}% | {tests[0]} | {tests[1]} |", flush=True)
    COND_MAX = keep


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "sens":
    sensitivity(*[Path(a) for a in sys.argv[2:7]])
    sys.exit(0)


if __name__ == "__main__":
    main()
