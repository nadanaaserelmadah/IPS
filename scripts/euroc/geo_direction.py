"""EuRoC: translation direction from image geometry, given the gyro rotation (the P8-D fix to test).

P8-D showed that the learned translation *direction* does not transfer across rooms (55-90 deg),
while the network's step length is usable: GT direction x network length beats the zero-motion
floor on every sequence. Here the direction comes from the images instead of the network:
  1. track corners from frame A (t_end - 100 ms) to frame B (t_end) with pyramidal LK
     (forward-backward check), undistort to normalised bearings with cam0's radtan model;
  2. take the A->B rotation R from the gyro (P8-F: static-start bias, T_BS into the camera frame);
  3. with R known, each match gives one linear constraint t . ((R x_B) x x_A) = 0, so two matches
     fix the direction: 2-point RANSAC (1 px epipolar distance), SVD refit on the inliers,
     sign by cheirality (positive depths in both views);
  4. slow motion (de-rotated parallax < 2 px at 100 ms): the direction is re-measured over 200 / 300 ms
     (A' -> B) and rotated into A with the gyro (GEO_ADAPTIVE=0 disables this);
  5. edges with too little parallax or too few inliers keep the network direction.
TUM-VI (fisheye, no cam0/sensor.yaml): intrinsics and the equidistant model come from Kalibr's
dso/camchain.yaml; only corners within 60 deg of the optical axis are used (FISHEYE_MAX_TAN, fixed
before any TUM-VI run), where the normalised-plane pixel errors stay close to the image's.
Final pose per edge: R_gyro, t = d_geo * |t_net|. Nothing is trained or fitted on test data.

  python scripts/euroc/geo_direction.py EUROC_ROOT OUT_DIR TS_DIR:SEQS NAME=DUMP_DIR [NAME=DUMP_DIR ...]
  python scripts/euroc/geo_direction.py --selftest
TS_DIR holds <seq>_timestamps.npy / <seq>_gt_xyz.npy / <seq>_gt_poses.npy; SEQS is a comma list.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gyro_hybrid_euroc as gh  # noqa: E402  (also puts VIFT + shape_diag on the path)

EDGE_NS = gh.EDGE_NS
MIN_PARALLAX_PX = 0.5
MIN_INLIERS = 15
MIN_INLIER_RATIO = 0.5
RANSAC_ITERS = 300
EPI_PX = 1.0
ADAPTIVE = os.environ.get("GEO_ADAPTIVE", "1") == "1"
ADAPT_PX = 2.0  # below this de-rotated parallax at 100 ms, retry over 200 / 300 ms (fixed before any run)
FISHEYE_MAX_TAN = np.tan(np.radians(60.0))
SNAP_NS = 10_000_000  # TUM-VI frame times drift off the 50 ms grid (50.002 ms steps); EuRoC's are exact


def read_cam(cam_dir: Path):
    """K, distortion, timestamps, files. EuRoC: radtan array from sensor.yaml. TUM-VI: ("equidistant", k1..k4)
    from <seq>/dso/camchain.yaml."""
    import yaml

    if (cam_dir / "sensor.yaml").is_file():
        cfg = yaml.safe_load((cam_dir / "sensor.yaml").read_text())
        fu, fv, cu, cv = cfg["intrinsics"]
        dist = np.array(cfg["distortion_coefficients"], dtype=np.float64)
    else:
        cfg = yaml.safe_load((cam_dir.parents[1] / "dso" / "camchain.yaml").read_text())[cam_dir.name]
        fu, fv, cu, cv = cfg["intrinsics"]
        assert cfg["distortion_model"] == "equidistant", cfg["distortion_model"]
        dist = ("equidistant", np.array(cfg["distortion_coeffs"], dtype=np.float64))
    K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1.0]])
    rows = np.loadtxt(cam_dir / "data.csv", delimiter=",", comments="#", dtype=str)
    ts = rows[:, 0].astype(np.int64)
    return K, dist, ts, [cam_dir / "data" / f.strip() for f in rows[:, 1]]


def track(img_a, img_b, max_corners=500):
    import cv2

    pa = cv2.goodFeaturesToTrack(img_a, max_corners, 0.01, 8)
    if pa is None or len(pa) < MIN_INLIERS:
        return None, None
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    pb, st, _ = cv2.calcOpticalFlowPyrLK(img_a, img_b, pa, None, **lk)
    pa2, st2, _ = cv2.calcOpticalFlowPyrLK(img_b, img_a, pb, None, **lk)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(pa - pa2, axis=2).ravel() < 1.0)
    return pa[ok].reshape(-1, 2), pb[ok].reshape(-1, 2)


def bearings(px, K, dist):
    """Normalised-plane bearings (x, y, 1); fisheye corners beyond FISHEYE_MAX_TAN are NaN."""
    import cv2

    pts = px.reshape(-1, 1, 2).astype(np.float64)
    if len(pts) == 0:
        return np.zeros((0, 3))
    if isinstance(dist, tuple):
        n = cv2.fisheye.undistortPoints(pts, K, dist[1]).reshape(-1, 2)
        n[np.linalg.norm(n, axis=1) > FISHEYE_MAX_TAN] = np.nan
    else:
        n = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    return np.hstack([n, np.ones((len(n), 1))])


def direction(xa, xb, R, f_px, rng):
    """Unit t (A->B, X_A = R X_B + t) from normalised bearings with R known, or (None, info)."""
    ok = np.isfinite(xa).all(1) & np.isfinite(xb).all(1)
    xa, xb = xa[ok], xb[ok]
    rb = xb @ R.T  # R x_B, expressed in A
    par = np.linalg.norm(xa[:, :2] / xa[:, 2:] - rb[:, :2] / rb[:, 2:], axis=1) * f_px
    info = {"n": int(len(xa)), "parallax_px": float(np.median(par)) if len(par) else 0.0}
    if len(xa) < MIN_INLIERS or info["parallax_px"] < MIN_PARALLAX_PX:
        return None, info
    N = np.cross(rb, xa)  # t . N_i = 0

    def epi_px(t):  # distance of x_A to the epipolar line l = t x (R x_B), in pixels
        lines = np.cross(t, rb)
        return np.abs(np.sum(lines * xa, 1)) / (np.linalg.norm(lines[:, :2], axis=1) + 1e-12) * f_px

    best, best_in = None, np.zeros(len(xa), bool)
    for _ in range(RANSAC_ITERS):
        i, j = rng.choice(len(xa), 2, replace=False)
        t = np.cross(N[i], N[j])
        if np.linalg.norm(t) < 1e-12:
            continue
        t /= np.linalg.norm(t)
        inl = epi_px(t) < EPI_PX
        if inl.sum() > best_in.sum():
            best, best_in = t, inl
    if best is None or best_in.sum() < MIN_INLIERS or best_in.mean() < MIN_INLIER_RATIO:
        info["inliers"] = int(best_in.sum())
        return None, info
    Ni = N[best_in] / np.linalg.norm(N[best_in], axis=1, keepdims=True)
    _, sv, Vt = np.linalg.svd(Ni, full_matrices=False)
    t = Vt[-1]
    # first-order angular uncertainty of the null vector: residual std / smallest tangent singular value
    info["sigma_deg"] = float(np.degrees(sv[2] / (np.sqrt(max(len(Ni) - 2, 1)) * max(sv[1], 1e-12))))
    info["sv"] = [float(x) for x in sv]
    inl = epi_px(t) < EPI_PX
    # cheirality: lambda_A x_A - lambda_B (R x_B) = t, both depths positive for the right sign
    votes = 0
    for a, b in zip(xa[inl], rb[inl]):
        lam = np.linalg.lstsq(np.stack([a, -b], 1), t, rcond=None)[0]
        votes += 1 if (lam > 0).all() else (-1 if (lam < 0).all() else 0)
    info["inliers"] = int(inl.sum())
    return (t if votes >= 0 else -t), info


def stereo_points(img_l, img_r, cam_l, cam_r, R_rl, t_rl, max_corners=500):
    """Corners of the left image with metric 3D positions (left camera frame) from the stereo pair."""
    import cv2

    pl, pr = track(img_l, img_r, max_corners)
    if pl is None or len(pl) < MIN_INLIERS:
        return None, None
    xl, xr = bearings(pl, *cam_l), bearings(pr, *cam_r)
    X = cv2.triangulatePoints(np.hstack([np.eye(3), np.zeros((3, 1))]), np.hstack([R_rl, t_rl[:, None]]),
                              xl[:, :2].T, xr[:, :2].T)
    X = (X[:3] / X[3]).T
    Xr = X @ R_rl.T + t_rl
    rep = np.linalg.norm(Xr[:, :2] / Xr[:, 2:] - xr[:, :2], axis=1) * cam_r[0][0, 0]
    ok = (X[:, 2] > 0.3) & (X[:, 2] < 30) & (Xr[:, 2] > 0.3) & (rep < 1.0)
    return pl[ok], X[ok]


def metric_translation(X_a, xb, R, f_px, rng):
    """Metric t (X_A = R X_B + t) from 3D points in A and their bearings in B, R known (linear PnP)."""
    info = {"n_stereo": int(len(X_a))}
    if len(X_a) < MIN_INLIERS:
        return None, info
    Y = X_a @ R  # rows: R^T X_A
    Rt = R.T

    def skew(v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    A = np.stack([skew(x) @ Rt for x in xb])  # [x_B]x R^T t = [x_B]x R^T X_A
    b = np.stack([skew(x) @ y for x, y in zip(xb, Y)])

    def reproj_px(t):
        P = (X_a - t) @ R
        return np.linalg.norm(P[:, :2] / P[:, 2:] - xb[:, :2] / xb[:, 2:], axis=1) * f_px

    best, best_in = None, np.zeros(len(X_a), bool)
    for _ in range(RANSAC_ITERS):
        i = rng.choice(len(X_a), 2, replace=False)
        t = np.linalg.lstsq(A[i].reshape(-1, 3), b[i].reshape(-1), rcond=None)[0]
        inl = reproj_px(t) < 1.5
        if inl.sum() > best_in.sum():
            best, best_in = t, inl
    if best is None or best_in.sum() < MIN_INLIERS:
        info["st_inliers"] = int(best_in.sum())
        return None, info
    t = np.linalg.lstsq(A[best_in].reshape(-1, 3), b[best_in].reshape(-1), rcond=None)[0]
    info["st_inliers"] = int((reproj_px(t) < 1.5).sum())
    return t, info


def geo_sequence(root: Path, seq: str, t_end: np.ndarray, cache: Path):
    """Per-edge gyro rotations, geometric directions and stereo metric translations (camera frame),
    cached per sequence. The stereo translation is NaN where cam1 or the solve is missing."""
    f = cache / f"{seq}_geo.npz"
    if f.is_file():
        z = np.load(f)
        return z["R"], z["d"], z["t_st"], json.loads(str(z["info"]))
    import cv2

    mav = root / seq / "mav0"
    imu = np.loadtxt(mav / "imu0/data.csv", delimiter=",", comments="#")
    t_ns, gyro, acc = imu[:, 0].astype(np.int64), imu[:, 1:4], imu[:, 4:7]
    b = gh.bias_for(seq, t_ns, gyro, acc)
    T_bs0 = gh.load_T_BS(root / seq)
    R_bs = T_bs0[:3, :3]
    K, dist, cam_ts, files = read_cam(mav / "cam0")
    idx = {int(t): i for i, t in enumerate(cam_ts)}

    def frame_at(t):  # timestamp of the cam0 frame within SNAP_NS of t, or None
        j = int(np.clip(np.searchsorted(cam_ts, t), 1, len(cam_ts) - 1))
        j = j - 1 if abs(cam_ts[j - 1] - t) < abs(cam_ts[j] - t) else j
        return int(cam_ts[j]) if abs(int(cam_ts[j]) - t) < SNAP_NS else None
    stereo = (mav / "cam1" / "sensor.yaml").is_file()
    if stereo:
        K1, dist1, cam1_ts, files1 = read_cam(mav / "cam1")
        idx1 = {int(t): i for i, t in enumerate(cam1_ts)}
        T_10 = np.linalg.inv(gh.load_T_BS(root / seq, cam="cam1")) @ T_bs0  # cam0 -> cam1
    rng = np.random.default_rng(0)
    Rs, ds, ts_st, infos = [], [], [], []
    for t1 in t_end:
        t1 = frame_at(int(t1)) or int(t1)  # the loaders parse times via float: TUM-VI's come back ~100 ns off
        t0 = frame_at(int(t1) - EDGE_NS) or int(t1) - EDGE_NS
        Rb = gh.integrate(t_ns, gyro - b, t0, int(t1))
        Rc = R_bs.T @ Rb @ R_bs
        Rs.append(Rc)
        d, t_st, info = None, None, {"missing_image": True}
        if t0 in idx and int(t1) in idx:
            ia = cv2.imread(str(files[idx[t0]]), cv2.IMREAD_GRAYSCALE)
            ib = cv2.imread(str(files[idx[int(t1)]]), cv2.IMREAD_GRAYSCALE)
            pa, pb = track(ia, ib)
            info = {"n": 0}
            if pa is not None:
                d, info = direction(bearings(pa, K, dist), bearings(pb, K, dist), Rc, K[0, 0], rng)
            if ADAPTIVE and (d is None or info.get("parallax_px", 0.0) < ADAPT_PX):
                # slow motion: measure the direction over a longer baseline A' -> B and rotate it into A
                for m in (2, 3):
                    ta = frame_at(int(t1) - m * EDGE_NS)
                    if ta is None:
                        continue
                    R_ab = R_bs.T @ gh.integrate(t_ns, gyro - b, ta, int(t1)) @ R_bs
                    i2 = cv2.imread(str(files[idx[ta]]), cv2.IMREAD_GRAYSCALE)
                    q2, qb = track(i2, ib)
                    if q2 is None:
                        continue
                    d2, info2 = direction(bearings(q2, K, dist), bearings(qb, K, dist), R_ab, K[0, 0], rng)
                    if d2 is not None and info2["parallax_px"] >= ADAPT_PX:
                        R_aa = R_bs.T @ gh.integrate(t_ns, gyro - b, ta, t0) @ R_bs  # A' -> A
                        d = R_aa.T @ d2
                        info = dict(info2, baseline_ms=m * EDGE_NS // 1_000_000)
                        break
            if stereo and t0 in idx1:
                ir = cv2.imread(str(files1[idx1[t0]]), cv2.IMREAD_GRAYSCALE)
                ps, X = stereo_points(ia, ir, (K, dist), (K1, dist1), T_10[:3, :3], T_10[:3, 3])
                if ps is not None:
                    pa2, pb2 = track_points(ia, ib, ps)
                    keep = ~np.isnan(pa2[:, 0])
                    t_st, sinfo = metric_translation(X[keep], bearings(pb2[keep], K, dist), Rc, K[0, 0], rng)
                    info.update(sinfo)
        ds.append(np.full(3, np.nan) if d is None else d)
        ts_st.append(np.full(3, np.nan) if t_st is None else t_st)
        infos.append(info)
    R, d, t_st = np.array(Rs), np.array(ds), np.array(ts_st)
    np.savez(f, R=R, d=d, t_st=t_st, info=json.dumps(infos))
    return R, d, t_st, infos



def visual_gyro_bias(root: Path, seq: str, seconds=1e9, min_inl=30):
    """Gyro bias from the images, for sequences without a static start (no ground truth). Per 100 ms edge in the
    first `seconds`: 5-point essential matrix (RANSAC, 1 px) on tracked undistorted corners -> recoverPose rotation;
    bias = component-wise median of (raw gyro rotation vector - visual rotation vector, body frame) / dt. The
    frame convention of recoverPose is picked as the one with the smaller median residual. Returns (bias, info)."""
    import cv2
    from scipy.spatial.transform import Rotation

    mav = root / seq / "mav0"
    imu = np.loadtxt(mav / "imu0/data.csv", delimiter=",", comments="#")
    t_ns, gyro = imu[:, 0].astype(np.int64), imu[:, 1:4]
    R_bs = gh.load_T_BS(root / seq)[:3, :3]
    K, dist, cam_ts, files = read_cam(mav / "cam0")
    res = {0: [], 1: []}
    dts = []
    i, t_stop = 0, cam_ts[0] + int(seconds * 1e9)
    while i + 2 < len(cam_ts) and cam_ts[i + 2] < t_stop:
        t0, t1 = int(cam_ts[i]), int(cam_ts[i + 2])
        i += 2
        if not (t_ns[0] < t0 and t1 < t_ns[-1]):
            continue
        ia = cv2.imread(str(files[i - 2]), cv2.IMREAD_GRAYSCALE)
        ib = cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)
        pa, pb = track(ia, ib)
        if pa is None:
            continue
        xa, xb = bearings(pa, K, dist), bearings(pb, K, dist)
        ok = np.isfinite(xa).all(1) & np.isfinite(xb).all(1)
        xa, xb = xa[ok, :2], xb[ok, :2]
        if len(xa) < min_inl:
            continue
        E, mask = cv2.findEssentialMat(xa, xb, focal=1.0, pp=(0.0, 0.0), method=cv2.RANSAC, prob=0.999,
                                       threshold=1.0 / K[0, 0])
        if E is None or E.shape != (3, 3):
            continue
        n_in, Rr, _, _ = cv2.recoverPose(E, xa, xb, focal=1.0, pp=(0.0, 0.0), mask=mask)
        if n_in < min_inl or n_in < 0.5 * len(xa):
            continue
        th_gyro = Rotation.from_matrix(gh.integrate(t_ns, gyro, t0, t1)).as_rotvec()
        for c, Rc in ((0, Rr), (1, Rr.T)):
            th_vis = Rotation.from_matrix(R_bs @ Rc @ R_bs.T).as_rotvec()
            res[c].append(th_gyro - th_vis)
        dts.append((t1 - t0) * 1e-9)
    if len(dts) < 20:
        return None, {"edges": len(dts)}
    dts = np.array(dts)[:, None]
    c = min(res, key=lambda k: np.median(np.linalg.norm(np.array(res[k]), axis=1)))
    r = np.array(res[c]) / dts
    return np.median(r, axis=0), {"edges": len(dts), "convention": c,
                                  "residual_deg_per_edge": float(np.degrees(np.median(np.linalg.norm(np.array(res[c]), axis=1))))}

def track_points(img_a, img_b, pa):
    """LK-track given points A -> B with a forward-backward check; failed points are NaN."""
    import cv2

    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p = pa.reshape(-1, 1, 2).astype(np.float32)
    pb, st, _ = cv2.calcOpticalFlowPyrLK(img_a, img_b, p, None, **lk)
    pa2, st2, _ = cv2.calcOpticalFlowPyrLK(img_b, img_a, pb, None, **lk)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(p - pa2, axis=2).ravel() < 1.0)
    pb = pb.reshape(-1, 2).astype(np.float64)
    pb[~ok] = np.nan
    return np.where(ok[:, None], pa, np.nan), pb


def dir_err(t, g, moving):
    c = np.sum(t * g, 1) / (np.linalg.norm(t, axis=1) * np.linalg.norm(g, axis=1) + 1e-12)
    return float(np.median(np.degrees(np.arccos(np.clip(c[moving], -1, 1)))))


def evaluate(root, out, ts_dir, seqs, dumps):
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for seq in seqs:
        t_end = np.load(ts_dir / f"{seq}_timestamps.npy").astype(np.int64)
        gxyz = np.load(ts_dir / f"{seq}_gt_xyz.npy").reshape(-1, 3).astype(np.float64)
        gtp = np.load(ts_dir / f"{seq}_gt_poses.npy").astype(np.float64)
        Rg = [gh.euler_to_rotation_matrix(p[:3]) for p in gtp]
        tg = gtp[:, 3:]
        moving = np.linalg.norm(tg, axis=1) > 0.01
        R, d, t_st, infos = geo_sequence(root, seq, t_end, out)
        ok = ~np.isnan(d[:, 0])
        ok_st = ~np.isnan(t_st[:, 0])
        floor = gh.ate(np.zeros_like(gxyz), gxyz)[0]
        cov = float(ok.mean())
        geo_err = dir_err(d[ok & moving], tg[ok & moving], np.ones((ok & moving).sum(), bool))
        ng = np.linalg.norm(tg, axis=1)
        st_m = ok_st & moving
        st_len = float(np.median(np.linalg.norm(t_st[st_m], axis=1) / ng[st_m])) if st_m.any() else float("nan")
        st_dir = dir_err(t_st[st_m], tg[st_m], np.ones(st_m.sum(), bool)) if st_m.any() else float("nan")
        # network-free trajectory: stereo t, else geometric direction x last stereo length, else last stereo t
        t_pure, last = [], np.zeros(3)
        for k in range(len(t_end)):
            if ok_st[k]:
                last = t_st[k]
            t_pure.append(t_st[k] if ok_st[k] else (d[k] * np.linalg.norm(last) if ok[k] else last))
        pure = gh.ate(gh.compose(list(R), np.array(t_pure)), gxyz)
        print(f"\n## {seq}: floor {floor:.2f} m | geometric direction on {cov:.0%} of edges, "
              f"median error {geo_err:.1f} deg (moving edges with a geometric estimate)")
        print(f"stereo metric t on {ok_st.mean():.0%} of edges: median |t_st|/|t_gt| {st_len:.3f}, direction error "
              f"{st_dir:.1f} deg | **network-free (R_gyro + stereo t): SE3 {pure[0]:.2f} m**, Sim3 {pure[1]:.2f}, scale {pure[2]:.2f}")
        report.append({"seq": seq, "run": "network-free", "floor": floor, "stereo_cov": float(ok_st.mean()),
                       "stereo_len_ratio": st_len, "stereo_dir_err": st_dir, "se3": pure[0], "sim3": pure[1]})
        # mono-inertial, network-free (same sensors as the learned baselines): geometric direction
        # (previous one where missing) x accelerometer step length (imu_scale.py)
        import imu_scale

        d_full, last_d = [], np.array([0.0, 0.0, 1.0])
        for k in range(len(t_end)):
            if ok[k]:
                last_d = d[k]
            d_full.append(last_d)
        d_full = np.array(d_full)
        l_imu, gn = imu_scale.imu_lengths(root, seq, t_end, d_full)
        l_ratio = float(np.nanmedian(l_imu[moving] / ng[moving]))
        mono = gh.ate(gh.compose(list(R), d_full * np.nan_to_num(l_imu)[:, None]), gxyz)
        # with the gravity plausibility check (|g| within 0.5 of 9.81; rejected windows keep the previous speed)
        l_chk, _ = imu_scale.imu_lengths(root, seq, t_end, d_full, g_tol=0.5)
        L, last_l = [], 0.0
        for x in l_chk:
            last_l = x if not np.isnan(x) else last_l
            L.append(last_l)
        mono_chk = gh.ate(gh.compose(list(R), d_full * np.array(L)[:, None]), gxyz)
        print(f"accelerometer length: median |l|/|t_gt| {l_ratio:.3f}, |g| {np.median(gn):.2f} | "
              f"**mono-inertial network-free (R_gyro + d_geo·l_imu): SE3 {mono[0]:.2f} m**, Sim3 {mono[1]:.2f}, scale {mono[2]:.2f}"
              f" | with |g| check: **SE3 {mono_chk[0]:.2f} m**")
        report.append({"seq": seq, "run": "mono-inertial-geo", "floor": floor, "imu_len_ratio": l_ratio,
                       "se3": mono[0], "sim3": mono[1], "scale": mono[2], "se3_gcheck": mono_chk[0]})
        print("| run | net dir err | SE3 net | R_gyro + t_net | **R_gyro + d_geo·|t_net|** | R_gyro + d_geo·|t_gt| (oracle length) |")
        print("|---|---|---|---|---|---|")
        for name, dump in dumps.items():
            if not (dump / f"{seq}_estimated_poses.npy").is_file():
                continue
            est = np.load(dump / f"{seq}_estimated_poses.npy").astype(np.float64)
            assert np.allclose(np.load(dump / f"{seq}_gt_poses.npy"), gtp, atol=1e-6), (name, seq)
            Re = [gh.euler_to_rotation_matrix(p[:3]) for p in est]
            te = est[:, 3:]
            n_te = np.linalg.norm(te, axis=1, keepdims=True)
            t_geo = np.where(ok[:, None], np.nan_to_num(d) * n_te, te)
            t_geo_gt = np.where(ok[:, None], np.nan_to_num(d) * np.linalg.norm(tg, axis=1, keepdims=True), te)
            row = {
                "seq": seq, "run": name, "floor": floor, "coverage": cov, "geo_dir_err": geo_err,
                "net_dir_err": dir_err(te, tg, moving),
                "net": gh.ate(gh.compose(Re, te), gxyz)[0],
                "gyro_net": gh.ate(gh.compose(list(R), te), gxyz)[0],
                "gyro_geo": gh.ate(gh.compose(list(R), t_geo), gxyz)[0],
                "gyro_geo_gtlen": gh.ate(gh.compose(list(R), t_geo_gt), gxyz)[0],
            }
            report.append(row)
            print(f"| {name} | {row['net_dir_err']:.0f}° | {row['net']:.2f} | {row['gyro_net']:.2f} | "
                  f"**{row['gyro_geo']:.2f}** | {row['gyro_geo_gtlen']:.2f} |")
    (out / "geo_report.json").write_text(json.dumps(report, indent=1))


def selftest():
    import cv2

    rng = np.random.default_rng(1)
    K = np.array([[458.0, 0, 367.0], [0, 457.0, 248.0], [0, 0, 1]])
    dist = np.array([-0.28, 0.074, 2e-4, 2e-5])
    errs = []
    for _ in range(50):
        X_B = np.column_stack([rng.uniform(-3, 3, 300), rng.uniform(-2, 2, 300), rng.uniform(2, 8, 300)])
        rv = rng.normal(0, 0.02, 3)
        R = cv2.Rodrigues(rv)[0]
        t = rng.normal(0, 1, 3)
        t *= 0.07 / np.linalg.norm(t)
        X_A = X_B @ R.T + t
        keep = (X_A[:, 2] > 0.5)
        pa = cv2.projectPoints(X_A[keep], np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
        pb = cv2.projectPoints(X_B[keep], np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
        pa += rng.normal(0, 0.3, pa.shape)
        pb += rng.normal(0, 0.3, pb.shape)
        out = rng.random(len(pa)) < 0.2  # 20% gross outliers
        pb[out] += rng.uniform(-20, 20, (out.sum(), 2))
        d, info = direction(bearings(pa, K, dist), bearings(pb, K, dist), R, K[0, 0], rng)
        errs.append(np.degrees(np.arccos(np.clip(d @ (t / np.linalg.norm(t)), -1, 1))) if d is not None else 180)
    print(f"selftest: median direction error {np.median(errs):.2f} deg, max {np.max(errs):.2f} deg over 50 cases")
    assert np.median(errs) < 2.0 and np.max(errs) < 20, errs
    # TUM-VI-like equidistant fisheye (512 px, f ~ 190 px), points all around the camera
    Kf = np.array([[190.98, 0, 254.93], [0, 190.98, 256.90], [0, 0, 1]])
    D = ("equidistant", np.array([0.00348, 0.00072, -0.00205, 0.00020]))
    errs = []
    for _ in range(50):
        u = rng.normal(0, 1, (400, 3))
        u[:, 2] = np.abs(u[:, 2]) + 0.2
        X_B = u / np.linalg.norm(u, axis=1, keepdims=True) * rng.uniform(1.5, 5, (400, 1))
        R = cv2.Rodrigues(rng.normal(0, 0.02, 3))[0]
        t = rng.normal(0, 1, 3)
        t *= 0.07 / np.linalg.norm(t)
        X_A = X_B @ R.T + t
        keep = (X_A[:, 2] > 0.2) & (X_B[:, 2] > 0.2)
        pa = cv2.fisheye.projectPoints(X_A[keep][:, None], np.zeros(3), np.zeros(3), Kf, D[1])[0].reshape(-1, 2)
        pb = cv2.fisheye.projectPoints(X_B[keep][:, None], np.zeros(3), np.zeros(3), Kf, D[1])[0].reshape(-1, 2)
        pa += rng.normal(0, 0.3, pa.shape)
        pb += rng.normal(0, 0.3, pb.shape)
        out = rng.random(len(pa)) < 0.2
        pb[out] += rng.uniform(-20, 20, (out.sum(), 2))
        d, info = direction(bearings(pa, Kf, D), bearings(pb, Kf, D), R, Kf[0, 0], rng)
        errs.append(np.degrees(np.arccos(np.clip(d @ (t / np.linalg.norm(t)), -1, 1))) if d is not None else 180)
    print(f"selftest fisheye: median direction error {np.median(errs):.2f} deg, max {np.max(errs):.2f} deg over 50 cases")
    assert np.median(errs) < 3.0 and np.max(errs) < 25, errs


def main():
    if sys.argv[1] == "--selftest":
        return selftest()
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    ts_dir, seqs = sys.argv[3].split(":")
    dumps = dict(a.split("=", 1) for a in sys.argv[4:])
    evaluate(root, out, Path(ts_dir), seqs.split(","), {k: Path(v) for k, v in dumps.items()})


if __name__ == "__main__":
    main()
