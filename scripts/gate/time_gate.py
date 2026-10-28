"""Runtime of the measurement side of gate v3 per 100 ms edge, single CPU thread (OpenCV and BLAS pinned to 1).

Parts, timed on the first N edges of a sequence:
  image I/O (two grayscale frames), corner tracking (goodFeaturesToTrack + forward-backward LK),
  two-view direction with known rotation (undistortion + 2-point RANSAC + SVD + cheirality),
  the full geometry pass as run (geo_sequence, incl. the adaptive 200/300 ms retries), gyro integration,
  the accelerometer window solves (twice, as in C5b: network and fused directions), and the gate / fusion.
The network forward pass is not included (it runs on the GPU and is the same with or without the gate).
  python scripts/gate/time_gate.py NAME:ROOT:SEQ:TS_DIR [NAME:ROOT:SEQ:TS_DIR ...] [--n 300]
"""
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for sub in ("scripts/euroc", "scripts/gate"):
    sys.path.insert(0, str(ROOT / sub))
import gate_v3_select as V  # noqa: E402
import geo_direction as GD  # noqa: E402
import gyro_hybrid_euroc as gh  # noqa: E402
import imu_scale  # noqa: E402
import observability_gate as G  # noqa: E402

cv2.setNumThreads(1)


def ms(f, reps=1):
    t = time.perf_counter()
    for _ in range(reps):
        out = f()
    return 1e3 * (time.perf_counter() - t) / reps, out


def time_sequence(root: Path, seq: str, ts_dir: Path, n: int) -> dict:
    t_end = np.load(ts_dir / f"{seq}_timestamps.npy").astype(np.int64)[:n]
    mav = root / seq / "mav0"
    K, dist, cam_ts, files = GD.read_cam(mav / "cam0")
    imu = np.loadtxt(mav / "imu0/data.csv", delimiter=",", comments="#")
    t_ns, gyro, acc = imu[:, 0].astype(np.int64), imu[:, 1:4], imu[:, 4:7]
    b, _ = gh.static_bias(t_ns, gyro, acc)
    b = np.zeros(3) if b is None else b
    R_bs = gh.load_T_BS(root / seq)[:3, :3]
    near = lambda t: int(cam_ts[np.argmin(np.abs(cam_ts - t))])  # noqa: E731
    idx = {int(t): i for i, t in enumerate(cam_ts)}
    rng = np.random.default_rng(0)
    io, trk, dirn, gyr = [], [], [], []
    for t1 in t_end:
        t1 = near(int(t1))
        t0 = near(t1 - GD.EDGE_NS)
        a, (ia, ib) = ms(lambda: (cv2.imread(str(files[idx[t0]]), cv2.IMREAD_GRAYSCALE),
                                  cv2.imread(str(files[idx[t1]]), cv2.IMREAD_GRAYSCALE)))
        io.append(a)
        a, R = ms(lambda: R_bs.T @ gh.integrate(t_ns, gyro - b, t0, t1) @ R_bs, reps=20)
        gyr.append(a)
        a, (pa, pb) = ms(lambda: GD.track(ia, ib))
        trk.append(a)
        if pa is not None:
            a, _ = ms(lambda: GD.direction(GD.bearings(pa, K, dist), GD.bearings(pb, K, dist), R, K[0, 0], rng))
            dirn.append(a)
    with tempfile.TemporaryDirectory() as tmp:
        a, (Rg, d_geo, _, inf) = ms(lambda: GD.geo_sequence(root, seq, t_end, Path(tmp)))
    geo_full = a / len(t_end)
    nn = len(t_end)
    d_net = d_geo.copy()
    bad = np.isnan(d_net[:, 0])
    d_net[bad] = [0.0, 0.0, 1.0]
    d_net = d_net + rng.normal(0, 0.02, d_net.shape)  # stand-in for the network direction (timing only)
    d_net /= np.linalg.norm(d_net, axis=1, keepdims=True)
    sig, fl = G.sigmas(inf, nn), G.floor_v2(inf, nn, V.A_V2)
    a, (d_v, _) = ms(lambda: V.fuse(d_net, d_geo, sig, fl, 0.829), reps=50)
    fuse = a / nn
    a, _ = ms(lambda: (imu_scale.imu_lengths(root, seq, t_end, d_v, g_tol=G.G_TOL),
                       imu_scale.imu_lengths(root, seq, t_end, d_net, g_tol=G.G_TOL)))
    accel = a / nn
    med = lambda v: float(np.median(v)) if len(v) else float("nan")  # noqa: E731
    return {"edges": nn, "image_io": med(io), "tracking": med(trk), "direction": med(dirn),
            "geometry_full_pass": geo_full, "gyro": med(gyr), "accel_two_solves": accel, "fusion_gate": fuse,
            "total": geo_full + med(gyr) + accel + fuse}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 300
    args = [a for a in args if a != str(n)]
    rows = {}
    for spec in args:
        name, root, seq, ts = spec.split(":")
        rows[name] = time_sequence(Path(root), seq, Path(ts), n)
        print(name, json.dumps({k: round(v, 3) for k, v in rows[name].items()}), flush=True)
    keys = [("image_io", "image I/O (2 frames)"), ("tracking", "corner tracking (LK, fwd-bwd)"),
            ("direction", "two-view direction (RANSAC + SVD)"), ("geometry_full_pass", "geometry as run (incl. adaptive retries)"),
            ("gyro", "gyro integration"), ("accel_two_solves", "accelerometer windows (2 solves, amortised)"),
            ("fusion_gate", "fusion + gate"), ("total", "**total per edge**")]
    print("\n| ms per edge, 1 CPU thread | " + " | ".join(rows) + " |\n|---|" + "---|" * len(rows))
    for k, lab in keys:
        print(f"| {lab} | " + " | ".join(f"{r[k]:.2f}" for r in rows.values()) + " |")


if __name__ == "__main__":
    main()
