"""Gate v4 design on DESIGN data only: KITTI 08 (three no-08 models, as for v3) and TUM-VI room1-4 (three
zero-shot EuRoC models; room5/6 are the test rooms and are never read here).

Fixed in advance from first principles (not searched): P_CONF = 5 px, S_CONF = 1 deg (geometry error well below
the network's typical KITTI error), M = 100, M_MIN = 20. Only TAU is chosen, from {3, 5, 10, 20} deg.
Selection rule (fixed before running this): the TAU with the lowest mean TUM room1-4 ATE among those whose KITTI 08
t_rel (vs OXTS) is within 0.02 of v3 (C5b); ties go to the larger TAU (closer to v3).
  python scripts/gate/gate_v4_select.py kitti PRED08_DIRS(comma) GEO08_NPZ NPZ_DIR GYRO_DIR RAW_CALIB_DIR
  python scripts/gate/gate_v4_select.py tumvi ROOT GEO_DIR DUMPS_DIR RUN [RUN ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("scripts/kitti", "scripts/euroc", "scripts/gate"):
    sys.path.insert(0, str(ROOT / sub))
import gate_v3_select as V  # noqa: E402
import gate_v4 as W  # noqa: E402
import gyro_hybrid as h  # noqa: E402
import imu_scale_kitti as isk  # noqa: E402

FIXED = {"P_CONF": 5.0, "S_CONF": 1.0, "M": 100, "M_MIN": 20}
TAUS = (3.0, 5.0, 10.0, 20.0)
DESIGN_ROOMS = ("dataset-room1_512_16", "dataset-room2_512_16", "dataset-room3_512_16", "dataset-room4_512_16")


def kitti(preds, geo_npz, npz_dir, gyro_dir, raw_calib):
    s = "08"
    calib = json.loads((gyro_dir / f"calib_{isk.DAY[s]}.json").read_text())
    p_c, _ = isk.cam_in_imu(raw_calib, isk.DAY[s])
    gap = h.load_gap(gyro_dir, s)
    gR = np.load(gyro_dir / f"{s}_gyro_R.npy")
    npz = npz_dir / f"{s}.npz"
    rows = {}
    for p in preds:
        gt = np.load(p / f"{s}_gt_poses.npy").astype(np.float64)
        est = np.load(p / f"{s}_estimated_poses.npy").astype(np.float64)[: len(gt)]
        n = len(gt)
        te = est[:, 3:]
        L_net = np.linalg.norm(te, axis=1)
        d_net = te / np.maximum(L_net, 1e-9)[:, None]
        z = np.load(geo_npz)
        d_geo, infos = z["d"][:n], json.loads(str(z["info"]))[:n]
        lengths = lambda dirs: V.lengths(npz, calib, p_c, dirs, gap)  # noqa: E731
        res, _, _ = V.candidates(s, est, gt, geo_npz, npz_dir, gyro_dir, raw_calib, W.SIGMA_NET_PRIOR)
        rows.setdefault("v3 (C5b)", []).append(h.score(h.hybrid(res["C5b (fusion, per-window min misfit)"], gR, gap), gt)[0])
        for tau in TAUS:
            d, L, sh = W.v4(d_net, d_geo, infos, lengths, dict(FIXED, TAU=tau))
            e6 = est.copy()
            e6[:, 3:] = d * np.where(np.isnan(L), L_net, L)[:, None]
            rows.setdefault(f"v4 TAU={tau:g}", []).append(h.score(h.hybrid(e6, gR, gap), gt)[0])
            if tau == TAUS[0]:
                print(f"{p.parent.name}: sigma_hat median {np.median(sh):.2f} deg, p90 {np.percentile(sh, 90):.2f}, "
                      f"distrusted at TAU=3: {np.mean(sh > 3):.0%}", flush=True)
    print("| KITTI 08 (design) | t_rel % vs OXTS per model | mean |\n|---|---|---|")
    for k, v in rows.items():
        print(f"| {k} | " + " / ".join(f"{x:.2f}" for x in v) + f" | **{np.mean(v):.2f}** |")
    return {k: float(np.mean(v)) for k, v in rows.items()}


def tumvi(root, geo_dir, dumps, runs):
    import gyro_hybrid_euroc as gh
    import imu_scale
    import observability_gate as G

    rows = {}
    for run in runs:
        for seq in DESIGN_ROOMS:
            d = dumps / run
            t_end = np.load(d / f"{seq}_timestamps.npy").astype(np.int64)
            gx = np.load(d / f"{seq}_gt_xyz.npy").reshape(-1, 3)
            est = np.load(d / f"{seq}_estimated_poses.npy").astype(np.float64)
            z = np.load(geo_dir / f"{seq}_geo.npz")
            R, d_geo, infos = z["R"], z["d"], json.loads(str(z["info"]))
            n = len(t_end)
            te = est[:, 3:]
            L_net = np.linalg.norm(te, axis=1)
            d_net = te / np.maximum(L_net, 1e-9)[:, None]

            def lengths(dirs):
                rms = np.full(n, np.nan)
                L, _ = imu_scale.imu_lengths(root, seq, t_end, dirs, g_tol=G.G_TOL, rms_out=rms)
                return L, rms

            sig, fl = G.sigmas(infos, n), G.floor_v2(infos, n, V.A_V2)
            d3, _ = V.fuse(d_net, d_geo, sig, fl, W.SIGMA_NET_PRIOR)
            L3, r3 = lengths(d3)
            Ln, rn = lengths(d_net)
            L3 = np.where(rn < r3, Ln, L3)
            ate = lambda dd, LL: gh.ate(gh.compose(list(R), dd * np.where(np.isnan(LL), L_net, LL)[:, None]), gx)[0]  # noqa: E731
            rows.setdefault("v3 (C5b)", []).append(ate(d3, L3))
            okg = ~np.isnan(d_geo[:, 0])
            df, last = d_geo.copy(), np.array([0.0, 0.0, 1.0])
            for k in range(n):
                last = d_geo[k] if okg[k] else last
                df[k] = last
            Lf, _ = imu_scale.imu_lengths(root, seq, t_end, df, g_tol=G.G_TOL)
            rows.setdefault("network-free", []).append(ate(df, np.where(np.isnan(Lf), 0.0, Lf)))
            rows.setdefault("network", []).append(gh.ate(gh.compose([gh.euler_to_rotation_matrix(q[:3]) for q in est], te), gx)[0])
            for tau in TAUS:
                dv, Lv, sh = W.v4(d_net, d_geo, infos, lengths, dict(FIXED, TAU=tau))
                rows.setdefault(f"v4 TAU={tau:g}", []).append(ate(dv, Lv))
            print(f"{run} {seq}: sigma_hat median {np.median(sh):.1f} deg; v3 {rows['v3 (C5b)'][-1]:.2f}, "
                  f"v4 TAU=20 {rows['v4 TAU=20'][-1]:.2f}, network-free {rows['network-free'][-1]:.2f}", flush=True)
    print("| TUM-VI room1-4 (design) | mean SE(3) ATE (m) over runs x rooms |\n|---|---|")
    for k, v in rows.items():
        print(f"| {k} | **{np.mean(v):.3f}** |")
    return {k: float(np.mean(v)) for k, v in rows.items()}


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "kitti":
        kitti([Path(p) for p in a[2].split(",")], *(Path(x) for x in a[3:7]))
    else:
        tumvi(Path(a[2]), Path(a[3]), Path(a[4]), a[5:])
