"""Gate v3 design on HELD-OUT KITTI 08 only (no-08 models, checkpoint picked on 08 val loss but never trained
on it). Candidates differ only in how the gated direction is coupled into the accelerometer length solve:
  C0  v2 as pre-registered (lengths solved with the gated direction)
  C2  per window, solve lengths with the network direction and with the gated direction; keep the solution
      with the smaller alignment misfit (sensor-only)
  C4  never switch to geometry on IMU-stationary edges (|gyro| < 0.02 rad/s and accel-norm std < 0.05 m/s^2
      over the interval: physical thresholds fixed in advance), then as C0
  C5  inverse-variance fusion of the two directions where they are consistent (angle <= 3 sigma_combined):
      d ~ d_net / sigma_net^2 + d_geo / sigma_geo_eff^2, sigma_geo_eff^2 = sigma_geo^2 + (A / parallax)^2;
      sigma_net = median network direction error on held-out 08 (validation, not test); inconsistent edges
      follow the v2 gate. C5a: lengths solved with the fused direction; C5b: per-window min misfit (as C2).
All other gate constants as v2 (K = 3, A = 6.755 deg*px, cond < 1e4, |g| tol 0.5).
  python scripts/gate/gate_v3_select.py PRED08_DIRS(comma) GEO08_NPZ NPZ_DIR GYRO_DIR RAW_CALIB_DIR SUMA_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/kitti"))
sys.path.insert(0, str(ROOT / "scripts/gate"))
import gyro_hybrid as h  # noqa: E402
import imu_scale_kitti as isk  # noqa: E402
import observability_gate as G  # noqa: E402

A_V2 = 6.755


def stationary(npz, lag, n):
    z = np.load(npz)
    t_ox, ox, t_img = z["t_ox"], z["ox"], z["t_img"]
    w, a = ox[:, 17:20], ox[:, 11:14]
    out = np.zeros(n, bool)
    for k in range(n):
        m = (t_ox >= t_img[k] + lag) & (t_ox <= t_img[k + 1] + lag)
        if m.sum() >= 3:
            out[k] = np.linalg.norm(w[m], axis=1).max() < 0.02 and np.linalg.norm(a[m], axis=1).std() < 0.05
    return out


def lengths(npz, calib, p_c, dirs, gap):
    rms = np.full(len(dirs), np.nan)
    L, _, _ = isk.seq_lengths(npz, calib, p_c, dirs, 500, gap=gap, rms_out=rms)
    return L, rms


def fuse(d_net, d_geo, sig, fl, sigma_net):
    """Inverse-variance fusion where consistent, v2 gate where not. Returns direction and a 'fused' mask."""
    ok = ~np.isnan(d_geo[:, 0])
    sg2 = np.nan_to_num(sig, nan=1e3) ** 2 + np.asarray(fl) ** 2
    dg = np.nan_to_num(d_geo)
    ang = G.angle_deg(d_net, dg)
    consistent = ok & (ang <= G.K_SIGMA * np.sqrt(sg2 + sigma_net ** 2))
    d_gate, _ = G.direction_gate(d_net, d_geo, sig, fl)
    fused = d_net / sigma_net ** 2 + dg / sg2[:, None]
    fused /= np.linalg.norm(fused, axis=1, keepdims=True)
    return np.where(consistent[:, None], fused, d_gate), consistent


def candidates(s, est, gt, geo_npz, npz_dir, gyro_dir, raw_calib, sigma_net=None):
    n = len(gt)
    te = est[:, 3:]
    L_net = np.linalg.norm(te, axis=1)
    d_net = te / np.maximum(L_net, 1e-9)[:, None]
    z = np.load(geo_npz)
    d_geo, infos = z["d"][:n], json.loads(str(z["info"]))
    sig, fl = G.sigmas(infos, n), G.floor_v2(infos, n, A_V2)
    d_gate, use = G.direction_gate(d_net, d_geo, sig, fl)
    calib = json.loads((gyro_dir / f"calib_{isk.DAY[s]}.json").read_text())
    p_c, _ = isk.cam_in_imu(raw_calib, isk.DAY[s])
    gap = h.load_gap(gyro_dir, s)
    npz = npz_dir / f"{s}.npz"
    out = {}
    L_fix, _ = lengths(npz, calib, p_c, d_net, gap)
    out["hand-chosen (net dir + accel len)"] = (d_net, L_fix)
    L0, r0 = lengths(npz, calib, p_c, d_gate, gap)
    out["C0 (v2)"] = (d_gate, L0)
    Ln, rn = L_fix, lengths(npz, calib, p_c, d_net, gap)[1]
    pick_net = rn < r0
    out["C2 (per-window min misfit)"] = (d_gate, np.where(pick_net, Ln, L0))
    if sigma_net is not None:
        d5, cons = fuse(d_net, d_geo, sig, fl, sigma_net)
        L5, r5 = lengths(npz, calib, p_c, d5, gap)
        out["C5a (fusion, lengths from fused dir)"] = (d5, L5)
        out["C5b (fusion, per-window min misfit)"] = (d5, np.where(rn < r5, Ln, L5))
    st = stationary(npz, calib["lag_s"], n)
    d4 = np.where((use & ~st)[:, None], d_gate, d_net)
    L4, _ = lengths(npz, calib, p_c, d4, gap)
    out["C4 (no switch when stationary)"] = (d4, L4)
    res = {}
    for name, (d, L) in out.items():
        e6 = est.copy()
        e6[:, 3:] = d * np.where(np.isnan(L), L_net, L)[:, None]
        res[name] = e6
    return res, use.mean(), st.mean()


def main():
    preds = [Path(p) for p in sys.argv[1].split(",")]
    geo_npz, npz_dir, gyro_dir, raw_calib, suma = (Path(a) for a in sys.argv[2:7])
    rows = {}
    # sigma_net: median network direction error on held-out 08 (validation), pooled over the models
    errs = []
    for p in preds:
        gt = np.load(p / "08_gt_poses.npy").astype(np.float64)
        te = np.load(p / "08_estimated_poses.npy").astype(np.float64)[: len(gt), 3:]
        mv = np.linalg.norm(gt[:, 3:], axis=1) > 0.1
        errs.extend(G.angle_deg(te[mv], gt[mv, 3:]).tolist())
    sigma_net = float(np.median(errs))
    print(f"sigma_net (median network direction error on held-out 08) = {sigma_net:.3f} deg")
    for p in preds:
        gt = np.load(p / "08_gt_poses.npy").astype(np.float64)
        est = np.load(p / "08_estimated_poses.npy").astype(np.float64)[: len(gt)]
        res, used, st = candidates("08", est, gt, geo_npz, npz_dir, gyro_dir, raw_calib, sigma_net)
        gR, gap = np.load(gyro_dir / "08_gyro_R.npy"), h.load_gap(gyro_dir, "08")
        sm = isk.gt6_from_poses(suma / "08.txt")[: len(gt)]
        for name, e6 in res.items():
            hy = h.hybrid(e6, gR, gap)
            rows.setdefault(name, []).append((h.score(hy, gt)[0], h.score(hy, sm)[0]))
        rows.setdefault("network only", []).append((h.score(est, gt)[0], h.score(est, sm)[0]))
        rows.setdefault("gyro + network t", []).append((h.score(h.hybrid(est, gR, gap), gt)[0], h.score(h.hybrid(est, gR, gap), sm)[0]))
        print(f"{p}: geometry switched on {used:.0%}, IMU-stationary edges {st:.0%}")
    print("| candidate (held-out 08) | t_rel % vs OXTS (per model) | mean | vs SuMa mean |\n|---|---|---|---|")
    for name, v in rows.items():
        v = np.array(v)
        print(f"| {name} | " + " / ".join(f"{x:.2f}" for x in v[:, 0]) + f" | **{v[:, 0].mean():.2f}** | {v[:, 1].mean():.2f} |")


if __name__ == "__main__":
    main()
