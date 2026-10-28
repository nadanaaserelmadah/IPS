"""Observability-gated pose: per edge and per component, measure what the sensors observe, keep the
learned network elsewhere. One rule, one configuration, applied unchanged to KITTI and EuRoC.

  rotation   : gyro, always (bias-corrected, timestamp-correct).
  step length: accelerometer (VINS-style window alignment) where the window is observable
               (condition number < COND_MAX and |g| within G_TOL of 9.81), else the network's length.
  direction  : the network's, unless it is inconsistent with the two-view epipolar measurement:
               angle(d_net, d_geo) > K * sqrt(sigma_geo^2 + FLOOR^2) -> use d_geo.
               sigma_geo is the solver's first-order uncertainty; FLOOR (systematic error of the geometric
               direction) is the median geometric direction error on KITTI TRAINING drives 04/06/09; K = 3
               fixed in advance. Nothing is fitted on test sequences or on EuRoC.
  python scripts/gate/observability_gate.py floor KITTI_GEO_DIR OXTS_GT_DIR      (v1: FLOOR in degrees)
  python scripts/gate/observability_gate.py floorA KITTI_GEO_DIR OXTS_GT_DIR     (v2: A in deg x px; pass FLOOR as A=<value>)
  python scripts/gate/observability_gate.py kitti RUNS_JSON KITTI_GEO_DIR NPZ_DIR GYRO_DIR RAW_CALIB_DIR SUMA_DIR FLOOR_DEG
  python scripts/gate/observability_gate.py euroc EUROC_ROOT GEO_CACHE_DIR DUMPS_DIR FLOOR_DEG RUN [RUN ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/kitti"))
sys.path.insert(0, str(ROOT / "scripts/euroc"))

K_SIGMA = 3.0
COND_MAX = 1e4
G_TOL = 0.5


def angle_deg(a, b):
    c = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def direction_gate(d_net, d_geo, sigma, floor):
    """Returns the gated unit direction and a boolean 'geometry used' per edge.
    floor: scalar (v1, degrees) or per-edge array (v2, A / parallax_px)."""
    ok = ~np.isnan(d_geo[:, 0])
    thr = K_SIGMA * np.sqrt(np.nan_to_num(sigma, nan=1e3) ** 2 + np.asarray(floor) ** 2)
    use_geo = ok & (angle_deg(d_net, np.nan_to_num(d_geo)) > thr)
    d = np.where(use_geo[:, None], np.nan_to_num(d_geo), d_net)
    return d / np.linalg.norm(d, axis=1, keepdims=True), use_geo


def sigmas(infos, n):
    return np.array([float(infos[k].get("sigma_deg", np.nan)) if k < len(infos) else np.nan for k in range(n)])


def parallax(infos, n):
    return np.array([float(infos[k].get("parallax_px", np.nan)) if k < len(infos) else np.nan for k in range(n)])


def floor_v2(infos, n, A):
    """v2 systematic floor: A [deg x px] / de-rotated parallax [px]; edges without parallax get no geometry."""
    p = parallax(infos, n)
    return np.where(np.isfinite(p) & (p > 0), A / np.maximum(p, 1e-6), 1e3)


def floor_A_from_training(geo_dir: Path, gt_dir: Path, seqs=("04", "06", "09")):
    """v2: A = median over training frames of (geometric direction error [deg] x parallax [px])."""
    import imu_scale_kitti as isk

    prods = []
    for s in seqs:
        gt = isk.gt6_from_poses(gt_dir / f"{s}.txt")
        z = np.load(geo_dir / f"{s}_geo_dir.npz")
        d = z["d"]
        n = min(len(d), len(gt))
        p = parallax(json.loads(str(z["info"])), n)
        t, d = gt[:n, 3:], d[:n]
        ok = ~np.isnan(d[:, 0]) & (np.linalg.norm(t, axis=1) > 0.1) & np.isfinite(p)
        prods.extend((angle_deg(d[ok], t[ok]) * p[ok]).tolist())
    A = float(np.median(prods))
    print(f"A = median(direction error x parallax) on KITTI training drives {seqs}: {A:.3f} deg*px ({len(prods)} frames)")
    return A


def floor_from_training(geo_dir: Path, gt_dir: Path, seqs=("04", "06", "09")):
    import imu_scale_kitti as isk

    errs = []
    for s in seqs:
        gt = isk.gt6_from_poses(gt_dir / f"{s}.txt")
        d = np.load(geo_dir / f"{s}_geo_dir.npz")["d"]
        n = min(len(d), len(gt))
        t, d = gt[:n, 3:], d[:n]
        ok = ~np.isnan(d[:, 0]) & (np.linalg.norm(t, axis=1) > 0.1)
        errs.extend(angle_deg(d[ok], t[ok]).tolist())
    floor = float(np.median(errs))
    print(f"FLOOR = median geometric direction error on KITTI training drives {seqs}: {floor:.3f} deg ({len(errs)} frames)")
    return floor


def kitti(runs_json, geo_dir, npz_dir, gyro_dir, raw_calib, suma_dir, floor):
    import gyro_hybrid as h
    import imu_scale_kitti as isk

    runs = json.loads(Path(runs_json).read_text())
    print("| recipe | reference | network | gyro + network t | fixed: gyro + net dir + accel len | **gated** | geo dir used | accel len used |")
    print("|---|---|---|---|---|---|---|---|")
    for recipe, dirs in runs.items():
        res = {ref: {k: [] for k in ("net", "gyro", "fixed", "gate")} for ref in ("OXTS", "SuMa")}
        use_geo_all, use_acc_all = [], []
        for dd in dirs:
            dd = Path(dd)
            per = {ref: {k: [] for k in res[ref]} for ref in res}
            for s in ("05", "07", "10"):
                gt = np.load(dd / f"{s}_gt_poses.npy").astype(np.float64)
                n = len(gt)
                est = np.load(dd / f"{s}_estimated_poses.npy").astype(np.float64)[:n]
                gR, gap = np.load(gyro_dir / f"{s}_gyro_R.npy"), h.load_gap(gyro_dir, s)
                calib = json.loads((gyro_dir / f"calib_{isk.DAY[s]}.json").read_text())
                p_c, _ = isk.cam_in_imu(raw_calib, isk.DAY[s])
                te = est[:, 3:]
                L_net = np.linalg.norm(te, axis=1)
                d_net = te / np.maximum(L_net, 1e-9)[:, None]
                z = np.load(geo_dir / f"{s}_geo_dir.npz")
                d_geo = z["d"][:n]
                infos = json.loads(str(z["info"]))
                sig = sigmas(infos, n)
                fl = floor_v2(infos, n, floor[1]) if isinstance(floor, tuple) else floor
                d_gate, use_geo = direction_gate(d_net, d_geo, sig, fl)
                variants = {}
                for name, dirs_ in (("fixed", d_net), ("gate", d_gate)):
                    lens, _, gs = isk.seq_lengths(npz_dir / f"{s}.npz", calib, p_c, dirs_, 500, gap=gap)
                    use_acc = ~np.isnan(lens)
                    e6 = est.copy()
                    e6[:, 3:] = dirs_ * np.where(use_acc, lens, L_net)[:, None]
                    variants[name] = h.hybrid(e6, gR, gap)
                    if name == "gate":
                        use_acc_all.append(use_acc.mean())
                use_geo_all.append(use_geo.mean())
                refs = {"OXTS": gt, "SuMa": isk.gt6_from_poses(suma_dir / f"{s}.txt")[:n]}
                for ref, g in refs.items():
                    per[ref]["net"].append(h.score(est, g))
                    per[ref]["gyro"].append(h.score(h.hybrid(est, gR, gap), g))
                    per[ref]["fixed"].append(h.score(variants["fixed"], g))
                    per[ref]["gate"].append(h.score(variants["gate"], g))
            for ref in per:
                for k in per[ref]:
                    res[ref][k].append(np.mean(per[ref][k], 0)[0])
        for ref in ("OXTS", "SuMa"):
            f = lambda v: f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}" if len(v) > 1 else f"{v[0]:.2f}"  # noqa: E731
            print(f"| {recipe} | {ref} | {f(res[ref]['net'])} | {f(res[ref]['gyro'])} | {f(res[ref]['fixed'])} | **{f(res[ref]['gate'])}** | "
                  f"{np.mean(use_geo_all):.0%} | {np.mean(use_acc_all):.0%} |", flush=True)


def euroc(root, cache, dumps, floor, runs):
    import gyro_hybrid_euroc as gh
    import imu_scale

    seqs = ("MH_05_difficult", "V2_02_medium", "V1_01_easy", "V1_02_medium")
    print("| run | seq | floor | network | gyro + network t | fixed: network-free (geo dir + accel len) | **gated** | geo dir used | accel len used |")
    print("|---|---|---|---|---|---|---|---|---|")
    means = {}
    for run in runs:
        for seq in seqs:
            dump = dumps / run
            if not (dump / f"{seq}_estimated_poses.npy").is_file():
                continue
            ts = dumps / "pivift_dir_best"
            t_end = np.load(ts / f"{seq}_timestamps.npy").astype(np.int64)
            gx = np.load(ts / f"{seq}_gt_xyz.npy").reshape(-1, 3)
            est = np.load(dump / f"{seq}_estimated_poses.npy").astype(np.float64)
            z = np.load(cache / f"{seq}_geo.npz")
            R, d_geo = z["R"], z["d"]
            infos = json.loads(str(z["info"]))
            sig = sigmas(infos, len(t_end))
            fl = floor_v2(infos, len(t_end), floor[1]) if isinstance(floor, tuple) else floor
            te = est[:, 3:]
            L_net = np.linalg.norm(te, axis=1)
            d_net = te / np.maximum(L_net, 1e-9)[:, None]
            d_gate, use_geo = direction_gate(d_net, d_geo, sig, fl)
            l_acc, _ = imu_scale.imu_lengths(root, seq, t_end, d_gate, g_tol=G_TOL)
            use_acc = ~np.isnan(l_acc)
            t_gate = d_gate * np.where(use_acc, l_acc, L_net)[:, None]
            # fixed network-free reference (P8-H): geometric direction (previous where missing), accel length, previous speed
            okg = ~np.isnan(d_geo[:, 0])
            df, last = [], np.array([0.0, 0.0, 1.0])
            for k in range(len(t_end)):
                last = d_geo[k] if okg[k] else last
                df.append(last)
            df = np.array(df)
            lf, _ = imu_scale.imu_lengths(root, seq, t_end, df, g_tol=G_TOL)
            L, last_l = [], 0.0
            for x in lf:
                last_l = x if not np.isnan(x) else last_l
                L.append(last_l)
            Re = [gh.euler_to_rotation_matrix(p[:3]) for p in est]
            row = {"floor": gh.ate(np.zeros_like(gx), gx)[0], "net": gh.ate(gh.compose(Re, te), gx)[0],
                   "gyro": gh.ate(gh.compose(list(R), te), gx)[0],
                   "fixed": gh.ate(gh.compose(list(R), df * np.array(L)[:, None]), gx)[0],
                   "gate": gh.ate(gh.compose(list(R), t_gate), gx)[0]}
            means.setdefault(run, []).append(row)
            print(f"| {run} | {seq.split('_')[0]}_{seq.split('_')[1]} | {row['floor']:.2f} | {row['net']:.2f} | {row['gyro']:.2f} | "
                  f"{row['fixed']:.2f} | **{row['gate']:.2f}** | {use_geo.mean():.0%} | {use_acc.mean():.0%} |", flush=True)
    for run, rows in means.items():
        m = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
        print(f"| {run} | mean ({len(rows)} seqs) | {m['floor']:.2f} | {m['net']:.2f} | {m['gyro']:.2f} | {m['fixed']:.2f} | **{m['gate']:.2f}** | | |")


def parse_floor(a):
    """'0.685' -> v1 degrees floor; 'A=12.3' -> v2 pixel-based floor (A / parallax)."""
    return ("A", float(a[2:])) if a.startswith("A=") else float(a)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "floor":
        floor_from_training(Path(sys.argv[2]), Path(sys.argv[3]))
    elif mode == "floorA":
        floor_A_from_training(Path(sys.argv[2]), Path(sys.argv[3]))
    elif mode == "kitti":
        kitti(sys.argv[2], *(Path(a) for a in sys.argv[3:8]), parse_floor(sys.argv[8]))
    elif mode == "euroc":
        euroc(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), parse_floor(sys.argv[5]), sys.argv[6:])
