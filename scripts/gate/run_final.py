"""Final, single run of the chosen gate v3 variant on the TEST data (KITTI 05/07/10 for VIFT and VS-VIO,
EuRoC MH_05/V2_02/V1_01/V1_02). The variant must have been chosen on held-out KITTI 08 beforehand
(gate_v3_select.py) and pre-registered in RESULTS.md; this script is run once per variant.

Variants (coupling of the gated direction into the accelerometer length solve; see gate_v3_select.py):
  C0, C2, C4, C5a, C5b.  SIGMA_NET (C5) is the held-out-08 value, used unchanged on EuRoC.
  python scripts/gate/run_final.py VARIANT SIGMA_NET KITTI_RUNS_JSON KITTI_GEO NPZ_DIR GYRO_DIR RAW_CALIB SUMA \
      EUROC_ROOT EUROC_GEO EUROC_DUMPS RUN [RUN ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("scripts/kitti", "scripts/euroc", "scripts/gate"):
    sys.path.insert(0, str(ROOT / sub))
import gate_v3_select as V  # noqa: E402
import gyro_hybrid as h  # noqa: E402
import imu_scale_kitti as isk  # noqa: E402
import observability_gate as G  # noqa: E402

NAMES = {"C0": "C0 (v2)", "C2": "C2 (per-window min misfit)", "C4": "C4 (no switch when stationary)",
         "C5a": "C5a (fusion, lengths from fused dir)", "C5b": "C5b (fusion, per-window min misfit)"}


def kitti(variant, sigma_net, runs_json, geo, npz_dir, gyro_dir, raw_calib, suma):
    runs = json.loads(Path(runs_json).read_text())
    print(f"\n### KITTI test, variant {variant}")
    print("| model | reference | network | gyro + net t | hand-chosen (net dir + accel len) | **gate v3** |\n|---|---|---|---|---|---|")
    for recipe, dirs in runs.items():
        acc = {ref: {k: [] for k in ("net", "gyro", "hand", "gate")} for ref in ("OXTS", "SuMa")}
        for d in dirs:
            d = Path(d)
            per = {ref: {k: [] for k in acc[ref]} for ref in acc}
            for s in ("05", "07", "10"):
                gt = np.load(d / f"{s}_gt_poses.npy").astype(np.float64)
                est = np.load(d / f"{s}_estimated_poses.npy").astype(np.float64)[: len(gt)]
                res, _, _ = V.candidates(s, est, gt, geo / f"{s}_geo_dir.npz", npz_dir, gyro_dir, raw_calib, sigma_net)
                gR, gap = np.load(gyro_dir / f"{s}_gyro_R.npy"), h.load_gap(gyro_dir, s)
                refs = {"OXTS": gt, "SuMa": isk.gt6_from_poses(suma / f"{s}.txt")[: len(gt)]}
                for ref, g in refs.items():
                    per[ref]["net"].append(h.score(est, g)[0])
                    per[ref]["gyro"].append(h.score(h.hybrid(est, gR, gap), g)[0])
                    per[ref]["hand"].append(h.score(h.hybrid(res["hand-chosen (net dir + accel len)"], gR, gap), g)[0])
                    per[ref]["gate"].append(h.score(h.hybrid(res[NAMES[variant]], gR, gap), g)[0])
            for ref in per:
                for k in per[ref]:
                    acc[ref][k].append(np.mean(per[ref][k]))
        for ref in acc:
            f = lambda v: f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}" if len(v) > 1 else f"{v[0]:.2f}"  # noqa: E731
            print(f"| {recipe} | {ref} | {f(acc[ref]['net'])} | {f(acc[ref]['gyro'])} | {f(acc[ref]['hand'])} | **{f(acc[ref]['gate'])}** |", flush=True)


def euroc(variant, sigma_net, root, cache, dumps, runs):
    import gyro_hybrid_euroc as gh
    import imu_scale

    print(f"\n### EuRoC test, variant {variant}")
    print("| run | network | gyro + net t | hand-chosen (network-free) | **gate v3** | seqs |\n|---|---|---|---|---|---|")
    seqs = ("MH_05_difficult", "V2_02_medium", "V1_01_easy", "V1_02_medium")
    for run in runs:
        rows = []
        for seq in seqs:
            dump = dumps / run
            if not (dump / f"{seq}_estimated_poses.npy").is_file():
                continue
            ts = dumps / "pivift_dir_best"
            t_end = np.load(ts / f"{seq}_timestamps.npy").astype(np.int64)
            gx = np.load(ts / f"{seq}_gt_xyz.npy").reshape(-1, 3)
            est = np.load(dump / f"{seq}_estimated_poses.npy").astype(np.float64)
            z = np.load(cache / f"{seq}_geo.npz")
            R, d_geo, inf = z["R"], z["d"], json.loads(str(z["info"]))
            n = len(t_end)
            sig, fl = G.sigmas(inf, n), G.floor_v2(inf, n, V.A_V2)
            te = est[:, 3:]
            L_net = np.linalg.norm(te, axis=1)
            d_net = te / np.maximum(L_net, 1e-9)[:, None]
            d_gate, use = G.direction_gate(d_net, d_geo, sig, fl)

            def lengths(dirs):
                rms = np.full(n, np.nan)
                L, _ = imu_scale.imu_lengths(root, seq, t_end, dirs, g_tol=G.G_TOL, rms_out=rms)
                return L, rms

            if variant in ("C5a", "C5b"):
                d_v, _ = V.fuse(d_net, d_geo, sig, fl, sigma_net)
            elif variant == "C4":
                imu = np.loadtxt(root / seq / "mav0/imu0/data.csv", delimiter=",", comments="#")
                t_ns, w, a = imu[:, 0].astype(np.int64), imu[:, 1:4], imu[:, 4:7]
                st = np.zeros(n, bool)
                for k, t1 in enumerate(t_end):
                    m = (t_ns >= t1 - 100_000_000) & (t_ns <= t1)
                    st[k] = m.sum() >= 3 and np.linalg.norm(w[m], axis=1).max() < 0.02 and np.linalg.norm(a[m], axis=1).std() < 0.05
                d_v = np.where((use & ~st)[:, None], d_gate, d_net)
            else:
                d_v = d_gate
            L_v, r_v = lengths(d_v)
            if variant in ("C2", "C5b"):
                L_n, r_n = lengths(d_net)
                L_v = np.where(r_n < r_v, L_n, L_v)
            t_v = d_v * np.where(np.isnan(L_v), L_net, L_v)[:, None]
            # hand-chosen network-free reference (P8-H, |g| check)
            okg = ~np.isnan(d_geo[:, 0])
            df, last = [], np.array([0.0, 0.0, 1.0])
            for k in range(n):
                last = d_geo[k] if okg[k] else last
                df.append(last)
            df = np.array(df)
            lf, _ = imu_scale.imu_lengths(root, seq, t_end, df, g_tol=G.G_TOL)
            Lf, ll = [], 0.0
            for x in lf:
                ll = x if not np.isnan(x) else ll
                Lf.append(ll)
            Re = [gh.euler_to_rotation_matrix(p[:3]) for p in est]
            rows.append([gh.ate(gh.compose(Re, te), gx)[0], gh.ate(gh.compose(list(R), te), gx)[0],
                         gh.ate(gh.compose(list(R), df * np.array(Lf)[:, None]), gx)[0], gh.ate(gh.compose(list(R), t_v), gx)[0]])
        if rows:
            m = np.mean(rows, 0)
            print(f"| {run} | {m[0]:.2f} | {m[1]:.2f} | {m[2]:.2f} | **{m[3]:.2f}** | {len(rows)} |", flush=True)


if __name__ == "__main__":
    a = sys.argv
    variant, sigma_net = a[1], float(a[2])
    kitti(variant, sigma_net, a[3], *(Path(x) for x in a[4:9]))
    euroc(variant, sigma_net, Path(a[9]), Path(a[10]), Path(a[11]), a[12:])
