"""KITTI hybrid: gyro rotation (integrated over the real ~104 ms image interval) + a network's translation.

calibrate: fit the IMU-to-pose time offset and the IMU->camera rotation on training sequences
           (default 04, 06, 08, 09) from raw OXTS packed by oxts_pack.py; cache per-frame gyro rotations
           for all packed sequences.
apply:     score each dump directory (<seq>_estimated_poses.npy / <seq>_gt_poses.npy) as the network
           predicted it, and with its rotation replaced by the gyro's.
features:  calibrate each recording day on its own training sequences (2011_10_03: 00 01 02;
           2011_09_30: 04 06 08 09) and write per-frame gyro rotations and rotation vectors (deg,
           camera frame) for all 11 sequences, the inputs of the gyro-conditioned model.
  python scripts/kitti/gyro_hybrid.py calibrate NPZ_DIR POSES_DIR OUT_DIR
  python scripts/kitti/gyro_hybrid.py features NPZ_DIR POSES_DIR OUT_DIR
  python scripts/kitti/gyro_hybrid.py ensemble OUT_DIR DUMP_DIR DUMP_DIR [...]   (mean translation, gyro rotation)
  python scripts/kitti/gyro_hybrid.py apply OUT_DIR DUMP_DIR [DUMP_DIR ...]
"""
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gyro_rotation_test as g  # noqa: E402

TRAIN, ALL = ["04", "06", "08", "09"], ["04", "05", "06", "07", "08", "09", "10"]
DAYS = {"2011_10_03": (["00", "01", "02"], ["00", "01", "02"]), "2011_09_30": (TRAIN, ALL)}


def max_gaps(npz: Path, lag: float) -> np.ndarray:
    """Largest spacing (s) between raw OXTS samples bracketing each image interval (gaps up to 1.9 s exist)."""
    z = np.load(npz)
    t_ox, t_img = z["t_ox"], z["t_img"]
    out = []
    for a, b in zip(t_img[:-1] + lag, t_img[1:] + lag):
        i0, i1 = max(np.searchsorted(t_ox, a) - 1, 0), min(np.searchsorted(t_ox, b), len(t_ox) - 1)
        out.append(float(np.diff(t_ox[i0:i1 + 1]).max()) if i1 > i0 else 0.0)
    return np.array(out)


def integrate(npz: Path, lag: float) -> list:
    z = np.load(npz)
    t_ox, w, t_img = z["t_ox"], z["ox"][:, 17:20], z["t_img"]
    out = []
    for i in range(len(t_img) - 1):
        a, b = t_img[i] + lag, t_img[i + 1] + lag
        tk = np.r_[a, t_ox[(t_ox > a) & (t_ox < b)], b]
        wk = np.stack([np.interp(tk, t_ox, w[:, c]) for c in range(3)], 1)
        R = np.eye(3)
        for k in range(len(tk) - 1):
            R = R @ g.so3_exp(0.5 * (wk[k] + wk[k + 1]) * (tk[k + 1] - tk[k]))
        out.append(R)
    return out


def calibrate(npz_dir: Path, poses_dir: Path, out: Path, train=TRAIN, seqs=ALL, tag=""):
    P = {s: g.read_pose_from_text(str(poses_dir / f"{s}.txt"))[0] for s in train}

    def fit(lag):
        ai, ac = [], []
        for s in train:
            Ri, Rc = integrate(npz_dir / f"{s}.npz", lag), g.gt_rel_R(P[s])
            n = min(len(Ri), len(Rc))
            ai += [g.so3_log(r) for r in Ri[:n]]
            ac += [g.so3_log(r) for r in Rc[:n]]
        ai, ac = np.array(ai), np.array(ac)
        R = g.kabsch(ai, ac)
        return float(np.median(np.linalg.norm((R @ ai.T).T - ac, axis=1))), R

    coarse = {lag: fit(lag)[0] for lag in np.arange(-0.08, 0.081, 0.01)}
    l0 = min(coarse, key=coarse.get)
    fine = {lag: fit(lag)[0] for lag in np.arange(l0 - 0.01, l0 + 0.0101, 0.0025)}
    lag = float(min(fine, key=fine.get))
    _, R = fit(lag)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"calib{tag}.json").write_text(json.dumps({"lag_s": lag, "R_cam_imu": R.tolist(), "train": train,
                                                      "train_median_err_deg": float(np.degrees(fine[lag]))}, indent=2))
    for s in seqs:
        Rs = np.array([R @ r @ R.T for r in integrate(npz_dir / f"{s}.npz", lag)])
        np.save(out / f"{s}_gyro_R.npy", Rs)
        np.save(out / f"{s}_gyro_rotvec_deg.npy", np.degrees(np.array([g.so3_log(r) for r in Rs])).astype(np.float32))
        np.save(out / f"{s}_gyro_maxgap_s.npy", max_gaps(npz_dir / f"{s}.npz", lag))
    print(f"{tag or 'calib'}: lag {lag * 1e3:+.1f} ms, train {train} median per-frame error {np.degrees(fine[lag]):.4f} deg -> {out}")


GAP_S = 0.05  # OXTS spacing above which the gyro is not trusted for that frame (nominal 0.01 s)


def hybrid(est: np.ndarray, gyro_R: np.ndarray, maxgap: np.ndarray | None = None) -> np.ndarray:
    """Gyro rotation + network translation; frames inside an OXTS gap keep the network's rotation."""
    n = len(est)
    rot = np.array([g.euler_from_matrix(r) for r in gyro_R[:n]])
    if maxgap is not None:
        bad = maxgap[:n] > GAP_S
        rot[bad] = est[bad, :3]
    return np.hstack([rot, est[:, 3:]])


def load_gap(calib: Path, s: str):
    f = calib / f"{s}_gyro_maxgap_s.npy"
    return np.load(f) if (USE_GAP and f.is_file()) else None


USE_GAP = os.environ.get("GYRO_GAP_FALLBACK", "1") == "1"


def score(e6, gt):
    with contextlib.redirect_stdout(io.StringIO()):
        return g.score(e6, gt)


def apply(calib: Path, dumps: list):
    print("| dump | rotation | 05 | 07 | 10 | mean t_rel | mean r_rel |\n|---|---|---|---|---|---|---|")
    for d in dumps:
        seqs = [s for s in ("05", "07", "10") if (d / f"{s}_estimated_poses.npy").is_file()]
        rows = {"network": [], "gyro": []}
        for s in seqs:
            est = np.load(d / f"{s}_estimated_poses.npy").astype(np.float64)
            gt = np.load(d / f"{s}_gt_poses.npy").astype(np.float64)
            rows["network"].append(score(est, gt))
            rows["gyro"].append(score(hybrid(est, np.load(calib / f"{s}_gyro_R.npy"), load_gap(calib, s)), gt))
        for k, v in rows.items():
            v = np.array(v)
            print(f"| {d} | {k} | " + " | ".join(f"{a:.2f}/{b:.2f}" for a, b in v) + f" | **{v[:, 0].mean():.2f}** | {v[:, 1].mean():.2f} |")


def ensemble(calib: Path, dumps: list):
    """Average the networks' per-frame translations (and rotations, for the network row); rotation from the gyro."""
    rows = {"network (mean)": [], "gyro + mean translation": []}
    for s in ("05", "07", "10"):
        gt = np.load(dumps[0] / f"{s}_gt_poses.npy").astype(np.float64)
        ests = [np.load(d / f"{s}_estimated_poses.npy").astype(np.float64)[: len(gt)] for d in dumps]
        for d in dumps:
            assert np.allclose(np.load(d / f"{s}_gt_poses.npy")[: len(gt)], gt, atol=1e-5), d
        mean = np.mean(ests, 0)
        rows["network (mean)"].append(score(mean, gt))
        rows["gyro + mean translation"].append(score(hybrid(mean, np.load(calib / f"{s}_gyro_R.npy"), load_gap(calib, s)), gt))
    print(f"ensemble of {len(dumps)}: " + ", ".join(str(d) for d in dumps))
    for k, v in rows.items():
        v = np.array(v)
        print(f"| {k} | " + " | ".join(f"{a:.2f}/{b:.2f}" for a, b in v) + f" | **{v[:, 0].mean():.2f}** | {v[:, 1].mean():.2f} |")


if __name__ == "__main__":
    if sys.argv[1] == "calibrate":
        calibrate(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    elif sys.argv[1] == "ensemble":
        ensemble(Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]])
    elif sys.argv[1] == "features":
        for day, (train, seqs) in DAYS.items():
            calibrate(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), train, seqs, f"_{day}")
    else:
        apply(Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]])
