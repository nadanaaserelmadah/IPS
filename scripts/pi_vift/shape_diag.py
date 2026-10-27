"""Why does the estimated trajectory shape carry no information on V2_02 / V1_xx?

For one pose-dump dir (estimated_poses, gt_poses = per-edge 6-DoF [rx ry rz tx ty tz],
gt_xyz world positions), report per-edge errors and SE3 / Sim3 ATE of hybrid
trajectories that swap one component of the estimate for GT:
  est             : estimate as tested
  R_gt + t_est    : GT rotations, estimated translations   -> cost of translation error
  R_est + t_gt    : estimated rotations, GT translations   -> cost of rotation drift
  R_gt + dir_est·|t_gt| / R_gt + dir_gt·|t_est| : direction vs magnitude
  R_gt + t_mean   : GT rotations, every step = mean GT translation (body frame)
  zero motion     : the floor
  python scripts/pi_vift/shape_diag.py DUMP_DIR [SEQ ...]   (dir with <seq>_estimated_poses.npy, <seq>_gt_poses.npy[, <seq>_gt_xyz.npy])
"""
import sys
from pathlib import Path

import numpy as np

VIFT = Path(__file__).resolve().parents[2] / "third_party/VIFT"
sys.path.insert(0, str(VIFT))
from src.metrics.euroc_metrics_calculator import ate_rmse_m, sim3_scale, align_trajectory_lengths  # noqa: E402
from src.utils.euroc_utils import euler_to_rotation_matrix  # noqa: E402


def compose(R_list, t_list):
    T = np.eye(4)
    xyz = [T[:3, 3].copy()]
    for R, t in zip(R_list, t_list):
        S = np.eye(4)
        S[:3, :3], S[:3, 3] = R, t
        T = T @ S
        xyz.append(T[:3, 3].copy())
    return np.array(xyz)


def rot_angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def rotvec(R):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (0.5 if a < 1e-8 else a / (2 * np.sin(a)))


def ate(xyz, gt):
    e, g = align_trajectory_lengths(xyz, gt)
    return ate_rmse_m(e, g, alignment="se3"), ate_rmse_m(e, g, alignment="sim3"), sim3_scale(e, g)


def run(d: Path, seq: str):
    est = np.load(d / f"{seq}_estimated_poses.npy").astype(np.float64)
    gtp = np.load(d / f"{seq}_gt_poses.npy").astype(np.float64)
    Re = [euler_to_rotation_matrix(p[:3]) for p in est]
    Rg = [euler_to_rotation_matrix(p[:3]) for p in gtp]
    gx = d / f"{seq}_gt_xyz.npy"  # latent tester dumps no gt_xyz: integrated GT (= world up to SE3)
    gxyz = np.load(gx).reshape(-1, 3).astype(np.float64) if gx.is_file() else compose(Rg, gtp[:, 3:])
    te, tg = est[:, 3:], gtp[:, 3:]
    rot_err = np.array([rot_angle(a.T @ b) for a, b in zip(Re, Rg)])
    rot_gt = np.array([rot_angle(b) for b in Rg])
    ne, ng = np.linalg.norm(te, axis=1), np.linalg.norm(tg, axis=1)
    cosd = np.sum(te * tg, 1) / np.maximum(ne * ng, 1e-9)
    dir_err = np.degrees(np.arccos(np.clip(cosd, -1, 1)))
    moving = ng > 0.02
    # magnitude correlation at lags (detects an off-by-k edge alignment)
    lags = {k: np.corrcoef(ne[max(0, k):len(ne) + min(0, k)], ng[max(0, -k):len(ng) - max(0, k)])[0, 1] for k in range(-2, 3)}
    ue, ug = te / np.maximum(ne, 1e-9)[:, None], tg / np.maximum(ng, 1e-9)[:, None]
    zero = np.zeros_like(tg)
    hyb = {
        "est": (Re, te),
        "R_gt + t_est": (Rg, te),
        "R_est + t_gt": (Re, tg),
        "R_gt + dir_est*|t_gt|": (Rg, ue * ng[:, None]),
        "R_gt + dir_gt*|t_est|": (Rg, ug * ne[:, None]),
        "R_gt + t_mean(gt)": (Rg, np.repeat(tg.mean(0, keepdims=True), len(tg), 0)),
        "gt (sanity)": (Rg, tg),
        "zero motion": (Rg, zero),
    }
    print(f"\n== {d.parent.name}/{d.name} {seq}: N={len(est)}  GT step {ng.mean()*100:.1f} cm, GT rot {rot_gt.mean():.2f} deg/step")
    print(f"   per-edge: rot err median {np.median(rot_err):.2f} / mean {rot_err.mean():.2f} deg | "
          f"dir err (moving) median {np.median(dir_err[moving]):.1f} deg | |t| est/gt {ne.sum()/ng.sum():.2f} | "
          f"corr(|t|) {np.corrcoef(ne, ng)[0,1]:.2f} | lag corr " + " ".join(f"{k:+d}:{v:.2f}" for k, v in lags.items()))
    # per-axis translation correlation (camera frame)
    print("   t corr per axis x/y/z: " + " ".join(f"{np.corrcoef(te[:, i], tg[:, i])[0,1]:.2f}" for i in range(3))
          + " | gt t std x/y/z cm: " + " ".join(f"{tg[:, i].std()*100:.1f}" for i in range(3))
          + " | est t std: " + " ".join(f"{te[:, i].std()*100:.1f}" for i in range(3)))
    for name, (R, t) in hyb.items():
        se3, sim3, sc = ate(compose(R, t), gxyz)
        print(f"   {name:24s} SE3 {se3:6.2f}  Sim3 {sim3:5.2f}  scale {sc:5.2f}")
    # accumulated heading drift of the estimate
    Ae, Ag = np.eye(3), np.eye(3)
    drift = []
    for a, b in zip(Re, Rg):
        Ae, Ag = Ae @ a, Ag @ b
        drift.append(rot_angle(Ae.T @ Ag))
    drift = np.array(drift)
    q = [len(drift) // 4, len(drift) // 2, len(drift) - 1]
    w = np.array([rotvec(a.T @ b) for a, b in zip(Re, Rg)])  # per-edge R_est^-1 R_gt, camera frame
    print("   per-edge rot err mean vector (bias) deg x/y/z: " + " ".join(f"{np.degrees(w[:, i].mean()):+.3f}" for i in range(3))
          + f" | |bias| x N = {np.degrees(np.linalg.norm(w.mean(0)))*len(w):.0f} deg")
    print("   accumulated orientation error at 25/50/100%: " + " / ".join(f"{drift[i]:.1f}" for i in q) + " deg")


if __name__ == "__main__":
    d = Path(sys.argv[1])
    seqs = sys.argv[2:] or sorted({p.name.split("_estimated")[0] for p in d.glob("*_estimated_poses.npy")})
    for s in seqs:
        run(d, s)
