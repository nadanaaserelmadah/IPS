"""EuRoC: network translation + gyro rotation (the KITTI hybrid), scored against the zero-motion floor.

Each dumped edge k is the last edge of a test window: camera frames t_end - 100 ms -> t_end (cam_stride 2
at 20 Hz), in the camera frame (T_WC = T_WB @ T_BS). The raw 200 Hz gyro (body = IMU frame) is integrated
over that interval with the trapezoidal rule on the true sample times and rotated into the camera frame
with the dataset's T_BS. Nothing is fitted on test data: EuRoC's camera and IMU are hardware-synchronised
(lag 0) and T_BS comes from cam0/sensor.yaml. Gyro bias, three ways:
  raw     : no bias removed
  static  : mean gyro over the static start of the recording (IMU only, before take-off)
  gt_bias : the ground-truth estimator's gyro bias (oracle upper bound, not a method)
  python scripts/euroc/gyro_hybrid_euroc.py EUROC_ROOT DUMP_DIR [TS_DIR] [SEQ ...]
TS_DIR holds <seq>_timestamps.npy / <seq>_gt_xyz.npy when DUMP_DIR lacks them (VIFT-style csv dumps);
the GT edges must match (checked).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/VIFT"))
sys.path.insert(0, str(ROOT / "scripts/pi_vift"))
from src.utils.euroc_utils import euler_from_matrix, euler_to_rotation_matrix  # noqa: E402
from src.utils.imu_canonical import load_T_BS  # noqa: E402
from shape_diag import ate, compose, rot_angle  # noqa: E402

EDGE_NS = 100_000_000


def so3_exp(w):
    a = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if a < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(a) / a * K + (1 - np.cos(a)) / a**2 * K @ K


def integrate(t_ns, gyro, t0, t1):
    """Body-frame rotation from t0 to t1: trapezoid on the raw samples, linear interpolation at the ends."""
    i0, i1 = np.searchsorted(t_ns, t0), np.searchsorted(t_ns, t1)
    ts = np.concatenate([[t0], t_ns[i0:i1], [t1]]).astype(np.float64)
    ws = np.stack([np.interp(ts, t_ns, gyro[:, j]) for j in range(3)], 1)
    R = np.eye(3)
    for k in range(len(ts) - 1):
        R = R @ so3_exp(0.5 * (ws[k] + ws[k + 1]) * (ts[k + 1] - ts[k]) * 1e-9)
    return R


def static_bias(t_ns, gyro, acc, block_s=0.25, min_s=0.5):
    """Mean gyro over the longest static prefix (low gyro and accel-norm spread per block)."""
    n = int(round(block_s * 200))
    end = 0
    for s in range(0, len(t_ns) - n, n):
        g, a = gyro[s:s + n], np.linalg.norm(acc[s:s + n], axis=1)
        if np.linalg.norm(g - g.mean(0), axis=1).max() > 0.1 or a.std() > 0.3:  # rotors may idle before take-off
            break
        end = s + n
    if (t_ns[max(end - 1, 0)] - t_ns[0]) * 1e-9 < min_s:
        return None, 0.0
    return gyro[:end].mean(0), (t_ns[end - 1] - t_ns[0]) * 1e-9


def run(root: Path, dump: Path, ts_dir: Path, seq: str):
    mav = root / seq / "mav0"
    imu = np.loadtxt(mav / "imu0/data.csv", delimiter=",", comments="#")
    t_ns, gyro, acc = imu[:, 0].astype(np.int64), imu[:, 1:4], imu[:, 4:7]
    gt = np.loadtxt(mav / "state_groundtruth_estimate0/data.csv", delimiter=",", comments="#")
    R_bs = load_T_BS(root / seq)[:3, :3]

    est = np.load(dump / f"{seq}_estimated_poses.npy").astype(np.float64)
    gtp = np.load(dump / f"{seq}_gt_poses.npy").astype(np.float64)
    ref = np.load(ts_dir / f"{seq}_gt_poses.npy").astype(np.float64)
    assert np.allclose(ref, gtp, atol=1e-6), "GT edges differ from the timestamp source"
    t_end = np.load(ts_dir / f"{seq}_timestamps.npy").astype(np.int64)
    gxyz = np.load(ts_dir / f"{seq}_gt_xyz.npy").reshape(-1, 3).astype(np.float64)

    b_static, static_s = static_bias(t_ns, gyro, acc)
    b_gt = np.stack([np.interp(t_end - EDGE_NS // 2, gt[:, 0], gt[:, 11 + j]) for j in range(3)], 1)
    biases = {"raw": np.zeros(3), "static": b_static, "gt_bias": b_gt}

    Rg = [euler_to_rotation_matrix(p[:3]) for p in gtp]
    Re = [euler_to_rotation_matrix(p[:3]) for p in est]
    te = est[:, 3:]
    rows = {"network": ate(compose(Re, te), gxyz), "R_gt + t_net (oracle)": ate(compose(Rg, te), gxyz)}
    per_edge = {}
    for name, b in biases.items():
        if b is None:
            continue
        Rc = []
        for k, t1 in enumerate(t_end):
            bk = b[k] if b.ndim == 2 else b
            Rb = integrate(t_ns, gyro - bk, t1 - EDGE_NS, t1)
            Rc.append(euler_to_rotation_matrix(euler_from_matrix(R_bs.T @ Rb @ R_bs)))  # same Euler round trip as GT
        per_edge[name] = np.median([rot_angle(a.T @ g) for a, g in zip(Rc, Rg)])
        rows[f"R_gyro[{name}] + t_net"] = ate(compose(Rc, te), gxyz)
    floor = ate(np.zeros_like(gxyz), gxyz)[0]
    net_err = np.median([rot_angle(a.T @ g) for a, g in zip(Re, Rg)])
    return rows, floor, per_edge, net_err, static_s, b_static


def main():
    root, dump = Path(sys.argv[1]), Path(sys.argv[2])
    rest = sys.argv[3:]
    ts_dir = Path(rest.pop(0)) if rest and Path(rest[0]).is_dir() else dump
    seqs = rest or ["MH_05_difficult", "V2_02_medium"]
    for seq in seqs:
        rows, floor, per_edge, net_err, static_s, b = run(root, dump, ts_dir, seq)
        print(f"\n## {seq}  (zero-motion floor SE3 {floor:.2f} m; static start {static_s:.1f} s"
              + (f", bias {np.round(b, 4).tolist()} rad/s" if b is not None else "") + ")")
        print(f"median rotation error per edge: network {net_err:.3f}°, "
              + ", ".join(f"gyro[{k}] {v:.3f}°" for k, v in per_edge.items()))
        print("| rotation + translation | SE3 ATE m | Sim3 ATE m | sim3 scale |\n|---|---|---|---|")
        for k, (se3, sim3, sc) in rows.items():
            print(f"| {k} | {se3:.2f} | {sim3:.2f} | {sc:.2f} |")


if __name__ == "__main__":
    main()
