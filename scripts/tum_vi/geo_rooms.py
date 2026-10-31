"""Shadow tree + window timestamps + two-view geometry for the given TUM-VI rooms (design rooms for gate v4)."""
import sys
from pathlib import Path

import numpy as np

IPS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(IPS / "scripts/euroc"))
sys.path.insert(0, str(IPS / "third_party/VIFT"))
import geo_direction as G  # noqa: E402
from src.utils.euroc_utils import build_sequence_samples  # noqa: E402

raw, shadow, ts_dir, geo_dir = (Path(a) for a in sys.argv[1:5])
for r in sys.argv[5:]:
    src, dst = raw / r, shadow / r
    (dst / "mav0").mkdir(parents=True, exist_ok=True)
    for name, target in (("dso", src / "dso"),):
        if not (dst / name).exists():
            (dst / name).symlink_to(target)
    for s in ("cam0", "cam1", "imu0", "mocap0"):
        if not (dst / "mav0" / s).exists():
            (dst / "mav0" / s).symlink_to(src / "mav0" / s)
    if not (dst / "mav0/state_groundtruth_estimate0").exists():
        (dst / "mav0/state_groundtruth_estimate0").symlink_to(src / "mav0/mocap0")
    samples, _ = build_sequence_samples(dst)
    te = np.array([s["t_end"] for s in samples], np.int64)
    gt = np.array([np.asarray(s["gts"])[-1] for s in samples])
    np.save(ts_dir / f"{r}_timestamps.npy", te)
    np.save(ts_dir / f"{r}_gt_poses.npy", gt)
    np.save(ts_dir / f"{r}_gt_xyz.npy", np.array([s["gt_xyz_end"] for s in samples]))
    R, d, _, inf = G.geo_sequence(shadow, r, te, geo_dir)
    Lg = np.linalg.norm(gt[:, 3:], axis=1)
    ok = ~np.isnan(d[:, 0]) & (Lg > 0.01)
    e = np.degrees(np.arccos(np.clip(np.sum(d * gt[:, 3:], 1) / np.maximum(Lg, 1e-12), -1, 1)))
    print(f"{r}: edges {len(te)}, geo {100 * np.mean(~np.isnan(d[:, 0])):.1f}%, dir err median {np.median(e[ok]):.2f} deg", flush=True)
