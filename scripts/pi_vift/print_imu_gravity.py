"""Print EuRoC/TUM IMU column means so gravity axis / channel order can be checked.

  python scripts/pi_vift/print_imu_gravity.py --seq-dir /path/to/MH_01_easy
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "VIFT"))

import numpy as np

from src.utils.euroc_utils import load_imu, resolve_sequence_root
from src.utils.imu_canonical import canonicalize_imu_table, imu_canonical_enabled


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-dir", type=Path, required=True)
    args = p.parse_args()
    root = resolve_sequence_root(args.seq_dir)
    raw = load_imu(root)
    acc_raw = raw[:, 4:7].mean(axis=0)
    gyr_raw = raw[:, 1:4].mean(axis=0)
    print(f"legacy CSV [gx,gy,gz,ax,ay,az]  n={len(raw)}")
    print(f"  mean gyro  {gyr_raw}")
    print(f"  mean accel {acc_raw}  |a|={np.linalg.norm(acc_raw):.3f}")
    can = canonicalize_imu_table(raw, args.seq_dir, enabled=True)
    print(f"canonical [ax,ay,az,wx,wy,wz]  enabled_default={imu_canonical_enabled()}")
    print(f"  mean accel {can[:, 1:4].mean(axis=0)}  |a|={np.linalg.norm(can[:, 1:4].mean(axis=0)):.3f}")
    print(f"  mean gyro  {can[:, 4:7].mean(axis=0)}")


if __name__ == "__main__":
    main()
