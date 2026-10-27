"""Create minimal synthetic EuRoC sequence for pipeline testing (no network)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

VIFT_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "VIFT"
SEQ = VIFT_ROOT / "data" / "euroc" / "MH_01_easy"


def main() -> None:
    cam_data = SEQ / "cam0" / "data"
    cam_data.mkdir(parents=True, exist_ok=True)
    (SEQ / "imu0").mkdir(parents=True, exist_ok=True)
    (SEQ / "state_groundtruth_estimate_0").mkdir(parents=True, exist_ok=True)

    n_frames = 40
    t0 = 1_400_000_000_000_000
    dt_cam = int(5e7)  # 20 Hz
    dt_imu = int(5e6)  # 200 Hz

    # Spiral trajectory GT
    with open(SEQ / "state_groundtruth_estimate_0" / "data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "#timestamp",
                "p_x",
                "p_y",
                "p_z",
                "q_w",
                "q_x",
                "q_y",
                "q_z",
            ]
        )
        for i in range(n_frames * 10):
            t = t0 + i * dt_imu
            ang = i * 0.01
            px, py, pz = np.cos(ang), np.sin(ang), 0.1 * i * 0.001
            w.writerow([int(t), px, py, pz, 1.0, 0.0, 0.0, 0.0])

    with open(SEQ / "cam0" / "data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#timestamp", "filename"])
        for i in range(n_frames):
            name = f"{i:05d}.png"
            t = t0 + i * dt_cam
            w.writerow([int(t), name])
            Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255)).save(cam_data / name)

    with open(SEQ / "imu0" / "data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "#timestamp",
                "w_x",
                "w_y",
                "w_z",
                "a_x",
                "a_y",
                "a_z",
            ]
        )
        for i in range(n_frames * 10):
            t = t0 + i * dt_imu
            w.writerow([int(t), 0.01, 0.0, 0.02, 0.0, 0.0, 9.81])

    print(f"Created synthetic EuRoC at {SEQ}")


if __name__ == "__main__":
    main()
