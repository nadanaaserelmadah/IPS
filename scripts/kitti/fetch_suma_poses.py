"""Range-read only sequences/<s>/poses.txt from the SemanticKITTI label archive (SuMa lidar-SLAM poses).

SemanticKITTI ships, with its labels, per-sequence poses estimated by SuMa from the Velodyne scans
(Behley et al., ICCV 2019). They do not come from the OXTS GPS/INS unit, so they are an independent
reference for the KITTI odometry ground truth that shares the OXTS sensor with our gyro / accelerometer
estimates. The poses.txt files are already in the KITTI odometry cam0 frame (checked 2026-10-28: per-frame
difference to the KITTI GT median 1.2-1.5 cm and 0.03-0.05 deg on 05 / 07 / 10, path lengths within 0.3%).
`to_camera` is kept only for velodyne-frame pose files.
  python scripts/kitti/fetch_suma_poses.py OUT_DIR [SEQS]
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_oxts import RangeFile  # noqa: E402

URL = "https://www.semantic-kitti.org/assets/data_odometry_labels.zip"


def main():
    out = Path(sys.argv[1])
    seqs = sys.argv[2].split(",") if len(sys.argv) > 2 else [f"{i:02d}" for i in range(11)]
    out.mkdir(parents=True, exist_ok=True)
    rf = RangeFile(URL)
    with zipfile.ZipFile(rf) as z:
        names = {n for n in z.namelist() if n.endswith("poses.txt")}
        for s in seqs:
            hit = [n for n in names if f"sequences/{s}/poses.txt" in n]
            if not hit:
                print(s, "no poses.txt in archive")
                continue
            (out / f"{s}.txt").write_bytes(z.read(hit[0]))
            print(s, hit[0], (out / f"{s}.txt").stat().st_size, "bytes")
    print(f"{rf.nreq} requests, {rf.nbytes / 1e6:.1f} MB read of {rf.size / 1e6:.0f} MB")


def to_camera(poses_velo, Tr):
    """KITTI-format velodyne-frame poses (N x 12) -> cam0-frame 4x4 poses."""
    import numpy as np

    P = np.tile(np.eye(4), (len(poses_velo), 1, 1))
    P[:, :3, :] = poses_velo.reshape(-1, 3, 4)
    Ti = np.linalg.inv(Tr)
    return Tr @ P @ Ti


if __name__ == "__main__":
    main()
