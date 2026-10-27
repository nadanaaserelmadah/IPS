#!/usr/bin/env python3
"""
One-shot Kaggle setup for minimal KITTI eval.

Finds IPS/VIFT under /kaggle/working/IPS or /kaggle/working/IPS/IPS,
symlinks KITTI from /kaggle/input/*, downloads IMU .mat files.

Usage (from anywhere on Kaggle):
  python /kaggle/working/IPS/IPS/scripts/phase0/kaggle_kitti_setup.py
  python /kaggle/working/IPS/IPS/scripts/phase0/kaggle_kitti_setup.py --kitti-slug my-kitti-dataset
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

IMU_BASE = "https://raw.githubusercontent.com/mingyuyng/Visual-Selective-VIO/main/data/imus"
MINIMAL_SEQS = ["05", "07", "10"]


def find_ips_roots() -> tuple[Path, Path]:
    """Return (ips_root, vift_root)."""
    candidates = [
        Path("/kaggle/working/IPS/IPS"),
        Path("/kaggle/working/IPS"),
        Path.cwd(),
    ]
    if os.environ.get("IPS_ROOT"):
        candidates.insert(0, Path(os.environ["IPS_ROOT"]))
    for ips in candidates:
        vift = ips / "third_party" / "VIFT"
        if (vift / "src" / "train.py").is_file():
            return ips.resolve(), vift.resolve()
    raise FileNotFoundError(
        "Could not find third_party/VIFT. Set IPS_ROOT or copy IPS to /kaggle/working/IPS/IPS/"
    )


def has_kitti_layout(root: Path) -> bool:
    poses = root / "poses" / "05.txt"
    imgs = root / "sequences" / "05" / "image_2"
    return poses.is_file() and imgs.is_dir()


def find_kitti_source(kitti_slug: str | None) -> Path | None:
    input_dir = Path("/kaggle/input")
    if not input_dir.is_dir():
        return None

    def try_candidates(base: Path) -> Path | None:
        for candidate in [base / "kitti_data", base, base / "dataset"]:
            if candidate.is_dir() and has_kitti_layout(candidate):
                return candidate
        return None

    if kitti_slug:
        for base in [input_dir / kitti_slug, input_dir / "datasets" / kitti_slug]:
            found = try_candidates(base)
            if found:
                return found
        # e.g. datasets/hocop1/kitti-odometry
        for base in input_dir.glob(f"datasets/*/{kitti_slug}"):
            found = try_candidates(base)
            if found:
                return found

    # Auto-discover: any poses/05.txt under /kaggle/input
    for poses_file in input_dir.rglob("poses/05.txt"):
        root = poses_file.parent.parent
        if has_kitti_layout(root):
            return root

    for slug in sorted(os.listdir(input_dir)):
        found = try_candidates(input_dir / slug)
        if found:
            return found
    return None


def setup_writable_kitti(vift: Path, source: Path) -> Path:
    """Writable kitti_data: symlink read-only sequences/poses, real imus/ dir."""
    import shutil

    target = vift / "data" / "kitti_data"
    source = source.resolve()

    if target.is_symlink():
        target.unlink()
    elif target.is_dir():
        # Reuse if already our composite layout with imus
        if (target / "imus").is_dir() and not (target / "imus").is_symlink():
            if has_kitti_layout(target):
                print(f"KITTI composite already at {target}")
                return target
        shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=True)
    seq_link = target / "sequences"
    pose_link = target / "poses"
    if seq_link.exists():
        seq_link.unlink() if seq_link.is_symlink() else shutil.rmtree(seq_link)
    if pose_link.exists():
        pose_link.unlink() if pose_link.is_symlink() else shutil.rmtree(pose_link)

    os.symlink(source / "sequences", seq_link, target_is_directory=True)
    os.symlink(source / "poses", pose_link, target_is_directory=True)
    (target / "imus").mkdir(exist_ok=True)
    print(f"KITTI composite at {target}")
    print(f"  sequences -> {source / 'sequences'}")
    print(f"  poses     -> {source / 'poses'}")
    print(f"  imus/     (writable)")
    return target


def download_imus(kitti_data: Path, seqs: list[str]) -> None:
    imus_dir = kitti_data / "imus"
    imus_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading IMU to {imus_dir}")
    for seq in seqs:
        dest = imus_dir / f"{seq}.mat"
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"  skip {seq}.mat")
            continue
        url = f"{IMU_BASE}/{seq}.mat"
        print(f"  {url}")
        urllib.request.urlretrieve(url, dest)


def ensure_pretrained(vift: Path, use_gdown: bool = False) -> bool:
    script = vift.parent.parent / "scripts" / "phase0" / "download_pretrained_encoder.py"
    if script.is_file():
        import subprocess

        cmd = [sys.executable, str(script), "--vift-root", str(vift)]
        if use_gdown:
            cmd.append("--gdown")
        return subprocess.call(cmd) == 0
    dest = vift / "pretrained_models" / "vf_512_if_256_3e-05.model"
    return dest.is_file()


def check(kitti_data: Path, seqs: list[str], vift: Path) -> bool:
    ok = True
    pretrained = vift / "pretrained_models" / "vf_512_if_256_3e-05.model"
    if not pretrained.is_file():
        print(f"MISSING: {pretrained}")
        print("  Attach Kaggle dataset with vf_512_if_256_3e-05.model, or:")
        print("  pip install gdown && python .../download_pretrained_encoder.py --gdown")
        ok = False
    else:
        print(f"OK: {pretrained.name} ({pretrained.stat().st_size // 1_000_000} MB)")

    for seq in seqs:
        p = kitti_data / "poses" / f"{seq}.txt"
        i = kitti_data / "imus" / f"{seq}.mat"
        g = list((kitti_data / "sequences" / seq / "image_2").glob("*.png"))[:1]
        if not p.is_file():
            print(f"MISSING pose: {p}")
            ok = False
        elif not g:
            print(f"MISSING images: .../sequences/{seq}/image_2/")
            ok = False
        elif not i.is_file():
            print(f"MISSING imu: {i}")
            ok = False
        else:
            print(f"OK seq {seq}: pose + images + imu")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitti-slug", default=None, help="Folder name under /kaggle/input")
    parser.add_argument("--seq", default="05,07,10")
    parser.add_argument("--skip-imu", action="store_true")
    parser.add_argument(
        "--download-pretrained",
        action="store_true",
        help="Try Kaggle input copy or gdown for encoder weights",
    )
    args = parser.parse_args()
    seqs = [s.strip() for s in args.seq.split(",") if s.strip()]

    ips, vift = find_ips_roots()
    os.environ["PROJECT_ROOT"] = str(vift)
    print(f"IPS_ROOT={ips}")
    print(f"VIFT_ROOT={vift}")
    print(f"Scripts: {ips / 'scripts' / 'phase0'}")

    source = find_kitti_source(args.kitti_slug)
    if source is None:
        print("\nERROR: No KITTI layout found under /kaggle/input/")
        print("Attach a KITTI Odometry dataset, then re-run with --kitti-slug <name>")
        print("\nContents of /kaggle/input:")
        inp = Path("/kaggle/input")
        if inp.is_dir():
            for name in sorted(os.listdir(inp))[:20]:
                p = inp / name
                print(f"  {name}/ -> {list(p.iterdir())[:6]}...")
        return 1

    kitti_data = setup_writable_kitti(vift, source)
    if not args.skip_imu:
        download_imus(kitti_data, seqs)

    if args.download_pretrained:
        ensure_pretrained(vift, use_gdown=True)

    ok = check(kitti_data, seqs, vift)
    if ok:
        print("\nReady. Next:")
        print(f"  %cd {vift / 'data'}")
        print("  !python latent_val_caching_minimal.py")
        print(f"  %cd {vift}")
        print("  !python src/train.py experiment=kitti_eval_minimal trainer=gpu trainer.max_epochs=5")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
