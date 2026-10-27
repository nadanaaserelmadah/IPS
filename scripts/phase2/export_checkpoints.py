#!/usr/bin/env python3
"""
Zip EuRoC Phase 2 checkpoints for download (Kaggle / local).

Kaggle: run after training, then download from the notebook **Output** tab
        (or use the notebook cell in docs/PHASE2.md).

Usage:
  python scripts/phase2/export_checkpoints.py
  python scripts/phase2/export_checkpoints.py --run 2026-05-25_03-14-02
  python scripts/phase2/export_checkpoints.py --output /kaggle/working/ips_phase2_ckpt.zip
  python scripts/phase2/export_checkpoints.py --latest-only
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def ips_root() -> Path:
    return Path(__file__).resolve().parents[2]


def vift_root() -> Path:
    return ips_root() / "third_party" / "VIFT"


def runs_dir(vift: Path) -> Path:
    return vift / "logs" / "train" / "runs"


def list_runs(vift: Path) -> list[Path]:
    rd = runs_dir(vift)
    if not rd.is_dir():
        return []
    return sorted(
        [p for p in rd.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def find_run(vift: Path, run_name: str | None) -> Path | None:
    runs = list_runs(vift)
    if not runs:
        return None
    if run_name:
        for r in runs:
            if r.name == run_name:
                return r
        return None
    return runs[0]


def zip_directory(src: Path, zip_path: Path, arc_prefix: str) -> list[str]:
    """Zip all files under src; return list of archived relative paths."""
    added: list[str] = []
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if not src.is_dir():
            return added
        for f in sorted(src.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(src)
            arc = f"{arc_prefix}/{rel.as_posix()}"
            zf.write(f, arcname=arc)
            added.append(arc)
    return added


def try_ipython_download(zip_path: Path) -> None:
    try:
        from IPython.display import FileLink, display

        display(FileLink(str(zip_path), result_html_prefix="Download: "))
        print("Click the link above, or use Kaggle Output tab →", zip_path)
    except ImportError:
        print("Download manually:", zip_path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description="Export VIFT checkpoints to a zip file")
    parser.add_argument(
        "--run",
        default=None,
        help="Run folder name under logs/train/runs (default: newest)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .zip path (default: IPS/exports/checkpoints/<run>.zip)",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Zip only last.ckpt (+ best if present), smaller download",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Zip every run's checkpoints into one archive",
    )
    parser.add_argument(
        "--copy-to-working",
        action="store_true",
        default=True,
        help="Also copy zip to /kaggle/working/ when on Kaggle (default: on)",
    )
    parser.add_argument(
        "--no-copy-to-working",
        action="store_false",
        dest="copy_to_working",
        help="Do not copy to /kaggle/working/",
    )
    parser.add_argument(
        "--download-link",
        action="store_true",
        help="Show IPython FileLink when in a notebook",
    )
    args = parser.parse_args()

    vift = vift_root()
    if not vift.is_dir():
        print(f"ERROR: VIFT not found: {vift}")
        return 1

    export_root = ips_root() / "exports" / "checkpoints"
    export_root.mkdir(parents=True, exist_ok=True)

    if args.all_runs:
        runs = list_runs(vift)
        if not runs:
            print("ERROR: No runs under", runs_dir(vift))
            return 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        zip_path = args.output or (export_root / f"all_runs_{stamp}.zip")
        zip_path = Path(zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for run in runs:
                ckpt_dir = run / "checkpoints"
                if not ckpt_dir.is_dir():
                    continue
                for f in sorted(ckpt_dir.rglob("*")):
                    if f.is_file():
                        rel = f.relative_to(ckpt_dir)
                        zf.write(f, arcname=f"{run.name}/checkpoints/{rel.as_posix()}")
        print(f"Zipped {len(runs)} runs -> {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    else:
        run = find_run(vift, args.run)
        if run is None:
            print("ERROR: No run found.", runs_dir(vift))
            return 1
        ckpt_dir = run / "checkpoints"
        if not ckpt_dir.is_dir():
            print(f"ERROR: No checkpoints in {ckpt_dir}")
            return 1

        zip_path = args.output or (export_root / f"{run.name}.zip")
        zip_path = Path(zip_path)

        if args.latest_only:
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for name in ("last.ckpt",):
                    p = ckpt_dir / name
                    if p.is_file():
                        zf.write(p, arcname=f"{run.name}/checkpoints/{name}")
                for p in sorted(ckpt_dir.glob("epoch_*.ckpt"))[:1]:
                    zf.write(p, arcname=f"{run.name}/checkpoints/{p.name}")
            print(f"Zipped latest checkpoints from {run.name}")
        else:
            zip_directory(ckpt_dir, zip_path, f"{run.name}/checkpoints")
            print(f"Zipped all checkpoints from {run.name}")

        if not zip_path.is_file():
            print("ERROR: Zip failed")
            return 1
        print(f"Output: {zip_path.resolve()} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    # Kaggle-friendly copy
    kaggle_working = Path("/kaggle/working")
    if args.copy_to_working and kaggle_working.is_dir():
        dest = kaggle_working / zip_path.name
        shutil.copy2(zip_path, dest)
        print(f"Copied for Kaggle Output tab: {dest}")

    print("\nNext session: attach zip as Kaggle Dataset, then:")
    print("  python scripts/phase2/import_checkpoints.py --zip /kaggle/input/<dataset>/your.zip")

    if args.download_link:
        try_ipython_download(zip_path if zip_path.is_file() else dest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
