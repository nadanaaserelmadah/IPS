"""Lightning callback that exports checkpoint ZIPs at a fixed epoch interval."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import torch
from lightning import Callback, Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class CheckpointZipCallback(Callback):
    """
    Zip ``last.ckpt`` (+ matching epoch_*.ckpt when present) every N epochs.

    Prefer resuming from ``last.ckpt`` inside the newest ZIP. The ZIP also
    writes ``resume_info.json`` with the true Lightning epoch to detect stale packs.

    Hook: ``on_train_epoch_end`` (NOT ``on_validation_end``) so ModelCheckpoint
    has already written ``last.ckpt`` for this epoch. Keep this callback *after*
    the periodic ModelCheckpoint in the Hydra callbacks list.
    """

    def __init__(
        self,
        every_n_epochs: int = 5,
        output_dir: str | None = None,
        copy_to_kaggle_working: bool = True,
    ) -> None:
        super().__init__()
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be >= 1")
        self.every_n_epochs = every_n_epochs
        self.output_dir = Path(output_dir) if output_dir else None
        self.copy_to_kaggle_working = copy_to_kaggle_working
        self._last_exported_epoch = -1

    def setup(self, trainer: Trainer, pl_module, stage: str) -> None:
        if stage != "fit":
            return
        # Ensure we run *after* ModelCheckpoint on the same hook.
        cbs = trainer.callbacks
        if self in cbs:
            cbs.remove(self)
            cbs.append(self)
        print(
            f"CheckpointZipCallback active: every {self.every_n_epochs} epochs, "
            f"copy_to_kaggle_working={self.copy_to_kaggle_working}, "
            f"callback_index={cbs.index(self) if self in cbs else -1}/{len(cbs)}"
        )

    def _checkpoint_dir(self, trainer: Trainer) -> Path | None:
        # Prefer the periodic callback that owns last.ckpt when several exist.
        for cb in trainer.checkpoint_callbacks:
            if getattr(cb, "save_last", False) and cb.dirpath:
                return Path(cb.dirpath)
        if trainer.checkpoint_callback is not None and trainer.checkpoint_callback.dirpath:
            return Path(trainer.checkpoint_callback.dirpath)
        return None

    def _read_ckpt_epoch(self, path: Path) -> int | None:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 — report and skip validation
            print(f"WARN: CheckpointZipCallback could not read {path}: {exc}")
            return None
        if "epoch" not in ckpt:
            return None
        return int(ckpt["epoch"])

    def _should_export(self, trainer: Trainer) -> bool:
        if trainer.sanity_checking:
            return False
        completed = trainer.current_epoch + 1
        if completed % self.every_n_epochs != 0:
            return False
        if completed == self._last_exported_epoch:
            return False
        return True

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module) -> None:
        """Run after ModelCheckpoint's on_train_epoch_end when listed last in callbacks."""
        if not self._should_export(trainer):
            return
        self._export_zip(trainer)

    def _export_zip(self, trainer: Trainer) -> None:
        completed = trainer.current_epoch + 1
        checkpoint_dir = self._checkpoint_dir(trainer)
        if checkpoint_dir is None:
            print("WARN: CheckpointZipCallback found no ModelCheckpoint directory")
            return

        last_path = checkpoint_dir / "last.ckpt"
        epoch_path = checkpoint_dir / f"epoch_{trainer.current_epoch:03d}.ckpt"

        # Resolve files to pack (prefer real last.ckpt).
        pack: list[tuple[str, Path]] = []
        if last_path.is_file():
            pack.append(("checkpoints/last.ckpt", last_path))
        elif epoch_path.is_file():
            print(f"WARN: last.ckpt missing; packing {epoch_path.name} as last.ckpt")
            pack.append(("checkpoints/last.ckpt", epoch_path))
        else:
            # Last resort: newest epoch_*.ckpt
            epoch_globs = sorted(checkpoint_dir.glob("epoch_*.ckpt"))
            if not epoch_globs:
                print(
                    f"WARN: CheckpointZipCallback: no checkpoints in {checkpoint_dir} "
                    f"after epoch {completed} — skip ZIP. "
                    f"Check that callbacks=phase2_checkpoint is active."
                )
                return
            newest = epoch_globs[-1]
            print(f"WARN: packing newest {newest.name} as last.ckpt")
            pack.append(("checkpoints/last.ckpt", newest))

        if epoch_path.is_file() and epoch_path.resolve() != pack[0][1].resolve():
            pack.append((f"checkpoints/{epoch_path.name}", epoch_path))

        source_for_meta = pack[0][1]
        ckpt_epoch = self._read_ckpt_epoch(source_for_meta)
        if ckpt_epoch is not None and ckpt_epoch != trainer.current_epoch:
            print(
                f"WARN: ckpt epoch={ckpt_epoch} but trainer.current_epoch="
                f"{trainer.current_epoch}. ZIP may be slightly stale."
            )

        export_dir = self.output_dir or (checkpoint_dir.parent / "checkpoint_zips")
        export_dir.mkdir(parents=True, exist_ok=True)
        zip_path = export_dir / f"checkpoint_epoch_{completed:03d}.zip"
        temp_path = zip_path.with_suffix(".zip.tmp")

        resume_info = {
            "completed_epochs": completed,
            "lightning_epoch": int(ckpt_epoch)
            if ckpt_epoch is not None
            else trainer.current_epoch,
            "resume_from": "checkpoints/last.ckpt",
            "note": (
                "After resume, training continues at lightning_epoch+1. "
                "Always resume from last.ckpt inside this ZIP."
            ),
        }

        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for arcname, source in pack:
                archive.write(source, arcname=arcname)
            archive.writestr("resume_info.json", json.dumps(resume_info, indent=2))
        temp_path.replace(zip_path)

        copied = None
        if self.copy_to_kaggle_working:
            kaggle_dir = Path("/kaggle/working")
            if kaggle_dir.is_dir():
                dest = kaggle_dir / zip_path.name
                try:
                    shutil.copy2(zip_path, dest)
                    copied = dest
                except OSError as exc:
                    print(f"WARN: failed to copy ZIP to {dest}: {exc}")
            else:
                print("WARN: /kaggle/working not found — ZIP only under run dir")

        self._last_exported_epoch = completed
        next_epoch = (ckpt_epoch + 1) if ckpt_epoch is not None else completed
        msg = (
            f"Checkpoint ZIP saved: {zip_path} "
            f"({zip_path.stat().st_size / 1e6:.1f} MB) "
            f"— resume continues at epoch {next_epoch}"
        )
        if copied is not None:
            msg += f" | copied to {copied}"
        print(msg)
