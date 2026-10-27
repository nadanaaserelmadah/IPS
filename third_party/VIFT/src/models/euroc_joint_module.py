"""Lightning module for Phase 3/5: fine-tune encoder + fusion on raw EuRoC/TUM."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from lightning import LightningModule

from src.models.components.euroc_joint_net import EurocJointNet


class EurocJointLitModule(LightningModule):
    def __init__(
        self,
        net: EurocJointNet,
        optimizer,
        scheduler,
        criterion,
        compile: bool,
        tester,
        metrics_calculator,
        pretrained_encoder: str,
        phase2_ckpt: Optional[str] = None,
        encoder_ckpt: Optional[str] = None,
        fusion_lr: float = 1e-4,
        freeze_encoder_epochs: int = 0,
        freeze_fusion_epochs: int = 0,
        encoder_heads_only: bool = False,
        encoder_lr_scale: float = 0.1,
        gradient_clip_val: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["net", "criterion", "tester", "metrics_calculator"],
        )
        self.fusion_lr = fusion_lr
        self.net = net
        self.criterion = criterion
        self.tester = tester
        self.metrics_calculator = metrics_calculator
        self.encoder_lr_scale = encoder_lr_scale
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.freeze_fusion_epochs = freeze_fusion_epochs
        self.encoder_heads_only = bool(encoder_heads_only)
        self.gradient_clip_val = float(gradient_clip_val)
        self._encoder_frozen = False
        self._fusion_frozen = False
        self._val_edges: list[tuple[torch.Tensor, torch.Tensor]] = []

        if pretrained_encoder:
            self.net.load_pretrained_encoder(pretrained_encoder)
        if encoder_ckpt:
            self._load_encoder_from_ckpt(encoder_ckpt)
        if phase2_ckpt:
            if type(self.net.fusion).__name__ == "PIViftFusion":
                print(
                    "Skipping Phase-2 fusion load: PI-VIFT weights are not PoseTransformer."
                )
            else:
                self._load_fusion_from_phase2(phase2_ckpt)
        if freeze_encoder_epochs > 0:
            self.net.freeze_encoder()
            self._encoder_frozen = True
        elif self.encoder_heads_only:
            self.net.freeze_encoder_backbone()
        if freeze_fusion_epochs > 0:
            self.net.freeze_fusion()
            self._fusion_frozen = True

    def _load_fusion_from_phase2(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        fusion_state = {
            k.replace("net.", "", 1): v
            for k, v in state.items()
            if k.startswith("net.")
        }
        if not fusion_state:
            raise RuntimeError(f"No net.* weights in Phase 2 checkpoint: {ckpt_path}")
        self.net.fusion.load_state_dict(fusion_state, strict=True)
        print(
            f"Fusion warm-start: loaded {len(fusion_state)} tensors from "
            f"{ckpt_path} (source epoch {ckpt.get('epoch', 'unknown')})"
        )

    def _load_encoder_from_ckpt(self, ckpt_path: str) -> None:
        """Warm-start the encoder from a joint (P3 / TUM-FT) Lightning checkpoint."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        enc_state = {
            k[len("net.encoder."):]: v for k, v in state.items() if k.startswith("net.encoder.")
        }
        if not enc_state:
            raise RuntimeError(f"No net.encoder.* weights in checkpoint: {ckpt_path}")
        self.net.encoder.load_state_dict(enc_state, strict=True)
        print(
            f"Encoder warm-start: loaded {len(enc_state)} tensors from "
            f"{ckpt_path} (source epoch {ckpt.get('epoch', 'unknown')})"
        )

    def _add_encoder_to_optimizer(self) -> None:
        """Register newly unfrozen encoder params with the optimizer and scheduler.

        configure_optimizers only sees params that are trainable at fit start, so an
        encoder frozen then would otherwise get gradients but never be updated.
        """
        opt = self.optimizers()
        opt = getattr(opt, "optimizer", opt)
        known = {id(p) for g in opt.param_groups for p in g["params"]}
        new = [p for p in self.net.encoder.parameters() if p.requires_grad and id(p) not in known]
        if not new:
            return
        enc_lr = float(self.fusion_lr) * self.encoder_lr_scale
        opt.add_param_group({"params": new, "lr": enc_lr, "initial_lr": enc_lr})
        sched = self.lr_schedulers()
        if sched is not None and hasattr(sched, "base_lrs"):
            sched.base_lrs.append(enc_lr)
            # Put the new group on the schedule's current point of the cosine.
            fusion_base = sched.base_lrs[-2]
            frac = opt.param_groups[-2]["lr"] / fusion_base if fusion_base else 1.0
            opt.param_groups[-1]["lr"] = enc_lr * frac
        print(f"Optimizer: added {len(new)} encoder tensors (base lr {enc_lr:g})")

    def forward(self, imgs: torch.Tensor, imus: torch.Tensor) -> torch.Tensor:
        return self.net(imgs, imus)

    def _unpack_batch(self, batch):
        (imgs, imus, _rot, weight), target = batch
        return imgs, imus, weight, target

    def on_train_epoch_start(self) -> None:
        if (
            self.freeze_encoder_epochs > 0
            and self.current_epoch >= self.freeze_encoder_epochs
            and self._encoder_frozen
        ):
            if self.encoder_heads_only:
                self.net.freeze_encoder_backbone()
            else:
                self.net.unfreeze_encoder()
            self._add_encoder_to_optimizer()
            self._encoder_frozen = False
            print(
                f"Epoch {self.current_epoch}: encoder unfrozen "
                f"(lr scale {self.encoder_lr_scale}"
                f"{', heads-only' if self.encoder_heads_only else ''})"
            )
        if (
            self.freeze_fusion_epochs > 0
            and self.current_epoch >= self.freeze_fusion_epochs
            and self._fusion_frozen
        ):
            self.net.unfreeze_fusion()
            self._fusion_frozen = False
            print(f"Epoch {self.current_epoch}: fusion unfrozen (lr {self.fusion_lr})")

    def _log_criterion_components(self, prefix: str) -> None:
        crit = self.criterion
        for attr, name in (
            ("last_translation_loss", "translation_loss"),
            ("last_rotation_loss", "rotation_loss"),
            ("last_multihorizon_loss", "multihorizon_loss"),
            ("last_temporal_loss", "temporal_loss"),
            ("last_velocity_loss", "velocity_loss"),
            ("last_gravity_loss", "gravity_loss"),
            ("last_direction_loss", "direction_loss"),
        ):
            if hasattr(crit, attr):
                self.log(f"{prefix}/{name}", float(getattr(crit, attr)), on_step=False, on_epoch=True)

    def _chunked_training_loss(self, imgs, imus, weight, target):
        """Truncated BPTT: K back-to-back windows, carrying the inertial state (detached).

        Window 0 starts fresh (state from latents + IMU); window k > 0 starts from
        window k-1's end-of-window state, which is exactly frame 0 of window k.
        """
        losses = []
        carry = None
        for k in range(imgs.shape[1]):
            out, aux = self.net.forward_with_aux(imgs[:, k], imus[:, k], carry)
            losses.append(self.criterion(out, target[:, k], weight[:, k], aux=aux))
            carry = {name: t.detach() for name, t in aux["carry"].items()}
        self.log("train/loss_fresh", losses[0], on_step=False, on_epoch=True)
        self.log("train/loss_carried", torch.stack(losses[1:]).mean(), on_step=False, on_epoch=True)
        return torch.stack(losses).mean()

    def training_step(self, batch, batch_idx):
        imgs, imus, weight, target = self._unpack_batch(batch)
        if imgs.dim() == 6:  # (B, K, T, C, H, W) from EuRoC(chunk_windows=K)
            loss = self._chunked_training_loss(imgs, imus, weight, target)
        elif hasattr(self.net, "forward_with_aux"):
            out, aux = self.net.forward_with_aux(imgs, imus)
            try:
                loss = self.criterion(out, target, weight, aux=aux)
            except TypeError:
                loss = self.criterion(out, target, weight)
        else:
            out = self.forward(imgs, imus)
            loss = self.criterion(out, target, weight)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite train loss at step {batch_idx}")
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self._log_criterion_components("train")
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, imus, weight, target = self._unpack_batch(batch)
        if hasattr(self.net, "forward_with_aux"):
            out, aux = self.net.forward_with_aux(imgs, imus)
            try:
                loss = self.criterion(out, target, weight, use_weighted_loss=False, aux=aux)
            except TypeError:
                loss = self.criterion(out, target, weight, use_weighted_loss=False)
        else:
            out = self.forward(imgs, imus)
            loss = self.criterion(out, target, weight, use_weighted_loss=False)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_criterion_components("val")
        if out.dim() == 3:  # last edge of each stride-1 window, for val/ate_se3
            self._val_edges.append((out[:, -1, :6].detach().float().cpu(), target[:, -1, :6].float().cpu()))
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_edges = []

    def on_validation_epoch_end(self) -> None:
        """SE3 ATE of the val sequence from window_last edges (val loader is unshuffled,
        stride-1 windows), against the integrated GT relatives. Assumes one val sequence."""
        if len(self._val_edges) == 0 or self.trainer.sanity_checking:
            return
        import numpy as np

        from src.metrics.euroc_metrics_calculator import ate_rmse_m
        from src.utils.euroc_utils import path_accu

        est = torch.cat([e for e, _ in self._val_edges]).numpy().astype(np.float64)
        gt = torch.cat([g for _, g in self._val_edges]).numpy().astype(np.float64)
        self._val_edges = []
        if len(est) < 3:
            return
        est_xyz = np.array([T[:3, 3] for T in path_accu(est)])
        gt_xyz = np.array([T[:3, 3] for T in path_accu(gt)])
        self.log("val/ate_se3", ate_rmse_m(est_xyz, gt_xyz, alignment="se3"), prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass

    def _run_custom_sequence_test(self) -> None:
        if getattr(self, "_custom_sequence_test_done", False):
            return
        self._custom_sequence_test_done = True
        print("Running full sequence evaluation (raw images + encoder; slow)...")
        with torch.inference_mode():
            results = self.tester.test(self.net)
        metrics = self.metrics_calculator.calculate_metrics(results)
        for name, value in metrics.items():
            self.log(f"test/{name}", value)

        # hp_metric = mean valid ATE (EuRoC/TUM) or mean t_rel (KITTI joint ablation)
        if "mean_ate_m" in metrics and metrics["mean_ate_m"] == metrics["mean_ate_m"]:
            hp = float(metrics["mean_ate_m"])
        else:
            t_rel_keys = [k for k in metrics if k.endswith("_t_rel")]
            if t_rel_keys:
                hp = float(sum(metrics[k] for k in t_rel_keys) / len(t_rel_keys))
                self.log("test/mean_t_rel", hp)
            else:
                ate_keys = [
                    k
                    for k, v in metrics.items()
                    if k.endswith("_ate_m")
                    and not k.endswith("_ate_origin_m")
                    and not k.endswith("_ate_se3_m")
                    and not k.endswith("_ate_sim3_m")
                    and metrics.get(k.replace("_ate_m", "_valid"), 1.0) >= 0.5
                ]
                vals = [metrics[k] for k in ate_keys]
                hp = float(sum(vals) / len(vals)) if vals else float("nan")
        self.log("hp_metric", hp)

        # Always dump poses for Phase 6 / offline (even when logger=null).
        save_dir = None
        logger = self.trainer.logger
        if logger is not None and getattr(logger, "log_dir", None):
            save_dir = str(logger.log_dir)
        elif getattr(self.trainer, "default_root_dir", None):
            save_dir = str(Path(self.trainer.default_root_dir) / "pose_dumps")
        else:
            save_dir = str(Path.cwd() / "pose_dumps")

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.tester.save_results(results, save_dir)
        print(f"Saved pose dumps to: {save_dir}")

        kaggle_dir = Path("/kaggle/working")
        if kaggle_dir.is_dir():
            kaggle_poses = kaggle_dir / "phase5_pose_dumps"
            kaggle_poses.mkdir(parents=True, exist_ok=True)
            self.tester.save_results(results, str(kaggle_poses))
            print(f"Also saved pose dumps to: {kaggle_poses}")

    def on_test_start(self) -> None:
        self._custom_sequence_test_done = False

    def on_test_epoch_end(self):
        self._run_custom_sequence_test()

    def on_test_end(self) -> None:
        # Lightning skips on_test_epoch_end when test_dataloader length is 0.
        self._run_custom_sequence_test()

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        fusion_lr = float(self.fusion_lr)
        enc_lr = fusion_lr * self.encoder_lr_scale
        groups = []
        enc_params = [p for p in self.net.encoder.parameters() if p.requires_grad]
        fusion_params = [p for p in self.net.fusion.parameters() if p.requires_grad]
        if enc_params:
            groups.append({"params": enc_params, "lr": enc_lr})
        if fusion_params:
            groups.append({"params": fusion_params, "lr": fusion_lr})
        if not groups:
            raise RuntimeError("No trainable parameters (encoder and fusion both frozen)")
        opt = self.hparams.optimizer(params=groups)
        if self.hparams.scheduler is not None:
            sched = self.hparams.scheduler(optimizer=opt)
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sched,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": opt}
