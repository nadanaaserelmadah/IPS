import torch
import torch.nn as nn
try:
    from utils.kitti_utils import eulerAnglesToRotationMatrixTorch as etr
    from utils import rpmg
except ImportError:  # Hydra rootutils puts project root on path; tests may only have src/
    from src.utils.kitti_utils import eulerAnglesToRotationMatrixTorch as etr
    from src.utils import rpmg

class WeightedMSEPoseLoss(nn.Module):
    def __init__(self, angle_weight=100):
        super(WeightedMSEPoseLoss, self).__init__()
        self.angle_weight = angle_weight

    def forward(self, poses, gts):
        angle_loss = torch.nn.functional.mse_loss(poses[:,:,:3], gts[:, :, :3])
        translation_loss = torch.nn.functional.mse_loss(poses[:,:,3:], gts[:, :, 3:])
        
        pose_loss = self.angle_weight * angle_loss + translation_loss
        return pose_loss
    
class WeightedMAEPoseLoss(nn.Module):
    def __init__(self, angle_weight=10):
        super(WeightedMAEPoseLoss, self).__init__()
        self.angle_weight = angle_weight

    def forward(self, poses, gts):
        angle_loss = torch.nn.functional.l1_loss(poses[:,:,:3], gts[:, :, :3])
        translation_loss = torch.nn.functional.l1_loss(poses[:,:,3:], gts[:, :, 3:])
        
        pose_loss = self.angle_weight * angle_loss + translation_loss
        return pose_loss

class LieTorchPoseLoss(nn.Module):
    def __init__(self, angle_weight=100):
        super().__init__()
        self.angle_weight= angle_weight
    def forward(self, poses, gts): # poses : translation + SE3 matrix, gts : translation + Euler Angles

        pass

class RPMGPoseLoss(nn.Module):
    def __init__(self, base_loss_fn, angle_weight=100):
        super().__init__()
        self.angle_weight = angle_weight
    def forward(self, poses, gts, weights, use_weighted_loss=True):
        angle_loss = torch.nn.functional.l1_loss(
            rpmg.simple_RPMG.apply(
                                   etr(poses[:,:,:3]).view(poses.shape[0]*poses.shape[1],9),
                                   1/4,
                                   0.01
                                   ).view(-1,9),
            etr(gts[:,:,:3]).view(poses.shape[0]*poses.shape[1],9)
        )
        
        translation_loss = torch.nn.functional.l1_loss(poses[:,:,3:], gts[:, :, 3:])
        
        pose_loss = self.angle_weight * angle_loss + translation_loss
        return pose_loss

class DataWeightedRPMGPoseLoss(nn.Module):
    def __init__(self, base_loss_fn, angle_weight=100):
        super().__init__()
        self.angle_weight = angle_weight
        self.base_loss_fn = base_loss_fn # make sure reduction is None
    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        angle_loss = self.base_loss_fn(
            rpmg.simple_RPMG.apply(
                                   etr(poses[:,:,:3]).view(poses.shape[0]*poses.shape[1],9),
                                   1/4,
                                   0.01
                                   ).view(-1,9),
            etr(gts[:,:,:3]).view(poses.shape[0]*poses.shape[1],9)
        )
        
        translation_loss = self.base_loss_fn(poses[:,:,3:], gts[:, :, 3:])
        
        total_loss = self.angle_weight * torch.sum(angle_loss,-1).view(poses.shape[0],-1) + torch.sum(translation_loss,-1).view(poses.shape[0],-1)
        if use_weighted_loss and (weights is not None):
            # Normalize weights
            weights = weights / weights.sum()
            weights = weights.view(-1, 1)  # reshape to (batch_size, 1, 1)
            
            # Apply weights
            total_loss = weights * total_loss.sum(dim=(1)).view(-1, 1)
        return total_loss.mean()

class DataWeightedPoseLoss(nn.Module):
    def __init__(self, base_loss_fn, angle_weight=100):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.angle_weight = angle_weight

    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        batch_size, seq_len, _ = poses.shape

        # Calculate losses for rotation and translation separately
        angle_loss = self.base_loss_fn(poses[:,:,:3], gts[:,:,:3])
        translation_loss = self.base_loss_fn(poses[:,:,3:], gts[:,:,3:])

        # Combine losses
        total_loss = self.angle_weight * angle_loss + translation_loss

        if use_weighted_loss and (weights is not None):
            # Normalize weights
            weights = weights / weights.sum()
            weights = weights.view(-1, 1, 1)  # reshape to (batch_size, 1, 1)
            
            # Apply weights
            total_loss = weights * total_loss.sum(dim=(1, 2)).view(-1, 1, 1)
        
        return total_loss.mean()

class CustomWeightedPoseLoss(nn.Module):
    def __init__(self, base_loss_fn, angle_weight=100):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.angle_weight = angle_weight

    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        batch_size, seq_len, _ = poses.shape
        loss1 = self.base_loss_fn(poses[:,:,0], gts[:,:,0])
        loss2 = self.base_loss_fn(poses[:,:,1], gts[:,:,1])
        loss3 = self.base_loss_fn(poses[:,:,2], gts[:,:,2])
        loss4 = self.base_loss_fn(poses[:,:,3], gts[:,:,3])
        loss5 = self.base_loss_fn(poses[:,:,4], gts[:,:,4])
        loss6 = self.base_loss_fn(poses[:,:,5], gts[:,:,5])

        total_loss = loss1 * (2/3) + loss2 * (1/5) + loss3 + loss4 * (0.1) + loss5 * (0.1) + loss6 * (0.03)
 
        if use_weighted_loss and (weights is not None):
            # Normalize weights
            weights = weights / weights.sum()
            weights = weights.view(-1, 1, 1)  # reshape to (batch_size, 1, 1)
            
            # Apply weights
            total_loss = weights * total_loss.sum(dim=(1)).view(-1, 1, 1)
        
        return total_loss.mean()


class GeodesicRot6DPoseLoss(nn.Module):
    """Horizon-1 Smooth-L1 translation + geodesic rotation on 9-D (6D+xyz) predictions.

    Ground truth remains KITTI 6-D Euler+xyz. Gradients go through Gram–Schmidt, not Euler.
    """

    def __init__(
        self,
        translation_weight: float = 1.0,
        rotation_weight: float = 10.0,
        smooth_l1_beta: float = 0.05,
    ):
        super().__init__()
        self.translation_weight = float(translation_weight)
        self.rotation_weight = float(rotation_weight)
        self.smooth_l1 = nn.SmoothL1Loss(beta=smooth_l1_beta, reduction="mean")

    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        from src.utils.rotation6d import rotation_6d_to_matrix

        if poses.shape[-1] != 9:
            raise ValueError(f"GeodesicRot6DPoseLoss expects 9-D preds, got {tuple(poses.shape)}")
        if gts.shape[-1] != 6:
            raise ValueError(f"GeodesicRot6DPoseLoss expects 6-D GT Euler+xyz, got {tuple(gts.shape)}")
        r_pred = rotation_6d_to_matrix(poses[..., :6])
        r_gt = etr(gts[..., :3])
        trans_loss = self.smooth_l1(poses[..., 6:9], gts[..., 3:6])
        rot_loss = _geodesic_rotation_loss(r_pred, r_gt)
        total = self.translation_weight * trans_loss + self.rotation_weight * rot_loss
        if not torch.isfinite(total):
            raise RuntimeError(f"Non-finite GeodesicRot6DPoseLoss trans={trans_loss} rot={rot_loss}")
        return total


class AngleWeightedPoseLoss(nn.Module):
    def __init__(self, base_loss_fn, angle_weight=100):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.angle_weight = angle_weight

    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        batch_size, seq_len, _ = poses.shape

        # Calculate losses for rotation and translation separately
        angle_loss = self.base_loss_fn(poses[:,:,:3], gts[:,:,:3])
        translation_loss = self.base_loss_fn(poses[:,:,3:], gts[:,:,3:])

        # Combine losses
        total_loss = self.angle_weight * angle_loss + translation_loss
        
        return total_loss.mean()


class CompoundedRotationPoseLoss(AngleWeightedPoseLoss):
    """
    AngleWeightedPoseLoss plus an L1 on the error of rotations compounded from the window start.

    For k = 2..T, R_0k = R_01 R_12 ... R_(k-1)k is built from the predicted and from the GT
    relative Euler angles, and the error E = R_0k_pred^T R_0k_gt is scored by the L1 of its
    skew part vee((E - E^T) / 2) = sin(theta) * axis (per axis; ~theta for small errors, and
    smooth at zero, unlike acos). A constant per-frame bias b grows to k*b here, so a bias far
    below the per-frame noise (KITTI yaw: ~0.0025 deg/frame) becomes visible to the loss.
    """

    def __init__(self, base_loss_fn, angle_weight=10, compound_weight=10.0, min_horizon=2):
        super().__init__(base_loss_fn, angle_weight)
        self.compound_weight = float(compound_weight)
        self.min_horizon = int(min_horizon)

    @staticmethod
    def _prefix_products(rot: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, 3) relative rotations -> (B, T, 3, 3) products R_01 ... R_(k-1)k."""
        out = [rot[:, 0]]
        for k in range(1, rot.shape[1]):
            out.append(torch.matmul(out[-1], rot[:, k]))
        return torch.stack(out, dim=1)

    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        base = super().forward(poses, gts, weights, use_weighted_loss)
        if self.compound_weight == 0 or poses.shape[1] < self.min_horizon:
            return base
        rp = self._prefix_products(etr(poses[:, :, :3]))
        rg = self._prefix_products(etr(gts[:, :, :3]))
        e = torch.matmul(rp.transpose(-1, -2), rg)[:, self.min_horizon - 1:]
        vee = 0.5 * torch.stack([e[..., 2, 1] - e[..., 1, 2], e[..., 0, 2] - e[..., 2, 0], e[..., 1, 0] - e[..., 0, 1]], dim=-1)
        return base + self.compound_weight * vee.abs().mean()


def _pose6_to_matrix_torch(pose: torch.Tensor) -> torch.Tensor:
    """Convert (..., 6) euler+xyz to (..., 4, 4) SE(3)."""
    r = etr(pose[..., :3])
    t = pose[..., 3:]
    eye = torch.eye(4, device=pose.device, dtype=pose.dtype)
    out = eye.expand(pose.shape[:-1] + (4, 4)).clone()
    out[..., :3, :3] = r
    out[..., :3, 3] = t
    return out


def _geodesic_rotation_loss(r_pred: torch.Tensor, r_gt: torch.Tensor) -> torch.Tensor:
    """Mean geodesic angle between rotation matrices (..., 3, 3)."""
    r_rel = torch.matmul(r_pred.transpose(-1, -2), r_gt)
    # trace = 1 + 2 cos(theta) for SO(3)
    trace = r_rel[..., 0, 0] + r_rel[..., 1, 1] + r_rel[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_theta).mean()


def _compose_horizon(poses: torch.Tensor, horizon: int) -> torch.Tensor:
    """
    Compose ``horizon`` consecutive relative SE(3) poses.

    poses: (B, T, 6) → returns (B, T-horizon+1, 6) for the composed edge,
    but we only need xyz+rotation matrices for loss; return matrices (B, T-h+1, 4, 4).
    """
    b, t, _ = poses.shape
    mats = _pose6_to_matrix_torch(poses)  # (B, T, 4, 4)
    # cumulative product along time
    # out[i] = mats[i] @ mats[i+1] @ ... @ mats[i+horizon-1]
    out = []
    for i in range(t - horizon + 1):
        acc = mats[:, i]
        for k in range(1, horizon):
            acc = torch.matmul(acc, mats[:, i + k])
        out.append(acc)
    return torch.stack(out, dim=1)


def _direction_loss(pairs, min_motion: float, ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean ||t_pred/|t_pred| - t_gt/|t_gt||| over (pred, gt) translation pairs where |t_gt| > min_motion."""
    terms = []
    for pred, gt in pairs:
        n_gt = gt.norm(dim=-1)
        mask = n_gt > min_motion
        if mask.any():
            u_p = pred[mask] / pred[mask].norm(dim=-1, keepdim=True).clamp_min(eps)
            u_g = gt[mask] / n_gt[mask].unsqueeze(-1)
            terms.append((u_p - u_g).norm(dim=-1).mean())
    return torch.stack(terms).mean() if terms else ref.new_zeros(())


class MultiHorizonPoseLoss(nn.Module):
    """
    Phase 5 geometric pose loss.

    - Smooth-L1 on translation
    - Geodesic rotation via rotation matrices
    - Multi-horizon SE(3) composition consistency (default horizons 2/5/10)
    - Optional supervised first-difference temporal term (gt_delta)

    - Optional translation-direction term (direction_weight > 0): L2 between the unit vectors
      of predicted and GT translation (TartanVO's up-to-scale loss, arXiv 2011.00359 Eq. 3),
      per step and on the composed horizons, only where the GT moves more than
      direction_min_motion metres. EuRoC models fail on direction (31-90 deg per edge), and an
      L1/Smooth-L1 on small per-frame translations barely penalises a wrong direction.

    With horizon_weight=0 and temporal_weight=0 this reduces to a weighted
    translation + geodesic rotation loss (not identical to AngleWeightedPoseLoss
    because rotation is geodesic rather than Euler L1).
    """

    def __init__(
        self,
        translation_weight: float = 1.0,
        rotation_weight: float = 10.0,
        horizon_weight: float = 0.1,
        temporal_weight: float = 0.0,
        horizons=(2, 5, 10),
        smooth_l1_beta: float = 0.05,
        temporal_mode: str = "gt_delta",
        direction_weight: float = 0.0,
        direction_min_motion: float = 0.01,
    ):
        super().__init__()
        self.direction_weight = float(direction_weight)
        self.direction_min_motion = float(direction_min_motion)
        self.last_direction_loss = 0.0
        self.translation_weight = float(translation_weight)
        self.rotation_weight = float(rotation_weight)
        self.horizon_weight = float(horizon_weight)
        self.temporal_weight = float(temporal_weight)
        self.horizons = tuple(int(h) for h in horizons)
        self.smooth_l1 = nn.SmoothL1Loss(beta=smooth_l1_beta, reduction="mean")
        self.temporal_mode = temporal_mode
        self.last_translation_loss = 0.0
        self.last_rotation_loss = 0.0
        self.last_multihorizon_loss = 0.0
        self.last_temporal_loss = 0.0

    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        if poses.shape != gts.shape:
            raise ValueError(f"Shape mismatch poses={poses.shape} gts={gts.shape}")

        # Per-step translation (Smooth-L1) and geodesic rotation
        trans_loss = self.smooth_l1(poses[..., 3:], gts[..., 3:])
        r_pred = etr(poses[..., :3])
        r_gt = etr(gts[..., :3])
        rot_loss = _geodesic_rotation_loss(r_pred, r_gt)

        # Multi-horizon SE(3) composition
        mh_terms = []
        dir_pairs = [(poses[..., 3:], gts[..., 3:])]
        t = poses.shape[1]
        for h in self.horizons:
            if h > t:
                continue
            pred_h = _compose_horizon(poses, h)
            gt_h = _compose_horizon(gts, h)
            dir_pairs.append((pred_h[..., :3, 3], gt_h[..., :3, 3]))
            mh_trans = self.smooth_l1(pred_h[..., :3, 3], gt_h[..., :3, 3])
            mh_rot = _geodesic_rotation_loss(pred_h[..., :3, :3], gt_h[..., :3, :3])
            mh_terms.append(self.rotation_weight * mh_rot + self.translation_weight * mh_trans)
        if mh_terms:
            mh_loss = torch.stack(mh_terms).mean()
        else:
            mh_loss = poses.new_zeros(())

        # Optional supervised temporal consistency on first differences
        if self.temporal_weight > 0 and t >= 2 and self.temporal_mode == "gt_delta":
            d_pred = poses[:, 1:] - poses[:, :-1]
            d_gt = gts[:, 1:] - gts[:, :-1]
            temp_rot = self.smooth_l1(d_pred[..., :3], d_gt[..., :3])
            temp_trans = self.smooth_l1(d_pred[..., 3:], d_gt[..., 3:])
            temporal_loss = self.rotation_weight * temp_rot + self.translation_weight * temp_trans
        else:
            temporal_loss = poses.new_zeros(())

        if self.direction_weight > 0:
            direction_loss = _direction_loss(dir_pairs, self.direction_min_motion, poses)
        else:
            direction_loss = poses.new_zeros(())

        total = (
            self.translation_weight * trans_loss
            + self.rotation_weight * rot_loss
            + self.horizon_weight * mh_loss
            + self.temporal_weight * temporal_loss
            + self.direction_weight * direction_loss
        )
        self.last_direction_loss = float(direction_loss.detach())

        self.last_translation_loss = float(trans_loss.detach())
        self.last_rotation_loss = float(rot_loss.detach())
        self.last_multihorizon_loss = float(mh_loss.detach())
        self.last_temporal_loss = float(temporal_loss.detach())

        if not torch.isfinite(total):
            raise RuntimeError(
                f"Non-finite MultiHorizonPoseLoss: trans={trans_loss.item()} "
                f"rot={rot_loss.item()} mh={mh_loss.item()} temp={temporal_loss.item()}"
            )
        return total


class TokenizedPoseLoss(nn.Module):
    def __init__(self, angle_weight=100):
        super().__init__()

    def forward(self, poses, gts):
        
        out, loss = poses
        return loss


class WeightedTokenizedPoseLoss(nn.Module):
    def __init__(self, base_loss_fn, angle_weight=100):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.angle_weight = angle_weight

    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        out, ce_loss = poses
        return ce_loss


class PIViftLoss(nn.Module):
    """Multi-horizon pose loss plus optional velocity / gravity supervision."""

    def __init__(
        self,
        translation_weight: float = 1.0,
        rotation_weight: float = 40.0,
        horizon_weight: float = 0.1,
        temporal_weight: float = 0.0,
        velocity_weight: float = 0.1,
        gravity_weight: float = 0.05,
        horizons=(2, 5, 10),
        smooth_l1_beta: float = 0.05,
        direction_weight: float = 0.0,
        direction_min_motion: float = 0.01,
    ):
        super().__init__()
        self.pose_loss = MultiHorizonPoseLoss(
            translation_weight=translation_weight,
            rotation_weight=rotation_weight,
            horizon_weight=horizon_weight,
            temporal_weight=temporal_weight,
            horizons=horizons,
            smooth_l1_beta=smooth_l1_beta,
            direction_weight=direction_weight,
            direction_min_motion=direction_min_motion,
        )
        self.velocity_weight = float(velocity_weight)
        self.gravity_weight = float(gravity_weight)
        self.smooth_l1 = nn.SmoothL1Loss(beta=smooth_l1_beta, reduction="mean")
        self.last_velocity_loss = 0.0
        self.last_gravity_loss = 0.0

    def forward(self, poses, gts, weights=None, use_weighted_loss=True, aux=None):
        pose_gt = gts[..., :6]
        total = self.pose_loss(poses, pose_gt, weights, use_weighted_loss)
        vel_loss = poses.new_zeros(())
        g_loss = poses.new_zeros(())
        if aux is not None and gts.shape[-1] >= 12:
            vel_loss = self.smooth_l1(aux["vel"], gts[..., 6:9])
            g_hat = torch.nn.functional.normalize(aux["g"], dim=-1)
            g_gt = torch.nn.functional.normalize(gts[..., 9:12], dim=-1)
            g_loss = (1.0 - (g_hat * g_gt).sum(dim=-1)).mean()
            total = total + self.velocity_weight * vel_loss + self.gravity_weight * g_loss
        self.last_translation_loss = self.pose_loss.last_translation_loss
        self.last_rotation_loss = self.pose_loss.last_rotation_loss
        self.last_multihorizon_loss = self.pose_loss.last_multihorizon_loss
        self.last_temporal_loss = self.pose_loss.last_temporal_loss
        self.last_direction_loss = self.pose_loss.last_direction_loss
        self.last_velocity_loss = float(vel_loss.detach()) if torch.is_tensor(vel_loss) else 0.0
        self.last_gravity_loss = float(g_loss.detach()) if torch.is_tensor(g_loss) else 0.0
        return total
