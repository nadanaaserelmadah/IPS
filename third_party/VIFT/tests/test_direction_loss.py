"""Translation-direction term of MultiHorizonPoseLoss / PIViftLoss (direction_weight)."""
import math

import torch

from src.metrics.weighted_loss import MultiHorizonPoseLoss, PIViftLoss, _direction_loss


def _poses(b=2, t=10, seed=0, speed=0.05):
    g = torch.Generator().manual_seed(seed)
    rot = torch.randn(b, t, 3, generator=g) * 0.01
    trans = torch.randn(b, t, 3, generator=g)
    trans = trans / trans.norm(dim=-1, keepdim=True) * speed
    return torch.cat([rot, trans], -1)


def test_zero_at_gt_and_scale_invariant():
    gt = _poses()
    pairs = [(gt[..., 3:], gt[..., 3:])]
    assert _direction_loss(pairs, 0.01, gt).item() < 1e-6
    assert _direction_loss([(3.0 * gt[..., 3:], gt[..., 3:])], 0.01, gt).item() < 1e-6  # only direction counts


def test_value_matches_angle():
    """Unit vectors at angle a are 2 sin(a/2) apart."""
    gt = torch.zeros(1, 1, 3); gt[..., 2] = 0.05
    for deg in (10.0, 45.0, 90.0, 180.0):
        a = math.radians(deg)
        pred = torch.tensor([[[0.05 * math.sin(a), 0.0, 0.05 * math.cos(a)]]])
        got = _direction_loss([(pred, gt)], 0.01, gt).item()
        assert abs(got - 2 * math.sin(a / 2)) < 1e-5, (deg, got)


def test_small_motion_is_masked():
    gt = _poses(speed=0.005)  # 5 mm per frame < 1 cm threshold
    pred = -gt
    assert _direction_loss([(pred[..., 3:], gt[..., 3:])], 0.01, gt).item() == 0.0


def test_default_weight_unchanged_and_term_adds():
    gt = _poses()
    pred = gt.clone(); pred[..., 3:] = pred[..., 3:].roll(1, dims=-1)  # wrong directions
    base = MultiHorizonPoseLoss(rotation_weight=40.0)(pred, gt)
    zero = MultiHorizonPoseLoss(rotation_weight=40.0, direction_weight=0.0)(pred, gt)
    assert torch.allclose(base, zero)
    crit = MultiHorizonPoseLoss(rotation_weight=40.0, direction_weight=0.1)
    with_dir = crit(pred, gt)
    assert with_dir > base and crit.last_direction_loss > 0.1


def test_pivift_passes_through_and_backward_finite():
    gt = torch.cat([_poses(), torch.zeros(2, 10, 6)], -1)  # pose + vel + g columns
    pred = (gt[..., :6] + 0.01 * torch.randn(2, 10, 6)).requires_grad_()
    crit = PIViftLoss(direction_weight=0.1)
    loss = crit(pred, gt)
    loss.backward()
    assert torch.isfinite(pred.grad).all() and crit.last_direction_loss >= 0.0
