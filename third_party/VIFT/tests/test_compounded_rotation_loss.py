"""CompoundedRotationPoseLoss: zero at GT, grows with a constant per-frame bias, matches numpy composition."""
import numpy as np
import torch

from src.metrics.weighted_loss import AngleWeightedPoseLoss, CompoundedRotationPoseLoss
from src.utils.kitti_utils import eulerAnglesToRotationMatrix


def _gts(b=4, t=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    rot = torch.randn(b, t, 3, generator=g) * 0.02
    trans = torch.randn(b, t, 3, generator=g)
    return torch.cat([rot, trans], -1)


def test_zero_at_gt_and_equal_to_base_without_compound_term():
    gts = _gts()
    loss = CompoundedRotationPoseLoss(torch.nn.L1Loss(reduction="none"), angle_weight=10, compound_weight=10)
    assert loss(gts.clone(), gts).item() < 1e-6
    pred = gts + 0.01
    base = AngleWeightedPoseLoss(torch.nn.L1Loss(reduction="none"), angle_weight=10)(pred, gts)
    off = CompoundedRotationPoseLoss(torch.nn.L1Loss(reduction="none"), angle_weight=10, compound_weight=0)(pred, gts)
    assert torch.allclose(base, off)


def test_constant_yaw_bias_grows_with_horizon():
    """A yaw bias b per frame gives a compounded yaw error ~k*b; the term is ~ mean_k(k*b)/3 per axis."""
    gts = _gts()
    gts[:, :, :3] = 0.0  # straight motion: compounded error is exactly k*b about y
    b = 1e-3
    pred = gts.clone()
    pred[:, :, 1] += b
    loss = CompoundedRotationPoseLoss(torch.nn.L1Loss(reduction="none"), angle_weight=0, compound_weight=1)
    comp = (loss(pred, gts) - AngleWeightedPoseLoss(torch.nn.L1Loss(reduction="none"), angle_weight=0)(pred, gts)).item()
    k = np.arange(2, 11)
    expected = np.mean(np.sin(k * b)) / 3  # only the y component is non-zero
    assert abs(comp - expected) < 1e-6, (comp, expected)
    assert comp > 5 * b / 3  # well above the per-frame size


def test_composition_matches_numpy():
    gts = _gts(b=1, t=6, seed=3)
    rp = CompoundedRotationPoseLoss._prefix_products(
        torch.stack([torch.as_tensor(eulerAnglesToRotationMatrix(e.numpy()), dtype=torch.float32) for e in gts[0, :, :3]])[None]
    )
    acc = np.eye(3)
    for k in range(6):
        acc = acc @ eulerAnglesToRotationMatrix(gts[0, k, :3].numpy())
        assert np.allclose(rp[0, k].numpy(), acc, atol=1e-5)


def test_backward_finite_at_gt():
    gts = _gts()
    pred = gts.clone().requires_grad_()
    CompoundedRotationPoseLoss(torch.nn.L1Loss(reduction="none"))(pred, gts).backward()
    assert torch.isfinite(pred.grad).all()
