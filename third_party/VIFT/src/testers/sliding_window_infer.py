"""Shared sliding-window fusion helpers for EuRoC / TUM joint evaluation."""

from __future__ import annotations

from typing import Any, Optional, Tuple

EVAL_MODE_WINDOW_LAST = "window_last"
EVAL_MODE_STREAMING_HISTORY = "streaming_history"
# PI-VIFT only: carry gravity + biases across windows, re-estimate velocity per window.
EVAL_MODE_STREAMING_GB = "streaming_gb"


def resolve_eval_mode(use_history_in_eval: bool, eval_mode: Optional[str] = None) -> str:
    """Map legacy bool / explicit string to a canonical eval mode."""
    if eval_mode:
        mode = str(eval_mode).strip().lower()
        if mode in {"false", "0", "off", "none", "window", "window_last"}:
            return EVAL_MODE_WINDOW_LAST
        if mode in {"true", "1", "on", "history", "streaming", "streaming_history"}:
            return EVAL_MODE_STREAMING_HISTORY
        if mode in {EVAL_MODE_WINDOW_LAST, EVAL_MODE_STREAMING_HISTORY, EVAL_MODE_STREAMING_GB}:
            return mode
        raise ValueError(f"Unknown eval_mode={eval_mode!r}")
    return EVAL_MODE_STREAMING_HISTORY if use_history_in_eval else EVAL_MODE_WINDOW_LAST


def fuse_sliding_window(
    fusion: Any,
    latents: Any,
    hist: Optional[Any],
    gt_seq: Any,
    eval_mode: str = EVAL_MODE_WINDOW_LAST,
) -> Tuple[Any, Any]:
    """
    Fuse one sliding window and return (pose, updated_hist).

    ``window_last``: evaluate the full window independently; keep last edge.
    ``streaming_history``: after the first window, advance history **once** with
    only the newest latent (correct for stride-1 overlapping windows).
    """
    import torch

    mode = resolve_eval_mode(False, eval_mode)
    if mode != EVAL_MODE_STREAMING_HISTORY or hist is None:
        hist_out = latents
        pose = fusion((latents, None, None), gt_seq)
        return pose, hist_out

    # streaming_history: one new edge per window
    hist_out = torch.roll(hist, -1, dims=1)
    hist_out = hist_out.clone()
    hist_out[:, -1, :] = latents[:, -1, :]
    pose = fusion((hist_out, None, None), gt_seq)
    return pose, hist_out


def last_edge_numpy(pose: Any, gt_seq: Any):
    """Extract the final relative pose from a window prediction / GT."""
    import numpy as np
    import torch

    p = pose[0].detach().cpu().numpy() if isinstance(pose, torch.Tensor) else np.asarray(pose)
    if isinstance(gt_seq, torch.Tensor):
        g = gt_seq.detach().cpu().numpy()
    else:
        g = np.asarray(gt_seq)
    if p.ndim == 2:
        p = p[-1]
    if g.ndim == 2:
        g = g[-1]
    return p, g
