"""Phase 6-C backend entry (sliding-window factor graph)."""

from __future__ import annotations

from typing import Sequence

from src.phase6.factor_graph_6c import NoiseModel, refine_trajectory_6c

__all__ = ["NoiseModel", "refine_trajectory_6c"]


def refine_trajectory_backend_6c(
    rel_poses,
    imu,
    edge_times_ns: Sequence[tuple[int, int]],
    **kwargs,
):
    """Alias used by the runner."""
    return refine_trajectory_6c(rel_poses, imu, edge_times_ns, **kwargs)
