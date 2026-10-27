"""Phase 6 package: IMU fusion backends (6-A scaffold + 6-B gravity-aware)."""

from .backend_6b import refine_trajectory_6b
from .factor_graph_6c import NoiseModel, refine_trajectory_6c
from .imu_preintegration import (
    PreintegrationResult,
    preintegrate_segment,
    relative_rotation_geodesic_blend,
    so3_exp,
    so3_log,
)
from .imu_preintegration_se3 import (
    BiasState,
    PreintegrationSE3,
    estimate_gravity_in_world,
    estimate_gyro_bias,
    preintegrate_segment_se3,
)
from .landmark_ba_6e import refine_trajectory_6e
from .loop_closure import RelativeConstraint, detect_spatial_loop_candidates
from .pose_graph_6d import refine_trajectory_6d
from .window_smoother import (
    edge_times_from_window_ends,
    fuse_relative_poses_with_imu,
    refine_trajectory_scaffold,
    relative_poses_to_xyz,
    smooth_absolute_xyz,
)

__all__ = [
    "BiasState",
    "NoiseModel",
    "PreintegrationResult",
    "PreintegrationSE3",
    "RelativeConstraint",
    "detect_spatial_loop_candidates",
    "edge_times_from_window_ends",
    "estimate_gravity_in_world",
    "estimate_gyro_bias",
    "fuse_relative_poses_with_imu",
    "preintegrate_segment",
    "preintegrate_segment_se3",
    "refine_trajectory_6b",
    "refine_trajectory_6c",
    "refine_trajectory_6d",
    "refine_trajectory_6e",
    "refine_trajectory_scaffold",
    "relative_poses_to_xyz",
    "relative_rotation_geodesic_blend",
    "smooth_absolute_xyz",
    "so3_exp",
    "so3_log",
]
