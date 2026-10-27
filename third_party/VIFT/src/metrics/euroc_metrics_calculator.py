"""EuRoC / TUM trajectory metrics with origin, SE(3), and Sim(3) alignment."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.utils.euroc_utils import path_accu

from .base_metrics_calculator import BaseMetricsCalculator


def align_trajectory_lengths(
    est_xyz: np.ndarray, gt_xyz: np.ndarray, *, strict: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """``path_accu`` yields N+1 poses for N relative measurements — align to GT length."""
    est_xyz = np.asarray(est_xyz, dtype=np.float64)
    gt_xyz = np.asarray(gt_xyz, dtype=np.float64)
    if len(est_xyz) == len(gt_xyz) + 1:
        est_xyz = est_xyz[1:]
    elif len(gt_xyz) == len(est_xyz) + 1:
        gt_xyz = gt_xyz[1:]
    elif len(est_xyz) != len(gt_xyz):
        msg = (
            f"Trajectory length mismatch: est={len(est_xyz)} gt={len(gt_xyz)}. "
            "Refusing silent truncation."
        )
        if strict:
            raise ValueError(msg)
        n = min(len(est_xyz), len(gt_xyz))
        print(f"WARN: {msg} Truncating to {n}.")
        est_xyz = est_xyz[:n]
        gt_xyz = gt_xyz[:n]
    return est_xyz, gt_xyz


def align_trajectory_translation(est_xyz: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    """Legacy: align estimated trajectory to GT by matching the first position."""
    return est_xyz + (gt_xyz[0] - est_xyz[0])


def _umeyama(
    src: np.ndarray, dst: np.ndarray, *, with_scale: bool
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama alignment: dst ≈ s * R @ src + t.

    Returns (scale, R[3,3], t[3]).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 3
    n = src.shape[0]
    if n < 3:
        return 1.0, np.eye(3), (dst[0] - src[0]) if n else np.zeros(3)

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(cov)
    s = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1] = -1.0
    r = u @ np.diag(s) @ vt
    var_src = np.mean(np.sum(src_c**2, axis=1))
    scale = 1.0
    if with_scale and var_src > 1e-12:
        scale = float(np.trace(np.diag(d * s)) / var_src)
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t


def align_trajectory_se3(est_xyz: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    """Rigid SE(3) alignment (rotation + translation, no scale)."""
    _, r, t = _umeyama(est_xyz, gt_xyz, with_scale=False)
    return (r @ est_xyz.T).T + t


def align_trajectory_sim3(est_xyz: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    """Similarity Sim(3) alignment (scale + rotation + translation)."""
    s, r, t = _umeyama(est_xyz, gt_xyz, with_scale=True)
    return (s * (r @ est_xyz.T)).T + t


def sim3_scale(est_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    """Umeyama Sim(3) scale s with gt ≈ s·R·est + t (>1 means est is too short)."""
    est_xyz, gt_xyz = align_trajectory_lengths(est_xyz, gt_xyz)
    s, _, _ = _umeyama(est_xyz, gt_xyz, with_scale=True)
    return float(s)


def ate_rmse_m(
    est_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    *,
    alignment: str = "origin",
) -> float:
    est_xyz, gt_xyz = align_trajectory_lengths(est_xyz, gt_xyz)
    mode = alignment.lower()
    if mode in {"origin", "translation", "legacy"}:
        est_a = align_trajectory_translation(est_xyz, gt_xyz)
    elif mode in {"se3", "ate_se3"}:
        est_a = align_trajectory_se3(est_xyz, gt_xyz)
    elif mode in {"sim3", "ate_sim3"}:
        est_a = align_trajectory_sim3(est_xyz, gt_xyz)
    else:
        raise ValueError(f"Unknown alignment mode: {alignment}")
    return float(np.sqrt(np.mean(np.sum((est_a - gt_xyz) ** 2, axis=1))))


def zero_motion_ate_m(gt_xyz: np.ndarray) -> float:
    """SE(3) / Sim(3) ATE of an estimate that never moves: RMS spread of GT about its centroid.

    Any run at or above this floor carries no information about the trajectory.
    """
    gt_xyz = np.asarray(gt_xyz, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum((gt_xyz - gt_xyz.mean(axis=0)) ** 2, axis=1))))


def rpe_translation_rmse(
    est_xyz: np.ndarray, gt_xyz: np.ndarray, delta: int = 1
) -> float:
    est_xyz, gt_xyz = align_trajectory_lengths(est_xyz, gt_xyz)
    if len(est_xyz) <= delta:
        return float("nan")
    est_d = est_xyz[delta:] - est_xyz[:-delta]
    gt_d = gt_xyz[delta:] - gt_xyz[:-delta]
    err = np.linalg.norm(est_d - gt_d, axis=1)
    return float(np.sqrt(np.mean(err**2)))


def path_length_m(xyz: np.ndarray) -> float:
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))


def validate_trajectory(
    est_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    *,
    max_step_m: float = 5.0,
    max_abs_coord_m: float = 1e4,
    max_path_ratio: float = 50.0,
) -> Dict[str, Any]:
    """Return validity report; does not mutate inputs."""
    est_xyz = np.asarray(est_xyz, dtype=np.float64)
    gt_xyz = np.asarray(gt_xyz, dtype=np.float64)
    reasons: list[str] = []
    if not np.isfinite(est_xyz).all():
        reasons.append("non_finite_est")
    if not np.isfinite(gt_xyz).all():
        reasons.append("non_finite_gt")
    if len(est_xyz) < 2 or len(gt_xyz) < 2:
        reasons.append("too_short")
    else:
        est_steps = np.linalg.norm(np.diff(est_xyz, axis=0), axis=1)
        gt_steps = np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1)
        if est_steps.max(initial=0.0) > max_step_m:
            reasons.append("est_max_step")
        if gt_steps.max(initial=0.0) > max_step_m * 10:  # GT spikes are data issues
            reasons.append("gt_max_step")
        if np.max(np.abs(est_xyz)) > max_abs_coord_m:
            reasons.append("est_abs_coord")
        est_len = path_length_m(est_xyz)
        gt_len = path_length_m(gt_xyz)
        if gt_len > 1e-6 and est_len / gt_len > max_path_ratio:
            reasons.append("path_ratio")
        if gt_len > 1e4:  # e.g. V2_03-style GT blow-up
            reasons.append("gt_path_implausible")
    return {
        "valid": len(reasons) == 0,
        "reasons": reasons,
        "est_path_len_m": path_length_m(est_xyz),
        "gt_path_len_m": path_length_m(gt_xyz),
    }


class EuroCMetricsCalculator(BaseMetricsCalculator):
    """
    Compute trajectory metrics.

    By default reports:
      - ``{seq}_ate_m``            : legacy origin alignment (backward compatible)
      - ``{seq}_ate_origin_m``     : same as ate_m
      - ``{seq}_ate_se3_m``        : rigid SE(3) ATE (primary for peer comparison)
      - ``{seq}_ate_sim3_m``       : Sim(3) ATE (scale diagnostic)
      - ``{seq}_sim3_scale``       : Sim(3) scale s (gt ≈ s·est; 1.0 = metric scale)
      - ``{seq}_ate_zero_motion_m``: SE(3)/Sim(3) ATE of a never-moving estimate (GT spread floor)
      - ``{seq}_rpe_t_m``          : translational RPE (after SE(3) alignment)
      - ``{seq}_path_len_m``       : GT path length
      - ``{seq}_valid``            : 1.0 if checks pass else 0.0
    """

    def __init__(
        self,
        primary_alignment: str = "origin",
        report_all_alignments: bool = True,
        max_step_m: float = 5.0,
        invalid_policy: str = "flag",  # flag | exclude
    ):
        self.primary_alignment = primary_alignment
        self.report_all_alignments = report_all_alignments
        self.max_step_m = max_step_m
        self.invalid_policy = invalid_policy

    def _xyz_from_results(self, seq_data: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        pose_est = seq_data["estimated_poses"]
        pose_gt = seq_data.get("gt_poses")

        est_mats = path_accu(pose_est)
        est_xyz = np.array([T[:3, 3] for T in est_mats])

        gt_xyz_integrated = None
        if pose_gt is not None:
            gt_mats = path_accu(pose_gt)
            gt_xyz_integrated = np.array([T[:3, 3] for T in gt_mats])

        gt_xyz_world = None
        if "gt_xyz" in seq_data and seq_data["gt_xyz"] is not None:
            gt_xyz_world = np.asarray(seq_data["gt_xyz"], dtype=np.float64)

        if gt_xyz_integrated is None and gt_xyz_world is None:
            raise KeyError("No GT trajectory (gt_xyz or gt_poses)")
        return est_xyz, gt_xyz_integrated, gt_xyz_world

    def calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        valid_ates: list[float] = []
        for seq_name, seq_data in results.items():
            est_xyz, gt_int, gt_world = self._xyz_from_results(seq_data)

            # Legacy origin ATE: compare integrated relative trajectories (same local frame).
            if gt_int is not None:
                est_o, gt_o = align_trajectory_lengths(est_xyz, gt_int)
                ate_origin = ate_rmse_m(est_o, gt_o, alignment="origin")
                gt_for_valid = gt_o
                est_for_valid = est_o
            else:
                est_o, gt_o = align_trajectory_lengths(est_xyz, gt_world)
                ate_origin = ate_rmse_m(est_o, gt_o, alignment="origin")
                gt_for_valid = gt_o
                est_for_valid = est_o

            # SE(3) / Sim(3): prefer world GT when available (TUM mocap / EuRoC absolute).
            gt_abs = gt_world if gt_world is not None else gt_int
            est_a, gt_a = align_trajectory_lengths(est_xyz, gt_abs)
            ate_se3 = ate_rmse_m(est_a, gt_a, alignment="se3")
            ate_sim3 = ate_rmse_m(est_a, gt_a, alignment="sim3")
            scale_sim3 = sim3_scale(est_a, gt_a)
            # est is in the camera-0 frame and gt_abs may be world-frame mocap, so
            # displacements are compared only after rigid SE(3) alignment.
            rpe_t = rpe_translation_rmse(align_trajectory_se3(est_a, gt_a), gt_a, delta=1)

            report = validate_trajectory(
                est_for_valid, gt_for_valid, max_step_m=self.max_step_m
            )
            seq_data["validity"] = report
            if report["reasons"]:
                print(
                    f"WARN: sequence {seq_name} validity issues: {report['reasons']}"
                )

            if self.primary_alignment.lower() in {"se3", "ate_se3"}:
                primary = ate_se3
            elif self.primary_alignment.lower() in {"sim3", "ate_sim3"}:
                primary = ate_sim3
            else:
                primary = ate_origin

            metrics[f"{seq_name}_ate_m"] = primary
            metrics[f"{seq_name}_ate_origin_m"] = ate_origin
            metrics[f"{seq_name}_ate_se3_m"] = ate_se3
            metrics[f"{seq_name}_ate_sim3_m"] = ate_sim3
            metrics[f"{seq_name}_sim3_scale"] = scale_sim3
            metrics[f"{seq_name}_ate_zero_motion_m"] = zero_motion_ate_m(gt_a)
            metrics[f"{seq_name}_rpe_t_m"] = rpe_t
            metrics[f"{seq_name}_path_len_m"] = float(
                report["gt_path_len_m"]
                if gt_int is not None
                else path_length_m(gt_a)
            )
            metrics[f"{seq_name}_est_path_len_m"] = report["est_path_len_m"]
            metrics[f"{seq_name}_valid"] = 1.0 if report["valid"] else 0.0

            if report["valid"]:
                valid_ates.append(primary)

        if valid_ates:
            metrics["mean_ate_m"] = float(np.mean(valid_ates))
            metrics["n_valid_seq"] = float(len(valid_ates))
        else:
            metrics["mean_ate_m"] = float("nan")
            metrics["n_valid_seq"] = 0.0
        return metrics

    def primary_metric_names(self, results: Dict[str, Any]) -> Sequence[str]:
        """Names that should contribute to hp_metric (mean ATE only)."""
        return ["mean_ate_m"]
