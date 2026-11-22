"""Causal online gyro-bias estimator on the per-edge visual residual series (docs/ONLINE_GYRO_BIAS_DESIGN.md).

estimate(series, M, N_MIN, P_MIN): the bias after edge k is the component-wise median of the residual rates
r_j = res_j / dt_j (rad/s, body frame) over the last M usable edges j <= k (usable: parallax >= P_MIN px); zero before
N_MIN usable edges exist. It never looks at edges after k. Convention fixed from physics + the design sequences: the rotation
pairing is res1 (recoverPose R^T) with par0 (de-rotation by recoverPose R).

  python scripts/audit/bias_online.py tune SERIES_DIR SEQ[,SEQ...]            # design sequences, GT bias from the copied csv
  python scripts/audit/bias_online.py eval SERIES_DIR SEQ[,SEQ...] M N_MIN P_MIN
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

WARM_S = 20.0
GRID_M = [50, 100, 200, 400, 0]  # 0 = all edges so far
GRID_NMIN = [10, 20, 50]
GRID_PMIN = [0.0, 1.0, 2.0, 5.0]


def load(series_dir, seq):
    z = np.load(Path(series_dir) / f"{seq}_bias_series.npz")
    return {"t0": z["t0"], "t1": z["t1"], "dt": z["dt"], "rate": z["res1"] / z["dt"][:, None], "par": z["par0"],
            "n_inl": z["n_inl"]}


def estimate(s, M, N_MIN, P_MIN, prior=None):
    """Per-edge causal bias, shape (n, 3); `prior` (3,) used before N_MIN usable edges (zero by default)."""
    n = len(s["t1"])
    use = s["par"] >= P_MIN
    out = np.zeros((n, 3)) if prior is None else np.tile(np.asarray(prior, float), (n, 1))
    hist = []
    for k in range(n):
        if use[k]:
            hist.append(s["rate"][k])
        if len(hist) >= N_MIN:
            h = hist if M == 0 else hist[-M:]
            out[k] = np.median(np.asarray(h), axis=0)
    return out


def gt_bias(series_dir, seq, t):
    g = np.loadtxt(Path(series_dir) / seq / "state_groundtruth_estimate0.csv", delimiter=",", comments="#")
    return np.stack([np.interp(t, g[:, 0], g[:, 11 + j]) for j in range(3)], 1)


def score(series_dir, seq, M, N_MIN, P_MIN):
    s = load(series_dir, seq)
    est = estimate(s, M, N_MIN, P_MIN)
    gt = gt_bias(series_dir, seq, s["t1"])
    err = np.abs(est - gt).max(1)
    late = (s["t1"] - s["t1"][0]) * 1e-9 > WARM_S
    return float(np.median(err[late])), float(np.median(np.abs(gt[late]).max(1)))


def tune(series_dir, seqs):
    rows = []
    for M, N_MIN, P_MIN in itertools.product(GRID_M, GRID_NMIN, GRID_PMIN):
        per = [score(series_dir, q, M, N_MIN, P_MIN) for q in seqs]
        rows.append(((M, N_MIN, P_MIN), float(np.mean([p[0] for p in per])), [p[0] for p in per]))
    rows.sort(key=lambda r: r[1])
    best = rows[0][1]
    near = [r for r in rows if r[1] <= best + 0.001]
    pick = min(near, key=lambda r: (r[0][0] if r[0][0] else 10**9, r[0][1], r[0][2]))
    return rows, pick


def main():
    cmd, sd, seqs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    if cmd == "tune":
        rows, pick = tune(sd, seqs)
        print("| M | N_MIN | P_MIN | score (rad/s) | per sequence |\n|---|---|---|---|---|")
        for (M, N, P), sc, per in rows[:15]:
            print(f"| {M or 'all'} | {N} | {P} | {sc:.4f} | " + " / ".join(f"{x:.4f}" for x in per) + " |")
        print("\nzero-bias score (the static/zero fallback):", " / ".join(f"{np.median(np.abs(gt_bias(sd, q, load(sd, q)['t1'])).max(1)):.4f}" for q in seqs))
        print("PICK", json.dumps({"M": pick[0][0], "N_MIN": pick[0][1], "P_MIN": pick[0][2], "score": pick[1]}))
    elif cmd == "make":  # make SERIES_DIR SEQS M N_MIN P_MIN OUT_DIR -> <seq>_online_bias.npz for IPS_ONLINE_BIAS_DIR
        M, N, P, out = int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), Path(sys.argv[7])
        b_min = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0  # optional: zero the bias while its norm is below b_min
        out.mkdir(parents=True, exist_ok=True)
        for q in seqs:
            s = load(sd, q)
            b = estimate(s, M, N, P)
            b[np.linalg.norm(b, axis=1) < b_min] = 0.0
            np.savez(out / f"{q}_online_bias.npz", t1=s["t1"], b=b)
            print(f"{q}: {len(b)} edges, final bias {np.round(b[-1], 4).tolist()} rad/s")
    elif cmd == "eval":
        M, N, P = int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
        for q in seqs:
            e, b = score(sd, q, M, N, P)
            print(f"{q}: median max-axis error after {WARM_S:.0f} s {e:.4f} rad/s (|GT bias| {b:.4f})")


if __name__ == "__main__":
    main()
