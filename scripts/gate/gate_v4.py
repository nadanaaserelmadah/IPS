"""Gate v4: v3 (C5b) with the network's direction uncertainty estimated ONLINE instead of fixed from KITTI 08.

Motivation (TUM-VI seed 123, RESULTS): v3 uses sigma_net = 0.829 deg, the network's typical direction error on
KITTI 08. On TUM-VI the network is 40-87 deg off, so wherever it happens to agree with geometry the fusion
over-weights it, edges without geometry take its direction, and the per-window length solve can pick its
direction. v4 measures how far the network is from geometry where geometry is reliable, causally:
  sigma_hat(k) = median angle(d_net, d_geo) over the last M "confident" edges before k
                 (geometry present, de-rotated parallax >= P_CONF px, sigma_geo_eff <= S_CONF deg),
                 sigma_net_prior (0.829, KITTI 08) until M_MIN confident edges have been seen.
Direction: v3 fusion with sigma_hat(k) in place of sigma_net. If sigma_hat(k) > TAU the network is distrusted:
edges without geometry hold the last gated direction instead of taking d_net, and each accelerometer window is
solved with the gated direction only (no min-misfit pick of the network-direction solve).
Rotation, the length gate (cond < 1e4, |g| tol 0.5) and every v3 constant are unchanged. No ground truth is used.
The free constants (P_CONF, S_CONF, M, M_MIN, TAU) are chosen on design data only: KITTI 08 (no-08 models) and
TUM-VI room1-4 (zero-shot models); see gate_v4_select.py.
"""
import numpy as np

import gate_v3_select as V
import observability_gate as G

SIGMA_NET_PRIOR = 0.829


def confident(d_geo, sig, fl, par, p_conf, s_conf):
    ok = ~np.isnan(d_geo[:, 0]) & np.isfinite(par) & (par >= p_conf)
    sg = np.sqrt(np.nan_to_num(sig, nan=1e3) ** 2 + np.asarray(fl) ** 2)
    return ok & (sg <= s_conf)


def online_sigma(d_net, d_geo, conf, m=100, m_min=20, prior=SIGMA_NET_PRIOR):
    """Causal per-edge estimate of the network direction error (deg), from confident edges strictly before k."""
    ang = G.angle_deg(d_net, np.nan_to_num(d_geo))
    out = np.full(len(d_net), prior)
    buf = []
    for k in range(len(d_net)):
        if len(buf) >= m_min:
            out[k] = float(np.median(buf[-m:]))
        if conf[k]:
            buf.append(ang[k])
    return out


def fuse_v4(d_net, d_geo, sig, fl, sig_hat):
    """v3 fusion (inverse variance where consistent, v2 gate elsewhere) with a per-edge sigma_net."""
    ok = ~np.isnan(d_geo[:, 0])
    sg2 = np.nan_to_num(sig, nan=1e3) ** 2 + np.asarray(fl) ** 2
    dg = np.nan_to_num(d_geo)
    ang = G.angle_deg(d_net, dg)
    consistent = ok & (ang <= G.K_SIGMA * np.sqrt(sg2 + sig_hat ** 2))
    d_gate, _ = G.direction_gate(d_net, d_geo, sig, fl)
    fused = d_net / sig_hat[:, None] ** 2 + dg / sg2[:, None]
    fused /= np.linalg.norm(fused, axis=1, keepdims=True)
    return np.where(consistent[:, None], fused, d_gate)


def v4(d_net, d_geo, infos, lengths, p):
    """Direction and length per edge. lengths(dirs) -> (L, rms) with NaN where the IMU is not used.
    p: dict with P_CONF, S_CONF, M, M_MIN, TAU. Returns (d, L, sig_hat)."""
    n = len(d_net)
    sig, fl, par = G.sigmas(infos, n), G.floor_v2(infos, n, V.A_V2), G.parallax(infos, n)
    conf = confident(d_geo, sig, fl, par, p["P_CONF"], p["S_CONF"])
    sig_hat = online_sigma(d_net, d_geo, conf, p["M"], p["M_MIN"])
    d = fuse_v4(d_net, d_geo, sig, fl, sig_hat)
    distrust = sig_hat > p["TAU"]
    has_geo = ~np.isnan(d_geo[:, 0])
    last = None
    for k in range(n):  # no geometry and an untrusted network: hold the last gated direction
        if has_geo[k]:
            last = d[k]
        elif distrust[k] and last is not None:
            d[k] = last
    L_v, r_v = lengths(d)
    L_n, r_n = lengths(d_net)
    L = np.where((r_n < r_v) & ~distrust, L_n, L_v)
    return d, L, sig_hat
